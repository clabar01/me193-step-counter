// Read the MPU-6050 at 50 Hz, ping the HC-SR04 once per sample, and send both
// to the Python side.
//
// IMU: MPU-6050 at 0x68 on Wire (labeled SDA/SCL header pins), 100 kHz.
// Ultrasonic: HC-SR04 on 5V, Trig on D2, Echo on D3 through a 1k/2k divider
// (see HC-SR04 below; it never delays an IMU read).
// Each sample goes to Python with Bridge.notify("imu_sample", t, ax, ay, az,
// gx, gy, gz, us_mm, us_hit) (fire-and-forget, never waits on Python), which
// publishes it to MQTT and records it. Status messages go to both Monitor ('#'
// lines) and Bridge.notify("imu_status", text), so they also appear in the
// app's Python logs.
//   t       integer ms since start, steps of exactly 20 ms; a skipped sample
//           shows up as a gap in t
//   a*      g, body frame (see AXIS REMAP below)
//   g*      deg/s, body frame
//   us_mm   distance of the ping fired right after the PREVIOUS sample's IMU
//           read (so it is 15-20 ms older than the IMU values), mm
//   us_hit  1 = real echo; 0 = no echo within US_MAX_MM (the other thigh is
//           out of the beam, or no sensor), and then us_mm = US_MAX_MM
// PRINT_CSV also prints every sample to Monitor as
// "t,ax,ay,az,gx,gy,gz,us_mm,us_hit".
//
// LED matrix: Python calls Bridge.call("show_icon", name) with the recording
// state and the matching 13x8 icon is drawn (see LED MATRIX ICONS).
//
// Libraries: Arduino_RouterBridge, Adafruit MPU6050 (+ Adafruit Unified
// Sensor and Adafruit BusIO, which it depends on). Arduino_LED_Matrix comes
// with the board package.

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino_LED_Matrix.h>

const uint8_t MPU_ADDR = 0x68;
const uint32_t I2C_HZ = 100000;
const uint32_t SAMPLE_MS = 20;                 // 50 Hz
const uint32_t SAMPLE_US = SAMPLE_MS * 1000;
const uint8_t READ_TRIES = 3;
const uint32_t REINIT_PERIOD_MS = 500;         // how often to retry while the sensor is down
const bool PRINT_CSV = false;                  // true: also print every sample to Monitor (testing)
const uint32_t US_STATS_MS = 10000;            // how often to report ping stats as a status line

// Gyro zero-rate bias of THIS board's MPU-6050, deg/s, in SENSOR axes (X, Y, Z
// as printed on the board, before the remap). Measured at rest over 10 s on
// 2026-10-23; the monitor showed body gx 1.46, gy -1.11, gz -0.07, which the
// default remap maps back to sensor X 1.11, Y 1.46, Z -0.07. Re-measure if the
// sensor board is swapped. No startup calibration on purpose: the wearer may be
// moving when it boots.
const float GYRO_BIAS_DPS[3] = {1.11f, 1.46f, -0.07f};

// ===================== AXIS REMAP: edit after the mounting check =====================
// Body frame (CLAUDE.md), right-handed:
//   x along the thigh toward the hip   (+1 g standing)
//   y lateral, pointing to your right
//   z out of the front of the thigh    (+1 g sitting, thigh horizontal)
// body[i] = BODY_SIGN[i] * sensor[BODY_FROM[i]], sensor axes 0=X 1=Y 2=Z as
// printed on the board. The same remap is applied to accel and gyro.
//
// Current mounting (axis check 2026-10-23): board flat on the front of the
// thigh, chip side facing out, with the board's +X arrow pointing DOWN toward
// the knee. Then
//   body x = -sensor X,  body y = -sensor Y,  body z = +sensor Z.
// Expected with this remap: standing ax ~ +1, sitting az ~ +1, and gy goes
// POSITIVE while sitting down (the thigh rotates about +y, to your right).
// Keep the frame right-handed: if you flip one sign, also flip another or
// swap two axes, otherwise the gyro signs won't match the accel.
const uint8_t BODY_FROM[3] = {0, 1, 2};
const int8_t  BODY_SIGN[3] = {-1, -1, +1};
// =====================================================================================

// ===================== HC-SR04 (non-blocking) =====================
// One ping per IMU sample (every 20 ms), fired right after the IMU read; its
// result is collected just before the next IMU read and sent with that sample.
// The only busy-wait is the 10 us trigger pulse.
//
// The echo pulse is timed by an interrupt on D3 (both edges, micros()). With
// US_USE_INTERRUPT false, loop() polls D3 instead: still non-blocking, but an
// edge that comes while loop() is busy (I2C read, Bridge write) is timestamped
// late, so prefer the interrupt.
//
// Timeouts: a pulse longer than US_MAX_MM (~5.8 ms) or no pulse at all means
// nothing in range: us_mm = US_MAX_MM, us_hit = 0. With no target a real
// HC-SR04 holds Echo high for ~38 ms and ignores triggers meanwhile, so the
// next 1-2 pings are reported as timeouts too ("busy" in the stats). Unplugged,
// the 2k divider resistor holds D3 low and every ping times out.
const uint8_t TRIG_PIN = 2;
const uint8_t ECHO_PIN = 3;
const bool US_USE_INTERRUPT = true;
const int32_t US_MAX_MM = 1000;
const uint32_t US_MAX_ECHO_US = (uint32_t)US_MAX_MM * 2000 / 343;   // round trip at 343 m/s: ~5831 us

enum : uint8_t { US_IDLE, US_WAIT_RISE, US_HIGH, US_DONE };
volatile uint8_t usState = US_IDLE;
volatile uint32_t usRiseUs = 0;
volatile uint32_t usFallUs = 0;
int32_t usMm = US_MAX_MM;    // result sent with the next sample
uint8_t usHit = 0;

// Ping stats since the last report
uint32_t usPings = 0, usHits = 0, usNoEcho = 0, usTooFar = 0, usBusy = 0;
uint32_t lastUsStatsMs = 0;
// =====================================================================

// ===================== LED MATRIX ICONS (13 x 8) =====================
// One string per row, '#' = LED on. Edit freely, but keep 8 rows of exactly
// 13 characters. board/run_board_app.py prints these as ASCII.
// ready = idle, waiting for a trial; rec = recording (blinks); count =
// stopped, waiting for the counted steps; saved / error flash for a few s.
const char *const ICON_READY[8] = {
  ".............",
  "..##.........",
  ".####........",
  ".####...##...",
  "..##...####..",
  "..##...####..",
  ".......##....",
  ".............",
};
const char *const ICON_REC[8] = {
  ".............",
  ".....###.....",
  "....#####....",
  "...#######...",
  "...#######...",
  "....#####....",
  ".....###.....",
  ".............",
};
const char *const ICON_COUNT[8] = {
  ".....###.....",
  "....#...#....",
  "........#....",
  ".......#.....",
  "......#......",
  "......#......",
  ".............",
  "......#......",
};
const char *const ICON_SAVED[8] = {
  ".............",
  "...........#.",
  "..........#..",
  ".........#...",
  ".#......#....",
  "..#....#.....",
  "...#..#......",
  "....##.......",
};
const char *const ICON_ERROR[8] = {
  ".............",
  "..#.......#..",
  "...#.....#...",
  ".....#.#.....",
  "......#......",
  ".....#.#.....",
  "...#.....#...",
  "..#.......#..",
};

struct Icon {
  const char *label;
  const char *const *rows;
  bool blink;
};
const Icon ICONS[] = {
  {"ready", ICON_READY, false},
  {"rec", ICON_REC, true},
  {"count", ICON_COUNT, false},
  {"saved", ICON_SAVED, false},
  {"error", ICON_ERROR, false},
};
const int N_ICONS = sizeof(ICONS) / sizeof(ICONS[0]);
const uint32_t BLINK_MS = 500;
// =====================================================================

Arduino_LED_Matrix matrix;
volatile int requestedIcon = -1;   // set by show_icon (Bridge thread); -1 = blank
int shownIcon = -2;                // what loop() last drew
bool blinkOn = true;
uint32_t lastBlinkMs = 0;

Adafruit_MPU6050 mpu;
bool sensorOk = false;
uint32_t lastReinitMs = 0;
uint32_t errorCount = 0;     // failed reads (after all retries)
uint32_t reinitCount = 0;

uint32_t nextUs;             // when the next sample is due
uint32_t tick = 0;           // sample slot number; t = tick * SAMPLE_MS

bool initSensor() {
  Wire.begin();
  if (!mpu.begin(MPU_ADDR, &Wire)) {
    return false;
  }
  Wire.setClock(I2C_HZ);     // after begin(), which may reset the bus clock
  // +/-8 g and +/-1000 deg/s saturated while running (2026-10-23)
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  return true;
}

// One read in sensor axes. getEvent() doesn't report I2C errors, so also check
// that the sensor still ACKs and that the data isn't a stuck-bus pattern (all
// three axes identical, e.g. every byte 0x00 or 0xFF).
bool readOnce(float acc[3], float gyr[3]) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) {
    return false;
  }
  if (a.acceleration.x == a.acceleration.y && a.acceleration.y == a.acceleration.z &&
      g.gyro.x == g.gyro.y && g.gyro.y == g.gyro.z) {
    return false;
  }

  acc[0] = a.acceleration.x / SENSORS_GRAVITY_STANDARD;   // m/s^2 -> g
  acc[1] = a.acceleration.y / SENSORS_GRAVITY_STANDARD;
  acc[2] = a.acceleration.z / SENSORS_GRAVITY_STANDARD;
  gyr[0] = g.gyro.x * RAD_TO_DEG - GYRO_BIAS_DPS[0];       // rad/s -> deg/s, minus bias
  gyr[1] = g.gyro.y * RAD_TO_DEG - GYRO_BIAS_DPS[1];
  gyr[2] = g.gyro.z * RAD_TO_DEG - GYRO_BIAS_DPS[2];
  return true;
}

void toBody(const float s[3], float b[3]) {
  for (int i = 0; i < 3; i++) {
    b[i] = BODY_SIGN[i] * s[BODY_FROM[i]];
  }
}

// printf's %f isn't reliably supported on this core, so format fixed-point by hand.
int appendFixed(char *buf, size_t size, float v, int decimals) {
  long scale = 1;
  for (int i = 0; i < decimals; i++) {
    scale *= 10;
  }
  long q = lroundf(v * scale);
  const char *sign = q < 0 ? "-" : "";
  q = labs(q);
  return snprintf(buf, size, ",%s%ld.%0*ld", sign, q / scale, decimals, q % scale);
}

void printSample(uint32_t t, const float acc[3], const float gyr[3], int32_t mm, uint8_t hit) {
  char line[112];
  int len = snprintf(line, sizeof(line), "%lu", (unsigned long)t);
  for (int i = 0; i < 3; i++) {
    len += appendFixed(line + len, sizeof(line) - len, acc[i], 4);
  }
  for (int i = 0; i < 3; i++) {
    len += appendFixed(line + len, sizeof(line) - len, gyr[i], 2);
  }
  snprintf(line + len, sizeof(line) - len, ",%ld,%u", (long)mm, hit);
  Monitor.println(line);
}

void status(const char *msg) {
  char line[96];
  snprintf(line, sizeof(line), "%s (errors=%lu, reinits=%lu)", msg,
           (unsigned long)errorCount, (unsigned long)reinitCount);
  Monitor.print("# ");
  Monitor.println(line);
  Bridge.notify("imu_status", line);
}

// ---------- HC-SR04 ----------

// Echo edge (interrupt context, or called from loop() when polling).
void echoEdge() {
  uint32_t now = micros();
  bool high = digitalRead(ECHO_PIN) == HIGH;
  if (high && usState == US_WAIT_RISE) {
    usRiseUs = now;
    usState = US_HIGH;
  } else if (!high && usState == US_HIGH) {
    usFallUs = now;
    usState = US_DONE;
  }
}

// Polling mode: call often from loop(); acts on level changes like the ISR.
void pollEcho() {
  static bool lastHigh = false;
  bool high = digitalRead(ECHO_PIN) == HIGH;
  if (high != lastHigh) {
    lastHigh = high;
    echoEdge();
  }
}

// Read the result of the ping fired one sample ago into usMm / usHit.
void collectPing() {
  noInterrupts();
  uint8_t state = usState;
  uint32_t rise = usRiseUs, fall = usFallUs;
  usState = US_IDLE;          // ignore any later edges of this ping
  interrupts();

  usMm = US_MAX_MM;
  usHit = 0;
  if (state == US_DONE && fall - rise <= US_MAX_ECHO_US) {
    usMm = (int32_t)(((fall - rise) * 343UL + 1000) / 2000);   // us -> mm, round trip
    usHit = 1;
    usHits++;
  } else if (state == US_DONE || state == US_HIGH) {
    usTooFar++;               // echo longer than US_MAX_MM, or still high
  } else if (state == US_WAIT_RISE) {
    usNoEcho++;               // never went high (no sensor, or nothing at all)
  }
}

// Fire the next ping, unless Echo is still high from the last one (then the
// sensor ignores triggers; this sample's result will be a timeout).
void firePing() {
  usPings++;
  if (digitalRead(ECHO_PIN) == HIGH) {
    usBusy++;
    return;                   // stays US_IDLE -> timeout
  }
  usState = US_WAIT_RISE;
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
}

void reportPingStats() {
  char line[96];
  snprintf(line, sizeof(line), "us pings=%lu hits=%lu no_echo=%lu too_far=%lu busy=%lu",
           (unsigned long)usPings, (unsigned long)usHits, (unsigned long)usNoEcho,
           (unsigned long)usTooFar, (unsigned long)usBusy);
  status(line);
  usPings = usHits = usNoEcho = usTooFar = usBusy = 0;
}

// ---------- LED matrix ----------

// Python calls this with the recording state. Bridge providers run on their
// own thread, so this only records the request; loop() does the drawing.
void show_icon(String label) {
  int idx = -1;
  for (int i = 0; i < N_ICONS; i++) {
    if (label == ICONS[i].label) {
      idx = i;
    }
  }
  requestedIcon = idx;
}

void drawIcon(int idx, bool on) {
  uint8_t frame[8][13] = {};
  if (idx >= 0 && on) {
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 13; c++) {
        frame[r][c] = ICONS[idx].rows[r][c] == '#';
      }
    }
  }
  matrix.renderBitmap(frame, 8, 13);
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  matrix.begin();
  matrix.clear();
  Bridge.provide("show_icon", show_icon);

  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);
  if (US_USE_INTERRUPT) {
    attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echoEdge, CHANGE);
  }

  sensorOk = initSensor();
  lastReinitMs = millis();
  status(sensorOk ? "MPU-6050 ready" : "MPU-6050 not found, will keep retrying");
  if (PRINT_CSV) {
    Monitor.println("t,ax,ay,az,gx,gy,gz,us_mm,us_hit");
  }

  nextUs = micros();
  lastUsStatsMs = millis();
}

void loop() {
  if (!US_USE_INTERRUPT) {
    pollEcho();
  }

  int wanted = requestedIcon;
  bool blink = wanted >= 0 && ICONS[wanted].blink;
  if (wanted != shownIcon) {
    blinkOn = true;
    lastBlinkMs = millis();
    drawIcon(wanted, blinkOn);
    shownIcon = wanted;
  } else if (blink && millis() - lastBlinkMs >= BLINK_MS) {
    lastBlinkMs = millis();
    blinkOn = !blinkOn;
    drawIcon(wanted, blinkOn);
  }

  // Fixed-rate scheduler on micros(): no delay(), and no drift because the
  // next deadline is always the previous one + 20 ms. I2C reads can't run
  // inside a timer interrupt, so the deadline is polled here.
  uint32_t now = micros();
  if ((int32_t)(now - nextUs) < 0) {
    return;
  }
  // Fell behind by whole periods (e.g. during a re-init): skip those slots so
  // the gap is visible in t instead of bursting out late samples.
  while ((int32_t)(now - nextUs) >= (int32_t)SAMPLE_US) {
    nextUs += SAMPLE_US;
    tick++;
  }
  uint32_t t = tick * SAMPLE_MS;
  nextUs += SAMPLE_US;
  tick++;

  // Last slot's ping is over (its echo window is ~6 ms of the 20 ms slot).
  collectPing();
  if (millis() - lastUsStatsMs >= US_STATS_MS) {
    lastUsStatsMs = millis();
    reportPingStats();
  }

  if (!sensorOk) {
    if (millis() - lastReinitMs >= REINIT_PERIOD_MS) {
      lastReinitMs = millis();
      reinitCount++;
      sensorOk = initSensor();
      status(sensorOk ? "re-init OK" : "re-init failed");
    }
    firePing();               // keep the ping cadence while the IMU is down
    return;
  }

  float sAcc[3], sGyr[3];
  for (uint8_t i = 0; i < READ_TRIES; i++) {
    if (readOnce(sAcc, sGyr)) {
      float acc[3], gyr[3];
      toBody(sAcc, acc);
      toBody(sGyr, gyr);
      Bridge.notify("imu_sample", t, acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2], usMm, usHit);
      if (PRINT_CSV) {
        printSample(t, acc, gyr, usMm, usHit);
      }
      firePing();             // after the IMU read, so its echo never overlaps one
      return;
    }
  }

  firePing();
  errorCount++;
  sensorOk = false;
  lastReinitMs = millis() - REINIT_PERIOD_MS;   // re-init on the next slot
  status("read failed, re-initializing");
}
