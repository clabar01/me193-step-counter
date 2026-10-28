// I2C scanner for the Uno Q microcontroller.
//
// Scans the labeled SDA/SCL header pins (Wire) at 100 kHz every 3 s and prints
// every address that answers. The MPU-6050 should show up at 0x68 (0x69 if its
// AD0 pin is pulled high).
//
// Output goes to the App Lab serial monitor through Arduino_RouterBridge's
// Monitor; on the Uno Q, Serial is the D0/D1 UART, not the monitor.

#include <Arduino_RouterBridge.h>
#include <Wire.h>

const uint32_t I2C_HZ = 100000;
const uint32_t SCAN_PERIOD_MS = 3000;

void setup() {
  Bridge.begin();
  Monitor.begin();
  Wire.begin();
  Wire.setClock(I2C_HZ);
}

void loop() {
  char line[48];
  int found = 0;

  Monitor.println("Scanning I2C (Wire, 100 kHz)...");
  // 0x08-0x77 is the usable 7-bit range; the rest are reserved.
  for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      snprintf(line, sizeof(line), "  found device at 0x%02X%s", addr,
               addr == 0x68 || addr == 0x69 ? " (MPU-6050?)" : "");
      Monitor.println(line);
      found++;
    } else if (err == 4) {
      snprintf(line, sizeof(line), "  bus error at 0x%02X", addr);
      Monitor.println(line);
    }
  }

  if (found == 0) {
    Monitor.println("  no devices found: check SDA/SCL, 3.3V and GND");
  } else {
    snprintf(line, sizeof(line), "  %d device(s) found", found);
    Monitor.println(line);
  }

  delay(SCAN_PERIOD_MS);
}
