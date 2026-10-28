"""Publish simulated thigh-IMU + ultrasonic samples to MQTT, so the board app,
recorder and dashboard can be tested end-to-end without the hardware.

*** FAKE DATA FOR TESTING ONLY. ***
These signals are hand-made approximations, not recordings. Never put them in
data/raw/, train on them, or report results from them.

Message format (the board's, see board/python/main.py):
    ~10 messages/s, 5 samples each (50 Hz), JSON list of samples:
        [{"t": 1240, "ax": 0.98, "ay": 0.01, "az": 0.12, "gx": -0.4, "gy": 12.7,
          "gz": 0.9, "us_mm": 112, "us_hit": 1}, ...]
    t ms since the stream started, accel g, gyro deg/s; us_hit 0 -> us_mm 1000.

Topics: by default it publishes to <prefix>/sim/raw, which board/run_board_app.py
feeds into the real board app (that app then publishes <prefix>/raw, status and
trial state, and records trials). --direct publishes to <prefix>/raw and
<prefix>/status itself, for watching the dashboard plots without the board app
(trials can't be recorded then).

Walking with a known step count: each time a trial starts (<prefix>/trial/state
goes to "recording") it stands for --lead-in s, walks exactly --steps steps at
--speed, then stands again, and prints the true count and distance. Keys: w
walks --steps steps now, s stands, q quits. --continuous walks all the time.

The model (sensor on the front of one thigh, HC-SR04 on its inner side pointing
at the other thigh):
  - thigh pitch = mean + swing * sin(2 pi phase), phase in strides (2 steps);
    gy = d(pitch)/dt (positive = thigh swinging forward, as on the real sensor)
  - heel strikes (= steps) just before each thigh's maximum forward swing:
    this leg's at phase 0.22, the other leg's half a stride later, as spikes
    mostly along the thigh (ax)
  - the other thigh swings in antiphase; the beam hits it while the two thighs
    are within BEAM_HALF_M of each other front-to-back, i.e. as they pass at
    mid-stance, twice per stride: the distance dips to the gap between the
    thighs. Otherwise the ping times out (us_mm 1000, us_hit 0), and the next
    ping after a timeout is lost too (a real HC-SR04 holds Echo high ~38 ms).
  - noise, a few dropped echoes, and a few stray echoes (arms, clothing)
  - the ultrasonic value is one sample late, like the sketch's
"""

import argparse
import json
import math
import queue
import random
import select
import sys
import termios
import threading
import time
import tty
import zlib

import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
PREFIX = "ME193/ceci/steps/test"
FS = 50                  # sample rate, Hz
BATCH = 5                # samples per message -> 10 messages/s
STATUS_PERIOD_S = 5
THIGH_RADIUS_M = 0.15    # hip joint to IMU, for tangential accel
US_HEIGHT_M = 0.20       # hip joint to the ultrasonic beam
BEAM_HALF_M = 0.05       # beam hits the other thigh within this front-to-back offset
US_MAX_MM = 1000
US_NOISE_MM = 3.0
US_DROP = 0.04           # chance an echo off the thigh is missed anyway
US_STRAY = 0.02          # chance of a stray echo when the thigh is out of the beam
STRIKE_PHASE = 0.22      # this leg's heel strike, strides after the thighs pass
ACCEL_NOISE_G = 0.015
GYRO_NOISE_DPS = 1.0
BLEND_S = 0.5            # how fast speed parameters change (mixed)

# Sensor axes (same as the gait project): x along the thigh toward the hip
# (+1 g standing), y to the wearer's right, z out of the front of the thigh.
#
# cadence   steps/min        swing    thigh swing amplitude, deg
# pitch     mean thigh pitch while moving, deg (stairs: thigh raised)
# impact    this leg's heel-strike spike along x, g; other = the other leg's, as a fraction
# step_m    step length, m (for the true distance)
SPEEDS = {
    "slow":   dict(cadence=90,  swing=18, pitch=4,  impact=0.5, other=0.40, step_m=0.55),
    "normal": dict(cadence=110, swing=23, pitch=5,  impact=0.8, other=0.40, step_m=0.70),
    "fast":   dict(cadence=125, swing=27, pitch=6,  impact=1.2, other=0.40, step_m=0.80),
    "jog":    dict(cadence=155, swing=33, pitch=12, impact=2.0, other=0.45, step_m=0.95),
    "run":    dict(cadence=170, swing=38, pitch=15, impact=2.8, other=0.45, step_m=1.20),
    "stairs": dict(cadence=95,  swing=28, pitch=28, impact=0.4, other=0.30, step_m=0.28),
}
MIXED_FROM = ("slow", "normal", "fast", "jog")
MIXED_EVERY = (6, 12)    # mixed: new speed every this many steps
PULSE_WIDTH_S = 0.03


class StepSim:
    """One 50 Hz sample at a time; walk(n, speed) takes exactly n steps."""

    def __init__(self, subject, rng):
        self.rng = rng
        # The subject name seeds a fixed "person": same name, same person.
        srng = random.Random(zlib.crc32(subject.encode()))
        self.amp_scale = srng.uniform(0.85, 1.15)
        self.cadence_scale = srng.uniform(0.92, 1.08)
        self.mount_roll = math.radians(srng.uniform(-8, 8))   # strap tilt
        self.gyro_bias = [srng.gauss(0, 1.5) for _ in range(3)]
        self.gap_mm = srng.uniform(70, 130)                   # between the thighs at the beam
        self.phase = 0.0          # strides
        self.amp = 0.0            # 0 standing .. 1 walking
        self.target_amp = 0.0
        self.params = dict(SPEEDS["normal"])
        self.target_params = dict(SPEEDS["normal"])
        self.speed = None
        self.steps_target = None  # None = walk forever
        self.steps_done = 0
        self.distance_m = 0.0
        self.next_switch = None
        self.done_callback = None
        self.prev_pitch = None
        self.prev_rate = 0.0
        self.us_pending = (int(self.gap_mm), 1)
        self.us_busy = 0
        self._new_stride()

    # ---------- control ----------

    def walk(self, steps, speed, on_done=None):
        """Start walking from standing: exactly `steps` steps (None = forever)."""
        self.speed = speed
        self.steps_target = steps
        self.steps_done = 0
        self.distance_m = 0.0
        self.done_callback = on_done
        self.phase = 0.0          # thighs side by side, as when standing
        self.target_amp = 1.0
        self._set_speed(self.rng.choice(MIXED_FROM) if speed == "mixed" else speed)
        self.params = dict(self.target_params)
        self.next_switch = self.rng.randint(*MIXED_EVERY) if speed == "mixed" else None

    def stand(self):
        self.target_amp = 0.0     # no further strikes are counted

    def _set_speed(self, name):
        self.current = name
        self.target_params = dict(SPEEDS[name])

    def _new_stride(self):
        self.stride_amp = self.amp_scale * self.rng.uniform(0.9, 1.1)
        self.stride_freq_jitter = self.rng.gauss(1.0, 0.03)

    def _freq(self):
        return self.params["cadence"] / 120 * self.cadence_scale * self.stride_freq_jitter

    # ---------- one sample ----------

    def sample(self, t):
        dt = 1.0 / FS
        rng = self.rng
        k = min(1.0, dt / BLEND_S)
        for key, target in self.target_params.items():
            self.params[key] += (target - self.params[key]) * k
        p = self.params
        freq = self._freq()

        # Amplitude ramps over half a stride when starting or stopping.
        ramp = dt * 2 * max(freq, 0.7)
        self.amp += max(-ramp, min(ramp, self.target_amp - self.amp))
        moving = self.amp > 0 or self.target_amp > 0

        pulse = 0.0
        if moving:
            old = self.phase
            self.phase += freq * dt
            if int(self.phase) != int(old):
                self._new_stride()
            # Heel strikes: step k (0-based) at 2 * (phase - STRIKE_PHASE) = k.
            u_old, u = 2 * (old - STRIKE_PHASE), 2 * (self.phase - STRIKE_PHASE)
            if self.target_amp > 0 and math.floor(u) > math.floor(u_old) and u >= 0:
                self._on_step(p)
            # Spike around the nearest strike c, only if it is (or will be) a counted step.
            c = round(u)
            upcoming = (u < c and self.target_amp > 0
                        and (self.steps_target is None or c < self.steps_target))
            if c >= 0 and (c < self.steps_done or upcoming):
                d = abs(u - c) / (2 * freq)          # seconds from that strike
                size = p["impact"] * (1.0 if c % 2 == 0 else p["other"])
                pulse = size * self.stride_amp * math.exp(-0.5 * (d / PULSE_WIDTH_S) ** 2)

        swing = math.radians(p["swing"]) * self.stride_amp * self.amp
        mean = math.radians(p["pitch"]) * self.amp
        s = math.sin(2 * math.pi * self.phase)
        sway = math.radians(1.0) * math.sin(2 * math.pi * 0.25 * t)   # postural sway
        pitch = mean + swing * s + sway
        other_pitch = mean - swing * s

        rate = 0.0 if self.prev_pitch is None else (pitch - self.prev_pitch) / dt
        acc_ang = (rate - self.prev_rate) / dt
        self.prev_pitch, self.prev_rate = pitch, rate

        ax = math.cos(pitch) + pulse
        az = math.sin(pitch) + THIGH_RADIUS_M * acc_ang / 9.81 + 0.3 * pulse
        r = self.mount_roll                     # a tilted strap leaks z into y
        ay = -az * math.sin(r)
        az = az * math.cos(r)
        gy_true = math.degrees(rate)
        gx = 0.1 * abs(gy_true) * math.sin(4 * math.pi * self.phase)   # small hip rotation
        gy = gy_true * math.cos(r)
        gz = gy_true * math.sin(r)

        us_mm, us_hit = self.us_pending          # last sample's ping, like the sketch
        self.us_pending = self._ping(pitch - sway, other_pitch)

        n = rng.gauss
        return {
            "t": round(t * 1000),
            "ax": round(ax + n(0, ACCEL_NOISE_G), 4),
            "ay": round(ay + n(0, ACCEL_NOISE_G), 4),
            "az": round(az + n(0, ACCEL_NOISE_G), 4),
            "gx": round(gx + self.gyro_bias[0] + n(0, GYRO_NOISE_DPS), 2),
            "gy": round(gy + self.gyro_bias[1] + n(0, GYRO_NOISE_DPS), 2),
            "gz": round(gz + self.gyro_bias[2] + n(0, GYRO_NOISE_DPS), 2),
            "us_mm": us_mm,
            "us_hit": us_hit,
        }

    def _on_step(self, p):
        self.steps_done += 1
        self.distance_m += p["step_m"]
        if self.next_switch is not None and self.steps_done >= self.next_switch:
            self._set_speed(self.rng.choice([s for s in MIXED_FROM if s != self.current]))
            self.next_switch = self.steps_done + self.rng.randint(*MIXED_EVERY)
        if self.steps_target is not None and self.steps_done >= self.steps_target:
            self.target_amp = 0.0
            if self.done_callback:
                self.done_callback(self.steps_done, self.distance_m)

    def _ping(self, pitch, other_pitch):
        """(us_mm, us_hit) of one HC-SR04 ping at the other thigh."""
        rng = self.rng
        if self.us_busy:
            self.us_busy -= 1
            return US_MAX_MM, 0
        offset = US_HEIGHT_M * (math.sin(pitch) - math.sin(other_pitch))   # front-to-back, m
        if abs(offset) < BEAM_HALF_M and rng.random() > US_DROP:
            mm = (self.gap_mm + 20 * (offset / BEAM_HALF_M) ** 2
                  + 8 * math.sin(4 * math.pi * self.phase) + rng.gauss(0, US_NOISE_MM))
            return int(min(max(mm, 20), US_MAX_MM - 1)), 1
        if rng.random() < US_STRAY:
            return int(rng.uniform(150, US_MAX_MM - 1)), 1
        self.us_busy = 1          # no echo: the sensor ignores the next trigger
        return US_MAX_MM, 0


def read_keys(out):
    """Push single keypresses (or typed lines if stdin isn't a terminal) to out."""
    if not sys.stdin.isatty():
        for line in sys.stdin:
            for ch in line.strip():
                out.put(ch)
        return
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            out.put(sys.stdin.read(1))


def main():
    ap = argparse.ArgumentParser(description="Publish FAKE thigh IMU + ultrasonic data over MQTT (testing only).")
    ap.add_argument("--broker", default=BROKER)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--prefix", default=PREFIX, help=f"topic prefix (default {PREFIX})")
    ap.add_argument("--speed", choices=list(SPEEDS) + ["mixed"], default="normal")
    ap.add_argument("--steps", type=int, default=40, help="steps per walk (the true count)")
    ap.add_argument("--lead-in", type=float, default=2.0,
                    help="seconds of standing after a trial starts before walking")
    ap.add_argument("--continuous", action="store_true", help="walk all the time; ignore trials")
    ap.add_argument("--direct", action="store_true",
                    help="publish <prefix>/raw and status like the board (no board app needed; no recording)")
    ap.add_argument("--subject", default="p1",
                    help="fake person; each name gives a different amplitude, cadence, strap tilt and thigh gap")
    ap.add_argument("--seed", type=int, help="random seed (repeatable noise)")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("--steps must be at least 1")

    raw_topic = f"{args.prefix}/raw" if args.direct else f"{args.prefix}/sim/raw"
    state_topic = f"{args.prefix}/trial/state"
    sim = StepSim(args.subject, random.Random(args.seed))
    events = queue.Queue()        # from MQTT / keys to the sample loop

    def done(steps, distance):
        print(f"  done: {steps} steps, {distance:.1f} m (TRUE count)", flush=True)

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(state_topic)

    last_state = [None]

    def on_message(client, userdata, msg):
        try:
            state = json.loads(msg.payload)
        except ValueError:
            return
        # Retained state from before we connected only sets the baseline.
        prev, last_state[0] = last_state[0], state.get("state")
        if prev is not None and prev != "recording" and state.get("state") == "recording":
            events.put(("trial", state.get("name")))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port)
    client.loop_start()

    keys = queue.Queue()
    is_tty = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if is_tty else None
    if is_tty:
        tty.setcbreak(sys.stdin.fileno())
    threading.Thread(target=read_keys, args=(keys,), daemon=True).start()

    print("FAKE IMU + ULTRASONIC DATA - for testing only")
    print(f"Publishing to {args.broker}:{args.port} {raw_topic} as subject '{args.subject}'")
    if args.continuous:
        print(f"Walking continuously at {args.speed}")
        sim.walk(None, args.speed)
    else:
        print(f"Each trial on {state_topic}: {args.lead_in:g} s standing, then {args.steps} steps at {args.speed}")
    print("Keys: w walk now   s stand   q quit", flush=True)

    start = time.monotonic()
    walk_at = None
    n = 0
    batch = []
    try:
        while True:
            t = n / FS
            while not keys.empty():
                ch = keys.get().lower()
                if ch == "q":
                    return
                if ch == "w":
                    events.put(("key", None))
                if ch == "s":
                    sim.stand()
                    walk_at = None
                    print(f"[{t:7.1f}s] standing", flush=True)
            while not events.empty():
                kind, name = events.get()
                if kind == "trial":
                    print(f"[{t:7.1f}s] trial {name} started", flush=True)
                    walk_at = t + args.lead_in
                else:
                    walk_at = t
            if walk_at is not None and t >= walk_at and not args.continuous:
                walk_at = None
                print(f"[{t:7.1f}s] walking {args.steps} steps at {args.speed}", flush=True)
                sim.walk(args.steps, args.speed, on_done=done)

            time.sleep(max(0.0, start + t - time.monotonic()))
            batch.append(sim.sample(t))
            n += 1
            if len(batch) == BATCH:
                client.publish(raw_topic, json.dumps(batch))
                batch = []
            if args.direct and n % (FS * STATUS_PERIOD_S) == 0:
                client.publish(f"{args.prefix}/status", json.dumps({
                    "wall": round(time.time(), 3), "t": round(t * 1000), "rate_hz": float(FS),
                    "queued": 0, "dropped": 0, "mqtt": True, "mcu": "simulator --direct"}))
    except KeyboardInterrupt:
        pass
    finally:
        if is_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        client.loop_stop()
        client.disconnect()
        print("stopped")


if __name__ == "__main__":
    main()
