"""Publish simulated thigh-IMU samples to MQTT so the dashboard and live
classifier can be tested without the hardware.

*** FAKE DATA FOR TESTING ONLY. ***
These signals are hand-made approximations, not recordings. Never put them in
data/raw/, train on them, or report results from them.

Message format (this is the format the Uno Q will use):
    topic   ME193/ceci/steps/raw
    rate    ~10 messages/s, 5 samples each (50 Hz)
    payload JSON list of samples:
        [{"t": 1240, "ax": 0.98, "ay": 0.01, "az": 0.12,
          "gx": -0.4, "gy": 12.7, "gz": 0.9}, ...]
    t is milliseconds since the stream started, accel is in g, gyro in deg/s.

Simulated activities. The sensor is on ONE thigh, so its swing repeats once
per stride (2 steps). The gyro shows the stride rate; the accel also shows the
step rate because of the sensor leg's strike plus a weaker pulse from the
other foot halfway through the stride.
    activity     stride Hz   gyro (thigh swing)   accel pulses
    standing     -           ~0 (sway only)       none
    walking      ~0.9        ~0.9 Hz              ~1.8 Hz (strike + other foot)
    running      ~1.4        ~1.4 Hz              ~2.8 Hz (strike + other foot)
    biking       ~1.2        ~1.2 Hz              ~1.2 Hz, small and smooth
    air-biking   ~1.1        ~1.1 Hz              ~1.1 Hz, small and smooth

It also publishes a board-style status message every 5 s to
ME193/ceci/steps/status (see board/python/main.py), so the dashboard's
connection indicator works without the board.

--pred also publishes fake predictions once per second to ME193/ceci/steps/pred,
in the format the board will use (see CLAUDE.md):
    {"t": <ms, last sample>, "model1": "walking", "model2": "walking",
     "cadence_hz": 0.9}
model2 is always the true activity (air-biking -> "biking"). model1 calls
biking "walking" about PRED_M1_BIKING_MISS of the time, to mimic its expected
weakness. cadence_hz is the stride rate (0 when standing).

Keys while running:
    s standing   w walking   r running   b biking   a air-biking   q quit

Usage:
    python training/simulator.py
    python training/simulator.py --random --subject p2
    python training/simulator.py --random --pred      # for the dashboard
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
TOPIC = "ME193/ceci/steps/raw"
PRED_TOPIC = "ME193/ceci/steps/pred"
STATUS_TOPIC = "ME193/ceci/steps/status"
PRED_PERIOD_S = 1        # one prediction per 1 s window step, like the board
STATUS_PERIOD_S = 5
PRED_M1_BIKING_MISS = 0.4   # fraction of biking predictions model1 gets wrong (says walking)
FS = 50                  # sample rate, Hz
BATCH = 5                # samples per message -> 10 messages/s
RANDOM_PERIOD_S = 20     # seconds between activity changes with --random
THIGH_RADIUS_M = 0.15    # hip joint to sensor, for tangential accel

# Sensor axes, with the IMU on the front of the mid-thigh:
#   x along the thigh, pointing toward the hip (reads +1 g standing)
#   y lateral, pointing to the wearer's right (y = z x x, right-handed)
#   z out of the front of the thigh
# The thigh pitch is a rotation about +y, so gy = +d(pitch)/dt: positive when
# sitting down or swinging the thigh forward, as on the real sensor.
# 0 deg = thigh vertical,
# 90 deg = thigh horizontal (seated), >90 deg = knee above hip (lying on back
# with legs raised), which puts gravity on -x/+z instead of +x.
#
# freq      stride frequency of the sensor leg (pedal RPM / 60 for biking), Hz
# pitch     mean thigh pitch, deg
# swing     thigh swing amplitude around the mean, deg
# impact    peak of the sensor leg's foot-strike pulse along x, g
# other     other foot's strike, as a fraction of impact, half a stride later
# width     pulse width (std dev), s
ACTIVITIES = {
    "standing":   dict(key="s", freq=0.0, pitch=0,   swing=0,  impact=0.0, other=0.0, width=0.03),
    "walking":    dict(key="w", freq=0.9, pitch=5,   swing=18, impact=0.8, other=0.4, width=0.03),
    "running":    dict(key="r", freq=1.4, pitch=15,  swing=35, impact=2.5, other=0.4, width=0.03),
    "biking":     dict(key="b", freq=1.2, pitch=65,  swing=22, impact=0.1, other=0.0, width=0.12),
    "air-biking": dict(key="a", freq=1.1, pitch=130, swing=25, impact=0.1, other=0.0, width=0.12),
}
KEYS = {p["key"]: name for name, p in ACTIVITIES.items()}

ACCEL_NOISE_G = 0.015
GYRO_NOISE_DPS = 1.0


class ThighImuSim:
    """Generates one 50 Hz sample at a time for the current activity."""

    def __init__(self, subject, rng):
        self.rng = rng
        # The subject name seeds a fixed "person": same name, same person.
        srng = random.Random(zlib.crc32(subject.encode()))
        self.amp_scale = srng.uniform(0.85, 1.15)
        self.cadence_scale = srng.uniform(0.90, 1.10)
        self.mount_roll = math.radians(srng.uniform(-8, 8))  # strap tilt
        self.gyro_bias = [srng.gauss(0, 1.5) for _ in range(3)]
        self.phase = 0.0  # in cycles
        self.set_activity("standing")

    def set_activity(self, name):
        self.name = name
        self.p = ACTIVITIES[name]
        # Each bout gets its own slightly different cadence.
        self.bout_freq = self.p["freq"] * self.cadence_scale * self.rng.uniform(0.95, 1.05)
        self._new_cycle()

    def _new_cycle(self):
        # Every stride / pedal stroke is a little different.
        self.cycle_amp = self.amp_scale * self.rng.uniform(0.88, 1.12)
        self.cycle_freq = self.bout_freq * self.rng.gauss(1.0, 0.03)

    def sample(self, t):
        p = self.p
        dt = 1.0 / FS

        prev_cycle = int(self.phase)
        self.phase += self.cycle_freq * dt
        if int(self.phase) != prev_cycle:
            self._new_cycle()

        w = 2 * math.pi * self.cycle_freq
        theta = 2 * math.pi * self.phase
        swing = math.radians(p["swing"]) * self.cycle_amp

        # Slow postural sway so standing isn't perfectly still.
        sway_w = 2 * math.pi * 0.25
        sway = math.radians(1.0) * math.sin(sway_w * t)

        pitch = math.radians(p["pitch"]) + swing * math.sin(theta) + sway
        pitch_rate = swing * w * math.cos(theta) + math.radians(1.0) * sway_w * math.cos(sway_w * t)
        pitch_acc = -swing * w * w * math.sin(theta)

        # Gravity in the sensor frame, plus tangential accel from the swing.
        ax = math.cos(pitch)
        az = math.sin(pitch) + THIGH_RADIUS_M * pitch_acc / 9.81

        # Sensor leg's strike (or pedal push) at the start of each stride, plus
        # the weaker other-foot strike half a stride later. Mostly along the thigh.
        if p["impact"]:
            frac = self.phase % 1.0
            d_own = min(frac, 1.0 - frac) / self.cycle_freq        # seconds
            d_other = abs(frac - 0.5) / self.cycle_freq
            shape = (math.exp(-0.5 * (d_own / p["width"]) ** 2)
                     + p["other"] * math.exp(-0.5 * (d_other / p["width"]) ** 2))
            pulse = p["impact"] * self.cycle_amp * shape
            ax += pulse
            az += 0.3 * pulse

        # A tilted strap leaks some of z into y.
        r = self.mount_roll
        ay = -az * math.sin(r)
        az = az * math.cos(r)

        gy_true = math.degrees(pitch_rate)
        gx = 0.1 * abs(gy_true) * math.sin(2 * theta)  # small hip rotation
        gy = gy_true * math.cos(r)
        gz = gy_true * math.sin(r)

        n = self.rng.gauss
        return {
            "t": round(t * 1000),
            "ax": round(ax + n(0, ACCEL_NOISE_G), 4),
            "ay": round(ay + n(0, ACCEL_NOISE_G), 4),
            "az": round(az + n(0, ACCEL_NOISE_G), 4),
            "gx": round(gx + self.gyro_bias[0] + n(0, GYRO_NOISE_DPS), 2),
            "gy": round(gy + self.gyro_bias[1] + n(0, GYRO_NOISE_DPS), 2),
            "gz": round(gz + self.gyro_bias[2] + n(0, GYRO_NOISE_DPS), 2),
        }


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


def make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:  # paho-mqtt < 2.0
        return mqtt.Client()


def fake_prediction(sim, t_ms):
    """A prediction message for the current activity (FAKE, for the dashboard)."""
    label = "biking" if sim.name == "air-biking" else sim.name
    model1 = label
    if label == "biking" and random.random() < PRED_M1_BIKING_MISS:
        model1 = "walking"
    cadence = 0.0 if sim.name == "standing" else round(sim.cycle_freq, 2)
    return {"t": t_ms, "model1": model1, "model2": label, "cadence_hz": cadence}


def main():
    ap = argparse.ArgumentParser(description="Publish FAKE thigh-IMU data over MQTT (testing only).")
    ap.add_argument("--broker", default=BROKER)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--subject", default="p1",
                    help="fake person; each name gives a different amplitude, cadence and strap tilt")
    ap.add_argument("--random", action="store_true",
                    help=f"switch to a random activity every {RANDOM_PERIOD_S} s")
    ap.add_argument("--activity", choices=ACTIVITIES, default="standing", help="starting activity")
    ap.add_argument("--pred", action="store_true",
                    help=f"also publish fake model predictions to {PRED_TOPIC}")
    args = ap.parse_args()
    # pred/status topics sit next to the raw topic, so a test --topic
    # (e.g. ME193/ceci/steps/test/raw) never touches the real ones
    base = args.topic[:-len("/raw")] if args.topic.endswith("/raw") else args.topic
    pred_topic, status_topic = f"{base}/pred", f"{base}/status"

    sim = ThighImuSim(args.subject, random.Random())
    sim.set_activity(args.activity)

    client = make_client()
    client.connect(args.broker, args.port)
    client.loop_start()

    keys = queue.Queue()
    is_tty = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if is_tty else None
    if is_tty:
        tty.setcbreak(sys.stdin.fileno())
    threading.Thread(target=read_keys, args=(keys,), daemon=True).start()

    print("FAKE IMU DATA - for testing only")
    print(f"Publishing to {args.broker}:{args.port} {args.topic} as subject '{args.subject}'")
    print("Keys: s standing  w walking  r running  b biking  a air-biking  q quit")
    print(f"[   0.0s] {sim.name}")

    start = time.monotonic()
    next_switch = RANDOM_PERIOD_S
    n = 0
    batch = []
    try:
        while True:
            t = n / FS

            while not keys.empty():
                ch = keys.get().lower()
                if ch == "q":
                    return
                if ch in KEYS:
                    sim.set_activity(KEYS[ch])
                    next_switch = t + RANDOM_PERIOD_S
                    print(f"[{t:6.1f}s] {sim.name}")

            if args.random and t >= next_switch:
                sim.set_activity(random.choice([a for a in ACTIVITIES if a != sim.name]))
                next_switch = t + RANDOM_PERIOD_S
                print(f"[{t:6.1f}s] {sim.name} (random)")

            time.sleep(max(0.0, start + t - time.monotonic()))
            batch.append(sim.sample(t))
            n += 1
            if len(batch) == BATCH:
                client.publish(args.topic, json.dumps(batch))
                batch = []
            if args.pred and n % (FS * PRED_PERIOD_S) == 0:
                client.publish(pred_topic, json.dumps(fake_prediction(sim, round(t * 1000))))
            if n % (FS * STATUS_PERIOD_S) == 0:
                client.publish(status_topic, json.dumps({
                    "wall": round(time.time(), 3), "t": round(t * 1000), "rate_hz": float(FS),
                    "queued": 0, "dropped": 0, "mqtt": True, "mcu": "simulator"}))
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
