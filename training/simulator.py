"""Simulated knee-mount IMU + HC-SR04 samples, for testing without the hardware.

*** FAKE DATA FOR TESTING ONLY. ***
These signals are an approximation built from two real trials (bench_04 and
A_01), not recordings. They are for testing the board app, the dashboard and
the step-detector pipeline end to end. Never put them in data/raw/; results on
simulated data say nothing about accuracy on real walking.

Two ways to use it:
  - live over MQTT (this file's main()): publishes 50 Hz samples so the board
    app, recorder and dashboard can be tested (see "MQTT" below);
  - offline: scripts/make_sim_dataset.py drives StepSim directly to write fake
    trial recordings into data/sim/raw/ in the board's format.

The model (IMU and HC-SR04 just above the knee, current BODY_R frame: x along
the thigh toward the hip, y to the right, z out of the front; standing reads
about (0.95, 0, 0) g and the forward swing is positive gy):
  - Walking plays back the average real stride from training/sim_template.json
    (fitted by scripts/fit_sim_template.py): phase 0 is the forward-swing gy
    peak. Each speed scales its cadence and size. Every stride gets its own
    size and length, and smooth noise is added on top (the real strides differ
    by about the template's stride_spread).
  - Steps are heel strikes: this leg's at phase 0.23 (where gy crosses zero
    after the swing), the other leg's half a stride later. Each adds a short
    accel spike. walk(n) takes exactly n steps; their times are kept in
    step_times_ms.
  - Ultrasonic, from what bench_04 and A_01 show:
      * walking: a close burst (16-95 mm, 60-120 ms, a few samples missing)
        each time the thighs pass, at phases 0.48 and 0.96, i.e. once while
        each leg swings; now and then a pass is missed;
      * standing or sitting: either a steady close echo at 17-30 mm (thighs
        nearly touching) or no close echo at all, chosen per standing period;
      * otherwise room clutter: stretches of far echoes from a wall at
        300-950 mm, or timeouts (us_mm 1000, us_hit 0), plus rare strays;
      * the value in a row is from the ping one sample earlier, like the sketch.
  - Per trial (a new StepSim): a random strap offset (a small rotation of the
    whole frame), cadence and step-length factor, and ultrasonic clutter level.
  - Extras for zero-step and harder trials: fidgeting (small weight shifts and
    leg twitches), sitting down / standing up, and 180-degree turns (rotation
    about the thigh's long axis, shorter and weaker steps).

MQTT (live): by default it publishes to <prefix>/sim/raw, which
board/run_board_app.py feeds into the real board app. --direct publishes to
<prefix>/raw and <prefix>/status itself, for the dashboard plots without the
board app. Each time a trial starts (<prefix>/trial/state goes to "recording")
it stands for --lead-in s, walks exactly --steps steps at --speed, then stands
again, and prints the true count and distance. Keys: w walks now, s stands,
q quits. --continuous walks all the time.
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
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
PREFIX = "ME193/ceci/steps/test"
FS = 50                      # sample rate, Hz
DT = 1.0 / FS
BATCH = 5                    # samples per MQTT message -> 10 messages/s
STATUS_PERIOD_S = 5
TEMPLATE = json.loads((Path(__file__).with_name("sim_template.json")).read_text())

G_STAND = np.array([0.946, 0.0, 0.0])   # standing accel with BODY_R (this MPU-6050 reads ~5% low)
HEEL_STRIKE = TEMPLATE["heel_strike_phase"]            # this leg; the other leg's is +0.5
HEEL_STRIKE_SD = 0.017       # stride-to-stride scatter of the heel strike phase (bench_04, A_01)
THIGH_PASS = TEMPLATE["thigh_pass_phase"]              # [other leg swinging, this leg swinging]
STRAP_DEG = 6.0              # strap offset per trial: up to this rotation about each axis
NOISE_FRAC = 0.6             # smooth noise = this x the real stride-to-stride spread
NOISE_TAU_S = 0.08
ACCEL_WHITE_G = 0.004        # sensor noise standing still (real: ~0.003-0.005 g)
GYRO_WHITE_DPS = 0.15
RAMP_STRIDES = 0.3           # amplitude ramp when starting / stopping

# cadence  steps/min        gyro   size of the gyro stride vs the template (normal)
# accel    size of the accel stride (minus gravity) vs the template
# impact   heel-strike spike size; step_m step length, m
SPEEDS = {
    "slow":   dict(cadence=88,  gyro=0.75, accel=0.6, impact=0.6, step_m=0.60),
    "normal": dict(cadence=102, gyro=1.00, accel=1.0, impact=1.0, step_m=0.72),
    "fast":   dict(cadence=120, gyro=1.25, accel=1.4, impact=1.5, step_m=0.82),
    "jog":    dict(cadence=158, gyro=1.90, accel=2.2, impact=3.0, step_m=1.00),
}
MIXED_FROM = tuple(SPEEDS)
MIXED_EVERY = (6, 12)        # mixed: new speed every this many steps

# Ultrasonic
US_MAX_MM = 1000
PASS_MS = (60, 120)          # close burst length at walking speeds (shorter when jogging)
PASS_MISS = 0.04             # chance a whole pass shows no close echo
PASS_SAMPLE_DROP = 0.15      # chance one sample inside a burst has no echo
PASS_MM = (18, 55)           # typical distance of a pass; samples scatter around it
STAND_CLOSE = 0.7            # chance a standing period shows the steady close echo
STAND_MM = (17, 30)
SIT_MM = (20, 60)
WALL_MM = (300, 950)
STRAY = 0.004                # chance per sample of a stray 100-300 mm echo (arm, clothing)
TURN_STEPS = 3               # a 180-degree turn takes this many steps


def _series(coef_cos, coef_sin, phase):
    k = np.arange(len(coef_cos))
    return np.asarray(coef_cos) @ np.cos(2 * np.pi * k * phase) + np.asarray(coef_sin) @ np.sin(2 * np.pi * k * phase)


def stride_template(phase):
    """Average real stride at this phase (0..1): accel g (3), gyro deg/s (3)."""
    return np.array([_series(c, s, phase) for c, s in zip(TEMPLATE["cos"], TEMPLATE["sin"])])


def rotation(rng, max_deg):
    """Small random rotation: up to max_deg about each axis."""
    ax, ay, az = (math.radians(rng.uniform(-max_deg, max_deg)) for _ in range(3))
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def smoothstep(u):
    """0..1 minimum-jerk ramp."""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class StepSim:
    """One 50 Hz sample at a time. A new StepSim is a new trial (new strap
    offset, cadence and ultrasonic clutter). walk(n, speed) takes exactly n
    steps; their times are appended to step_times_ms."""

    def __init__(self, rng):
        self.rng = rng
        self.strap = rotation(rng, STRAP_DEG)
        self.cadence_factor = rng.gauss(1.0, 0.05)
        self.step_factor = rng.gauss(1.0, 0.05)
        self.gyro_bias = np.array([rng.gauss(0, 0.3) for _ in range(3)])
        self.clutter = rng.uniform(0.0, 1.0)          # share of time the wall echo is there
        self.spread = np.array(TEMPLATE["stride_spread"]) * NOISE_FRAC
        self.noise = np.zeros(6)

        self.phase = 0.0            # strides; phase 0 = forward-swing gy peak
        self.amp = 0.0              # 0 standing .. 1 walking
        self.target_amp = 0.0
        self.params = dict(SPEEDS["normal"])
        self.target_params = dict(SPEEDS["normal"])
        self.speed = None
        self.steps_target = 0
        self.steps_done = 0
        self.distance_m = 0.0
        self.step_times_ms = []
        self.next_event = None      # phase of the next heel strike (on the template)
        self.event_jitter = 0.0     # how early/late that heel strike actually lands
        self.next_switch = None
        self.done_callback = None
        self.turns = {}             # step index -> +1 / -1 (left / right)
        self.turn = None            # (start_t, duration_s, sign) while turning
        self.stride_amp, self.stride_rate = 1.0, 1.0
        self.passes = {}            # thigh-pass index -> its random burst

        self.pitch = 0.0            # posture: thigh flexion, rad (0 standing, ~1.4 sitting)
        self.posture_move = None    # (start_t, duration_s, from, to)
        self.sitting = False
        self.fidgeting = False
        self.fidget = None          # (start_t, duration_s, gyro amplitudes)
        self.next_fidget_t = 0.0

        self.t = 0.0
        self.us_pending = (US_MAX_MM, 0)
        self.wall_mm = rng.uniform(*WALL_MM)
        self.wall_on = rng.random() < self.clutter
        self._new_still_period()

    # ---------- control ----------

    def walk(self, steps, speed, turns=None, on_done=None):
        """Start walking: exactly `steps` steps (None = forever). turns maps a
        step index to +1/-1: a 180-degree turn left/right starts at that step."""
        self.speed = speed
        self.steps_target = steps
        self.steps_done = 0
        self.distance_m = 0.0
        self.done_callback = on_done
        self.turns = dict(turns or {})
        self._set_speed(self.rng.choice(MIXED_FROM) if speed == "mixed" else speed)
        self.params = dict(self.target_params)
        self.next_switch = self.rng.randint(*MIXED_EVERY) if speed == "mixed" else None
        # Start just before a swing: this leg first (toe-off at ~0.8) or the other leg first.
        self.phase = 0.80 if self.rng.random() < 0.5 else 0.30
        self.next_event = self._event_after(self.phase)
        self.event_jitter = 0.0
        self.target_amp = 1.0

    def stand(self):
        self.target_amp = 0.0       # no further steps are counted

    def sit_down(self, duration_s):
        self.sitting = True
        self.posture_move = (self.t, duration_s, self.pitch, math.radians(self.rng.uniform(75, 90)))
        self._new_still_period()

    def stand_up(self, duration_s):
        self.sitting = False
        self.posture_move = (self.t, duration_s, self.pitch, 0.0)
        self._new_still_period()

    def set_fidget(self, on):
        self.fidgeting = on
        self.next_fidget_t = self.t + self.rng.uniform(0.5, 2.0)

    @property
    def busy(self):
        """Still walking or moving posture (a trial script waits for this)."""
        return self.amp > 0 or self.target_amp > 0 or self.posture_move is not None

    def _set_speed(self, name):
        self.current = name
        self.target_params = dict(SPEEDS[name])

    def _freq(self):
        """Strides per second."""
        return self.params["cadence"] / 120 * self.cadence_factor * self.stride_rate

    @staticmethod
    def _event_after(phase):
        """Phase of the first heel strike (this leg or the other) after `phase`."""
        base = math.floor(phase)
        for k in range(3):
            for e in (HEEL_STRIKE, HEEL_STRIKE + 0.5):
                if base + k + e > phase:
                    return base + k + e
        raise AssertionError

    # ---------- one sample ----------

    def sample(self, t):
        """Board-format sample at time t (s): accel g, gyro deg/s, ultrasonic."""
        rng = self.rng
        self.t = t
        for key, target in self.target_params.items():
            self.params[key] += (target - self.params[key]) * min(1.0, DT / 0.5)
        p = self.params
        freq = self._freq()

        ramp = DT * freq / RAMP_STRIDES
        self.amp += max(-ramp, min(ramp, self.target_amp - self.amp))
        walking = self.amp > 0

        acc = np.zeros(3)
        gyr = np.zeros(3)
        turning = 1.0
        if walking:
            old = self.phase
            self.phase += freq * DT
            if math.floor(self.phase) != math.floor(old):
                self.stride_amp = rng.gauss(1.0, 0.06)
                self.stride_rate = rng.gauss(1.0, 0.03)
            if self.target_amp > 0 and self.phase >= self.next_event + self.event_jitter:
                self._on_step(t)
                # Heel strikes alternate legs every half stride on the template; real ones
                # scatter around the gy zero-crossing by ~0.017 stride (sd).
                self.next_event += 0.5
                self.event_jitter = rng.gauss(0, HEEL_STRIKE_SD)
            if self.turn:
                start, dur, sign = self.turn
                u = (t - start) / dur
                if u >= 1:
                    self.turn = None
                else:
                    # 180 degrees about the thigh's long axis (about vertical), half-sine rate
                    gyr[0] += sign * 180 * math.pi / (2 * dur) * math.sin(math.pi * u)
                    turning = 0.7
            tmpl = stride_template(self.phase % 1.0)
            size = self.amp * self.stride_amp * turning
            acc += size * p["accel"] * (tmpl[:3] - G_STAND)
            gyr += size * p["gyro"] * tmpl[3:]
            acc += self._impact(p) * np.array([1.0, 0.1, 0.4])

        # posture (sitting down / standing up) and slow sway when still
        rate = 0.0
        if self.posture_move:
            start, dur, a, b = self.posture_move
            u = (t - start) / dur
            new = a + (b - a) * smoothstep(u)
            rate = (new - self.pitch) / DT
            self.pitch = new
            if u >= 1:
                self.posture_move = None
        sway = math.radians(0.8) * math.sin(2 * math.pi * 0.23 * t) * (1 - self.amp)
        pitch = self.pitch + sway
        acc += G_STAND[0] * np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        gyr[1] += math.degrees(rate)

        if self.fidgeting and not walking:
            gyr += self._fidget(t)

        # smooth stride-to-stride noise (scaled with walking) + white sensor noise
        k = DT / NOISE_TAU_S
        self.noise += -k * self.noise + math.sqrt(2 * k) * np.array([rng.gauss(0, 1) for _ in range(6)])
        acc += self.amp * self.spread[:3] * self.noise[:3]
        gyr += self.amp * self.spread[3:] * self.noise[3:]
        acc = self.strap @ acc + np.array([rng.gauss(0, ACCEL_WHITE_G) for _ in range(3)])
        gyr = self.strap @ gyr + self.gyro_bias + np.array([rng.gauss(0, GYRO_WHITE_DPS) for _ in range(3)])

        us_mm, us_hit = self.us_pending              # last sample's ping, like the sketch
        self.us_pending = self._ping(freq)
        return {
            "t": round(t * 1000),
            "ax": round(float(acc[0]), 4), "ay": round(float(acc[1]), 4), "az": round(float(acc[2]), 4),
            "gx": round(float(gyr[0]), 2), "gy": round(float(gyr[1]), 2), "gz": round(float(gyr[2]), 2),
            "us_mm": us_mm, "us_hit": us_hit,
        }

    def _on_step(self, t):
        p = self.params
        turning = self.turn is not None
        self.steps_done += 1
        self.step_times_ms.append(round(t * 1000))
        self.distance_m += p["step_m"] * self.step_factor * self.rng.gauss(1.0, 0.04) * (0.5 if turning else 1.0)
        if self.steps_done in self.turns:
            dur = TURN_STEPS / (p["cadence"] / 60 * self.cadence_factor)
            self.turn = (t, dur, self.turns[self.steps_done])
        if self.next_switch is not None and self.steps_done >= self.next_switch:
            self._set_speed(self.rng.choice([s for s in MIXED_FROM if s != self.current]))
            self.next_switch = self.steps_done + self.rng.randint(*MIXED_EVERY)
        if self.steps_target is not None and self.steps_done >= self.steps_target:
            self.target_amp = 0.0
            self._new_still_period()
            if self.done_callback:
                self.done_callback(self.steps_done, self.distance_m)

    def _impact(self, p):
        """Heel-strike accel spike (g) around the nearest step that is (or will be) counted."""
        e = round((self.phase - HEEL_STRIKE) * 2) / 2 + HEEL_STRIKE
        d = (self.phase - e) / self._freq()          # seconds from that heel strike
        if abs(d) > 0.1 or self.amp < 0.5:
            return 0.0
        this_leg = abs((e - HEEL_STRIKE) % 1.0) < 0.25
        size = p["impact"] * (0.4 if this_leg else 0.15) * self.stride_amp
        return size * math.exp(-0.5 * (d / 0.025) ** 2)

    def _fidget(self, t):
        """Small weight shifts / leg twitches while standing: well below a swing."""
        rng = self.rng
        if self.fidget is None and t >= self.next_fidget_t:
            amps = np.array([rng.uniform(-50, 50), rng.uniform(-40, 40), rng.uniform(-20, 20)])
            self.fidget = (t, rng.uniform(0.4, 1.0), amps)
        if self.fidget is None:
            return np.zeros(3)
        start, dur, amps = self.fidget
        u = (t - start) / dur
        if u >= 1:
            self.fidget = None
            self.next_fidget_t = t + rng.uniform(1.0, 3.5)
            if rng.random() < 0.3:
                self._new_still_period()             # feet moved: the thigh gap changes
            return np.zeros(3)
        return amps * math.sin(2 * math.pi * u)      # out and back: no net rotation

    # ---------- ultrasonic ----------

    def _new_still_period(self):
        """Standing/sitting: steady close echo (thighs nearly touching) or none."""
        rng = self.rng
        self.still_close = rng.random() < STAND_CLOSE
        self.still_mm = rng.uniform(*(SIT_MM if self.sitting else STAND_MM))

    def _pass_burst(self, freq):
        """(in burst, burst) for the thigh pass nearest the current phase."""
        k_stride = math.floor(self.phase)
        best = None
        for s in (k_stride - 1, k_stride, k_stride + 1):
            for j, c in enumerate(THIGH_PASS):
                key = (s, j)
                if key not in self.passes:
                    scale = min(1.0, 105 / self.params["cadence"]) ** 0.7    # faster legs pass quicker
                    self.passes[key] = dict(
                        center=s + c + self.rng.gauss(0, 0.02),
                        half_s=self.rng.uniform(*PASS_MS) * scale / 2000,
                        miss=self.rng.random() < PASS_MISS * (6 if self.turn else 1),
                        mm=self.rng.uniform(*PASS_MM))
                b = self.passes[key]
                d = abs(self.phase - b["center"]) / freq
                if best is None or d < best[0]:
                    best = (d, b)
        for key in [k for k in self.passes if k[0] < k_stride - 1]:
            del self.passes[key]
        d, b = best
        return d <= b["half_s"] and not b["miss"], b

    def _ping(self, freq):
        rng = self.rng
        if self.amp > 0.5:
            inside, b = self._pass_burst(freq)
            if inside:
                if rng.random() < PASS_SAMPLE_DROP:
                    return US_MAX_MM, 0
                return int(min(max(b["mm"] + rng.gauss(0, 8), 16), 95)), 1
        elif self.amp == 0 and self.still_close:
            if rng.random() < 0.005:                 # small shifts of the stance
                self.still_mm = min(max(self.still_mm + rng.gauss(0, 3), 16), 35 if not self.sitting else 65)
            if rng.random() < 0.98:
                return int(round(self.still_mm + rng.gauss(0, 1.5))), 1
            return US_MAX_MM, 0
        if rng.random() < STRAY:
            return int(rng.uniform(100, 300)), 1
        # room clutter: a wall echo that comes and goes, else timeouts
        if rng.random() < DT / 3.0:                  # re-decide about every 3 s
            self.wall_on = rng.random() < self.clutter
        if self.wall_on:
            self.wall_mm = min(max(self.wall_mm + rng.gauss(0, 4), WALL_MM[0]), WALL_MM[1])
            if rng.random() < 0.85:
                mm = self.wall_mm + rng.gauss(0, 25) if rng.random() < 0.75 else rng.uniform(300, 999)
                return int(min(max(mm, 150), US_MAX_MM - 1)), 1
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
    ap = argparse.ArgumentParser(description="Publish FAKE knee-mount IMU + ultrasonic data over MQTT (testing only).")
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
    ap.add_argument("--seed", type=int, help="random seed (repeatable noise)")
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("--steps must be at least 1")

    raw_topic = f"{args.prefix}/raw" if args.direct else f"{args.prefix}/sim/raw"
    state_topic = f"{args.prefix}/trial/state"
    sim = StepSim(random.Random(args.seed))
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
    print(f"Publishing to {args.broker}:{args.port} {raw_topic}")
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
