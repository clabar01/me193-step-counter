"""Record labeled IMU data from MQTT to CSV files for training.

Subscribes to ME193/ceci/steps/raw (the format is described in simulator.py),
asks for the subject's name, then records one activity at a time.

Keys:
    0 standing   1 walking   2 running   3 biking   4 air-biking
    space  stop and save the current recording (or cancel a countdown)
    q      quit (saves the current recording first)
Pressing a number while recording saves the current recording and starts a new one.

Timed mode (default, --duration 60): after an activity key there is a 5 s
countdown with a beep each second, then a "go" beep. Recording stops by itself
after exactly --duration seconds of samples (by t), plays a distinct end beep
and saves. Nothing is trimmed: the countdown covers getting into position. If
you stop early with space, the last 3 s are trimmed off.

Manual mode (--duration 0): recording starts at once and runs until space; the
first and last 3 s are trimmed off (getting into position / pressing the key).

Beeps are macOS system sounds played with afplay, loud enough to hear away from
the screen (turn the volume up); --no-beep turns them off.

Each recording is saved as data/raw/<subject>_<activity>_<timestamp>.csv with
columns t, ax, ay, az, gx, gy, gz, label, subject. After saving, a quality
check prints a big warning (and plays a warning sound) if:
  - a non-standing recording has more than 15% of its 1 s windows with mean
    |gyro| under 20 deg/s ("REDO: lots of stillness ..."), or
  - any sample reached the accel (+/-16 g) or gyro (+/-2000 deg/s) range limit.

Air-biking keeps its own label here. Training merges it into biking.

--sim saves to data/sim/ instead, with "_sim" in the filename. Use it whenever
simulator.py is the source. Training scripts only ever read data/raw/.

Usage:
    python training/logger.py                 # timed, 60 s
    python training/logger.py --duration 30
    python training/logger.py --duration 0    # manual: space to stop
    python training/logger.py --sim
"""

import argparse
import collections
import csv
import json
import queue
import math
import re
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "ME193/ceci/steps/raw"
FS = 50
SAMPLE_MS = 1000 / FS
MIN_RATE_HZ = 45
RATE_WINDOW_S = 2.0      # rate is averaged over this long; messages carry 5 samples
TRIM_MS = 3000
COUNTDOWN_S = 5
STALL_GRACE_S = 5        # timed mode: stop anyway this long after the duration if data stalls

# Quality check
STILL_DPS = 20.0         # a 1 s window with mean |gyro| under this counts as still
STILL_FRACTION = 0.15    # REDO if more than this fraction of windows are still
ACCEL_LIMIT_G = 16.0     # sensor ranges set in the sketch (+/-8 g / +/-1000 deg/s
GYRO_LIMIT_DPS = 2000.0  # before 2026-10-23 evening)
LIMIT_FRACTION = 0.995   # "hit the limit" = within 0.5% of full scale

# macOS system sounds (afplay); terminal bell elsewhere
SOUND_DIR = Path("/System/Library/Sounds")
SOUND_TICK = "Ping"      # countdown, once per second
SOUND_GO = "Glass"       # recording starts
SOUND_END = "Submarine"  # timed recording finished and saved
SOUND_WARN = "Basso"     # quality check failed
BEEP_VOLUME = 2          # afplay -v; 1 is normal

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "data" / "raw"
SIM_DIR = REPO / "data" / "sim"

FIELDS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
LABELS = {"0": "standing", "1": "walking", "2": "running", "3": "biking", "4": "air-biking"}


class Recorder:
    """Receives samples on the MQTT thread; the main thread reads stats and saves."""

    def __init__(self):
        self.lock = threading.Lock()
        self.arrivals = collections.deque()  # wall time of each sample, for the rate
        self.first_arrival = None
        self.last_t = None
        self.missing = 0         # since the last status report
        self.restarts = 0        # t went backwards since the last status report
        self.bad = 0             # malformed messages since the last status report
        self.activity = None     # None = not recording
        self.rows = []
        self.rec_missing = 0
        self.rec_started = 0.0
        self.duration_ms = None  # timed recording length, None = manual
        self.start_t = None      # t of the first sample in a timed recording
        self.done = False        # timed recording has all its samples

    def on_message(self, client, userdata, msg):
        try:
            samples = json.loads(msg.payload)
            samples = [{k: s[k] for k in FIELDS} for s in samples]
        except (ValueError, TypeError, KeyError):
            with self.lock:
                self.bad += 1
            return
        now = time.monotonic()
        with self.lock:
            if self.first_arrival is None:
                self.first_arrival = now
            for s in samples:
                if self.last_t is not None:
                    dt = s["t"] - self.last_t
                    if dt <= 0:
                        self.restarts += 1
                    elif dt > 1.5 * SAMPLE_MS:
                        n = round(dt / SAMPLE_MS) - 1
                        self.missing += n
                        if self.activity:
                            self.rec_missing += n
                self.last_t = s["t"]
                self.arrivals.append(now)
                if self.activity and not self.done:
                    if self.duration_ms is not None:
                        if self.start_t is None:
                            self.start_t = s["t"]
                        if s["t"] >= self.start_t + self.duration_ms:
                            self.done = True
                            continue
                    self.rows.append(s)

    def start(self, activity, duration_s=0):
        """Start recording; duration_s > 0 records exactly that much by t."""
        with self.lock:
            self.activity = activity
            self.rows = []
            self.rec_missing = 0
            self.rec_started = time.monotonic()
            self.duration_ms = duration_s * 1000 if duration_s > 0 else None
            self.start_t = None
            self.done = False

    def stop(self):
        """Stop recording and return (activity, rows, missing samples, timed)."""
        with self.lock:
            out = (self.activity, self.rows, self.rec_missing, self.duration_ms is not None)
            self.activity = None
            self.rows = []
            self.done = False
            return out

    def report(self):
        """Return (rate Hz, missing, restarts, bad) and reset the counters.

        rate is None until the first full rate window after data starts.
        """
        now = time.monotonic()
        with self.lock:
            while self.arrivals and self.arrivals[0] < now - RATE_WINDOW_S:
                self.arrivals.popleft()
            if self.first_arrival is None or now - self.first_arrival < RATE_WINDOW_S:
                rate = None
            else:
                rate = len(self.arrivals) / RATE_WINDOW_S
            out = (rate, self.missing, self.restarts, self.bad)
            self.missing = self.restarts = self.bad = 0
            return out


def clean_name(name):
    """Lowercase, and only letters, digits and '-', so '_' still splits filenames."""
    return re.sub(r"[^a-z0-9-]+", "-", name.strip().lower()).strip("-")


def save(activity, rows, missing, timed, subject, sim, beeper, early_stop=True):
    """Save a recording. Manual recordings lose 3 s at both ends; timed ones
    lose nothing, or only the last 3 s if stopped early with a key."""
    if not rows:
        say(f"{activity}: no samples received, nothing saved")
        return
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    lo = t0 if timed else t0 + TRIM_MS
    hi = t1 - TRIM_MS if (early_stop or not timed) else t1
    kept = [r for r in rows if lo <= r["t"] <= hi]
    if not kept:
        say(f"{activity}: only {(t1 - t0) / 1000:.1f} s, too short after trimming, nothing saved")
        return

    out_dir = SIM_DIR if sim else RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{subject}_{activity}_{stamp}{'_sim' if sim else ''}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["label", "subject"])
        w.writeheader()
        for r in kept:
            w.writerow({**r, "label": activity, "subject": subject})

    secs = len(kept) / FS
    trimmed = "" if (timed and not early_stop) else " after trimming"
    msg = f"saved {path.relative_to(REPO)}: {len(kept)} samples ({secs:.1f} s){trimmed}"
    if missing:
        msg += f", WARNING {missing} samples missing"
    say(msg)

    problems = quality_check(activity, kept)
    for title, detail in problems:
        banner(title, detail)
    if problems:
        beeper(SOUND_WARN)


def quality_check(activity, rows):
    """Return a list of (title, detail) problems with a saved recording."""
    problems = []

    if activity != "standing":
        windows = collections.defaultdict(list)
        t0 = rows[0]["t"]
        for r in rows:
            windows[(r["t"] - t0) // 1000].append(math.sqrt(r["gx"] ** 2 + r["gy"] ** 2 + r["gz"] ** 2))
        full = [w for w in windows.values() if len(w) >= 0.8 * FS]   # skip a partial last window
        still = sum(1 for w in full if sum(w) / len(w) < STILL_DPS)
        if full and still / len(full) > STILL_FRACTION:
            problems.append((f"REDO: lots of stillness in this {activity} recording",
                             f"{still} of {len(full)} 1 s windows ({still / len(full):.0%}) have mean "
                             f"|gyro| under {STILL_DPS:.0f} deg/s (limit {STILL_FRACTION:.0%})"))

    a_lim, g_lim = LIMIT_FRACTION * ACCEL_LIMIT_G, LIMIT_FRACTION * GYRO_LIMIT_DPS
    n_acc = sum(1 for r in rows if max(abs(r["ax"]), abs(r["ay"]), abs(r["az"])) >= a_lim)
    n_gyr = sum(1 for r in rows if max(abs(r["gx"]), abs(r["gy"]), abs(r["gz"])) >= g_lim)
    if n_acc or n_gyr:
        problems.append(("RANGE LIMIT: the sensor saturated in this recording",
                         f"{n_acc} samples at the +/-{ACCEL_LIMIT_G:g} g accel limit, "
                         f"{n_gyr} at the +/-{GYRO_LIMIT_DPS:g} deg/s gyro limit"))
    return problems


def banner(title, detail):
    line = "#" * 78
    print("\r" + line)
    print(f"#  {title}".ljust(77) + "#")
    print(f"#  {detail}".ljust(77) + "#")
    print(line, flush=True)


def make_beeper(enabled):
    """Return beep(name): plays a macOS system sound without blocking."""
    have_afplay = shutil.which("afplay") is not None

    def beep(name):
        if not enabled:
            return
        sound = SOUND_DIR / f"{name}.aiff"
        if have_afplay and sound.exists():
            subprocess.Popen(["afplay", "-v", str(BEEP_VOLUME), str(sound)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)
    return beep


def say(msg):
    """Print a line above the status line."""
    print("\r" + msg.ljust(78), flush=True)


def read_keys(out):
    """Push single keypresses (or typed lines if stdin isn't a terminal) to out."""
    if not sys.stdin.isatty():
        for line in sys.stdin:
            for ch in line.rstrip("\n"):
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


def main():
    ap = argparse.ArgumentParser(description="Record labeled IMU data from MQTT to CSV.")
    ap.add_argument("--broker", default=BROKER)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--sim", action="store_true",
                    help="data comes from simulator.py: save to data/sim/ with '_sim' in the name")
    ap.add_argument("--duration", type=float, default=60,
                    help="timed recording length in seconds after a 5 s countdown; 0 = manual (space to stop)")
    ap.add_argument("--no-beep", action="store_true", help="don't play sounds")
    args = ap.parse_args()
    beep = make_beeper(not args.no_beep)

    if args.sim:
        print("SIM MODE: saving to data/sim/ (never used for training)")
    subject = ""
    while not subject:
        subject = clean_name(input("Subject name: "))
    print(f"Recording subject '{subject}' to {(SIM_DIR if args.sim else RAW_DIR).relative_to(REPO)}/")

    rec = Recorder()
    client = make_client()
    client.on_message = rec.on_message
    client.on_connect = lambda c, *a: c.subscribe(args.topic)  # resubscribe after reconnects
    client.connect(args.broker, args.port)
    client.loop_start()

    keys = queue.Queue()
    is_tty = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if is_tty else None
    if is_tty:
        tty.setcbreak(sys.stdin.fileno())
    threading.Thread(target=read_keys, args=(keys,), daemon=True).start()

    print("Keys: 0 standing  1 walking  2 running  3 biking  4 air-biking  space stop  q quit")
    if args.duration > 0:
        print(f"Timed mode: {COUNTDOWN_S} s countdown, then {args.duration:g} s recording")
    else:
        print("Manual mode: space stops the recording")

    countdown = None     # (activity, monotonic start) while counting down
    beeps_done = 0
    next_report = time.monotonic() + 1.0
    try:
        while True:
            while not keys.empty():
                ch = keys.get().lower()
                if ch in LABELS or ch in (" ", "q"):
                    if countdown:
                        say(f"{countdown[0]}: countdown cancelled")
                        countdown = None
                    if rec.activity:
                        save(*rec.stop(), subject, args.sim, beep)
                    if ch == "q":
                        return
                    if ch in LABELS:
                        if args.duration > 0:
                            countdown = (LABELS[ch], time.monotonic())
                            beeps_done = 0
                            say(f"get ready: {LABELS[ch]} starts in {COUNTDOWN_S} s")
                        else:
                            rec.start(LABELS[ch])
                            say(f"recording {LABELS[ch]}")

            if countdown:
                elapsed = time.monotonic() - countdown[1]
                if beeps_done < COUNTDOWN_S and elapsed >= beeps_done:
                    beep(SOUND_TICK)
                    beeps_done += 1
                if elapsed >= COUNTDOWN_S:
                    rec.start(countdown[0], args.duration)
                    beep(SOUND_GO)
                    say(f"recording {countdown[0]} for {args.duration:g} s")
                    countdown = None

            if rec.activity and rec.duration_ms is not None:
                stalled = time.monotonic() - rec.rec_started > args.duration + STALL_GRACE_S
                if rec.done or stalled:
                    if stalled and not rec.done:
                        say(f"WARNING: data stalled, stopping {rec.activity} early")
                    save(*rec.stop(), subject, args.sim, beep, early_stop=False)
                    beep(SOUND_END)

            if time.monotonic() >= next_report:
                next_report += 1.0
                rate, missing, restarts, bad = rec.report()
                if missing:
                    say(f"WARNING: {missing} samples missing (gaps in t)")
                if restarts:
                    say("WARNING: t went backwards, did the Uno Q restart?")
                if bad:
                    say(f"WARNING: {bad} malformed messages ignored")
                if rec.first_arrival is None:
                    rate_txt = f"waiting for data on {args.topic}"
                elif rate is None:
                    rate_txt = "measuring rate..."
                else:
                    rate_txt = f"{rate:4.1f} Hz"
                    if rate < MIN_RATE_HZ:
                        rate_txt += f"  LOW RATE (<{MIN_RATE_HZ} Hz)"
                if countdown:
                    left = COUNTDOWN_S - (time.monotonic() - countdown[1])
                    status = f"GET READY {countdown[0]} in {max(left, 0):.0f}  {rate_txt}"
                elif rec.activity:
                    secs = time.monotonic() - rec.rec_started
                    of = f"/{args.duration:g}" if rec.duration_ms is not None else ""
                    status = f"REC {rec.activity} {secs:5.1f}{of} s  {len(rec.rows)} samples  {rate_txt}"
                else:
                    status = f"idle  {rate_txt}"
                print("\r" + status.ljust(78), end="", flush=True)

            time.sleep(0.05)
    except KeyboardInterrupt:
        if rec.activity:
            save(*rec.stop(), subject, args.sim, beep)
    finally:
        if is_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        client.loop_stop()
        client.disconnect()
        print("\nstopped")


if __name__ == "__main__":
    main()
