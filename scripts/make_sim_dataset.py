"""Write a FAKE dataset of trial recordings from the simulator.

*** FAKE DATA FOR TESTING THE PIPELINE ONLY. ***
Recordings go to data/sim/raw/ (never data/raw/) in exactly the format the
board's recorder writes: <session>_<trial>.csv (t_ms, ax, ay, az, gx, gy, gz,
us_mm, us_hit) + <session>_<trial>.json. The JSON's counted_steps is what a
person would have counted, so now and then it is off by one, like a real
miscount. The true step count and heel-strike times go to data/sim/truth/,
which only evaluation reads; training never sees them.

Sessions (hallway only; speeds slow / normal / fast / jog):
  A  count trials: SESSION_A_WALKS straight passes of 20-60 steps, spread
     evenly over the speeds, plus zero-step trials (standing, fidgeting,
     sitting down and standing up again).
  B  distance trials: one-way passes of a 20 m hallway, 5 per speed. The step
     count follows from the person's step length at that speed, plus noise.
  C  test trials (held out): passes at each speed, plus one back-and-forth
     trial (3 lengths of the hallway with two 180-degree turns).
Each trial stands still for 3-5 s before and after walking, like the real ones.

Usage:
    python scripts/make_sim_dataset.py              # writes data/sim/raw/ + data/sim/truth/
    python scripts/make_sim_dataset.py --seed 2 --out data/sim2
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from simulator import FS, SPEEDS, StepSim  # noqa: E402

FIELDS = ("t_ms", "ax", "ay", "az", "gx", "gy", "gz", "us_mm", "us_hit")
FORMATS = ("{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}", "{:.2f}", "{:.2f}", "{}", "{}")   # as the recorder
SPEED_ORDER = tuple(SPEEDS)                 # slow, normal, fast, jog
SESSION_A_WALKS = 35
HALLWAY_M = 20.0
MISCOUNT = 0.1                              # chance a counted trial is off by one
STEP_LENGTH_SD = 0.05                       # per-trial step-length noise (fraction)


def run(sim, seconds, t0):
    """Advance the simulator by `seconds`; returns the samples and the new time."""
    out = []
    n = round(seconds * FS)
    for i in range(n):
        out.append(sim.sample(t0 + i / FS))
    return out, t0 + n / FS


def run_until_still(sim, t0):
    out = []
    t = t0
    while sim.busy:
        out.append(sim.sample(t))
        t += 1 / FS
    return out, t


def walk_trial(rng, steps, speed, turns=None):
    sim = StepSim(rng)
    t = 0.0
    a, t = run(sim, rng.uniform(3, 5), t)
    sim.walk(steps, speed, turns=turns)
    b, t = run_until_still(sim, t)
    c, t = run(sim, rng.uniform(3, 5), t)
    return a + b + c, sim


def zero_step_trial(rng, kind):
    sim = StepSim(rng)
    t = 0.0
    if kind == "standing":
        samples, t = run(sim, rng.uniform(10, 20), t)
    elif kind == "fidgeting":
        a, t = run(sim, rng.uniform(2, 3), t)
        sim.set_fidget(True)
        b, t = run(sim, rng.uniform(12, 20), t)
        sim.set_fidget(False)
        c, t = run(sim, 2, t)
        samples = a + b + c
    else:                                   # sitting down, then standing up again
        a, t = run(sim, rng.uniform(3, 4), t)
        sim.sit_down(rng.uniform(1.2, 1.8))
        b, t = run(sim, rng.uniform(6, 11), t)
        sim.stand_up(rng.uniform(1.2, 1.8))
        c, t = run(sim, rng.uniform(4, 5), t)
        samples = a + b + c
    return samples, sim


def counted(rng, true_steps):
    """What a person counting by hand writes down: now and then off by one."""
    if true_steps and rng.random() < MISCOUNT:
        return true_steps + rng.choice((-1, 1))
    return true_steps


def write_trial(out, session, trial, samples, sim, rng, kind, speed, notes, counted_steps,
                distance_m=None, start=None):
    raw, truth = out / "raw", out / "truth"
    name = f"{session}_{trial:02d}"
    boot_ms = rng.randrange(60_000, 3_000_000, 20)       # the board's millis() when the trial started
    lines = [",".join(FIELDS)]
    for s in samples:
        row = (s["t"] + boot_ms, s["ax"], s["ay"], s["az"], s["gx"], s["gy"], s["gz"], s["us_mm"], s["us_hit"])
        lines.append(",".join(f.format(v) for f, v in zip(FORMATS, row)))
    (raw / f"{name}.csv").write_text("\n".join(lines) + "\n")
    hits = sum(s["us_hit"] for s in samples)
    stop = start + timedelta(seconds=len(samples) / FS)
    meta = {
        "session": session, "trial": f"{trial:02d}", "type": kind,
        "counted_steps": counted_steps, "distance_m": distance_m, "speed": speed,
        "phone_steps": None, "notes": notes, "csv": f"{name}.csv",
        "samples": len(samples), "seconds": round(len(samples) / FS, 2),
        "first_t_ms": samples[0]["t"] + boot_ms, "last_t_ms": samples[-1]["t"] + boot_ms,
        "missing_samples": 0, "t_resets": 0, "us_hit_pct": round(100 * hits / len(samples), 1),
        "sample_hz": FS, "us_lag_samples": 1,
        "started_at": start.isoformat(timespec="seconds"), "stopped_at": stop.isoformat(timespec="seconds"),
        "saved_at": (stop + timedelta(seconds=8)).isoformat(timespec="seconds"),
        "source": "simulator",
    }
    (raw / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    (truth / f"{name}.json").write_text(json.dumps({
        "name": name, "true_steps": len(sim.step_times_ms),
        "step_t_ms": [t + boot_ms for t in sim.step_times_ms],
        "true_distance_m": round(sim.distance_m, 2), "counted_steps": counted_steps,
    }) + "\n")
    return name, len(sim.step_times_ms), counted_steps


def main():
    ap = argparse.ArgumentParser(description="Write a FAKE trial dataset from the simulator (testing only).")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "sim",
                    help="folder for raw/ and truth/ (default data/sim)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = args.out.resolve()
    if out == (REPO / "data").resolve() or (REPO / "data" / "raw").resolve() in (out, *out.parents):
        raise SystemExit("refusing to write simulated data into data/raw/")
    for sub in ("raw", "truth"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)

    rng = random.Random(args.seed)
    clock = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    made = []

    def save(session, trial, samples, sim, kind, speed, notes, counted_steps, distance_m=None):
        nonlocal clock
        made.append(write_trial(out, session, trial, samples, sim, rng, kind, speed, notes,
                                counted_steps, distance_m, start=clock))
        clock += timedelta(seconds=len(samples) / FS + rng.uniform(30, 90))

    # Session A: count trials, then zero-step trials, in a shuffled order like a real session
    plan = [("walk", SPEED_ORDER[i % 4]) for i in range(SESSION_A_WALKS)]
    plan += [("zero", k) for k in ("standing", "fidgeting", "sitting") for _ in range(2)]
    rng.shuffle(plan)
    for i, (what, arg) in enumerate(plan, start=1):
        if what == "walk":
            steps = rng.randint(20, 60)
            samples, sim = walk_trial(rng, steps, arg)
            save("A", i, samples, sim, "count", arg, f"SIM straight pass, {arg}", counted(rng, steps))
        else:
            samples, sim = zero_step_trial(rng, arg)
            save("A", i, samples, sim, "count", "slow", f"SIM zero steps: {arg}", 0)

    # Session B: one-way 20 m passes, 5 per speed
    plan = [s for s in SPEED_ORDER for _ in range(5)]
    rng.shuffle(plan)
    for i, speed in enumerate(plan, start=1):
        step_m = SPEEDS[speed]["step_m"] * rng.gauss(1.0, STEP_LENGTH_SD)
        steps = round(HALLWAY_M / step_m)
        samples, sim = walk_trial(rng, steps, speed)
        save("B", i, samples, sim, "distance", speed, f"SIM {HALLWAY_M:g} m hallway, {speed}",
             counted(rng, steps), distance_m=HALLWAY_M)

    # Session C (test): 2 passes per speed + one back-and-forth with two turns
    plan = [s for s in SPEED_ORDER for _ in range(2)]
    rng.shuffle(plan)
    for i, speed in enumerate(plan, start=1):
        steps = rng.randint(20, 60)
        samples, sim = walk_trial(rng, steps, speed)
        save("C", i, samples, sim, "test", speed, f"SIM test pass, {speed}", counted(rng, steps))
    per_length = round(HALLWAY_M / (SPEEDS["normal"]["step_m"] * rng.gauss(1.0, STEP_LENGTH_SD)))
    turns = {per_length - 1: 1, 2 * per_length - 1: -1}
    samples, sim = walk_trial(rng, 3 * per_length, "normal", turns=turns)
    save("C", len(plan) + 1, samples, sim, "test", "normal",
         f"SIM back-and-forth, 3 x {HALLWAY_M:g} m with 2 turns", counted(rng, 3 * per_length),
         distance_m=3 * HALLWAY_M)

    off = sum(c != n for _, n, c in made)
    print(f"FAKE dataset: {len(made)} trials in {out.relative_to(REPO)}/raw "
          f"(truth in {out.relative_to(REPO)}/truth), {off} with a miscount")
    for s in "ABC":
        rows = [m for m in made if m[0].startswith(s + "_")]
        print(f"  session {s}: {len(rows)} trials, {sum(n for _, n, _ in rows)} true steps")


if __name__ == "__main__":
    main()
