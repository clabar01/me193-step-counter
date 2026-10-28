"""Trial recorder: samples -> <session>_<trial>.csv + .json in one folder.

Pure Python (no Arduino imports), so the laptop runs exactly what the board
runs. main.py calls add() from the Bridge handler (cheap: appends to a list),
flush() from a worker thread (writes to disk), and handle() for each command
from the dashboard.

States: idle -> (start) -> recording -> (stop) -> stopped -> (save | discard) -> idle.
While recording, rows go to <name>.csv.part. save writes <name>.json, then
renames the .part file to <name>.csv, so a .csv always has its .json next to
it and board/pull_recordings.py never copies a half-written trial.

CSV columns: t_ms, ax, ay, az, gx, gy, gz, us_mm, us_hit (units in sketch.ino).
"""

import datetime
import json
import math
import re
import threading
from pathlib import Path

FIELDS = ("t_ms", "ax", "ay", "az", "gx", "gy", "gz", "us_mm", "us_hit")
SAMPLE_KEYS = ("t", "ax", "ay", "az", "gx", "gy", "gz", "us_mm", "us_hit")
FORMATS = ("{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}", "{:.2f}", "{:.2f}", "{}", "{}")
SAMPLE_MS = 20
TYPES = ("count", "distance", "test")
SPEEDS = ("slow", "normal", "fast", "jog", "run", "stairs", "mixed")
NAME_RE = re.compile(r"^[A-Za-z0-9-]{1,40}$")    # no "_": it separates session and trial
MAX_NOTES = 1000


class TrialError(ValueError):
    pass


def _now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _name(value, what):
    value = str(value or "").strip()
    if not NAME_RE.match(value):
        raise TrialError(f"{what} must be 1-40 letters, digits or '-' (got {value!r})")
    return value


def _optional_int(value, what):
    if value in (None, ""):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise TrialError(f"{what} must be a whole number (got {value!r})") from None
    if n < 0:
        raise TrialError(f"{what} can't be negative")
    return n


def _optional_float(value, what):
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise TrialError(f"{what} must be a number (got {value!r})") from None
    if not math.isfinite(x) or x <= 0:
        raise TrialError(f"{what} must be more than 0")
    return x


def _notes(value):
    return str(value or "").strip()[:MAX_NOTES]


class Recorder:
    def __init__(self, folder, source="board"):
        self.folder = Path(folder)
        self.source = source
        self.lock = threading.Lock()      # state + pending rows
        self.io_lock = threading.Lock()   # the open file
        self.state = "idle"
        self.meta = None
        self.file = None
        self.rows = []
        self.last_saved = None
        self.last_cmd = None
        self._reset_counts()

    def _reset_counts(self):
        self.samples = 0
        self.hits = 0
        self.first_t = None
        self.last_t = None
        self.missing = 0
        self.t_resets = 0
        self.started_at = self.stopped_at = None
        self.started_wall = None

    # ---------- called from the Bridge handler ----------

    def add(self, sample):
        if self.state != "recording":     # unlocked read: a sample at the edge may go either way
            return
        row = tuple(sample[k] for k in SAMPLE_KEYS)
        with self.lock:
            if self.state != "recording":
                return
            t = row[0]
            if self.last_t is not None:
                dt = t - self.last_t
                if dt <= 0:
                    self.t_resets += 1    # sketch restarted
                elif dt > SAMPLE_MS:
                    self.missing += round(dt / SAMPLE_MS) - 1
            else:
                self.first_t = t
            self.last_t = t
            self.samples += 1
            self.hits += row[8]
            self.rows.append(row)

    # ---------- called from the worker thread ----------

    def flush(self):
        with self.io_lock:
            with self.lock:
                rows, self.rows = self.rows, []
            if rows and self.file:
                self.file.write("".join(",".join(f.format(v) for f, v in zip(FORMATS, r)) + "\n"
                                        for r in rows))
                self.file.flush()

    def info(self):
        with self.lock:
            m = self.meta or {}
            return {
                "state": self.state,
                "name": m.get("name"),
                "session": m.get("session"),
                "trial": m.get("trial"),
                "meta": {k: m.get(k) for k in ("type", "speed", "distance_m", "phone_steps", "notes")}
                        if m else None,
                "samples": self.samples,
                "seconds": round(self.samples * SAMPLE_MS / 1000, 1),
                "missing_samples": self.missing,
                "us_hit_pct": round(100 * self.hits / self.samples, 1) if self.samples else None,
                "last_saved": self.last_saved,
                "last_cmd": self.last_cmd,
            }

    # ---------- commands ----------

    def handle(self, cmd):
        """Run one dashboard command. Returns (ok, message); never raises."""
        name = cmd.get("cmd")
        handlers = {"start": self.start, "stop": self.stop, "save": self.save, "discard": self.discard}
        try:
            if name not in handlers:
                raise TrialError(f"unknown command {name!r}")
            message = handlers[name](cmd)
            ok = True
        except TrialError as e:
            ok, message = False, str(e)
        except OSError as e:
            ok, message = False, f"file error: {e}"
        with self.lock:
            self.last_cmd = {"id": cmd.get("id"), "cmd": name, "ok": ok, "message": message}
        return ok, message

    def _paths(self, name):
        return (self.folder / f"{name}.csv", self.folder / f"{name}.json",
                self.folder / f"{name}.csv.part")

    def _next_trial(self, session):
        used = [int(m.group(1)) for p in self.folder.glob(f"{session}_*")
                if (m := re.match(rf"^{re.escape(session)}_(\d+)\.", p.name))]
        return f"{max(used, default=0) + 1:02d}"

    def start(self, cmd):
        if self.state != "idle":
            raise TrialError("already recording" if self.state == "recording"
                             else "save or discard the last trial first")
        session = _name(cmd.get("session"), "session")
        self.folder.mkdir(parents=True, exist_ok=True)
        trial = _name(cmd.get("trial"), "trial") if cmd.get("trial") else self._next_trial(session)
        ttype = cmd.get("type", "count")
        if ttype not in TYPES:
            raise TrialError(f"type must be one of {', '.join(TYPES)}")
        speed = cmd.get("speed", "normal")
        if speed not in SPEEDS:
            raise TrialError(f"speed must be one of {', '.join(SPEEDS)}")
        distance = _optional_float(cmd.get("distance_m"), "distance")
        if ttype == "distance" and distance is None:
            raise TrialError("a distance trial needs the known distance in m")
        name = f"{session}_{trial}"
        if any(p.exists() for p in self._paths(name)):
            raise TrialError(f"{name} already exists; pick another trial name")

        meta = {"name": name, "session": session, "trial": trial, "type": ttype, "speed": speed,
                "distance_m": distance,
                "phone_steps": _optional_int(cmd.get("phone_steps"), "phone steps"),
                "notes": _notes(cmd.get("notes"))}
        part = self._paths(name)[2]
        with self.io_lock:
            self.file = part.open("w", newline="")
            self.file.write(",".join(FIELDS) + "\n")
        with self.lock:
            self._reset_counts()
            self.rows = []
            self.meta = meta
            self.started_at = _now_iso()
            self.state = "recording"
        return f"recording {name}"

    def stop(self, cmd):
        if self.state != "recording":
            raise TrialError("not recording")
        with self.lock:
            self.state = "stopped"
            self.stopped_at = _now_iso()
        self.flush()
        with self.io_lock:
            self.file.close()
            self.file = None
        return f"stopped {self.meta['name']}: {self.samples} samples; waiting for the step count"

    def save(self, cmd):
        if self.state != "stopped":
            raise TrialError("stop the trial before saving" if self.state == "recording"
                             else "nothing to save")
        m = self.meta
        counted = _optional_int(cmd.get("counted_steps"), "counted steps")
        if m["type"] == "count" and counted is None:
            raise TrialError("a count trial needs your counted steps")
        if self.samples == 0:
            raise TrialError("no samples were recorded (is the sensor streaming?); discard this trial")
        phone = _optional_int(cmd.get("phone_steps"), "phone steps") if "phone_steps" in cmd \
            else m["phone_steps"]
        notes = _notes(cmd["notes"]) if "notes" in cmd else m["notes"]

        csv_path, json_path, part = self._paths(m["name"])
        doc = {
            "session": m["session"],
            "trial": m["trial"],
            "type": m["type"],
            "counted_steps": counted,
            "distance_m": m["distance_m"],
            "speed": m["speed"],
            "phone_steps": phone,
            "notes": notes,
            "csv": csv_path.name,
            "samples": self.samples,
            "seconds": round(self.samples * SAMPLE_MS / 1000, 2),
            "first_t_ms": self.first_t,
            "last_t_ms": self.last_t,
            "missing_samples": self.missing,
            "t_resets": self.t_resets,
            "us_hit_pct": round(100 * self.hits / self.samples, 1),
            "sample_hz": 1000 // SAMPLE_MS,
            "us_lag_samples": 1,     # a row's us_mm/us_hit are from the ping fired after the previous row
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "saved_at": _now_iso(),
            "source": self.source,
        }
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2) + "\n")
        tmp.replace(json_path)
        part.replace(csv_path)
        with self.lock:
            self.state = "idle"
            self.last_saved = m["name"]
            self.meta = None
        return f"saved {csv_path.name} + {json_path.name} ({doc['samples']} samples, {counted} steps)"

    def discard(self, cmd):
        if self.state == "idle":
            raise TrialError("nothing to discard")
        with self.lock:
            self.state = "idle"
        with self.io_lock:
            if self.file:
                self.file.close()
                self.file = None
        name = self.meta["name"]
        self._paths(name)[2].unlink(missing_ok=True)
        with self.lock:
            self.meta = None
            self.rows = []
        return f"discarded {name}"
