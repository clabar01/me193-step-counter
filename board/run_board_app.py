"""Run the board's Python app (board/python/main.py) on the laptop, for testing
without the board.

It stands in for the board runtime (arduino.app_utils: App, Bridge, Logger) and
feeds the app's imu_sample Bridge handler from an MQTT raw stream, e.g. the
simulator. The app then buffers, publishes and records trials exactly as it
does on the board, but under --prefix so the real topics are never touched.
LED icon requests (Bridge.call "show_icon") are printed as the ASCII icon read
from board/sketch/sketch.ino when the icon changes.

Trials are saved to --record-dir (default data/sim/, never data/raw/: this is
not real data) and their JSON says where the samples came from.

Usage (three terminals):
    python training/simulator.py --speed normal --steps 40
    python board/run_board_app.py
    python dashboard/serve.py --query "prefix=ME193/ceci/steps/test"

    python board/run_board_app.py --icons      # just print the LED icons
"""

import argparse
import json
import os
import re
import runpy
import sys
import threading
import time
import types
from pathlib import Path

import paho.mqtt.client as mqtt

REPO = Path(__file__).resolve().parent.parent

APP_DIR = REPO / "board" / "python"
SKETCH = REPO / "board" / "sketch" / "sketch.ino"
FIELDS = ("t", "ax", "ay", "az", "gx", "gy", "gz", "us_mm", "us_hit")
DEFAULTS = {"us_mm": 1000, "us_hit": 0}     # a source without the ultrasonic = every ping timed out


def read_icons():
    """{label: [8 row strings]} from the ICON_* arrays in the sketch."""
    text = SKETCH.read_text()
    icons = {}
    for name, body in re.findall(r"ICON_(\w+)\[8\]\s*=\s*\{(.*?)\};", text, re.S):
        icons[name.lower()] = re.findall(r'"([.#]{13})"', body)
    return icons


def draw(label, icons):
    rows = icons.get(label)
    if rows is None:
        return f"    (no icon for {label!r}: matrix blank)"
    return "\n".join("    " + "".join("██" if c == "#" else "· " for c in r) for r in rows)


def make_runtime(icons, deadline):
    """A stand-in arduino.app_utils module."""
    handlers = {}

    class Logger:
        def __init__(self, name):
            self.name = name

        def _log(self, level, msg):
            print(f"{time.strftime('%H:%M:%S')} {level:7s} {self.name}: {msg}", flush=True)

        def info(self, msg):
            self._log("INFO", msg)

        def warning(self, msg):
            self._log("WARNING", msg)

        def error(self, msg):
            self._log("ERROR", msg)

    shown = [None]

    class Bridge:
        @staticmethod
        def provide(name, fn):
            handlers[name] = fn

        @staticmethod
        def call(name, *args):
            if name == "show_icon" and args[0] != shown[0]:
                shown[0] = args[0]
                print(f"LED matrix -> {args[0]}\n{draw(args[0], icons)}", flush=True)
            return None

        @staticmethod
        def notify(name, *args):
            pass

    class App:
        @staticmethod
        def run(user_loop=None):
            while deadline is None or time.monotonic() < deadline:
                if user_loop:
                    user_loop()
                else:
                    time.sleep(0.5)

    module = types.ModuleType("arduino.app_utils")
    module.App, module.Bridge, module.Logger = App, Bridge, Logger
    package = types.ModuleType("arduino")
    package.app_utils = module
    sys.modules["arduino"], sys.modules["arduino.app_utils"] = package, module
    return handlers


def feed(source, handlers):
    """Subscribe to the raw source topic and call the app's imu_sample handler."""
    while "imu_sample" not in handlers:
        time.sleep(0.05)

    def on_message(client, userdata, msg):
        try:
            samples = json.loads(msg.payload)
        except ValueError:
            return
        for s in samples:
            handlers["imu_sample"](*(s.get(k, DEFAULTS.get(k)) for k in FIELDS))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.on_connect = lambda c, *a: c.subscribe(source)
    client.connect("broker.hivemq.com", 1883)
    client.loop_start()


def main():
    ap = argparse.ArgumentParser(description="Run the board app on the laptop against an MQTT raw stream.")
    ap.add_argument("--prefix", default="ME193/ceci/steps/test",
                    help="topic prefix the app publishes under (never the real one by default)")
    ap.add_argument("--source", default=None, help="raw topic to read samples from (default <prefix>/sim/raw)")
    ap.add_argument("--record-dir", type=Path, default=REPO / "data" / "sim",
                    help="where trials are saved (default data/sim/)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--icons", action="store_true", help="print the LED icons from the sketch and exit")
    args = ap.parse_args()

    icons = read_icons()
    if args.icons:
        for label in icons:
            print(f"{label}\n{draw(label, icons)}\n")
        return
    if args.prefix == "ME193/ceci/steps":
        print("WARNING: publishing on the REAL topics; the logger and dashboard will see this data")

    source = args.source or f"{args.prefix}/sim/raw"
    os.environ["STEPS_TOPIC_PREFIX"] = args.prefix
    os.environ["STEPS_RECORD_DIR"] = str(args.record_dir.resolve())
    os.environ["STEPS_DATA_SOURCE"] = f"run_board_app.py on the laptop, fed from MQTT {source}"
    deadline = time.monotonic() + args.seconds if args.seconds else None
    handlers = make_runtime(icons, deadline)
    threading.Thread(target=feed, args=(source, handlers), daemon=True).start()
    print(f"board app on the laptop: reading {source}, publishing under {args.prefix}/", flush=True)

    sys.path.insert(0, str(APP_DIR))       # so main.py finds recorder.py like on the board
    os.chdir(APP_DIR)
    try:
        runpy.run_path(str(APP_DIR / "main.py"), run_name="__main__")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
