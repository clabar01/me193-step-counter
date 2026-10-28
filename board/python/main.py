"""me193_steps: forward IMU + ultrasonic samples to MQTT and record trials.

The sketch reads the MPU-6050 at 50 Hz, pings the HC-SR04 once per sample, and
calls Bridge.notify("imu_sample", t, ax, ay, az, gx, gy, gz, us_mm, us_hit)
for every sample. This side batches them and publishes to RAW_TOPIC in the
format training/simulator.py also uses:
    ~10 messages/s, each a JSON list of 5 samples:
    [{"t": 1240, "ax": 0.98, "ay": 0.01, "az": 0.12,
      "gx": -0.4, "gy": 12.7, "gz": 0.9, "us_mm": 112, "us_hit": 1}, ...]
    t is integer ms since the sketch started; accel in g, gyro in deg/s;
    us_mm/us_hit as described in sketch.ino (us_hit 0 -> us_mm 1000, no echo).

MQTT never slows the sensor: the sketch's notify is fire-and-forget, the Bridge
handler only appends to a batch, and a separate thread publishes from a bounded
queue. While MQTT is down (e.g. WiFi drop) up to QUEUE_SECONDS of batches are
kept and sent on reconnect; older ones are dropped, which shows up as a gap in
t. paho reconnects by itself (connect_async + loop_start).

Trials (recorder.py): the dashboard's Trial page sends commands to CMD_TOPIC
and this side records to RECORD_DIR:
    {"cmd": "start", "session": "...", "trial": "" (= next free number),
     "type": "count", "speed": "normal", "distance_m": null, "phone_steps": null, "notes": ""}
    {"cmd": "stop"}
    {"cmd": "save", "counted_steps": 42, "phone_steps": null, "notes": "..."}
    {"cmd": "discard"}
The recording state goes to STATE_TOPIC (retained, so a page opened later sees
it) on every change and every second while recording, and to the LED matrix
(Bridge.call "show_icon": ready / rec / count / saved / error).

Every STATUS_PERIOD_S this also publishes STATUS_TOPIC:
    {"wall": <board Unix time>, "t": <last sample t>, "rate_hz": ...,
     "queued": <batches waiting>, "dropped": ..., "mqtt": true/false,
     "mcu": <last sketch status line>, "us_hit_pct": ..., "rec": <state>}
"wall" and "t" let the laptop estimate end-to-end lag.

Environment (only set by board/run_board_app.py when this runs on the laptop):
    STEPS_TOPIC_PREFIX   topic prefix instead of ME193/ceci/steps
    STEPS_RECORD_DIR     where recordings go instead of <app>/recordings
    STEPS_DATA_SOURCE    written into each trial's JSON (default "board")
"""
import collections
import json
import os
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge, Logger

from recorder import Recorder

logger = Logger("me193_steps")

BROKER = ("broker.hivemq.com", 1883)
TOPIC_PREFIX = os.environ.get("STEPS_TOPIC_PREFIX", "ME193/ceci/steps")
RAW_TOPIC = f"{TOPIC_PREFIX}/raw"
STATUS_TOPIC = f"{TOPIC_PREFIX}/status"
CMD_TOPIC = f"{TOPIC_PREFIX}/trial/cmd"
STATE_TOPIC = f"{TOPIC_PREFIX}/trial/state"
# <app folder>/recordings, i.e. ~/ArduinoApps/me193_steps/recordings on the
# board if App Lab maps the app folder into the container (check on first run:
# the path is logged at startup and sent in the trial state).
RECORD_DIR = Path(os.environ.get("STEPS_RECORD_DIR",
                                 Path(__file__).resolve().parent.parent / "recordings"))
DATA_SOURCE = os.environ.get("STEPS_DATA_SOURCE", "board")

LED_REFRESH_S = 5.0         # re-send the icon this often in case the sketch restarted
FLASH_S = 3.0               # how long "saved" / "error" stay on the matrix
STATE_PERIOD_S = 1.0        # trial state while recording
FS = 50
BATCH = 5                   # samples per message -> 10 messages/s
QUEUE_SECONDS = 30          # buffer this much while MQTT is down
STATUS_PERIOD_S = 5.0

batch = []
queue = collections.deque(maxlen=QUEUE_SECONDS * FS // BATCH)
queue_ready = threading.Event()
lock = threading.Lock()

received = 0                # samples from the sketch since the last status
us_hits = 0                 # of those, pings with an echo
dropped = 0                 # batches dropped because the queue was full
last_t = None
last_mcu_status = ""
mqtt_up = False

recorder = Recorder(RECORD_DIR, source=DATA_SOURCE)
state_changed = threading.Event()
led_flash = None            # ("saved" | "error", until monotonic time, trial state it belongs to)


def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_up
    mqtt_up = not reason_code.is_failure
    logger.info(f"MQTT connected to {BROKER[0]} ({reason_code})")
    client.subscribe(CMD_TOPIC, qos=1)
    queue_ready.set()       # flush anything queued while we were down
    state_changed.set()     # re-publish the trial state


def on_disconnect(client, userdata, flags, reason_code, properties):
    global mqtt_up
    if mqtt_up:
        logger.warning(f"MQTT disconnected ({reason_code}); reconnecting")
    mqtt_up = False


def on_message(client, userdata, msg):
    """Trial command from the dashboard (paho's thread)."""
    global led_flash
    try:
        cmd = json.loads(msg.payload)
        if not isinstance(cmd, dict):
            raise ValueError("not a JSON object")
    except ValueError as e:
        logger.warning(f"bad trial command {msg.payload[:80]!r}: {e}")
        return
    ok, message = recorder.handle(cmd)
    state = recorder.info()["state"]
    if ok:
        logger.info(f"trial {cmd.get('cmd')}: {message}")
        if cmd.get("cmd") == "save":
            led_flash = ("saved", time.monotonic() + FLASH_S, state)
    else:
        logger.warning(f"trial {cmd.get('cmd')} refused: {message}")
        led_flash = ("error", time.monotonic() + FLASH_S, state)
    state_changed.set()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.reconnect_delay_set(1, 10)
client.connect_async(*BROKER, keepalive=30)
client.loop_start()


def imu_sample(t, ax, ay, az, gx, gy, gz, us_mm, us_hit):
    """Bridge handler, once per sample. Must stay fast: no I/O here."""
    global received, us_hits, dropped, last_t
    sample = {"t": int(t),
              "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
              "gx": round(gx, 2), "gy": round(gy, 2), "gz": round(gz, 2),
              "us_mm": int(us_mm), "us_hit": int(us_hit)}
    recorder.add(sample)
    with lock:
        received += 1
        us_hits += sample["us_hit"]
        last_t = sample["t"]
        batch.append(sample)
        if len(batch) == BATCH:
            if len(queue) == queue.maxlen:
                dropped += 1    # deque drops the oldest batch
            queue.append(json.dumps(batch))
            batch.clear()
            queue_ready.set()


def imu_status(text):
    global last_mcu_status
    last_mcu_status = text
    logger.info(f"sketch: {text}")


def publisher():
    """Publish queued batches in order while MQTT is up."""
    while True:
        queue_ready.wait()
        queue_ready.clear()
        while mqtt_up:
            with lock:
                if not queue:
                    break
                payload = queue[0]
            info = client.publish(RAW_TOPIC, payload, qos=0)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                break           # not connected after all; retry on reconnect
            with lock:
                if queue and queue[0] is payload:
                    queue.popleft()


def led_icon(state):
    # A flash ends early when the trial state changes (e.g. start right after an error).
    if led_flash and time.monotonic() < led_flash[1] and led_flash[2] == state:
        return led_flash[0]
    return {"idle": "ready", "recording": "rec", "stopped": "count"}[state]


def trial_loop():
    """Write recorded rows to disk, publish the trial state, drive the LED."""
    last_state_pub = 0.0
    last_icon, last_led = None, 0.0
    while True:
        state_changed.wait(timeout=0.25)
        changed = state_changed.is_set()
        state_changed.clear()
        try:
            recorder.flush()
        except OSError as e:
            logger.error(f"writing the recording failed: {e}")
        info = recorder.info()
        now = time.monotonic()
        if mqtt_up and (changed or (info["state"] == "recording" and now - last_state_pub >= STATE_PERIOD_S)):
            last_state_pub = now
            info["dir"] = str(RECORD_DIR)
            info["wall"] = round(time.time(), 3)
            client.publish(STATE_TOPIC, json.dumps(info), qos=1, retain=True)
        icon = led_icon(info["state"])
        if icon != last_icon or now - last_led >= LED_REFRESH_S:
            last_led = now      # also on failure, so a missing sketch method warns every 5 s
            try:
                Bridge.call("show_icon", icon)
                last_icon = icon
            except Exception as e:
                logger.warning(f"show_icon failed: {e}")


def status_loop():
    """App loop: log and publish a status line every STATUS_PERIOD_S."""
    global received, us_hits, dropped
    time.sleep(STATUS_PERIOD_S)
    with lock:
        rate = received / STATUS_PERIOD_S
        hit_pct = round(100 * us_hits / received, 1) if received else None
        n_queued, n_dropped = len(queue), dropped
        received = us_hits = dropped = 0
    rec = recorder.info()["state"]
    status = {"wall": round(time.time(), 3), "t": last_t, "rate_hz": rate,
              "queued": n_queued, "dropped": n_dropped, "mqtt": mqtt_up,
              "mcu": last_mcu_status, "us_hit_pct": hit_pct, "rec": rec}
    logger.info(f"rate {rate:.1f} Hz, us hits {hit_pct}%, queued {n_queued}, dropped {n_dropped}, "
                f"MQTT {'up' if mqtt_up else 'DOWN'}, trial {rec}")
    if mqtt_up:
        client.publish(STATUS_TOPIC, json.dumps(status), qos=0)


logger.info(f"recording to {RECORD_DIR} (source {DATA_SOURCE}); commands on {CMD_TOPIC}")
Bridge.provide("imu_sample", imu_sample)
Bridge.provide("imu_status", imu_status)
threading.Thread(target=publisher, daemon=True).start()
threading.Thread(target=trial_loop, daemon=True).start()

App.run(user_loop=status_loop)
