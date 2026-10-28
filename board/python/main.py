"""me193_steps: forward IMU samples from the sketch to MQTT, and classify them live.

The sketch reads the MPU-6050 at 50 Hz and calls Bridge.notify("imu_sample",
t, ax, ay, az, gx, gy, gz) for every sample. This side batches them and
publishes to RAW_TOPIC in the format training/logger.py and training/simulator.py
use:
    ~10 messages/s, each a JSON list of 5 samples:
    [{"t": 1240, "ax": 0.98, "ay": 0.01, "az": 0.12,
      "gx": -0.4, "gy": 12.7, "gz": 0.9}, ...]
    t is integer ms since the sketch started; accel in g, gyro in deg/s.

MQTT never slows the sensor: the sketch's notify is fire-and-forget, the Bridge
handler only appends to a batch, and a separate thread publishes from a bounded
queue. While MQTT is down (e.g. WiFi drop) up to QUEUE_SECONDS of batches are
kept and sent on reconnect; older ones are dropped, which shows up as a gap in
t. paho reconnects by itself (connect_async + loop_start).

Live classification (live.py): every 0.5 s the last 2 s of samples are
despiked and turned into features (features.py, shared with training), then
Model 1 (models/model1.json) and Model 2 (models/model2_rules.py, majority vote
over its last 3 predictions) run on a worker thread. Each result goes to
PRED_TOPIC, and Model 2's label goes to the sketch (Bridge.call
"show_activity") for the LED matrix icon. If the models can't be loaded the
app logs why and keeps forwarding raw samples.

Every STATUS_PERIOD_S this also publishes STATUS_TOPIC:
    {"wall": <board Unix time>, "t": <last sample t>, "rate_hz": ...,
     "queued": <batches waiting>, "dropped": ..., "mqtt": true/false,
     "mcu": <last sketch status line>, "live": true/false, "model2": <last label>}
"wall" and "t" let the laptop estimate end-to-end lag.
"""
import collections
import json
import os
import threading
import time
from queue import Full, Queue

import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge, Logger

logger = Logger("me193_steps")

BROKER = ("broker.hivemq.com", 1883)
# STEPS_TOPIC_PREFIX is only set when board/run_board_app.py runs this file on
# the laptop for testing; on the board the topics are the real ones.
TOPIC_PREFIX = os.environ.get("STEPS_TOPIC_PREFIX", "ME193/ceci/steps")
RAW_TOPIC = f"{TOPIC_PREFIX}/raw"
STATUS_TOPIC = f"{TOPIC_PREFIX}/status"
PRED_TOPIC = f"{TOPIC_PREFIX}/pred"
LED_REFRESH_S = 5.0         # re-send the icon this often in case the sketch restarted

FS = 50
BATCH = 5                   # samples per message -> 10 messages/s
QUEUE_SECONDS = 30          # buffer this much while MQTT is down
STATUS_PERIOD_S = 5.0

batch = []
queue = collections.deque(maxlen=QUEUE_SECONDS * FS // BATCH)
queue_ready = threading.Event()
lock = threading.Lock()

received = 0                # samples from the sketch since the last status
dropped = 0                 # batches dropped because the queue was full
last_t = None
last_mcu_status = ""
mqtt_up = False

try:
    # imported here, not at the top: if numpy or the model files are missing,
    # the app must still forward raw samples for the logger
    from live import LiveClassifier
    live = LiveClassifier()
    logger.info(f"models loaded: model1 on {live.model1['feature']} "
                f"({', '.join(live.model1['classes'])}), model2 classes {live.model2.CLASSES}")
except Exception as e:      # missing/broken model files: keep forwarding raw data
    live = None
    logger.error(f"live classification OFF, could not load models: {e}")
work = Queue(maxsize=4)   # buffer snapshots waiting to be classified
last_label = None


def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_up
    mqtt_up = not reason_code.is_failure
    logger.info(f"MQTT connected to {BROKER[0]} ({reason_code})")
    queue_ready.set()   # flush anything queued while we were down


def on_disconnect(client, userdata, flags, reason_code, properties):
    global mqtt_up
    if mqtt_up:
        logger.warning(f"MQTT disconnected ({reason_code}); reconnecting")
    mqtt_up = False


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.reconnect_delay_set(1, 10)
client.connect_async(*BROKER, keepalive=30)
client.loop_start()


def imu_sample(t, ax, ay, az, gx, gy, gz):
    """Bridge handler, once per sample. Must stay fast: no I/O here."""
    global received, dropped, last_t
    sample = {"t": int(t),
              "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
              "gx": round(gx, 2), "gy": round(gy, 2), "gz": round(gz, 2)}
    with lock:
        received += 1
        last_t = sample["t"]
        batch.append(sample)
        snapshot = live.add(sample) if live else None
        if len(batch) == BATCH:
            if len(queue) == queue.maxlen:
                dropped += 1    # deque drops the oldest batch
            queue.append(json.dumps(batch))
            batch.clear()
            queue_ready.set()
    if snapshot is not None:
        try:
            work.put_nowait(snapshot)
        except Full:        # classifier is behind: skip this snapshot
            pass


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


def classifier_loop():
    """Classify buffer snapshots, publish predictions, drive the LED icon."""
    global last_label
    last_led = 0.0
    while True:
        result = live.classify(work.get())
        if result is None:          # window had a gap in t
            continue
        pred, _ = result
        if mqtt_up:
            client.publish(PRED_TOPIC, json.dumps(pred), qos=0)
        now = time.monotonic()
        if pred["model2"] != last_label or now - last_led >= LED_REFRESH_S:
            last_led = now          # also on failure, so a missing sketch method warns every 5 s, not every 0.5 s
            try:
                Bridge.call("show_activity", pred["model2"])
            except Exception as e:
                logger.warning(f"show_activity failed: {e}")
            if pred["model2"] != last_label:
                logger.info(f"model2: {pred['model2']} (model1: {pred['model1']}, "
                            f"cadence {pred['cadence_hz']} Hz)")
        last_label = pred["model2"]


def status_loop():
    """App loop: log and publish a status line every STATUS_PERIOD_S."""
    global received, dropped
    time.sleep(STATUS_PERIOD_S)
    with lock:
        rate = received / STATUS_PERIOD_S
        n_queued, n_dropped = len(queue), dropped
        received = 0
        dropped = 0
    status = {"wall": round(time.time(), 3), "t": last_t, "rate_hz": rate,
              "queued": n_queued, "dropped": n_dropped, "mqtt": mqtt_up,
              "mcu": last_mcu_status, "live": live is not None, "model2": last_label}
    logger.info(f"rate {rate:.1f} Hz, queued {n_queued}, dropped {n_dropped}, "
                f"MQTT {'up' if mqtt_up else 'DOWN'}")
    if mqtt_up:
        client.publish(STATUS_TOPIC, json.dumps(status), qos=0)


Bridge.provide("imu_sample", imu_sample)
Bridge.provide("imu_status", imu_status)
threading.Thread(target=publisher, daemon=True).start()
if live:
    threading.Thread(target=classifier_loop, daemon=True).start()

App.run(user_loop=status_loop)
