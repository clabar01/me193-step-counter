"""Measure the delay from a heel strike / stomp (accel spike) to the step count updating.

Listens to the board's MQTT topics (like the dashboard does):
  <prefix>/raw    50 Hz samples: accel spikes (|a| peaks above --spike-g) are the stomps
  <prefix>/live   every 0.25 s: "new_steps" = board times of newly counted steps
Each counted step is matched to the spike closest to its time (within
--match-ms). Two delays per step:
  detect lag   on the board: when the count changed minus when the spike
               happened, both on the board's sample clock (the CNN's 1.2 s
               look-ahead plus waiting for the next 0.25 s run)
  network      the live message's arrival here minus its "wall" publish time
               (needs the board's and this computer's clocks in sync: NTP)
  count delay  the sum: what you see on the dashboard
Spikes with no counted step (e.g. a stomp in place that the model doesn't take
for a step) are listed too.

Usage: walk or stomp for ~30 s while this runs:
    python analysis/measure_latency.py --seconds 40
    python analysis/measure_latency.py --prefix ME193/ceci/steps/test     # simulator
"""

import argparse
import json
import math
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

BROKER = ("broker.hivemq.com", 1883)


def main():
    ap = argparse.ArgumentParser(description="Delay from accel spike (heel strike/stomp) to the step count.")
    ap.add_argument("--prefix", default="ME193/ceci/steps")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--spike-g", type=float, default=1.6, help="|accel| peak that counts as a stomp")
    ap.add_argument("--match-ms", type=float, default=250, help="max offset between a spike and its step")
    args = ap.parse_args()

    lock = threading.Lock()
    samples = []              # (board t, |a|)
    counted = []              # (board t of step, network delay s, board t newest processed)

    def on_message(client, userdata, msg):
        now = time.time()
        data = json.loads(msg.payload)
        with lock:
            if msg.topic.endswith("/raw"):
                for s in data:
                    samples.append((s["t"], math.sqrt(s["ax"] ** 2 + s["ay"] ** 2 + s["az"] ** 2)))
            else:
                for t in data.get("new_steps", []):
                    counted.append((t, now - data["wall"], data["t"]))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.on_connect = lambda c, u, f, rc, p: (c.subscribe(f"{args.prefix}/raw"), c.subscribe(f"{args.prefix}/live"))
    client.connect(*BROKER)
    client.loop_start()
    print(f"listening on {args.prefix}/raw + /live for {args.seconds:g} s: walk with firm heel strikes or stomp")
    time.sleep(args.seconds)
    client.loop_stop()

    if not samples:
        raise SystemExit("no raw samples arrived: is the board (or simulator) running?")
    t = np.array([s[0] for s in samples], float)
    a = np.array([s[1] for s in samples])
    spikes = [i for i in range(1, len(a) - 1)
              if a[i] >= args.spike_g and a[i] >= a[i - 1] and a[i] > a[i + 1]]
    spikes = [i for k, i in enumerate(spikes) if k == 0 or t[i] - t[spikes[k - 1]] > 300]
    rows, used = [], set()
    for st, network, newest in counted:
        if not len(spikes):
            break
        i = min(spikes, key=lambda j: abs(t[j] - st))
        if abs(t[i] - st) <= args.match_ms:
            used.add(i)
            lag = newest - t[i]
            rows.append((st - t[i], lag, network * 1000, lag + network * 1000))
    print(f"\n{len(counted)} steps counted, {len(spikes)} spikes >= {args.spike_g} g, {len(rows)} matched")
    if rows:
        r = np.array(rows)
        for name, col in (("step time - spike time (ms)", 0), ("detect lag on the board (ms)", 1),
                          ("network to here (ms)", 2), ("count delay, total (ms)", 3)):
            print(f"  {name:30s} median {np.median(r[:, col]):6.0f}   min {r[:, col].min():6.0f}   "
                  f"max {r[:, col].max():6.0f}")
    missed = [i for i in spikes if i not in used and t[i] < t[-1] - 2000]
    if missed:
        print(f"  {len(missed)} spike(s) with no counted step (not taken for a step), at "
              + ", ".join(f"{(t[i] - t[0]) / 1000:.1f} s" for i in missed[:10]))


if __name__ == "__main__":
    main()
