# me193-step-counter

ME193 project: count steps with a PyTorch neural net and estimate the distance
walked. Extends [me193-gait](https://github.com/clabar01/me193-gait), whose board
app, MQTT pipeline, dashboard and 3D avatar this repo starts from.

## Overview

_TBD_

## Hardware

| Part | Notes |
| --- | --- |
| Arduino Uno Q ("AirFour") | Linux side runs Python, MCU side runs the sketch; they talk through Arduino_RouterBridge. |
| MPU-6050 IMU (accel + gyro) | I2C 0x68, front of the mid-thigh. |
| HC-SR04 ultrasonic sensor | From [me193-sumo](https://github.com/clabar01/me193-sumo). On the same thigh strap, pointing at the other thigh. |
| Power bank + fanny pack | Power and carry the Uno Q on the hip. |

There are no foot sensors. Step labels come from trials where the steps are
counted by hand.

## Wiring

MPU-6050: 3.3V, GND, and the Uno Q's SDA/SCL header pins (I2C 0x68).

HC-SR04 (same wiring as me193-sumo). It runs at 5 V and its Echo output is 5 V,
but the Uno Q's MCU pins are 3.3 V, so Echo goes through a 1 kΩ / 2 kΩ divider:

```
HC-SR04 VCC  ── 5V
HC-SR04 GND  ── GND
HC-SR04 Trig ── D2
HC-SR04 Echo ── 1 kΩ ──┬── D3
                       └── 2 kΩ ── GND
```

_TBD: final pin choice and cable routing on the strap._

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Folder | Contents |
| --- | --- |
| `board/` | App Lab app `me193_steps` (`sketch/`, `python/`), `deploy_board.sh`, `pull_recordings.py`, `run_board_app.py`, `tools/i2c_scanner/` |
| `data/raw/` | trial recordings pulled from the board (gitignored) |
| `data/sim/` | trials recorded from the simulator (gitignored, never used for training) |
| `data/processed/` | windowed / feature data |
| `training/` | simulator, windowing, evaluation splits |
| `models/` | exported models |
| `analysis/` | `plot_raw.py` (one trial); figures in `analysis/figures/` |
| `dashboard/` | `trial.html` (record trials, live plots) and `dashboard.html`: `python dashboard/serve.py` |
| `animation/` | 3D avatar (`avatar.glb`) and `shrink_glb.py` |
| `docs/` | write-up |

MQTT: `broker.hivemq.com:1883`, topics under `ME193/ceci/steps/`:

| Topic | From | Contents |
| --- | --- | --- |
| `raw` | board | 50 Hz samples, 5 per message: `t, ax, ay, az, gx, gy, gz, us_mm, us_hit` |
| `status` | board | every 5 s: rate, ultrasonic echo %, MQTT queue, recording state |
| `trial/cmd` | Trial page | `start` / `stop` / `save` / `discard` (fields in `board/python/main.py`) |
| `trial/state` | board | retained: idle / recording / stopped, sample count, the answer to the last command |

This is a public broker: anyone can read these topics or send a trial command.

### Recording trials

1. Power on the board (once `me193_steps` is deployed and set as the default app).
2. `python dashboard/serve.py --lan` and open the printed `phone:` address on your
   phone (same Wi-Fi as the laptop; the phone needs internet for the broker).
3. Fill in the trial (type, speed, known distance for distance trials), press
   **Start**, walk, press **Stop**, enter the steps you counted, **Save**.
   The LED matrix shows the state: footprints = ready, blinking dot =
   recording, ? = waiting for your count, check = saved, X = command refused.
4. `python board/pull_recordings.py` copies finished trials into `data/raw/`
   (`BOARD=arduino@192.168.1.173` if `AirFour.local` doesn't resolve).

Each trial is `<session>_<trial>.csv` (`t_ms, ax, ay, az, gx, gy, gz, us_mm, us_hit`)
plus `<session>_<trial>.json` (type, counted steps, distance, speed, phone steps,
notes, sample count, missing samples, echo %, times). `us_hit = 0` means no echo
within 1 m (the other thigh was out of the beam), and then `us_mm = 1000`. The
ultrasonic values in a row are from the ping fired right after the previous row
(`us_lag_samples: 1`). `python analysis/plot_raw.py data/raw/<name>.csv` plots one.

### Simulator (no hardware)

`training/simulator.py` makes **fake** thigh-IMU and ultrasonic data: the gyro
swings once per stride, the accel spikes at each heel strike, and the ultrasonic
distance dips each time the other thigh passes, with noise, missed echoes and
stray echoes. Each time a trial starts it stands for 2 s, then walks exactly
`--steps` steps at `--speed` and prints the true count. Three terminals:

```bash
python training/simulator.py --speed normal --steps 40     # also slow, fast, jog, run, stairs, mixed
python board/run_board_app.py                              # the real board app, on the laptop
python dashboard/serve.py --trial --query "prefix=ME193/ceci/steps/test"
```

Then record a trial on the Trial page and enter the count the simulator printed.
Everything uses the `ME193/ceci/steps/test` topics, and trials are saved to
`data/sim/`, never `data/raw/`. Other options: `--continuous` (walk all the time),
`--direct` (publish straight to `<prefix>/raw`, for the plots without the board
app), `--subject p2` (a different fake person), `--seed N` (repeatable noise).
`python board/run_board_app.py --icons` prints the LED icons.

## Results

_TBD_

## Reflection

_TBD_
