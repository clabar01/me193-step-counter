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
| MPU-6050 IMU (accel + gyro) | I2C 0x68, low on the thigh just above the knee. Axis remap measured from `bench_04` (`BODY_R` in the sketch). |
| HC-SR04 ultrasonic sensor | From [me193-sumo](https://github.com/clabar01/me193-sumo). Just above the knee, horizontal with pins up, pointing at the other thigh. |
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
| `data/sim/` | FAKE trials from the simulator (gitignored): `raw/` + `truth/` from `scripts/make_sim_dataset.py`, for testing the pipeline only |
| `data/processed/` | windowed / feature data |
| `training/` | `simulator.py` (+ `sim_template.json`), `step_detector/` (CNN step counter: data, model, losses, training, tests), `step_length/` (step-length MLP + baselines) |
| `scripts/` | `make_sim_dataset.py` (fake dataset), `fit_sim_template.py` (simulator stride from real trials) |
| `models/` | trained models: `step_detector/<sim or real>/` (weights `model.pt` stay local) |
| `analysis/` | `plot_raw.py` (one trial), `plot_us_gyro.py` (gyro vs ultrasonic check); figures in `docs/figures/` |
| `dashboard/` | `trial.html` (record trials, live plots) and `dashboard.html`: `python dashboard/serve.py` |
| `animation/` | 3D avatar (`avatar.glb`) and `shrink_glb.py` |
| `docs/` | write-up, [`step_detector.md`](docs/step_detector.md) (how the CNN works), figures |

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
   (`BOARD=arduino@<board IP>` if `AirFour.local` doesn't resolve).

Each trial is `<session>_<trial>.csv` (`t_ms, ax, ay, az, gx, gy, gz, us_mm, us_hit`)
plus `<session>_<trial>.json` (type, counted steps, distance, speed, phone steps,
notes, sample count, missing samples, echo %, times). `us_hit = 0` means no echo
within 1 m (the other thigh was out of the beam), and then `us_mm = 1000`. The
ultrasonic values in a row are from the ping fired right after the previous row
(`us_lag_samples: 1`). `python analysis/plot_raw.py data/raw/<name>.csv` plots one.

**For training:** the other thigh reads 16–100 mm. Echoes beyond about 150 mm
are walls and furniture (e.g. a steady ~560 mm band in `A_01`) and depend on the
room, so clip `us_mm` at 150 mm (or treat anything farther as no echo) before
making features; never use the far readings themselves. Skip `bench_*` trials:
they are hardware checks, and `bench_01`/`bench_04` predate the current axis
remap.

### Simulator (no hardware)

`training/simulator.py` makes **fake** knee-mount IMU and ultrasonic data in the
current `BODY_R` frame. Walking plays back the average real stride from bench_04
and A_01 (`training/sim_template.json`, from `scripts/fit_sim_template.py`), with
stride-to-stride variation, a heel-strike accel spike per step, and close
ultrasonic bursts (60-120 ms) each time the thighs pass, from both legs. Standing
gives a steady 17-30 mm echo or none, and walls show up as far echoes that come
and go. Each new trial gets a random strap offset. Each time a trial starts it
stands for 2 s, then walks exactly `--steps` steps at `--speed` and prints the
true count. Three terminals:

```bash
python training/simulator.py --speed normal --steps 40     # also slow, fast, jog, mixed
python board/run_board_app.py                              # the real board app, on the laptop
python dashboard/serve.py --trial --query "prefix=ME193/ceci/steps/test"
```

Then record a trial on the Trial page and enter the count the simulator printed.
Everything uses the `ME193/ceci/steps/test` topics, and trials are saved to
`data/sim/`, never `data/raw/`. Other options: `--continuous` (walk all the time),
`--direct` (publish straight to `<prefix>/raw`, for the plots without the board
app), `--seed N` (repeatable noise).
`python board/run_board_app.py --icons` prints the LED icons.

### Step detector (CNN)

```bash
python scripts/make_sim_dataset.py                          # FAKE sessions A/B/C -> data/sim/raw, truth in data/sim/truth
python training/step_detector/train.py                      # train + evaluate on them
python training/step_detector/train.py --data-dir data/raw  # the same on real trials, once recorded
```

A small 1D CNN (17k weights) outputs a step mass per sample. It is warm-started
on step times from the gyro stride detector, then trained on each trial's
counted steps only. Sessions A and B train/validate and session C is the test
set. The gyro detector alone is the baseline. Results, the model and
normalization go to `models/step_detector/<sim|real>/` and figures to
`docs/figures/step_detector/<sim|real>/`. How it works, in plain language:
[docs/step_detector.md](docs/step_detector.md).

### Step length

```bash
python training/step_length/train.py        # Session B distance trials in data/raw
```

Each distance trial is split into steps with the gyro stride detector, and
each step gets five features (step time, thigh swing, accel range, closest
ultrasonic, ultrasonic burst). A small MLP (5 -> 16 -> 16 -> 1) predicts each
step's length and is trained only on whole-trial distances. It is compared,
leave-one-trial-out, with a fixed step length, Weinberg and a pendulum model.
Results and the model go to `models/step_length/real/`; explanation and the
honest limits of 8 trials: [docs/step_length.md](docs/step_length.md).

## Results

_TBD_

## Reflection

_TBD_
