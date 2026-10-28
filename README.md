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
| `board/` | App Lab app `me193_steps` (`sketch/`, `python/`), `deploy_board.sh`, `run_board_app.py`, `tools/i2c_scanner/` |
| `data/raw/` | recordings from `training/logger.py` (gitignored) |
| `data/processed/` | windowed / feature data |
| `training/` | logger, simulator, windowing, evaluation splits |
| `models/` | exported models |
| `analysis/` | plotting scripts; figures in `analysis/figures/` |
| `dashboard/` | live MQTT dashboard: `python dashboard/serve.py` |
| `animation/` | 3D avatar (`avatar.glb`) and `shrink_glb.py` |
| `docs/` | write-up |

MQTT: `broker.hivemq.com:1883`, topics under `ME193/ceci/steps/`.

## Results

_TBD_

## Reflection

_TBD_
