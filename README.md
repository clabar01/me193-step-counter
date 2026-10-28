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
| VL53L0X time-of-flight sensor | Inner ankle, pointing at the other ankle. |
| 2x FSR | Under each heel. **Training labels only**, never a model input. |
| Power bank + fanny pack | Power and carry the Uno Q on the hip. |

## Wiring

_TBD_

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
