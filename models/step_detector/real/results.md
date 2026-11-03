# Step detector results (real)

Data: real trials, raw/. Train 14, val 7, test 0 trials. Command: `python training/step_detector/train.py --data-dir data/raw`. Kept epoch 526 of up to 1200.

MAE = mean absolute count error per trial vs counted_steps; total % = summed count vs summed counted steps.

|  | trials | steps (counted) | gyro detector MAE | gyro detector total % | CNN peaks MAE | CNN peaks total % | CNN sum MAE | CNN sum total % |
|---|---|---|---|---|---|---|---|---|
| train | 14.0 | 301.0 | 0.43 | 1.3 | 0.71 | -3.3 | 0.18 | -0.2 |
| val | 7.0 | 180.0 | 0.29 | 1.1 | 1.14 | -3.3 | 0.38 | -0.7 |

## Per trial

| trial | split | speed | type | counted | true | gyro | cnn_peaks | cnn_sum |
|---|---|---|---|---|---|---|---|---|
| A_10 | train | fast | count | 30 |  | 30 | 29 | 30.05 |
| A_07 | train | fast | count | 35 |  | 36 | 34 | 35.28 |
| A_06 | train | normal | count | 20 |  | 20 | 19 | 19.56 |
| A_03 | train | normal | count | 20 |  | 20 | 20 | 20.05 |
| A_01 | train | normal | count | 30 |  | 30 | 29 | 29.83 |
| A_08 | train | slow | count | 25 |  | 26 | 25 | 25.37 |
| A_05 | train | slow | count | 20 |  | 20 | 18 | 19.69 |
| A_13 | train | normal | count | 0 |  | 0 | 0 | 0.05 |
| A_12 | train | normal | count | 0 |  | 0 | 0 | 0.01 |
| B_07 | train | fast | distance | 23 |  | 22 | 22 | 22.75 |
| B_01 | train | normal | distance | 24 |  | 24 | 23 | 23.88 |
| B_04 | train | normal | distance | 25 |  | 26 | 25 | 25.13 |
| B_02 | train | normal | distance | 24 |  | 24 | 23 | 23.86 |
| B_05 | train | slow | distance | 25 |  | 27 | 24 | 24.85 |
| A_04 | val | fast | count | 30 |  | 30 | 29 | 29.98 |
| A_09 | val | normal | count | 30 |  | 30 | 29 | 29.96 |
| A_02 | val | slow | count | 30 |  | 30 | 27 | 28.77 |
| A_11 | val | normal | count | 0 |  | 0 | 0 | 0.01 |
| B_08 | val | fast | distance | 12 |  | 12 | 13 | 12.57 |
| B_03 | val | normal | distance | 37 |  | 38 | 36 | 36.34 |
| B_06 | val | slow | distance | 41 |  | 42 | 40 | 41.15 |
