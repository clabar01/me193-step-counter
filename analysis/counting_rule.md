# Choosing the CNN's counting rule (training trials only)

Written by `analysis/counting_rule.py`. The CNN was retrained 14 times inside the 14 training trials (7 folds x 2 seeds), each time without 2 of them, with the committed run's settings. Each rule was scored on the trials each model never saw. Each model picked its kept epoch on its own training trials. The validation trials and Session C were not used.

Scoring the committed model on its own training trials (last column) would be unfair. There, the count loss fitted sum = counted, so rounding the sum is exact on every trial by construction. On trials the model hadn't seen, it is the worst rule.

Rules marked offline-only run the gyro stride detector on the whole trial, which the board doesn't have. The windowed form (`counting.count_steps`) runs it only on a window around each bout edge, which the board keeps in memory, so offline and live compute the same thing.

## All rules (error = predicted - counted, over 28 held-out trial predictions)

| rule | mean abs error (steps) | bias (steps) | exact trials % | MAE seed 0 | MAE seed 1 | committed model, own training trials |
|---|---|---|---|---|---|---|
| (a) peaks >= 0.5 | 1.50 | -1.43 | 42.86 | 1.36 | 1.64 | 0.71 |
| (a) peaks >= 0.4 | 1.50 | -1.07 | 39.29 | 1.00 | 2.00 | 0.64 |
| (a) peaks >= 0.3 | 1.39 | -0.75 | 46.43 | 0.79 | 2.00 | 0.64 |
| (a) peaks >= 0.2 | 1.46 | -0.54 | 46.43 | 0.64 | 2.29 | 0.64 |
| (a) peaks >= 0.1 | 2.79 | 1.14 | 42.86 | 0.57 | 5.00 | 0.57 |
| (a) peaks >= 0.05 | 2.79 | 1.57 | 46.43 | 0.50 | 5.07 | 0.71 |
| (b) round(sum) | 4.29 | 1.50 | 42.86 | 1.57 | 7.00 | 0.00 |
| (c) edge, bouts >= 1 | 1.07 | -0.14 | 50.00 | 0.86 | 1.29 | 0.29 |
| (c) edge, bouts >= 2 | 1.14 | -0.43 | 50.00 | 1.00 | 1.29 | 0.21 |
| (c) edge, bouts >= 3 | 1.21 | -0.50 | 50.00 | 1.07 | 1.36 | 0.21 |
| (c) edge, bouts >= 4 | 1.29 | -0.57 | 50.00 | 1.21 | 1.36 | 0.21 |
| (c) edge, bouts >= 6 | 1.32 | -0.61 | 50.00 | 1.21 | 1.43 | 0.21 |
| (c) edge-gyro, bouts >= 1 (offline only) | 0.93 | -0.21 | 46.43 | 0.79 | 1.07 | 0.21 |
| (c) edge-gyro, bouts >= 2 (offline only) | 1.11 | -0.39 | 46.43 | 0.93 | 1.29 | 0.21 |
| (c) edge-gyro, bouts >= 3 (offline only) | 1.21 | -0.50 | 46.43 | 1.07 | 1.36 | 0.21 |
| (c) edge-gyro, bouts >= 4 (offline only) | 1.29 | -0.57 | 46.43 | 1.21 | 1.36 | 0.21 |
| (c) edge-gyro, bouts >= 6 (offline only) | 1.36 | -0.64 | 46.43 | 1.21 | 1.50 | 0.21 |
| (c) edge-gyro-windowed (board form) | 0.96 | -0.11 | 46.43 | 0.86 | 1.07 | 0.29 |

## Best setting of each rule, mean abs error per held-out trial

| trial | (c) edge-gyro, bouts >= 1 (offline only) | (c) edge-gyro-windowed (board form) | (c) edge, bouts >= 1 | (a) peaks >= 0.3 | (b) round(sum) |
|---|---|---|---|---|---|
| A_12 | 0.00 | 0.00 | 0.00 | 1.00 | 5.50 |
| A_05 | 4.50 | 4.50 | 5.50 | 6.50 | 8.50 |
| A_13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| A_06 | 0.50 | 0.50 | 0.50 | 0.50 | 1.50 |
| A_03 | 0.50 | 0.50 | 1.00 | 0.50 | 0.00 |
| B_01 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| B_05 | 0.50 | 0.50 | 0.50 | 2.50 | 2.50 |
| B_02 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| B_04 | 0.50 | 0.50 | 0.50 | 0.00 | 0.50 |
| A_07 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| A_08 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 |
| A_01 | 1.00 | 1.00 | 1.00 | 0.00 | 0.50 |
| B_07 | 2.50 | 2.50 | 1.50 | 1.00 | 2.00 |
| A_10 | 1.50 | 2.00 | 3.00 | 5.50 | 36.50 |

**Chosen (best that runs on the board): (c) edge-gyro-windowed (board form).** Frozen in `training/step_detector/counting.py` (`count_steps`) and `board/python/live_steps.py`.
