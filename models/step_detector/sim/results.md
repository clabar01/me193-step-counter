# Step detector results (sim)

Data: SIMULATED data (scripts/make_sim_dataset.py): pipeline test only, raw/. Train 48, val 13, test 9 trials. Command: `python training/step_detector/train.py`. Kept epoch 386 of up to 400.

MAE = mean absolute count error per trial vs counted_steps; total % = summed count vs summed counted steps.

|  | trials | steps (counted) | gyro detector MAE | gyro detector total % | CNN peaks MAE | CNN peaks total % | CNN sum MAE | CNN sum total % | CNN peaks MAE vs true | F1 CNN | F1 gyro |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 48.0 | 1618.0 | 0.71 | 0.6 | 0.48 | -0.3 | 0.31 | 0.5 | 0.44 | 0.981 | 0.943 |
| val | 13.0 | 450.0 | 0.62 | 0.0 | 0.23 | 0.2 | 0.45 | 0.4 | 0.15 | 0.987 | 0.987 |
| test | 9.0 | 446.0 | 0.89 | -0.4 | 0.56 | -0.7 | 0.99 | -0.9 | 0.44 | 0.981 | 0.986 |
| test: fast | 2.0 | 102.0 | 1.0 | 0.0 | 0.5 | 1.0 | 1.62 | 0.8 | 0.0 | 0.98 | 0.995 |
| test: jog | 2.0 | 94.0 | 1.0 | -2.1 | 0.0 | 0.0 | 0.48 | -0.4 | 0.0 | 1.0 | 0.988 |
| test: normal | 3.0 | 163.0 | 1.0 | -0.6 | 0.67 | -1.2 | 1.19 | -2.2 | 0.67 | 0.988 | 0.988 |
| test: slow | 2.0 | 87.0 | 0.5 | 1.1 | 1.0 | -2.3 | 0.57 | -0.9 | 1.0 | 0.952 | 0.97 |

## Per trial

| trial | split | speed | type | counted | true | gyro | cnn_peaks | cnn_sum | f1_cnn | f1_gyro |
|---|---|---|---|---|---|---|---|---|---|---|
| A_10 | train | fast | count | 49 | 50 | 50 | 50 | 49.42 | 1.0 | 0.98 |
| A_19 | train | fast | count | 49 | 49 | 50 | 50 | 49.51 | 0.99 | 0.99 |
| A_22 | train | fast | count | 48 | 48 | 48 | 49 | 49.0 | 0.99 | 1.0 |
| A_17 | train | fast | count | 59 | 58 | 58 | 59 | 59.47 | 0.991 | 1.0 |
| A_02 | train | fast | count | 49 | 49 | 48 | 48 | 49.54 | 0.99 | 0.99 |
| A_34 | train | fast | count | 27 | 27 | 26 | 27 | 27.59 | 1.0 | 0.981 |
| A_24 | train | fast | count | 26 | 26 | 26 | 26 | 25.81 | 1.0 | 0.962 |
| A_25 | train | jog | count | 56 | 56 | 56 | 56 | 56.19 | 1.0 | 0.982 |
| A_31 | train | jog | count | 48 | 48 | 48 | 48 | 48.2 | 1.0 | 1.0 |
| A_06 | train | jog | count | 46 | 46 | 46 | 47 | 45.97 | 0.989 | 1.0 |
| A_30 | train | jog | count | 51 | 51 | 52 | 51 | 51.0 | 1.0 | 0.99 |
| A_16 | train | jog | count | 60 | 60 | 60 | 60 | 59.68 | 1.0 | 0.983 |
| A_37 | train | jog | count | 51 | 51 | 52 | 51 | 50.52 | 1.0 | 0.971 |
| A_09 | train | normal | count | 57 | 57 | 58 | 57 | 57.64 | 0.982 | 0.991 |
| A_32 | train | normal | count | 31 | 31 | 32 | 31 | 31.2 | 0.968 | 0.984 |
| A_15 | train | normal | count | 31 | 31 | 32 | 31 | 31.42 | 1.0 | 0.984 |
| A_14 | train | normal | count | 48 | 48 | 48 | 49 | 48.31 | 0.969 | 1.0 |
| A_18 | train | normal | count | 38 | 38 | 38 | 38 | 38.05 | 0.974 | 1.0 |
| A_11 | train | normal | count | 39 | 39 | 40 | 39 | 39.02 | 0.974 | 0.987 |
| A_07 | train | normal | count | 35 | 35 | 34 | 34 | 35.1 | 0.986 | 0.986 |
| A_41 | train | slow | count | 25 | 24 | 24 | 23 | 24.37 | 0.809 | 0.958 |
| A_27 | train | slow | count | 33 | 33 | 32 | 32 | 33.4 | 0.985 | 0.985 |
| A_01 | train | slow | count | 55 | 55 | 56 | 55 | 55.3 | 0.945 | 0.991 |
| A_33 | train | slow | count | 53 | 53 | 52 | 52 | 52.95 | 0.99 | 0.99 |
| A_29 | train | slow | count | 39 | 39 | 40 | 39 | 39.3 | 0.974 | 0.987 |
| A_35 | train | slow | count | 53 | 53 | 54 | 52 | 53.0 | 0.952 | 0.991 |
| A_12 | train | slow | count | 42 | 42 | 42 | 43 | 42.72 | 0.965 | 0.976 |
| A_20 | train | slow | count | 0 | 0 | 2 | 0 | 0.09 | 1.0 | 0.0 |
| A_40 | train | slow | count | 0 | 0 | 0 | 0 | 0.07 | 1.0 | 1.0 |
| A_21 | train | slow | count | 0 | 0 | 2 | 0 | 0.07 | 1.0 | 0.0 |
| A_13 | train | slow | count | 0 | 0 | 0 | 0 | 0.1 | 1.0 | 1.0 |
| A_26 | train | slow | count | 0 | 0 | 0 | 0 | 0.32 | 1.0 | 1.0 |
| B_12 | train | fast | distance | 24 | 24 | 24 | 25 | 23.87 | 0.98 | 0.958 |
| B_18 | train | fast | distance | 23 | 23 | 22 | 22 | 22.55 | 0.978 | 0.978 |
| B_20 | train | fast | distance | 23 | 24 | 24 | 24 | 23.66 | 1.0 | 0.958 |
| B_01 | train | fast | distance | 25 | 25 | 24 | 24 | 24.31 | 0.98 | 0.98 |
| B_14 | train | jog | distance | 19 | 19 | 20 | 19 | 18.97 | 1.0 | 0.974 |
| B_03 | train | jog | distance | 19 | 19 | 18 | 18 | 19.05 | 0.973 | 0.973 |
| B_10 | train | jog | distance | 21 | 21 | 20 | 20 | 20.43 | 0.976 | 0.976 |
| B_02 | train | jog | distance | 19 | 19 | 20 | 20 | 19.68 | 0.974 | 0.974 |
| B_19 | train | normal | distance | 27 | 27 | 28 | 27 | 26.92 | 0.963 | 0.982 |
| B_16 | train | normal | distance | 28 | 28 | 28 | 28 | 28.32 | 0.964 | 1.0 |
| B_06 | train | normal | distance | 31 | 31 | 32 | 31 | 31.54 | 0.968 | 0.984 |
| B_09 | train | normal | distance | 30 | 30 | 30 | 29 | 30.12 | 0.983 | 0.967 |
| B_13 | train | slow | distance | 31 | 31 | 32 | 31 | 31.4 | 0.968 | 0.984 |
| B_11 | train | slow | distance | 35 | 35 | 36 | 34 | 35.09 | 0.986 | 0.986 |
| B_17 | train | slow | distance | 31 | 31 | 30 | 30 | 30.8 | 0.984 | 0.984 |
| B_07 | train | slow | distance | 34 | 34 | 34 | 34 | 34.26 | 0.971 | 0.971 |
| A_28 | val | fast | count | 58 | 58 | 58 | 58 | 56.83 | 1.0 | 0.983 |
| A_23 | val | fast | count | 46 | 46 | 46 | 46 | 47.05 | 1.0 | 0.978 |
| A_36 | val | jog | count | 41 | 41 | 40 | 41 | 40.7 | 1.0 | 0.988 |
| A_03 | val | jog | count | 28 | 28 | 28 | 28 | 28.55 | 1.0 | 0.964 |
| A_08 | val | normal | count | 26 | 26 | 26 | 26 | 26.25 | 0.962 | 1.0 |
| A_04 | val | normal | count | 51 | 51 | 50 | 51 | 51.21 | 0.98 | 0.99 |
| A_39 | val | slow | count | 59 | 59 | 58 | 59 | 58.81 | 0.983 | 0.991 |
| A_38 | val | slow | count | 31 | 31 | 32 | 31 | 31.25 | 0.968 | 0.984 |
| A_05 | val | slow | count | 0 | 0 | 0 | 0 | 0.13 | 1.0 | 1.0 |
| B_15 | val | fast | distance | 25 | 25 | 24 | 24 | 24.72 | 0.98 | 0.98 |
| B_05 | val | jog | distance | 20 | 20 | 20 | 20 | 20.24 | 1.0 | 1.0 |
| B_08 | val | normal | distance | 29 | 29 | 30 | 30 | 29.27 | 0.983 | 0.983 |
| B_04 | val | slow | distance | 36 | 37 | 38 | 37 | 36.99 | 0.973 | 0.987 |
| C_01 | test | jog | test | 37 | 37 | 36 | 37 | 37.31 | 1.0 | 0.986 |
| C_02 | test | fast | test | 49 | 50 | 50 | 50 | 51.04 | 0.98 | 1.0 |
| C_03 | test | jog | test | 57 | 57 | 56 | 57 | 56.36 | 1.0 | 0.991 |
| C_04 | test | slow | test | 40 | 40 | 40 | 39 | 39.03 | 0.937 | 0.95 |
| C_05 | test | normal | test | 27 | 27 | 26 | 26 | 26.05 | 0.981 | 0.981 |
| C_06 | test | slow | test | 47 | 47 | 48 | 46 | 47.17 | 0.968 | 0.989 |
| C_07 | test | fast | test | 53 | 53 | 52 | 53 | 51.79 | 0.981 | 0.99 |
| C_08 | test | normal | test | 43 | 43 | 42 | 43 | 42.48 | 1.0 | 0.988 |
| C_09 | test | normal | test | 93 | 93 | 94 | 92 | 90.91 | 0.984 | 0.995 |
