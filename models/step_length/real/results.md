# Step length results (real)

Leave-one-trial-out over 8 distance trials (real). Each trial is predicted by methods fitted on the other 7.

## Mean |distance error| %, by speed (number of trials)

| speed | fixed mean step | Weinberg | pendulum | MLP |
|---|---|---|---|---|
| slow (2) | 7.8 | 2.3 | 6.0 | 1.7 |
| normal (4) | 2.8 | 1.7 | 1.1 | 2.8 |
| fast (2) | 12.9 | 4.2 | 7.8 | 2.2 |
| all (8) | 6.6 | 2.5 | 4.0 | 2.4 |

## Per trial, distance error %

| trial | speed | distance_m | steps | fixed_err% | weinberg_err% | pendulum_err% | mlp_err% |
|---|---|---|---|---|---|---|---|
| B_01 | normal | 19.0 | 24 | -0.9 | -1.1 | 2.4 | -1.8 |
| B_02 | normal | 19.0 | 24 | -0.9 | 0.5 | -1.1 | -3.1 |
| B_03 | normal | 30.0 | 38 | -0.6 | -0.6 | 0.8 | 0.8 |
| B_04 | normal | 19.0 | 26 | 8.8 | 4.4 | 0.2 | 5.4 |
| B_05 | slow | 19.0 | 25 | 3.9 | 0.3 | 3.3 | -1.4 |
| B_06 | slow | 30.0 | 42 | 11.8 | 4.4 | 8.7 | 2.0 |
| B_07 | fast | 19.0 | 22 | -10.1 | -3.2 | -7.7 | -1.6 |
| B_08 | fast | 11.0 | 12 | -15.8 | -5.2 | -7.9 | 2.8 |

Fitted on all trials: fixed 0.7856, weinberg 0.7506, pendulum 1.0371
