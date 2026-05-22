# MotionBench WPS Oracle Upper Bound

## Gate Subset

| Coverage | N | Base acc | Raw Wan acc | WPS acc |
|---|---|---|---|---|
| 0.2083 | 20 | 0.3000 | 0.3000 | 0.2000 |

## Switch Comparison

| Switch | Acc | Correct/Total | Gain vs base | Helps | Hurts | Net |
|---|---|---|---|---|---|---|
| base everywhere | 0.4583 | 44/96 | 0.0000 | 0 | 0 | 0 |
| raw Wan everywhere | 0.4792 | 46/96 | 0.0208 | 5 | 3 | 2 |
| WPS everywhere | 0.2500 | 24/96 | -0.2083 | 12 | 32 | -20 |
| gate -> raw Wan | 0.4583 | 44/96 | 0.0000 | 1 | 1 | 0 |
| gate -> WPS | 0.4375 | 42/96 | -0.0208 | 4 | 6 | -2 |
| gate -> max(raw Wan, WPS) oracle | 0.5000 | 48/96 | 0.0417 | 5 | 1 | 4 |

## Temporal-Sensitive Oracle

| Coverage | N | Base acc | Raw Wan acc | Raw switch acc | Raw switch gain | Max(base, raw) oracle acc | Oracle gain |
|---|---|---|---|---|---|---|---|
| 0.0833 | 8 | 0.2500 | 0.6250 | 0.4896 | 0.0312 | 0.4896 | 0.0312 |

## Verdict

Gate direction has limited headroom: using the existing gate to switch to raw Wan is 0.4583, while the gate-only max(raw Wan, WPS) oracle is 0.5000. The temporal-sensitive perfect detector raw-Wan switch is 0.4896, a +0.0312 gain over base.
