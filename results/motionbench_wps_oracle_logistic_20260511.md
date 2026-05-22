# MotionBench WPS Oracle Upper Bound

## Gate Subset

| Coverage | N | Base acc | Raw Wan acc | WPS acc |
|---|---|---|---|---|
| 0.2083 | 20 | 0.4500 | 0.4500 | 0.1500 |

## Switch Comparison

| Switch | Acc | Correct/Total | Gain vs base | Helps | Hurts | Net |
|---|---|---|---|---|---|---|
| base everywhere | 0.4583 | 44/96 | 0.0000 | 0 | 0 | 0 |
| raw Wan everywhere | 0.4792 | 46/96 | 0.0208 | 5 | 3 | 2 |
| WPS everywhere | 0.2812 | 27/96 | -0.1771 | 14 | 31 | -17 |
| gate -> raw Wan | 0.4583 | 44/96 | 0.0000 | 1 | 1 | 0 |
| gate -> WPS | 0.3958 | 38/96 | -0.0625 | 3 | 9 | -6 |
| gate -> max(raw Wan, WPS) oracle | 0.4896 | 47/96 | 0.0312 | 4 | 1 | 3 |

## Temporal-Sensitive Oracle

| Coverage | N | Base acc | Raw Wan acc | Raw switch acc | Raw switch gain | Max(base, raw) oracle acc | Oracle gain |
|---|---|---|---|---|---|---|---|
| 0.0833 | 8 | 0.2500 | 0.6250 | 0.4896 | 0.0312 | 0.4896 | 0.0312 |

## Verdict

Gate direction has limited headroom: using the existing gate to switch to raw Wan is 0.4583, while the gate-only max(raw Wan, WPS) oracle is 0.4896. The temporal-sensitive perfect detector raw-Wan switch is 0.4896, a +0.0312 gain over base.
