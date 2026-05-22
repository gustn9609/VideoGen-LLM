# Wan-Motion-REPA Methods Execution Summary

## Implemented

Code:

- `experiments/motionbench_wan_motion_repa_targets.py`
- `experiments/motionbench_wan_motion_repa_eval.py`

Target recipes implemented:

- Step 1 raw Wan alignment baseline proxy
- Step 2 dynamics-relation Wan target
- Step 3 residualized dynamics-relation target with train-fold-only Ridge residualization
- Step 4 relation-only target
- Step 5 equivariance target from reverse/shuffle/first/time-average responses
- Step 6 multi-target proxy by concatenated target and score-level ensemble

Main output directories:

- `results/motionbench_wan_motion_repa_20260511/` for 2x2, 16-frame target
- `results/motionbench_wan_motion_repa_20260511_pool1f16/` for 1x1, 16-frame target
- `results/motionbench_wan_motion_repa_20260511_pool1f8/` for 1x1, 8-frame target

## Target Dimensions

For 2x2, 16-frame Wan target:

| Target | Dim |
|---|---:|
| raw_pooled | 224 |
| structured_compact | 236 |
| dynamics_relation | 702 |
| relation_only | 20 |
| equivariance | 2760 |
| multi_target | 3706 |

## Proxy QA Rerank Results

| Setting | Text | Raw Wan | Structured | Dynamics | Residual Dynamics | Relation | Equivariance | Multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x2 frames16 | 0.4589 | 0.3847 | 0.4053 | 0.3542 | 0.2905 | 0.4589 | 0.3532 | 0.3111 |
| 1x1 frames16 | 0.4589 | 0.4484 | 0.4163 | 0.3542 | 0.4274 | 0.4589 | 0.4047 | 0.4047 |
| 1x1 frames8 | 0.4589 | 0.4800 | 0.4368 | 0.3532 | 0.4074 | 0.4589 | 0.2911 | 0.3105 |

## Multi-Target Score Ensemble

Run: `results/motionbench_wan_motion_repa_20260511_pool1f8/wmrepa_score_ensemble.md`

| Model | Acc | Correct/total |
|---|---:|---:|
| text | 0.4583 | 44/96 |
| pixel | 0.4688 | 45/96 |
| flow | 0.4479 | 43/96 |
| raw_wan | 0.4792 | 46/96 |
| structured | 0.4375 | 42/96 |
| dynamics | 0.3542 | 34/96 |
| residual_dyn | 0.4062 | 39/96 |
| relation | 0.4583 | 44/96 |
| equivariance | 0.2917 | 28/96 |
| multi_target | 0.3125 | 30/96 |
| calibrated_residual_ensemble | 0.4271 | 41/96 |

Best non-held-out weight sweep reached 0.4896, 47/96, but the calibrated held-out ensemble was 0.4271, 41/96.

## Stop-Rule Assessment

- Raw Wan alignment remains the best Wan branch in the strongest setting, 1x1 frames8: 0.4800.
- Dynamics-relation target did not outperform raw Wan or text-only.
- Residualized dynamics-relation did not improve temporal QA overall, despite being fold-safe.
- Relation-only matched text-only but did not add incremental gain.
- Equivariance target was weak as a direct rerank feature.
- Multi-target direct concat and held-out calibrated ensemble did not improve over raw Wan.

Conclusion: under this proxy evaluation, the current Wan-Motion-REPA target transformations do not yet justify replacing raw Wan. If this direction continues, it should move from direct candidate rerank proxy to actual adapter pretraining/evaluation or frame-selection regularization, because the target-as-feature test fails most branches.
