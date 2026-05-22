# Wan-Motion-REPA Actual Adapter Training Summary

## Code

- `experiments/motionbench_wan_motion_repa_adapter_train.py`

This trains a temporal motion adapter with:

```text
visual tokens -> temporal adapter -> QA residual scorer

L = L_QA
  + lambda_align   * Align(StudentMotion, WanTarget)
  + lambda_relation* AlignRelation(StudentSegments, WanSegments)
  + lambda_equiv   * AlignEquivariance(StudentPerturbResponse, WanPerturbResponse)
  + lambda_contrast* TemporalContrast
  + lambda_preserve* PreserveBase
```

Cached `pixel_grid_sequence + flow_grid_sequence` are used as the frozen visual-token stream because no real VideoLLM token dump/checkpoint output was available in the current result files.

## Runs

- `results/motionbench_wan_motion_repa_adapter_20260511_pool1f8/adapter_eval.md`
- `results/motionbench_wan_motion_repa_adapter_20260511_pool2f16/adapter_eval.md`

## Accuracy

| Setting | QA-only | Raw Wan align | Structured | Dynamics | Residual Dynamics | Relation | Equivariance | Multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x1 frames8 | 0.4789 | 0.4684 | 0.4895 | 0.4789 | 0.5211 | 0.4795 | 0.4789 | 0.4789 |
| 2x2 frames16 | 0.4795 | 0.4789 | 0.4895 | 0.4684 | 0.4584 | 0.4689 | 0.4789 | 0.4789 |

## Best Result

Best branch:

```text
1x1 frames8 + residualized_dynamics_relation adapter
```

Result:

```text
50/96 = 0.5211
```

Compared to:

```text
text-only fold baseline: 44/96 = 0.4589
QA-only adapter:         46/96 = 0.4789
raw Wan align adapter:   45/96 = 0.4684
previous raw Wan probe:  46/96 = 0.4792
```

Question type breakdown for the best branch:

| Question Type | Acc |
|---|---:|
| Action Order | 0.5000 |
| Motion Recognition | 0.5000 |
| Motion-related Objects | 0.5833 |
| Repetition Count | 0.5000 |

## Interpretation

The actual adapter training result is different from the earlier direct-feature proxy:

- Direct target-as-feature proxy failed most transformed targets.
- Actual adapter training makes residualized dynamics useful.
- The best branch beats QA-only by +4/96 and text-only by +6/96.
- Structured compact is stable at 47/96 in both 1x1/8 and 2x2/16.
- Raw Wan alignment does not beat QA-only.
- Multi-target did not improve over the best single target.

Current best claim candidate:

```text
Raw Wan alignment is not the best adapter target.
Fold-safe residualized Wan dynamics-relation alignment improves a small temporal adapter on real MotionBench.
```

Remaining caveat:

```text
This is actual adapter training with QA + REPA losses, but the visual-token stream is cached pixel/flow tokens, not a dumped hidden state from Ask-Anything/VideoChat or another full VideoLLM.
```
