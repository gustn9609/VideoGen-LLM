# VideoChat2-HD Wan-REPA 5-Fold CV Summary

Dataset: MotionBench high_motion+camera_comp, 96 examples.

Evaluation: stratified 5-fold CV. Each example is in the test split exactly once.

| Setting | Initial | Final | Gain vs initial | Gain vs CE-only |
|---|---:|---:|---:|---:|
| CE-only adapter | 37/96 = 0.3854 | 44/96 = 0.4583 | +7 | - |
| CE + Wan-REPA equivariance | 37/96 = 0.3854 | 46/96 = 0.4792 | +9 | +2 |

## Per-Type Final Accuracy

| Question type | CE-only | CE + Wan-REPA | Delta |
|---|---:|---:|---:|
| Action Order | 7/24 = 0.2917 | 8/24 = 0.3333 | +1 |
| Motion Recognition | 14/24 = 0.5833 | 15/24 = 0.6250 | +1 |
| Motion-related Objects | 14/24 = 0.5833 | 14/24 = 0.5833 | 0 |
| Repetition Count | 9/24 = 0.3750 | 9/24 = 0.3750 | 0 |

## Paired Comparison

| Category | Count |
|---|---:|
| Both correct | 44 |
| CE-only correct, REPA wrong | 0 |
| REPA correct, CE-only wrong | 2 |
| Both wrong | 50 |

Paired mean gain: +0.0208 accuracy.

Bootstrap 95% CI for paired gain: [0.0000, 0.0521].

## Assessment

Wan-REPA gives a small positive incremental gain over CE-only in 5-fold CV: +2 samples out of 96. The gain appears in Action Order and Motion Recognition, not in Motion-related Objects or Repetition Count.

This is better evidence than the previous 24-example split, but still a weak effect. The direction is positive and non-destructive in this run, but the confidence interval touches zero, so this is not yet strong enough to claim a robust VideoLLM improvement without more data or repeated seeds.

Primary outputs:

- CE-only: `results/videochat2_hd_wan_repa_finetune_cv5_ce_only/finetune_eval.md`
- CE + Wan-REPA: `results/videochat2_hd_wan_repa_finetune_cv5_repa_eq/finetune_eval.md`
