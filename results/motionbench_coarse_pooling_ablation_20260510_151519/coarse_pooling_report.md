# MotionBench Coarse Pooling Diagnostics

| Mode | Pool | Random same-dim | PCA same-dim | Raw ridge | Text-only |
|---|---:|---:|---:|---:|---:|
| all | 1x1 | 0.4088 | 0.4088 | 0.3260 | 0.3468 |
| all | 2x2 | 0.3646 | 0.3582 | 0.3330 | 0.3468 |
| all | 4x4 | 0.3184 | 0.3433 | 0.3298 | 0.3468 |
| all | 8x8 | 0.3705 | 0.3470 | 0.3463 | 0.3468 |
| all | 16x16 | 0.3404 | 0.3716 | 0.3533 | 0.3468 |
| high-motion | 1x1 | 0.3946 | 0.3946 | 0.3226 | 0.3468 |
| high-motion | 2x2 | 0.3882 | 0.3684 | 0.3330 | 0.3468 |
| high-motion | 4x4 | 0.3404 | 0.3054 | 0.3505 | 0.3468 |
| high-motion | 8x8 | 0.3749 | 0.3437 | 0.3428 | 0.3468 |
| high-motion | 16x16 | 0.3572 | 0.4061 | 0.3498 | 0.3468 |

## Ridge Alpha Sweep

| Alpha | Mode | Pool | Acc |
|---:|---|---:|---:|
| 0p01 | all | 1x1 | 0.3746 |
| 0p01 | all | 2x2 | 0.3228 |
| 0p01 | all | 4x4 | 0.3647 |
| 0p01 | all | 8x8 | 0.3330 |
| 0p01 | all | 16x16 | 0.3433 |
| 0p01 | high-motion | 1x1 | 0.3540 |
| 0p01 | high-motion | 2x2 | 0.3095 |
| 0p01 | high-motion | 4x4 | 0.2875 |
| 0p01 | high-motion | 8x8 | 0.3267 |
| 0p01 | high-motion | 16x16 | 0.4104 |
| 0p1 | all | 1x1 | 0.4404 |
| 0p1 | all | 2x2 | 0.3404 |
| 0p1 | all | 4x4 | 0.3398 |
| 0p1 | all | 8x8 | 0.3365 |
| 0p1 | all | 16x16 | 0.3398 |
| 0p1 | high-motion | 1x1 | 0.4053 |
| 0p1 | high-motion | 2x2 | 0.3305 |
| 0p1 | high-motion | 4x4 | 0.3161 |
| 0p1 | high-motion | 8x8 | 0.3439 |
| 0p1 | high-motion | 16x16 | 0.4098 |
| 1 | all | 1x1 | 0.4088 |
| 1 | all | 2x2 | 0.3582 |
| 1 | all | 4x4 | 0.3433 |
| 1 | all | 8x8 | 0.3470 |
| 1 | all | 16x16 | 0.3716 |
| 1 | high-motion | 1x1 | 0.3946 |
| 1 | high-motion | 2x2 | 0.3684 |
| 1 | high-motion | 4x4 | 0.3054 |
| 1 | high-motion | 8x8 | 0.3437 |
| 1 | high-motion | 16x16 | 0.4061 |
| 10 | all | 1x1 | 0.3812 |
| 10 | all | 2x2 | 0.3958 |
| 10 | all | 4x4 | 0.3640 |
| 10 | all | 8x8 | 0.3537 |
| 10 | all | 16x16 | 0.3567 |
| 10 | high-motion | 1x1 | 0.3986 |
| 10 | high-motion | 2x2 | 0.3579 |
| 10 | high-motion | 4x4 | 0.3018 |
| 10 | high-motion | 8x8 | 0.3646 |
| 10 | high-motion | 16x16 | 0.3751 |
| 100 | all | 1x1 | 0.3639 |
| 100 | all | 2x2 | 0.4200 |
| 100 | all | 4x4 | 0.3226 |
| 100 | all | 8x8 | 0.3358 |
| 100 | all | 16x16 | 0.3425 |
| 100 | high-motion | 1x1 | 0.3742 |
| 100 | high-motion | 2x2 | 0.3472 |
| 100 | high-motion | 4x4 | 0.3158 |
| 100 | high-motion | 8x8 | 0.3540 |
| 100 | high-motion | 16x16 | 0.3328 |

## Sample Size Sweep

| Samples/type | Mode | Pool | Acc |
|---:|---|---:|---:|
| 8 | all | 1x1 | 0.3429 |
| 8 | all | 2x2 | 0.3286 |
| 8 | all | 4x4 | 0.2683 |
| 8 | all | 8x8 | 0.2905 |
| 8 | all | 16x16 | 0.3190 |
| 8 | high-motion | 1x1 | 0.2095 |
| 8 | high-motion | 2x2 | 0.2952 |
| 8 | high-motion | 4x4 | 0.2508 |
| 8 | high-motion | 8x8 | 0.2587 |
| 8 | high-motion | 16x16 | 0.2603 |
| 16 | all | 1x1 | 0.3491 |
| 16 | all | 2x2 | 0.2701 |
| 16 | all | 4x4 | 0.2812 |
| 16 | all | 8x8 | 0.3406 |
| 16 | all | 16x16 | 0.2662 |
| 16 | high-motion | 1x1 | 0.4158 |
| 16 | high-motion | 2x2 | 0.3808 |
| 16 | high-motion | 4x4 | 0.4073 |
| 16 | high-motion | 8x8 | 0.4688 |
| 16 | high-motion | 16x16 | 0.4432 |
| 24 | all | 1x1 | 0.4088 |
| 24 | all | 2x2 | 0.3582 |
| 24 | all | 4x4 | 0.3433 |
| 24 | all | 8x8 | 0.3470 |
| 24 | all | 16x16 | 0.3716 |
| 24 | high-motion | 1x1 | 0.3946 |
| 24 | high-motion | 2x2 | 0.3684 |
| 24 | high-motion | 4x4 | 0.3054 |
| 24 | high-motion | 8x8 | 0.3437 |
| 24 | high-motion | 16x16 | 0.4061 |

## PCA Same-Dim Per-Type

| Mode | Pool | Question type | Acc | Correct/total |
|---|---:|---|---:|---:|
| all | 1x1 | Action Order | 0.3750 | 27/72 |
| all | 1x1 | Motion Recognition | 0.4306 | 31/72 |
| all | 1x1 | Motion-related Objects | 0.3750 | 27/72 |
| all | 1x1 | Repetition Count | 0.4583 | 33/72 |
| all | 2x2 | Action Order | 0.3611 | 26/72 |
| all | 2x2 | Motion Recognition | 0.3194 | 23/72 |
| all | 2x2 | Motion-related Objects | 0.2917 | 21/72 |
| all | 2x2 | Repetition Count | 0.4583 | 33/72 |
| all | 4x4 | Action Order | 0.2917 | 21/72 |
| all | 4x4 | Motion Recognition | 0.2778 | 20/72 |
| all | 4x4 | Motion-related Objects | 0.3750 | 27/72 |
| all | 4x4 | Repetition Count | 0.4306 | 31/72 |
| all | 8x8 | Action Order | 0.3611 | 26/72 |
| all | 8x8 | Motion Recognition | 0.3194 | 23/72 |
| all | 8x8 | Motion-related Objects | 0.2778 | 20/72 |
| all | 8x8 | Repetition Count | 0.4306 | 31/72 |
| all | 16x16 | Action Order | 0.4028 | 29/72 |
| all | 16x16 | Motion Recognition | 0.2917 | 21/72 |
| all | 16x16 | Motion-related Objects | 0.3750 | 27/72 |
| all | 16x16 | Repetition Count | 0.4167 | 30/72 |
| high-motion | 1x1 | Action Order | 0.3750 | 27/72 |
| high-motion | 1x1 | Motion Recognition | 0.4167 | 30/72 |
| high-motion | 1x1 | Motion-related Objects | 0.3750 | 27/72 |
| high-motion | 1x1 | Repetition Count | 0.4167 | 30/72 |
| high-motion | 2x2 | Action Order | 0.3611 | 26/72 |
| high-motion | 2x2 | Motion Recognition | 0.4167 | 30/72 |
| high-motion | 2x2 | Motion-related Objects | 0.2500 | 18/72 |
| high-motion | 2x2 | Repetition Count | 0.4444 | 32/72 |
| high-motion | 4x4 | Action Order | 0.2917 | 21/72 |
| high-motion | 4x4 | Motion Recognition | 0.2222 | 16/72 |
| high-motion | 4x4 | Motion-related Objects | 0.2778 | 20/72 |
| high-motion | 4x4 | Repetition Count | 0.4306 | 31/72 |
| high-motion | 8x8 | Action Order | 0.2639 | 19/72 |
| high-motion | 8x8 | Motion Recognition | 0.3611 | 26/72 |
| high-motion | 8x8 | Motion-related Objects | 0.3472 | 25/72 |
| high-motion | 8x8 | Repetition Count | 0.4028 | 29/72 |
| high-motion | 16x16 | Action Order | 0.3889 | 28/72 |
| high-motion | 16x16 | Motion Recognition | 0.4444 | 32/72 |
| high-motion | 16x16 | Motion-related Objects | 0.3472 | 25/72 |
| high-motion | 16x16 | Repetition Count | 0.4444 | 32/72 |
