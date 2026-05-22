# MotionBench Wan-MoREPA Residual Reranker

| mode | feature | teacher | acc | text-only | gain | correct/total |
|---|---|---|---:|---:|---:|---:|
| high_motion+camera_comp | wan_vae_grid_1x1 | motion_teacher_summary | 0.3426 | 0.3216 | 0.0211 | 33/96 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.2917 | 7/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.3333 | 8/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.3333 | 8/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.4167 | 10/24 |
