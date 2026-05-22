# MotionBench Wan-MoREPA Residual Reranker

| mode | feature | teacher | acc | text-only | gain | correct/total |
|---|---|---|---:|---:|---:|---:|
| high_motion+camera_comp | wan_vae_grid_1x1 | motion_teacher_summary | 0.4800 | 0.4589 | 0.0211 | 46/96 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.3750 | 9/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.5000 | 12/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5000 | 12/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5417 | 13/24 |
