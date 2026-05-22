# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| high_motion+camera_comp | wan_moft_wan_vae_grid_1x1 | hash | ridge | 0.4379 | 0.1282 | 0.3438 | 0.5312 | 42/96 | 0.6584 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| high_motion+camera_comp | wan_moft_wan_vae_grid_1x1 | Action Order | 0.2917 | 7/24 |
| high_motion+camera_comp | wan_moft_wan_vae_grid_1x1 | Motion Recognition | 0.4583 | 11/24 |
| high_motion+camera_comp | wan_moft_wan_vae_grid_1x1 | Motion-related Objects | 0.5000 | 12/24 |
| high_motion+camera_comp | wan_moft_wan_vae_grid_1x1 | Repetition Count | 0.5000 | 12/24 |
