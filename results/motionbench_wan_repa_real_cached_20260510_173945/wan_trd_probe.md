# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| high_motion+camera_comp | wan_trd_wan_vae_grid_1x1_pooled | hash | ridge | 0.3868 | 0.1115 | 0.2917 | 0.4792 | 37/96 | 0.6435 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| high_motion+camera_comp | wan_trd_wan_vae_grid_1x1_pooled | Action Order | 0.3750 | 9/24 |
| high_motion+camera_comp | wan_trd_wan_vae_grid_1x1_pooled | Motion Recognition | 0.2917 | 7/24 |
| high_motion+camera_comp | wan_trd_wan_vae_grid_1x1_pooled | Motion-related Objects | 0.4167 | 10/24 |
| high_motion+camera_comp | wan_trd_wan_vae_grid_1x1_pooled | Repetition Count | 0.4583 | 11/24 |
