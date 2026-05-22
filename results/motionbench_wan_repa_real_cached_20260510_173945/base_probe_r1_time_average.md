# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| high_motion+camera_comp | text_only | hash | ridge | 0.4589 | 0.1443 | 0.3643 | 0.5521 | 44/96 | 0.6627 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4695 | 0.1127 | 0.3750 | 0.5625 | 45/96 | 0.7006 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4484 | 0.1485 | 0.3542 | 0.5417 | 43/96 | 0.6824 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4800 | 0.1329 | 0.3750 | 0.5833 | 46/96 | 0.6983 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| high_motion+camera_comp | text_only | Action Order | 0.3333 | 8/24 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.4583 | 11/24 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 13/24 |
| high_motion+camera_comp | text_only | Repetition Count | 0.5000 | 12/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.2917 | 7/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4583 | 11/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5833 | 14/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5417 | 13/24 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3750 | 9/24 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4167 | 10/24 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5417 | 13/24 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 11/24 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.4583 | 11/24 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4167 | 10/24 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5000 | 12/24 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.5417 | 13/24 |
