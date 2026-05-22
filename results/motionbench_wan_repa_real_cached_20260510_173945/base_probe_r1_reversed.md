# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| high_motion+camera_comp | text_only | hash | ridge | 0.4589 | 0.1443 | 0.3643 | 0.5521 | 44/96 | 0.6627 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4700 | 0.1141 | 0.3646 | 0.5628 | 45/96 | 0.7311 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4805 | 0.1341 | 0.3750 | 0.5729 | 46/96 | 0.7189 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.5005 | 0.1198 | 0.4062 | 0.5938 | 48/96 | 0.6311 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| high_motion+camera_comp | text_only | Action Order | 0.3333 | 8/24 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.4583 | 11/24 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 13/24 |
| high_motion+camera_comp | text_only | Repetition Count | 0.5000 | 12/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.3750 | 9/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4583 | 11/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5000 | 12/24 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5417 | 13/24 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.4167 | 10/24 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4167 | 10/24 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5417 | 13/24 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5417 | 13/24 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.4167 | 10/24 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 12/24 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5833 | 14/24 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.5000 | 12/24 |
