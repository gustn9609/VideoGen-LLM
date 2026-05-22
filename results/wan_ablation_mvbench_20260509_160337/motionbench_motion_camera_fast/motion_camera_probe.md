# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.3788 | 0.1395 | 0.3229 | 0.4375 | 109/288 | 2.4469 |
| none | pixel_grid_sequence | hash | logistic | 0.4130 | 0.1139 | 0.3542 | 0.4688 | 119/288 | 2.3753 |
| none | flow_grid_sequence | hash | logistic | 0.3718 | 0.0937 | 0.3160 | 0.4271 | 107/288 | 2.3894 |
| high_motion | wan_vae_grid_4x4 | hash | logistic | 0.4374 | 0.1297 | 0.3819 | 0.4965 | 126/288 | 2.2433 |
| high_motion | pixel_grid_sequence | hash | logistic | 0.4270 | 0.1073 | 0.3715 | 0.4861 | 123/288 | 2.3965 |
| high_motion | flow_grid_sequence | hash | logistic | 0.3923 | 0.0774 | 0.3368 | 0.4479 | 113/288 | 2.3846 |
| camera_comp | wan_vae_grid_4x4 | hash | logistic | 0.4133 | 0.0790 | 0.3576 | 0.4688 | 119/288 | 2.4900 |
| camera_comp | pixel_grid_sequence | hash | logistic | 0.3995 | 0.1278 | 0.3438 | 0.4583 | 115/288 | 2.1496 |
| camera_comp | flow_grid_sequence | hash | logistic | 0.3544 | 0.0960 | 0.2986 | 0.4062 | 102/288 | 2.3858 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | Action Order | 0.3333 | 24/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3889 | 28/72 |
| none | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| none | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.3333 | 24/72 |
| none | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.5833 | 42/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3056 | 22/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.2639 | 19/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.5278 | 38/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4028 | 29/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2361 | 17/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.2917 | 21/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.3889 | 28/72 |
