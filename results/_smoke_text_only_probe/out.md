# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4271 | 0.0312 | 0.3021 | 0.5208 | 41/96 | 0.8639 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3958 | 0.0417 | 0.3229 | 0.5000 | 38/96 | 0.6248 |
| none | pixel_grid_sequence | hash | ridge | 0.3021 | 0.0312 | 0.2133 | 0.3805 | 29/96 | 0.7166 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3333 | 8/24 |
| none | text_only | Motion Recognition | 0.4583 | 11/24 |
| none | text_only | Motion-related Objects | 0.4167 | 10/24 |
| none | text_only | Repetition Count | 0.5000 | 12/24 |
| none | wan_vae_grid_4x4 | Action Order | 0.4167 | 10/24 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3333 | 8/24 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 10/24 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 10/24 |
| none | pixel_grid_sequence | Action Order | 0.2083 | 5/24 |
| none | pixel_grid_sequence | Motion Recognition | 0.3333 | 8/24 |
| none | pixel_grid_sequence | Motion-related Objects | 0.3333 | 8/24 |
| none | pixel_grid_sequence | Repetition Count | 0.3333 | 8/24 |
