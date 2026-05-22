# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | hash | logistic | 0.3958 | 0.0208 | 0.2966 | 0.4951 | 38/96 | 2.1877 |
| none | pixel_grid_sequence | hash | logistic | 0.3958 | 0.0417 | 0.2862 | 0.4951 | 38/96 | 1.6502 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_sequence | Action Order | 0.3750 | 9/24 |
| none | wan_vae_grid_sequence | Motion Recognition | 0.4167 | 10/24 |
| none | wan_vae_grid_sequence | Motion-related Objects | 0.4583 | 11/24 |
| none | wan_vae_grid_sequence | Repetition Count | 0.3333 | 8/24 |
| none | pixel_grid_sequence | Action Order | 0.3750 | 9/24 |
| none | pixel_grid_sequence | Motion Recognition | 0.3750 | 9/24 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4583 | 11/24 |
| none | pixel_grid_sequence | Repetition Count | 0.3750 | 9/24 |
