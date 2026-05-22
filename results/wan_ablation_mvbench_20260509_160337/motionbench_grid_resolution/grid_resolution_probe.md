# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_1x1 | hash | logistic | 0.4621 | 0.1372 | 0.4028 | 0.5208 | 133/288 | 2.6354 |
| none | wan_vae_grid_2x2 | hash | logistic | 0.4509 | 0.1148 | 0.3958 | 0.5069 | 130/288 | 2.4459 |
| none | wan_vae_grid_4x4 | hash | logistic | 0.4096 | 0.1050 | 0.3542 | 0.4653 | 118/288 | 2.4251 |
| none | wan_vae_grid_8x8 | hash | logistic | 0.4133 | 0.0910 | 0.3576 | 0.4688 | 119/288 | 2.3244 |
| none | wan_vae_grid_16x16 | hash | logistic | 0.3821 | 0.1249 | 0.3264 | 0.4375 | 110/288 | 2.2431 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_1x1 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.5278 | 38/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.5278 | 38/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2500 | 18/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.5000 | 36/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4722 | 34/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.3333 | 24/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.5000 | 36/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.4583 | 33/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.4028 | 29/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.3889 | 28/72 |
