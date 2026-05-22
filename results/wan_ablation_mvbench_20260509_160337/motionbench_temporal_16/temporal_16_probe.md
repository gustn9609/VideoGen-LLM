# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.4133 | 0.1151 | 0.3576 | 0.4688 | 119/288 | 2.6697 |
| none | pixel_grid_sequence | hash | logistic | 0.3860 | 0.0930 | 0.3264 | 0.4410 | 111/288 | 2.4048 |
| none | flow_grid_sequence | hash | logistic | 0.3681 | 0.0995 | 0.3125 | 0.4236 | 106/288 | 2.3527 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4861 | 35/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4028 | 29/72 |
| none | pixel_grid_sequence | Action Order | 0.3056 | 22/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| none | pixel_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| none | flow_grid_sequence | Action Order | 0.2361 | 17/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.3333 | 24/72 |
| none | flow_grid_sequence | Repetition Count | 0.4028 | 29/72 |
