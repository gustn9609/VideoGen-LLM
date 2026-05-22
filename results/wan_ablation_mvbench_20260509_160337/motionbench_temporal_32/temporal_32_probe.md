# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.3854 | 0.0981 | 0.3299 | 0.4411 | 111/288 | 2.6604 |
| none | pixel_grid_sequence | hash | logistic | 0.4377 | 0.1075 | 0.3785 | 0.4965 | 126/288 | 2.1956 |
| none | flow_grid_sequence | hash | logistic | 0.3747 | 0.0770 | 0.3194 | 0.4306 | 108/288 | 2.5169 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4722 | 34/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4306 | 31/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| none | pixel_grid_sequence | Action Order | 0.3889 | 28/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| none | flow_grid_sequence | Action Order | 0.1944 | 14/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.3333 | 24/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
