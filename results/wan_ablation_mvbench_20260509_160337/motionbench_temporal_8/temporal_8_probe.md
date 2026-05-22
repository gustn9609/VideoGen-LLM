# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.4104 | 0.1257 | 0.3507 | 0.4688 | 118/288 | 2.6507 |
| none | pixel_grid_sequence | hash | logistic | 0.3889 | 0.1315 | 0.3332 | 0.4479 | 112/288 | 2.2699 |
| none | flow_grid_sequence | hash | logistic | 0.3751 | 0.0840 | 0.3194 | 0.4306 | 108/288 | 2.4758 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | Action Order | 0.3333 | 24/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4306 | 31/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.5000 | 36/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Repetition Count | 0.3889 | 28/72 |
| none | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.2917 | 21/72 |
| none | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
