# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.3919 | 0.1171 | 0.3333 | 0.4479 | 113/288 | 2.2369 |
| none | pixel_grid_sequence | hash | logistic | 0.4237 | 0.0872 | 0.3646 | 0.4826 | 122/288 | 2.3638 |
| none | flow_grid_sequence | hash | logistic | 0.3749 | 0.1057 | 0.3194 | 0.4306 | 108/288 | 2.6657 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.5139 | 37/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Action Order | 0.3056 | 22/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| none | flow_grid_sequence | Action Order | 0.3889 | 28/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.2361 | 17/72 |
| none | flow_grid_sequence | Repetition Count | 0.3889 | 28/72 |
