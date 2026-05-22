# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_4x4 | hash | logistic | 0.3989 | 0.0882 | 0.3403 | 0.4549 | 115/288 | 2.4429 |
| none | pixel_grid_sequence | hash | logistic | 0.4372 | 0.1058 | 0.3785 | 0.4931 | 126/288 | 2.4028 |
| none | flow_grid_sequence | hash | logistic | 0.4375 | 0.0983 | 0.3785 | 0.4965 | 126/288 | 2.5781 |
| center_crop | wan_vae_grid_4x4 | hash | logistic | 0.3956 | 0.1402 | 0.3403 | 0.4549 | 114/288 | 2.5835 |
| center_crop | pixel_grid_sequence | hash | logistic | 0.3889 | 0.0933 | 0.3333 | 0.4479 | 112/288 | 2.3025 |
| center_crop | flow_grid_sequence | hash | logistic | 0.3684 | 0.1013 | 0.3125 | 0.4236 | 106/288 | 2.2977 |
| object_crop | wan_vae_grid_4x4 | hash | logistic | 0.3954 | 0.0862 | 0.3403 | 0.4514 | 114/288 | 2.5630 |
| object_crop | pixel_grid_sequence | hash | logistic | 0.4274 | 0.1184 | 0.3681 | 0.4826 | 123/288 | 2.3533 |
| object_crop | flow_grid_sequence | hash | logistic | 0.3644 | 0.1051 | 0.3090 | 0.4201 | 105/288 | 2.5232 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_4x4 | Action Order | 0.2639 | 19/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.5278 | 38/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Action Order | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.5417 | 39/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Repetition Count | 0.3889 | 28/72 |
| none | flow_grid_sequence | Action Order | 0.4444 | 32/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.3611 | 26/72 |
| none | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| center_crop | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| center_crop | wan_vae_grid_4x4 | Motion Recognition | 0.3750 | 27/72 |
| center_crop | wan_vae_grid_4x4 | Motion-related Objects | 0.4722 | 34/72 |
| center_crop | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| center_crop | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| center_crop | pixel_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| center_crop | pixel_grid_sequence | Motion-related Objects | 0.3472 | 25/72 |
| center_crop | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| center_crop | flow_grid_sequence | Action Order | 0.3056 | 22/72 |
| center_crop | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| center_crop | flow_grid_sequence | Motion-related Objects | 0.2778 | 20/72 |
| center_crop | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| object_crop | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| object_crop | wan_vae_grid_4x4 | Motion Recognition | 0.4861 | 35/72 |
| object_crop | wan_vae_grid_4x4 | Motion-related Objects | 0.3611 | 26/72 |
| object_crop | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| object_crop | pixel_grid_sequence | Action Order | 0.3750 | 27/72 |
| object_crop | pixel_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| object_crop | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| object_crop | pixel_grid_sequence | Repetition Count | 0.3889 | 28/72 |
| object_crop | flow_grid_sequence | Action Order | 0.3056 | 22/72 |
| object_crop | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| object_crop | flow_grid_sequence | Motion-related Objects | 0.3056 | 22/72 |
| object_crop | flow_grid_sequence | Repetition Count | 0.3889 | 28/72 |
