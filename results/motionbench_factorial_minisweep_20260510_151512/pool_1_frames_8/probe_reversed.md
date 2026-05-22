# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4246 | 0.1076 | 0.3681 | 0.4793 | 122/288 | 0.6562 |
| none | pixel_grid_sequence | hash | ridge | 0.4414 | 0.1041 | 0.3854 | 0.4965 | 127/288 | 0.6379 |
| none | flow_grid_sequence | hash | ridge | 0.4523 | 0.1318 | 0.3958 | 0.5104 | 130/288 | 0.6802 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4347 | 0.1051 | 0.3750 | 0.4931 | 125/288 | 0.6579 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4519 | 0.1110 | 0.3924 | 0.5070 | 130/288 | 0.6356 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4549 | 0.1154 | 0.3958 | 0.5139 | 131/288 | 0.6072 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4211 | 0.1068 | 0.3611 | 0.4792 | 121/288 | 0.6533 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4414 | 0.1041 | 0.3854 | 0.4965 | 127/288 | 0.6386 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4519 | 0.1130 | 0.3958 | 0.5104 | 130/288 | 0.6518 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4382 | 0.1089 | 0.3785 | 0.4965 | 126/288 | 0.6553 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4588 | 0.1066 | 0.4027 | 0.5139 | 132/288 | 0.6341 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4621 | 0.1319 | 0.4028 | 0.5208 | 133/288 | 0.5930 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.4861 | 35/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| none | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.4722 | 34/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.3333 | 24/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5556 | 40/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
