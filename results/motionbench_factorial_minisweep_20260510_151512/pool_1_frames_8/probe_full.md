# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4418 | 0.1286 | 0.3819 | 0.4965 | 127/288 | 0.6593 |
| none | pixel_grid_sequence | hash | ridge | 0.4447 | 0.1054 | 0.3854 | 0.5000 | 128/288 | 0.6392 |
| none | flow_grid_sequence | hash | ridge | 0.4309 | 0.1270 | 0.3750 | 0.4896 | 124/288 | 0.6414 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4381 | 0.1094 | 0.3785 | 0.4965 | 126/288 | 0.6629 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4379 | 0.1172 | 0.3785 | 0.4931 | 126/288 | 0.6344 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4549 | 0.1071 | 0.3992 | 0.5139 | 131/288 | 0.5841 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4347 | 0.1337 | 0.3750 | 0.4931 | 125/288 | 0.6590 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4447 | 0.1171 | 0.3854 | 0.5000 | 128/288 | 0.6451 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4412 | 0.1182 | 0.3819 | 0.5000 | 127/288 | 0.6423 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4519 | 0.1125 | 0.3924 | 0.5104 | 130/288 | 0.6546 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4379 | 0.1172 | 0.3785 | 0.4931 | 126/288 | 0.6372 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4446 | 0.1065 | 0.3888 | 0.5000 | 128/288 | 0.5949 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| none | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4861 | 35/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.4722 | 34/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.2917 | 21/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.4861 | 35/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.4722 | 34/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3611 | 26/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
