# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4277 | 0.1054 | 0.3715 | 0.4861 | 123/288 | 0.6620 |
| none | pixel_grid_sequence | hash | ridge | 0.4453 | 0.0984 | 0.3889 | 0.5035 | 128/288 | 0.6382 |
| none | flow_grid_sequence | hash | ridge | 0.4518 | 0.1068 | 0.3924 | 0.5070 | 130/288 | 0.5991 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4382 | 0.1033 | 0.3785 | 0.4965 | 126/288 | 0.6621 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4489 | 0.1027 | 0.3924 | 0.5036 | 129/288 | 0.6431 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4481 | 0.1447 | 0.3924 | 0.5069 | 129/288 | 0.6371 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4277 | 0.1054 | 0.3715 | 0.4861 | 123/288 | 0.6609 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4521 | 0.0998 | 0.3958 | 0.5104 | 130/288 | 0.6408 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4446 | 0.1065 | 0.3854 | 0.5001 | 128/288 | 0.6387 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4418 | 0.1105 | 0.3819 | 0.5000 | 127/288 | 0.6626 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4591 | 0.0935 | 0.4028 | 0.5174 | 132/288 | 0.6403 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4451 | 0.1166 | 0.3854 | 0.5035 | 128/288 | 0.6489 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4028 | 29/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| none | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | flow_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4028 | 29/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.3333 | 24/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
