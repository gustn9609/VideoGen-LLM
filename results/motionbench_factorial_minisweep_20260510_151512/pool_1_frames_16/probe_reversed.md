# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4344 | 0.1068 | 0.3784 | 0.4896 | 125/288 | 0.6512 |
| none | pixel_grid_sequence | hash | ridge | 0.4418 | 0.0927 | 0.3854 | 0.4965 | 127/288 | 0.6437 |
| none | flow_grid_sequence | hash | ridge | 0.4554 | 0.1192 | 0.3958 | 0.5139 | 131/288 | 0.6216 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4414 | 0.0926 | 0.3819 | 0.4966 | 127/288 | 0.6594 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4311 | 0.0926 | 0.3750 | 0.4896 | 124/288 | 0.6475 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4658 | 0.1104 | 0.4097 | 0.5243 | 134/288 | 0.6482 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4309 | 0.1063 | 0.3749 | 0.4861 | 124/288 | 0.6475 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4453 | 0.0948 | 0.3889 | 0.5035 | 128/288 | 0.6419 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1331 | 0.3958 | 0.5139 | 131/288 | 0.6446 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4414 | 0.0926 | 0.3819 | 0.4965 | 127/288 | 0.6584 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4381 | 0.0955 | 0.3819 | 0.4931 | 126/288 | 0.6511 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4519 | 0.1251 | 0.3958 | 0.5104 | 130/288 | 0.6617 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5556 | 40/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4861 | 35/72 |
| none | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| none | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Action Order | 0.4444 | 32/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.4861 | 35/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3750 | 27/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
