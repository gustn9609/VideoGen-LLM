# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3995 | 0.0957 | 0.3437 | 0.4549 | 115/288 | 0.6673 |
| none | pixel_grid_sequence | hash | ridge | 0.4346 | 0.0913 | 0.3785 | 0.4931 | 125/288 | 0.6435 |
| none | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3998 | 0.1024 | 0.3403 | 0.4549 | 115/288 | 0.6453 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4412 | 0.0904 | 0.3854 | 0.5000 | 127/288 | 0.6418 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3995 | 0.0957 | 0.3437 | 0.4549 | 115/288 | 0.6673 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4346 | 0.0913 | 0.3785 | 0.4931 | 125/288 | 0.6435 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3998 | 0.1024 | 0.3403 | 0.4549 | 115/288 | 0.6453 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4412 | 0.0904 | 0.3854 | 0.5000 | 127/288 | 0.6418 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| none | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4583 | 33/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
