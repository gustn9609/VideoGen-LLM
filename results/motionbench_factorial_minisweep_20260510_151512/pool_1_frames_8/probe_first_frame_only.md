# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4521 | 0.1212 | 0.3924 | 0.5104 | 130/288 | 0.6493 |
| none | pixel_grid_sequence | hash | ridge | 0.4449 | 0.1254 | 0.3854 | 0.5035 | 128/288 | 0.6238 |
| none | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4488 | 0.1217 | 0.3889 | 0.5069 | 129/288 | 0.6478 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4516 | 0.1228 | 0.3924 | 0.5104 | 130/288 | 0.6246 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4521 | 0.1212 | 0.3924 | 0.5104 | 130/288 | 0.6493 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4449 | 0.1254 | 0.3854 | 0.5035 | 128/288 | 0.6238 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4488 | 0.1217 | 0.3889 | 0.5069 | 129/288 | 0.6478 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4516 | 0.1228 | 0.3924 | 0.5104 | 130/288 | 0.6246 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4722 | 34/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| none | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.5278 | 38/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4722 | 34/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5278 | 38/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
