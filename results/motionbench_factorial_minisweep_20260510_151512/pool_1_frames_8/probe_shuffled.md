# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4416 | 0.1096 | 0.3853 | 0.4966 | 127/288 | 0.6630 |
| none | pixel_grid_sequence | hash | ridge | 0.4623 | 0.1113 | 0.4028 | 0.5208 | 133/288 | 0.6419 |
| none | flow_grid_sequence | hash | ridge | 0.4453 | 0.1072 | 0.3854 | 0.5035 | 128/288 | 0.6435 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4312 | 0.1060 | 0.3715 | 0.4896 | 124/288 | 0.6728 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4623 | 0.1113 | 0.4028 | 0.5208 | 133/288 | 0.6401 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4307 | 0.1146 | 0.3715 | 0.4896 | 124/288 | 0.6064 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4312 | 0.1187 | 0.3749 | 0.4896 | 124/288 | 0.6645 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4589 | 0.1071 | 0.3958 | 0.5139 | 132/288 | 0.6395 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4314 | 0.1243 | 0.3715 | 0.4896 | 124/288 | 0.6403 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4312 | 0.1094 | 0.3715 | 0.4862 | 124/288 | 0.6739 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4623 | 0.1096 | 0.4028 | 0.5208 | 133/288 | 0.6411 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4379 | 0.1191 | 0.3785 | 0.4931 | 126/288 | 0.5963 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.2639 | 19/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| none | flow_grid_sequence | Action Order | 0.2639 | 19/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| none | flow_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.4861 | 35/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3611 | 26/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| high_motion | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.2500 | 18/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2778 | 20/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.2639 | 19/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.4861 | 35/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5278 | 38/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3611 | 26/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
