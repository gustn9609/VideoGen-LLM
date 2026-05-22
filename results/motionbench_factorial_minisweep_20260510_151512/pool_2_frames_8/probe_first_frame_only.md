# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.4205 | 0.0919 | 0.3611 | 0.4757 | 121/288 | 0.6624 |
| none | pixel_grid_sequence | hash | ridge | 0.4453 | 0.1220 | 0.3854 | 0.5035 | 128/288 | 0.6383 |
| none | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.4172 | 0.0937 | 0.3576 | 0.4722 | 120/288 | 0.6552 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4418 | 0.1157 | 0.3819 | 0.5000 | 127/288 | 0.6326 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.4205 | 0.0919 | 0.3611 | 0.4757 | 121/288 | 0.6624 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4453 | 0.1220 | 0.3854 | 0.5035 | 128/288 | 0.6383 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.4172 | 0.0937 | 0.3576 | 0.4722 | 120/288 | 0.6552 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4418 | 0.1157 | 0.3819 | 0.5000 | 127/288 | 0.6326 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| none | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2778 | 20/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.2917 | 21/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3750 | 27/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.5000 | 36/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2778 | 20/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3611 | 26/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
