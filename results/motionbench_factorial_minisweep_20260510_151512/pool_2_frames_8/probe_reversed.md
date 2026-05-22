# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3856 | 0.1343 | 0.3264 | 0.4410 | 111/288 | 0.6620 |
| none | pixel_grid_sequence | hash | ridge | 0.4132 | 0.1133 | 0.3576 | 0.4722 | 119/288 | 0.6611 |
| none | flow_grid_sequence | hash | ridge | 0.4588 | 0.1274 | 0.3993 | 0.5174 | 132/288 | 0.6672 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3891 | 0.1199 | 0.3299 | 0.4444 | 112/288 | 0.6564 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4165 | 0.1256 | 0.3611 | 0.4722 | 120/288 | 0.6492 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4793 | 0.1278 | 0.4201 | 0.5382 | 138/288 | 0.6839 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3891 | 0.1274 | 0.3299 | 0.4444 | 112/288 | 0.6624 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4130 | 0.1161 | 0.3576 | 0.4722 | 119/288 | 0.6639 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4451 | 0.1330 | 0.3854 | 0.5000 | 128/288 | 0.6491 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3926 | 0.1201 | 0.3368 | 0.4479 | 113/288 | 0.6568 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4165 | 0.1256 | 0.3611 | 0.4722 | 120/288 | 0.6522 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4586 | 0.1309 | 0.4028 | 0.5174 | 132/288 | 0.6661 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.5000 | 36/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| none | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5694 | 41/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.4028 | 29/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.5000 | 36/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4167 | 30/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.5000 | 36/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.5000 | 36/72 |
