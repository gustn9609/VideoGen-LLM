# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3993 | 0.0949 | 0.3438 | 0.4549 | 115/288 | 0.6413 |
| none | pixel_grid_sequence | hash | ridge | 0.4096 | 0.0941 | 0.3542 | 0.4618 | 118/288 | 0.6137 |
| none | flow_grid_sequence | hash | ridge | 0.4516 | 0.1213 | 0.3924 | 0.5104 | 130/288 | 0.6044 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.4028 | 0.0906 | 0.3438 | 0.4583 | 116/288 | 0.6457 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4237 | 0.1058 | 0.3681 | 0.4792 | 122/288 | 0.6082 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4623 | 0.1268 | 0.4028 | 0.5208 | 133/288 | 0.6489 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3851 | 0.1046 | 0.3333 | 0.4411 | 111/288 | 0.6390 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4167 | 0.1040 | 0.3611 | 0.4722 | 120/288 | 0.6146 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4446 | 0.1085 | 0.3889 | 0.5000 | 128/288 | 0.5725 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3923 | 0.0930 | 0.3368 | 0.4479 | 113/288 | 0.6493 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4237 | 0.1058 | 0.3681 | 0.4792 | 122/288 | 0.6059 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4551 | 0.1080 | 0.3958 | 0.5139 | 131/288 | 0.6377 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2500 | 18/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.4444 | 32/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.4306 | 31/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| none | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| none | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2500 | 18/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3750 | 27/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.2083 | 15/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4306 | 31/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4583 | 33/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3056 | 22/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4306 | 31/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
