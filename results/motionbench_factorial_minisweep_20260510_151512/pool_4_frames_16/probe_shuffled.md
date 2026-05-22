# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3675 | 0.1078 | 0.3125 | 0.4236 | 106/288 | 0.5606 |
| none | pixel_grid_sequence | hash | ridge | 0.4025 | 0.0979 | 0.3438 | 0.4583 | 116/288 | 0.5771 |
| none | flow_grid_sequence | hash | ridge | 0.4063 | 0.0878 | 0.3507 | 0.4653 | 117/288 | 0.5833 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3854 | 0.1195 | 0.3298 | 0.4444 | 111/288 | 0.5654 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3851 | 0.1150 | 0.3264 | 0.4410 | 111/288 | 0.5520 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4270 | 0.1004 | 0.3681 | 0.4826 | 123/288 | 0.5785 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3995 | 0.1032 | 0.3438 | 0.4583 | 115/288 | 0.6093 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.3923 | 0.1037 | 0.3368 | 0.4479 | 113/288 | 0.6239 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4165 | 0.1111 | 0.3610 | 0.4723 | 120/288 | 0.6090 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3747 | 0.0956 | 0.3160 | 0.4306 | 108/288 | 0.5867 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3189 | 0.1239 | 0.2674 | 0.3715 | 92/288 | 0.6320 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4023 | 0.0735 | 0.3472 | 0.4583 | 116/288 | 0.5173 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| none | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| none | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.3889 | 28/72 |
| none | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.4722 | 34/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Action Order | 0.4028 | 29/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.3889 | 28/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.3056 | 22/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4861 | 35/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.3889 | 28/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3611 | 26/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.4028 | 29/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.3472 | 25/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3333 | 24/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4028 | 29/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.3333 | 24/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.3472 | 25/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4028 | 29/72 |
