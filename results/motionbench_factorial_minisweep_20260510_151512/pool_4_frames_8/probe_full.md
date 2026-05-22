# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3679 | 0.1168 | 0.3090 | 0.4236 | 106/288 | 0.6723 |
| none | pixel_grid_sequence | hash | ridge | 0.3465 | 0.1009 | 0.2951 | 0.3993 | 100/288 | 0.5780 |
| none | flow_grid_sequence | hash | ridge | 0.4307 | 0.1254 | 0.3750 | 0.4861 | 124/288 | 0.5793 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3649 | 0.1078 | 0.3056 | 0.4201 | 105/288 | 0.6462 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3639 | 0.0816 | 0.3125 | 0.4201 | 105/288 | 0.5949 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4339 | 0.1041 | 0.3785 | 0.4896 | 125/288 | 0.5709 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3679 | 0.1136 | 0.3125 | 0.4236 | 106/288 | 0.6616 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.3502 | 0.1018 | 0.2951 | 0.4062 | 101/288 | 0.5789 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4374 | 0.1209 | 0.3819 | 0.4965 | 126/288 | 0.5944 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3788 | 0.1099 | 0.3229 | 0.4375 | 109/288 | 0.6387 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3674 | 0.0850 | 0.3159 | 0.4236 | 106/288 | 0.5913 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4340 | 0.1216 | 0.3785 | 0.4896 | 125/288 | 0.5366 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3889 | 28/72 |
| none | pixel_grid_sequence | Action Order | 0.1944 | 14/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.3611 | 26/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| none | flow_grid_sequence | Action Order | 0.3750 | 27/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| none | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion | flow_grid_sequence | Action Order | 0.4167 | 30/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3611 | 26/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4028 | 29/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2083 | 15/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.3611 | 26/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3889 | 28/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.3611 | 26/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3889 | 28/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4028 | 29/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3750 | 27/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
