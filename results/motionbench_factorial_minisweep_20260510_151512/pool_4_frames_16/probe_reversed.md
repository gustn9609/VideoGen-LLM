# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3995 | 0.0914 | 0.3438 | 0.4549 | 115/288 | 0.6799 |
| none | pixel_grid_sequence | hash | ridge | 0.3360 | 0.1058 | 0.2812 | 0.3924 | 97/288 | 0.6289 |
| none | flow_grid_sequence | hash | ridge | 0.3996 | 0.1082 | 0.3403 | 0.4549 | 115/288 | 0.5594 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3372 | 0.1157 | 0.2812 | 0.3890 | 97/288 | 0.6281 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3544 | 0.1182 | 0.2986 | 0.4097 | 102/288 | 0.5745 |
| high_motion | flow_grid_sequence | hash | ridge | 0.3895 | 0.0968 | 0.3333 | 0.4444 | 112/288 | 0.6236 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3705 | 0.1119 | 0.3125 | 0.4271 | 107/288 | 0.5902 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.3511 | 0.0956 | 0.2951 | 0.4062 | 101/288 | 0.7061 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.3818 | 0.0897 | 0.3264 | 0.4375 | 110/288 | 0.5368 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3712 | 0.0635 | 0.3160 | 0.4271 | 107/288 | 0.6494 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3575 | 0.1000 | 0.3021 | 0.4132 | 103/288 | 0.5933 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4168 | 0.0879 | 0.3611 | 0.4688 | 120/288 | 0.6021 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4306 | 31/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| none | flow_grid_sequence | Action Order | 0.2500 | 18/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3889 | 28/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.2639 | 19/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.3056 | 22/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3611 | 26/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.3611 | 26/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.3750 | 27/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.1944 | 14/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.3889 | 28/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.2639 | 19/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3194 | 23/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.5000 | 36/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4028 | 29/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.3194 | 23/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
