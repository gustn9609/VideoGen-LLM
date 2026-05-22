# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3718 | 0.0667 | 0.3160 | 0.4271 | 107/288 | 0.6750 |
| none | pixel_grid_sequence | hash | ridge | 0.4202 | 0.0924 | 0.3646 | 0.4757 | 121/288 | 0.6139 |
| none | flow_grid_sequence | hash | ridge | 0.4693 | 0.1407 | 0.4097 | 0.5278 | 135/288 | 0.6673 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3891 | 0.0694 | 0.3333 | 0.4444 | 112/288 | 0.6753 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4202 | 0.1037 | 0.3646 | 0.4757 | 121/288 | 0.6094 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4482 | 0.1333 | 0.3889 | 0.5035 | 129/288 | 0.6394 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3647 | 0.0667 | 0.3090 | 0.4201 | 105/288 | 0.6822 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4202 | 0.0924 | 0.3646 | 0.4757 | 121/288 | 0.6144 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4621 | 0.1123 | 0.4062 | 0.5174 | 133/288 | 0.6764 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3788 | 0.0602 | 0.3228 | 0.4340 | 109/288 | 0.6765 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4202 | 0.1037 | 0.3646 | 0.4757 | 121/288 | 0.6100 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4344 | 0.1092 | 0.3750 | 0.4896 | 125/288 | 0.6846 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2083 | 15/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Action Order | 0.3611 | 26/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| none | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2500 | 18/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4861 | 35/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.1944 | 14/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4861 | 35/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3611 | 26/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3472 | 25/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4028 | 29/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
