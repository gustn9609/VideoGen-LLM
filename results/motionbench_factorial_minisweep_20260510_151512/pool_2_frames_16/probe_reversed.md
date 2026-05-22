# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3784 | 0.0903 | 0.3229 | 0.4340 | 109/288 | 0.6611 |
| none | pixel_grid_sequence | hash | ridge | 0.3993 | 0.1023 | 0.3438 | 0.4549 | 115/288 | 0.6033 |
| none | flow_grid_sequence | hash | ridge | 0.4409 | 0.1155 | 0.3819 | 0.4965 | 127/288 | 0.6140 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3853 | 0.0773 | 0.3299 | 0.4410 | 111/288 | 0.6415 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3995 | 0.0936 | 0.3438 | 0.4549 | 115/288 | 0.6041 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4553 | 0.1259 | 0.3993 | 0.5139 | 131/288 | 0.6606 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3819 | 0.0908 | 0.3264 | 0.4375 | 110/288 | 0.6596 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.3923 | 0.0930 | 0.3368 | 0.4479 | 113/288 | 0.6077 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4411 | 0.0991 | 0.3819 | 0.4965 | 127/288 | 0.6222 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3925 | 0.0765 | 0.3368 | 0.4479 | 113/288 | 0.6457 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3821 | 0.0934 | 0.3264 | 0.4375 | 110/288 | 0.5999 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4549 | 0.1304 | 0.3992 | 0.5139 | 131/288 | 0.6529 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.1806 | 13/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.4028 | 29/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.4444 | 32/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4861 | 35/72 |
| none | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| none | flow_grid_sequence | Action Order | 0.3611 | 26/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.4722 | 34/72 |
| none | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2222 | 16/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.3889 | 28/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3611 | 26/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.1944 | 14/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.4167 | 30/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4444 | 32/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3056 | 22/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.3611 | 26/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.3611 | 26/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.4167 | 30/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4444 | 32/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4722 | 34/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.3472 | 25/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
