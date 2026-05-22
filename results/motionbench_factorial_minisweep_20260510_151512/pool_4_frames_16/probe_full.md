# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3609 | 0.1204 | 0.3056 | 0.4132 | 104/288 | 0.6408 |
| none | pixel_grid_sequence | hash | ridge | 0.2984 | 0.0991 | 0.2465 | 0.3508 | 86/288 | 0.5990 |
| none | flow_grid_sequence | hash | ridge | 0.4132 | 0.0507 | 0.3576 | 0.4688 | 119/288 | 0.5844 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3198 | 0.1213 | 0.2673 | 0.3750 | 92/288 | 0.6817 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3367 | 0.0642 | 0.2812 | 0.3889 | 97/288 | 0.6128 |
| high_motion | flow_grid_sequence | hash | ridge | 0.3674 | 0.0999 | 0.3090 | 0.4236 | 106/288 | 0.5675 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3474 | 0.1075 | 0.2917 | 0.3993 | 100/288 | 0.5703 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4096 | 0.1075 | 0.3507 | 0.4653 | 118/288 | 0.5846 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4033 | 0.0985 | 0.3438 | 0.4583 | 116/288 | 0.5614 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3546 | 0.1044 | 0.2986 | 0.4097 | 102/288 | 0.5939 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3612 | 0.0847 | 0.3056 | 0.4132 | 104/288 | 0.6263 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4409 | 0.1070 | 0.3819 | 0.4965 | 127/288 | 0.5892 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3333 | 24/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Action Order | 0.1389 | 10/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.3611 | 26/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.3333 | 24/72 |
| none | pixel_grid_sequence | Repetition Count | 0.3611 | 26/72 |
| none | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5000 | 36/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.3889 | 28/72 |
| none | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2361 | 17/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.3611 | 26/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.3472 | 25/72 |
| high_motion | flow_grid_sequence | Action Order | 0.2639 | 19/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.3611 | 26/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.3750 | 27/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.2361 | 17/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3333 | 24/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4028 | 29/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.3472 | 25/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.2639 | 19/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3333 | 24/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.3611 | 26/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.3750 | 27/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
