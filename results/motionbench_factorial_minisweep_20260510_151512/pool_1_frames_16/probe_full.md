# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4346 | 0.1006 | 0.3750 | 0.4896 | 125/288 | 0.6479 |
| none | pixel_grid_sequence | hash | ridge | 0.4586 | 0.0929 | 0.4028 | 0.5174 | 132/288 | 0.6379 |
| none | flow_grid_sequence | hash | ridge | 0.4063 | 0.1122 | 0.3472 | 0.4618 | 117/288 | 0.6470 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4381 | 0.1045 | 0.3785 | 0.4931 | 126/288 | 0.6479 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4451 | 0.0957 | 0.3889 | 0.5035 | 128/288 | 0.6346 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4414 | 0.1255 | 0.3853 | 0.4965 | 127/288 | 0.6632 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4275 | 0.1063 | 0.3681 | 0.4826 | 123/288 | 0.6495 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4518 | 0.0962 | 0.3958 | 0.5104 | 130/288 | 0.6479 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4307 | 0.1277 | 0.3750 | 0.4861 | 124/288 | 0.6385 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_1x1 | hash | ridge | 0.4311 | 0.1072 | 0.3715 | 0.4861 | 124/288 | 0.6444 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4382 | 0.0980 | 0.3819 | 0.4965 | 126/288 | 0.6381 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4419 | 0.1160 | 0.3819 | 0.5000 | 127/288 | 0.6895 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| none | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| none | flow_grid_sequence | Action Order | 0.2222 | 16/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| none | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5000 | 36/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5556 | 40/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_1x1 | Action Order | 0.3056 | 22/72 |
| camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5000 | 36/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3472 | 25/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4861 | 35/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2361 | 17/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | wan_vae_grid_1x1 | Repetition Count | 0.5139 | 37/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.3889 | 28/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5278 | 38/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
