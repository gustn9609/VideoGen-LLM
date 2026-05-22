# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3751 | 0.1000 | 0.3194 | 0.4306 | 108/288 | 0.5972 |
| none | pixel_grid_sequence | hash | ridge | 0.3709 | 0.1087 | 0.3160 | 0.4237 | 107/288 | 0.5734 |
| none | flow_grid_sequence | hash | ridge | 0.3884 | 0.1065 | 0.3332 | 0.4444 | 112/288 | 0.5961 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3861 | 0.0990 | 0.3298 | 0.4444 | 111/288 | 0.6327 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3775 | 0.0895 | 0.3229 | 0.4340 | 109/288 | 0.5758 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4474 | 0.1167 | 0.3924 | 0.5035 | 129/288 | 0.5579 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3854 | 0.1007 | 0.3264 | 0.4376 | 111/288 | 0.5992 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.3709 | 0.0998 | 0.3160 | 0.4236 | 107/288 | 0.5737 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.3818 | 0.1006 | 0.3264 | 0.4375 | 110/288 | 0.5810 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3896 | 0.1029 | 0.3333 | 0.4444 | 112/288 | 0.6339 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3707 | 0.0911 | 0.3160 | 0.4271 | 107/288 | 0.5783 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4442 | 0.1272 | 0.3889 | 0.4965 | 128/288 | 0.5552 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3472 | 25/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.4028 | 29/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| none | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| none | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.3889 | 28/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3750 | 27/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4861 | 35/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.3750 | 27/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.4028 | 29/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2361 | 17/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.3750 | 27/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2500 | 18/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.3750 | 27/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.5278 | 38/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
