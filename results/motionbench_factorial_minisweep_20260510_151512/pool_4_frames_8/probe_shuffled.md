# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3754 | 0.1041 | 0.3194 | 0.4271 | 108/288 | 0.6142 |
| none | pixel_grid_sequence | hash | ridge | 0.3674 | 0.0893 | 0.3125 | 0.4201 | 106/288 | 0.5846 |
| none | flow_grid_sequence | hash | ridge | 0.3914 | 0.0958 | 0.3368 | 0.4479 | 113/288 | 0.5694 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3895 | 0.0953 | 0.3333 | 0.4444 | 112/288 | 0.6429 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3674 | 0.0913 | 0.3125 | 0.4236 | 106/288 | 0.5885 |
| high_motion | flow_grid_sequence | hash | ridge | 0.3956 | 0.0992 | 0.3403 | 0.4514 | 114/288 | 0.6046 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3651 | 0.1142 | 0.3090 | 0.4201 | 105/288 | 0.6151 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.3707 | 0.0931 | 0.3160 | 0.4271 | 107/288 | 0.5833 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4019 | 0.1032 | 0.3472 | 0.4618 | 116/288 | 0.5741 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_4x4 | hash | ridge | 0.3860 | 0.0969 | 0.3299 | 0.4410 | 111/288 | 0.6512 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.3672 | 0.0838 | 0.3125 | 0.4201 | 106/288 | 0.5884 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4025 | 0.1162 | 0.3472 | 0.4583 | 116/288 | 0.5929 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4167 | 30/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3889 | 28/72 |
| none | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| none | flow_grid_sequence | Action Order | 0.1944 | 14/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.4583 | 33/72 |
| none | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2361 | 17/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_4x4 | Action Order | 0.2778 | 20/72 |
| camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.4028 | 29/72 |
| camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.3889 | 28/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2500 | 18/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2361 | 17/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Action Order | 0.3472 | 25/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion Recognition | 0.4306 | 31/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| high_motion+camera_comp | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 27/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2500 | 18/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.4167 | 30/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3056 | 22/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.4028 | 29/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
