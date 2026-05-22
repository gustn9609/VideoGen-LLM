# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3754 | 0.1198 | 0.3194 | 0.4306 | 108/288 | 0.6715 |
| none | pixel_grid_sequence | hash | ridge | 0.4100 | 0.1397 | 0.3542 | 0.4654 | 118/288 | 0.6547 |
| none | flow_grid_sequence | hash | ridge | 0.4479 | 0.1222 | 0.3889 | 0.5036 | 129/288 | 0.6234 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3686 | 0.1189 | 0.3125 | 0.4236 | 106/288 | 0.6608 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.3961 | 0.1332 | 0.3403 | 0.4514 | 114/288 | 0.6514 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4139 | 0.1352 | 0.3576 | 0.4688 | 119/288 | 0.6369 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3754 | 0.1198 | 0.3194 | 0.4306 | 108/288 | 0.6716 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4268 | 0.1345 | 0.3715 | 0.4826 | 123/288 | 0.6541 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4202 | 0.1205 | 0.3611 | 0.4757 | 121/288 | 0.6480 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3721 | 0.1256 | 0.3160 | 0.4271 | 107/288 | 0.6631 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4028 | 0.1245 | 0.3472 | 0.4618 | 116/288 | 0.6516 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4556 | 0.1252 | 0.3958 | 0.5139 | 131/288 | 0.6621 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3333 | 24/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| none | flow_grid_sequence | Action Order | 0.2639 | 19/72 |
| none | flow_grid_sequence | Motion Recognition | 0.5556 | 40/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5556 | 40/72 |
| none | flow_grid_sequence | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4722 | 34/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.4861 | 35/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion | flow_grid_sequence | Action Order | 0.2361 | 17/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3333 | 24/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.3056 | 22/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2500 | 18/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4028 | 29/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2361 | 17/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4722 | 34/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5556 | 40/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4722 | 34/72 |
