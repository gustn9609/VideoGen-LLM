# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3649 | 0.1205 | 0.3124 | 0.4201 | 105/288 | 0.6757 |
| none | pixel_grid_sequence | hash | ridge | 0.4100 | 0.1170 | 0.3542 | 0.4688 | 118/288 | 0.6493 |
| none | flow_grid_sequence | hash | ridge | 0.4244 | 0.1089 | 0.3681 | 0.4792 | 122/288 | 0.6636 |
| high_motion | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3647 | 0.1216 | 0.3125 | 0.4201 | 105/288 | 0.6777 |
| high_motion | pixel_grid_sequence | hash | ridge | 0.4167 | 0.1251 | 0.3611 | 0.4757 | 120/288 | 0.6567 |
| high_motion | flow_grid_sequence | hash | ridge | 0.4346 | 0.1266 | 0.3750 | 0.4965 | 125/288 | 0.6358 |
| camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3682 | 0.1221 | 0.3125 | 0.4236 | 106/288 | 0.6754 |
| camera_comp | pixel_grid_sequence | hash | ridge | 0.4198 | 0.1233 | 0.3646 | 0.4792 | 121/288 | 0.6533 |
| camera_comp | flow_grid_sequence | hash | ridge | 0.4312 | 0.1210 | 0.3750 | 0.4861 | 124/288 | 0.6774 |
| high_motion+camera_comp | text_only | hash | ridge | 0.4553 | 0.1218 | 0.3958 | 0.5139 | 131/288 | 0.6107 |
| high_motion+camera_comp | wan_vae_grid_2x2 | hash | ridge | 0.3647 | 0.1216 | 0.3125 | 0.4201 | 105/288 | 0.6739 |
| high_motion+camera_comp | pixel_grid_sequence | hash | ridge | 0.4204 | 0.1190 | 0.3646 | 0.4758 | 121/288 | 0.6561 |
| high_motion+camera_comp | flow_grid_sequence | hash | ridge | 0.4312 | 0.1160 | 0.3715 | 0.4861 | 124/288 | 0.6454 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.5000 | 36/72 |
| none | text_only | Motion-related Objects | 0.5417 | 39/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2083 | 15/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| none | pixel_grid_sequence | Action Order | 0.2639 | 19/72 |
| none | pixel_grid_sequence | Motion Recognition | 0.4306 | 31/72 |
| none | pixel_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| none | pixel_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| none | flow_grid_sequence | Action Order | 0.2500 | 18/72 |
| none | flow_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| none | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| none | flow_grid_sequence | Repetition Count | 0.4861 | 35/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2083 | 15/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| high_motion | pixel_grid_sequence | Motion Recognition | 0.4167 | 30/72 |
| high_motion | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion | flow_grid_sequence | Action Order | 0.3333 | 24/72 |
| high_motion | flow_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| high_motion | flow_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| camera_comp | wan_vae_grid_2x2 | Action Order | 0.2222 | 16/72 |
| camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| camera_comp | pixel_grid_sequence | Action Order | 0.2778 | 20/72 |
| camera_comp | pixel_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| camera_comp | pixel_grid_sequence | Repetition Count | 0.4306 | 31/72 |
| camera_comp | flow_grid_sequence | Action Order | 0.2778 | 20/72 |
| camera_comp | flow_grid_sequence | Motion Recognition | 0.4444 | 32/72 |
| camera_comp | flow_grid_sequence | Motion-related Objects | 0.5556 | 40/72 |
| camera_comp | flow_grid_sequence | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | text_only | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | text_only | Motion Recognition | 0.5000 | 36/72 |
| high_motion+camera_comp | text_only | Motion-related Objects | 0.5417 | 39/72 |
| high_motion+camera_comp | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Action Order | 0.2083 | 15/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Motion-related Objects | 0.4861 | 35/72 |
| high_motion+camera_comp | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion+camera_comp | pixel_grid_sequence | Action Order | 0.3194 | 23/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion Recognition | 0.4028 | 29/72 |
| high_motion+camera_comp | pixel_grid_sequence | Motion-related Objects | 0.5000 | 36/72 |
| high_motion+camera_comp | pixel_grid_sequence | Repetition Count | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Action Order | 0.2917 | 21/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion Recognition | 0.4583 | 33/72 |
| high_motion+camera_comp | flow_grid_sequence | Motion-related Objects | 0.5139 | 37/72 |
| high_motion+camera_comp | flow_grid_sequence | Repetition Count | 0.4583 | 33/72 |
