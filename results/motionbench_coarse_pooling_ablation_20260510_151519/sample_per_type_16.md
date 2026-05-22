# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.2303 | 0.1105 | 0.1719 | 0.2865 | 44/192 | 0.6999 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.3491 | 0.1507 | 0.2812 | 0.4167 | 67/192 | 0.5837 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.2701 | 0.1207 | 0.2135 | 0.3333 | 52/192 | 0.7421 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.2812 | 0.0969 | 0.2240 | 0.3438 | 54/192 | 0.7722 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3406 | 0.1226 | 0.2760 | 0.4010 | 65/192 | 0.7735 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.2662 | 0.1189 | 0.2031 | 0.3281 | 51/192 | 0.6402 |
| high_motion | text_only | hash | ridge | 0.3752 | 0.0984 | 0.3073 | 0.4479 | 72/192 | 0.7858 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4158 | 0.1250 | 0.3438 | 0.4896 | 80/192 | 0.7259 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3808 | 0.0906 | 0.3125 | 0.4531 | 73/192 | 0.8625 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.4073 | 0.1333 | 0.3385 | 0.4792 | 78/192 | 0.7831 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.4688 | 0.1231 | 0.4010 | 0.5417 | 90/192 | 0.7300 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.4432 | 0.1267 | 0.3750 | 0.5104 | 85/192 | 0.7570 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.1458 | 7/48 |
| none | text_only | Motion Recognition | 0.3333 | 16/48 |
| none | text_only | Motion-related Objects | 0.2083 | 10/48 |
| none | text_only | Repetition Count | 0.2292 | 11/48 |
| none | wan_vae_grid_1x1 | Action Order | 0.4792 | 23/48 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.3542 | 17/48 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.3125 | 15/48 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.2500 | 12/48 |
| none | wan_vae_grid_2x2 | Action Order | 0.3542 | 17/48 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.2917 | 14/48 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.1875 | 9/48 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.2500 | 12/48 |
| none | wan_vae_grid_4x4 | Action Order | 0.3333 | 16/48 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.2708 | 13/48 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.2708 | 13/48 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.2500 | 12/48 |
| none | wan_vae_grid_8x8 | Action Order | 0.4167 | 20/48 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3333 | 16/48 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.3542 | 17/48 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.2500 | 12/48 |
| none | wan_vae_grid_16x16 | Action Order | 0.2917 | 14/48 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.2917 | 14/48 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.2500 | 12/48 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.2292 | 11/48 |
| high_motion | text_only | Action Order | 0.3125 | 15/48 |
| high_motion | text_only | Motion Recognition | 0.5417 | 26/48 |
| high_motion | text_only | Motion-related Objects | 0.1042 | 5/48 |
| high_motion | text_only | Repetition Count | 0.5417 | 26/48 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.4792 | 23/48 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.3958 | 19/48 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.2500 | 12/48 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.5417 | 26/48 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.3125 | 15/48 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.5000 | 24/48 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.1875 | 9/48 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.5208 | 25/48 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.5000 | 24/48 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.3958 | 19/48 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.1458 | 7/48 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.5833 | 28/48 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.4583 | 22/48 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.3958 | 19/48 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.4167 | 20/48 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.6042 | 29/48 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.5625 | 27/48 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.3750 | 18/48 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.2500 | 12/48 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.5833 | 28/48 |
