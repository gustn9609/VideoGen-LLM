# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.2984 | 0.1773 | 0.2083 | 0.3958 | 29/96 | 0.8785 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.3429 | 0.1586 | 0.2604 | 0.4375 | 33/96 | 0.6600 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3286 | 0.1589 | 0.2500 | 0.4271 | 32/96 | 0.6035 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.2683 | 0.1515 | 0.1875 | 0.3544 | 26/96 | 0.5381 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.2905 | 0.1302 | 0.2083 | 0.3854 | 28/96 | 0.5986 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3190 | 0.1274 | 0.2396 | 0.4167 | 31/96 | 0.6231 |
| high_motion | text_only | hash | ridge | 0.1873 | 0.1508 | 0.1146 | 0.2708 | 18/96 | 0.6835 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.2095 | 0.1752 | 0.1354 | 0.2917 | 20/96 | 0.7896 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.2952 | 0.1500 | 0.1979 | 0.3854 | 28/96 | 0.5565 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.2508 | 0.1232 | 0.1667 | 0.3438 | 24/96 | 0.5418 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.2587 | 0.1545 | 0.1771 | 0.3438 | 25/96 | 0.7100 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.2603 | 0.1345 | 0.1771 | 0.3542 | 25/96 | 0.7848 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.2500 | 6/24 |
| none | text_only | Motion Recognition | 0.1250 | 3/24 |
| none | text_only | Motion-related Objects | 0.4167 | 10/24 |
| none | text_only | Repetition Count | 0.4167 | 10/24 |
| none | wan_vae_grid_1x1 | Action Order | 0.5000 | 12/24 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 10/24 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.1667 | 4/24 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.2917 | 7/24 |
| none | wan_vae_grid_2x2 | Action Order | 0.4583 | 11/24 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.2917 | 7/24 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.2500 | 6/24 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.3333 | 8/24 |
| none | wan_vae_grid_4x4 | Action Order | 0.3750 | 9/24 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.0833 | 2/24 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.2500 | 6/24 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3750 | 9/24 |
| none | wan_vae_grid_8x8 | Action Order | 0.4167 | 10/24 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.1667 | 4/24 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.2500 | 6/24 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.3333 | 8/24 |
| none | wan_vae_grid_16x16 | Action Order | 0.3750 | 9/24 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.2917 | 7/24 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.2917 | 7/24 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.3333 | 8/24 |
| high_motion | text_only | Action Order | 0.1250 | 3/24 |
| high_motion | text_only | Motion Recognition | 0.0000 | 0/24 |
| high_motion | text_only | Motion-related Objects | 0.3750 | 9/24 |
| high_motion | text_only | Repetition Count | 0.2500 | 6/24 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.2500 | 6/24 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.0417 | 1/24 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.2917 | 7/24 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.2500 | 6/24 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2917 | 7/24 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.1667 | 4/24 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.4583 | 11/24 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.2500 | 6/24 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2083 | 5/24 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.1250 | 3/24 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.4167 | 10/24 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.2500 | 6/24 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.2917 | 7/24 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.1250 | 3/24 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3750 | 9/24 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.2500 | 6/24 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.2083 | 5/24 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.1250 | 3/24 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.4583 | 11/24 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.2500 | 6/24 |
