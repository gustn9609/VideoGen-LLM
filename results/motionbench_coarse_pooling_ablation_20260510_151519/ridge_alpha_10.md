# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3433 | 0.0629 | 0.2882 | 0.3993 | 99/288 | 0.3562 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.3812 | 0.1266 | 0.3264 | 0.4375 | 110/288 | 0.3299 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3958 | 0.0825 | 0.3438 | 0.4514 | 114/288 | 0.3874 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3640 | 0.1092 | 0.3090 | 0.4201 | 105/288 | 0.3350 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3537 | 0.0954 | 0.3021 | 0.4097 | 102/288 | 0.3185 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3567 | 0.1199 | 0.3021 | 0.4132 | 103/288 | 0.3462 |
| high_motion | text_only | hash | ridge | 0.3433 | 0.0629 | 0.2882 | 0.3993 | 99/288 | 0.3562 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.3986 | 0.1247 | 0.3438 | 0.4549 | 115/288 | 0.3239 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3579 | 0.0770 | 0.3021 | 0.4132 | 103/288 | 0.3472 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3018 | 0.0704 | 0.2499 | 0.3542 | 87/288 | 0.3596 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3646 | 0.1098 | 0.3090 | 0.4167 | 105/288 | 0.3354 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.3751 | 0.0840 | 0.3194 | 0.4272 | 108/288 | 0.3024 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.3194 | 23/72 |
| none | text_only | Motion Recognition | 0.3194 | 23/72 |
| none | text_only | Motion-related Objects | 0.3194 | 23/72 |
| none | text_only | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.3333 | 24/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.3472 | 25/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.4028 | 29/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.2778 | 20/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.3056 | 22/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.3194 | 23/72 |
| high_motion | text_only | Motion Recognition | 0.3194 | 23/72 |
| high_motion | text_only | Motion-related Objects | 0.3194 | 23/72 |
| high_motion | text_only | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.1944 | 14/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.2222 | 16/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.2361 | 17/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.4167 | 30/72 |
