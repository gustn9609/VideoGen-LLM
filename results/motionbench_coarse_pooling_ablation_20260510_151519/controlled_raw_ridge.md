# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3225 | 0.1109 | 0.2708 | 0.3750 | 93/288 | 0.6266 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.3260 | 0.1064 | 0.2742 | 0.3819 | 94/288 | 0.6238 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3330 | 0.1008 | 0.2778 | 0.3889 | 96/288 | 0.6209 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3298 | 0.1025 | 0.2778 | 0.3854 | 95/288 | 1.5233 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3463 | 0.1293 | 0.2951 | 0.3993 | 100/288 | 0.6589 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3533 | 0.1212 | 0.2986 | 0.4062 | 102/288 | 0.6565 |
| high_motion | text_only | hash | ridge | 0.3225 | 0.1109 | 0.2708 | 0.3750 | 93/288 | 0.6266 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.3226 | 0.1084 | 0.2674 | 0.3785 | 93/288 | 0.6239 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3330 | 0.1008 | 0.2778 | 0.3889 | 96/288 | 0.6304 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3505 | 0.0925 | 0.2986 | 0.4062 | 101/288 | 0.9977 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3428 | 0.1294 | 0.2917 | 0.3958 | 99/288 | 0.6623 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.3498 | 0.1230 | 0.2951 | 0.4029 | 101/288 | 0.6573 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.2778 | 20/72 |
| none | text_only | Motion Recognition | 0.3750 | 27/72 |
| none | text_only | Motion-related Objects | 0.2778 | 20/72 |
| none | text_only | Repetition Count | 0.3611 | 26/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.2917 | 21/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.3611 | 26/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.2778 | 20/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.4028 | 29/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.2361 | 17/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3611 | 26/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.3056 | 22/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.3611 | 26/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.3056 | 22/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.3611 | 26/72 |
| high_motion | text_only | Action Order | 0.2778 | 20/72 |
| high_motion | text_only | Motion Recognition | 0.3750 | 27/72 |
| high_motion | text_only | Motion-related Objects | 0.2778 | 20/72 |
| high_motion | text_only | Repetition Count | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.2500 | 18/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.2778 | 20/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.3472 | 25/72 |
