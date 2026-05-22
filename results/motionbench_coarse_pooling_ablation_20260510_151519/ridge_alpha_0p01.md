# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3504 | 0.0755 | 0.2951 | 0.4062 | 101/288 | 0.6614 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.3746 | 0.1236 | 0.3194 | 0.4306 | 108/288 | 1.2070 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3228 | 0.1032 | 0.2674 | 0.3785 | 93/288 | 2.0801 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3647 | 0.0816 | 0.3124 | 0.4201 | 105/288 | 1.5239 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3330 | 0.0916 | 0.2812 | 0.3854 | 96/288 | 1.2976 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3433 | 0.0685 | 0.2882 | 0.3993 | 99/288 | 1.2319 |
| high_motion | text_only | hash | ridge | 0.3504 | 0.0755 | 0.2951 | 0.4062 | 101/288 | 0.6614 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.3540 | 0.1358 | 0.2986 | 0.4062 | 102/288 | 1.2334 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3095 | 0.0950 | 0.2569 | 0.3611 | 89/288 | 2.1713 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.2875 | 0.0925 | 0.2361 | 0.3368 | 83/288 | 1.5601 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3267 | 0.1161 | 0.2743 | 0.3819 | 94/288 | 1.1875 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.4104 | 0.1108 | 0.3542 | 0.4653 | 118/288 | 1.1538 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.2639 | 19/72 |
| none | text_only | Motion Recognition | 0.3333 | 24/72 |
| none | text_only | Motion-related Objects | 0.3472 | 25/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.2778 | 20/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.4444 | 32/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.4028 | 29/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.2917 | 21/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.1806 | 13/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3333 | 24/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3611 | 26/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.2500 | 18/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.2639 | 19/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.3333 | 24/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.3056 | 22/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.2639 | 19/72 |
| high_motion | text_only | Motion Recognition | 0.3333 | 24/72 |
| high_motion | text_only | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.1944 | 14/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2500 | 18/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.1667 | 12/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.2083 | 15/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.4722 | 34/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.4306 | 31/72 |
