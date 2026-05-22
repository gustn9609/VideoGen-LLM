# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3330 | 0.0940 | 0.2812 | 0.3854 | 96/288 | 0.1427 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.3639 | 0.1281 | 0.3090 | 0.4201 | 105/288 | 0.1321 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.4200 | 0.0808 | 0.3646 | 0.4792 | 121/288 | 0.1350 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3226 | 0.0802 | 0.2707 | 0.3750 | 93/288 | 0.1007 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3358 | 0.0965 | 0.2812 | 0.3924 | 97/288 | 0.0995 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3425 | 0.1153 | 0.2917 | 0.3993 | 99/288 | 0.1053 |
| high_motion | text_only | hash | ridge | 0.3330 | 0.0940 | 0.2812 | 0.3854 | 96/288 | 0.1427 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.3742 | 0.1064 | 0.3194 | 0.4306 | 108/288 | 0.1324 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3472 | 0.0697 | 0.2917 | 0.4028 | 100/288 | 0.1299 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3158 | 0.0889 | 0.2604 | 0.3681 | 91/288 | 0.1091 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3540 | 0.1141 | 0.3021 | 0.4097 | 102/288 | 0.1087 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.3328 | 0.0868 | 0.2812 | 0.3855 | 96/288 | 0.1005 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.2917 | 21/72 |
| none | text_only | Motion Recognition | 0.2917 | 21/72 |
| none | text_only | Motion-related Objects | 0.2917 | 21/72 |
| none | text_only | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.2778 | 20/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4583 | 33/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.2917 | 21/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.4167 | 30/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.4306 | 31/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3056 | 22/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.2361 | 17/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.2639 | 19/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.2639 | 19/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.3056 | 22/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.4028 | 29/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.2500 | 18/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.2917 | 21/72 |
| high_motion | text_only | Motion Recognition | 0.2917 | 21/72 |
| high_motion | text_only | Motion-related Objects | 0.2917 | 21/72 |
| high_motion | text_only | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.1667 | 12/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.1528 | 11/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.2083 | 15/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.3889 | 28/72 |
