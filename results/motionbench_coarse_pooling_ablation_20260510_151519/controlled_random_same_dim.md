# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3468 | 0.0782 | 0.2951 | 0.4028 | 100/288 | 0.5655 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4088 | 0.1349 | 0.3542 | 0.4688 | 118/288 | 0.5578 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3646 | 0.1070 | 0.3125 | 0.4201 | 105/288 | 0.5433 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3184 | 0.0866 | 0.2673 | 0.3750 | 92/288 | 0.6204 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3705 | 0.1201 | 0.3160 | 0.4271 | 107/288 | 0.6206 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3404 | 0.1131 | 0.2882 | 0.3924 | 98/288 | 0.7352 |
| high_motion | text_only | hash | ridge | 0.3468 | 0.0782 | 0.2951 | 0.4028 | 100/288 | 0.5655 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.3946 | 0.1504 | 0.3403 | 0.4514 | 114/288 | 0.5397 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3882 | 0.1265 | 0.3333 | 0.4445 | 112/288 | 0.6118 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3404 | 0.1176 | 0.2847 | 0.3958 | 98/288 | 0.6224 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3749 | 0.1101 | 0.3194 | 0.4306 | 108/288 | 0.5640 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.3572 | 0.0800 | 0.3021 | 0.4097 | 103/288 | 0.6584 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.2778 | 20/72 |
| none | text_only | Motion Recognition | 0.3194 | 23/72 |
| none | text_only | Motion-related Objects | 0.3472 | 25/72 |
| none | text_only | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.3750 | 27/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.3194 | 23/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3056 | 22/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.1528 | 11/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.3056 | 22/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.4028 | 29/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.3889 | 28/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.3889 | 28/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.2361 | 17/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.3611 | 26/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.4444 | 32/72 |
| high_motion | text_only | Action Order | 0.2778 | 20/72 |
| high_motion | text_only | Motion Recognition | 0.3194 | 23/72 |
| high_motion | text_only | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | text_only | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.3056 | 22/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.2222 | 16/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.4722 | 34/72 |
