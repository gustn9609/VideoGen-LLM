# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3468 | 0.0782 | 0.2951 | 0.4028 | 100/288 | 0.5655 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4088 | 0.1349 | 0.3542 | 0.4688 | 118/288 | 0.5578 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3582 | 0.0935 | 0.3021 | 0.4097 | 103/288 | 0.7701 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3433 | 0.0823 | 0.2882 | 0.3993 | 99/288 | 0.7053 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3470 | 0.1016 | 0.2917 | 0.3993 | 100/288 | 0.6429 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3716 | 0.0935 | 0.3160 | 0.4271 | 107/288 | 0.6938 |
| high_motion | text_only | hash | ridge | 0.3468 | 0.0782 | 0.2951 | 0.4028 | 100/288 | 0.5655 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.3946 | 0.1504 | 0.3403 | 0.4514 | 114/288 | 0.5397 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3684 | 0.0979 | 0.3125 | 0.4236 | 106/288 | 0.7127 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3054 | 0.0910 | 0.2534 | 0.3576 | 88/288 | 0.6845 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3437 | 0.0730 | 0.2882 | 0.3958 | 99/288 | 0.6554 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.4061 | 0.0851 | 0.3472 | 0.4618 | 117/288 | 0.6437 |

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
| none | wan_vae_grid_2x2 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.2917 | 21/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.2778 | 20/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.3611 | 26/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3194 | 23/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.2778 | 20/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.4306 | 31/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.4028 | 29/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.2917 | 21/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.3750 | 27/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.2778 | 20/72 |
| high_motion | text_only | Motion Recognition | 0.3194 | 23/72 |
| high_motion | text_only | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | text_only | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.2500 | 18/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.2222 | 16/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.2778 | 20/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.3611 | 26/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3889 | 28/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.4444 | 32/72 |
