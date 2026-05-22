# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | text_only | hash | ridge | 0.3468 | 0.0758 | 0.2917 | 0.4028 | 100/288 | 0.6499 |
| none | wan_vae_grid_1x1 | hash | ridge | 0.4404 | 0.1238 | 0.3819 | 0.5000 | 127/288 | 0.7952 |
| none | wan_vae_grid_2x2 | hash | ridge | 0.3404 | 0.1101 | 0.2812 | 0.3958 | 98/288 | 1.3170 |
| none | wan_vae_grid_4x4 | hash | ridge | 0.3398 | 0.0762 | 0.2882 | 0.3958 | 98/288 | 1.0719 |
| none | wan_vae_grid_8x8 | hash | ridge | 0.3365 | 0.1199 | 0.2812 | 0.3889 | 97/288 | 1.0382 |
| none | wan_vae_grid_16x16 | hash | ridge | 0.3398 | 0.0762 | 0.2882 | 0.3959 | 98/288 | 1.0294 |
| high_motion | text_only | hash | ridge | 0.3468 | 0.0758 | 0.2917 | 0.4028 | 100/288 | 0.6499 |
| high_motion | wan_vae_grid_1x1 | hash | ridge | 0.4053 | 0.1536 | 0.3507 | 0.4618 | 117/288 | 0.8059 |
| high_motion | wan_vae_grid_2x2 | hash | ridge | 0.3305 | 0.1071 | 0.2778 | 0.3819 | 95/288 | 1.3590 |
| high_motion | wan_vae_grid_4x4 | hash | ridge | 0.3161 | 0.0886 | 0.2604 | 0.3681 | 91/288 | 1.0338 |
| high_motion | wan_vae_grid_8x8 | hash | ridge | 0.3439 | 0.1060 | 0.2916 | 0.3958 | 99/288 | 0.9315 |
| high_motion | wan_vae_grid_16x16 | hash | ridge | 0.4098 | 0.0980 | 0.3542 | 0.4653 | 118/288 | 0.9979 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | text_only | Action Order | 0.2639 | 19/72 |
| none | text_only | Motion Recognition | 0.3333 | 24/72 |
| none | text_only | Motion-related Objects | 0.3472 | 25/72 |
| none | text_only | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_1x1 | Action Order | 0.4444 | 32/72 |
| none | wan_vae_grid_1x1 | Motion Recognition | 0.3889 | 28/72 |
| none | wan_vae_grid_1x1 | Motion-related Objects | 0.4722 | 34/72 |
| none | wan_vae_grid_1x1 | Repetition Count | 0.4583 | 33/72 |
| none | wan_vae_grid_2x2 | Action Order | 0.3750 | 27/72 |
| none | wan_vae_grid_2x2 | Motion Recognition | 0.3056 | 22/72 |
| none | wan_vae_grid_2x2 | Motion-related Objects | 0.2639 | 19/72 |
| none | wan_vae_grid_2x2 | Repetition Count | 0.4167 | 30/72 |
| none | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| none | wan_vae_grid_4x4 | Motion Recognition | 0.3056 | 22/72 |
| none | wan_vae_grid_4x4 | Motion-related Objects | 0.3194 | 23/72 |
| none | wan_vae_grid_4x4 | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_8x8 | Action Order | 0.3056 | 22/72 |
| none | wan_vae_grid_8x8 | Motion Recognition | 0.3333 | 24/72 |
| none | wan_vae_grid_8x8 | Motion-related Objects | 0.2639 | 19/72 |
| none | wan_vae_grid_8x8 | Repetition Count | 0.4444 | 32/72 |
| none | wan_vae_grid_16x16 | Action Order | 0.4028 | 29/72 |
| none | wan_vae_grid_16x16 | Motion Recognition | 0.2639 | 19/72 |
| none | wan_vae_grid_16x16 | Motion-related Objects | 0.2778 | 20/72 |
| none | wan_vae_grid_16x16 | Repetition Count | 0.4167 | 30/72 |
| high_motion | text_only | Action Order | 0.2639 | 19/72 |
| high_motion | text_only | Motion Recognition | 0.3333 | 24/72 |
| high_motion | text_only | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | text_only | Repetition Count | 0.4444 | 32/72 |
| high_motion | wan_vae_grid_1x1 | Action Order | 0.4167 | 30/72 |
| high_motion | wan_vae_grid_1x1 | Motion Recognition | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_1x1 | Motion-related Objects | 0.3750 | 27/72 |
| high_motion | wan_vae_grid_1x1 | Repetition Count | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_2x2 | Action Order | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_2x2 | Motion Recognition | 0.3194 | 23/72 |
| high_motion | wan_vae_grid_2x2 | Motion-related Objects | 0.2083 | 15/72 |
| high_motion | wan_vae_grid_2x2 | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_4x4 | Action Order | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_4x4 | Motion Recognition | 0.2222 | 16/72 |
| high_motion | wan_vae_grid_4x4 | Motion-related Objects | 0.2917 | 21/72 |
| high_motion | wan_vae_grid_4x4 | Repetition Count | 0.4583 | 33/72 |
| high_motion | wan_vae_grid_8x8 | Action Order | 0.2639 | 19/72 |
| high_motion | wan_vae_grid_8x8 | Motion Recognition | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_8x8 | Motion-related Objects | 0.3472 | 25/72 |
| high_motion | wan_vae_grid_8x8 | Repetition Count | 0.4306 | 31/72 |
| high_motion | wan_vae_grid_16x16 | Action Order | 0.3333 | 24/72 |
| high_motion | wan_vae_grid_16x16 | Motion Recognition | 0.4722 | 34/72 |
| high_motion | wan_vae_grid_16x16 | Motion-related Objects | 0.4028 | 29/72 |
| high_motion | wan_vae_grid_16x16 | Repetition Count | 0.4306 | 31/72 |
