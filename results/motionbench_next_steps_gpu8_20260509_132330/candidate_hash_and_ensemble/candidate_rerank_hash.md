# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | hash | logistic | 0.3811 | 0.0867 | 0.3375 | 0.4250 | 183/480 | 2.2460 |
| none | wan_vae_global_sequence | hash | logistic | 0.4396 | 0.1282 | 0.3937 | 0.4854 | 211/480 | 2.6558 |
| none | wan_vae_global_delta | hash | logistic | 0.4539 | 0.1188 | 0.4083 | 0.4958 | 218/480 | 2.6827 |
| none | pixel_grid_sequence | hash | logistic | 0.3942 | 0.1125 | 0.3500 | 0.4375 | 189/480 | 2.4971 |
| none | flow_grid_sequence | hash | logistic | 0.4138 | 0.0882 | 0.3688 | 0.4604 | 199/480 | 2.3373 |
| uniform5 | wan_vae_grid_sequence | hash | logistic | 0.3848 | 0.1150 | 0.3417 | 0.4292 | 185/480 | 2.3836 |
| uniform5 | wan_vae_global_sequence | hash | logistic | 0.4353 | 0.1186 | 0.3916 | 0.4792 | 209/480 | 2.6494 |
| uniform5 | wan_vae_global_delta | hash | logistic | 0.4311 | 0.1226 | 0.3875 | 0.4750 | 207/480 | 2.6688 |
| uniform5 | pixel_grid_sequence | hash | logistic | 0.3974 | 0.0987 | 0.3563 | 0.4417 | 191/480 | 2.4535 |
| uniform5 | flow_grid_sequence | hash | logistic | 0.4040 | 0.0840 | 0.3625 | 0.4500 | 194/480 | 2.2587 |
| nonuniform5 | wan_vae_grid_sequence | hash | logistic | 0.3752 | 0.1077 | 0.3312 | 0.4188 | 180/480 | 2.3393 |
| nonuniform5 | wan_vae_global_sequence | hash | logistic | 0.4479 | 0.1220 | 0.4042 | 0.4917 | 215/480 | 2.6503 |
| nonuniform5 | wan_vae_global_delta | hash | logistic | 0.4539 | 0.1068 | 0.4104 | 0.5000 | 218/480 | 2.6409 |
| nonuniform5 | pixel_grid_sequence | hash | logistic | 0.4185 | 0.0887 | 0.3750 | 0.4625 | 201/480 | 2.1502 |
| nonuniform5 | flow_grid_sequence | hash | logistic | 0.4269 | 0.1061 | 0.3854 | 0.4750 | 205/480 | 2.1319 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_sequence | Action Order | 0.2417 | 29/120 |
| none | wan_vae_grid_sequence | Motion Recognition | 0.4333 | 52/120 |
| none | wan_vae_grid_sequence | Motion-related Objects | 0.4667 | 56/120 |
| none | wan_vae_grid_sequence | Repetition Count | 0.3833 | 46/120 |
| none | wan_vae_global_sequence | Action Order | 0.2917 | 35/120 |
| none | wan_vae_global_sequence | Motion Recognition | 0.5333 | 64/120 |
| none | wan_vae_global_sequence | Motion-related Objects | 0.5083 | 61/120 |
| none | wan_vae_global_sequence | Repetition Count | 0.4250 | 51/120 |
| none | wan_vae_global_delta | Action Order | 0.3417 | 41/120 |
| none | wan_vae_global_delta | Motion Recognition | 0.5417 | 65/120 |
| none | wan_vae_global_delta | Motion-related Objects | 0.5167 | 62/120 |
| none | wan_vae_global_delta | Repetition Count | 0.4167 | 50/120 |
| none | pixel_grid_sequence | Action Order | 0.3500 | 42/120 |
| none | pixel_grid_sequence | Motion Recognition | 0.4000 | 48/120 |
| none | pixel_grid_sequence | Motion-related Objects | 0.4500 | 54/120 |
| none | pixel_grid_sequence | Repetition Count | 0.3750 | 45/120 |
| none | flow_grid_sequence | Action Order | 0.3500 | 42/120 |
| none | flow_grid_sequence | Motion Recognition | 0.4583 | 55/120 |
| none | flow_grid_sequence | Motion-related Objects | 0.3917 | 47/120 |
| none | flow_grid_sequence | Repetition Count | 0.4583 | 55/120 |
| uniform5 | wan_vae_grid_sequence | Action Order | 0.2500 | 30/120 |
| uniform5 | wan_vae_grid_sequence | Motion Recognition | 0.4583 | 55/120 |
| uniform5 | wan_vae_grid_sequence | Motion-related Objects | 0.4250 | 51/120 |
| uniform5 | wan_vae_grid_sequence | Repetition Count | 0.4083 | 49/120 |
| uniform5 | wan_vae_global_sequence | Action Order | 0.2750 | 33/120 |
| uniform5 | wan_vae_global_sequence | Motion Recognition | 0.5250 | 63/120 |
| uniform5 | wan_vae_global_sequence | Motion-related Objects | 0.5250 | 63/120 |
| uniform5 | wan_vae_global_sequence | Repetition Count | 0.4167 | 50/120 |
| uniform5 | wan_vae_global_delta | Action Order | 0.2833 | 34/120 |
| uniform5 | wan_vae_global_delta | Motion Recognition | 0.5083 | 61/120 |
| uniform5 | wan_vae_global_delta | Motion-related Objects | 0.5167 | 62/120 |
| uniform5 | wan_vae_global_delta | Repetition Count | 0.4167 | 50/120 |
| uniform5 | pixel_grid_sequence | Action Order | 0.2917 | 35/120 |
| uniform5 | pixel_grid_sequence | Motion Recognition | 0.4500 | 54/120 |
| uniform5 | pixel_grid_sequence | Motion-related Objects | 0.4250 | 51/120 |
| uniform5 | pixel_grid_sequence | Repetition Count | 0.4250 | 51/120 |
| uniform5 | flow_grid_sequence | Action Order | 0.3083 | 37/120 |
| uniform5 | flow_grid_sequence | Motion Recognition | 0.5333 | 64/120 |
| uniform5 | flow_grid_sequence | Motion-related Objects | 0.3417 | 41/120 |
| uniform5 | flow_grid_sequence | Repetition Count | 0.4333 | 52/120 |
| nonuniform5 | wan_vae_grid_sequence | Action Order | 0.2917 | 35/120 |
| nonuniform5 | wan_vae_grid_sequence | Motion Recognition | 0.4000 | 48/120 |
| nonuniform5 | wan_vae_grid_sequence | Motion-related Objects | 0.4417 | 53/120 |
| nonuniform5 | wan_vae_grid_sequence | Repetition Count | 0.3667 | 44/120 |
| nonuniform5 | wan_vae_global_sequence | Action Order | 0.3083 | 37/120 |
| nonuniform5 | wan_vae_global_sequence | Motion Recognition | 0.5333 | 64/120 |
| nonuniform5 | wan_vae_global_sequence | Motion-related Objects | 0.5250 | 63/120 |
| nonuniform5 | wan_vae_global_sequence | Repetition Count | 0.4250 | 51/120 |
| nonuniform5 | wan_vae_global_delta | Action Order | 0.3083 | 37/120 |
| nonuniform5 | wan_vae_global_delta | Motion Recognition | 0.5333 | 64/120 |
| nonuniform5 | wan_vae_global_delta | Motion-related Objects | 0.5417 | 65/120 |
| nonuniform5 | wan_vae_global_delta | Repetition Count | 0.4333 | 52/120 |
| nonuniform5 | pixel_grid_sequence | Action Order | 0.3333 | 40/120 |
| nonuniform5 | pixel_grid_sequence | Motion Recognition | 0.4917 | 59/120 |
| nonuniform5 | pixel_grid_sequence | Motion-related Objects | 0.4083 | 49/120 |
| nonuniform5 | pixel_grid_sequence | Repetition Count | 0.4417 | 53/120 |
| nonuniform5 | flow_grid_sequence | Action Order | 0.3750 | 45/120 |
| nonuniform5 | flow_grid_sequence | Motion Recognition | 0.5667 | 68/120 |
| nonuniform5 | flow_grid_sequence | Motion-related Objects | 0.3333 | 40/120 |
| nonuniform5 | flow_grid_sequence | Repetition Count | 0.4333 | 52/120 |
