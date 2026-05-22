# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | clip | logistic | 0.4333 | 0.1203 | 0.3916 | 0.4771 | 208/480 | 3.0304 |
| none | pixel_grid_sequence | clip | logistic | 0.3849 | 0.1262 | 0.3417 | 0.4271 | 185/480 | 2.8149 |
| none | flow_grid_sequence | clip | logistic | 0.4294 | 0.1237 | 0.3854 | 0.4708 | 206/480 | 2.7964 |
| uniform5 | wan_vae_grid_sequence | clip | logistic | 0.4564 | 0.1075 | 0.4124 | 0.5000 | 219/480 | 2.8248 |
| uniform5 | pixel_grid_sequence | clip | logistic | 0.4721 | 0.0985 | 0.4292 | 0.5167 | 227/480 | 2.4605 |
| uniform5 | flow_grid_sequence | clip | logistic | 0.4624 | 0.1140 | 0.4188 | 0.5062 | 222/480 | 2.8415 |
| nonuniform5 | wan_vae_grid_sequence | clip | logistic | 0.4457 | 0.1213 | 0.4020 | 0.4896 | 214/480 | 2.9682 |
| nonuniform5 | pixel_grid_sequence | clip | logistic | 0.4417 | 0.0984 | 0.3979 | 0.4834 | 212/480 | 2.7743 |
| nonuniform5 | flow_grid_sequence | clip | logistic | 0.4418 | 0.1108 | 0.3979 | 0.4833 | 212/480 | 2.9219 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_sequence | Action Order | 0.3583 | 43/120 |
| none | wan_vae_grid_sequence | Motion Recognition | 0.4833 | 58/120 |
| none | wan_vae_grid_sequence | Motion-related Objects | 0.3500 | 42/120 |
| none | wan_vae_grid_sequence | Repetition Count | 0.5417 | 65/120 |
| none | pixel_grid_sequence | Action Order | 0.3167 | 38/120 |
| none | pixel_grid_sequence | Motion Recognition | 0.4917 | 59/120 |
| none | pixel_grid_sequence | Motion-related Objects | 0.3250 | 39/120 |
| none | pixel_grid_sequence | Repetition Count | 0.4083 | 49/120 |
| none | flow_grid_sequence | Action Order | 0.4417 | 53/120 |
| none | flow_grid_sequence | Motion Recognition | 0.5583 | 67/120 |
| none | flow_grid_sequence | Motion-related Objects | 0.2500 | 30/120 |
| none | flow_grid_sequence | Repetition Count | 0.4667 | 56/120 |
| uniform5 | wan_vae_grid_sequence | Action Order | 0.4417 | 53/120 |
| uniform5 | wan_vae_grid_sequence | Motion Recognition | 0.5000 | 60/120 |
| uniform5 | wan_vae_grid_sequence | Motion-related Objects | 0.3583 | 43/120 |
| uniform5 | wan_vae_grid_sequence | Repetition Count | 0.5250 | 63/120 |
| uniform5 | pixel_grid_sequence | Action Order | 0.4500 | 54/120 |
| uniform5 | pixel_grid_sequence | Motion Recognition | 0.5333 | 64/120 |
| uniform5 | pixel_grid_sequence | Motion-related Objects | 0.4500 | 54/120 |
| uniform5 | pixel_grid_sequence | Repetition Count | 0.4583 | 55/120 |
| uniform5 | flow_grid_sequence | Action Order | 0.4583 | 55/120 |
| uniform5 | flow_grid_sequence | Motion Recognition | 0.5667 | 68/120 |
| uniform5 | flow_grid_sequence | Motion-related Objects | 0.3167 | 38/120 |
| uniform5 | flow_grid_sequence | Repetition Count | 0.5083 | 61/120 |
| nonuniform5 | wan_vae_grid_sequence | Action Order | 0.4667 | 56/120 |
| nonuniform5 | wan_vae_grid_sequence | Motion Recognition | 0.5000 | 60/120 |
| nonuniform5 | wan_vae_grid_sequence | Motion-related Objects | 0.3583 | 43/120 |
| nonuniform5 | wan_vae_grid_sequence | Repetition Count | 0.4583 | 55/120 |
| nonuniform5 | pixel_grid_sequence | Action Order | 0.3917 | 47/120 |
| nonuniform5 | pixel_grid_sequence | Motion Recognition | 0.5500 | 66/120 |
| nonuniform5 | pixel_grid_sequence | Motion-related Objects | 0.3917 | 47/120 |
| nonuniform5 | pixel_grid_sequence | Repetition Count | 0.4333 | 52/120 |
| nonuniform5 | flow_grid_sequence | Action Order | 0.4417 | 53/120 |
| nonuniform5 | flow_grid_sequence | Motion Recognition | 0.5500 | 66/120 |
| nonuniform5 | flow_grid_sequence | Motion-related Objects | 0.3333 | 40/120 |
| nonuniform5 | flow_grid_sequence | Repetition Count | 0.4417 | 53/120 |
