# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | tfidf | logistic | 0.4227 | 0.1058 | 0.3792 | 0.4667 | 203/480 | 2.5333 |
| none | wan_vae_global_sequence | tfidf | logistic | 0.4168 | 0.1070 | 0.3729 | 0.4604 | 200/480 | 2.5019 |
| none | wan_vae_global_delta | tfidf | logistic | 0.4228 | 0.1090 | 0.3792 | 0.4667 | 203/480 | 2.3620 |
| none | pixel_grid_sequence | tfidf | logistic | 0.3599 | 0.0906 | 0.3208 | 0.4021 | 173/480 | 2.3036 |
| none | flow_grid_sequence | tfidf | logistic | 0.4288 | 0.0930 | 0.3854 | 0.4750 | 206/480 | 2.5957 |
| uniform5 | wan_vae_grid_sequence | tfidf | logistic | 0.3812 | 0.1032 | 0.3375 | 0.4250 | 183/480 | 2.3298 |
| uniform5 | wan_vae_global_sequence | tfidf | logistic | 0.4166 | 0.0965 | 0.3729 | 0.4625 | 200/480 | 2.4676 |
| uniform5 | wan_vae_global_delta | tfidf | logistic | 0.4226 | 0.1134 | 0.3792 | 0.4667 | 203/480 | 2.4461 |
| uniform5 | pixel_grid_sequence | tfidf | logistic | 0.4644 | 0.0915 | 0.4208 | 0.5104 | 223/480 | 2.4015 |
| uniform5 | flow_grid_sequence | tfidf | logistic | 0.4643 | 0.1030 | 0.4208 | 0.5104 | 223/480 | 2.4543 |
| nonuniform5 | wan_vae_grid_sequence | tfidf | logistic | 0.4269 | 0.0957 | 0.3812 | 0.4729 | 205/480 | 2.4571 |
| nonuniform5 | wan_vae_global_sequence | tfidf | logistic | 0.4123 | 0.1024 | 0.3688 | 0.4562 | 198/480 | 2.4481 |
| nonuniform5 | wan_vae_global_delta | tfidf | logistic | 0.4082 | 0.1127 | 0.3646 | 0.4521 | 196/480 | 2.4400 |
| nonuniform5 | pixel_grid_sequence | tfidf | logistic | 0.3855 | 0.1002 | 0.3438 | 0.4292 | 185/480 | 2.3937 |
| nonuniform5 | flow_grid_sequence | tfidf | logistic | 0.4120 | 0.0909 | 0.3688 | 0.4583 | 198/480 | 2.4768 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | wan_vae_grid_sequence | Action Order | 0.4083 | 49/120 |
| none | wan_vae_grid_sequence | Motion Recognition | 0.4917 | 59/120 |
| none | wan_vae_grid_sequence | Motion-related Objects | 0.3417 | 41/120 |
| none | wan_vae_grid_sequence | Repetition Count | 0.4500 | 54/120 |
| none | wan_vae_global_sequence | Action Order | 0.3667 | 44/120 |
| none | wan_vae_global_sequence | Motion Recognition | 0.4417 | 53/120 |
| none | wan_vae_global_sequence | Motion-related Objects | 0.4417 | 53/120 |
| none | wan_vae_global_sequence | Repetition Count | 0.4167 | 50/120 |
| none | wan_vae_global_delta | Action Order | 0.3917 | 47/120 |
| none | wan_vae_global_delta | Motion Recognition | 0.4667 | 56/120 |
| none | wan_vae_global_delta | Motion-related Objects | 0.4000 | 48/120 |
| none | wan_vae_global_delta | Repetition Count | 0.4333 | 52/120 |
| none | pixel_grid_sequence | Action Order | 0.2500 | 30/120 |
| none | pixel_grid_sequence | Motion Recognition | 0.4167 | 50/120 |
| none | pixel_grid_sequence | Motion-related Objects | 0.3833 | 46/120 |
| none | pixel_grid_sequence | Repetition Count | 0.3917 | 47/120 |
| none | flow_grid_sequence | Action Order | 0.3750 | 45/120 |
| none | flow_grid_sequence | Motion Recognition | 0.4917 | 59/120 |
| none | flow_grid_sequence | Motion-related Objects | 0.4000 | 48/120 |
| none | flow_grid_sequence | Repetition Count | 0.4500 | 54/120 |
| uniform5 | wan_vae_grid_sequence | Action Order | 0.3000 | 36/120 |
| uniform5 | wan_vae_grid_sequence | Motion Recognition | 0.4000 | 48/120 |
| uniform5 | wan_vae_grid_sequence | Motion-related Objects | 0.4333 | 52/120 |
| uniform5 | wan_vae_grid_sequence | Repetition Count | 0.3917 | 47/120 |
| uniform5 | wan_vae_global_sequence | Action Order | 0.3750 | 45/120 |
| uniform5 | wan_vae_global_sequence | Motion Recognition | 0.4417 | 53/120 |
| uniform5 | wan_vae_global_sequence | Motion-related Objects | 0.4333 | 52/120 |
| uniform5 | wan_vae_global_sequence | Repetition Count | 0.4167 | 50/120 |
| uniform5 | wan_vae_global_delta | Action Order | 0.3750 | 45/120 |
| uniform5 | wan_vae_global_delta | Motion Recognition | 0.4583 | 55/120 |
| uniform5 | wan_vae_global_delta | Motion-related Objects | 0.4167 | 50/120 |
| uniform5 | wan_vae_global_delta | Repetition Count | 0.4417 | 53/120 |
| uniform5 | pixel_grid_sequence | Action Order | 0.4583 | 55/120 |
| uniform5 | pixel_grid_sequence | Motion Recognition | 0.4917 | 59/120 |
| uniform5 | pixel_grid_sequence | Motion-related Objects | 0.4750 | 57/120 |
| uniform5 | pixel_grid_sequence | Repetition Count | 0.4333 | 52/120 |
| uniform5 | flow_grid_sequence | Action Order | 0.4917 | 59/120 |
| uniform5 | flow_grid_sequence | Motion Recognition | 0.4917 | 59/120 |
| uniform5 | flow_grid_sequence | Motion-related Objects | 0.4000 | 48/120 |
| uniform5 | flow_grid_sequence | Repetition Count | 0.4750 | 57/120 |
| nonuniform5 | wan_vae_grid_sequence | Action Order | 0.3750 | 45/120 |
| nonuniform5 | wan_vae_grid_sequence | Motion Recognition | 0.4417 | 53/120 |
| nonuniform5 | wan_vae_grid_sequence | Motion-related Objects | 0.4667 | 56/120 |
| nonuniform5 | wan_vae_grid_sequence | Repetition Count | 0.4250 | 51/120 |
| nonuniform5 | wan_vae_global_sequence | Action Order | 0.3583 | 43/120 |
| nonuniform5 | wan_vae_global_sequence | Motion Recognition | 0.4583 | 55/120 |
| nonuniform5 | wan_vae_global_sequence | Motion-related Objects | 0.4250 | 51/120 |
| nonuniform5 | wan_vae_global_sequence | Repetition Count | 0.4083 | 49/120 |
| nonuniform5 | wan_vae_global_delta | Action Order | 0.3583 | 43/120 |
| nonuniform5 | wan_vae_global_delta | Motion Recognition | 0.4583 | 55/120 |
| nonuniform5 | wan_vae_global_delta | Motion-related Objects | 0.3917 | 47/120 |
| nonuniform5 | wan_vae_global_delta | Repetition Count | 0.4250 | 51/120 |
| nonuniform5 | pixel_grid_sequence | Action Order | 0.2833 | 34/120 |
| nonuniform5 | pixel_grid_sequence | Motion Recognition | 0.4750 | 57/120 |
| nonuniform5 | pixel_grid_sequence | Motion-related Objects | 0.3667 | 44/120 |
| nonuniform5 | pixel_grid_sequence | Repetition Count | 0.4167 | 50/120 |
| nonuniform5 | flow_grid_sequence | Action Order | 0.2500 | 30/120 |
| nonuniform5 | flow_grid_sequence | Motion Recognition | 0.5500 | 66/120 |
| nonuniform5 | flow_grid_sequence | Motion-related Objects | 0.3750 | 45/120 |
| nonuniform5 | flow_grid_sequence | Repetition Count | 0.4750 | 57/120 |
