# MotionBench Candidate Rerank Probe

| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| none | dit_l14_t900_token_mean | hash | logistic | 0.3782 | 0.1177 | 0.3229 | 0.4340 | 109/288 | 2.8498 |
| none | dit_l14_t900_spatial_tokens | hash | logistic | 0.4446 | 0.0949 | 0.3889 | 0.5000 | 128/288 | 2.3888 |

## Per Question Type

| mode | feature | question_type | acc | correct/total |
|---|---|---|---:|---:|
| none | dit_l14_t900_token_mean | Action Order | 0.2778 | 20/72 |
| none | dit_l14_t900_token_mean | Motion Recognition | 0.4167 | 30/72 |
| none | dit_l14_t900_token_mean | Motion-related Objects | 0.4167 | 30/72 |
| none | dit_l14_t900_token_mean | Repetition Count | 0.4028 | 29/72 |
| none | dit_l14_t900_spatial_tokens | Action Order | 0.3611 | 26/72 |
| none | dit_l14_t900_spatial_tokens | Motion Recognition | 0.5278 | 38/72 |
| none | dit_l14_t900_spatial_tokens | Motion-related Objects | 0.5000 | 36/72 |
| none | dit_l14_t900_spatial_tokens | Repetition Count | 0.3889 | 28/72 |
