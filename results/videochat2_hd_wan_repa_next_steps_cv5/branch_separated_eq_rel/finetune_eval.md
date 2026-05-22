# VideoChat2-HD Wan-REPA Fine-tuning

| Fold | Stage | Split | Acc | Correct/total |
|---|---|---|---:|---:|
| 0 | initial | test | 0.3000 | 6/20 |
| 0 | epoch_1 | test | 0.3500 | 7/20 |
| 0 | final | train | 0.5921 | 45/76 |
| 0 | final | test | 0.3500 | 7/20 |
| 1 | initial | test | 0.6316 | 12/19 |
| 1 | epoch_1 | test | 0.4211 | 8/19 |
| 1 | final | train | 0.6234 | 48/77 |
| 1 | final | test | 0.4211 | 8/19 |
| 2 | initial | test | 0.3684 | 7/19 |
| 2 | epoch_1 | test | 0.4737 | 9/19 |
| 2 | final | train | 0.6364 | 49/77 |
| 2 | final | test | 0.4737 | 9/19 |
| 3 | initial | test | 0.4211 | 8/19 |
| 3 | epoch_1 | test | 0.1053 | 2/19 |
| 3 | final | train | 0.3636 | 28/77 |
| 3 | final | test | 0.1053 | 2/19 |
| 4 | initial | test | 0.2632 | 5/19 |
| 4 | epoch_1 | test | 0.4211 | 8/19 |
| 4 | final | train | 0.4935 | 38/77 |
| 4 | final | test | 0.4211 | 8/19 |
| all | initial | test | 0.3958 | 38/96 |
| all | epoch_1 | test | 0.3542 | 34/96 |
| all | final | test | 0.3542 | 34/96 |

## Training

| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss | Eff REPA λ | Eff Rel λ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.3868 | 0.2081 | 0.4391 | 0.0000 | 0.1000 | 0.1000 |
| 1 | 1 | 1.3393 | 0.2197 | 0.4378 | 0.0000 | 0.1000 | 0.1000 |
| 2 | 1 | 1.5813 | 0.2145 | 0.4374 | 0.0000 | 0.1000 | 0.1000 |
| 3 | 1 | 1.7037 | 0.2057 | 0.4387 | 0.0000 | 0.1000 | 0.1000 |
| 4 | 1 | 1.2956 | 0.2134 | 0.4345 | 0.0000 | 0.1000 | 0.1000 |
