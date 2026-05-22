# VideoChat2-HD Wan-REPA Fine-tuning

| Fold | Stage | Split | Acc | Correct/total |
|---|---|---|---:|---:|
| 0 | initial | test | 0.3333 | 2/6 |
| 0 | epoch_1 | test | 0.3333 | 2/6 |
| 0 | final | train | 0.6667 | 4/6 |
| 0 | final | test | 0.3333 | 2/6 |
| 1 | initial | test | 0.5000 | 3/6 |
| 1 | epoch_1 | test | 0.3333 | 2/6 |
| 1 | final | train | 0.1667 | 1/6 |
| 1 | final | test | 0.3333 | 2/6 |
| all | initial | test | 0.4167 | 5/12 |
| all | epoch_1 | test | 0.3333 | 4/12 |
| all | final | test | 0.3333 | 4/12 |

## Training

| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 6.6525 | 0.5962 | 0.0000 | 0.0000 |
| 1 | 1 | 6.1942 | 0.5666 | 0.0000 | 0.0000 |
