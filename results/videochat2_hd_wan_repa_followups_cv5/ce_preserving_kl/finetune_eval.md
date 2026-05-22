# VideoChat2-HD Wan-REPA Fine-tuning

| Fold | Stage | Split | Acc | Correct/total |
|---|---|---|---:|---:|
| 0 | initial | test | 0.3500 | 7/20 |
| 0 | epoch_1 | test | 0.5000 | 10/20 |
| 0 | final | train | 0.5789 | 44/76 |
| 0 | final | test | 0.5000 | 10/20 |
| 1 | initial | test | 0.5789 | 11/19 |
| 1 | epoch_1 | test | 0.4737 | 9/19 |
| 1 | final | train | 0.5974 | 46/77 |
| 1 | final | test | 0.4737 | 9/19 |
| 2 | initial | test | 0.3684 | 7/19 |
| 2 | epoch_1 | test | 0.3684 | 7/19 |
| 2 | final | train | 0.5714 | 44/77 |
| 2 | final | test | 0.3684 | 7/19 |
| 3 | initial | test | 0.3684 | 7/19 |
| 3 | epoch_1 | test | 0.5263 | 10/19 |
| 3 | final | train | 0.4156 | 32/77 |
| 3 | final | test | 0.5263 | 10/19 |
| 4 | initial | test | 0.2632 | 5/19 |
| 4 | epoch_1 | test | 0.4737 | 9/19 |
| 4 | final | train | 0.5974 | 46/77 |
| 4 | final | test | 0.4737 | 9/19 |
| all | initial | test | 0.3854 | 37/96 |
| all | epoch_1 | test | 0.4688 | 45/96 |
| all | final | test | 0.4688 | 45/96 |

## Training

| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 1.7564 | 0.2047 | 0.0000 | 1.1429 |
| 1 | 1 | 1.6295 | 0.2200 | 0.0000 | 0.9091 |
| 2 | 1 | 1.8056 | 0.2032 | 0.0000 | 1.1794 |
| 3 | 1 | 1.9776 | 0.2046 | 0.0000 | 1.2381 |
| 4 | 1 | 1.5294 | 0.1996 | 0.0000 | 0.8508 |
