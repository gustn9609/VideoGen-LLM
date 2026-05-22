# VideoChat2-HD Wan-REPA Fine-tuning

| Fold | Stage | Split | Acc | Correct/total |
|---|---|---|---:|---:|
| 0 | initial | test | 0.3500 | 7/20 |
| 0 | epoch_1 | test | 0.5000 | 10/20 |
| 0 | final | train | 0.5658 | 43/76 |
| 0 | final | test | 0.5000 | 10/20 |
| 1 | initial | test | 0.5789 | 11/19 |
| 1 | epoch_1 | test | 0.4737 | 9/19 |
| 1 | final | train | 0.5974 | 46/77 |
| 1 | final | test | 0.4737 | 9/19 |
| 2 | initial | test | 0.3684 | 7/19 |
| 2 | epoch_1 | test | 0.3684 | 7/19 |
| 2 | final | train | 0.5844 | 45/77 |
| 2 | final | test | 0.3684 | 7/19 |
| 3 | initial | test | 0.3684 | 7/19 |
| 3 | epoch_1 | test | 0.4737 | 9/19 |
| 3 | final | train | 0.4286 | 33/77 |
| 3 | final | test | 0.4737 | 9/19 |
| 4 | initial | test | 0.2632 | 5/19 |
| 4 | epoch_1 | test | 0.5263 | 10/19 |
| 4 | final | train | 0.6234 | 48/77 |
| 4 | final | test | 0.5263 | 10/19 |
| all | initial | test | 0.3854 | 37/96 |
| all | epoch_1 | test | 0.4688 | 45/96 |
| all | final | test | 0.4688 | 45/96 |

## Training

| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss | Eff REPA λ | Eff Rel λ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.7219 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| 1 | 1 | 1.6880 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| 2 | 1 | 1.8313 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| 3 | 1 | 2.0631 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| 4 | 1 | 1.5863 | 1.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
