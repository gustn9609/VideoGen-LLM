# VideoChat2-HD Wan-REPA Fine-tuning

| Fold | Stage | Split | Acc | Correct/total |
|---|---|---|---:|---:|
| 0 | initial | test | 0.4359 | 17/39 |
| 0 | epoch_1 | test | 0.4872 | 19/39 |
| 0 | final | train | 0.5294 | 81/153 |
| 0 | final | test | 0.4872 | 19/39 |
| 1 | initial | test | 0.4103 | 16/39 |
| 1 | epoch_1 | test | 0.4615 | 18/39 |
| 1 | final | train | 0.5752 | 88/153 |
| 1 | final | test | 0.4615 | 18/39 |
| 2 | initial | test | 0.4737 | 18/38 |
| 2 | epoch_1 | test | 0.3684 | 14/38 |
| 2 | final | train | 0.5909 | 91/154 |
| 2 | final | test | 0.3684 | 14/38 |
| 3 | initial | test | 0.3684 | 14/38 |
| 3 | epoch_1 | test | 0.4474 | 17/38 |
| 3 | final | train | 0.5195 | 80/154 |
| 3 | final | test | 0.4474 | 17/38 |
| 4 | initial | test | 0.4211 | 16/38 |
| 4 | epoch_1 | test | 0.4474 | 17/38 |
| 4 | final | train | 0.5130 | 79/154 |
| 4 | final | test | 0.4474 | 17/38 |
| all | initial | test | 0.4219 | 81/192 |
| all | epoch_1 | test | 0.4427 | 85/192 |
| all | final | test | 0.4427 | 85/192 |

## Training

| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss | Eff REPA λ | Eff Rel λ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.3425 | 1.0731 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 1 | 1.2579 | 1.0065 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 1 | 1.2890 | 0.9459 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 1 | 1.2209 | 0.9178 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 1 | 1.2109 | 0.8146 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
