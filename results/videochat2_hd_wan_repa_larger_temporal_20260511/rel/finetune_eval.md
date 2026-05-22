# VideoChat2-HD Wan-REPA Fine-tuning

| Fold | Stage | Split | Acc | Correct/total |
|---|---|---|---:|---:|
| 0 | initial | test | 0.4359 | 17/39 |
| 0 | epoch_1 | test | 0.4103 | 16/39 |
| 0 | final | train | 0.5556 | 85/153 |
| 0 | final | test | 0.4103 | 16/39 |
| 1 | initial | test | 0.4103 | 16/39 |
| 1 | epoch_1 | test | 0.3846 | 15/39 |
| 1 | final | train | 0.5817 | 89/153 |
| 1 | final | test | 0.3846 | 15/39 |
| 2 | initial | test | 0.4737 | 18/38 |
| 2 | epoch_1 | test | 0.3947 | 15/38 |
| 2 | final | train | 0.6104 | 94/154 |
| 2 | final | test | 0.3947 | 15/38 |
| 3 | initial | test | 0.3684 | 14/38 |
| 3 | epoch_1 | test | 0.4211 | 16/38 |
| 3 | final | train | 0.5195 | 80/154 |
| 3 | final | test | 0.4211 | 16/38 |
| 4 | initial | test | 0.4211 | 16/38 |
| 4 | epoch_1 | test | 0.5000 | 19/38 |
| 4 | final | train | 0.5584 | 86/154 |
| 4 | final | test | 0.5000 | 19/38 |
| all | initial | test | 0.4219 | 81/192 |
| all | epoch_1 | test | 0.4219 | 81/192 |
| all | final | test | 0.4219 | 81/192 |

## Training

| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss | Eff REPA λ | Eff Rel λ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1.3647 | 1.0781 | 0.4388 | 0.0000 | 0.0000 | 0.1000 |
| 1 | 1 | 1.3070 | 0.9958 | 0.4402 | 0.0000 | 0.0000 | 0.1000 |
| 2 | 1 | 1.2873 | 0.9424 | 0.4386 | 0.0000 | 0.0000 | 0.1000 |
| 3 | 1 | 1.2213 | 0.9163 | 0.4385 | 0.0000 | 0.0000 | 0.1000 |
| 4 | 1 | 1.1686 | 0.8206 | 0.4383 | 0.0000 | 0.0000 | 0.1000 |
