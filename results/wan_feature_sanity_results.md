# Wan Feature Sanity Results

## VAE linear probes

| task | feature | accuracy | balanced_accuracy | dim |
|---|---:|---:|---:|---:|
| direction4 | frame_rgb_mean | 0.1979 | 0.1979 | 51 |
| direction4 | pixel_grid | 0.9479 | 0.9479 | 13056 |
| direction4 | pixel_grid_delta | 0.9792 | 0.9792 | 12288 |
| direction4 | wan_vae_global | 0.9688 | 0.9688 | 80 |
| direction4 | wan_vae_grid | 1.0000 | 1.0000 | 20480 |
| direction4 | wan_vae_delta_grid | 1.0000 | 1.0000 | 36864 |
| reversal | frame_rgb_mean | 1.0000 | 1.0000 | 51 |
| reversal | pixel_grid | 1.0000 | 1.0000 | 13056 |
| reversal | pixel_grid_delta | 1.0000 | 1.0000 | 12288 |
| reversal | wan_vae_global | 1.0000 | 1.0000 | 80 |
| reversal | wan_vae_grid | 1.0000 | 1.0000 | 20480 |
| reversal | wan_vae_delta_grid | 1.0000 | 1.0000 | 36864 |
| shuffle | frame_rgb_mean | 0.3750 | 0.3750 | 51 |
| shuffle | pixel_grid | 0.7917 | 0.7917 | 13056 |
| shuffle | pixel_grid_delta | 0.6042 | 0.6042 | 12288 |
| shuffle | wan_vae_global | 1.0000 | 1.0000 | 80 |
| shuffle | wan_vae_grid | 1.0000 | 1.0000 | 20480 |
| shuffle | wan_vae_delta_grid | 0.9792 | 0.9792 | 36864 |

## DiT denoising loss

| timestep | normal | reversed | shuffled | normal<reversed | normal<shuffled |
|---:|---:|---:|---:|---:|---:|
| 900 | 1.625805 | 1.632518 | 1.665083 | 0.458 | 0.792 |
| 500 | 4.732586 | 4.719334 | 4.635388 | 0.375 | 0.042 |
| 100 | 5.738333 | 5.752999 | 5.765300 | 0.542 | 0.750 |
