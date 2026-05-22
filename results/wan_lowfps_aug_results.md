# Wan low-fps augmentation experiment results

| task | feature | full->full | full->low | low->low | full+low->low | aug gain |
|---|---:|---:|---:|---:|---:|---:|
| direction4 | dit_l14_t900_token_mean | 0.7083 | 0.2865 | 0.4948 | 0.5781 | 0.2917 |
| direction4 | wan_vae_global_delta | 0.7188 | 0.2708 | 0.4740 | 0.4271 | 0.1562 |
| direction4 | pixel_grid_delta | 0.9844 | 0.9583 | 0.9948 | 0.9948 | 0.0365 |
| direction4 | wan_vae_global_sequence | 0.9010 | 0.2500 | 0.2812 | 0.2812 | 0.0312 |
| direction4 | pixel_grid_sequence | 0.9740 | 0.9740 | 0.9635 | 0.9740 | 0.0000 |
| direction4 | wan_vae_grid_sequence | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| before_after_cycle | wan_vae_global_delta | 0.9896 | 0.5312 | 0.9375 | 0.9792 | 0.4479 |
| before_after_cycle | wan_vae_global_sequence | 0.9583 | 0.5729 | 0.8229 | 0.8958 | 0.3229 |
| before_after_cycle | pixel_grid_delta | 1.0000 | 0.6979 | 1.0000 | 1.0000 | 0.3021 |
| before_after_cycle | dit_l14_t900_token_mean | 0.6458 | 0.5104 | 0.6458 | 0.6562 | 0.1458 |
| before_after_cycle | pixel_grid_sequence | 1.0000 | 0.9792 | 1.0000 | 1.0000 | 0.0208 |
| before_after_cycle | wan_vae_grid_sequence | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
