# Wan temporal reasoning experiment results

## Temporal reasoning probes

| task | feature | balanced_accuracy_mean | std | n |
|---|---:|---:|---:|---:|
| before_after | pixel_grid_time_avg | 1.0000 | 0.0000 | 3 |
| before_after | pixel_grid_sequence | 1.0000 | 0.0000 | 3 |
| before_after | pixel_grid_delta | 1.0000 | 0.0000 | 3 |
| before_after | wan_vae_grid_sequence | 1.0000 | 0.0000 | 3 |
| before_after | wan_vae_global_delta | 0.9583 | 0.0147 | 3 |
| before_after | wan_vae_global_sequence | 0.8646 | 0.0531 | 3 |
| before_after | dit_l14_t900_token_mean | 0.7708 | 0.0295 | 3 |
| before_after | wan_vae_channel_avg | 0.5729 | 0.1062 | 3 |
| before_after | pixel_rgb_clip_like_avg | 0.5521 | 0.0642 | 3 |
| before_after_cycle | pixel_grid_sequence | 1.0000 | 0.0000 | 3 |
| before_after_cycle | pixel_grid_delta | 1.0000 | 0.0000 | 3 |
| before_after_cycle | wan_vae_grid_sequence | 1.0000 | 0.0000 | 3 |
| before_after_cycle | wan_vae_global_delta | 0.9896 | 0.0147 | 3 |
| before_after_cycle | wan_vae_global_sequence | 0.9479 | 0.0390 | 3 |
| before_after_cycle | dit_l14_t900_token_mean | 0.6979 | 0.0820 | 3 |
| before_after_cycle | wan_vae_channel_avg | 0.6042 | 0.0896 | 3 |
| before_after_cycle | pixel_rgb_clip_like_avg | 0.5729 | 0.0820 | 3 |
| before_after_cycle | pixel_grid_time_avg | 0.5312 | 0.0255 | 3 |
| interaction | wan_vae_grid_sequence | 1.0000 | 0.0000 | 3 |
| interaction | wan_vae_global_delta | 1.0000 | 0.0000 | 3 |
| interaction | pixel_grid_time_avg | 0.9896 | 0.0147 | 3 |
| interaction | pixel_grid_sequence | 0.9896 | 0.0147 | 3 |
| interaction | wan_vae_global_sequence | 0.9792 | 0.0147 | 3 |
| interaction | pixel_grid_delta | 0.9583 | 0.0390 | 3 |
| interaction | dit_l14_t900_token_mean | 0.9167 | 0.0531 | 3 |
| interaction | wan_vae_channel_avg | 0.5521 | 0.0147 | 3 |
| interaction | pixel_rgb_clip_like_avg | 0.5417 | 0.0295 | 3 |
| repetition4 | pixel_grid_sequence | 1.0000 | 0.0000 | 3 |
| repetition4 | wan_vae_global_sequence | 1.0000 | 0.0000 | 3 |
| repetition4 | wan_vae_grid_sequence | 1.0000 | 0.0000 | 3 |
| repetition4 | wan_vae_global_delta | 1.0000 | 0.0000 | 3 |
| repetition4 | pixel_grid_delta | 0.9896 | 0.0074 | 3 |
| repetition4 | wan_vae_channel_avg | 0.7344 | 0.0442 | 3 |
| repetition4 | dit_l14_t900_token_mean | 0.5781 | 0.0255 | 3 |
| repetition4 | pixel_grid_time_avg | 0.4219 | 0.0675 | 3 |
| repetition4 | pixel_rgb_clip_like_avg | 0.2969 | 0.0383 | 3 |

## Low-frame-rate robustness

| task | feature | full | lowfps | drop | n |
|---|---:|---:|---:|---:|---:|
| direction4 | wan_vae_grid_sequence | 1.0000 | 1.0000 | 0.0000 | 3 |
| direction4 | pixel_grid_sequence | 0.9948 | 0.9896 | 0.0052 | 3 |
| direction4 | pixel_grid_delta | 0.9792 | 0.9792 | 0.0000 | 3 |
| direction4 | pixel_grid_time_avg | 0.3490 | 0.3281 | 0.0208 | 3 |
| direction4 | wan_vae_global_delta | 0.7760 | 0.2760 | 0.5000 | 3 |
| direction4 | wan_vae_global_sequence | 0.9271 | 0.2656 | 0.6615 | 3 |
| direction4 | wan_vae_channel_avg | 0.7656 | 0.2500 | 0.5156 | 3 |
| direction4 | dit_l14_t900_token_mean | 0.7083 | 0.2500 | 0.4583 | 3 |
| direction4 | pixel_rgb_clip_like_avg | 0.2292 | 0.2292 | 0.0000 | 3 |
| before_after_cycle | wan_vae_grid_sequence | 1.0000 | 1.0000 | 0.0000 | 3 |
| before_after_cycle | pixel_grid_sequence | 1.0000 | 0.9583 | 0.0417 | 3 |
| before_after_cycle | pixel_grid_delta | 1.0000 | 0.7396 | 0.2604 | 3 |
| before_after_cycle | wan_vae_global_sequence | 0.9167 | 0.6458 | 0.2708 | 3 |
| before_after_cycle | pixel_rgb_clip_like_avg | 0.5312 | 0.5312 | 0.0000 | 3 |
| before_after_cycle | pixel_grid_time_avg | 0.5625 | 0.5208 | 0.0417 | 3 |
| before_after_cycle | dit_l14_t900_token_mean | 0.6667 | 0.5104 | 0.1562 | 3 |
| before_after_cycle | wan_vae_channel_avg | 0.5000 | 0.5000 | 0.0000 | 3 |
| before_after_cycle | wan_vae_global_delta | 1.0000 | 0.5000 | 0.5000 | 3 |
