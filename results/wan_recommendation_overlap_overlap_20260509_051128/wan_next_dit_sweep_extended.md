# Wan next experiment results

## VAE/pixel pooling ablation

| task | feature | balanced_accuracy_mean | std | n |
|---|---:|---:|---:|---:|
| direction4 | wan_vae_grid_sequence | 1.0000 | 0.0000 | 3 |
| direction4 | pixel_grid_sequence | 0.9792 | 0.0074 | 3 |
| direction4 | pixel_grid_delta | 0.9635 | 0.0195 | 3 |
| direction4 | wan_vae_global_sequence | 0.9219 | 0.0128 | 3 |
| direction4 | wan_vae_grid_time_avg | 0.8854 | 0.0266 | 3 |
| direction4 | wan_vae_global_delta | 0.7917 | 0.0321 | 3 |
| direction4 | wan_vae_channel_avg | 0.7188 | 0.0255 | 3 |
| direction4 | pixel_grid_time_avg | 0.3854 | 0.0448 | 3 |
| direction4 | pixel_rgb_clip_like_avg | 0.2240 | 0.0575 | 3 |
| reversal | pixel_grid_sequence | 1.0000 | 0.0000 | 3 |
| reversal | pixel_grid_delta | 1.0000 | 0.0000 | 3 |
| reversal | wan_vae_global_sequence | 1.0000 | 0.0000 | 3 |
| reversal | wan_vae_grid_sequence | 1.0000 | 0.0000 | 3 |
| reversal | wan_vae_global_delta | 1.0000 | 0.0000 | 3 |
| reversal | wan_vae_grid_time_avg | 0.9688 | 0.0255 | 3 |
| reversal | wan_vae_channel_avg | 0.8854 | 0.0147 | 3 |
| reversal | pixel_rgb_clip_like_avg | 0.5000 | 0.0000 | 3 |
| reversal | pixel_grid_time_avg | 0.5000 | 0.0000 | 3 |
| shuffle | wan_vae_channel_avg | 0.9896 | 0.0147 | 3 |
| shuffle | wan_vae_global_sequence | 0.9896 | 0.0147 | 3 |
| shuffle | wan_vae_grid_sequence | 0.9896 | 0.0147 | 3 |
| shuffle | wan_vae_global_delta | 0.9896 | 0.0147 | 3 |
| shuffle | wan_vae_grid_time_avg | 0.9792 | 0.0147 | 3 |
| shuffle | pixel_grid_sequence | 0.7500 | 0.0920 | 3 |
| shuffle | pixel_grid_delta | 0.5938 | 0.0675 | 3 |
| shuffle | pixel_rgb_clip_like_avg | 0.5000 | 0.0000 | 3 |
| shuffle | pixel_grid_time_avg | 0.5000 | 0.0000 | 3 |

## DiT hidden-state sweep: top features

| task | feature | layer | timestep | pooling | balanced_accuracy | dim |
|---|---:|---:|---:|---:|---:|---:|
| direction4 | dit_l10_t900_temporal_sequence | 10 | 900 | temporal_sequence | 0.8906 | 7680 |
| direction4 | dit_l10_t950_token_mean | 10 | 950 | token_mean | 0.8594 | 1536 |
| direction4 | dit_l10_t950_temporal_sequence | 10 | 950 | temporal_sequence | 0.8594 | 7680 |
| direction4 | dit_l12_t900_temporal_sequence | 12 | 900 | temporal_sequence | 0.7969 | 7680 |
| direction4 | dit_l16_t950_token_mean | 16 | 950 | token_mean | 0.7969 | 1536 |
| direction4 | dit_l12_t950_token_mean | 12 | 950 | token_mean | 0.7812 | 1536 |
| direction4 | dit_l14_t950_token_mean | 14 | 950 | token_mean | 0.7812 | 1536 |
| direction4 | dit_l16_t950_temporal_sequence | 16 | 950 | temporal_sequence | 0.7812 | 7680 |
| direction4 | dit_l18_t950_token_mean | 18 | 950 | token_mean | 0.7812 | 1536 |
| direction4 | dit_l10_t900_token_mean | 10 | 900 | token_mean | 0.7656 | 1536 |
| shuffle | dit_l10_t900_temporal_sequence | 10 | 900 | temporal_sequence | 1.0000 | 7680 |
| shuffle | dit_l14_t900_token_mean | 14 | 900 | token_mean | 1.0000 | 1536 |
| shuffle | dit_l12_t900_temporal_sequence | 12 | 900 | temporal_sequence | 0.9688 | 7680 |
| shuffle | dit_l14_t900_temporal_sequence | 14 | 900 | temporal_sequence | 0.9688 | 7680 |
| shuffle | dit_l18_t900_temporal_sequence | 18 | 900 | temporal_sequence | 0.9688 | 7680 |
| shuffle | dit_l10_t950_temporal_delta | 10 | 950 | temporal_delta | 0.9688 | 6144 |
| shuffle | dit_l14_t950_token_mean | 14 | 950 | token_mean | 0.9688 | 1536 |
| shuffle | dit_l14_t950_temporal_sequence | 14 | 950 | temporal_sequence | 0.9688 | 7680 |
| shuffle | dit_l14_t950_temporal_delta | 14 | 950 | temporal_delta | 0.9688 | 6144 |
| shuffle | dit_l16_t950_token_mean | 16 | 950 | token_mean | 0.9688 | 1536 |
