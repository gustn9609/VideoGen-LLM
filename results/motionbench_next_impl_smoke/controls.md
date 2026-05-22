# MotionBench Temporal Controls

| feature | condition | acc mean | std | CI low | CI high | correct/total | normal gap |
|---|---|---:|---:|---:|---:|---:|---:|
| wan_vae_grid_sequence | normal | 0.2083 | 0.0000 | 0.1250 | 0.2971 | 20/96 | 0.0000 |
| wan_vae_grid_sequence | first_frame_only | 0.2396 | 0.0104 | 0.1612 | 0.3180 | 23/96 | -0.0312 |
| wan_vae_grid_sequence | time_average | 0.2188 | 0.0312 | 0.1562 | 0.2812 | 21/96 | -0.0104 |
| wan_vae_grid_sequence | shuffled | 0.1979 | 0.0521 | 0.1250 | 0.2714 | 19/96 | 0.0104 |
| wan_vae_grid_sequence | reversed | 0.2083 | 0.0625 | 0.1245 | 0.2763 | 20/96 | 0.0000 |
| wan_vae_grid_sequence | uniform5 | 0.2083 | 0.0208 | 0.1458 | 0.2818 | 20/96 | 0.0000 |
| wan_vae_grid_sequence | metadata_only | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 96/96 | -0.7917 |
