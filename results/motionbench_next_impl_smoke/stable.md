# MotionBench Stable Probe

| mode | feature | transform | acc mean | std | CI low | CI high | correct/total |
|---|---|---|---:|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | full | 0.1979 | 0.0312 | 0.1146 | 0.2604 | 19/96 |
| none | pixel_grid_sequence | full | 0.1771 | 0.0521 | 0.1042 | 0.2500 | 17/96 |
| uniform5 | wan_vae_grid_sequence | full | 0.2292 | 0.0208 | 0.1349 | 0.2917 | 22/96 |
| uniform5 | pixel_grid_sequence | full | 0.1979 | 0.0104 | 0.1250 | 0.2812 | 19/96 |

## Paired Comparisons

| mode | feature | baseline | diff | CI low | CI high | McNemar p |
|---|---|---|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | pixel_grid_sequence | 0.0208 | -0.0839 | 0.0938 | 0.8238 |
| uniform5 | wan_vae_grid_sequence | pixel_grid_sequence | 0.0312 | -0.0576 | 0.1042 | 0.6476 |

## Low-FPS Consistency

| feature | lowfps mode | consistency | normal only | low only | both correct | both wrong |
|---|---|---:|---:|---:|---:|---:|
| wan_vae_grid_sequence | uniform5 | 0.6042 | 8 | 11 | 11 | 66 |
| pixel_grid_sequence | uniform5 | 0.6354 | 7 | 9 | 10 | 70 |
