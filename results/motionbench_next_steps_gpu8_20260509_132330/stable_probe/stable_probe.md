# MotionBench Stable Probe

| mode | feature | transform | acc mean | std | CI low | CI high | correct/total |
|---|---|---|---:|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | full | 0.2558 | 0.0927 | 0.2167 | 0.2958 | 123/480 |
| none | wan_vae_global_sequence | full | 0.2812 | 0.0793 | 0.2417 | 0.3208 | 135/480 |
| none | wan_vae_global_delta | full | 0.2379 | 0.1144 | 0.2000 | 0.2751 | 114/480 |
| none | pixel_grid_sequence | full | 0.2567 | 0.1040 | 0.2167 | 0.2958 | 123/480 |
| none | flow_grid_sequence | full | 0.2645 | 0.0588 | 0.2271 | 0.3042 | 127/480 |
| none | metadata_only | full | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 480/480 |
| uniform5 | wan_vae_grid_sequence | full | 0.2563 | 0.1065 | 0.2167 | 0.2938 | 123/480 |
| uniform5 | wan_vae_global_sequence | full | 0.2704 | 0.0914 | 0.2333 | 0.3104 | 130/480 |
| uniform5 | wan_vae_global_delta | full | 0.2213 | 0.0710 | 0.1833 | 0.2562 | 106/480 |
| uniform5 | pixel_grid_sequence | full | 0.2560 | 0.0874 | 0.2188 | 0.2958 | 123/480 |
| uniform5 | flow_grid_sequence | full | 0.3271 | 0.0627 | 0.2854 | 0.3688 | 157/480 |
| uniform5 | metadata_only | full | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 480/480 |
| nonuniform5 | wan_vae_grid_sequence | full | 0.2602 | 0.1072 | 0.2208 | 0.3000 | 125/480 |
| nonuniform5 | wan_vae_global_sequence | full | 0.2835 | 0.0648 | 0.2437 | 0.3209 | 136/480 |
| nonuniform5 | wan_vae_global_delta | full | 0.2542 | 0.0840 | 0.2167 | 0.2917 | 122/480 |
| nonuniform5 | pixel_grid_sequence | full | 0.2353 | 0.0807 | 0.2000 | 0.2730 | 113/480 |
| nonuniform5 | flow_grid_sequence | full | 0.2736 | 0.0728 | 0.2313 | 0.3104 | 131/480 |
| nonuniform5 | metadata_only | full | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 480/480 |

## Paired Comparisons

| mode | feature | baseline | diff | CI low | CI high | McNemar p |
|---|---|---|---:|---:|---:|---:|
| none | wan_vae_grid_sequence | pixel_grid_sequence | 0.0000 | -0.0417 | 0.0396 | 1.0000 |
| none | wan_vae_global_sequence | pixel_grid_sequence | 0.0250 | -0.0292 | 0.0771 | 0.3875 |
| none | wan_vae_global_delta | pixel_grid_sequence | -0.0187 | -0.0667 | 0.0312 | 0.5124 |
| none | flow_grid_sequence | pixel_grid_sequence | 0.0083 | -0.0458 | 0.0583 | 0.8181 |
| none | metadata_only | pixel_grid_sequence | 0.7438 | 0.7042 | 0.7833 | 0.0000 |
| uniform5 | wan_vae_grid_sequence | pixel_grid_sequence | 0.0000 | -0.0417 | 0.0396 | 1.0000 |
| uniform5 | wan_vae_global_sequence | pixel_grid_sequence | 0.0146 | -0.0375 | 0.0625 | 0.6184 |
| uniform5 | wan_vae_global_delta | pixel_grid_sequence | -0.0354 | -0.0854 | 0.0125 | 0.1868 |
| uniform5 | flow_grid_sequence | pixel_grid_sequence | 0.0708 | 0.0166 | 0.1271 | 0.0153 |
| uniform5 | metadata_only | pixel_grid_sequence | 0.7438 | 0.7042 | 0.7812 | 0.0000 |
| nonuniform5 | wan_vae_grid_sequence | pixel_grid_sequence | 0.0250 | -0.0146 | 0.0646 | 0.2664 |
| nonuniform5 | wan_vae_global_sequence | pixel_grid_sequence | 0.0479 | -0.0021 | 0.0958 | 0.0711 |
| nonuniform5 | wan_vae_global_delta | pixel_grid_sequence | 0.0187 | -0.0333 | 0.0688 | 0.5259 |
| nonuniform5 | flow_grid_sequence | pixel_grid_sequence | 0.0375 | -0.0146 | 0.0875 | 0.1705 |
| nonuniform5 | metadata_only | pixel_grid_sequence | 0.7646 | 0.7270 | 0.8000 | 0.0000 |

## Low-FPS Consistency

| feature | lowfps mode | consistency | normal only | low only | both correct | both wrong |
|---|---|---:|---:|---:|---:|---:|
| wan_vae_grid_sequence | uniform5 | 0.8167 | 20 | 20 | 103 | 337 |
| wan_vae_grid_sequence | nonuniform5 | 0.7750 | 21 | 23 | 102 | 334 |
| wan_vae_global_sequence | uniform5 | 0.6292 | 53 | 48 | 82 | 297 |
| wan_vae_global_sequence | nonuniform5 | 0.5938 | 53 | 54 | 82 | 291 |
| wan_vae_global_delta | uniform5 | 0.5000 | 61 | 53 | 53 | 313 |
| wan_vae_global_delta | nonuniform5 | 0.4000 | 69 | 77 | 45 | 289 |
| pixel_grid_sequence | uniform5 | 0.8187 | 20 | 20 | 103 | 337 |
| pixel_grid_sequence | nonuniform5 | 0.7688 | 35 | 25 | 88 | 332 |
| flow_grid_sequence | uniform5 | 0.7021 | 21 | 51 | 106 | 302 |
| flow_grid_sequence | nonuniform5 | 0.5875 | 47 | 51 | 80 | 302 |
| metadata_only | uniform5 | 1.0000 | 0 | 0 | 480 | 0 |
| metadata_only | nonuniform5 | 1.0000 | 0 | 0 | 480 | 0 |
