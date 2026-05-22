# MotionBench Temporal Controls

| feature | condition | acc mean | std | CI low | CI high | correct/total | normal gap |
|---|---|---:|---:|---:|---:|---:|---:|
| wan_vae_grid_sequence | normal | 0.2647 | 0.1028 | 0.2271 | 0.3042 | 127/480 | 0.0000 |
| wan_vae_grid_sequence | first_frame_only | 0.2707 | 0.0827 | 0.2292 | 0.3104 | 130/480 | -0.0060 |
| wan_vae_grid_sequence | time_average | 0.2546 | 0.0773 | 0.2146 | 0.2938 | 122/480 | 0.0101 |
| wan_vae_grid_sequence | shuffled | 0.2418 | 0.0937 | 0.2062 | 0.2792 | 116/480 | 0.0229 |
| wan_vae_grid_sequence | reversed | 0.2625 | 0.0984 | 0.2229 | 0.3021 | 126/480 | 0.0022 |
| wan_vae_grid_sequence | uniform5 | 0.2755 | 0.0933 | 0.2333 | 0.3167 | 132/480 | -0.0107 |
| wan_vae_grid_sequence | nonuniform5 | 0.2541 | 0.0955 | 0.2146 | 0.2938 | 122/480 | 0.0106 |
| wan_vae_grid_sequence | metadata_only | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 480/480 | -0.7353 |
| pixel_grid_sequence | normal | 0.2562 | 0.0842 | 0.2187 | 0.2958 | 123/480 | 0.0000 |
| pixel_grid_sequence | first_frame_only | 0.2478 | 0.0868 | 0.2083 | 0.2854 | 119/480 | 0.0084 |
| pixel_grid_sequence | time_average | 0.2332 | 0.0834 | 0.1958 | 0.2729 | 112/480 | 0.0231 |
| pixel_grid_sequence | shuffled | 0.2352 | 0.0901 | 0.1979 | 0.2750 | 113/480 | 0.0211 |
| pixel_grid_sequence | reversed | 0.2518 | 0.0806 | 0.2146 | 0.2917 | 121/480 | 0.0044 |
| pixel_grid_sequence | uniform5 | 0.2645 | 0.0962 | 0.2271 | 0.3042 | 127/480 | -0.0083 |
| pixel_grid_sequence | nonuniform5 | 0.2502 | 0.0925 | 0.2125 | 0.2875 | 120/480 | 0.0060 |
| pixel_grid_sequence | metadata_only | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 480/480 | -0.7438 |
| flow_grid_sequence | normal | 0.2604 | 0.0755 | 0.2208 | 0.3000 | 125/480 | 0.0000 |
| flow_grid_sequence | first_frame_only | 0.2184 | 0.0158 | 0.1812 | 0.2583 | 105/480 | 0.0420 |
| flow_grid_sequence | time_average | 0.2540 | 0.0700 | 0.2146 | 0.2938 | 122/480 | 0.0064 |
| flow_grid_sequence | shuffled | 0.2373 | 0.0858 | 0.2000 | 0.2771 | 114/480 | 0.0232 |
| flow_grid_sequence | reversed | 0.2705 | 0.0489 | 0.2313 | 0.3104 | 130/480 | -0.0101 |
| flow_grid_sequence | uniform5 | 0.3312 | 0.0649 | 0.2896 | 0.3750 | 159/480 | -0.0707 |
| flow_grid_sequence | nonuniform5 | 0.2591 | 0.0649 | 0.2208 | 0.2979 | 124/480 | 0.0014 |
| flow_grid_sequence | metadata_only | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 480/480 | -0.7396 |

## Skipped Conditions

- wan_vae_grid_sequence / none+crop_shift: missing mode none+crop_shift
- pixel_grid_sequence / none+crop_shift: missing mode none+crop_shift
- flow_grid_sequence / none+crop_shift: missing mode none+crop_shift
