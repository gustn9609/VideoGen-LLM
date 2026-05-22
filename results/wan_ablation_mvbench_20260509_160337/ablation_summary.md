# Wan Ablation + MVBench Summary

- run_dir: `/home/hskim/VideoGen-LLM/results/wan_ablation_mvbench_20260509_160337`
- metric: candidate rerank accuracy with hash text encoder, logistic classifier, 5 folds x 3 repeats
- MotionBench rows: 96 videos; totals are 288 because each split-repeat evaluates every row once
- MVBench: annotations found, but local source videos were not resolved; see `mvbench_ablation/mvbench/mvbench_missing.jsonl`

## Wan-VAE Grid Resolution

| grid | acc | CI95 | correct/total |
|---|---:|---:|---:|
| 1x1 | 0.4621 | 0.4028-0.5208 | 133/288 |
| 2x2 | 0.4509 | 0.3958-0.5069 | 130/288 |
| 4x4 | 0.4096 | 0.3542-0.4653 | 118/288 |
| 8x8 | 0.4133 | 0.3576-0.4688 | 119/288 |
| 16x16 | 0.3821 | 0.3264-0.4375 | 110/288 |

## Temporal Resolution

| frames | wan_vae_grid_4x4 | pixel_grid_sequence | flow_grid_sequence |
|---:|---:|---:|---:|
| 8 | 0.4104 | 0.3889 | 0.3751 |
| 16 | 0.4133 | 0.3860 | 0.3681 |
| 32 | 0.3854 | 0.4377 | 0.3747 |
| 64 | 0.3919 | 0.4237 | 0.3749 |

## Object-Centric Crop

| feature | full | center_crop | object_crop |
|---|---:|---:|---:|
| wan_vae_grid_4x4 | 0.3989 | 0.3956 | 0.3954 |
| pixel_grid_sequence | 0.4372 | 0.3889 | 0.4274 |
| flow_grid_sequence | 0.4375 | 0.3684 | 0.3644 |

## Motion Saliency / Camera Compensation

| feature | all frames | high_motion | camera_comp |
|---|---:|---:|---:|
| wan_vae_grid_4x4 | 0.3788 | 0.4374 | 0.4133 |
| pixel_grid_sequence | 0.4130 | 0.4270 | 0.3995 |
| flow_grid_sequence | 0.3718 | 0.3923 | 0.3544 |

## Notes

- Grid pooling favored coarse Wan-VAE grids: 1x1 was best, then 2x2; 16x16 was worst.
- Increasing frame count did not monotonically improve Wan-VAE; 8/16 frames were slightly better than 32/64 for Wan 4x4.
- Center/object crops did not improve Wan-VAE on this subset. Pixel features recovered under object crop, but flow degraded under crops.
- High-motion segment sampling improved Wan-VAE and modestly improved pixel/flow; camera compensation helped Wan versus full-frame baseline but hurt flow.
- The `object_crop` implementation uses motion-saliency region proposals because no local detector weights were available.
- The `camera_comp` implementation uses FFT phase-correlation global translation compensation.
