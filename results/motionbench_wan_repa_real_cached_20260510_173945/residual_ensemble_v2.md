# MotionBench Residual Score Ensemble

| Model | Overall | Hard subset | Action Order | Repetition Count | Motion Recognition | Motion Objects | Correct/total |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 0.4583 | 0.1746 | 0.3333 | 0.5000 | 0.4583 | 0.5417 | 44/96 |
| pixel | 0.4688 | 0.2063 | 0.3750 | 0.5417 | 0.4167 | 0.5417 | 45/96 |
| flow | 0.4479 | 0.2063 | 0.3333 | 0.5000 | 0.4167 | 0.5417 | 43/96 |
| raw_wan | 0.4792 | 0.2222 | 0.3750 | 0.5417 | 0.5000 | 0.5000 | 46/96 |
| wan_moft | 0.4375 | 0.1587 | 0.2917 | 0.5000 | 0.4583 | 0.5000 | 42/96 |
| wan_trd | 0.3854 | 0.1905 | 0.3750 | 0.4583 | 0.2917 | 0.4167 | 37/96 |
| wan_morepa | 0.4792 | 0.2063 | 0.3750 | 0.5417 | 0.5000 | 0.5000 | 46/96 |
| calibrated_residual_ensemble | 0.4479 | 0.1746 | 0.3333 | 0.5417 | 0.4583 | 0.4583 | 43/96 |

## Best Weight Sweep

| rank | weights | acc | hard acc | correct/total |
|---:|---|---:|---:|---:|
| 1 | text=0, pixel=0, flow=0.25, raw_wan=0, wan_moft=0, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 2 | text=0, pixel=0, flow=0.25, raw_wan=0, wan_moft=0.25, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 3 | text=0, pixel=0, flow=0.25, raw_wan=0, wan_moft=0.5, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 4 | text=0, pixel=0, flow=0.25, raw_wan=0.25, wan_moft=0.25, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 5 | text=0, pixel=0, flow=0.25, raw_wan=0.25, wan_moft=0.5, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 6 | text=0, pixel=0.25, flow=0, raw_wan=0, wan_moft=0.25, wan_trd=0, wan_morepa=1 | 0.5000 | 0.2381 | 48/96 |
| 7 | text=0, pixel=0.25, flow=0, raw_wan=0, wan_moft=0.5, wan_trd=0, wan_morepa=1 | 0.5000 | 0.2381 | 48/96 |
| 8 | text=0, pixel=0.25, flow=0, raw_wan=0, wan_moft=0.5, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 9 | text=0, pixel=0.25, flow=0, raw_wan=0, wan_moft=1, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 10 | text=0, pixel=0.25, flow=0, raw_wan=0.5, wan_moft=0, wan_trd=0, wan_morepa=0 | 0.5000 | 0.2381 | 48/96 |
| 11 | text=0, pixel=0.25, flow=0.25, raw_wan=0, wan_moft=0, wan_trd=0, wan_morepa=1 | 0.5000 | 0.2381 | 48/96 |
| 12 | text=0, pixel=0.25, flow=0.25, raw_wan=0, wan_moft=0, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 13 | text=0, pixel=0.25, flow=0.25, raw_wan=0, wan_moft=0.25, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 14 | text=0, pixel=0.25, flow=0.25, raw_wan=0, wan_moft=0.5, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 15 | text=0, pixel=0.25, flow=0.25, raw_wan=0.25, wan_moft=0.5, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 16 | text=0, pixel=0.25, flow=0.5, raw_wan=0, wan_moft=0, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 17 | text=0, pixel=0.25, flow=0.5, raw_wan=0, wan_moft=0, wan_trd=0.25, wan_morepa=1 | 0.5000 | 0.2381 | 48/96 |
| 18 | text=0, pixel=0.25, flow=0.5, raw_wan=0, wan_moft=0, wan_trd=0.25, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 19 | text=0, pixel=0.25, flow=0.5, raw_wan=0, wan_moft=0.25, wan_trd=0, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |
| 20 | text=0, pixel=0.25, flow=0.5, raw_wan=0, wan_moft=0.25, wan_trd=0.25, wan_morepa=2 | 0.5000 | 0.2381 | 48/96 |

## Complementarity

- flow_only_vs_pixel: 4
- flow_only_vs_raw_wan: 3
- flow_only_vs_text: 4
- flow_only_vs_wan_moft: 4
- flow_only_vs_wan_morepa: 3
- flow_only_vs_wan_trd: 11
- oracle_union_accuracy: 0.5416666666666666
- oracle_union_correct: 52
- oracle_union_total: 96
- pixel_only_vs_flow: 6
- pixel_only_vs_raw_wan: 2
- pixel_only_vs_text: 2
- pixel_only_vs_wan_moft: 5
- pixel_only_vs_wan_morepa: 2
- pixel_only_vs_wan_trd: 10
- raw_wan_only_vs_flow: 6
- raw_wan_only_vs_pixel: 3
- raw_wan_only_vs_text: 4
- raw_wan_only_vs_wan_moft: 5
- raw_wan_only_vs_wan_morepa: 3
- raw_wan_only_vs_wan_trd: 11
- text_only_vs_flow: 5
- text_only_vs_pixel: 1
- text_only_vs_raw_wan: 2
- text_only_vs_wan_moft: 3
- text_only_vs_wan_morepa: 1
- text_only_vs_wan_trd: 10
- text_pixel_flow_wrong_raw_wan_right: 1
- text_pixel_flow_wrong_wan_moft_right: 0
- text_pixel_flow_wrong_wan_morepa_right: 1
- text_pixel_flow_wrong_wan_trd_right: 1
- wan_moft_only_vs_flow: 3
- wan_moft_only_vs_pixel: 2
- wan_moft_only_vs_raw_wan: 1
- wan_moft_only_vs_text: 1
- wan_moft_only_vs_wan_morepa: 1
- wan_moft_only_vs_wan_trd: 10
- wan_morepa_only_vs_flow: 6
- wan_morepa_only_vs_pixel: 3
- wan_morepa_only_vs_raw_wan: 3
- wan_morepa_only_vs_text: 3
- wan_morepa_only_vs_wan_moft: 5
- wan_morepa_only_vs_wan_trd: 11
- wan_trd_only_vs_flow: 5
- wan_trd_only_vs_pixel: 2
- wan_trd_only_vs_raw_wan: 2
- wan_trd_only_vs_text: 3
- wan_trd_only_vs_wan_moft: 5
- wan_trd_only_vs_wan_morepa: 2
