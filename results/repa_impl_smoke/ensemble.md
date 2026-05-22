# MotionBench Residual Score Ensemble

| Model | Overall | Hard subset | Action Order | Repetition Count | Motion Recognition | Motion Objects | Correct/total |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 | 3/12 |
| wan_morepa | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 | 3/12 |
| calibrated_residual_ensemble | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0000 | 3/12 |

## Best Weight Sweep

| rank | weights | acc | hard acc | correct/total |
|---:|---|---:|---:|---:|
| 1 | text=0, wan_morepa=1 | 0.2500 | 0.2500 | 3/12 |
| 2 | text=1, wan_morepa=0 | 0.2500 | 0.2500 | 3/12 |
| 3 | text=1, wan_morepa=1 | 0.2500 | 0.2500 | 3/12 |

## Complementarity

- oracle_union_accuracy: 0.25
- oracle_union_correct: 3
- oracle_union_total: 12
- text_only_vs_wan_morepa: 0
- wan_morepa_only_vs_text: 0
