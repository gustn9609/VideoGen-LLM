# OOF Candidate Score Analysis

| Run | N | Acc | Mean rank | Mean margin | Median margin | Mean NLL |
|---|---:|---:|---:|---:|---:|---:|
| ce | 96 | 0.4583 | 1.927 | -0.0097 | -0.1737 | 1.6264 |
| repa | 96 | 0.4792 | 1.906 | 0.0053 | -0.2359 | 1.5298 |

## Per-Type

### ce
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.2917 | 2.333 | -1.1084 | 2.2175 |
| Motion Recognition | 24 | 0.5833 | 1.583 | 0.1781 | 1.7420 |
| Motion-related Objects | 24 | 0.5833 | 1.583 | 1.4514 | 0.9282 |
| Repetition Count | 24 | 0.3750 | 2.208 | -0.5601 | 1.6178 |

### repa
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.3333 | 2.250 | -1.0064 | 2.0740 |
| Motion Recognition | 24 | 0.6250 | 1.542 | 0.1743 | 1.6106 |
| Motion-related Objects | 24 | 0.5833 | 1.583 | 1.3810 | 0.8945 |
| Repetition Count | 24 | 0.3750 | 2.250 | -0.5275 | 1.5401 |

## Paired

### ce_vs_repa
- `n`: 96
- `b_only_correct`: 2
- `a_only_correct`: 0
- `both_correct`: 44
- `both_wrong`: 50
- `mean_delta_correct`: 0.020833333333333332
- `mean_delta_margin`: 0.015056134999194152
- `mean_delta_candidate_nll`: -0.09657990964535614
- `mean_delta_gold_rank`: -0.020833333333333332
