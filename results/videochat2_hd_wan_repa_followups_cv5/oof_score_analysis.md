# OOF Candidate Score Analysis

| Run | N | Acc | Mean rank | Mean margin | Median margin | Mean NLL |
|---|---:|---:|---:|---:|---:|---:|
| ce | 96 | 0.4583 | 1.927 | -0.0097 | -0.1737 | 1.6264 |
| equiv | 96 | 0.4792 | 1.906 | 0.0053 | -0.2359 | 1.5298 |
| relation | 96 | 0.4792 | 1.969 | 0.0443 | -0.1574 | 1.6099 |
| dynamics | 96 | 0.4688 | 1.917 | -0.0158 | -0.1348 | 1.6304 |
| type_cond | 96 | 0.4688 | 1.927 | 0.0014 | -0.2236 | 1.5129 |
| kl | 96 | 0.4688 | 1.969 | -0.1064 | -0.2714 | 1.7136 |

## Per-Type

### ce
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.2917 | 2.333 | -1.1084 | 2.2175 |
| Motion Recognition | 24 | 0.5833 | 1.583 | 0.1781 | 1.7420 |
| Motion-related Objects | 24 | 0.5833 | 1.583 | 1.4514 | 0.9282 |
| Repetition Count | 24 | 0.3750 | 2.208 | -0.5601 | 1.6178 |

### equiv
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.3333 | 2.250 | -1.0064 | 2.0740 |
| Motion Recognition | 24 | 0.6250 | 1.542 | 0.1743 | 1.6106 |
| Motion-related Objects | 24 | 0.5833 | 1.583 | 1.3810 | 0.8945 |
| Repetition Count | 24 | 0.3750 | 2.250 | -0.5275 | 1.5401 |

### relation
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.3333 | 2.375 | -1.0983 | 2.2020 |
| Motion Recognition | 24 | 0.5833 | 1.625 | 0.0185 | 1.8575 |
| Motion-related Objects | 24 | 0.6250 | 1.625 | 1.7602 | 0.8158 |
| Repetition Count | 24 | 0.3750 | 2.250 | -0.5030 | 1.5644 |

### dynamics
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.2917 | 2.333 | -1.0926 | 2.2076 |
| Motion Recognition | 24 | 0.5833 | 1.583 | 0.1532 | 1.7451 |
| Motion-related Objects | 24 | 0.6250 | 1.542 | 1.4538 | 0.9310 |
| Repetition Count | 24 | 0.3750 | 2.208 | -0.5777 | 1.6381 |

### type_cond
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.2917 | 2.333 | -0.9807 | 2.0470 |
| Motion Recognition | 24 | 0.6250 | 1.542 | 0.1558 | 1.5971 |
| Motion-related Objects | 24 | 0.5833 | 1.583 | 1.3306 | 0.8922 |
| Repetition Count | 24 | 0.3750 | 2.250 | -0.5002 | 1.5152 |

### kl
| Type | N | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| Action Order | 24 | 0.3333 | 2.417 | -1.2164 | 2.3051 |
| Motion Recognition | 24 | 0.6250 | 1.583 | 0.0035 | 1.7451 |
| Motion-related Objects | 24 | 0.5417 | 1.625 | 1.5045 | 0.9906 |
| Repetition Count | 24 | 0.3750 | 2.250 | -0.7174 | 1.8135 |

## Paired

### ce_vs_equiv
- `n`: 96
- `b_only_correct`: 2
- `a_only_correct`: 0
- `both_correct`: 44
- `both_wrong`: 50
- `mean_delta_correct`: 0.020833333333333332
- `mean_delta_margin`: 0.015056134999194152
- `mean_delta_candidate_nll`: -0.09657990964535614
- `mean_delta_gold_rank`: -0.020833333333333332

### ce_vs_relation
- `n`: 96
- `b_only_correct`: 4
- `a_only_correct`: 2
- `both_correct`: 42
- `both_wrong`: 48
- `mean_delta_correct`: 0.020833333333333332
- `mean_delta_margin`: 0.05406948840497231
- `mean_delta_candidate_nll`: -0.01645350594568167
- `mean_delta_gold_rank`: 0.041666666666666664

### ce_vs_dynamics
- `n`: 96
- `b_only_correct`: 1
- `a_only_correct`: 0
- `both_correct`: 44
- `both_wrong`: 51
- `mean_delta_correct`: 0.010416666666666666
- `mean_delta_margin`: -0.00611988118453155
- `mean_delta_candidate_nll`: 0.004053469281077834
- `mean_delta_gold_rank`: -0.010416666666666666

### ce_vs_type_cond
- `n`: 96
- `b_only_correct`: 1
- `a_only_correct`: 0
- `both_correct`: 44
- `both_wrong`: 51
- `mean_delta_correct`: 0.010416666666666666
- `mean_delta_margin`: 0.011102829648393708
- `mean_delta_candidate_nll`: -0.11350013014588307
- `mean_delta_gold_rank`: 0.0

### ce_vs_kl
- `n`: 96
- `b_only_correct`: 2
- `a_only_correct`: 1
- `both_correct`: 43
- `both_wrong`: 50
- `mean_delta_correct`: 0.010416666666666666
- `mean_delta_margin`: -0.09671528604970565
- `mean_delta_candidate_nll`: 0.08720855830677393
- `mean_delta_gold_rank`: 0.041666666666666664
