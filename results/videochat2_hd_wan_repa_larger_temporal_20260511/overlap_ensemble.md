# VideoChat2-HD CE/Eq/Rel Overlap and Ensemble

## Single Runs

| Run | Acc | Correct/total | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| ce | 0.4427 | 85/192 | 1.964 | 0.5247 | 1.8060 |
| eq | 0.4427 | 85/192 | 1.964 | 0.5494 | 1.8392 |
| rel | 0.4219 | 81/192 | 2.026 | 0.4129 | 1.6263 |

## Correct-Set Overlap

| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 97 |
| 0 | 0 | 1 | 6 |
| 0 | 1 | 0 | 3 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 3 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 8 |
| 1 | 1 | 1 | 73 |

Oracle any-correct CE/Eq/Rel: 95/192 = 0.4948

## Per-Type Overlap

### Action Order
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 31 |
| 0 | 0 | 1 | 2 |
| 0 | 1 | 0 | 1 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 2 |
| 1 | 1 | 1 | 11 |

### Motion Recognition
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 18 |
| 0 | 0 | 1 | 2 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 2 |
| 1 | 1 | 1 | 25 |

### Motion-related Objects
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 15 |
| 0 | 0 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 31 |

### Repetition Count
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 33 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 2 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 0 | 5 |
| 1 | 1 | 1 | 6 |

## Margin/Rank By Overlap Group

| Group | N | Run | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---|---:|---:|---:|---:|
| all_correct | 73 | ce | 1.0000 | 1.000 | 4.9864 | 0.1660 |
| all_correct | 73 | eq | 1.0000 | 1.000 | 5.1825 | 0.1560 |
| all_correct | 73 | rel | 1.0000 | 1.000 | 4.1015 | 0.2292 |
| all_wrong | 97 | ce | 0.0000 | 2.773 | -2.7234 | 3.2504 |
| all_wrong | 97 | eq | 0.0000 | 2.804 | -2.8121 | 3.3178 |
| all_wrong | 97 | rel | 0.0000 | 2.835 | -2.1953 | 2.7728 |
| ce_eq_not_rel | 8 | ce | 1.0000 | 1.000 | 0.4744 | 0.6227 |
| ce_eq_not_rel | 8 | eq | 1.0000 | 1.000 | 0.3580 | 0.7263 |
| ce_eq_not_rel | 8 | rel | 0.0000 | 2.250 | -0.7188 | 1.5227 |
| ce_only | 3 | ce | 1.0000 | 1.000 | 0.1826 | 0.8043 |
| ce_only | 3 | eq | 0.0000 | 2.000 | -0.2617 | 0.9943 |
| ce_only | 3 | rel | 0.0000 | 2.333 | -0.5430 | 1.1938 |
| ce_rel_not_eq | 1 | ce | 1.0000 | 1.000 | 0.0781 | 0.7165 |
| ce_rel_not_eq | 1 | eq | 0.0000 | 2.000 | -0.3282 | 0.9359 |
| ce_rel_not_eq | 1 | rel | 1.0000 | 1.000 | 1.0939 | 0.4516 |
| eq_only | 3 | ce | 0.0000 | 2.333 | -0.2893 | 1.1275 |
| eq_only | 3 | eq | 1.0000 | 1.000 | 0.1446 | 0.9295 |
| eq_only | 3 | rel | 0.0000 | 2.667 | -1.1048 | 1.7584 |
| eq_rel_not_ce | 1 | ce | 0.0000 | 2.000 | -0.1896 | 0.8476 |
| eq_rel_not_ce | 1 | eq | 1.0000 | 1.000 | 0.7967 | 0.3807 |
| eq_rel_not_ce | 1 | rel | 1.0000 | 1.000 | 0.5143 | 0.5071 |
| rel_only | 6 | ce | 0.0000 | 2.333 | -0.4114 | 1.1654 |
| rel_only | 6 | eq | 0.0000 | 2.000 | -0.5055 | 1.1684 |
| rel_only | 6 | rel | 1.0000 | 1.000 | 0.3178 | 0.7606 |

## Nested Logit Ensemble

Nested ensemble: 84/192 = 0.4375, mean NLL 1.5673, mean margin 0.3440

| Fold | Norm | Weights | Val acc | Test acc | Test correct/total |
|---:|---|---|---:|---:|---:|
| 0 | logprob | ce=0.55, eq=0.45, rel=0.00 | 0.4444 | 0.4872 | 19/39 |
| 1 | zscore | ce=0.65, eq=0.15, rel=0.20 | 0.4510 | 0.4359 | 17/39 |
| 2 | raw | ce=0.25, eq=0.75, rel=0.00 | 0.4675 | 0.3947 | 15/38 |
| 3 | zscore | ce=0.25, eq=0.75, rel=0.00 | 0.4545 | 0.4211 | 16/38 |
| 4 | logprob | ce=0.55, eq=0.45, rel=0.00 | 0.4545 | 0.4474 | 17/38 |

## Reference Ensembles

| Ensemble | Acc | Correct/total | Mean rank | Mean margin | Mean NLL | Norm | Weights |
|---|---:|---:|---:|---:|---:|---|---|
| equal_logprob | 0.4271 | 82/192 | 2.005 | 0.5354 | 1.7228 | logprob | ce=0.33, eq=0.33, rel=0.33 |
| full_oof_tuned | 0.4531 | 87/192 | 1.943 | 0.5435 | 1.8125 | logprob | ce=0.55, eq=0.45, rel=0.00 |
