# VideoChat2-HD CE/Eq/Rel Overlap and Ensemble

## Single Runs

| Run | Acc | Correct/total | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| ce | 0.4583 | 44/96 | 1.927 | -0.0097 | 1.6264 |
| eq | 0.4792 | 46/96 | 1.906 | 0.0053 | 1.5298 |
| rel | 0.4792 | 46/96 | 1.969 | 0.0443 | 1.6099 |

## Correct-Set Overlap

| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 48 |
| 0 | 0 | 1 | 2 |
| 0 | 1 | 1 | 2 |
| 1 | 1 | 0 | 2 |
| 1 | 1 | 1 | 42 |

Oracle any-correct CE/Eq/Rel: 48/96 = 0.5000

## Per-Type Overlap

### Action Order
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 16 |
| 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 7 |

### Motion Recognition
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 9 |
| 0 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 13 |

### Motion-related Objects
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 8 |
| 0 | 0 | 1 | 2 |
| 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 13 |

### Repetition Count
| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 15 |
| 1 | 1 | 1 | 9 |

## Margin/Rank By Overlap Group

| Group | N | Run | Acc | Mean rank | Mean margin | Mean NLL |
|---|---:|---|---:|---:|---:|---:|
| all_correct | 42 | ce | 1.0000 | 1.000 | 2.5906 | 0.3619 |
| all_correct | 42 | eq | 1.0000 | 1.000 | 2.3854 | 0.3707 |
| all_correct | 42 | rel | 1.0000 | 1.000 | 2.6432 | 0.3693 |
| all_wrong | 48 | ce | 0.0000 | 2.771 | -2.2813 | 2.8040 |
| all_wrong | 48 | eq | 0.0000 | 2.771 | -2.0575 | 2.6011 |
| all_wrong | 48 | rel | 0.0000 | 2.875 | -2.2318 | 2.7787 |
| ce_eq_not_rel | 2 | ce | 1.0000 | 1.000 | 0.6408 | 0.6852 |
| ce_eq_not_rel | 2 | eq | 1.0000 | 1.000 | 0.1333 | 0.8602 |
| ce_eq_not_rel | 2 | rel | 0.0000 | 2.500 | -0.2819 | 1.0606 |
| eq_rel_not_ce | 2 | ce | 0.0000 | 2.000 | -0.0568 | 1.1317 |
| eq_rel_not_ce | 2 | eq | 1.0000 | 1.000 | 0.0487 | 1.0647 |
| eq_rel_not_ce | 2 | rel | 1.0000 | 1.000 | 0.0993 | 1.0649 |
| rel_only | 2 | ce | 0.0000 | 2.000 | -0.7026 | 1.3524 |
| rel_only | 2 | eq | 0.0000 | 2.000 | -0.6397 | 1.2936 |
| rel_only | 2 | rel | 1.0000 | 1.000 | 0.3681 | 0.7059 |

## Nested Logit Ensemble

Nested ensemble: 45/96 = 0.4688, mean NLL 1.6603, mean margin -0.2883

| Fold | Norm | Weights | Val acc | Test acc | Test correct/total |
|---:|---|---|---:|---:|---:|
| 0 | raw | ce=0.00, eq=0.00, rel=1.00 | 0.4868 | 0.4500 | 9/20 |
| 1 | zscore | ce=0.05, eq=0.55, rel=0.40 | 0.4935 | 0.4737 | 9/19 |
| 2 | zscore | ce=0.05, eq=0.55, rel=0.40 | 0.5195 | 0.3684 | 7/19 |
| 3 | zscore | ce=0.00, eq=0.65, rel=0.35 | 0.4675 | 0.5789 | 11/19 |
| 4 | zscore | ce=0.15, eq=0.30, rel=0.55 | 0.4805 | 0.4737 | 9/19 |

## Reference Ensembles

| Ensemble | Acc | Correct/total | Mean rank | Mean margin | Mean NLL | Norm | Weights |
|---|---:|---:|---:|---:|---:|---|---|
| equal_logprob | 0.4896 | 47/96 | 1.927 | 0.0196 | 1.5778 | logprob | ce=0.33, eq=0.33, rel=0.33 |
| full_oof_tuned | 0.4896 | 47/96 | 1.938 | -0.2679 | 1.3294 | zscore | ce=0.05, eq=0.55, rel=0.40 |
