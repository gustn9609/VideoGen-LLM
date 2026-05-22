# MotionBench Wan-Motion-REPA Evaluation

| Step | Feature | Acc | Correct/total | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | text_only | 0.4589 | 44/96 | 0.3333 | 0.4583 | 0.5417 | 0.5000 |
| 2 | raw_wan | 0.3847 | 37/96 | 0.3333 | 0.3333 | 0.4167 | 0.4583 |
| 3 | structured_compact | 0.4053 | 39/96 | 0.3333 | 0.2083 | 0.5833 | 0.5000 |
| 4 | dynamics_relation | 0.3542 | 34/96 | 0.2083 | 0.3750 | 0.4167 | 0.4167 |
| 5 | residualized_dynamics_relation | 0.2905 | 28/96 | 0.2500 | 0.2500 | 0.2083 | 0.4583 |
| 6 | relation_only | 0.4589 | 44/96 | 0.2917 | 0.4583 | 0.5417 | 0.5417 |
| 7 | equivariance | 0.3532 | 34/96 | 0.2500 | 0.4167 | 0.3333 | 0.4167 |
| 8 | multi_target | 0.3111 | 30/96 | 0.2083 | 0.3333 | 0.2917 | 0.4167 |

## Pairwise Gain vs Baselines
| Feature | Gain vs text | Gain vs raw Wan |
|---|---:|---:|
| text_only | +0.0000 | +0.0742 |
| raw_wan | -0.0742 | +0.0000 |
| structured_compact | -0.0537 | +0.0205 |
| dynamics_relation | -0.1047 | -0.0305 |
| residualized_dynamics_relation | -0.1684 | -0.0942 |
| relation_only | +0.0000 | +0.0742 |
| equivariance | -0.1058 | -0.0316 |
| multi_target | -0.1479 | -0.0737 |
