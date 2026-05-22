# MotionBench Wan-Motion-REPA Evaluation

| Step | Feature | Acc | Correct/total | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | text_only | 0.4589 | 44/96 | 0.3333 | 0.4583 | 0.5417 | 0.5000 |
| 2 | raw_wan | 0.4484 | 43/96 | 0.3333 | 0.4167 | 0.5417 | 0.5000 |
| 3 | structured_compact | 0.4163 | 40/96 | 0.2917 | 0.3750 | 0.5417 | 0.4583 |
| 4 | dynamics_relation | 0.3542 | 34/96 | 0.2500 | 0.2917 | 0.4167 | 0.4583 |
| 5 | residualized_dynamics_relation | 0.4274 | 41/96 | 0.2917 | 0.4167 | 0.5417 | 0.4583 |
| 6 | relation_only | 0.4589 | 44/96 | 0.2917 | 0.4583 | 0.5417 | 0.5417 |
| 7 | equivariance | 0.4047 | 39/96 | 0.3333 | 0.4583 | 0.4167 | 0.4167 |
| 8 | multi_target | 0.4047 | 39/96 | 0.2500 | 0.5000 | 0.4167 | 0.4583 |

## Pairwise Gain vs Baselines
| Feature | Gain vs text | Gain vs raw Wan |
|---|---:|---:|
| text_only | +0.0000 | +0.0105 |
| raw_wan | -0.0105 | +0.0000 |
| structured_compact | -0.0426 | -0.0321 |
| dynamics_relation | -0.1047 | -0.0942 |
| residualized_dynamics_relation | -0.0316 | -0.0211 |
| relation_only | +0.0000 | +0.0105 |
| equivariance | -0.0542 | -0.0437 |
| multi_target | -0.0542 | -0.0437 |
