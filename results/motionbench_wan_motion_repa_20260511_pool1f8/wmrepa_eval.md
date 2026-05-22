# MotionBench Wan-Motion-REPA Evaluation

| Step | Feature | Acc | Correct/total | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | text_only | 0.4589 | 44/96 | 0.3333 | 0.4583 | 0.5417 | 0.5000 |
| 2 | raw_wan | 0.4800 | 46/96 | 0.3750 | 0.5000 | 0.5000 | 0.5417 |
| 3 | structured_compact | 0.4368 | 42/96 | 0.2917 | 0.4583 | 0.5417 | 0.4583 |
| 4 | dynamics_relation | 0.3532 | 34/96 | 0.2500 | 0.2917 | 0.4167 | 0.4583 |
| 5 | residualized_dynamics_relation | 0.4074 | 39/96 | 0.4167 | 0.3333 | 0.4167 | 0.4583 |
| 6 | relation_only | 0.4589 | 44/96 | 0.3333 | 0.4583 | 0.5417 | 0.5000 |
| 7 | equivariance | 0.2911 | 28/96 | 0.1667 | 0.4167 | 0.2083 | 0.3750 |
| 8 | multi_target | 0.3105 | 30/96 | 0.2083 | 0.3333 | 0.3333 | 0.3750 |

## Pairwise Gain vs Baselines
| Feature | Gain vs text | Gain vs raw Wan |
|---|---:|---:|
| text_only | +0.0000 | -0.0211 |
| raw_wan | +0.0211 | +0.0000 |
| structured_compact | -0.0221 | -0.0432 |
| dynamics_relation | -0.1058 | -0.1268 |
| residualized_dynamics_relation | -0.0516 | -0.0726 |
| relation_only | +0.0000 | -0.0211 |
| equivariance | -0.1679 | -0.1889 |
| multi_target | -0.1484 | -0.1695 |
