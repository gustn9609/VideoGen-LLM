# MotionBench Wan-Motion-REPA Adapter Training

| Model | Acc | Text-only | Gain | Correct/total | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter_qa_only | 0.4789 | 0.4589 | +0.0200 | 46/96 | 0.3750 | 0.5000 | 0.5417 | 0.5000 |
| adapter_raw_wan | 0.4684 | 0.4589 | +0.0095 | 45/96 | 0.3333 | 0.5000 | 0.5000 | 0.5417 |
| adapter_structured_compact | 0.4895 | 0.4589 | +0.0305 | 47/96 | 0.4167 | 0.5000 | 0.5000 | 0.5417 |
| adapter_dynamics_relation | 0.4789 | 0.4589 | +0.0200 | 46/96 | 0.3750 | 0.5000 | 0.5000 | 0.5417 |
| adapter_residualized_dynamics_relation | 0.5211 | 0.4589 | +0.0621 | 50/96 | 0.5000 | 0.5000 | 0.5833 | 0.5000 |
| adapter_relation_only | 0.4795 | 0.4589 | +0.0205 | 46/96 | 0.3750 | 0.5000 | 0.5000 | 0.5417 |
| adapter_equivariance | 0.4789 | 0.4589 | +0.0200 | 46/96 | 0.4167 | 0.5000 | 0.5000 | 0.5000 |
| adapter_multi_target | 0.4789 | 0.4589 | +0.0200 | 46/96 | 0.4167 | 0.5000 | 0.5000 | 0.5000 |

## Loss Setup

`L = L_QA + lambda_align*L_align + lambda_relation*L_relation + lambda_equiv*L_equivariance + lambda_contrast*L_temporal_contrast + lambda_preserve*L_preserve_base`
