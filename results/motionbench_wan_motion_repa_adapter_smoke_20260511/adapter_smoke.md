# MotionBench Wan-Motion-REPA Adapter Training

| Model | Acc | Text-only | Gain | Correct/total | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapter_qa_only | 0.4896 | 0.4896 | +0.0000 | 47/96 | 0.4167 | 0.5417 | 0.5833 | 0.4167 |
| adapter_raw_wan | 0.4896 | 0.4896 | +0.0000 | 47/96 | 0.4167 | 0.5417 | 0.5833 | 0.4167 |
| adapter_multi_target | 0.4896 | 0.4896 | +0.0000 | 47/96 | 0.4167 | 0.5417 | 0.5833 | 0.4167 |

## Loss Setup

`L = L_QA + lambda_align*L_align + lambda_relation*L_relation + lambda_equiv*L_equivariance + lambda_contrast*L_temporal_contrast + lambda_preserve*L_preserve_base`
