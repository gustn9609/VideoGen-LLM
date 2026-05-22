# VideoChat2-HD Wan-REPA Step Experiments Summary

Date: 2026-05-11

## Setup

- Dataset: MotionBench real subset, 96 QA samples, 4-way candidate classification.
- VideoLLM hidden tokens: `results/videochat2_hd_hidden_motionbench_f8_hmcc/hidden_tokens.h5`
- Combined features: `results/videochat2_hd_hidden_motionbench_f8_hmcc/combined_features.h5`
- CV setting: same 5-fold split with `split_seed=123`.
- Baseline: CE-only VideoChat2-HD hidden-token adapter.

## Step 1. Robustness Check

Experiment: current equivariance Wan-REPA setting, 5 random seeds, lambda sweep, same 5-fold split.

| Lambda | Runs | Mean acc | Min | Max |
|---:|---:|---:|---:|---:|
| 0.0 | 5 | 0.4583 | 0.4271 | 0.4896 |
| 0.03 | 5 | 0.4604 | 0.4271 | 0.4896 |
| 0.1 | 5 | 0.4729 | 0.4375 | 0.5104 |
| 0.3 | 5 | 0.4688 | 0.4479 | 0.5104 |

Per-run results:

| Seed | Lambda | Acc | Correct/total |
|---:|---:|---:|---:|
| 123 | 0.0 | 0.4583 | 44/96 |
| 124 | 0.0 | 0.4479 | 43/96 |
| 125 | 0.0 | 0.4271 | 41/96 |
| 126 | 0.0 | 0.4688 | 45/96 |
| 127 | 0.0 | 0.4896 | 47/96 |
| 123 | 0.03 | 0.4688 | 45/96 |
| 124 | 0.03 | 0.4375 | 42/96 |
| 125 | 0.03 | 0.4271 | 41/96 |
| 126 | 0.03 | 0.4792 | 46/96 |
| 127 | 0.03 | 0.4896 | 47/96 |
| 123 | 0.1 | 0.4792 | 46/96 |
| 124 | 0.1 | 0.4583 | 44/96 |
| 125 | 0.1 | 0.4375 | 42/96 |
| 126 | 0.1 | 0.4792 | 46/96 |
| 127 | 0.1 | 0.5104 | 49/96 |
| 123 | 0.3 | 0.4792 | 46/96 |
| 124 | 0.3 | 0.4583 | 44/96 |
| 125 | 0.3 | 0.4479 | 43/96 |
| 126 | 0.3 | 0.4479 | 43/96 |
| 127 | 0.3 | 0.5104 | 49/96 |

Result: lambda 0.1 is the best mean setting. The trend is small but mostly non-destructive: +1.4 correct samples on average over CE-only.

## Step 2. OOF Logit / Margin Analysis

OOF candidate scores were saved and analyzed for accuracy, gold rank, gold-vs-best-negative margin, NLL, paired deltas, and per-type metrics.

| Run | N | Acc | Mean rank | Mean margin | Median margin | Mean NLL |
|---|---:|---:|---:|---:|---:|---:|
| ce | 96 | 0.4583 | 1.927 | -0.0097 | -0.1737 | 1.6264 |
| equiv | 96 | 0.4792 | 1.906 | 0.0053 | -0.2359 | 1.5298 |
| relation | 96 | 0.4792 | 1.969 | 0.0443 | -0.1574 | 1.6099 |
| dynamics | 96 | 0.4688 | 1.917 | -0.0158 | -0.1348 | 1.6304 |
| type_cond | 96 | 0.4688 | 1.927 | 0.0014 | -0.2236 | 1.5129 |
| kl | 96 | 0.4688 | 1.969 | -0.1064 | -0.2714 | 1.7136 |

Paired vs CE-only:

| Pair | New-only correct | CE-only correct | Both correct | Both wrong | Mean acc delta | Mean margin delta | Mean NLL delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ce_vs_equiv | 2 | 0 | 44 | 50 | +0.0208 | +0.0151 | -0.0966 |
| ce_vs_relation | 4 | 2 | 42 | 48 | +0.0208 | +0.0541 | -0.0165 |
| ce_vs_dynamics | 1 | 0 | 44 | 51 | +0.0104 | -0.0061 | +0.0041 |
| ce_vs_type_cond | 1 | 0 | 44 | 51 | +0.0104 | +0.0111 | -0.1135 |
| ce_vs_kl | 2 | 1 | 43 | 50 | +0.0104 | -0.0967 | +0.0872 |

Result: equivariance improves accuracy, mean rank, mean margin, and NLL. Relation REPA has the largest margin improvement but worse mean rank than CE-only.

## Step 3. CE-Preserving REPA

Implemented and evaluated:

- `L = CE + Wan-REPA + KL_to_CE_only`
- zero-initialized REPA residual branch

| Experiment | Acc | Correct/total |
|---|---:|---:|
| CE-only baseline | 0.4583 | 44/96 |
| CE-preserving KL | 0.4688 | 45/96 |
| zero-init REPA residual | 0.4062 | 39/96 |

Result: KL gives a small +1 correct gain but does not preserve margins/NLL. Zero-init residual branch fails in this setup.

## Step 4. Relation REPA

Implemented student segment relation matrix alignment against Wan temporal relation matrix.

| Experiment | Acc | Correct/total |
|---|---:|---:|
| relation_only | 0.4792 | 46/96 |
| relation_plus_equiv | 0.4688 | 45/96 |

Result: relation-only matches the best seed-123 equivariance run. Combining relation and equivariance did not help.

Per-type comparison:

| Run | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|
| CE-only | 7/24 | 14/24 | 14/24 | 9/24 |
| equiv | 8/24 | 15/24 | 14/24 | 9/24 |
| relation | 8/24 | 14/24 | 15/24 | 9/24 |

Relation REPA helps a different slice than equivariance: it improves Motion-related Objects while equivariance improves Motion Recognition.

## Step 5. Dynamics REPA

Implemented dynamics targets from Wan temporal sequence:

- first difference
- second difference
- temporal energy
- autocorrelation
- temporal curvature

| Experiment | Acc | Correct/total |
|---|---:|---:|
| dynamics_repa | 0.4688 | 45/96 |

Per-type:

| Run | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|
| CE-only | 7/24 | 14/24 | 14/24 | 9/24 |
| dynamics | 7/24 | 14/24 | 15/24 | 9/24 |

Result: dynamics REPA did not improve the intended Action Order / Repetition Count / Motion Recognition categories.

## Step 6. Type-Conditional Lambda

Implemented per-question-type Wan-REPA weights:

- Action Order: high lambda
- Motion Recognition: high lambda
- Motion-related Objects: low lambda
- Repetition Count: medium lambda

| Experiment | Acc | Correct/total |
|---|---:|---:|
| type_conditional_lambda | 0.4688 | 45/96 |

Per-type:

| Run | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|
| CE-only | 7/24 | 14/24 | 14/24 | 9/24 |
| type_conditional_lambda | 7/24 | 15/24 | 14/24 | 9/24 |

Result: type-conditional lambda does not outperform uniform lambda 0.1.

## Step 7. Unlabeled Pretraining

Implemented adapter pretraining with Wan-REPA before CE finetuning. Since no extra unlabeled pool was available in this run, the pretraining scope used all 96 videos as unlabeled video-only data.

| Experiment | Acc | Correct/total |
|---|---:|---:|
| unlabeled_pretrain_all | 0.4375 | 42/96 |

Result: unlabeled-style Wan-REPA pretraining hurts downstream CV accuracy in this setup.

## Overall Result

Best accuracy observed:

- Seed-robust mean best: equivariance Wan-REPA, lambda 0.1, mean 0.4729 over 5 seeds.
- Single seed best among follow-ups: equivariance or relation-only, 46/96 = 0.4792.

Main conclusions:

1. The original +2 correct samples from equivariance Wan-REPA is not just a single-run accident, but the average effect is small.
2. Equivariance gives the cleanest representation-level improvement: better accuracy, rank, margin, and NLL.
3. Relation REPA is promising because it matches equivariance accuracy and helps a different question type.
4. CE-preserving KL, zero-init residual, dynamics REPA, type-conditional lambda, and current unlabeled pretraining did not clearly improve the result.
5. Repetition Count remains unchanged across all tested variants, so the current Wan alignment objective is not solving count/order dynamics strongly enough.

## Output Files

- Robustness sweep: `results/videochat2_hd_wan_repa_robustness_equiv_cv5/sweep_summary.md`
- Follow-up summary: `results/videochat2_hd_wan_repa_followups_cv5/followup_summary.md`
- OOF score analysis: `results/videochat2_hd_wan_repa_followups_cv5/oof_score_analysis.md`
- Machine-readable robustness results: `results/videochat2_hd_wan_repa_robustness_equiv_cv5/sweep_summary.json`
- Machine-readable follow-up results: `results/videochat2_hd_wan_repa_followups_cv5/followup_summary.json`
- Machine-readable OOF analysis: `results/videochat2_hd_wan_repa_followups_cv5/oof_score_analysis.json`
