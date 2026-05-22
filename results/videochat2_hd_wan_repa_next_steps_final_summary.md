# VideoChat2-HD Wan-REPA Next Steps Final Summary

Date: 2026-05-11

## Scope

Implemented and ran the requested follow-up experiments:

1. Correct-set overlap: CE vs Eq vs Rel
2. Logit-level ensemble with nested weight calibration
3. Branch-separated Eq/Rel REPA
4. Lambda schedule sweep
5. Negative controls
6. Larger temporal set

The original 96-sample setting uses the cached high-motion+camera-comp VideoChat2-HD hidden tokens.  The larger set uses a newly built 192-sample MotionBench subset with 48 samples per question type.  For the larger set I used uniform 8-frame sampling (`mode=none`) because high-motion+camera-comp extraction on 192 videos was bottlenecked by full-video decode.

## Step 1. Correct-Set Overlap

96-sample CE/Eq/Rel single-run summary:

| Run | Acc | Correct/total | Mean rank | Mean margin | Mean NLL |
|---|---:|---:|---:|---:|---:|
| CE | 0.4583 | 44/96 | 1.927 | -0.0097 | 1.6264 |
| Eq | 0.4792 | 46/96 | 1.906 | 0.0053 | 1.5298 |
| Rel | 0.4792 | 46/96 | 1.969 | 0.0443 | 1.6099 |

Correct-set overlap:

| CE | Eq | Rel | N |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 48 |
| 0 | 0 | 1 | 2 |
| 0 | 1 | 1 | 2 |
| 1 | 1 | 0 | 2 |
| 1 | 1 | 1 | 42 |

Oracle any-correct CE/Eq/Rel is 48/96 = 0.5000.  Eq and Rel are only weakly complementary: Rel has 2 unique correct samples, Eq has no Eq-only sample, and Eq+Rel jointly recover 2 CE-wrong samples.

## Step 2. Logit-Level Ensemble

Nested calibration over CE/Eq/Rel logits:

| Setting | Acc | Correct/total |
|---|---:|---:|
| Best single, Eq or Rel | 0.4792 | 46/96 |
| Nested calibrated ensemble | 0.4688 | 45/96 |
| Equal logprob ensemble | 0.4896 | 47/96 |
| Full-OOF tuned ensemble | 0.4896 | 47/96 |

The non-nested reference ensembles can reach 47/96, but nested calibration does not beat the best single model.  So there is not enough reliable evidence that a calibrated CE/Eq/Rel logit ensemble is better than Eq/Rel alone.

## Step 3. Branch-Separated Eq/Rel REPA

Implemented a `branch_separated` adapter:

- Eq residual branch receives the equivariance REPA loss.
- Rel residual branch receives the relation REPA loss.
- LLM input uses a gated sum of both residual branches.

Result:

| Experiment | Acc | Correct/total |
|---|---:|---:|
| Branch-separated Eq+Rel | 0.3542 | 34/96 |

This failed clearly.  The simple branch split did not solve the Eq/Rel combination problem; it made Motion-related Objects and Repetition Count much worse.

## Step 4. Schedule Sweep

All schedule runs used Eq REPA with 3 epochs.

| Schedule | Acc | Correct/total |
|---|---:|---:|
| constant lambda=0.1 | 0.4271 | 41/96 |
| warm-up then CE polish | 0.4479 | 43/96 |
| late-start REPA | 0.4167 | 40/96 |
| cosine decay | 0.4167 | 40/96 |

The 3-epoch schedules hurt relative to the 1-epoch Eq run.  The main failure mode is Repetition Count collapse:

| Schedule | Repetition Count |
|---|---:|
| CE-only baseline | 9/24 |
| Eq 1 epoch | 9/24 |
| constant 3 epoch | 5/24 |
| warm-up then CE polish | 3/24 |
| late-start | 3/24 |
| cosine decay | 3/24 |

## Step 5. Negative Controls

Negative-control target alignment:

| Target | Acc | Correct/total |
|---|---:|---:|
| Wan Eq target | 0.4792 | 46/96 |
| random target | 0.4792 | 46/96 |
| pixel target | 0.4792 | 46/96 |
| first-frame target | 0.4688 | 45/96 |
| flow target | 0.4688 | 45/96 |
| time-average target | 0.4688 | 45/96 |

This is the strongest negative result.  Random and pixel target alignment match Wan Eq at 46/96.  Therefore the +2 on the 96-sample set is not convincing evidence of Wan-specific target information in the current adapter setup.

## Step 6. Larger Temporal Set

Built a larger MotionBench temporal subset:

- 192 videos total
- 48 Action Order
- 48 Motion Recognition
- 48 Motion-related Objects
- 48 Repetition Count
- Uniform 8-frame sampling, `mode=none`

Generated:

- Wan/pixel/flow features: `results/motionbench_larger_temporal_20260511/features_pool1f8_none.h5`
- Wan-REPA targets: `results/motionbench_larger_temporal_20260511/wmrepa_targets_none.h5`
- VideoChat2-HD hidden tokens: `results/motionbench_larger_temporal_20260511/hidden_tokens_none.h5`

Larger-set CE/Eq/Rel results:

| Run | Acc | Correct/total |
|---|---:|---:|
| CE | 0.4427 | 85/192 |
| Eq | 0.4427 | 85/192 |
| Rel | 0.4219 | 81/192 |

Per-type:

| Run | Action Order | Motion Recognition | Motion-related Objects | Repetition Count |
|---|---:|---:|---:|---:|
| CE | 13/48 | 28/48 | 32/48 | 12/48 |
| Eq | 13/48 | 27/48 | 32/48 | 13/48 |
| Rel | 14/48 | 28/48 | 32/48 | 7/48 |

Larger-set overlap/ensemble:

| Metric | Value |
|---|---:|
| Oracle any-correct CE/Eq/Rel | 95/192 = 0.4948 |
| Nested CE/Eq/Rel ensemble | 84/192 = 0.4375 |
| Full-OOF tuned ensemble | 87/192 = 0.4531 |

The larger set does not reproduce the 96-sample Eq gain.  Eq ties CE overall, and Rel is worse.  Eq only shifts one correct sample from Motion Recognition to Repetition Count.

## Overall Conclusion

The current Wan-REPA direction is weak in this VideoChat2-HD adapter setting.

1. Eq/Rel complementarity exists only at a small oracle level, and nested logit calibration cannot reliably exploit it.
2. Branch-separated Eq/Rel training fails.
3. Longer schedule variants hurt, mostly by damaging Repetition Count.
4. Negative controls match the Wan Eq result, so the observed +2 on 96 samples is not Wan-specific.
5. On the larger 192-sample set, Eq does not improve over CE and Rel drops below CE.

The practical interpretation is that the current auxiliary REPA loss is acting more like a small regularizer than a meaningful Wan-motion transfer signal.  I would not treat this as a solid paper result without changing the supervision target or moving to a stronger training/evaluation regime.

## Key Files

- 96-sample overlap/ensemble: `results/videochat2_hd_wan_repa_next_steps_cv5/overlap_ensemble.md`
- 96-sample branch/schedule/control summary: `results/videochat2_hd_wan_repa_next_steps_cv5/next_steps_summary.md`
- Larger-set summary: `results/videochat2_hd_wan_repa_larger_temporal_20260511/larger_temporal_summary.md`
- Larger-set overlap/ensemble: `results/videochat2_hd_wan_repa_larger_temporal_20260511/overlap_ensemble.md`
- Final summary: `results/videochat2_hd_wan_repa_next_steps_final_summary.md`
