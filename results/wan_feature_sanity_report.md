# Wan feature sanity experiment report

## Setup

- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Features tested:
  - `Wan-VAE latent`: frozen VAE encoder output, normalized with Wan latent mean/std.
  - `Wan-DiT denoising loss`: Wan transformer flow-prediction MSE, zero text conditioning.
- Synthetic videos: 17 frames, 128x128.
- Train/test: `train_base=48`, `test_base=24`, seed 7.
- Output files:
  - `results/wan_feature_sanity_results.md`
  - `results/wan_feature_sanity_results.json`

## Main results

| task | baseline | Wan feature | result |
|---|---:|---:|---|
| 4-way motion direction | frame RGB mean 19.8%, pixel delta 97.9% | VAE global 96.9%, VAE grid 100.0% | Wan-VAE carries strong direction signal. |
| frame shuffle detection | frame RGB mean 37.5%, pixel grid 79.2%, pixel delta 60.4% | VAE global 100.0%, VAE grid 100.0% | Wan-VAE carries very strong temporal-order/coherence signal. |
| gravity forward vs reversed | all baselines 100.0% | all Wan features 100.0% | Not diagnostic; synthetic data leaks easy global cues. |

## DiT denoising-loss check

Lower loss is better. This test used zero text conditioning, so it should be treated as a weak lower bound for a real prompt-conditioned reranker.

| timestep | normal | reversed | shuffled | normal<reversed | normal<shuffled |
|---:|---:|---:|---:|---:|---:|
| 900 | 1.625805 | 1.632518 | 1.665083 | 45.8% | 79.2% |
| 500 | 4.732586 | 4.719334 | 4.635388 | 37.5% | 4.2% |
| 100 | 5.738333 | 5.752999 | 5.765300 | 54.2% | 75.0% |

Interpretation:

- High-noise timestep 900 shows a useful signal for shuffled-frame detection: normal videos have lower loss than shuffled videos in 79.2% of pairs.
- Reversed-video detection is not reliable under zero text conditioning.
- Mid-noise timestep 500 is actively unhelpful for this synthetic setup.
- The DiT loss result is weaker than the VAE probe and should not yet be used as evidence that a zero-shot reranker will work.

## Conclusion

Wan feature is useful enough to justify the next experiment, but the strongest evidence here is for frozen feature probing rather than denoising-loss reranking.

The useful finding is that a compact `Wan-VAE global` feature with only 80 dimensions matches or beats much larger low-level pixel baselines on motion direction and shuffle detection. That suggests Wan latents expose temporal/motion structure in a form that a simple linear probe can use.

The less useful finding is that zero-text Wan-DiT denoising loss is only partially sensitive to shuffled order and does not robustly detect reversal. For VideoQA reranking, the next test should use real prompt-conditioned text embeddings and a real motion QA benchmark, not zero text and synthetic clips.

## Recommended next step

Use Wan features first as an adapter/probing signal:

1. Extract `Wan-VAE global/grid` and selected Wan-DiT hidden states.
2. Run linear/MLP probes on a real temporal benchmark such as SSV2, MVBench, or MotionBench.
3. Only then test prompt-conditioned Wan denoising loss for multiple-choice VideoQA reranking.

