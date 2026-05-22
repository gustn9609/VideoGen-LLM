# Wan follow-up experiment report

## What was tested

This follow-up used the previous result and the research note to test two practical questions.

1. Which pooling style makes Wan-VAE useful as a motion token?
2. Which Wan-DiT layer/timestep contains useful motion/order information?

The experiment script is `experiments/wan_next_experiments.py`.

## Experiment 1: VAE pooling ablation

Setup:

- Synthetic tasks: 4-way motion direction, forward-vs-reversed gravity, smooth-vs-shuffled order.
- Seeds: 3.
- Train/test per seed: `train_base=32`, `test_base=16`.
- Video: 17 frames, 128x128.
- Probe: frozen feature + Ridge linear classifier.

Key mean balanced accuracies:

| task | best Wan-VAE | best pixel baseline | interpretation |
|---|---:|---:|---|
| direction4 | `wan_vae_grid_sequence` 100.0% | `pixel_grid_sequence` 97.9% | Wan-VAE is competitive with low-level ordered pixels and better than temporal-average baselines. |
| reversal | sequence/delta Wan-VAE 100.0% | sequence/delta pixel 100.0% | Easy synthetic task; only shows order-aware features can solve it. |
| shuffle | `wan_vae_channel_avg/global_sequence/grid_sequence/global_delta` 99.0% | `pixel_grid_sequence` 75.0% | Strong evidence that Wan-VAE captures temporal coherence beyond low-level frame grids. |

Important ablation:

- CLIP-like time-averaged pixel features were at chance for reversal/shuffle: 50.0%.
- Pixel temporal sequence helped but was much weaker on shuffle: 75.0%.
- Wan-VAE stayed high even with compact pooling:
  - `wan_vae_channel_avg`: 16 dims, 99.0% on shuffle.
  - `wan_vae_global_sequence`: 80 dims, 99.0% on shuffle and 92.2% on direction.

Conclusion:

Wan-VAE is the best immediate feature for a cheap VideoLLM adapter. Use `global_sequence` first because it is compact and preserves temporal ordering. Use `grid_sequence` when accuracy matters more than token budget.

## Experiment 2: DiT hidden-state layer/timestep sweep

Setup:

- Tasks: direction4 and shuffle.
- Layers: early 0, middle 14, late 29.
- Timesteps: 900, 500, 100.
- Pooling: token mean, temporal sequence, temporal delta.
- Text condition: zero embedding. This isolates video/noise representation without prompt conditioning.

Top results:

| task | best DiT feature | balanced accuracy |
|---|---|---:|
| direction4 | layer 14, timestep 900, token mean | 70.3% |
| direction4 | layer 14, timestep 900, temporal sequence | 68.8% |
| shuffle | layer 14, timestep 900, token mean | 96.9% |
| shuffle | layer 14, timestep 900, temporal sequence | 96.9% |
| shuffle | layer 0, timestep 900, token mean | 93.8% |

Pattern:

- High-noise timestep 900 was consistently the best setting.
- Middle layer 14 was the strongest overall.
- Late layer 29 was not better for direction and only competitive for shuffle.
- Token mean often matched or beat temporal sequence, which is useful because it gives a 1536-dim compact feature.

Conclusion:

Wan-DiT hidden states are useful, especially for temporal coherence/shuffle detection. For fine-grained direction on this synthetic setup, VAE features were much stronger than DiT hidden states. If using DiT features as motion tokens, start with layer 14 at high noise and token-mean pooling.

## Updated research direction

The best near-term architecture is:

```text
video
  -> Wan-VAE latent
  -> global_sequence or grid_sequence pooling
  -> small projector / resampler
  -> VideoLLM motion tokens
```

An optional heavier branch can add:

```text
Wan-DiT hidden state at timestep 900, layer 14
  -> token_mean pooling
  -> projector
  -> extra dynamics token
```

Avoid relying on zero-text DiT denoising loss as the primary reranker. The hidden-state probe is much stronger than the loss-only result.

## Next experiment to run

Move from synthetic clips to a real temporal benchmark:

1. Extract `wan_vae_global_sequence`, `wan_vae_grid_sequence`, and `dit_l14_t900_token_mean`.
2. Compare against a CLIP/SigLIP frame-average baseline.
3. Use SSV2/MotionBench/MVBench-style labels for direction/order/action.
4. Train only a linear probe or small MLP first.

This will answer whether the synthetic signal transfers to real VideoLLM failure modes.

