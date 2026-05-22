# Wan low-frame-rate augmentation report

## Purpose

The previous temporal reasoning experiment showed that compact Wan features can fail under frame-rate shift:

- `wan_vae_global_sequence` was strong on full-frame clips but weak when full-frame-trained probes were tested on low-fps clips.
- `wan_vae_grid_sequence` stayed robust.

This experiment tested whether low-fps augmentation during probe training fixes that failure.

Script: `experiments/wan_lowfps_aug_experiments.py`

## Setup

- Tasks: `direction4`, `before_after_cycle`
- Train modes:
  - `full->low`: train on normal clips, test on low-fps clips
  - `low->low`: train on low-fps clips, test on low-fps clips
  - `full+low->low`: train on both normal and low-fps clips, test on low-fps clips
- Seeds: 3
- Low-fps simulation: 5 keyframes expanded back to 17 frames
- Probe: frozen feature + Ridge linear classifier
- DiT feature: `layer14/timestep900/token_mean`

## Main results

| task | feature | full->low | full+low->low | gain |
|---|---:|---:|---:|---:|
| direction4 | `dit_l14_t900_token_mean` | 28.6% | 57.8% | +29.2 |
| direction4 | `wan_vae_global_delta` | 27.1% | 42.7% | +15.6 |
| direction4 | `wan_vae_global_sequence` | 25.0% | 28.1% | +3.1 |
| direction4 | `wan_vae_grid_sequence` | 100.0% | 100.0% | 0.0 |
| before_after_cycle | `wan_vae_global_delta` | 53.1% | 97.9% | +44.8 |
| before_after_cycle | `wan_vae_global_sequence` | 57.3% | 89.6% | +32.3 |
| before_after_cycle | `dit_l14_t900_token_mean` | 51.0% | 65.6% | +14.6 |
| before_after_cycle | `wan_vae_grid_sequence` | 100.0% | 100.0% | 0.0 |

## Interpretation

Low-fps augmentation helps, but it does not replace preserving spatial grid structure.

- `wan_vae_grid_sequence` is robust without augmentation.
- `wan_vae_global_sequence` can be rescued for before/after order, but not for direction.
- `wan_vae_global_delta` benefits strongly from augmentation on before/after order.
- DiT global token benefits on direction, but remains weaker than VAE grid features.

## Adapter implication

For a VideoLLM motion adapter:

1. Use `wan_vae_grid_sequence` as the main feature.
2. Add low-fps augmentation during adapter/projector training.
3. Treat `wan_vae_global_sequence` as a compact ablation, not the production default.
4. Use DiT layer14/timestep900 as an auxiliary token only if compute budget allows.

