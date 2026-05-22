# Wan2.1 Feature for VideoLLM: Experiment Summary

This document summarizes all experiments run so far under `/home/hskim/VideoGen-LLM`, based on `wan21_videollm_research.md`.

## Files

### Experiment scripts

- `experiments/wan_feature_sanity.py`
- `experiments/wan_next_experiments.py`
- `experiments/wan_temporal_reasoning_experiments.py`
- `experiments/wan_lowfps_aug_experiments.py`
- `experiments/wan_stress_fair_baselines.py`
- `experiments/extract_wan_features.py`
- `experiments/wan_motion_adapter.py`

### Result reports

- `results/wan_feature_sanity_report.md`
- `results/wan_next_experiments_report.md`
- `results/wan_temporal_reasoning_report.md`
- `results/wan_lowfps_aug_report.md`
- `results/wan_stress_fair_baselines_report.md`

Raw result JSON/summary tables are also saved under `results/`.

## Experiment 1: Initial Wan Feature Sanity

Goal:

- Check whether Wan-VAE and Wan-DiT signals contain motion/order information.

Main results:

| task | baseline | Wan result | conclusion |
|---|---:|---:|---|
| direction4 | frame RGB mean 19.8% | VAE global 96.9%, VAE grid 100.0% | Wan-VAE carries motion direction. |
| shuffle | pixel grid 79.2% | VAE global/grid 100.0% | Wan-VAE strongly captures temporal coherence. |
| DiT denoising loss, shuffle | normal<shuffled 79.2% at t=900 | weak but non-random | Loss score is weaker than hidden/latent features. |
| DiT denoising loss, reversed | normal<reversed 45.8% at t=900 | unreliable | Zero-text loss is not enough for reversal. |

Conclusion:

- Wan-VAE features are immediately useful.
- Zero-text DiT denoising loss should not be the primary reranker.

## Experiment 2: Pooling Ablation and DiT Layer/Timestep Sweep

Goal:

- Identify the useful Wan-VAE pooling style.
- Identify the useful Wan-DiT layer/timestep.

Main results:

| setting | best result | conclusion |
|---|---:|---|
| VAE pooling, direction4 | `wan_vae_grid_sequence` 100.0% | Grid sequence is strongest. |
| VAE pooling, shuffle | compact/global VAE features about 99.0% | Wan-VAE keeps temporal coherence even with compact pooling. |
| DiT hidden, direction4 | layer14/timestep900/token_mean 70.3% | Middle layer at high noise is best. |
| DiT hidden, shuffle | layer14/timestep900 96.9% | DiT hidden states are useful for coherence. |

Conclusion:

- Best VAE feature: `wan_vae_grid_sequence`
- Best compact VAE feature: `wan_vae_global_sequence`
- Best DiT feature tested: `dit_l14_t900_token_mean`

## Experiment 3: Temporal Reasoning Probes

Goal:

- Test tasks closer to VideoLLM failures:
  - before/after order
  - object interaction
  - repetition count
  - low-frame-rate robustness

Important control:

- The first before/after task leaked static/time-average cues.
- A stricter `before_after_cycle` task was added; pixel time average was near chance at 53.1%.

Main results:

| task | key baseline | Wan result | conclusion |
|---|---:|---:|---|
| before_after_cycle | pixel time avg 53.1% | VAE global seq 94.8%, global delta 99.0%, grid seq 100.0% | Wan-VAE captures temporal order. |
| interaction | pixel time avg 99.0% | VAE global seq 97.9%, grid/delta 100.0%, DiT 91.7% | Wan works, but task has static cues. |
| repetition4 | pixel time avg 42.2% | VAE global/grid/delta 100.0% | Wan-VAE captures count-sensitive motion. |
| low-fps direction4 | global seq 92.7% -> 26.6% | grid seq 100.0% -> 100.0% | Compact pooling is brittle; grid sequence is robust. |
| low-fps before_after_cycle | global seq 91.7% -> 64.6% | grid seq 100.0% -> 100.0% | Same pattern. |

Conclusion:

- `wan_vae_grid_sequence` is the strongest default motion-token source.
- `wan_vae_global_sequence` is useful but fragile under frame-rate shift.
- DiT token is useful but weaker than VAE grid.

## Experiment 4: Low-FPS Augmentation

Goal:

- Test whether training with low-fps augmentation fixes compact feature brittleness.

Main results:

| task | feature | full->low | full+low->low | gain |
|---|---:|---:|---:|---:|
| direction4 | DiT layer14/t900 token | 28.6% | 57.8% | +29.2 |
| direction4 | VAE global delta | 27.1% | 42.7% | +15.6 |
| direction4 | VAE global sequence | 25.0% | 28.1% | +3.1 |
| direction4 | VAE grid sequence | 100.0% | 100.0% | 0.0 |
| before_after_cycle | VAE global delta | 53.1% | 97.9% | +44.8 |
| before_after_cycle | VAE global sequence | 57.3% | 89.6% | +32.3 |
| before_after_cycle | DiT layer14/t900 token | 51.0% | 65.6% | +14.6 |
| before_after_cycle | VAE grid sequence | 100.0% | 100.0% | 0.0 |

Conclusion:

- Augmentation helps compact/global features, especially for before/after order.
- It does not make `global_sequence` robust enough for direction.
- `grid_sequence` remains the safest default.

## Experiment 5: Hard Stress Tasks and Same-Token Controls

Goal:

- Test whether Wan grid remains strong when pixel shortcut tasks become harder.
- Compare Wan grid against pixel and motion-grid controls under the same token budget.

Implemented tasks:

- same first/last frame, different path
- camera motion vs object motion
- crossing objects identity
- target-object counting with distractor
- contact/causality push

Main results:

| task | Wan result | strongest control | conclusion |
|---|---:|---:|---|
| same_first_last_path | Wan grid 16/32 linear 100.0% | flow 16/32 100.0%, pixel 32 100.0% | Wan works, but controls also solve it. |
| camera_vs_object_motion | Wan grid/global 100.0% | flow 77.1%, pixel 66.7% | Clear Wan advantage in this control. |
| crossing_identity | Wan grid 16 93.8%, grid 32 100.0% | pixel 32 100.0% | Wan helps at tighter token budget. |
| target_count_distractor | Wan grid 16/32 100.0% | flow 99.0%, pixel 99.0% | Wan best, controls close. |
| contact_causality_push | Wan grid/global 100.0% | flow 95.8%, pixel 58.3% | Wan beats pixel and slightly beats flow proxy. |

Conclusion:

- `wan_vae_grid_sequence` stays strong under same-token compression.
- Strong pixel/flow controls can solve several synthetic tasks, so real benchmark evidence remains necessary.
- The strongest Wan-favorable stress tasks were `camera_vs_object_motion` and `contact_causality_push`.

## Infrastructure Implemented

### Dataset-facing extractor

`experiments/extract_wan_features.py` implements the recommended benchmark extractor.

Input:

```text
metadata.jsonl with at least:
{"video_id": "...", "path": "..."}
```

Supported feature outputs:

```text
wan_vae_grid_sequence
wan_vae_global_sequence
wan_vae_global_delta
dit_l14_t900_token_mean
pixel_grid_sequence
flow_grid_sequence
```

Supported low-FPS modes:

```text
none
uniform5
nonuniform5
```

Output:

```text
features.h5
metadata_aligned.jsonl
```

The extractor was smoke-tested with synthetic `.npy` videos and produced H5 datasets with expected shapes.

### Motion-token adapter skeleton

`experiments/wan_motion_adapter.py` implements:

- `FactorizedMotionResampler`
- `WanMotionTokenAdapter`

It maps Wan VAE latents `[B, C, T, H, W]` to fixed motion tokens `[B, K, D]`, preserving spatial structure before temporal resampling.

## Current Recommendation

Use a two-tier Wan motion adapter:

```text
Main path:
  video
    -> Wan-VAE latent
    -> grid_sequence
    -> spatial/temporal resampler
    -> 8-64 motion tokens
    -> VideoLLM

Compact ablation:
  video
    -> Wan-VAE latent
    -> global_sequence
    -> small projector
    -> 5 temporal tokens

Optional auxiliary:
  noisy Wan latent at timestep 900
    -> Wan-DiT layer 14 hidden state
    -> token_mean
    -> one global dynamics token
```

Training recommendation:

- Freeze Wan.
- Train only projector/resampler first.
- Include low-fps augmentation.
- Evaluate separately on motion direction, temporal order, interaction, and counting questions.

## What Has Not Been Run Yet

These are in the research note but need real video/QA data or a VideoLLM codebase:

- SSV2 / MVBench / MotionBench / Video-MME real benchmark probing.
- Multiple-choice VideoQA diffusion-loss reranking with real question-answer prompts.
- VideoLLM motion-token adapter training.
- Question-aware frame/region selector.
- Wan-generated counterfactual training data and real benchmark transfer.
- Student distillation from Wan feature teacher.

No local video benchmark files or VideoLLM training code were found in the current workspace scan, so these were not executable directly. The extractor and adapter modules were implemented so those experiments can run once data/model code is provided.

## Next Concrete Step

Prepare a dataset-facing extractor:

```text
input: video files + metadata CSV/JSONL
output:
  wan_vae_grid_sequence.npy
  wan_vae_global_sequence.npy
  dit_l14_t900_token_mean.npy
  labels/questions/answers metadata
```

Then run the same frozen linear/MLP probes on SSV2, MotionBench, MVBench, or a small local VideoQA subset.
