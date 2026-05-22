# Wan temporal reasoning follow-up report

## Purpose

This run extended the previous motion/shuffle probes toward VideoLLM failure modes from `wan21_videollm_research.md`:

- before/after temporal order
- object interaction
- repetition counting
- low-frame-rate robustness

The experiment used the previously selected features:

- `wan_vae_global_sequence`
- `wan_vae_grid_sequence`
- `wan_vae_global_delta`
- `dit_l14_t900_token_mean`
- pixel/grid baselines

Script: `experiments/wan_temporal_reasoning_experiments.py`

## Important control

The first `before_after` task had leakage: `pixel_grid_time_avg` reached 100%, meaning temporal order could be inferred from time-averaged dwell traces. I added `before_after_cycle`, where each object performs an out-and-back cycle so time-averaged pixels are near chance.

This stricter control is the more meaningful before/after result.

## Main results

| task | key baseline | Wan result | interpretation |
|---|---:|---:|---|
| before_after_cycle | pixel time avg 53.1% | VAE global seq 94.8%, global delta 99.0%, grid seq 100.0% | Strong evidence for temporal-order signal in Wan-VAE. |
| interaction | pixel time avg 99.0% | VAE global seq 97.9%, grid/delta 100.0%, DiT 91.7% | Wan works well, but this synthetic interaction task still has static spatial cues. |
| repetition4 | pixel time avg 42.2% | VAE global/grid/delta 100.0% | Strong evidence that Wan-VAE captures count-sensitive motion. |
| low-fps direction4 | global seq full 92.7% -> low 26.6% | grid seq full 100.0% -> low 100.0% | Compact global pooling is brittle under frame-rate shift; grid sequence is robust. |
| low-fps before_after_cycle | global seq full 91.7% -> low 64.6% | grid seq full 100.0% -> low 100.0% | Same pattern: grid tokens preserve sparse-frame information better. |

## Feature-level conclusion

`wan_vae_grid_sequence` is now the strongest default candidate for a VideoLLM adapter:

- 100% on strict before/after cycle
- 100% on interaction
- 100% on repetition count
- 100% under low-frame-rate direction and before/after tests

`wan_vae_global_sequence` is still attractive because it is compact:

- 94.8% on strict before/after cycle
- 97.9% on interaction
- 100% on repetition count

But it is not robust to low-frame-rate distribution shift when trained on full-frame clips and tested on sparse/held clips:

- direction4 drops from 92.7% to 26.6%
- before_after_cycle drops from 91.7% to 64.6%

`wan_vae_global_delta` is strong on full-frame temporal tasks but brittle under low frame rate:

- before_after_cycle 99.0% full task
- low-fps before_after_cycle drops to 50.0%

`dit_l14_t900_token_mean` remains useful but should not be the primary feature:

- before_after_cycle 69.8%
- interaction 91.7%
- repetition4 57.8%
- low-fps direction4 drops to 25.0%

## Updated adapter recommendation

Use a two-tier Wan motion adapter:

```text
Tier 1, default:
  Wan-VAE grid_sequence
    -> spatial/temporal resampler
    -> 8-64 motion tokens

Tier 2, compact ablation:
  Wan-VAE global_sequence
    -> projector
    -> 5 temporal tokens

Optional:
  DiT layer14 timestep900 token_mean
    -> one global dynamics token
```

Do not use only `global_sequence` if the target VideoLLM setting uses aggressive frame sampling. Either keep spatial grid structure or train with low-frame-rate augmentation.

## Next best experiment

The next experiment should stop using synthetic-only labels and move to a real or pseudo-real benchmark:

1. Build an extractor that saves `wan_vae_grid_sequence`, `wan_vae_global_sequence`, and `dit_l14_t900_token_mean` to disk.
2. Run frozen linear/MLP probes on a real temporal dataset: SSV2, MotionBench, MVBench, or a small local VideoQA subset.
3. Add low-frame-rate augmentation during probe training and compare:
   - train full, test full
   - train full, test low-fps
   - train full+low-fps, test low-fps
4. If the real benchmark trend matches this synthetic run, implement the VideoLLM motion-token adapter.

