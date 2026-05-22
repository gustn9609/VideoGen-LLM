# Wan hard-stress fair-baseline report

## Purpose

This experiment implements the first executable recommendations from `wan21_next_steps_recommendations.md`:

- same-token comparison
- pixel / flow / Wan grid baselines
- harder synthetic stress tasks
- linear and MLP probes with matched classifier setup
- DiT auxiliary token comparison

Script: `experiments/wan_stress_fair_baselines.py`

## Setup

- Stress tasks:
  - `same_first_last_path`
  - `camera_vs_object_motion`
  - `crossing_identity`
  - `target_count_distractor`
  - `contact_causality_push`
- Token budgets: 16 and 32
- Features:
  - `pixel_grid_same_token`
  - `flow_grid_same_token` using frame-difference motion grid as a lightweight optical-flow proxy
  - `wan_vae_grid_same_token`
  - `wan_vae_global_sequence`
  - `dit_l14_t900_token_mean`
- Classifiers:
  - Ridge linear probe
  - 1-hidden-layer MLP probe
- Seeds: 2

## Main observations

| task | best Wan | strongest control | interpretation |
|---|---:|---:|---|
| same_first_last_path | Wan grid 16/32 linear 100.0% | flow 16/32 linear 100.0%, pixel 32 linear 100.0% | Wan works, but motion/grid controls also solve it. |
| camera_vs_object_motion | Wan grid 16/32 linear 100.0%, global 100.0% | flow 77.1%, pixel 66.7% | Clear Wan advantage in this synthetic camera-motion control. |
| crossing_identity | Wan grid 32 linear 100.0%, 16 linear 93.8% | pixel 32 linear 100.0% | Wan helps at 16 tokens, but pixel catches up at 32. |
| target_count_distractor | Wan grid 16/32 linear 100.0% | flow 16 linear 99.0%, pixel 32 linear 99.0% | Wan is best but strong controls are close. |
| contact_causality_push | Wan grid 16/32 linear 100.0%, global 100.0% | flow 95.8%, pixel 58.3% | Wan beats pixel and slightly beats motion-grid proxy. |

## Important caveats

- Several tasks are still solvable by strong pixel/flow controls.
- MLP probes were unstable with the small synthetic sample size and often underperformed linear probes.
- The `flow_grid_same_token` feature is a frame-difference proxy, not a full optical-flow estimator.

## Updated claim strength

The new result strengthens the claim that `wan_vae_grid_sequence` is a strong motion feature under a fixed token budget. It does not fully prove Wan is always better than raw pixel or flow features, because strong controls solve some tasks.

The clearest Wan-favorable tasks here are:

- `camera_vs_object_motion`
- `contact_causality_push`
- lower-token `crossing_identity`

The next evidence still needed is real benchmark probing.

