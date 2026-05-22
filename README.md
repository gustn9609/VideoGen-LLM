# VideoGen-LLM

Research workspace for probing Wan-derived video features, MotionBench/MVBench evaluation, and VideoChat2-HD adapter training.

This repository contains the experiment code, analysis scripts, and lightweight result summaries. Large artifacts are intentionally excluded from GitHub:

- base model weights under `models/`
- virtual environment files
- downloaded videos and HDF5 feature caches
- trained adapter checkpoints
- raw JSONL score dumps

Selected adapter checkpoints are published separately on Hugging Face:

- `gustn9609/VideoGen-LLM-Wan-REPA-adapters`

## Main Experiment Areas

- Wan VAE feature extraction and low-FPS/control feature probes
- MotionBench real-subset probing
- MVBench feature ablations
- Wan-Motion-REPA target construction
- VideoChat2-HD hidden-token dumping
- VideoChat2-HD adapter fine-tuning with CE, Eq REPA, Relation REPA, schedules, controls, and ensemble analysis

## Key Summaries

- `results/videochat2_hd_wan_repa_next_steps_final_summary.md`
- `results/videochat2_hd_wan_repa_all_steps_summary.md`
- `results/videochat2_hd_wan_repa_robustness_equiv_cv5/sweep_summary.md`
- `results/videochat2_hd_wan_repa_followups_cv5/oof_score_analysis.md`
- `results/videochat2_hd_wan_repa_larger_temporal_20260511/larger_temporal_summary.md`

## Notes

The final VideoChat2-HD Wan-REPA experiments showed weak evidence for Wan-specific gains in the current adapter setup. The strongest negative-control result is that random/pixel target alignment matched the Wan Eq target on the 96-sample setting, and the larger 192-sample set did not reproduce a clear Eq gain over CE.
