# Wan 2.1 Next-Step Overlap Run

- run_tag: `next_overlap_fixed2_20260509_053429`
- host: `DCTN-0506023702`
- cuda_visible_devices: `2`
- result_dir: `/home/hskim/VideoGen-LLM/results/wan21_next_steps_next_overlap_fixed2_20260509_053429`

This run implements and exercises the next-step roadmap:

- same-token stress baselines with spatial/temporal/reverse/shuffle controls
- temporal augmentation modes: uniform, nonuniform, speed jitter, repeats, blur, JPEG, crop shift, reverse, shuffle
- JSONL -> H5 Wan/pixel/flow feature extraction with richer metadata
- cached-feature real benchmark probing scaffold
- feature-motion frame selector
- cached Wan grid motion-token adapter probe

## Main outputs

- `wan_stress_fair_controls_smoke.md`
- `wan_lowfps_aug_controls_smoke.md`
- `synthetic_wan_features.h5`
- `synthetic_wan_features_metadata.jsonl`
- `wan_real_probe_synthetic.md`
- `wan_frame_selector_synthetic.jsonl`
- `wan_adapter_probe_synthetic.md`
- `logs/`
