# Wan Ablation + MVBench Run

- run_tag: `wan_ablation_mvbench_20260509_160337`
- motionbench_jsonl: `/home/hskim/VideoGen-LLM/results/motionbench_real_20260509_062454/motionbench_subset.jsonl`
- mvbench_json_dir: `/home/hskim/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots/230a2d4fac8900333c61754641c7a13e069ac9c6/json`
- mvbench_video_root: ``
- gpus: `0,1,2,3,4,5,6,7`
- text_encoder: `hash`
- probe_folds: `5`
- probe_repeats: `3`

Each GPU slot runs an independent extraction/probe chain.
