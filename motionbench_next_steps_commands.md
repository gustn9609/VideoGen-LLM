# MotionBench Next-Step Experiment Commands

Assume:

```bash
cd /home/hskim/VideoGen-LLM
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

BASE_RUN=results/motionbench_real_20260509_062454
FEATURES_H5=$BASE_RUN/motionbench_features.h5
METADATA_JSONL=$BASE_RUN/motionbench_features_metadata.jsonl
OUT=results/motionbench_next_steps_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
```

## 1. Stabilized MotionBench Probe

```bash
python -u experiments/motionbench_stable_probe.py \
  --features-h5 "$FEATURES_H5" \
  --metadata-jsonl "$METADATA_JSONL" \
  --output-json "$OUT/stable_probe.json" \
  --output-md "$OUT/stable_probe.md" \
  --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence,metadata_only \
  --modes none,uniform5,nonuniform5 \
  --label-column label \
  --folds 5 \
  --repeats 5 \
  --max-feature-dim 4096 \
  --baseline-feature pixel_grid_sequence
```

## 2. Temporal Shortcut Controls

```bash
python -u experiments/motionbench_temporal_controls.py \
  --features-h5 "$FEATURES_H5" \
  --metadata-jsonl "$METADATA_JSONL" \
  --output-json "$OUT/temporal_controls.json" \
  --output-md "$OUT/temporal_controls.md" \
  --feature-names wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
  --conditions normal,first_frame_only,time_average,shuffled,reversed,uniform5,nonuniform5,metadata_only,none+crop_shift \
  --label-column label \
  --folds 5 \
  --repeats 5 \
  --max-feature-dim 4096
```

## 3. Candidate-Conditioned Wan Scorer

```bash
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 "$FEATURES_H5" \
  --metadata-jsonl "$METADATA_JSONL" \
  --output-json "$OUT/candidate_rerank.json" \
  --output-md "$OUT/candidate_rerank.md" \
  --output-scores-jsonl "$OUT/wan_candidate_scores.jsonl" \
  --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
  --modes none,uniform5,nonuniform5 \
  --text-encoder hash \
  --text-dim 1024 \
  --joint-dim 512 \
  --classifier logistic \
  --folds 5 \
  --repeats 5
```

Text encoder alternatives:

```bash
--text-encoder clip --text-model openai/clip-vit-base-patch32 --text-device cuda:0
--text-encoder tfidf
--text-encoder wan-t5 --text-model google/umt5-small --text-device cuda:0
--text-encoder sentence-transformer --text-model sentence-transformers/all-MiniLM-L6-v2
```

## 4. VideoLLM + Wan Ensemble

Wan-only sanity report:

```bash
python -u experiments/motionbench_videollm_ensemble.py \
  --wan-scores-jsonl "$OUT/wan_candidate_scores.jsonl" \
  --output-json "$OUT/ensemble_wan_only.json" \
  --output-md "$OUT/ensemble_wan_only.md" \
  --wan-mode none \
  --wan-feature wan_vae_grid_sequence
```

With VideoLLM candidate scores:

```bash
python -u experiments/motionbench_videollm_ensemble.py \
  --wan-scores-jsonl "$OUT/wan_candidate_scores.jsonl" \
  --videollm-scores-jsonl /path/to/videollm_candidate_scores.jsonl \
  --output-json "$OUT/ensemble_videollm_wan.json" \
  --output-md "$OUT/ensemble_videollm_wan.md" \
  --wan-mode none \
  --wan-feature wan_vae_grid_sequence \
  --normalizations zscore,rank,raw \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --betas 0,0.25,0.5,0.75,1.0
```

Expected VideoLLM JSONL schema can be either:

```json
{"video_id": "...", "answer_index": 2, "candidates": [{"score": -1.2}, {"score": -0.4}, {"score": -2.0}, {"score": -3.1}]}
```

or:

```json
{"video_id": "...", "answer_index": 2, "scores": [-1.2, -0.4, -2.0, -3.1]}
```

## 5. Question-Aware Frame Selector

```bash
python -u experiments/motionbench_frame_selector_eval.py \
  --features-h5 "$FEATURES_H5" \
  --metadata-jsonl "$METADATA_JSONL" \
  --output-jsonl "$OUT/frame_selector.jsonl" \
  --output-md "$OUT/frame_selector.md" \
  --mode none \
  --budgets 4,8,16,32,64 \
  --selectors uniform,random,pixel_motion,flow_motion,wan_saliency,question_text,candidate_text \
  --keep-endpoints
```

If VideoLLM frame-budget results exist:

```bash
python -u experiments/motionbench_frame_selector_eval.py \
  --features-h5 "$FEATURES_H5" \
  --metadata-jsonl "$METADATA_JSONL" \
  --output-jsonl "$OUT/frame_selector_with_acc.jsonl" \
  --output-md "$OUT/frame_selector_with_acc.md" \
  --mode none \
  --budgets 4,8,16,32,64 \
  --selectors uniform,random,pixel_motion,flow_motion,wan_saliency,question_text,candidate_text \
  --videollm-frame-results-jsonl /path/to/videollm_frame_budget_results.jsonl \
  --keep-endpoints
```

## 6. Extract DiT Spatial Tokens

```bash
python -u experiments/extract_wan_features.py \
  --video-jsonl "$BASE_RUN/motionbench_subset.jsonl" \
  --output-h5 "$OUT/motionbench_features_with_dit.h5" \
  --metadata-output "$OUT/motionbench_features_with_dit_metadata.jsonl" \
  --feature-types wan_vae_grid_sequence,wan_vae_global_sequence,pixel_grid_sequence,flow_grid_sequence,dit_l14_t900_token_mean,dit_l14_t900_spatial_tokens \
  --num-frames 17 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none
```

## 7. QA-Conditioned DiT Hidden Probe

```bash
python -u experiments/motionbench_dit_qa_probe.py \
  --features-h5 "$FEATURES_H5" \
  --metadata-jsonl "$METADATA_JSONL" \
  --output-json "$OUT/dit_qa_probe.json" \
  --output-md "$OUT/dit_qa_probe.md" \
  --output-scores-jsonl "$OUT/dit_qa_candidate_scores.jsonl" \
  --mode none \
  --feature-name wan_vae_grid_sequence \
  --layer 14 \
  --timestep 900 \
  --pooling spatial_tokens \
  --folds 5 \
  --batch-size 2 \
  --text-encoder hash \
  --device cuda:0
```

## 8. One-Command Runner

Core next-step experiments:

```bash
RUN_TAG=motionbench_next_steps_core CUDA_VISIBLE_DEVICES=1 \
  bash experiments/run_motionbench_next_steps.sh
```

With VideoLLM ensemble:

```bash
RUN_TAG=motionbench_next_steps_with_vlm CUDA_VISIBLE_DEVICES=1 \
RUN_ENSEMBLE=1 VIDEOLLM_SCORES_JSONL=/path/to/videollm_candidate_scores.jsonl \
  bash experiments/run_motionbench_next_steps.sh
```

With QA-conditioned DiT:

```bash
RUN_TAG=motionbench_next_steps_with_dit CUDA_VISIBLE_DEVICES=1 RUN_DIT=1 \
  bash experiments/run_motionbench_next_steps.sh
```

## 9. 8-GPU Overlap Runner

Run all next-step experiments split across eight GPU slots. Each job gets a
single visible GPU through `CUDA_VISIBLE_DEVICES`; this can be overlapped with
other GPU jobs if there is enough free memory.

```bash
RUN_TAG=motionbench_next_steps_gpu8_$(date +%Y%m%d_%H%M%S)
RESULT_DIR=/home/hskim/VideoGen-LLM/results/$RUN_TAG

setsid env \
  RUN_TAG="$RUN_TAG" \
  RESULT_DIR="$RESULT_DIR" \
  GPUS_CSV=0,1,2,3,4,5,6,7 \
  RUN_DIT=1 \
  RUN_CLIP=1 \
  /home/hskim/VideoGen-LLM/experiments/run_motionbench_next_steps_gpu8.sh \
  > "$RESULT_DIR.launcher.log" 2>&1 < /dev/null &
```

Current job map:

- GPU 0: stable probe
- GPU 1: temporal controls
- GPU 2: hash candidate rerank + WAN-only ensemble
- GPU 3: TF-IDF candidate rerank
- GPU 4: CLIP-text candidate rerank, or hash MLP if `RUN_CLIP=0`
- GPU 5: frame selector eval + DiT feature extraction + DiT-feature rerank
- GPU 6: QA-conditioned DiT probe with token-mean pooling
- GPU 7: QA-conditioned DiT probe with spatial-token pooling

Monitor:

```bash
tail -f "$RESULT_DIR"/logs/*.log
cat "$RESULT_DIR"/pids.tsv
nvidia-smi
```
