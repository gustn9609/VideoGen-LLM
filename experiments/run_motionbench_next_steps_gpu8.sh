#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
BASE_RUN="${BASE_RUN:-$ROOT/results/motionbench_real_20260509_062454}"
RUN_TAG="${RUN_TAG:-motionbench_next_steps_gpu8_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
VIDEOLLM_SCORES_JSONL="${VIDEOLLM_SCORES_JSONL:-}"
RUN_CLIP="${RUN_CLIP:-1}"
RUN_DIT="${RUN_DIT:-1}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

FEATURES_H5="${FEATURES_H5:-$BASE_RUN/motionbench_features.h5}"
METADATA_JSONL="${METADATA_JSONL:-$BASE_RUN/motionbench_features_metadata.jsonl}"
SUBSET_JSONL="${SUBSET_JSONL:-$BASE_RUN/motionbench_subset.jsonl}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 8 ]]; then
  echo "GPUS_CSV must contain at least 8 GPU ids; got $GPUS_CSV" >&2
  exit 2
fi

cat > "$RESULT_DIR/README.md" <<EOF
# MotionBench Next Steps GPU8 Run

- run_tag: \`$RUN_TAG\`
- base_run: \`$BASE_RUN\`
- features_h5: \`$FEATURES_H5\`
- metadata_jsonl: \`$METADATA_JSONL\`
- gpus: \`$GPUS_CSV\`
- run_clip: \`$RUN_CLIP\`
- run_dit: \`$RUN_DIT\`
- videollm_scores_jsonl: \`$VIDEOLLM_SCORES_JSONL\`

Jobs are independent unless explicitly chained inside one GPU slot.
EOF

launch_job() {
  local gpu="$1"
  local name="$2"
  local command="$3"
  local job_dir="$RESULT_DIR/$name"
  mkdir -p "$job_dir"
  {
    echo "[$(date --iso-8601=seconds)] START $name gpu=$gpu"
    echo "command=$command"
    CUDA_VISIBLE_DEVICES="$gpu" \
    ROOT="$ROOT" \
    BASE_RUN="$BASE_RUN" \
    RESULT_DIR="$job_dir" \
    FEATURES_H5="$FEATURES_H5" \
    METADATA_JSONL="$METADATA_JSONL" \
    SUBSET_JSONL="$SUBSET_JSONL" \
    VIDEOLLM_SCORES_JSONL="$VIDEOLLM_SCORES_JSONL" \
      bash -lc "cd '$ROOT'; source .venv/bin/activate; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false; $command"
    echo "[$(date --iso-8601=seconds)] DONE $name"
  } > "$LOG_DIR/${name}.log" 2>&1 &
  local pid=$!
  echo -e "$pid\t$gpu\t$name\t$job_dir" | tee -a "$RESULT_DIR/pids.tsv"
}

echo -e "pid\tgpu\tname\tjob_dir" > "$RESULT_DIR/pids.tsv"
echo "result_dir=$RESULT_DIR"
echo "features_h5=$FEATURES_H5"
echo "metadata_jsonl=$METADATA_JSONL"
echo "gpus=$GPUS_CSV"

launch_job "${GPUS[0]}" stable_probe "
python -u experiments/motionbench_stable_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/stable_probe.json\" \
  --output-md \"\$RESULT_DIR/stable_probe.md\" \
  --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence,metadata_only \
  --modes none,uniform5,nonuniform5 \
  --label-column label \
  --folds 5 \
  --repeats 5 \
  --max-feature-dim 4096 \
  --baseline-feature pixel_grid_sequence
"

launch_job "${GPUS[1]}" temporal_controls "
python -u experiments/motionbench_temporal_controls.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/temporal_controls.json\" \
  --output-md \"\$RESULT_DIR/temporal_controls.md\" \
  --feature-names wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
  --conditions normal,first_frame_only,time_average,shuffled,reversed,uniform5,nonuniform5,metadata_only,none+crop_shift \
  --label-column label \
  --folds 5 \
  --repeats 5 \
  --max-feature-dim 4096
"

launch_job "${GPUS[2]}" candidate_hash_and_ensemble "
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/candidate_rerank_hash.json\" \
  --output-md \"\$RESULT_DIR/candidate_rerank_hash.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_hash.jsonl\" \
  --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
  --modes none,uniform5,nonuniform5 \
  --text-encoder hash \
  --text-dim 1024 \
  --joint-dim 512 \
  --classifier logistic \
  --folds 5 \
  --repeats 5 && \
python -u experiments/motionbench_videollm_ensemble.py \
  --wan-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_hash.jsonl\" \
  --output-json \"\$RESULT_DIR/ensemble_wan_only_hash.json\" \
  --output-md \"\$RESULT_DIR/ensemble_wan_only_hash.md\" \
  --wan-mode none \
  --wan-feature wan_vae_grid_sequence && \
if [[ -n \"\$VIDEOLLM_SCORES_JSONL\" ]]; then \
python -u experiments/motionbench_videollm_ensemble.py \
  --wan-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_hash.jsonl\" \
  --videollm-scores-jsonl \"\$VIDEOLLM_SCORES_JSONL\" \
  --output-json \"\$RESULT_DIR/ensemble_videollm_wan_hash.json\" \
  --output-md \"\$RESULT_DIR/ensemble_videollm_wan_hash.md\" \
  --wan-mode none \
  --wan-feature wan_vae_grid_sequence \
  --normalizations zscore,rank,raw \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --betas 0,0.25,0.5,0.75,1.0; \
fi
"

launch_job "${GPUS[3]}" candidate_tfidf "
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/candidate_rerank_tfidf.json\" \
  --output-md \"\$RESULT_DIR/candidate_rerank_tfidf.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_tfidf.jsonl\" \
  --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
  --modes none,uniform5,nonuniform5 \
  --text-encoder tfidf \
  --text-dim 2048 \
  --joint-dim 512 \
  --classifier logistic \
  --folds 5 \
  --repeats 5
"

if [[ "$RUN_CLIP" == "1" ]]; then
  launch_job "${GPUS[4]}" candidate_clip "
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/candidate_rerank_clip.json\" \
  --output-md \"\$RESULT_DIR/candidate_rerank_clip.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_clip.jsonl\" \
  --feature-names wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
  --modes none,uniform5,nonuniform5 \
  --text-encoder clip \
  --text-model openai/clip-vit-base-patch32 \
  --text-device cuda:0 \
  --text-batch-size 16 \
  --joint-dim 512 \
  --classifier logistic \
  --folds 5 \
  --repeats 5
"
else
  launch_job "${GPUS[4]}" candidate_hash_mlp "
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/candidate_rerank_hash_mlp.json\" \
  --output-md \"\$RESULT_DIR/candidate_rerank_hash_mlp.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_hash_mlp.jsonl\" \
  --feature-names wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
  --modes none,uniform5,nonuniform5 \
  --text-encoder hash \
  --text-dim 1024 \
  --joint-dim 512 \
  --classifier mlp \
  --folds 5 \
  --repeats 3
"
fi

launch_job "${GPUS[5]}" frame_selector_and_dit_feature_scorer "
python -u experiments/motionbench_frame_selector_eval.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-jsonl \"\$RESULT_DIR/frame_selector.jsonl\" \
  --output-md \"\$RESULT_DIR/frame_selector.md\" \
  --mode none \
  --budgets 4,8,16,32,64 \
  --selectors uniform,random,pixel_motion,flow_motion,wan_saliency,question_text,candidate_text \
  --keep-endpoints && \
python -u experiments/extract_wan_features.py \
  --video-jsonl \"\$SUBSET_JSONL\" \
  --output-h5 \"\$RESULT_DIR/motionbench_features_with_dit.h5\" \
  --metadata-output \"\$RESULT_DIR/motionbench_features_with_dit_metadata.jsonl\" \
  --feature-types wan_vae_grid_sequence,wan_vae_global_sequence,pixel_grid_sequence,flow_grid_sequence,dit_l14_t900_token_mean,dit_l14_t900_spatial_tokens \
  --num-frames 17 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none && \
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 \"\$RESULT_DIR/motionbench_features_with_dit.h5\" \
  --metadata-jsonl \"\$RESULT_DIR/motionbench_features_with_dit_metadata.jsonl\" \
  --output-json \"\$RESULT_DIR/candidate_rerank_dit_features.json\" \
  --output-md \"\$RESULT_DIR/candidate_rerank_dit_features.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/dit_feature_candidate_scores.jsonl\" \
  --feature-names dit_l14_t900_token_mean,dit_l14_t900_spatial_tokens \
  --modes none \
  --text-encoder hash \
  --text-dim 1024 \
  --joint-dim 512 \
  --classifier logistic \
  --folds 5 \
  --repeats 3
"

if [[ "$RUN_DIT" == "1" ]]; then
  launch_job "${GPUS[6]}" dit_qa_token_mean "
python -u experiments/motionbench_dit_qa_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/dit_qa_token_mean.json\" \
  --output-md \"\$RESULT_DIR/dit_qa_token_mean.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/dit_qa_token_mean_scores.jsonl\" \
  --mode none \
  --feature-name wan_vae_grid_sequence \
  --layer 14 \
  --timestep 900 \
  --pooling token_mean \
  --folds 5 \
  --batch-size 2 \
  --text-encoder hash \
  --device cuda:0
"
  launch_job "${GPUS[7]}" dit_qa_spatial_tokens "
python -u experiments/motionbench_dit_qa_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/dit_qa_spatial_tokens.json\" \
  --output-md \"\$RESULT_DIR/dit_qa_spatial_tokens.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/dit_qa_spatial_tokens_scores.jsonl\" \
  --mode none \
  --feature-name wan_vae_grid_sequence \
  --layer 14 \
  --timestep 900 \
  --pooling spatial_tokens \
  --folds 5 \
  --batch-size 2 \
  --text-encoder hash \
  --device cuda:0
"
else
  launch_job "${GPUS[6]}" candidate_hash_ridge "
python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-json \"\$RESULT_DIR/candidate_rerank_hash_ridge.json\" \
  --output-md \"\$RESULT_DIR/candidate_rerank_hash_ridge.md\" \
  --output-scores-jsonl \"\$RESULT_DIR/wan_candidate_scores_hash_ridge.jsonl\" \
  --feature-names wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
  --modes none,uniform5,nonuniform5 \
  --text-encoder hash \
  --classifier ridge \
  --folds 5 \
  --repeats 5
"
  launch_job "${GPUS[7]}" frame_selector_wide "
python -u experiments/motionbench_frame_selector_eval.py \
  --features-h5 \"\$FEATURES_H5\" \
  --metadata-jsonl \"\$METADATA_JSONL\" \
  --output-jsonl \"\$RESULT_DIR/frame_selector_wide.jsonl\" \
  --output-md \"\$RESULT_DIR/frame_selector_wide.md\" \
  --mode none \
  --budgets 2,4,8,16,32,64 \
  --selectors uniform,random,pixel_motion,flow_motion,wan_saliency,question_text,candidate_text \
  --keep-endpoints
"
fi

echo "Launched jobs:"
cat "$RESULT_DIR/pids.tsv"
echo "Logs: $LOG_DIR"
echo "Use: tail -f $LOG_DIR/<job>.log"
