#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
BASE_RUN="${BASE_RUN:-$ROOT/results/motionbench_real_20260509_062454}"
RUN_TAG="${RUN_TAG:-motionbench_next_steps_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_DIT="${RUN_DIT:-0}"
RUN_ENSEMBLE="${RUN_ENSEMBLE:-0}"
VIDEOLLM_SCORES_JSONL="${VIDEOLLM_SCORES_JSONL:-}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

FEATURES_H5="${FEATURES_H5:-$BASE_RUN/motionbench_features.h5}"
METADATA_JSONL="${METADATA_JSONL:-$BASE_RUN/motionbench_features_metadata.jsonl}"

run_step() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date --iso-8601=seconds)] DONE $name"
}

echo "run_tag=$RUN_TAG"
echo "result_dir=$RESULT_DIR"
echo "features_h5=$FEATURES_H5"
echo "metadata_jsonl=$METADATA_JSONL"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

run_step stable_probe \
  python -u experiments/motionbench_stable_probe.py \
    --features-h5 "$FEATURES_H5" \
    --metadata-jsonl "$METADATA_JSONL" \
    --output-json "$RESULT_DIR/stable_probe.json" \
    --output-md "$RESULT_DIR/stable_probe.md" \
    --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence,metadata_only \
    --modes none,uniform5,nonuniform5 \
    --label-column label \
    --folds 5 \
    --repeats 5 \
    --max-feature-dim 4096 \
    --baseline-feature pixel_grid_sequence

run_step temporal_controls \
  python -u experiments/motionbench_temporal_controls.py \
    --features-h5 "$FEATURES_H5" \
    --metadata-jsonl "$METADATA_JSONL" \
    --output-json "$RESULT_DIR/temporal_controls.json" \
    --output-md "$RESULT_DIR/temporal_controls.md" \
    --feature-names wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
    --conditions normal,first_frame_only,time_average,shuffled,reversed,uniform5,nonuniform5,metadata_only,none+crop_shift \
    --label-column label \
    --folds 5 \
    --repeats 5 \
    --max-feature-dim 4096

run_step candidate_rerank \
  python -u experiments/motionbench_candidate_rerank_probe.py \
    --features-h5 "$FEATURES_H5" \
    --metadata-jsonl "$METADATA_JSONL" \
    --output-json "$RESULT_DIR/candidate_rerank.json" \
    --output-md "$RESULT_DIR/candidate_rerank.md" \
    --output-scores-jsonl "$RESULT_DIR/wan_candidate_scores.jsonl" \
    --feature-names wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
    --modes none,uniform5,nonuniform5 \
    --text-encoder hash \
    --text-dim 1024 \
    --joint-dim 512 \
    --classifier logistic \
    --folds 5 \
    --repeats 5

run_step frame_selector \
  python -u experiments/motionbench_frame_selector_eval.py \
    --features-h5 "$FEATURES_H5" \
    --metadata-jsonl "$METADATA_JSONL" \
    --output-jsonl "$RESULT_DIR/frame_selector.jsonl" \
    --output-md "$RESULT_DIR/frame_selector.md" \
    --mode none \
    --budgets 4,8,16,32,64 \
    --selectors uniform,random,pixel_motion,flow_motion,wan_saliency,question_text,candidate_text \
    --keep-endpoints

run_step wan_only_ensemble_report \
  python -u experiments/motionbench_videollm_ensemble.py \
    --wan-scores-jsonl "$RESULT_DIR/wan_candidate_scores.jsonl" \
    --output-json "$RESULT_DIR/ensemble_wan_only.json" \
    --output-md "$RESULT_DIR/ensemble_wan_only.md" \
    --wan-mode none \
    --wan-feature wan_vae_grid_sequence

if [[ "$RUN_ENSEMBLE" == "1" ]]; then
  if [[ -z "$VIDEOLLM_SCORES_JSONL" ]]; then
    echo "RUN_ENSEMBLE=1 requires VIDEOLLM_SCORES_JSONL" >&2
    exit 2
  fi
  run_step videollm_ensemble \
    python -u experiments/motionbench_videollm_ensemble.py \
      --wan-scores-jsonl "$RESULT_DIR/wan_candidate_scores.jsonl" \
      --videollm-scores-jsonl "$VIDEOLLM_SCORES_JSONL" \
      --output-json "$RESULT_DIR/ensemble_videollm_wan.json" \
      --output-md "$RESULT_DIR/ensemble_videollm_wan.md" \
      --wan-mode none \
      --wan-feature wan_vae_grid_sequence \
      --normalizations zscore,rank,raw \
      --alphas 0,0.25,0.5,0.75,1.0 \
      --betas 0,0.25,0.5,0.75,1.0
fi

if [[ "$RUN_DIT" == "1" ]]; then
  run_step dit_qa_probe \
    python -u experiments/motionbench_dit_qa_probe.py \
      --features-h5 "$FEATURES_H5" \
      --metadata-jsonl "$METADATA_JSONL" \
      --output-json "$RESULT_DIR/dit_qa_probe.json" \
      --output-md "$RESULT_DIR/dit_qa_probe.md" \
      --output-scores-jsonl "$RESULT_DIR/dit_qa_candidate_scores.jsonl" \
      --mode none \
      --feature-name wan_vae_grid_sequence \
      --layer 14 \
      --timestep 900 \
      --pooling spatial_tokens \
      --folds 5 \
      --batch-size 2 \
      --text-encoder hash \
      --device cuda:0
fi

cat > "$RESULT_DIR/README.md" <<EOF
# MotionBench Next Steps

- base_run: \`$BASE_RUN\`
- features_h5: \`$FEATURES_H5\`
- metadata_jsonl: \`$METADATA_JSONL\`
- cuda_visible_devices: \`$CUDA_VISIBLE_DEVICES\`
- run_dit: \`$RUN_DIT\`
- run_ensemble: \`$RUN_ENSEMBLE\`

## Outputs

- \`stable_probe.md\`
- \`temporal_controls.md\`
- \`candidate_rerank.md\`
- \`wan_candidate_scores.jsonl\`
- \`frame_selector.md\`
- \`ensemble_wan_only.md\`
- optional: \`ensemble_videollm_wan.md\`
- optional: \`dit_qa_probe.md\`
EOF

echo "[$(date --iso-8601=seconds)] ALL_DONE"
