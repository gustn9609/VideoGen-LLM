#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
MOTIONBENCH_JSONL="${MOTIONBENCH_JSONL:-$ROOT/results/motionbench_real_20260509_062454/motionbench_subset.jsonl}"
RUN_TAG="${RUN_TAG:-motionbench_coarse_pooling_ablation_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
GPU="${GPU:-6}"
MODES="${MODES:-none,high_motion}"
FRAMES="${FRAMES:-16}"
PROBE_FOLDS="${PROBE_FOLDS:-5}"
PROBE_REPEATS="${PROBE_REPEATS:-3}"
TEXT_ENCODER="${TEXT_ENCODER:-hash}"
TEXT_DIM="${TEXT_DIM:-1024}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-0.01,0.1,1,10,100}"
SAMPLE_PER_TYPE="${SAMPLE_PER_TYPE:-8,16,24}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$ROOT"

FEATURE_NAMES="text_only,wan_vae_grid_1x1,wan_vae_grid_2x2,wan_vae_grid_4x4,wan_vae_grid_8x8,wan_vae_grid_16x16"

cat > "$RESULT_DIR/README.md" <<EOF
# MotionBench Coarse Pooling Ablation

- motionbench_jsonl: \`$MOTIONBENCH_JSONL\`
- frames: \`$FRAMES\`
- modes: \`$MODES\`
- ridge_alphas: \`$RIDGE_ALPHAS\`
- sample_per_type: \`$SAMPLE_PER_TYPE\`
EOF

{
  echo "[$(date --iso-8601=seconds)] START coarse pooling ablation gpu=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" \
  ROOT="$ROOT" \
  RESULT_DIR="$RESULT_DIR" \
  MOTIONBENCH_JSONL="$MOTIONBENCH_JSONL" \
  MODES="$MODES" \
  FRAMES="$FRAMES" \
  FEATURE_NAMES="$FEATURE_NAMES" \
  TEXT_ENCODER="$TEXT_ENCODER" \
  TEXT_DIM="$TEXT_DIM" \
  PROBE_FOLDS="$PROBE_FOLDS" \
  PROBE_REPEATS="$PROBE_REPEATS" \
  RIDGE_ALPHAS="$RIDGE_ALPHAS" \
  SAMPLE_PER_TYPE="$SAMPLE_PER_TYPE" \
  bash -lc '
    set -euo pipefail
    cd "$ROOT"
    source .venv/bin/activate
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

    FEATURES="$RESULT_DIR/features.h5"
    META="$RESULT_DIR/metadata.jsonl"
    python -u experiments/extract_wan_features.py \
      --video-jsonl "$MOTIONBENCH_JSONL" \
      --output-h5 "$FEATURES" \
      --metadata-output "$META" \
      --feature-types wan_vae_grid_1x1,wan_vae_grid_2x2,wan_vae_grid_4x4,wan_vae_grid_8x8,wan_vae_grid_16x16 \
      --num-frames "$FRAMES" \
      --height 128 \
      --width 128 \
      --batch-size 2 \
      --device cuda:0 \
      --lowfps-modes "$MODES"

    ONE_DIM=$(python - <<PY
import h5py, numpy as np
with h5py.File("$FEATURES", "r") as h5:
    shape = h5["none"]["wan_vae_grid_1x1"].shape
    print(int(np.prod(shape[1:])))
PY
)
    echo "one_by_one_raw_dim=$ONE_DIM" | tee "$RESULT_DIR/dimensions.txt"

    python -u experiments/motionbench_candidate_rerank_probe.py \
      --features-h5 "$FEATURES" \
      --metadata-jsonl "$META" \
      --output-json "$RESULT_DIR/controlled_random_same_dim.json" \
      --output-md "$RESULT_DIR/controlled_random_same_dim.md" \
      --output-scores-jsonl "$RESULT_DIR/controlled_random_same_dim_scores.jsonl" \
      --feature-names "$FEATURE_NAMES" \
      --modes "$MODES" \
      --text-encoder "$TEXT_ENCODER" \
      --text-dim "$TEXT_DIM" \
      --joint-dim "$ONE_DIM" \
      --video-reduction random \
      --classifier ridge \
      --ridge-alpha 1 \
      --folds "$PROBE_FOLDS" \
      --repeats "$PROBE_REPEATS"

    python -u experiments/motionbench_candidate_rerank_probe.py \
      --features-h5 "$FEATURES" \
      --metadata-jsonl "$META" \
      --output-json "$RESULT_DIR/controlled_pca_same_dim.json" \
      --output-md "$RESULT_DIR/controlled_pca_same_dim.md" \
      --output-scores-jsonl "$RESULT_DIR/controlled_pca_same_dim_scores.jsonl" \
      --feature-names "$FEATURE_NAMES" \
      --modes "$MODES" \
      --text-encoder "$TEXT_ENCODER" \
      --text-dim "$TEXT_DIM" \
      --joint-dim "$ONE_DIM" \
      --video-reduction pca \
      --classifier ridge \
      --ridge-alpha 1 \
      --folds "$PROBE_FOLDS" \
      --repeats "$PROBE_REPEATS"

    python -u experiments/motionbench_candidate_rerank_probe.py \
      --features-h5 "$FEATURES" \
      --metadata-jsonl "$META" \
      --output-json "$RESULT_DIR/controlled_raw_ridge.json" \
      --output-md "$RESULT_DIR/controlled_raw_ridge.md" \
      --output-scores-jsonl "$RESULT_DIR/controlled_raw_ridge_scores.jsonl" \
      --feature-names "$FEATURE_NAMES" \
      --modes "$MODES" \
      --text-encoder "$TEXT_ENCODER" \
      --text-dim "$TEXT_DIM" \
      --joint-dim 0 \
      --video-reduction none \
      --classifier ridge \
      --ridge-alpha 1 \
      --folds "$PROBE_FOLDS" \
      --repeats "$PROBE_REPEATS"

    IFS="," read -r -a ALPHAS <<< "$RIDGE_ALPHAS"
    for alpha in "${ALPHAS[@]}"; do
      safe_alpha="${alpha//./p}"
      python -u experiments/motionbench_candidate_rerank_probe.py \
        --features-h5 "$FEATURES" \
        --metadata-jsonl "$META" \
        --output-json "$RESULT_DIR/ridge_alpha_${safe_alpha}.json" \
        --output-md "$RESULT_DIR/ridge_alpha_${safe_alpha}.md" \
        --output-scores-jsonl "$RESULT_DIR/ridge_alpha_${safe_alpha}_scores.jsonl" \
        --feature-names "$FEATURE_NAMES" \
        --modes "$MODES" \
        --text-encoder "$TEXT_ENCODER" \
        --text-dim "$TEXT_DIM" \
        --joint-dim "$ONE_DIM" \
        --video-reduction pca \
        --classifier ridge \
        --ridge-alpha "$alpha" \
        --folds "$PROBE_FOLDS" \
        --repeats "$PROBE_REPEATS"
    done

    IFS="," read -r -a SIZES <<< "$SAMPLE_PER_TYPE"
    for size in "${SIZES[@]}"; do
      python -u experiments/motionbench_candidate_rerank_probe.py \
        --features-h5 "$FEATURES" \
        --metadata-jsonl "$META" \
        --output-json "$RESULT_DIR/sample_per_type_${size}.json" \
        --output-md "$RESULT_DIR/sample_per_type_${size}.md" \
        --output-scores-jsonl "$RESULT_DIR/sample_per_type_${size}_scores.jsonl" \
        --feature-names "$FEATURE_NAMES" \
        --modes "$MODES" \
        --text-encoder "$TEXT_ENCODER" \
        --text-dim "$TEXT_DIM" \
        --joint-dim "$ONE_DIM" \
        --video-reduction pca \
        --classifier ridge \
        --ridge-alpha 1 \
        --max-rows-per-type "$size" \
        --folds "$PROBE_FOLDS" \
        --repeats "$PROBE_REPEATS"
    done

    python -u experiments/motionbench_coarse_pooling_report.py \
      --result-dir "$RESULT_DIR" \
      --output-md "$RESULT_DIR/coarse_pooling_report.md" \
      --output-json "$RESULT_DIR/coarse_pooling_report.json"
  '
  echo "[$(date --iso-8601=seconds)] DONE coarse pooling ablation"
} > "$LOG_DIR/coarse_pooling.log" 2>&1

echo "Report: $RESULT_DIR/coarse_pooling_report.md"
