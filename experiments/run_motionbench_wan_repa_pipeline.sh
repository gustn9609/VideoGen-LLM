#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/hskim/VideoGen-LLM}
cd "$ROOT"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

RUN_TAG=${RUN_TAG:-motionbench_wan_repa_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"$ROOT/results/$RUN_TAG"}
mkdir -p "$OUT_DIR/logs"

WAN_RUN_DIR=${WAN_RUN_DIR:-"$ROOT/results/motionbench_factorial_minisweep_20260510_151512/pool_1_frames_8"}
WAN_FEATURES_H5=${WAN_FEATURES_H5:-"$WAN_RUN_DIR/features.h5"}
WAN_METADATA_JSONL=${WAN_METADATA_JSONL:-"$WAN_RUN_DIR/metadata.jsonl"}
BASE_SCORES_JSONL=${BASE_SCORES_JSONL:-"$WAN_RUN_DIR/probe_full_scores.jsonl"}

MODE=${MODE:-high_motion+camera_comp}
WAN_FEATURE=${WAN_FEATURE:-wan_vae_grid_1x1}
SAFE_WAN_FEATURE=$(printf '%s' "$WAN_FEATURE" | tr -c '[:alnum:]' '_')
MOFT_FEATURE=${MOFT_FEATURE:-wan_moft_${SAFE_WAN_FEATURE}}
TRD_FEATURE=${TRD_FEATURE:-wan_trd_${SAFE_WAN_FEATURE}_pooled}
MOREPA_FEATURE=${MOREPA_FEATURE:-wan_morepa_${SAFE_WAN_FEATURE}}

DEVICE=${DEVICE:-cuda:0}
PROBE_CLASSIFIER=${PROBE_CLASSIFIER:-ridge}
FOLDS=${FOLDS:-5}
REPEATS=${REPEATS:-1}
SEED=${SEED:-123}

echo "Output: $OUT_DIR"
echo "Wan features: $WAN_FEATURES_H5"
echo "Metadata: $WAN_METADATA_JSONL"
echo "Mode: $MODE"

python -u experiments/motionbench_motion_teacher.py \
  --video-jsonl "$WAN_METADATA_JSONL" \
  --output-h5 "$OUT_DIR/motion_teacher.h5" \
  --metadata-output "$OUT_DIR/motion_teacher_metadata.jsonl" \
  --input-mode-filter "$MODE" \
  --cached-features-h5 "$WAN_FEATURES_H5" \
  --flow-feature flow_grid_sequence \
  --lowfps-modes "$MODE" \
  --num-frames 8 \
  --height 128 \
  --width 128 \
  --segments 4 \
  --grid-hw 4 \
  --seed "$SEED" \
  2>&1 | tee "$OUT_DIR/logs/01_motion_teacher.log"

python -u experiments/motionbench_wan_moft.py \
  --features-h5 "$WAN_FEATURES_H5" \
  --metadata-jsonl "$WAN_METADATA_JSONL" \
  --output-h5 "$OUT_DIR/wan_moft_features.h5" \
  --output-metadata-jsonl "$OUT_DIR/wan_moft_metadata.jsonl" \
  --feature-names "$WAN_FEATURE" \
  --modes "$MODE" \
  --content-components 32 \
  --topk 256 \
  --flow-feature flow_grid_sequence \
  --report-json "$OUT_DIR/wan_moft_report.json" \
  --report-md "$OUT_DIR/wan_moft_report.md" \
  2>&1 | tee "$OUT_DIR/logs/02_wan_moft.log"

python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 "$OUT_DIR/wan_moft_features.h5" \
  --metadata-jsonl "$OUT_DIR/wan_moft_metadata.jsonl" \
  --output-json "$OUT_DIR/wan_moft_probe.json" \
  --output-md "$OUT_DIR/wan_moft_probe.md" \
  --output-scores-jsonl "$OUT_DIR/wan_moft_scores.jsonl" \
  --feature-names "$MOFT_FEATURE" \
  --modes "$MODE" \
  --classifier "$PROBE_CLASSIFIER" \
  --video-reduction random \
  --joint-dim 512 \
  --folds "$FOLDS" \
  --repeats "$REPEATS" \
  --seed "$SEED" \
  2>&1 | tee "$OUT_DIR/logs/03_wan_moft_probe.log"

python -u experiments/motionbench_relation_distill.py \
  --wan-features-h5 "$WAN_FEATURES_H5" \
  --wan-metadata-jsonl "$WAN_METADATA_JSONL" \
  --teacher-h5 "$OUT_DIR/motion_teacher.h5" \
  --output-h5 "$OUT_DIR/wan_trd_features.h5" \
  --output-metadata-jsonl "$OUT_DIR/wan_trd_metadata.jsonl" \
  --feature-names "$WAN_FEATURE" \
  --teacher-feature motion_teacher_segments \
  --modes "$MODE" \
  --segments 4 \
  --align-dim 128 \
  --hidden-dim 256 \
  --epochs 120 \
  --batch-size 64 \
  --device "$DEVICE" \
  --report-json "$OUT_DIR/wan_trd_report.json" \
  --report-md "$OUT_DIR/wan_trd_report.md" \
  --seed "$SEED" \
  2>&1 | tee "$OUT_DIR/logs/04_wan_trd.log"

python -u experiments/motionbench_candidate_rerank_probe.py \
  --features-h5 "$OUT_DIR/wan_trd_features.h5" \
  --metadata-jsonl "$OUT_DIR/wan_trd_metadata.jsonl" \
  --output-json "$OUT_DIR/wan_trd_probe.json" \
  --output-md "$OUT_DIR/wan_trd_probe.md" \
  --output-scores-jsonl "$OUT_DIR/wan_trd_scores.jsonl" \
  --feature-names "$TRD_FEATURE" \
  --modes "$MODE" \
  --classifier "$PROBE_CLASSIFIER" \
  --video-reduction random \
  --joint-dim 512 \
  --folds "$FOLDS" \
  --repeats "$REPEATS" \
  --seed "$SEED" \
  2>&1 | tee "$OUT_DIR/logs/05_wan_trd_probe.log"

python -u experiments/motionbench_wan_morepa_train.py \
  --wan-features-h5 "$WAN_FEATURES_H5" \
  --wan-metadata-jsonl "$WAN_METADATA_JSONL" \
  --teacher-h5 "$OUT_DIR/motion_teacher.h5" \
  --output-json "$OUT_DIR/wan_morepa.json" \
  --output-md "$OUT_DIR/wan_morepa.md" \
  --output-scores-jsonl "$OUT_DIR/wan_morepa_scores.jsonl" \
  --feature-names "$WAN_FEATURE" \
  --teacher-feature motion_teacher_summary \
  --modes "$MODE" \
  --align-dim 128 \
  --hidden-dim 256 \
  --epochs 80 \
  --batch-size 64 \
  --folds "$FOLDS" \
  --repeats "$REPEATS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  2>&1 | tee "$OUT_DIR/logs/06_wan_morepa.log"

python -u experiments/motionbench_residual_ensemble.py \
  --system "text=$BASE_SCORES_JSONL,mode=$MODE,feature=text_only" \
  --system "pixel=$BASE_SCORES_JSONL,mode=$MODE,feature=pixel_grid_sequence" \
  --system "flow=$BASE_SCORES_JSONL,mode=$MODE,feature=flow_grid_sequence" \
  --system "raw_wan=$BASE_SCORES_JSONL,mode=$MODE,feature=$WAN_FEATURE" \
  --system "wan_moft=$OUT_DIR/wan_moft_scores.jsonl,mode=$MODE,feature=$MOFT_FEATURE" \
  --system "wan_trd=$OUT_DIR/wan_trd_scores.jsonl,mode=$MODE,feature=$TRD_FEATURE" \
  --system "wan_morepa=$OUT_DIR/wan_morepa_scores.jsonl,mode=$MODE,feature=$MOREPA_FEATURE" \
  --output-json "$OUT_DIR/residual_ensemble.json" \
  --output-md "$OUT_DIR/residual_ensemble.md" \
  --score-normalization zscore \
  --text-system text \
  --folds "$FOLDS" \
  --seed "$SEED" \
  2>&1 | tee "$OUT_DIR/logs/07_residual_ensemble.log"

echo "Done: $OUT_DIR"
