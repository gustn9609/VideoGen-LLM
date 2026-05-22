#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
RUN_TAG="${RUN_TAG:-motionbench_real_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
PER_QTYPE="${PER_QTYPE:-24}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

run_step() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date --iso-8601=seconds)] DONE $name"
}

echo "run_tag=$RUN_TAG"
echo "result_dir=$RESULT_DIR"
echo "per_question_type=$PER_QTYPE"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
print({"cuda": torch.cuda.is_available(), "visible_devices": torch.cuda.device_count()})
if torch.cuda.is_available():
    print({"device": torch.cuda.get_device_name(0)})
PY

run_step make_motionbench_subset \
  python -u experiments/make_motionbench_subset.py \
    --output-jsonl "$RESULT_DIR/motionbench_subset.jsonl" \
    --download-dir "$RESULT_DIR/hf_download" \
    --per-question-type "$PER_QTYPE" \
    --seed 2026

run_step extract_features \
  python -u experiments/extract_wan_features.py \
    --video-jsonl "$RESULT_DIR/motionbench_subset.jsonl" \
    --output-h5 "$RESULT_DIR/motionbench_features.h5" \
    --metadata-output "$RESULT_DIR/motionbench_features_metadata.jsonl" \
    --feature-types wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
    --num-frames 17 \
    --height 128 \
    --width 128 \
    --batch-size 2 \
    --device cuda:0 \
    --lowfps-modes none,uniform5,nonuniform5

run_step category_probe \
  python -u experiments/wan_real_probe.py \
    --features-h5 "$RESULT_DIR/motionbench_features.h5" \
    --metadata-jsonl "$RESULT_DIR/motionbench_features_metadata.jsonl" \
    --output-json "$RESULT_DIR/motionbench_question_type_probe.json" \
    --output-md "$RESULT_DIR/motionbench_question_type_probe.md" \
    --feature-names wan_vae_global_sequence,wan_vae_grid_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
    --label-column label \
    --split-column split \
    --classifiers linear \
    --max-feature-dim 4096

run_step answer_letter_probe \
  python -u experiments/wan_real_probe.py \
    --features-h5 "$RESULT_DIR/motionbench_features.h5" \
    --metadata-jsonl "$RESULT_DIR/motionbench_features_metadata.jsonl" \
    --output-json "$RESULT_DIR/motionbench_answer_letter_probe.json" \
    --output-md "$RESULT_DIR/motionbench_answer_letter_probe.md" \
    --feature-names wan_vae_global_sequence,wan_vae_grid_sequence,wan_vae_global_delta,pixel_grid_sequence,flow_grid_sequence \
    --label-column answer \
    --task-column question_type \
    --split-column split \
    --classifiers linear \
    --max-feature-dim 4096

run_step adapter_probe \
  python -u experiments/wan_adapter_probe.py \
    --features-h5 "$RESULT_DIR/motionbench_features.h5" \
    --metadata-jsonl "$RESULT_DIR/motionbench_features_metadata.jsonl" \
    --output-json "$RESULT_DIR/motionbench_adapter_probe.json" \
    --output-md "$RESULT_DIR/motionbench_adapter_probe.md" \
    --feature-name wan_vae_grid_sequence \
    --modes none,uniform5,nonuniform5 \
    --label-column label \
    --split-column split \
    --epochs 8 \
    --batch-size 8 \
    --hidden-dim 96 \
    --output-tokens 8 \
    --tokens-per-frame 2 \
    --device cuda:0

cat > "$RESULT_DIR/README.md" <<EOF
# MotionBench Real Probe

- run_tag: \`$RUN_TAG\`
- per_question_type: \`$PER_QTYPE\`
- cuda_visible_devices: \`$CUDA_VISIBLE_DEVICES\`

This run uses publicly downloadable MotionBench DEV mp4 files only.

## Outputs

- \`motionbench_subset.jsonl\`
- \`motionbench_features.h5\`
- \`motionbench_features_metadata.jsonl\`
- \`motionbench_question_type_probe.md\`
- \`motionbench_answer_letter_probe.md\`
- \`motionbench_adapter_probe.md\`
- \`logs/\`
EOF

echo "[$(date --iso-8601=seconds)] ALL_DONE"
