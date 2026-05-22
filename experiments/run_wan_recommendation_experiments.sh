#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/wan_recommendation_overlap_$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "run_tag=$RUN_TAG"
echo "root=$ROOT"
echo "result_dir=$RESULT_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
print({"torch_cuda_available": torch.cuda.is_available(), "torch_cuda_device_count": torch.cuda.device_count()})
if torch.cuda.is_available():
    print({"visible_device_name": torch.cuda.get_device_name(0)})
PY

run_step() {
  local name="$1"
  shift
  echo "[$(date --iso-8601=seconds)] START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date --iso-8601=seconds)] DONE $name"
}

run_step adapter_smoke \
  python -u experiments/wan_motion_adapter.py

run_step stress_fair_baselines_extended \
  python -u experiments/wan_stress_fair_baselines.py \
    --seeds 3 \
    --train-base 24 \
    --test-base 12 \
    --token-budgets 8,16,32,64 \
    --classifiers linear,mlp \
    --batch-size 4 \
    --device cuda:0 \
    --output-json "$RESULT_DIR/wan_stress_fair_baselines_extended.json" \
    --output-md "$RESULT_DIR/wan_stress_fair_baselines_extended.md"

run_step lowfps_aug_full \
  python -u experiments/wan_lowfps_aug_experiments.py \
    --seeds 3 \
    --train-base 32 \
    --test-base 16 \
    --batch-size 4 \
    --device cuda:0 \
    --output-json "$RESULT_DIR/wan_lowfps_aug_full.json" \
    --output-md "$RESULT_DIR/wan_lowfps_aug_full.md"

run_step temporal_reasoning_full \
  python -u experiments/wan_temporal_reasoning_experiments.py \
    --seeds 3 \
    --train-base 32 \
    --test-base 16 \
    --batch-size 4 \
    --device cuda:0 \
    --output-json "$RESULT_DIR/wan_temporal_reasoning_full.json" \
    --output-md "$RESULT_DIR/wan_temporal_reasoning_full.md"

run_step next_dit_sweep_extended \
  python -u experiments/wan_next_experiments.py \
    --pooling-seeds 3 \
    --train-base 32 \
    --test-base 16 \
    --batch-size 4 \
    --device cuda:0 \
    --dit-tasks direction4,shuffle \
    --dit-layers 10,12,14,16,18 \
    --dit-timesteps 700,800,900,950 \
    --dit-pooling token_mean,temporal_sequence,temporal_delta \
    --output-json "$RESULT_DIR/wan_next_dit_sweep_extended.json" \
    --output-md "$RESULT_DIR/wan_next_dit_sweep_extended.md"

cat > "$RESULT_DIR/README.md" <<EOF
# Wan Recommendation Overlap Run

- run_tag: \`$RUN_TAG\`
- started_on: \`$(hostname)\`
- cuda_visible_devices: \`$CUDA_VISIBLE_DEVICES\`
- result_dir: \`$RESULT_DIR\`

This run executes the implemented recommendation experiments while another
8-GPU job may be active. The new run is restricted to the visible device above.

## Outputs

- \`wan_stress_fair_baselines_extended.md\`
- \`wan_lowfps_aug_full.md\`
- \`wan_temporal_reasoning_full.md\`
- \`wan_next_dit_sweep_extended.md\`
- \`logs/\`
EOF

echo "[$(date --iso-8601=seconds)] ALL_DONE"
