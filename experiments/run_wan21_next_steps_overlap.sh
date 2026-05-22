#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
RUN_TAG="${RUN_TAG:-next_overlap_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/wan21_next_steps_$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | sort -t',' -k2,2n -k3,3n \
        | head -n1 \
        | cut -d',' -f1 \
        | tr -d ' '
    )"
  else
    CUDA_VISIBLE_DEVICES="0"
  fi
  export CUDA_VISIBLE_DEVICES
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

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

run_step stress_fair_controls_smoke \
  python -u experiments/wan_stress_fair_baselines.py \
    --seeds 1 \
    --train-base 4 \
    --test-base 2 \
    --token-budgets 8,16 \
    --classifiers linear \
    --batch-size 2 \
    --device cuda:0 \
    --skip-dit \
    --output-json "$RESULT_DIR/wan_stress_fair_controls_smoke.json" \
    --output-md "$RESULT_DIR/wan_stress_fair_controls_smoke.md"

run_step lowfps_aug_controls_smoke \
  python -u experiments/wan_lowfps_aug_experiments.py \
    --seeds 1 \
    --train-base 4 \
    --test-base 2 \
    --batch-size 2 \
    --device cuda:0 \
    --skip-dit \
    --augmentation-modes uniform5,nonuniform5,speed0.5,speed2,repeat8,none+blur,none+jpeg45,none+crop_shift,reverse,shuffle \
    --output-json "$RESULT_DIR/wan_lowfps_aug_controls_smoke.json" \
    --output-md "$RESULT_DIR/wan_lowfps_aug_controls_smoke.md"

run_step make_synthetic_jsonl \
  python -u experiments/wan_make_synthetic_video_jsonl.py \
    --output-dir "$RESULT_DIR/synthetic_dataset" \
    --train-per-class 4 \
    --test-per-class 2 \
    --frames 24 \
    --height 128 \
    --width 128

run_step extract_cached_features \
  python -u experiments/extract_wan_features.py \
    --video-jsonl "$RESULT_DIR/synthetic_dataset/videos.jsonl" \
    --output-h5 "$RESULT_DIR/synthetic_wan_features.h5" \
    --metadata-output "$RESULT_DIR/synthetic_wan_features_metadata.jsonl" \
    --feature-types wan_vae_grid_sequence,wan_vae_global_sequence,pixel_grid_sequence,flow_grid_sequence \
    --num-frames 17 \
    --height 128 \
    --width 128 \
    --batch-size 2 \
    --device cuda:0 \
    --lowfps-modes none,uniform5,nonuniform5,speed0.5,speed2,repeat8,reverse,shuffle,none+blur,none+jpeg45,none+crop_shift

run_step real_probe_cached_features \
  python -u experiments/wan_real_probe.py \
    --features-h5 "$RESULT_DIR/synthetic_wan_features.h5" \
    --metadata-jsonl "$RESULT_DIR/synthetic_wan_features_metadata.jsonl" \
    --output-json "$RESULT_DIR/wan_real_probe_synthetic.json" \
    --output-md "$RESULT_DIR/wan_real_probe_synthetic.md" \
    --feature-names wan_vae_global_sequence,wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence \
    --label-column label \
    --task-column task \
    --split-column split \
    --classifiers linear \
    --max-feature-dim 4096

run_step frame_selector_cached_features \
  python -u experiments/wan_frame_selector.py \
    --features-h5 "$RESULT_DIR/synthetic_wan_features.h5" \
    --metadata-jsonl "$RESULT_DIR/synthetic_wan_features_metadata.jsonl" \
    --output-jsonl "$RESULT_DIR/wan_frame_selector_synthetic.jsonl" \
    --feature-name wan_vae_grid_sequence \
    --budgets 4,8,16 \
    --keep-endpoints

run_step adapter_probe_cached_features \
  python -u experiments/wan_adapter_probe.py \
    --features-h5 "$RESULT_DIR/synthetic_wan_features.h5" \
    --metadata-jsonl "$RESULT_DIR/synthetic_wan_features_metadata.jsonl" \
    --output-json "$RESULT_DIR/wan_adapter_probe_synthetic.json" \
    --output-md "$RESULT_DIR/wan_adapter_probe_synthetic.md" \
    --feature-name wan_vae_grid_sequence \
    --modes none,uniform5,nonuniform5 \
    --label-column label \
    --split-column split \
    --epochs 3 \
    --batch-size 4 \
    --hidden-dim 64 \
    --output-tokens 8 \
    --tokens-per-frame 2 \
    --device cuda:0

cat > "$RESULT_DIR/README.md" <<EOF
# Wan 2.1 Next-Step Overlap Run

- run_tag: \`$RUN_TAG\`
- host: \`$(hostname)\`
- cuda_visible_devices: \`$CUDA_VISIBLE_DEVICES\`
- result_dir: \`$RESULT_DIR\`

This run implements and exercises the next-step roadmap:

- same-token stress baselines with spatial/temporal/reverse/shuffle controls
- temporal augmentation modes: uniform, nonuniform, speed jitter, repeats, blur, JPEG, crop shift, reverse, shuffle
- JSONL -> H5 Wan/pixel/flow feature extraction with richer metadata
- cached-feature real benchmark probing scaffold
- feature-motion frame selector
- cached Wan grid motion-token adapter probe

## Main outputs

- \`wan_stress_fair_controls_smoke.md\`
- \`wan_lowfps_aug_controls_smoke.md\`
- \`synthetic_wan_features.h5\`
- \`synthetic_wan_features_metadata.jsonl\`
- \`wan_real_probe_synthetic.md\`
- \`wan_frame_selector_synthetic.jsonl\`
- \`wan_adapter_probe_synthetic.md\`
- \`logs/\`
EOF

echo "[$(date --iso-8601=seconds)] ALL_DONE"
