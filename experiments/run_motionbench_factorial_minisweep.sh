#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
MOTIONBENCH_JSONL="${MOTIONBENCH_JSONL:-$ROOT/results/motionbench_real_20260509_062454/motionbench_subset.jsonl}"
RUN_TAG="${RUN_TAG:-motionbench_factorial_minisweep_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5}"
POOLS="${POOLS:-1,2,4}"
FRAMES_LIST="${FRAMES_LIST:-8,16}"
SAMPLING_MODES="${SAMPLING_MODES:-none,high_motion,camera_comp,high_motion+camera_comp}"
PROBE_FOLDS="${PROBE_FOLDS:-5}"
PROBE_REPEATS="${PROBE_REPEATS:-3}"
TEXT_ENCODER="${TEXT_ENCODER:-hash}"
TEXT_DIM="${TEXT_DIM:-1024}"
JOINT_DIM="${JOINT_DIM:-512}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
IFS=',' read -r -a POOL_ARR <<< "$POOLS"
IFS=',' read -r -a FRAME_ARR <<< "$FRAMES_LIST"

cat > "$RESULT_DIR/README.md" <<EOF
# MotionBench Factorial Mini-Sweep

- motionbench_jsonl: \`$MOTIONBENCH_JSONL\`
- pools: \`$POOLS\`
- frames: \`$FRAMES_LIST\`
- sampling_modes: \`$SAMPLING_MODES\`
- text_encoder: \`$TEXT_ENCODER\`
- joint_dim: \`$JOINT_DIM\`
EOF

probe_all_transforms() {
  local features_h5="$1"
  local metadata_jsonl="$2"
  local out_dir="$3"
  local feature_names="$4"
  local modes="$5"
  for transform in full shuffled reversed first_frame_only; do
    python -u experiments/motionbench_candidate_rerank_probe.py \
      --features-h5 "$features_h5" \
      --metadata-jsonl "$metadata_jsonl" \
      --output-json "$out_dir/probe_${transform}.json" \
      --output-md "$out_dir/probe_${transform}.md" \
      --output-scores-jsonl "$out_dir/probe_${transform}_scores.jsonl" \
      --feature-names "$feature_names" \
      --modes "$modes" \
      --temporal-transform "$transform" \
      --text-encoder "$TEXT_ENCODER" \
      --text-dim "$TEXT_DIM" \
      --joint-dim "$JOINT_DIM" \
      --video-reduction random \
      --classifier logistic \
      --folds "$PROBE_FOLDS" \
      --repeats "$PROBE_REPEATS"
  done
}
export -f probe_all_transforms

launch_job() {
  local gpu="$1"
  local pool="$2"
  local frames="$3"
  local name="pool_${pool}_frames_${frames}"
  local job_dir="$RESULT_DIR/$name"
  local wan_feature="wan_vae_grid_${pool}x${pool}"
  mkdir -p "$job_dir"
  {
    echo "[$(date --iso-8601=seconds)] START $name gpu=$gpu"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    ROOT="$ROOT" \
    RESULT_DIR="$job_dir" \
    MOTIONBENCH_JSONL="$MOTIONBENCH_JSONL" \
    SAMPLING_MODES="$SAMPLING_MODES" \
    PROBE_FOLDS="$PROBE_FOLDS" \
    PROBE_REPEATS="$PROBE_REPEATS" \
    TEXT_ENCODER="$TEXT_ENCODER" \
    TEXT_DIM="$TEXT_DIM" \
    JOINT_DIM="$JOINT_DIM" \
    POOL="$pool" \
    FRAMES="$frames" \
    WAN_FEATURE="$wan_feature" \
      bash -lc '
        cd "$ROOT"
        source .venv/bin/activate
        export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
        FEATURES="$RESULT_DIR/features.h5"
        META="$RESULT_DIR/metadata.jsonl"
        python -u experiments/extract_wan_features.py \
          --video-jsonl "$MOTIONBENCH_JSONL" \
          --output-h5 "$FEATURES" \
          --metadata-output "$META" \
          --feature-types "$WAN_FEATURE,pixel_grid_sequence,flow_grid_sequence" \
          --num-frames "$FRAMES" \
          --height 128 \
          --width 128 \
          --pixel-grid-hw "$POOL" \
          --batch-size 2 \
          --device cuda:0 \
          --lowfps-modes "$SAMPLING_MODES"
        probe_all_transforms "$FEATURES" "$META" "$RESULT_DIR" \
          "text_only,$WAN_FEATURE,pixel_grid_sequence,flow_grid_sequence" \
          "$SAMPLING_MODES"
      '
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
      echo "[$(date --iso-8601=seconds)] DONE $name"
    else
      echo "[$(date --iso-8601=seconds)] FAILED $name status=$status"
      exit "$status"
    fi
  } > "$LOG_DIR/${name}.log" 2>&1 &
  local pid=$!
  echo -e "$pid\t$gpu\t$name\t$job_dir" | tee -a "$RESULT_DIR/pids.tsv"
}

echo -e "pid\tgpu\tname\tjob_dir" > "$RESULT_DIR/pids.tsv"
echo "result_dir=$RESULT_DIR"

job_index=0
for pool in "${POOL_ARR[@]}"; do
  for frames in "${FRAME_ARR[@]}"; do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    launch_job "$gpu" "$pool" "$frames"
    job_index=$((job_index + 1))
  done
done

failures=0
if [[ "$WAIT_FOR_JOBS" == "1" ]]; then
  while IFS=$'\t' read -r pid gpu name job_dir; do
    if [[ "$pid" == "pid" ]]; then
      continue
    fi
    if wait "$pid"; then
      echo "[$(date --iso-8601=seconds)] completed $name gpu=$gpu"
    else
      status=$?
      echo "[$(date --iso-8601=seconds)] failed $name gpu=$gpu status=$status"
      failures=$((failures + 1))
    fi
  done < "$RESULT_DIR/pids.tsv"
fi

if [[ "$failures" -gt 0 ]]; then
  exit 1
fi

python -u experiments/motionbench_factorial_report.py \
  --result-dir "$RESULT_DIR" \
  --output-md "$RESULT_DIR/factorial_report.md" \
  --output-csv "$RESULT_DIR/factorial_report.csv" \
  --output-json "$RESULT_DIR/factorial_report.json" \
  --pools "$POOLS" \
  --frames "$FRAMES_LIST" \
  --samplings "$SAMPLING_MODES"

echo "Report: $RESULT_DIR/factorial_report.md"
