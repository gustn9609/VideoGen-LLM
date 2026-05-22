#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
RESULT_DIR="${RESULT_DIR:?Set RESULT_DIR to an existing factorial sweep directory}"
POOLS="${POOLS:-1,2,4}"
FRAMES_LIST="${FRAMES_LIST:-8,16}"
SAMPLING_MODES="${SAMPLING_MODES:-none,high_motion,camera_comp,high_motion+camera_comp}"
CLASSIFIER="${CLASSIFIER:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1}"
PROBE_FOLDS="${PROBE_FOLDS:-5}"
PROBE_REPEATS="${PROBE_REPEATS:-3}"
TEXT_ENCODER="${TEXT_ENCODER:-hash}"
TEXT_DIM="${TEXT_DIM:-1024}"
JOINT_DIM="${JOINT_DIM:-512}"

cd "$ROOT"
mkdir -p "$RESULT_DIR/logs"
IFS=',' read -r -a POOL_ARR <<< "$POOLS"
IFS=',' read -r -a FRAME_ARR <<< "$FRAMES_LIST"

run_job() {
  local pool="$1"
  local frames="$2"
  local job_dir="$RESULT_DIR/pool_${pool}_frames_${frames}"
  local wan_feature="wan_vae_grid_${pool}x${pool}"
  {
    echo "[$(date --iso-8601=seconds)] START probes pool=$pool frames=$frames classifier=$CLASSIFIER"
    source .venv/bin/activate
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    for transform in full shuffled reversed first_frame_only; do
      python -u experiments/motionbench_candidate_rerank_probe.py \
        --features-h5 "$job_dir/features.h5" \
        --metadata-jsonl "$job_dir/metadata.jsonl" \
        --output-json "$job_dir/probe_${transform}.json" \
        --output-md "$job_dir/probe_${transform}.md" \
        --output-scores-jsonl "$job_dir/probe_${transform}_scores.jsonl" \
        --feature-names "text_only,$wan_feature,pixel_grid_sequence,flow_grid_sequence" \
        --modes "$SAMPLING_MODES" \
        --temporal-transform "$transform" \
        --text-encoder "$TEXT_ENCODER" \
        --text-dim "$TEXT_DIM" \
        --joint-dim "$JOINT_DIM" \
        --video-reduction random \
        --classifier "$CLASSIFIER" \
        --ridge-alpha "$RIDGE_ALPHA" \
        --folds "$PROBE_FOLDS" \
        --repeats "$PROBE_REPEATS"
    done
    echo "[$(date --iso-8601=seconds)] DONE probes pool=$pool frames=$frames"
  } > "$RESULT_DIR/logs/pool_${pool}_frames_${frames}_${CLASSIFIER}_probe.log" 2>&1 &
  echo -e "$!\tpool_${pool}_frames_${frames}" >> "$RESULT_DIR/probe_pids.tsv"
}

echo -e "pid\tname" > "$RESULT_DIR/probe_pids.tsv"
for pool in "${POOL_ARR[@]}"; do
  for frames in "${FRAME_ARR[@]}"; do
    run_job "$pool" "$frames"
  done
done

failures=0
while IFS=$'\t' read -r pid name; do
  if [[ "$pid" == "pid" ]]; then
    continue
  fi
  if wait "$pid"; then
    echo "[$(date --iso-8601=seconds)] completed $name"
  else
    status=$?
    echo "[$(date --iso-8601=seconds)] failed $name status=$status"
    failures=$((failures + 1))
  fi
done < "$RESULT_DIR/probe_pids.tsv"

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
