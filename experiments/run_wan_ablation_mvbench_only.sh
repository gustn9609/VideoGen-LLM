#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
RUN_TAG="${RUN_TAG:-wan_mvbench_only_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
MVBENCH_JSONL="${MVBENCH_JSONL:?Set MVBENCH_JSONL to the prepared MVBench JSONL path}"
ABLATION_FRAMES="${ABLATION_FRAMES:-32}"
PROBE_FOLDS="${PROBE_FOLDS:-5}"
PROBE_REPEATS="${PROBE_REPEATS:-3}"
TEXT_ENCODER="${TEXT_ENCODER:-hash}"
TEXT_DIM="${TEXT_DIM:-1024}"
JOINT_DIM="${JOINT_DIM:-512}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-0}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 7 ]]; then
  echo "GPUS_CSV must contain at least 7 GPU ids; got $GPUS_CSV" >&2
  exit 2
fi

cat > "$RESULT_DIR/README.md" <<EOF
# Wan MVBench-Only Ablation Run

- run_tag: \`$RUN_TAG\`
- mvbench_jsonl: \`$MVBENCH_JSONL\`
- gpus: \`$GPUS_CSV\`
- text_encoder: \`$TEXT_ENCODER\`
- probe_folds: \`$PROBE_FOLDS\`
- probe_repeats: \`$PROBE_REPEATS\`
EOF

probe_candidates() {
  local features_h5="$1"
  local metadata_jsonl="$2"
  local out_prefix="$3"
  local feature_names="$4"
  local modes="$5"
  python -u experiments/motionbench_candidate_rerank_probe.py \
    --features-h5 "$features_h5" \
    --metadata-jsonl "$metadata_jsonl" \
    --output-json "${out_prefix}.json" \
    --output-md "${out_prefix}.md" \
    --output-scores-jsonl "${out_prefix}_scores.jsonl" \
    --feature-names "$feature_names" \
    --modes "$modes" \
    --text-encoder "$TEXT_ENCODER" \
    --text-dim "$TEXT_DIM" \
    --joint-dim "$JOINT_DIM" \
    --classifier logistic \
    --folds "$PROBE_FOLDS" \
    --repeats "$PROBE_REPEATS"
}
export -f probe_candidates

launch_job() {
  local gpu="$1"
  local name="$2"
  local command="$3"
  local job_dir="$RESULT_DIR/$name"
  mkdir -p "$job_dir"
  {
    echo "[$(date --iso-8601=seconds)] START $name gpu=$gpu"
    echo "command=$command"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    ROOT="$ROOT" \
    RESULT_DIR="$job_dir" \
    MVBENCH_JSONL="$MVBENCH_JSONL" \
    ABLATION_FRAMES="$ABLATION_FRAMES" \
    PROBE_FOLDS="$PROBE_FOLDS" \
    PROBE_REPEATS="$PROBE_REPEATS" \
    TEXT_ENCODER="$TEXT_ENCODER" \
    TEXT_DIM="$TEXT_DIM" \
    JOINT_DIM="$JOINT_DIM" \
      bash -lc "cd '$ROOT'; source .venv/bin/activate; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false; $command"
    local status=$?
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
echo "mvbench_jsonl=$MVBENCH_JSONL"

launch_job "${GPUS[0]}" mvbench_grid_resolution '
FEATURES="$RESULT_DIR/grid_resolution_features.h5"
META="$RESULT_DIR/grid_resolution_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_1x1,wan_vae_grid_2x2,wan_vae_grid_4x4,wan_vae_grid_8x8,wan_vae_grid_16x16 \
  --num-frames 17 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/grid_resolution_probe" \
  "wan_vae_grid_1x1,wan_vae_grid_2x2,wan_vae_grid_4x4,wan_vae_grid_8x8,wan_vae_grid_16x16" \
  "none"
'

launch_job "${GPUS[1]}" mvbench_temporal_8 '
FEATURES="$RESULT_DIR/temporal_8_features.h5"
META="$RESULT_DIR/temporal_8_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames 8 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/temporal_8_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none"
'

launch_job "${GPUS[2]}" mvbench_temporal_16 '
FEATURES="$RESULT_DIR/temporal_16_features.h5"
META="$RESULT_DIR/temporal_16_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames 16 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/temporal_16_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none"
'

launch_job "${GPUS[3]}" mvbench_temporal_32 '
FEATURES="$RESULT_DIR/temporal_32_features.h5"
META="$RESULT_DIR/temporal_32_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames 32 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/temporal_32_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none"
'

launch_job "${GPUS[4]}" mvbench_temporal_64 '
FEATURES="$RESULT_DIR/temporal_64_features.h5"
META="$RESULT_DIR/temporal_64_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames 64 \
  --height 128 \
  --width 128 \
  --batch-size 1 \
  --device cuda:0 \
  --lowfps-modes none
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/temporal_64_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none"
'

launch_job "${GPUS[5]}" mvbench_object_crop '
FEATURES="$RESULT_DIR/object_crop_features.h5"
META="$RESULT_DIR/object_crop_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames "$ABLATION_FRAMES" \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none,center_crop,object_crop
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/object_crop_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none,center_crop,object_crop"
'

launch_job "${GPUS[6]}" mvbench_motion_camera '
FEATURES="$RESULT_DIR/motion_camera_features.h5"
META="$RESULT_DIR/motion_camera_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MVBENCH_JSONL" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames "$ABLATION_FRAMES" \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none,high_motion,camera_comp
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/motion_camera_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none,high_motion,camera_comp"
'

echo "Launched jobs:"
cat "$RESULT_DIR/pids.tsv"
echo "Logs: $LOG_DIR"
echo "Use: tail -f $LOG_DIR/<job>.log"

if [[ "$WAIT_FOR_JOBS" == "1" ]]; then
  failures=0
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
  if [[ "$failures" -gt 0 ]]; then
    exit 1
  fi
fi
