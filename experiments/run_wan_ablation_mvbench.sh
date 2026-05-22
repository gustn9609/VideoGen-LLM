#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
BASE_MOTIONBENCH="${BASE_MOTIONBENCH:-$ROOT/results/motionbench_real_20260509_062454}"
RUN_TAG="${RUN_TAG:-wan_ablation_mvbench_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/$RUN_TAG}"
LOG_DIR="$RESULT_DIR/logs"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
MOTIONBENCH_JSONL="${MOTIONBENCH_JSONL:-$BASE_MOTIONBENCH/motionbench_subset.jsonl}"
MVBENCH_JSON_DIR="${MVBENCH_JSON_DIR:-$HOME/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots/230a2d4fac8900333c61754641c7a13e069ac9c6/json}"
MVBENCH_VIDEO_ROOT="${MVBENCH_VIDEO_ROOT:-}"
PER_MVBENCH_TASK="${PER_MVBENCH_TASK:-24}"
ABLATION_FRAMES="${ABLATION_FRAMES:-32}"
PROBE_FOLDS="${PROBE_FOLDS:-5}"
PROBE_REPEATS="${PROBE_REPEATS:-3}"
TEXT_ENCODER="${TEXT_ENCODER:-hash}"
TEXT_DIM="${TEXT_DIM:-1024}"
JOINT_DIM="${JOINT_DIM:-512}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -lt 8 ]]; then
  echo "GPUS_CSV must contain at least 8 GPU ids; got $GPUS_CSV" >&2
  exit 2
fi

cat > "$RESULT_DIR/README.md" <<EOF
# Wan Ablation + MVBench Run

- run_tag: \`$RUN_TAG\`
- motionbench_jsonl: \`$MOTIONBENCH_JSONL\`
- mvbench_json_dir: \`$MVBENCH_JSON_DIR\`
- mvbench_video_root: \`$MVBENCH_VIDEO_ROOT\`
- gpus: \`$GPUS_CSV\`
- text_encoder: \`$TEXT_ENCODER\`
- probe_folds: \`$PROBE_FOLDS\`
- probe_repeats: \`$PROBE_REPEATS\`

Each GPU slot runs an independent extraction/probe chain.
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
    CUDA_VISIBLE_DEVICES="$gpu" \
    ROOT="$ROOT" \
    RESULT_DIR="$job_dir" \
    MOTIONBENCH_JSONL="$MOTIONBENCH_JSONL" \
    MVBENCH_JSON_DIR="$MVBENCH_JSON_DIR" \
    MVBENCH_VIDEO_ROOT="$MVBENCH_VIDEO_ROOT" \
    PER_MVBENCH_TASK="$PER_MVBENCH_TASK" \
    ABLATION_FRAMES="$ABLATION_FRAMES" \
    PROBE_FOLDS="$PROBE_FOLDS" \
    PROBE_REPEATS="$PROBE_REPEATS" \
    TEXT_ENCODER="$TEXT_ENCODER" \
    TEXT_DIM="$TEXT_DIM" \
    JOINT_DIM="$JOINT_DIM" \
      bash -lc "cd '$ROOT'; source .venv/bin/activate; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false; $command"
    echo "[$(date --iso-8601=seconds)] DONE $name"
  } > "$LOG_DIR/${name}.log" 2>&1 &
  local pid=$!
  echo -e "$pid\t$gpu\t$name\t$job_dir" | tee -a "$RESULT_DIR/pids.tsv"
}

echo -e "pid\tgpu\tname\tjob_dir" > "$RESULT_DIR/pids.tsv"
echo "result_dir=$RESULT_DIR"
echo "motionbench_jsonl=$MOTIONBENCH_JSONL"
echo "mvbench_video_root=$MVBENCH_VIDEO_ROOT"

launch_job "${GPUS[0]}" motionbench_grid_resolution '
FEATURES="$RESULT_DIR/grid_resolution_features.h5"
META="$RESULT_DIR/grid_resolution_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[1]}" motionbench_temporal_8 '
FEATURES="$RESULT_DIR/temporal_8_features.h5"
META="$RESULT_DIR/temporal_8_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[2]}" motionbench_temporal_16 '
FEATURES="$RESULT_DIR/temporal_16_features.h5"
META="$RESULT_DIR/temporal_16_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[3]}" motionbench_temporal_32 '
FEATURES="$RESULT_DIR/temporal_32_features.h5"
META="$RESULT_DIR/temporal_32_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[4]}" motionbench_temporal_64 '
FEATURES="$RESULT_DIR/temporal_64_features.h5"
META="$RESULT_DIR/temporal_64_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[5]}" motionbench_object_crop '
FEATURES="$RESULT_DIR/object_crop_features.h5"
META="$RESULT_DIR/object_crop_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[6]}" motionbench_motion_camera '
FEATURES="$RESULT_DIR/motion_camera_features.h5"
META="$RESULT_DIR/motion_camera_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$MOTIONBENCH_JSONL" \
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

launch_job "${GPUS[7]}" mvbench_ablation '
mkdir -p "$RESULT_DIR/mvbench"
if [[ -n "$MVBENCH_VIDEO_ROOT" ]]; then
  python -u experiments/make_mvbench_subset.py \
    --json-dir "$MVBENCH_JSON_DIR" \
    --video-root "$MVBENCH_VIDEO_ROOT" \
    --output-jsonl "$RESULT_DIR/mvbench/mvbench_subset.jsonl" \
    --missing-jsonl "$RESULT_DIR/mvbench/mvbench_missing.jsonl" \
    --per-task "$PER_MVBENCH_TASK"
else
  python -u experiments/make_mvbench_subset.py \
    --json-dir "$MVBENCH_JSON_DIR" \
    --output-jsonl "$RESULT_DIR/mvbench/mvbench_subset.jsonl" \
    --missing-jsonl "$RESULT_DIR/mvbench/mvbench_missing.jsonl" \
    --per-task "$PER_MVBENCH_TASK"
fi
if [[ ! -s "$RESULT_DIR/mvbench/mvbench_subset.jsonl" ]]; then
  cat > "$RESULT_DIR/mvbench/MVBENCH_SKIPPED.md" <<EOF
# MVBench Skipped

No local MVBench videos were resolved. Annotation JSONs were found, but the
actual source benchmark videos are not present under MVBENCH_VIDEO_ROOT.

Set MVBENCH_VIDEO_ROOT to the directory containing MVBench source videos and
rerun this script to execute the same ablations on MVBench.
EOF
  exit 0
fi
FEATURES="$RESULT_DIR/mvbench/grid_resolution_features.h5"
META="$RESULT_DIR/mvbench/grid_resolution_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$RESULT_DIR/mvbench/mvbench_subset.jsonl" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_1x1,wan_vae_grid_2x2,wan_vae_grid_4x4,wan_vae_grid_8x8,wan_vae_grid_16x16 \
  --num-frames 17 \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/mvbench/grid_resolution_probe" \
  "wan_vae_grid_1x1,wan_vae_grid_2x2,wan_vae_grid_4x4,wan_vae_grid_8x8,wan_vae_grid_16x16" \
  "none"
for frames in 8 16 32 64; do
  FEATURES="$RESULT_DIR/mvbench/temporal_${frames}_features.h5"
  META="$RESULT_DIR/mvbench/temporal_${frames}_metadata.jsonl"
  batch=2
  if [[ "$frames" == "64" ]]; then batch=1; fi
  python -u experiments/extract_wan_features.py \
    --video-jsonl "$RESULT_DIR/mvbench/mvbench_subset.jsonl" \
    --output-h5 "$FEATURES" \
    --metadata-output "$META" \
    --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
    --num-frames "$frames" \
    --height 128 \
    --width 128 \
    --batch-size "$batch" \
    --device cuda:0 \
    --lowfps-modes none
  probe_candidates "$FEATURES" "$META" "$RESULT_DIR/mvbench/temporal_${frames}_probe" \
    "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
    "none"
done
FEATURES="$RESULT_DIR/mvbench/crop_motion_camera_features.h5"
META="$RESULT_DIR/mvbench/crop_motion_camera_metadata.jsonl"
python -u experiments/extract_wan_features.py \
  --video-jsonl "$RESULT_DIR/mvbench/mvbench_subset.jsonl" \
  --output-h5 "$FEATURES" \
  --metadata-output "$META" \
  --feature-types wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence \
  --num-frames "$ABLATION_FRAMES" \
  --height 128 \
  --width 128 \
  --batch-size 2 \
  --device cuda:0 \
  --lowfps-modes none,center_crop,object_crop,high_motion,camera_comp
probe_candidates "$FEATURES" "$META" "$RESULT_DIR/mvbench/crop_motion_camera_probe" \
  "wan_vae_grid_4x4,pixel_grid_sequence,flow_grid_sequence" \
  "none,center_crop,object_crop,high_motion,camera_comp"
'

echo "Launched jobs:"
cat "$RESULT_DIR/pids.tsv"
echo "Logs: $LOG_DIR"
echo "Use: tail -f $LOG_DIR/<job>.log"
