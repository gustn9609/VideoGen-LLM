#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hskim/VideoGen-LLM}"
MVBENCH_JSONL="${MVBENCH_JSONL:?Set MVBENCH_JSONL}"
RUN_TAG="${RUN_TAG:-wan_mvbench_only_after_gpu_free_$(date +%Y%m%d_%H%M%S)}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
BUSY_MEMORY_MB="${BUSY_MEMORY_MB:-20000}"
LATEST_ITER_ROOT="${LATEST_ITER_ROOT:-/home/hskim/MinorPerception/flow_grpo/logs/r2i_pref_grpo/flux_v3_clip_grpo_stage3init_memfix_r512_bs1_ts025_lr5e6_clip1e3_beta0/rollouts/images}"

cd "$ROOT"

high_mem_process_count() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v threshold="$BUSY_MEMORY_MB" '
        {
          gsub(/ /, "", $2)
          if (($2 + 0) > threshold) {
            count++
          }
        }
        END { print count + 0 }
      '
}

latest_iter() {
  if [[ ! -d "$LATEST_ITER_ROOT" ]]; then
    echo "unknown"
    return
  fi
  find "$LATEST_ITER_ROOT" -maxdepth 1 -type d -name "iter_*" -printf "%f\n" 2>/dev/null \
    | sort \
    | tail -1
}

echo "[$(date --iso-8601=seconds)] Waiting for GPUs before MVBench ablation"
echo "run_tag=$RUN_TAG"
echo "mvbench_jsonl=$MVBENCH_JSONL"
echo "gpus=$GPUS_CSV"
echo "busy_memory_mb=$BUSY_MEMORY_MB"

while true; do
  busy="$(high_mem_process_count)"
  if [[ "$busy" -eq 0 ]]; then
    echo "[$(date --iso-8601=seconds)] GPUs are below busy threshold; launching MVBench"
    break
  fi
  echo "[$(date --iso-8601=seconds)] busy_high_mem_processes=$busy latest_minor_iter=$(latest_iter)"
  sleep "$POLL_SECONDS"
done

MVBENCH_JSONL="$MVBENCH_JSONL" \
RUN_TAG="$RUN_TAG" \
GPUS_CSV="$GPUS_CSV" \
WAIT_FOR_JOBS=1 \
bash experiments/run_wan_ablation_mvbench_only.sh
