#!/usr/bin/env python3
"""Run CE/Eq/Rel VideoChat2-HD adapter experiments on the larger temporal set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def final_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in reversed(data["eval"]):
        if row.get("stage") == "final" and row.get("split") == "test" and row.get("fold") == "all":
            return row
    raise ValueError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata-jsonl", default="results/motionbench_larger_temporal_20260511/hidden_tokens_none_metadata.jsonl")
    parser.add_argument("--hidden-features-h5", default="results/motionbench_larger_temporal_20260511/hidden_tokens_none.h5")
    parser.add_argument("--target-features-h5", default="results/motionbench_larger_temporal_20260511/wmrepa_targets_none.h5")
    parser.add_argument("--target-metadata-jsonl", default="results/motionbench_larger_temporal_20260511/wmrepa_targets_none_metadata.jsonl")
    parser.add_argument("--source-features-h5", default="results/motionbench_larger_temporal_20260511/features_pool1f8_none.h5")
    parser.add_argument("--mode", default="none")
    parser.add_argument("--gpus", default="5,6,7")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = [
        "--metadata-jsonl",
        args.metadata_jsonl,
        "--hidden-features-h5",
        args.hidden_features_h5,
        "--target-features-h5",
        args.target_features_h5,
        "--target-metadata-jsonl",
        args.target_metadata_jsonl,
        "--mode",
        args.mode,
        "--wan-feature",
        "wan_vae_grid_1x1",
        "--repa-target",
        "equivariance",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--folds",
        "5",
        "--split-seed",
        "123",
        "--device",
        "cuda:0",
        "--dtype",
        "fp16",
        "--seed",
        "123",
    ]
    cases = [
        {"name": "ce", "args": ["--lambda-repa", "0.0"]},
        {"name": "eq", "args": ["--lambda-repa", "0.1"]},
        {
            "name": "rel",
            "args": [
                "--lambda-repa",
                "0.0",
                "--lambda-relation",
                "0.1",
                "--source-features-h5",
                args.source_features_h5,
            ],
        },
    ]
    jobs = []
    for case in cases:
        out_dir = output_root / case["name"]
        if args.skip_existing and (out_dir / "finetune_eval.json").exists():
            continue
        cmd = [
            sys.executable,
            "-u",
            "experiments/videochat2_hd_wan_repa_finetune.py",
            *base,
            "--output-dir",
            str(out_dir),
            *case["args"],
        ]
        jobs.append({"name": case["name"], "out_dir": out_dir, "cmd": cmd})

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    running = []
    completed = []
    next_gpu = 0
    while jobs or running:
        while jobs and len(running) < int(args.max_parallel):
            job = jobs.pop(0)
            gpu = gpus[next_gpu % len(gpus)]
            next_gpu += 1
            job["out_dir"].mkdir(parents=True, exist_ok=True)
            log = open(job["out_dir"] / "run.log", "w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT, cwd=Path.cwd(), env=env)
            job.update({"proc": proc, "gpu": gpu, "log": log, "start": time.time()})
            running.append(job)
            print(f"started {job['name']} gpu={gpu} pid={proc.pid}", flush=True)
        time.sleep(2)
        still = []
        for job in running:
            code = job["proc"].poll()
            if code is None:
                still.append(job)
                continue
            job["log"].close()
            print(f"finished {job['name']} code={code}", flush=True)
            if code != 0:
                raise SystemExit(f"Job failed: {job['out_dir']}/run.log")
            job["elapsed_sec"] = time.time() - job["start"]
            completed.append(job)
        running = still

    rows = []
    for job in completed:
        row = final_row(job["out_dir"] / "finetune_eval.json")
        rows.append(
            {
                "name": job["name"],
                "accuracy": float(row["accuracy"]),
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "per_question_type": row.get("per_question_type", {}),
                "elapsed_sec": float(job.get("elapsed_sec", 0.0)),
                "output_dir": str(job["out_dir"]),
            }
        )
    rows = sorted(rows, key=lambda x: x["name"])
    (output_root / "larger_temporal_summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    lines = ["# Larger Temporal Set CE/Eq/Rel Summary", "", "| Run | Acc | Correct/total |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['accuracy']:.4f} | {row['correct']}/{row['total']} |")
    lines.append("")
    lines.append("## Per-Type")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['name']}")
        lines.append("| Type | Acc | Correct/total |")
        lines.append("|---|---:|---:|")
        for qtype, stats in sorted(row["per_question_type"].items()):
            lines.append(f"| {qtype} | {stats['accuracy']:.4f} | {stats['correct']}/{stats['total']} |")
    (output_root / "larger_temporal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
