#!/usr/bin/env python3
"""Run follow-up VideoChat2-HD Wan-REPA CV experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_ARGS = [
    "--metadata-jsonl",
    "results/videochat2_hd_hidden_motionbench_f8_hmcc/hidden_tokens_metadata.jsonl",
    "--hidden-features-h5",
    "results/videochat2_hd_hidden_motionbench_f8_hmcc/hidden_tokens.h5",
    "--target-features-h5",
    "results/motionbench_wan_motion_repa_20260511_pool1f8/wmrepa_targets.h5",
    "--target-metadata-jsonl",
    "results/motionbench_wan_motion_repa_20260511_pool1f8/wmrepa_targets_metadata.jsonl",
    "--mode",
    "high_motion+camera_comp",
    "--wan-feature",
    "wan_vae_grid_1x1",
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


def final_accuracy(path: Path) -> tuple[int, int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in reversed(data["eval"]):
        if row.get("stage") == "final" and row.get("split") == "test" and row.get("fold") == "all":
            return int(row["correct"]), int(row["total"]), float(row["accuracy"])
    raise ValueError(path)


def cases(output_root: Path) -> list[dict]:
    return [
        {
            "name": "ce_preserving_kl",
            "args": [
                "--repa-target",
                "equivariance",
                "--lambda-repa",
                "0.1",
                "--lambda-kl",
                "0.1",
                "--teacher-checkpoint",
                "results/videochat2_hd_wan_repa_robustness_equiv_cv5/seed123_lambda0p0/adapter_checkpoint.pt",
            ],
        },
        {
            "name": "zero_init_repa",
            "args": ["--repa-target", "equivariance", "--lambda-repa", "0.1", "--adapter-init-scale", "0.0"],
        },
        {
            "name": "relation_only",
            "args": [
                "--repa-target",
                "equivariance",
                "--lambda-repa",
                "0.0",
                "--lambda-relation",
                "0.1",
                "--source-features-h5",
                "results/videochat2_hd_hidden_motionbench_f8_hmcc/combined_features.h5",
            ],
        },
        {
            "name": "relation_plus_equiv",
            "args": [
                "--repa-target",
                "equivariance",
                "--lambda-repa",
                "0.1",
                "--lambda-relation",
                "0.1",
                "--source-features-h5",
                "results/videochat2_hd_hidden_motionbench_f8_hmcc/combined_features.h5",
            ],
        },
        {
            "name": "dynamics_repa",
            "args": ["--repa-target", "dynamics_relation", "--lambda-repa", "0.1"],
        },
        {
            "name": "type_conditional_lambda",
            "args": [
                "--repa-target",
                "equivariance",
                "--lambda-repa",
                "0.1",
                "--type-lambda-json",
                '{"Action Order": 2.0, "Motion Recognition": 2.0, "Motion-related Objects": 0.25, "Repetition Count": 1.0}',
            ],
        },
        {
            "name": "unlabeled_pretrain_all",
            "args": [
                "--repa-target",
                "equivariance",
                "--lambda-repa",
                "0.1",
                "--pretrain-epochs",
                "1",
                "--pretrain-scope",
                "all",
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=7)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    jobs = []
    for case in cases(output_root):
        out_dir = output_root / case["name"]
        if args.skip_existing and (out_dir / "finetune_eval.json").exists():
            continue
        cmd = [
            sys.executable,
            "-u",
            "experiments/videochat2_hd_wan_repa_finetune.py",
            *BASE_ARGS,
            "--output-dir",
            str(out_dir),
            *case["args"],
        ]
        jobs.append({"name": case["name"], "out_dir": out_dir, "cmd": cmd})

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
            job.update({"proc": proc, "gpu": gpu, "log": log})
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
            completed.append(job)
        running = still

    rows = []
    for job in completed:
        correct, total, accuracy = final_accuracy(job["out_dir"] / "finetune_eval.json")
        rows.append({"name": job["name"], "correct": correct, "total": total, "accuracy": accuracy, "output_dir": str(job["out_dir"])})
    rows = sorted(rows, key=lambda x: x["name"])
    (output_root / "followup_summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    lines = ["# VideoChat2-HD Wan-REPA Follow-up Summary", "", "| Experiment | Acc | Correct/total |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['accuracy']:.4f} | {row['correct']}/{row['total']} |")
    (output_root / "followup_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
