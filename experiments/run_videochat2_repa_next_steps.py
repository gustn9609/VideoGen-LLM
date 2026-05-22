#!/usr/bin/env python3
"""Run the next VideoChat2-HD Wan-REPA experiments requested after Eq/Rel."""

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


def final_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in reversed(data["eval"]):
        if row.get("stage") == "final" and row.get("split") == "test" and row.get("fold") == "all":
            return row
    raise ValueError(f"No final aggregate in {path}")


def cases() -> list[dict]:
    negative_root = Path("results/videochat2_hd_wan_repa_negative_targets")
    out = [
        {
            "name": "branch_separated_eq_rel",
            "group": "branch",
            "args": [
                "--adapter-type",
                "branch_separated",
                "--lambda-repa",
                "0.1",
                "--lambda-relation",
                "0.1",
                "--source-features-h5",
                "results/videochat2_hd_hidden_motionbench_f8_hmcc/combined_features.h5",
            ],
        },
        {
            "name": "schedule_constant_e3",
            "group": "schedule",
            "args": ["--lambda-repa", "0.1", "--epochs", "3", "--lambda-schedule", "constant"],
        },
        {
            "name": "schedule_warmup_ce_polish_e3",
            "group": "schedule",
            "args": ["--lambda-repa", "0.1", "--epochs", "3", "--lambda-schedule", "warmup_ce_polish"],
        },
        {
            "name": "schedule_late_start_e3",
            "group": "schedule",
            "args": ["--lambda-repa", "0.1", "--epochs", "3", "--lambda-schedule", "late_start"],
        },
        {
            "name": "schedule_cosine_decay_e3",
            "group": "schedule",
            "args": ["--lambda-repa", "0.1", "--epochs", "3", "--lambda-schedule", "cosine_decay"],
        },
    ]
    for control in ["random", "first_frame", "time_average", "pixel", "flow"]:
        out.append(
            {
                "name": f"negative_{control}",
                "group": "negative_control",
                "args": [
                    "--target-features-h5",
                    str(negative_root / control / "targets.h5"),
                    "--target-metadata-jsonl",
                    str(negative_root / control / "targets_metadata.jsonl"),
                    "--lambda-repa",
                    "0.1",
                ],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-groups", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    groups = {x.strip() for x in args.only_groups.split(",") if x.strip()}
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    jobs = []
    for case in cases():
        if groups and case["group"] not in groups:
            continue
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
        jobs.append({"name": case["name"], "group": case["group"], "out_dir": out_dir, "cmd": cmd})

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
            print(f"started {job['name']} group={job['group']} gpu={gpu} pid={proc.pid}", flush=True)
        time.sleep(2)
        still = []
        for job in running:
            code = job["proc"].poll()
            if code is None:
                still.append(job)
                continue
            job["log"].close()
            job["returncode"] = code
            job["elapsed_sec"] = time.time() - job["start"]
            print(f"finished {job['name']} code={code}", flush=True)
            if code != 0:
                raise SystemExit(f"Job failed: {job['out_dir']}/run.log")
            completed.append(job)
        running = still

    rows = []
    for job in completed:
        row = final_row(job["out_dir"] / "finetune_eval.json")
        rows.append(
            {
                "name": job["name"],
                "group": job["group"],
                "accuracy": float(row["accuracy"]),
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "per_question_type": row.get("per_question_type", {}),
                "elapsed_sec": float(job.get("elapsed_sec", 0.0)),
                "output_dir": str(job["out_dir"]),
            }
        )
    rows = sorted(rows, key=lambda x: (x["group"], x["name"]))
    (output_root / "next_steps_summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    lines = ["# VideoChat2-HD Wan-REPA Next Steps Summary", "", "| Group | Experiment | Acc | Correct/total |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['group']} | {row['name']} | {row['accuracy']:.4f} | {row['correct']}/{row['total']} |")
    lines.append("")
    lines.append("## Per-Type")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['name']}")
        lines.append("| Type | Acc | Correct/total |")
        lines.append("|---|---:|---:|")
        for qtype, stats in sorted(row["per_question_type"].items()):
            lines.append(f"| {qtype} | {stats['accuracy']:.4f} | {stats['correct']}/{stats['total']} |")
    (output_root / "next_steps_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
