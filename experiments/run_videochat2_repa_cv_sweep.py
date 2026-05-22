#!/usr/bin/env python3
"""Run VideoChat2-HD Wan-REPA CV sweeps across seeds/lambdas."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def fmt_lambda(value: float) -> str:
    return str(value).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="123,124,125,126,127")
    parser.add_argument("--lambdas", default="0,0.03,0.1,0.3")
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--metadata-jsonl", default="results/videochat2_hd_hidden_motionbench_f8_hmcc/hidden_tokens_metadata.jsonl")
    parser.add_argument("--hidden-features-h5", default="results/videochat2_hd_hidden_motionbench_f8_hmcc/hidden_tokens.h5")
    parser.add_argument("--target-features-h5", default="results/motionbench_wan_motion_repa_20260511_pool1f8/wmrepa_targets.h5")
    parser.add_argument("--target-metadata-jsonl", default="results/motionbench_wan_motion_repa_20260511_pool1f8/wmrepa_targets_metadata.jsonl")
    parser.add_argument("--source-features-h5", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def final_accuracy(path: Path) -> tuple[int, int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in reversed(data["eval"]):
        if row.get("stage") == "final" and row.get("split") == "test" and row.get("fold") == "all":
            return int(row["correct"]), int(row["total"]), float(row["accuracy"])
    raise ValueError(f"No final aggregate in {path}")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    jobs = []
    for seed in seeds:
        for lam in lambdas:
            out_dir = out_root / f"seed{seed}_lambda{fmt_lambda(lam)}"
            done = out_dir / "finetune_eval.json"
            if args.skip_existing and done.exists():
                continue
            cmd = [
                sys.executable,
                "-u",
                "experiments/videochat2_hd_wan_repa_finetune.py",
                "--metadata-jsonl",
                args.metadata_jsonl,
                "--hidden-features-h5",
                args.hidden_features_h5,
                "--target-features-h5",
                args.target_features_h5,
                "--target-metadata-jsonl",
                args.target_metadata_jsonl,
                "--output-dir",
                str(out_dir),
                "--mode",
                "high_motion+camera_comp",
                "--wan-feature",
                "wan_vae_grid_1x1",
                "--repa-target",
                "equivariance",
                "--lambda-repa",
                str(lam),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                "1",
                "--folds",
                str(args.folds),
                "--split-seed",
                str(args.split_seed),
                "--device",
                "cuda:0",
                "--dtype",
                "fp16",
                "--seed",
                str(seed),
            ]
            if args.source_features_h5:
                cmd.extend(["--source-features-h5", args.source_features_h5])
            jobs.append({"seed": seed, "lambda": lam, "out_dir": out_dir, "cmd": cmd})

    running: list[dict] = []
    completed = []
    next_gpu = 0
    while jobs or running:
        while jobs and len(running) < args.max_parallel:
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
            print(f"started seed={job['seed']} lambda={job['lambda']} gpu={gpu} pid={proc.pid}", flush=True)
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
            completed.append(job)
            print(f"finished seed={job['seed']} lambda={job['lambda']} code={code}", flush=True)
            if code != 0:
                raise SystemExit(f"Job failed: {job['out_dir']}/run.log")
        running = still

    rows = []
    for job in completed:
        eval_path = job["out_dir"] / "finetune_eval.json"
        correct, total, acc = final_accuracy(eval_path)
        rows.append(
            {
                "seed": job["seed"],
                "lambda_repa": job["lambda"],
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "output_dir": str(job["out_dir"]),
            }
        )
    rows = sorted(rows, key=lambda x: (x["lambda_repa"], x["seed"]))
    summary = {"rows": rows}
    by_lambda = {}
    for lam in lambdas:
        vals = [row["accuracy"] for row in rows if row["lambda_repa"] == lam]
        if vals:
            by_lambda[str(lam)] = {
                "mean_accuracy": sum(vals) / len(vals),
                "min_accuracy": min(vals),
                "max_accuracy": max(vals),
                "runs": len(vals),
            }
    summary["by_lambda"] = by_lambda
    (out_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# VideoChat2-HD Wan-REPA Robustness Sweep", ""]
    lines.append("| Lambda | Runs | Mean acc | Min | Max |")
    lines.append("|---:|---:|---:|---:|---:|")
    for lam, item in by_lambda.items():
        lines.append(f"| {lam} | {item['runs']} | {item['mean_accuracy']:.4f} | {item['min_accuracy']:.4f} | {item['max_accuracy']:.4f} |")
    lines.append("")
    lines.append("| Seed | Lambda | Acc | Correct/total |")
    lines.append("|---:|---:|---:|---:|")
    for row in rows:
        lines.append(f"| {row['seed']} | {row['lambda_repa']} | {row['accuracy']:.4f} | {row['correct']}/{row['total']} |")
    (out_root / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
