#!/usr/bin/env python3
"""Analyze out-of-fold multiple-choice candidate scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_scores(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test" and row.get("stage") == "final":
                    rows.append(row)
    return rows


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    scores = np.asarray([float(c["score"]) for c in row["candidates"]], dtype=np.float64)
    gold = int(row["answer_index"])
    pred = int(np.argmax(scores))
    order = np.argsort(-scores)
    rank = int(np.where(order == gold)[0][0]) + 1
    others = np.delete(scores, gold)
    margin = float(scores[gold] - np.max(others)) if others.size else 0.0
    logits = scores - np.max(scores)
    log_probs = logits - np.log(np.exp(logits).sum())
    nll = float(-log_probs[gold])
    return {
        "key": f"{row.get('fold', '')}:{row.get('row_index', row.get('video_id'))}",
        "video_id": row.get("video_id"),
        "fold": row.get("fold"),
        "row_index": row.get("row_index"),
        "question_type": row.get("question_type", ""),
        "correct": bool(row.get("correct", pred == gold)),
        "answer_index": gold,
        "prediction_index": pred,
        "gold_rank": rank,
        "margin": margin,
        "candidate_nll": nll,
        "gold_score": float(scores[gold]),
        "best_score": float(scores[pred]),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"n": 0}
    correct = np.asarray([x["correct"] for x in items], dtype=np.float64)
    rank = np.asarray([x["gold_rank"] for x in items], dtype=np.float64)
    margin = np.asarray([x["margin"] for x in items], dtype=np.float64)
    nll = np.asarray([x["candidate_nll"] for x in items], dtype=np.float64)
    return {
        "n": int(len(items)),
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "mean_gold_rank": float(rank.mean()),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "mean_candidate_nll": float(nll.mean()),
    }


def paired_summary(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    amap = {x["key"]: x for x in a}
    bmap = {x["key"]: x for x in b}
    keys = sorted(set(amap) & set(bmap))
    if not keys:
        return {"n": 0}
    diffs = {
        "correct": np.asarray([float(bmap[k]["correct"]) - float(amap[k]["correct"]) for k in keys]),
        "margin": np.asarray([bmap[k]["margin"] - amap[k]["margin"] for k in keys]),
        "candidate_nll": np.asarray([bmap[k]["candidate_nll"] - amap[k]["candidate_nll"] for k in keys]),
        "gold_rank": np.asarray([bmap[k]["gold_rank"] - amap[k]["gold_rank"] for k in keys]),
    }
    out = {
        "n": int(len(keys)),
        "b_only_correct": int(sum((not amap[k]["correct"]) and bmap[k]["correct"] for k in keys)),
        "a_only_correct": int(sum(amap[k]["correct"] and (not bmap[k]["correct"]) for k in keys)),
        "both_correct": int(sum(amap[k]["correct"] and bmap[k]["correct"] for k in keys)),
        "both_wrong": int(sum((not amap[k]["correct"]) and (not bmap[k]["correct"]) for k in keys)),
    }
    for name, values in diffs.items():
        out[f"mean_delta_{name}"] = float(values.mean())
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# OOF Candidate Score Analysis", ""]
    lines.append("| Run | N | Acc | Mean rank | Mean margin | Median margin | Mean NLL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, stats in payload["summary"].items():
        lines.append(
            f"| {name} | {stats['n']} | {stats['accuracy']:.4f} | {stats['mean_gold_rank']:.3f} | "
            f"{stats['mean_margin']:.4f} | {stats['median_margin']:.4f} | {stats['mean_candidate_nll']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-Type")
    for name, by_type in payload["per_type"].items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append("| Type | N | Acc | Mean rank | Mean margin | Mean NLL |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for qtype, stats in sorted(by_type.items()):
            lines.append(
                f"| {qtype} | {stats['n']} | {stats['accuracy']:.4f} | {stats['mean_gold_rank']:.3f} | "
                f"{stats['mean_margin']:.4f} | {stats['mean_candidate_nll']:.4f} |"
            )
    if payload.get("paired"):
        lines.append("")
        lines.append("## Paired")
        for name, stats in payload["paired"].items():
            lines.append("")
            lines.append(f"### {name}")
            for key, value in stats.items():
                lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="name=path pairs separated by commas")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = []
    for item in args.scores.split(","):
        name, path = item.split("=", 1)
        pairs.append((name, path))
    metrics_by_run = {name: [row_metrics(row) for row in read_scores(path)] for name, path in pairs}
    payload = {
        "summary": {name: summarize(items) for name, items in metrics_by_run.items()},
        "per_type": {},
        "paired": {},
    }
    for name, items in metrics_by_run.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[str(item["question_type"])].append(item)
        payload["per_type"][name] = {key: summarize(vals) for key, vals in groups.items()}
    if len(pairs) >= 2:
        base_name = pairs[0][0]
        for name, _ in pairs[1:]:
            payload["paired"][f"{base_name}_vs_{name}"] = paired_summary(metrics_by_run[base_name], metrics_by_run[name])
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(Path(args.output_md), payload)
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
