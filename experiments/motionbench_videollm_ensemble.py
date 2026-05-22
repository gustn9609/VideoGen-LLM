#!/usr/bin/env python3
"""Combine VideoLLM candidate scores with Wan candidate-reranker scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import normalize_scores, read_jsonl  # noqa: E402


def key_for(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("split", 0)),
        str(row.get("mode", "")),
        str(row.get("feature", "")),
        str(row.get("video_id", row.get("uid", row.get("id", row.get("row_index", ""))))),
    )


def sample_key(row: dict[str, Any]) -> str:
    return str(row.get("video_id", row.get("uid", row.get("id", row.get("row_index", "")))))


def candidate_scores(row: dict[str, Any]) -> np.ndarray:
    if "candidates" in row and isinstance(row["candidates"], list):
        return np.asarray([float(c.get("score", c.get("logprob", 0.0))) for c in row["candidates"]], dtype=np.float64)
    scores = row.get("scores")
    if isinstance(scores, dict):
        return np.asarray([float(scores[k]) for k in sorted(scores)], dtype=np.float64)
    if isinstance(scores, list):
        return np.asarray(scores, dtype=np.float64)
    raise KeyError("Could not find candidate scores")


def correct_index(row: dict[str, Any]) -> int:
    if "answer_index" in row:
        return int(row["answer_index"])
    if "correct_index" in row:
        return int(row["correct_index"])
    answer = str(row.get("answer", "A")).strip().upper()
    if len(answer) == 1 and "A" <= answer <= "Z":
        return ord(answer) - ord("A")
    return int(row.get("label", 0))


def pred_from_scores(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def eval_rows(rows: list[dict[str, Any]], score_fn) -> dict[str, Any]:
    correct = []
    by_type: dict[str, list[int]] = {}
    preds = []
    for row in rows:
        scores = score_fn(row)
        pred = pred_from_scores(scores)
        truth = correct_index(row)
        val = int(pred == truth)
        preds.append(pred)
        correct.append(val)
        by_type.setdefault(str(row.get("question_type", "all")), []).append(val)
    arr = np.asarray(correct, dtype=np.float64)
    return {
        "accuracy": float(arr.mean()) if arr.size else 0.0,
        "correct": int(arr.sum()),
        "total": int(arr.size),
        "per_question_type": {
            k: {"accuracy": float(np.mean(v)), "correct": int(np.sum(v)), "total": int(len(v))}
            for k, v in sorted(by_type.items())
        },
        "predictions": preds,
        "correct_vector": correct,
    }


def align_rows(wan_rows: list[dict[str, Any]], videollm_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not videollm_rows:
        return wan_rows
    video_by_key = {sample_key(row): row for row in videollm_rows}
    aligned = []
    for wan in wan_rows:
        key = sample_key(wan)
        base = video_by_key.get(key)
        if base is None:
            continue
        merged = dict(wan)
        merged["videollm_candidates"] = base.get("candidates")
        merged["videollm_scores"] = candidate_scores(base).tolist()
        aligned.append(merged)
    return aligned


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench VideoLLM + Wan Ensemble", ""]
    if payload.get("base_metrics"):
        lines.append("| system | acc | correct/total |")
        lines.append("|---|---:|---:|")
        for name, item in payload["base_metrics"].items():
            lines.append(f"| {name} | {item['accuracy']:.4f} | {item['correct']}/{item['total']} |")
    if payload.get("sweep"):
        lines.extend(["", "## Ensemble Sweep", ""])
        lines.append("| alpha | beta | norm | temp | acc | correct/total |")
        lines.append("|---:|---:|---|---:|---:|---:|")
        for item in payload["sweep"]:
            lines.append(
                f"| {item['alpha']:.3f} | {item['beta']:.3f} | {item['normalization']} | {item['temperature']:.3f} | "
                f"{item['accuracy']:.4f} | {item['correct']}/{item['total']} |"
            )
    if payload.get("complementarity"):
        comp = payload["complementarity"]
        lines.extend(["", "## Complementarity", ""])
        for key in ["videollm_only_correct", "wan_only_correct", "both_correct", "both_wrong", "oracle_union_accuracy", "disagreement_accuracy_wan"]:
            lines.append(f"- {key}: {comp[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-scores-jsonl", required=True)
    parser.add_argument("--videollm-scores-jsonl", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--wan-mode", default="")
    parser.add_argument("--wan-feature", default="")
    parser.add_argument("--normalizations", default="zscore,rank,raw")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--temperatures", default="1.0")
    args = parser.parse_args()

    wan_rows = read_jsonl(Path(args.wan_scores_jsonl))
    if args.wan_mode:
        wan_rows = [row for row in wan_rows if str(row.get("mode")) == args.wan_mode]
    if args.wan_feature:
        wan_rows = [row for row in wan_rows if str(row.get("feature")) == args.wan_feature]
    videollm_rows = read_jsonl(Path(args.videollm_scores_jsonl)) if args.videollm_scores_jsonl else []
    rows = align_rows(wan_rows, videollm_rows, args)

    base_metrics = {
        "wan_only": eval_rows(rows, lambda row: candidate_scores(row)),
    }
    if videollm_rows:
        base_metrics["videollm_only"] = eval_rows(rows, lambda row: np.asarray(row["videollm_scores"], dtype=np.float64))

    sweep = []
    if videollm_rows:
        normalizations = [x.strip() for x in args.normalizations.split(",") if x.strip()]
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        betas = [float(x) for x in args.betas.split(",") if x.strip()]
        temperatures = [float(x) for x in args.temperatures.split(",") if x.strip()]
        for norm in normalizations:
            for temp in temperatures:
                for alpha in alphas:
                    for beta in betas:
                        if alpha == 0 and beta == 0:
                            continue
                        metric = eval_rows(
                            rows,
                            lambda row, alpha=alpha, beta=beta, norm=norm, temp=temp: alpha
                            * normalize_scores(np.asarray(row["videollm_scores"], dtype=np.float64), norm, temp)
                            + beta * normalize_scores(candidate_scores(row), norm, temp),
                        )
                        sweep.append(
                            {
                                "alpha": alpha,
                                "beta": beta,
                                "normalization": norm,
                                "temperature": temp,
                                **{k: v for k, v in metric.items() if k not in {"predictions", "correct_vector"}},
                            }
                        )
        sweep.sort(key=lambda x: x["accuracy"], reverse=True)

    complementarity = {}
    if videollm_rows:
        wan_correct = np.asarray(base_metrics["wan_only"]["correct_vector"], dtype=bool)
        vlm_correct = np.asarray(base_metrics["videollm_only"]["correct_vector"], dtype=bool)
        disagree = np.asarray(base_metrics["wan_only"]["predictions"]) != np.asarray(base_metrics["videollm_only"]["predictions"])
        complementarity = {
            "videollm_only_correct": int((vlm_correct & ~wan_correct).sum()),
            "wan_only_correct": int((wan_correct & ~vlm_correct).sum()),
            "both_correct": int((wan_correct & vlm_correct).sum()),
            "both_wrong": int((~wan_correct & ~vlm_correct).sum()),
            "oracle_union_accuracy": float((wan_correct | vlm_correct).mean()) if len(wan_correct) else 0.0,
            "disagreement_count": int(disagree.sum()),
            "disagreement_accuracy_wan": float(wan_correct[disagree].mean()) if disagree.any() else 0.0,
            "disagreement_accuracy_videollm": float(vlm_correct[disagree].mean()) if disagree.any() else 0.0,
        }

    payload = {
        "config": vars(args),
        "rows": len(rows),
        "base_metrics": {k: {kk: vv for kk, vv in v.items() if kk not in {"predictions", "correct_vector"}} for k, v in base_metrics.items()},
        "sweep": sweep,
        "complementarity": complementarity,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({"rows": len(rows), "sweep": len(sweep), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
