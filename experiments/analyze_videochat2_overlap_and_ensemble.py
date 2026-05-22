#!/usr/bin/env python3
"""Analyze CE/Eq/Rel out-of-fold overlap and logit-level ensembles."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_score_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "test" and row.get("stage") == "final":
                rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("row_index", row.get("video_id", "")))


def row_scores(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(c["score"]) for c in row["candidates"]], dtype=np.float64)


def log_softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - np.max(scores)
    return z - np.log(np.exp(z).sum())


def metrics_for_row(row: dict[str, Any]) -> dict[str, Any]:
    scores = row_scores(row)
    label = int(row["answer_index"])
    pred = int(np.argmax(scores))
    order = np.argsort(-scores)
    rank = int(np.where(order == label)[0][0]) + 1
    others = np.delete(scores, label)
    margin = float(scores[label] - np.max(others)) if others.size else 0.0
    nll = float(-log_softmax(scores)[label])
    return {
        "key": row_key(row),
        "row_index": int(row.get("row_index", -1)),
        "fold": int(row.get("fold", -1)),
        "video_id": row.get("video_id"),
        "question_type": str(row.get("question_type", "")),
        "answer_index": label,
        "prediction_index": pred,
        "correct": bool(pred == label),
        "rank": rank,
        "margin": margin,
        "nll": nll,
        "scores": scores,
        "row": row,
    }


def summarize_bool(values: list[bool]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {"correct": int(arr.sum()), "total": int(arr.size), "accuracy": float(arr.mean()) if arr.size else 0.0}


def summarize_metrics(items: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(items)}
    for name in names:
        if not items:
            out[name] = {}
            continue
        out[name] = {
            "accuracy": float(np.mean([x[name]["correct"] for x in items])),
            "mean_rank": float(np.mean([x[name]["rank"] for x in items])),
            "mean_margin": float(np.mean([x[name]["margin"] for x in items])),
            "mean_nll": float(np.mean([x[name]["nll"] for x in items])),
        }
    return out


def normalize_for_ensemble(scores: np.ndarray, mode: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if mode == "raw":
        return scores
    if mode == "center":
        return scores - scores.mean()
    if mode == "zscore":
        centered = scores - scores.mean()
        return centered / max(float(centered.std()), 1e-6)
    if mode == "logprob":
        return log_softmax(scores)
    raise ValueError(mode)


def eval_weights(
    rows: list[dict[str, Any]],
    names: list[str],
    weights: tuple[float, ...],
    norm: str,
) -> dict[str, Any]:
    correct = []
    nlls = []
    margins = []
    ranks = []
    preds = []
    for item in rows:
        label = int(item[names[0]]["answer_index"])
        score = np.zeros_like(item[names[0]]["scores"], dtype=np.float64)
        for name, weight in zip(names, weights):
            score = score + float(weight) * normalize_for_ensemble(item[name]["scores"], norm)
        pred = int(np.argmax(score))
        order = np.argsort(-score)
        rank = int(np.where(order == label)[0][0]) + 1
        others = np.delete(score, label)
        margin = float(score[label] - np.max(others)) if others.size else 0.0
        lp = log_softmax(score)
        correct.append(pred == label)
        nlls.append(float(-lp[label]))
        margins.append(margin)
        ranks.append(rank)
        preds.append(pred)
    arr = np.asarray(correct, dtype=np.float64)
    return {
        "accuracy": float(arr.mean()) if arr.size else 0.0,
        "correct": int(arr.sum()),
        "total": int(arr.size),
        "mean_nll": float(np.mean(nlls)) if nlls else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "predictions": preds,
    }


def weight_grid(n: int, step: float) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    out: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining: int, depth: int) -> None:
        if depth == n - 1:
            out.append(tuple((prefix + [remaining])[i] / units for i in range(n)))
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, depth + 1)

    rec([], units, 0)
    return out


def choose_weights(
    rows: list[dict[str, Any]],
    names: list[str],
    grid: list[tuple[float, ...]],
    norms: list[str],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for norm in norms:
        for weights in grid:
            stats = eval_weights(rows, names, weights, norm)
            cand = {"norm": norm, "weights": weights, **{k: v for k, v in stats.items() if k != "predictions"}}
            key = (cand["accuracy"], -cand["mean_nll"], cand["mean_margin"], -cand["mean_rank"])
            if best is None or key > best["_key"]:
                cand["_key"] = key
                best = cand
    assert best is not None
    best.pop("_key", None)
    best["weights"] = [float(x) for x in best["weights"]]
    return best


def per_type_from_predictions(items: list[dict[str, Any]], predictions: list[int], base_name: str) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for item, pred in zip(items, predictions):
        groups[item[base_name]["question_type"]].append(int(pred) == int(item[base_name]["answer_index"]))
    return {key: summarize_bool(vals) for key, vals in sorted(groups.items())}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# VideoChat2-HD CE/Eq/Rel Overlap and Ensemble", ""]
    lines.append("## Single Runs")
    lines.append("")
    lines.append("| Run | Acc | Correct/total | Mean rank | Mean margin | Mean NLL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, stats in payload["single_run_summary"].items():
        lines.append(
            f"| {name} | {stats['accuracy']:.4f} | {stats['correct']}/{stats['total']} | "
            f"{stats['mean_rank']:.3f} | {stats['mean_margin']:.4f} | {stats['mean_nll']:.4f} |"
        )
    lines.append("")
    lines.append("## Correct-Set Overlap")
    lines.append("")
    lines.append("| CE | Eq | Rel | N |")
    lines.append("|---:|---:|---:|---:|")
    for row in payload["overlap_counts"]:
        lines.append(f"| {row['ce']} | {row['eq']} | {row['rel']} | {row['n']} |")
    lines.append("")
    oracle = payload["oracle"]
    lines.append(
        f"Oracle any-correct CE/Eq/Rel: {oracle['correct']}/{oracle['total']} = {oracle['accuracy']:.4f}"
    )
    lines.append("")
    lines.append("## Per-Type Overlap")
    for qtype, rows in payload["per_type_overlap"].items():
        lines.append("")
        lines.append(f"### {qtype}")
        lines.append("| CE | Eq | Rel | N |")
        lines.append("|---:|---:|---:|---:|")
        for row in rows:
            lines.append(f"| {row['ce']} | {row['eq']} | {row['rel']} | {row['n']} |")
    lines.append("")
    lines.append("## Margin/Rank By Overlap Group")
    lines.append("")
    lines.append("| Group | N | Run | Acc | Mean rank | Mean margin | Mean NLL |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for group, stats in payload["group_metric_summary"].items():
        for name in payload["names"]:
            vals = stats[name]
            lines.append(
                f"| {group} | {stats['n']} | {name} | {vals['accuracy']:.4f} | {vals['mean_rank']:.3f} | "
                f"{vals['mean_margin']:.4f} | {vals['mean_nll']:.4f} |"
            )
    lines.append("")
    lines.append("## Nested Logit Ensemble")
    lines.append("")
    ens = payload["nested_ensemble"]
    lines.append(
        f"Nested ensemble: {ens['correct']}/{ens['total']} = {ens['accuracy']:.4f}, "
        f"mean NLL {ens['mean_nll']:.4f}, mean margin {ens['mean_margin']:.4f}"
    )
    lines.append("")
    lines.append("| Fold | Norm | Weights | Val acc | Test acc | Test correct/total |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for row in ens["folds"]:
        weights = ", ".join(f"{name}={weight:.2f}" for name, weight in zip(payload["names"], row["weights"]))
        lines.append(
            f"| {row['fold']} | {row['norm']} | {weights} | {row['validation_accuracy']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_correct']}/{row['test_total']} |"
        )
    lines.append("")
    lines.append("## Reference Ensembles")
    lines.append("")
    lines.append("| Ensemble | Acc | Correct/total | Mean rank | Mean margin | Mean NLL | Norm | Weights |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for name, row in payload["reference_ensembles"].items():
        weights = ", ".join(f"{run}={weight:.2f}" for run, weight in zip(payload["names"], row["weights"]))
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['correct']}/{row['total']} | {row['mean_rank']:.3f} | "
            f"{row['mean_margin']:.4f} | {row['mean_nll']:.4f} | {row['norm']} | {weights} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="name=path pairs separated by commas. Use ce,eq,rel names for the standard report.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--ensemble-scores-jsonl", required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--norms", default="raw,center,zscore,logprob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = []
    for item in args.scores.split(","):
        name, path = item.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    names = [name for name, _ in pairs]
    if len(names) < 2:
        raise ValueError("Need at least two score files.")
    rows_by_run = {name: {row_key(row): metrics_for_row(row) for row in read_score_rows(path)} for name, path in pairs}
    keys = sorted(set.intersection(*(set(rows_by_run[name]) for name in names)), key=lambda x: int(x) if str(x).isdigit() else str(x))
    items = [{name: rows_by_run[name][key] for name in names} for key in keys]

    single_summary = {}
    for name in names:
        vals = [item[name] for item in items]
        single_summary[name] = {
            "accuracy": float(np.mean([x["correct"] for x in vals])),
            "correct": int(sum(x["correct"] for x in vals)),
            "total": len(vals),
            "mean_rank": float(np.mean([x["rank"] for x in vals])),
            "mean_margin": float(np.mean([x["margin"] for x in vals])),
            "mean_nll": float(np.mean([x["nll"] for x in vals])),
        }

    ce_name = "ce" if "ce" in names else names[0]
    eq_name = "eq" if "eq" in names else names[1]
    rel_name = "rel" if "rel" in names else names[min(2, len(names) - 1)]
    overlap_counter: Counter[tuple[int, int, int]] = Counter()
    per_type_counter: dict[str, Counter[tuple[int, int, int]]] = defaultdict(Counter)
    group_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oracle_correct = []
    for item in items:
        bits = (
            int(item[ce_name]["correct"]),
            int(item[eq_name]["correct"]),
            int(item[rel_name]["correct"]),
        )
        overlap_counter[bits] += 1
        per_type_counter[item[ce_name]["question_type"]][bits] += 1
        oracle_correct.append(any(bits))
        if bits == (1, 1, 1):
            group = "all_correct"
        elif bits == (0, 0, 0):
            group = "all_wrong"
        elif bits == (0, 1, 0):
            group = "eq_only"
        elif bits == (0, 0, 1):
            group = "rel_only"
        elif bits == (0, 1, 1):
            group = "eq_rel_not_ce"
        elif bits == (1, 0, 0):
            group = "ce_only"
        elif bits == (1, 1, 0):
            group = "ce_eq_not_rel"
        else:
            group = "ce_rel_not_eq"
        group_items[group].append(item)

    overlap_counts = [
        {"ce": ce, "eq": eq, "rel": rel, "n": n}
        for (ce, eq, rel), n in sorted(overlap_counter.items(), key=lambda x: x[0])
    ]
    per_type_overlap = {
        qtype: [
            {"ce": ce, "eq": eq, "rel": rel, "n": n}
            for (ce, eq, rel), n in sorted(counter.items(), key=lambda x: x[0])
        ]
        for qtype, counter in sorted(per_type_counter.items())
    }
    group_summary = {group: summarize_metrics(vals, names) for group, vals in sorted(group_items.items())}

    grid = weight_grid(len(names), args.grid_step)
    norms = [x.strip() for x in args.norms.split(",") if x.strip()]
    folds = sorted({item[names[0]]["fold"] for item in items})
    nested_rows: list[dict[str, Any]] = []
    ensemble_score_rows: list[dict[str, Any]] = []
    all_test_stats = []
    all_predictions: list[int] = []
    ordered_test_items: list[dict[str, Any]] = []
    for fold in folds:
        train_items = [item for item in items if item[names[0]]["fold"] != fold]
        test_items = [item for item in items if item[names[0]]["fold"] == fold]
        choice = choose_weights(train_items, names, grid, norms)
        test_stats = eval_weights(test_items, names, tuple(choice["weights"]), choice["norm"])
        nested_rows.append(
            {
                "fold": int(fold),
                "norm": choice["norm"],
                "weights": choice["weights"],
                "validation_accuracy": choice["accuracy"],
                "validation_nll": choice["mean_nll"],
                "test_accuracy": test_stats["accuracy"],
                "test_correct": test_stats["correct"],
                "test_total": test_stats["total"],
                "test_nll": test_stats["mean_nll"],
            }
        )
        all_test_stats.append(test_stats)
        all_predictions.extend(test_stats["predictions"])
        ordered_test_items.extend(test_items)
        for item, pred in zip(test_items, test_stats["predictions"]):
            base = item[names[0]]["row"]
            score = np.zeros_like(item[names[0]]["scores"], dtype=np.float64)
            for name, weight in zip(names, choice["weights"]):
                score = score + float(weight) * normalize_for_ensemble(item[name]["scores"], choice["norm"])
            row = {
                "split": "test",
                "stage": "final",
                "fold": int(fold),
                "row_index": int(base.get("row_index", -1)),
                "video_id": base.get("video_id"),
                "question_type": base.get("question_type", ""),
                "answer_index": int(base["answer_index"]),
                "prediction_index": int(pred),
                "correct": bool(int(pred) == int(base["answer_index"])),
                "ensemble_norm": choice["norm"],
                "ensemble_weights": {name: float(weight) for name, weight in zip(names, choice["weights"])},
                "candidates": [
                    {
                        "index": int(i),
                        "letter": base["candidates"][i].get("letter", chr(ord("A") + i)),
                        "text": base["candidates"][i].get("text", ""),
                        "score": float(score[i]),
                    }
                    for i in range(len(score))
                ],
            }
            ensemble_score_rows.append(row)

    nested_correct = sum(row["correct"] for row in all_test_stats)
    nested_total = sum(row["total"] for row in all_test_stats)
    nested_ensemble = {
        "accuracy": float(nested_correct / max(1, nested_total)),
        "correct": int(nested_correct),
        "total": int(nested_total),
        "mean_nll": float(np.average([row["mean_nll"] for row in all_test_stats], weights=[row["total"] for row in all_test_stats])),
        "mean_margin": float(np.average([row["mean_margin"] for row in all_test_stats], weights=[row["total"] for row in all_test_stats])),
        "mean_rank": float(np.average([row["mean_rank"] for row in all_test_stats], weights=[row["total"] for row in all_test_stats])),
        "per_question_type": per_type_from_predictions(ordered_test_items, all_predictions, names[0]),
        "folds": nested_rows,
    }

    equal_weights = tuple([1.0 / len(names)] * len(names))
    equal_stats = eval_weights(items, names, equal_weights, "logprob")
    full_choice = choose_weights(items, names, grid, norms)
    full_stats = eval_weights(items, names, tuple(full_choice["weights"]), full_choice["norm"])
    reference = {
        "equal_logprob": {
            "norm": "logprob",
            "weights": list(equal_weights),
            **{k: v for k, v in equal_stats.items() if k != "predictions"},
        },
        "full_oof_tuned": {
            "norm": full_choice["norm"],
            "weights": full_choice["weights"],
            **{k: v for k, v in full_stats.items() if k != "predictions"},
        },
    }

    payload = {
        "names": names,
        "single_run_summary": single_summary,
        "overlap_counts": overlap_counts,
        "per_type_overlap": per_type_overlap,
        "oracle": summarize_bool(oracle_correct),
        "group_metric_summary": group_summary,
        "nested_ensemble": nested_ensemble,
        "reference_ensembles": reference,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(Path(args.output_md), payload)
    write_jsonl(Path(args.ensemble_scores_jsonl), ensemble_score_rows)
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "nested_ensemble": nested_ensemble}, indent=2))


if __name__ == "__main__":
    main()
