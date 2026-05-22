#!/usr/bin/env python3
"""Calibrated score-level residual ensemble for MotionBench candidate scores."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import normalize_scores, read_jsonl  # noqa: E402
from motionbench_repa_common import candidate_scores_from_row, score_row_key  # noqa: E402


def parse_system_spec(spec: str) -> dict[str, str]:
    if "=" not in spec:
        raise ValueError(f"System spec must start with name=path: {spec}")
    name, rest = spec.split("=", 1)
    parts = [part.strip() for part in rest.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"System spec has no path: {spec}")
    out = {"name": name.strip(), "path": parts[0], "mode": "", "feature": ""}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load_system(spec: dict[str, str], include_split: bool) -> dict[tuple[str, ...], dict[str, Any]]:
    rows = read_jsonl(Path(spec["path"]))
    if spec.get("mode"):
        rows = [row for row in rows if str(row.get("mode", "")) == spec["mode"]]
    if spec.get("feature"):
        rows = [row for row in rows if str(row.get("feature", "")) == spec["feature"]]
    out = {}
    for row in rows:
        key = score_row_key(row, include_split=include_split)
        if key not in out:
            out[key] = row
    return out


def correct_index(row: dict[str, Any]) -> int:
    if "answer_index" in row:
        return int(row["answer_index"])
    if "correct_index" in row:
        return int(row["correct_index"])
    answer = str(row.get("answer", "A")).strip().upper()
    if len(answer) == 1 and "A" <= answer <= "Z":
        return ord(answer) - ord("A")
    return int(row.get("label", 0))


def build_records(system_maps: dict[str, dict[tuple[str, ...], dict[str, Any]]], norm: str, temperature: float) -> list[dict[str, Any]]:
    common_keys = set.intersection(*(set(rows.keys()) for rows in system_maps.values())) if system_maps else set()
    records = []
    for key in sorted(common_keys):
        base_row = next(iter(system_maps.values()))[key]
        scores = {}
        raw_scores = {}
        n_candidates = None
        ok = True
        for name, rows in system_maps.items():
            row = rows[key]
            arr = candidate_scores_from_row(row)
            if n_candidates is None:
                n_candidates = len(arr)
            elif len(arr) != n_candidates:
                ok = False
                break
            raw_scores[name] = arr.astype(np.float64)
            scores[name] = normalize_scores(arr, norm, temperature).astype(np.float64)
        if not ok or n_candidates is None or n_candidates < 2:
            continue
        truth = correct_index(base_row)
        if truth < 0 or truth >= n_candidates:
            continue
        records.append(
            {
                "key": key,
                "row": base_row,
                "truth": truth,
                "question_type": str(base_row.get("question_type", "all")),
                "scores": scores,
                "raw_scores": raw_scores,
                "n_candidates": n_candidates,
            }
        )
    return records


def predictions_for_weight(records: list[dict[str, Any]], names: list[str], weights: np.ndarray) -> list[int]:
    preds = []
    for rec in records:
        score = np.zeros((rec["n_candidates"],), dtype=np.float64)
        for name, weight in zip(names, weights):
            score += float(weight) * rec["scores"][name]
        preds.append(int(np.argmax(score)))
    return preds


def metric(records: list[dict[str, Any]], preds: list[int], hard_mask: np.ndarray | None = None) -> dict[str, Any]:
    if hard_mask is None:
        use = np.ones((len(records),), dtype=bool)
    else:
        use = hard_mask.astype(bool)
    truths = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    pred_arr = np.asarray(preds, dtype=np.int64)
    correct = (pred_arr == truths)
    by_type: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        if not use[i]:
            continue
        by_type.setdefault(rec["question_type"], []).append(int(correct[i]))
    selected = correct[use]
    return {
        "accuracy": float(selected.mean()) if selected.size else 0.0,
        "correct": int(selected.sum()) if selected.size else 0,
        "total": int(selected.size),
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(by_type.items())
        },
        "correct_vector": correct.astype(int).tolist(),
    }


def hard_subset(records: list[dict[str, Any]], text_name: str, margin_quantile: float) -> np.ndarray:
    if text_name not in records[0]["scores"]:
        text_name = next(iter(records[0]["scores"]))
    margins = []
    wrong = []
    for rec in records:
        score = rec["scores"][text_name]
        order = np.sort(score)
        margin = float(order[-1] - order[-2]) if len(order) > 1 else 0.0
        margins.append(margin)
        wrong.append(int(np.argmax(score)) != rec["truth"])
    threshold = float(np.quantile(np.asarray(margins), margin_quantile)) if margins else 0.0
    return np.asarray(wrong, dtype=bool) | (np.asarray(margins) <= threshold)


def train_calibrated(records: list[dict[str, Any]], names: list[str], args: argparse.Namespace) -> tuple[list[int], dict[str, Any]]:
    labels = np.asarray([rec["question_type"] for rec in records])
    enc = LabelEncoder()
    y_strat = enc.fit_transform(labels)
    counts = np.bincount(y_strat)
    folds = max(2, min(args.folds, int(counts[counts > 0].min()) if np.any(counts > 0) else 2))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    preds = np.zeros((len(records),), dtype=np.int64)
    split_rows = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(records)), y_strat)):
        x_train = []
        y_train = []
        for rec_idx in train_idx:
            rec = records[int(rec_idx)]
            for cand_idx in range(rec["n_candidates"]):
                x_train.append([rec["scores"][name][cand_idx] for name in names])
                y_train.append(1 if cand_idx == rec["truth"] else 0)
        if args.classifier == "ridge":
            clf = RidgeClassifier(alpha=args.ridge_alpha)
        else:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=args.logistic_c, random_state=args.seed + split_id)
        model = make_pipeline(StandardScaler(), clf)
        model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.int64))
        split_pred = []
        split_true = []
        for rec_idx in test_idx:
            rec = records[int(rec_idx)]
            x = np.asarray([[rec["scores"][name][cand_idx] for name in names] for cand_idx in range(rec["n_candidates"])], dtype=np.float64)
            final = model.steps[-1][1]
            if hasattr(model, "decision_function"):
                score = model.decision_function(x)
            else:
                prob = model.predict_proba(x)
                cls = list(getattr(final, "classes_", [0, 1]))
                pos = cls.index(1) if 1 in cls else min(1, prob.shape[1] - 1)
                score = prob[:, pos]
            if np.asarray(score).ndim == 2:
                cls = list(getattr(final, "classes_", [0, 1]))
                pos = cls.index(1) if 1 in cls else min(1, score.shape[1] - 1)
                score = score[:, pos]
            pred = int(np.argmax(score))
            preds[int(rec_idx)] = pred
            split_pred.append(pred)
            split_true.append(rec["truth"])
        split_rows.append(
            {
                "split": split_id,
                "accuracy": float(accuracy_score(split_true, split_pred)),
                "correct": int((np.asarray(split_true) == np.asarray(split_pred)).sum()),
                "total": int(len(split_true)),
            }
        )
    return preds.tolist(), {"splits": split_rows}


def complementarity(records: list[dict[str, Any]], names: list[str], system_preds: dict[str, list[int]]) -> dict[str, Any]:
    truths = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    correct = {name: np.asarray(system_preds[name]) == truths for name in names}
    oracle = np.zeros((len(records),), dtype=bool)
    for name in names:
        oracle |= correct[name]
    out: dict[str, Any] = {
        "oracle_union_accuracy": float(oracle.mean()) if oracle.size else 0.0,
        "oracle_union_correct": int(oracle.sum()),
        "oracle_union_total": int(oracle.size),
    }
    if {"text", "pixel", "flow"}.issubset(correct):
        base_wrong = ~correct["text"] & ~correct["pixel"] & ~correct["flow"]
        for name in names:
            if name not in {"text", "pixel", "flow"}:
                out[f"text_pixel_flow_wrong_{name}_right"] = int((base_wrong & correct[name]).sum())
    for a, b in itertools.combinations(names, 2):
        out[f"{a}_only_vs_{b}"] = int((correct[a] & ~correct[b]).sum())
        out[f"{b}_only_vs_{a}"] = int((correct[b] & ~correct[a]).sum())
    return out


def short_type_metrics(item: dict[str, Any]) -> dict[str, float]:
    per = item.get("per_question_type", {})
    return {
        "Action Order": per.get("Action Order", {}).get("accuracy", 0.0),
        "Repetition Count": per.get("Repetition Count", {}).get("accuracy", 0.0),
        "Motion Recognition": per.get("Motion Recognition", {}).get("accuracy", 0.0),
        "Motion Objects": per.get("Motion-related Objects", {}).get("accuracy", 0.0),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench Residual Score Ensemble", ""]
    lines.append("| Model | Overall | Hard subset | Action Order | Repetition Count | Motion Recognition | Motion Objects | Correct/total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in payload["models"].items():
        tm = short_type_metrics(item)
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item.get('hard_accuracy', 0.0):.4f} | "
            f"{tm['Action Order']:.4f} | {tm['Repetition Count']:.4f} | {tm['Motion Recognition']:.4f} | "
            f"{tm['Motion Objects']:.4f} | {item['correct']}/{item['total']} |"
        )
    if payload.get("weight_sweep"):
        lines.extend(["", "## Best Weight Sweep", ""])
        lines.append("| rank | weights | acc | hard acc | correct/total |")
        lines.append("|---:|---|---:|---:|---:|")
        for idx, item in enumerate(payload["weight_sweep"][:20], start=1):
            weight_text = ", ".join(f"{k}={v:.3g}" for k, v in item["weights"].items())
            lines.append(f"| {idx} | {weight_text} | {item['accuracy']:.4f} | {item['hard_accuracy']:.4f} | {item['correct']}/{item['total']} |")
    if payload.get("complementarity"):
        lines.extend(["", "## Complementarity", ""])
        for key, value in sorted(payload["complementarity"].items()):
            lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, help="name=score.jsonl[,mode=...][,feature=...]")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--score-normalization", default="zscore", choices=["raw", "zscore", "rank", "softmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-split-in-key", action="store_true")
    parser.add_argument("--text-system", default="text")
    parser.add_argument("--hard-margin-quantile", type=float, default=0.35)
    parser.add_argument("--weight-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--max-weight-combinations", type=int, default=20000)
    parser.add_argument("--classifier", default="logistic", choices=["logistic", "ridge"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [parse_system_spec(spec) for spec in args.system]
    names = [spec["name"] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"System names must be unique: {names}")
    system_maps = {spec["name"]: load_system(spec, include_split=args.include_split_in_key) for spec in specs}
    records = build_records(system_maps, args.score_normalization, args.temperature)
    if not records:
        raise ValueError("No aligned score rows found across systems")
    hard = hard_subset(records, args.text_system, args.hard_margin_quantile)
    models: dict[str, Any] = {}
    system_preds: dict[str, list[int]] = {}
    for name in names:
        preds = [int(np.argmax(rec["scores"][name])) for rec in records]
        system_preds[name] = preds
        item = metric(records, preds)
        item["hard_accuracy"] = metric(records, preds, hard)["accuracy"]
        models[name] = {k: v for k, v in item.items() if k != "correct_vector"}

    calibrated_preds, calibrated_extra = train_calibrated(records, names, args)
    calibrated = metric(records, calibrated_preds)
    calibrated["hard_accuracy"] = metric(records, calibrated_preds, hard)["accuracy"]
    calibrated.update(calibrated_extra)
    models["calibrated_residual_ensemble"] = {k: v for k, v in calibrated.items() if k != "correct_vector"}

    weights = [float(x) for x in args.weight_grid.split(",") if x.strip()]
    combos = itertools.product(weights, repeat=len(names))
    sweep = []
    for combo_id, combo in enumerate(combos):
        if combo_id >= args.max_weight_combinations:
            break
        combo_arr = np.asarray(combo, dtype=np.float64)
        if float(np.abs(combo_arr).sum()) <= 1e-12:
            continue
        preds = predictions_for_weight(records, names, combo_arr)
        item = metric(records, preds)
        hard_item = metric(records, preds, hard)
        sweep.append(
            {
                "weights": {name: float(weight) for name, weight in zip(names, combo_arr)},
                "accuracy": item["accuracy"],
                "hard_accuracy": hard_item["accuracy"],
                "correct": item["correct"],
                "total": item["total"],
            }
        )
    sweep.sort(key=lambda x: (x["accuracy"], x["hard_accuracy"]), reverse=True)

    payload = {
        "config": vars(args),
        "systems": specs,
        "rows": len(records),
        "hard_subset_rows": int(hard.sum()),
        "models": models,
        "weight_sweep": sweep,
        "complementarity": complementarity(records, names, system_preds),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({"rows": len(records), "systems": names, "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
