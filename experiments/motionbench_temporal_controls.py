#!/usr/bin/env python3
"""Temporal and metadata shortcut controls for MotionBench cached features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import (  # noqa: E402
    bootstrap_ci,
    flatten_dataset,
    metadata_features,
    random_project,
    read_jsonl,
    rows_by_mode,
)


def make_cv(labels: np.ndarray, folds: int, repeats: int, seed: int):
    counts = np.bincount(labels)
    max_folds = int(counts[counts > 0].min()) if np.any(counts > 0) else 2
    folds = max(2, min(folds, max_folds))
    if repeats <= 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(labels)), labels)
    return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(np.arange(len(labels)), labels)


def evaluate_matrix(x: np.ndarray, labels: np.ndarray, label_names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    split_rows = []
    all_true, all_pred = [], []
    for split_id, (train_idx, test_idx) in enumerate(make_cv(labels, args.folds, args.repeats, args.seed)):
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        model.fit(x[train_idx], labels[train_idx])
        pred = model.predict(x[test_idx])
        truth = labels[test_idx]
        split_rows.append(
            {
                "split": split_id,
                "accuracy": float(accuracy_score(truth, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                "correct": int((truth == pred).sum()),
                "total": int(len(test_idx)),
            }
        )
        all_true.extend(truth.tolist())
        all_pred.extend(pred.tolist())
    correct = (np.asarray(all_true) == np.asarray(all_pred)).astype(np.float32)
    ci = bootstrap_ci(correct, seed=args.seed, n_boot=args.bootstrap)
    return {
        "accuracy_mean": float(np.mean([x["accuracy"] for x in split_rows])),
        "accuracy_std": float(np.std([x["accuracy"] for x in split_rows], ddof=0)),
        "balanced_accuracy_mean": float(np.mean([x["balanced_accuracy"] for x in split_rows])),
        "balanced_accuracy_std": float(np.std([x["balanced_accuracy"] for x in split_rows], ddof=0)),
        "correct": int(correct.sum()),
        "total": int(correct.size),
        "accuracy_ci95": [ci[0], ci[1]],
        "confusion_matrix": confusion_matrix(all_true, all_pred, labels=list(range(len(label_names))).copy()).tolist(),
        "splits": split_rows,
    }


def condition_matrix(
    h5: h5py.File,
    rows_by: dict[str, list[dict[str, Any]]],
    condition: str,
    feature: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, list[dict[str, Any]] | None, str]:
    if condition == "metadata_only":
        rows = rows_by.get("none") or next(iter(rows_by.values()))
        x, _ = metadata_features(rows)
        return x, rows, "metadata"

    mode = "none"
    transform = "full"
    if condition in {"normal", "first_frame_only", "time_average", "shuffled", "reversed"}:
        transform = "full" if condition == "normal" else condition
    elif condition in h5:
        mode = condition
    elif condition in {"uniform5", "nonuniform5", "none+crop_shift", "none+blur", "none+jpeg45"}:
        return None, None, f"missing mode {condition}"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    rows = rows_by.get(mode)
    if rows is None or mode not in h5 or feature not in h5[mode]:
        return None, None, f"missing feature {feature} in mode {mode}"
    x = flatten_dataset(h5[mode][feature][:], rows, feature, transform=transform, seed=args.seed)
    x = random_project(x, args.max_feature_dim, args.seed + abs(hash((condition, feature))) % 100000)
    return x.astype(np.float32), rows, ""


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench Temporal Controls", ""]
    lines.append("| feature | condition | acc mean | std | CI low | CI high | correct/total | normal gap |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for item in payload["results"]:
        ci = item["accuracy_ci95"]
        gap = item.get("normal_gap")
        gap_text = "" if gap is None else f"{gap:.4f}"
        lines.append(
            f"| {item['feature']} | {item['condition']} | {item['accuracy_mean']:.4f} | "
            f"{item['accuracy_std']:.4f} | {ci[0]:.4f} | {ci[1]:.4f} | {item['correct']}/{item['total']} | {gap_text} |"
        )
    if payload.get("skipped"):
        lines.extend(["", "## Skipped Conditions", ""])
        for item in payload["skipped"]:
            lines.append(f"- {item['feature']} / {item['condition']}: {item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--feature-names", default="wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence")
    parser.add_argument(
        "--conditions",
        default="normal,first_frame_only,time_average,shuffled,reversed,uniform5,nonuniform5,metadata_only,none+crop_shift",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-feature-dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.metadata_jsonl))
    rows_by = rows_by_mode(rows)
    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]

    results = []
    skipped = []
    normal_acc: dict[str, float] = {}
    with h5py.File(args.features_h5, "r") as h5:
        for feature in feature_names:
            for condition in conditions:
                x, cond_rows, reason = condition_matrix(h5, rows_by, condition, feature, args)
                if x is None or cond_rows is None:
                    skipped.append({"feature": feature, "condition": condition, "reason": reason})
                    continue
                if args.label_column not in cond_rows[0]:
                    skipped.append({"feature": feature, "condition": condition, "reason": f"missing label column {args.label_column}"})
                    continue
                encoder = LabelEncoder()
                labels = encoder.fit_transform([str(row[args.label_column]) for row in cond_rows])
                if len(np.unique(labels)) < 2:
                    skipped.append({"feature": feature, "condition": condition, "reason": "fewer than two classes"})
                    continue
                item = evaluate_matrix(x, labels, [str(c) for c in encoder.classes_.tolist()], args)
                item.update({"feature": feature, "condition": condition, "dim": int(x.shape[1])})
                if condition == "normal":
                    normal_acc[feature] = item["accuracy_mean"]
                    item["normal_gap"] = 0.0
                else:
                    item["normal_gap"] = normal_acc.get(feature, float("nan")) - item["accuracy_mean"]
                results.append(item)

    payload = {"config": vars(args), "results": results, "skipped": skipped}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({"results": len(results), "skipped": len(skipped), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
