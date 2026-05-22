#!/usr/bin/env python3
"""Repeated CV probe with counts, CI, confusion, paired stats, and low-FPS consistency."""

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
    mcnemar_exact,
    metadata_features,
    paired_bootstrap_diff,
    random_project,
    read_jsonl,
    rows_by_mode,
)


def make_cv(labels: np.ndarray, folds: int, repeats: int, seed: int):
    class_counts = np.bincount(labels)
    max_folds = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 2
    folds = max(2, min(folds, max_folds))
    if repeats <= 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(labels)), labels)
    return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(np.arange(len(labels)), labels)


def fit_predict(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    model.fit(x[train_idx], y[train_idx])
    return model.predict(x[test_idx])


def load_matrix(
    h5: h5py.File,
    mode: str,
    feature: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    transform: str = "full",
) -> np.ndarray | None:
    if feature == "metadata_only":
        x, _ = metadata_features(rows)
        return x.astype(np.float32)
    if mode not in h5 or feature not in h5[mode]:
        return None
    x = flatten_dataset(h5[mode][feature][:], rows, feature, transform=transform, seed=args.seed)
    return random_project(x, args.max_feature_dim, args.seed + abs(hash((mode, feature, transform))) % 100000).astype(np.float32)


def evaluate_feature(
    h5: h5py.File,
    rows_by: dict[str, list[dict[str, Any]]],
    labels_by: dict[str, np.ndarray],
    label_names: list[str],
    mode: str,
    feature: str,
    args: argparse.Namespace,
    transform: str = "full",
) -> dict[str, Any] | None:
    rows = rows_by.get(mode)
    labels = labels_by.get(mode)
    if rows is None or labels is None or len(np.unique(labels)) < 2:
        return None
    x = load_matrix(h5, mode, feature, rows, args, transform=transform)
    if x is None:
        return None

    split_results = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for split_id, (train_idx, test_idx) in enumerate(make_cv(labels, args.folds, args.repeats, args.seed)):
        pred = fit_predict(x, labels, train_idx, test_idx)
        truth = labels[test_idx]
        split_results.append(
            {
                "split": split_id,
                "accuracy": float(accuracy_score(truth, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                "correct": int((truth == pred).sum()),
                "total": int(len(test_idx)),
                "test_indices": [int(i) for i in test_idx.tolist()],
                "pred": [int(i) for i in pred.tolist()],
                "true": [int(i) for i in truth.tolist()],
            }
        )
        all_true.extend(truth.tolist())
        all_pred.extend(pred.tolist())

    correct_arr = (np.asarray(all_true) == np.asarray(all_pred)).astype(np.float32)
    ci_low, ci_high = bootstrap_ci(correct_arr, seed=args.seed, n_boot=args.bootstrap)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(label_names)))).tolist()
    return {
        "mode": mode,
        "feature": feature,
        "transform": transform,
        "classes": label_names,
        "splits": split_results,
        "accuracy_mean": float(np.mean([x["accuracy"] for x in split_results])),
        "accuracy_std": float(np.std([x["accuracy"] for x in split_results], ddof=0)),
        "balanced_accuracy_mean": float(np.mean([x["balanced_accuracy"] for x in split_results])),
        "balanced_accuracy_std": float(np.std([x["balanced_accuracy"] for x in split_results], ddof=0)),
        "correct": int(correct_arr.sum()),
        "total": int(correct_arr.size),
        "accuracy_ci95": [ci_low, ci_high],
        "confusion_matrix": cm,
    }


def split_correct_vector(result: dict[str, Any]) -> np.ndarray:
    vals = []
    for split in result["splits"]:
        vals.extend([int(t == p) for t, p in zip(split["true"], split["pred"])])
    return np.asarray(vals, dtype=np.int64)


def consistency(result_a: dict[str, Any], result_b: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for split_a, split_b in zip(result_a["splits"], result_b["splits"]):
        pred_a = dict(zip(split_a["test_indices"], split_a["pred"]))
        true_a = dict(zip(split_a["test_indices"], split_a["true"]))
        pred_b = dict(zip(split_b["test_indices"], split_b["pred"]))
        for idx in sorted(set(pred_a) & set(pred_b)):
            pairs.append((true_a[idx], pred_a[idx], pred_b[idx]))
    if not pairs:
        return {}
    arr = np.asarray(pairs, dtype=np.int64)
    same = arr[:, 1] == arr[:, 2]
    a_correct = arr[:, 0] == arr[:, 1]
    b_correct = arr[:, 0] == arr[:, 2]
    return {
        "paired_total": int(len(arr)),
        "prediction_consistency": float(same.mean()),
        "normal_only_correct": int((a_correct & ~b_correct).sum()),
        "lowfps_only_correct": int((~a_correct & b_correct).sum()),
        "both_correct": int((a_correct & b_correct).sum()),
        "both_wrong": int((~a_correct & ~b_correct).sum()),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench Stable Probe", ""]
    lines.append("| mode | feature | transform | acc mean | std | CI low | CI high | correct/total |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for item in payload["results"]:
        ci = item["accuracy_ci95"]
        lines.append(
            f"| {item['mode']} | {item['feature']} | {item['transform']} | "
            f"{item['accuracy_mean']:.4f} | {item['accuracy_std']:.4f} | {ci[0]:.4f} | {ci[1]:.4f} | "
            f"{item['correct']}/{item['total']} |"
        )
    if payload.get("paired"):
        lines.extend(["", "## Paired Comparisons", ""])
        lines.append("| mode | feature | baseline | diff | CI low | CI high | McNemar p |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for item in payload["paired"]:
            lines.append(
                f"| {item['mode']} | {item['feature']} | {item['baseline']} | "
                f"{item['mean_diff']:.4f} | {item['ci_low']:.4f} | {item['ci_high']:.4f} | {item['mcnemar_p']:.4f} |"
            )
    if payload.get("consistency"):
        lines.extend(["", "## Low-FPS Consistency", ""])
        lines.append("| feature | lowfps mode | consistency | normal only | low only | both correct | both wrong |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for item in payload["consistency"]:
            lines.append(
                f"| {item['feature']} | {item['lowfps_mode']} | {item['prediction_consistency']:.4f} | "
                f"{item['normal_only_correct']} | {item['lowfps_only_correct']} | {item['both_correct']} | {item['both_wrong']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--feature-names", default="wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence")
    parser.add_argument("--modes", default="none,uniform5,nonuniform5")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-feature-dim", type=int, default=4096)
    parser.add_argument("--baseline-feature", default="pixel_grid_sequence")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.metadata_jsonl))
    rows_by = rows_by_mode(rows)
    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    label_encoders: dict[str, LabelEncoder] = {}
    labels_by: dict[str, np.ndarray] = {}
    label_names: list[str] | None = None
    for mode, mode_rows in rows_by.items():
        if not mode_rows or args.label_column not in mode_rows[0]:
            continue
        encoder = LabelEncoder()
        labels = encoder.fit_transform([str(row[args.label_column]) for row in mode_rows])
        label_encoders[mode] = encoder
        labels_by[mode] = labels
        label_names = [str(x) for x in encoder.classes_.tolist()]

    results = []
    with h5py.File(args.features_h5, "r") as h5:
        for mode in modes:
            for feature in feature_names:
                result = evaluate_feature(h5, rows_by, labels_by, label_names or [], mode, feature, args)
                if result is not None:
                    results.append(result)

    result_map = {(item["mode"], item["feature"], item["transform"]): item for item in results}
    paired = []
    for item in results:
        base = result_map.get((item["mode"], args.baseline_feature, item["transform"]))
        if base is None or base is item:
            continue
        a = split_correct_vector(item)
        b = split_correct_vector(base)
        diff = paired_bootstrap_diff(a, b, seed=args.seed, n_boot=args.bootstrap)
        mc = mcnemar_exact(a, b)
        paired.append(
            {
                "mode": item["mode"],
                "feature": item["feature"],
                "baseline": args.baseline_feature,
                **diff,
                "mcnemar_p": mc["p_value"],
                "mcnemar": mc,
            }
        )

    consistency_rows = []
    for feature in feature_names:
        normal = result_map.get(("none", feature, "full"))
        if normal is None:
            continue
        for mode in modes:
            if mode == "none":
                continue
            low = result_map.get((mode, feature, "full"))
            if low is None:
                continue
            item = consistency(normal, low)
            if item:
                item.update({"feature": feature, "lowfps_mode": mode})
                consistency_rows.append(item)

    payload = {
        "config": vars(args),
        "results": results,
        "paired": paired,
        "consistency": consistency_rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({"results": len(results), "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
