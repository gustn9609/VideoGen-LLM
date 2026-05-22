#!/usr/bin/env python3
"""Probe cached Wan features on real or JSONL-described video benchmarks.

The extractor writes one H5 group per low-FPS/augmentation mode and one JSONL
row per video/mode. This script keeps that layout and trains lightweight
classifiers for each feature, mode, and task slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def random_project(x: np.ndarray, max_dim: int, seed: int) -> np.ndarray:
    if max_dim <= 0 or x.shape[1] <= max_dim:
        return x
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(max_dim), size=(x.shape[1], max_dim)).astype(np.float32)
    return x @ projection


def few_shot_train_indices(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    if per_class <= 0:
        return np.arange(len(y), dtype=np.int64)
    indices = []
    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)
        take = min(per_class, len(candidates))
        indices.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def make_split(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if args.split_column and args.split_column in rows[0]:
        train_idx = np.asarray(
            [i for i, row in enumerate(rows) if str(row.get(args.split_column)) == args.train_split],
            dtype=np.int64,
        )
        test_idx = np.asarray(
            [i for i, row in enumerate(rows) if str(row.get(args.split_column)) == args.test_split],
            dtype=np.int64,
        )
        if len(train_idx) and len(test_idx):
            return train_idx, test_idx

    all_idx = np.arange(len(rows), dtype=np.int64)
    class_counts = np.bincount(labels)
    stratify = labels if len(class_counts) > 1 and np.all(class_counts[class_counts > 0] >= 2) else None
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=args.test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, classifier: str, seed: int) -> np.ndarray:
    if classifier == "linear":
        model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    elif classifier == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256,),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                random_state=seed,
            ),
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    model.fit(x_train, y_train)
    return model.predict(x_test)


def task_slices(rows: list[dict[str, Any]], task_column: str | None) -> dict[str, list[int]]:
    if not task_column or task_column not in rows[0]:
        return {"all": list(range(len(rows)))}
    output: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        output.setdefault(str(row.get(task_column)), []).append(idx)
    return output


def evaluate_mode_feature(
    h5: h5py.File,
    mode: str,
    mode_rows: list[dict[str, Any]],
    feature_name: str,
    args: argparse.Namespace,
    seed: int,
) -> list[dict[str, Any]]:
    if mode not in h5 or feature_name not in h5[mode]:
        return []
    if args.label_column not in mode_rows[0]:
        raise KeyError(f"Label column '{args.label_column}' missing from metadata")

    features = flatten_features(h5[mode][feature_name][:])
    features = random_project(features, args.max_feature_dim, seed)
    labels_text = np.asarray([str(row[args.label_column]) for row in mode_rows])
    labels = LabelEncoder().fit_transform(labels_text)
    if len(np.unique(labels)) < 2:
        return []

    results = []
    rng = np.random.default_rng(seed)
    for task_name, indices in task_slices(mode_rows, args.task_column).items():
        indices_np = np.asarray(indices, dtype=np.int64)
        if len(np.unique(labels[indices_np])) < 2 or len(indices_np) < 4:
            continue
        rows_slice = [mode_rows[i] for i in indices]
        train_local, test_local = make_split(rows_slice, labels[indices_np], args, seed)
        train_idx = indices_np[train_local]
        test_idx = indices_np[test_local]
        shot_idx = few_shot_train_indices(labels[train_idx], args.few_shot_per_class, rng)
        train_idx = train_idx[shot_idx]
        if len(np.unique(labels[train_idx])) < 2 or len(test_idx) == 0:
            continue

        for classifier in args.classifiers:
            pred = fit_predict(features[train_idx], labels[train_idx], features[test_idx], classifier, seed)
            results.append(
                {
                    "mode": mode,
                    "task": task_name,
                    "feature": feature_name,
                    "classifier": classifier,
                    "accuracy": float(accuracy_score(labels[test_idx], pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], pred)),
                    "feature_dim": int(features.shape[1]),
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "classes": int(len(np.unique(labels[indices_np]))),
                    "confusion_matrix": confusion_matrix(labels[test_idx], pred).tolist(),
                }
            )
    return results


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Wan Real Probe Results",
        "",
        "| mode | task | feature | classifier | acc | bal_acc | train | test | dim |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(results, key=lambda x: (x["mode"], x["task"], x["feature"], x["classifier"])):
        lines.append(
            "| {mode} | {task} | {feature} | {classifier} | {accuracy:.3f} | {balanced_accuracy:.3f} | {train_size} | {test_size} | {feature_dim} |".format(
                **item
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--feature-names", default="wan_vae_global_sequence,wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence")
    parser.add_argument("--modes", default="")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--task-column", default="")
    parser.add_argument("--split-column", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--few-shot-per-class", type=int, default=0)
    parser.add_argument("--classifiers", default="linear,mlp")
    parser.add_argument("--max-feature-dim", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names = [item.strip() for item in args.feature_names.split(",") if item.strip()]
    classifiers = [item.strip() for item in args.classifiers.split(",") if item.strip()]
    args.classifiers = classifiers
    rows = read_jsonl(Path(args.metadata_jsonl))
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_mode.setdefault(str(row.get("lowfps_mode", "none")), []).append(row)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    requested_modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    results: list[dict[str, Any]] = []
    with h5py.File(args.features_h5, "r") as h5:
        modes = requested_modes or list(h5.keys())
        for mode in modes:
            if mode not in rows_by_mode:
                continue
            for feature_name in feature_names:
                results.extend(evaluate_mode_feature(h5, mode, rows_by_mode[mode], feature_name, args, args.seed))

    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), results)
    print(json.dumps({"results": len(results), "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
