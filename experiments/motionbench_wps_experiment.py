#!/usr/bin/env python3
"""Real MotionBench Wan Perturbation Signature (WPS) experiment.

Builds candidate-level perturbation-signature features from cached normal /
shuffled / reversed / static-control score JSONL files and evaluates:

- raw Wan vs WPS vs raw Wan + WPS
- pixel WPS / flow WPS
- text / pixel / flow / text+pixel+flow baselines
- text+pixel+flow+WPS calibrated score model
- held-out WPS gate switch
- temporal-sensitive subset and normal-shuffle/reverse gaps
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import flatten_dataset, read_jsonl, rows_by_mode  # noqa: E402


FEATURE_BY_FAMILY = {
    "wan": "wan_vae_grid_1x1",
    "pixel": "pixel_grid_sequence",
    "flow": "flow_grid_sequence",
    "text": "text_only",
}


def read_scores(path: Path, mode: str, feature: str) -> dict[tuple[int, str], dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("mode")) != mode or str(row.get("feature")) != feature:
                continue
            key = (int(row["row_index"]), str(row.get("video_id", "")))
            if key not in out:
                out[key] = row
    return out


def candidate_scores(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(c.get("score", 0.0)) for c in row["candidates"]], dtype=np.float64)


def zscore_per_row(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    std = float(scores.std())
    return (scores - float(scores.mean())) / (std if std > 1e-8 else 1.0)


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(scores))
    return order.astype(np.float64) / max(1, len(scores) - 1)


def softmax(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    x = x - float(x.max())
    exp = np.exp(x)
    return exp / max(float(exp.sum()), 1e-12)


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    p = softmax(a)
    q = softmax(b)
    m = 0.5 * (p + q)
    kl_pm = float((p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(m, 1e-12)))).sum())
    kl_qm = float((q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(m, 1e-12)))).sum())
    return 0.5 * (kl_pm + kl_qm)


def margin(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    ordered = np.sort(scores)
    return float(ordered[-1] - ordered[-2])


def pred(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def normalize_scores(scores: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "raw":
        return np.asarray(scores, dtype=np.float64)
    if method == "zscore":
        return zscore_per_row(scores)
    if method == "rank":
        return ranks(scores)
    raise ValueError(method)


def score_row_metric(records: list[dict[str, Any]], score_map: dict[tuple[int, str], np.ndarray], mask: np.ndarray | None = None) -> dict[str, Any]:
    use = np.ones((len(records),), dtype=bool) if mask is None else mask.astype(bool)
    correct = []
    by_type: dict[str, list[int]] = {}
    preds = []
    for i, rec in enumerate(records):
        scores = score_map[rec["key"]]
        p = pred(scores)
        preds.append(p)
        val = int(p == rec["truth"])
        if use[i]:
            correct.append(val)
            by_type.setdefault(rec["question_type"], []).append(val)
    arr = np.asarray(correct, dtype=np.float64)
    return {
        "accuracy": float(arr.mean()) if arr.size else 0.0,
        "correct": int(arr.sum()) if arr.size else 0,
        "total": int(arr.size),
        "predictions": preds,
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(by_type.items())
        },
    }


def load_all_scores(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[tuple[int, str], np.ndarray]]]]:
    transform_paths = {
        "normal": Path(args.normal_scores_jsonl),
        "shuffle": Path(args.shuffle_scores_jsonl),
        "reverse": Path(args.reverse_scores_jsonl),
        "first": Path(args.first_scores_jsonl),
        "timeavg": Path(args.timeavg_scores_jsonl),
    }
    families = ["text", "wan", "pixel", "flow"]
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]] = {family: {} for family in families}
    rows_by_family_transform: dict[str, dict[str, dict[tuple[int, str], dict[str, Any]]]] = {family: {} for family in families}
    for family in families:
        feature = FEATURE_BY_FAMILY[family]
        for transform, path in transform_paths.items():
            loaded = read_scores(path, args.mode, feature)
            rows_by_family_transform[family][transform] = loaded
            scores[family][transform] = {key: candidate_scores(row) for key, row in loaded.items()}

    common = set(rows_by_family_transform["text"]["normal"].keys())
    for family in families:
        for transform in transform_paths:
            common &= set(rows_by_family_transform[family][transform].keys())
    records = []
    for key in sorted(common):
        row = rows_by_family_transform["text"]["normal"][key]
        records.append(
            {
                "key": key,
                "row_index": int(row["row_index"]),
                "video_id": str(row.get("video_id", "")),
                "question_type": str(row.get("question_type", "all")),
                "truth": int(row["answer_index"]),
                "n_candidates": len(row["candidates"]),
            }
        )
    return records, scores


def feature_distances(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, dict[str, dict[tuple[int, str], float]]]:
    if not args.features_h5 or not args.metadata_jsonl:
        return {}
    metadata_rows = read_jsonl(Path(args.metadata_jsonl))
    grouped = rows_by_mode(metadata_rows)
    mode_rows = grouped.get(args.mode, [])
    for idx, row in enumerate(mode_rows):
        row["__h5_index"] = idx
    row_by_index = {idx: row for idx, row in enumerate(mode_rows)}
    record_indices = [rec["row_index"] for rec in records]
    out: dict[str, dict[str, dict[tuple[int, str], float]]] = {}
    with h5py.File(args.features_h5, "r") as h5:
        if args.mode not in h5:
            return out
        for family in ["wan", "pixel", "flow"]:
            feature = FEATURE_BY_FAMILY[family]
            if feature not in h5[args.mode]:
                continue
            data = h5[args.mode][feature][:]
            selected_rows = [row_by_index[i] for i in record_indices]
            selected = data[record_indices]
            base = flatten_dataset(selected, selected_rows, feature, transform="full", seed=args.seed)
            out[family] = {}
            for transform, tf_name in [("shuffle", "shuffled"), ("reverse", "reversed"), ("first", "first_frame_only"), ("timeavg", "time_average")]:
                ctrl = flatten_dataset(selected, selected_rows, feature, transform=tf_name, seed=args.seed)
                if base.shape[1] != ctrl.shape[1]:
                    continue
                dist = np.linalg.norm(base - ctrl, axis=1)
                out[family][transform] = {rec["key"]: float(val) for rec, val in zip(records, dist)}
    return out


def sample_wps_features(
    rec: dict[str, Any],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    family: str,
    anchor: str,
    distances: dict[str, dict[str, dict[tuple[int, str], float]]] | None = None,
) -> np.ndarray:
    key = rec["key"]
    anchor_scores = scores[family][anchor][key]
    controls = [name for name in ["normal", "shuffle", "reverse", "first", "timeavg"] if name != anchor]
    pieces = []
    for ctrl_name in controls:
        ctrl_scores = scores[family][ctrl_name][key]
        pieces.extend(
            [
                margin(anchor_scores) - margin(ctrl_scores),
                js_divergence(anchor_scores, ctrl_scores),
                float(pred(anchor_scores) != pred(ctrl_scores)),
                float(np.linalg.norm(zscore_per_row(anchor_scores) - zscore_per_row(ctrl_scores))),
            ]
        )
        if distances and family in distances:
            if anchor == "normal" and ctrl_name in distances[family]:
                pieces.append(distances[family][ctrl_name].get(key, 0.0))
            elif ctrl_name == "normal" and anchor in distances[family]:
                pieces.append(distances[family][anchor].get(key, 0.0))
            else:
                pieces.append(0.0)
    return np.asarray(pieces, dtype=np.float64)


def candidate_wps_matrix(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    family: str,
    anchor: str,
    include_anchor_score: bool,
    distances: dict[str, dict[str, dict[tuple[int, str], float]]] | None = None,
    extra_score_families: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y = []
    owners = []
    controls = [name for name in ["normal", "shuffle", "reverse", "first", "timeavg"] if name != anchor]
    extra_score_families = extra_score_families or []
    for row_idx, rec in enumerate(records):
        key = rec["key"]
        anchor_scores = scores[family][anchor][key]
        sample_feat = sample_wps_features(rec, scores, family, anchor, distances)
        for cand_idx in range(rec["n_candidates"]):
            pieces = []
            if include_anchor_score:
                pieces.extend([anchor_scores[cand_idx], zscore_per_row(anchor_scores)[cand_idx], ranks(anchor_scores)[cand_idx]])
            for ctrl_name in controls:
                ctrl_scores = scores[family][ctrl_name][key]
                diff = anchor_scores - ctrl_scores
                z_diff = zscore_per_row(anchor_scores) - zscore_per_row(ctrl_scores)
                rank_diff = ranks(anchor_scores) - ranks(ctrl_scores)
                pieces.extend([diff[cand_idx], z_diff[cand_idx], rank_diff[cand_idx]])
            for extra in extra_score_families:
                ex_scores = scores[extra][anchor][key]
                pieces.extend([ex_scores[cand_idx], zscore_per_row(ex_scores)[cand_idx], ranks(ex_scores)[cand_idx]])
            pieces.extend(sample_feat.tolist())
            x_rows.append(pieces)
            y.append(1 if cand_idx == rec["truth"] else 0)
            owners.append(row_idx)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y, dtype=np.int64), np.asarray(owners, dtype=np.int64)


def train_binary_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, seed: int):
    if args.classifier == "logistic":
        clf = LogisticRegression(max_iter=2000, C=args.logistic_c, class_weight="balanced", random_state=seed)
    else:
        clf = RidgeClassifier(alpha=args.ridge_alpha)
    model = make_pipeline(StandardScaler(), clf)
    model.fit(x, y)
    return model


def model_scores(model, x: np.ndarray) -> np.ndarray:
    final = model.steps[-1][1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(x)
    else:
        prob = model.predict_proba(x)
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, prob.shape[1] - 1)
        s = prob[:, pos]
    if np.asarray(s).ndim == 2:
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, s.shape[1] - 1)
        s = s[:, pos]
    return np.asarray(s, dtype=np.float64)


def predict_from_candidate_scores(records: list[dict[str, Any]], owners: np.ndarray, flat_scores: np.ndarray) -> tuple[list[int], dict[tuple[int, str], np.ndarray]]:
    score_by_key: dict[tuple[int, str], list[float]] = {rec["key"]: [] for rec in records}
    for owner, score in zip(owners, flat_scores):
        key = records[int(owner)]["key"]
        score_by_key[key].append(float(score))
    preds = []
    arr_by_key = {}
    for rec in records:
        arr = np.asarray(score_by_key[rec["key"]], dtype=np.float64)
        arr_by_key[rec["key"]] = arr
        preds.append(pred(arr))
    return preds, arr_by_key


def cv_candidate_model(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    distances: dict[str, dict[str, dict[tuple[int, str], float]]],
    args: argparse.Namespace,
    family: str,
    include_anchor_score: bool = False,
    extra_score_families: list[str] | None = None,
) -> dict[str, Any]:
    labels = np.asarray([rec["question_type"] for rec in records])
    enc = LabelEncoder()
    y_strat = enc.fit_transform(labels)
    counts = np.bincount(y_strat)
    folds = max(2, min(args.folds, int(counts[counts > 0].min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    normal_pred = np.zeros((len(records),), dtype=np.int64)
    shuffle_pred = np.zeros((len(records),), dtype=np.int64)
    reverse_pred = np.zeros((len(records),), dtype=np.int64)
    normal_score_map: dict[tuple[int, str], np.ndarray] = {}
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(records)), y_strat)):
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        x_train, y_train, _ = candidate_wps_matrix(train_records, scores, family, "normal", include_anchor_score, distances, extra_score_families)
        model = train_binary_model(x_train, y_train, args, args.seed + split_id)
        for anchor, target in [("normal", normal_pred), ("shuffle", shuffle_pred), ("reverse", reverse_pred)]:
            x_test, _, owners = candidate_wps_matrix(test_records, scores, family, anchor, include_anchor_score, distances, extra_score_families)
            flat = model_scores(model, x_test)
            preds, score_map = predict_from_candidate_scores(test_records, owners, flat)
            for local_i, rec_idx in enumerate(test_idx):
                target[int(rec_idx)] = int(preds[local_i])
                if anchor == "normal":
                    normal_score_map[records[int(rec_idx)]["key"]] = score_map[test_records[local_i]["key"]]
    truth = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    return {
        "predictions": normal_pred.tolist(),
        "shuffle_predictions": shuffle_pred.tolist(),
        "reverse_predictions": reverse_pred.tolist(),
        "score_map": normal_score_map,
        "accuracy": float((normal_pred == truth).mean()),
        "shuffle_accuracy": float((shuffle_pred == truth).mean()),
        "reverse_accuracy": float((reverse_pred == truth).mean()),
    }


def candidate_score_final_matrix(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    wps_score_map: dict[tuple[int, str], np.ndarray],
    anchor: str,
    include_raw_wan: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y = []
    owners = []
    families = ["text", "pixel", "flow"]
    if include_raw_wan:
        families.append("wan")
    for row_idx, rec in enumerate(records):
        key = rec["key"]
        family_rows = [normalize_scores(scores[family][anchor][key], "zscore") for family in families]
        family_rows.append(normalize_scores(wps_score_map[key], "zscore"))
        for cand_idx in range(rec["n_candidates"]):
            x_rows.append([float(row[cand_idx]) for row in family_rows])
            y.append(1 if cand_idx == rec["truth"] else 0)
            owners.append(row_idx)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y, dtype=np.int64), np.asarray(owners, dtype=np.int64)


def score_wps_model_for_records(
    train_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    distances: dict[str, dict[str, dict[tuple[int, str], float]]],
    args: argparse.Namespace,
    anchor: str,
    seed: int,
) -> dict[tuple[int, str], np.ndarray]:
    x_train, y_train, _ = candidate_wps_matrix(train_records, scores, "wan", "normal", False, distances, [])
    model = train_binary_model(x_train, y_train, args, seed)
    x_target, _, owners = candidate_wps_matrix(target_records, scores, "wan", anchor, False, distances, [])
    flat = model_scores(model, x_target)
    _, score_map = predict_from_candidate_scores(target_records, owners, flat)
    return score_map


def oof_wps_score_map(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    distances: dict[str, dict[str, dict[tuple[int, str], float]]],
    args: argparse.Namespace,
    seed: int,
) -> dict[tuple[int, str], np.ndarray]:
    labels = np.asarray([rec["question_type"] for rec in records])
    enc = LabelEncoder()
    y_strat = enc.fit_transform(labels)
    counts = np.bincount(y_strat)
    folds = max(2, min(3, int(counts[counts > 0].min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out: dict[tuple[int, str], np.ndarray] = {}
    for split_id, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(records)), y_strat)):
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        score_map = score_wps_model_for_records(train_records, val_records, scores, distances, args, "normal", seed + split_id)
        out.update(score_map)
    return out


def cv_score_final_model(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    distances: dict[str, dict[str, dict[tuple[int, str], float]]],
    args: argparse.Namespace,
    include_raw_wan: bool = False,
) -> dict[str, Any]:
    labels = np.asarray([rec["question_type"] for rec in records])
    enc = LabelEncoder()
    y_strat = enc.fit_transform(labels)
    counts = np.bincount(y_strat)
    folds = max(2, min(args.folds, int(counts[counts > 0].min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed + 211)
    normal_pred = np.zeros((len(records),), dtype=np.int64)
    shuffle_pred = np.zeros((len(records),), dtype=np.int64)
    reverse_pred = np.zeros((len(records),), dtype=np.int64)
    normal_score_map: dict[tuple[int, str], np.ndarray] = {}
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(records)), y_strat)):
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        train_wps_map = oof_wps_score_map(train_records, scores, distances, args, args.seed + 300 + split_id)
        x_train, y_train, _ = candidate_score_final_matrix(train_records, scores, train_wps_map, "normal", include_raw_wan)
        final_model = train_binary_model(x_train, y_train, args, args.seed + 400 + split_id)

        for anchor, target in [("normal", normal_pred), ("shuffle", shuffle_pred), ("reverse", reverse_pred)]:
            test_wps_map = score_wps_model_for_records(
                train_records,
                test_records,
                scores,
                distances,
                args,
                anchor,
                args.seed + 500 + split_id,
            )
            x_test, _, owners = candidate_score_final_matrix(test_records, scores, test_wps_map, anchor, include_raw_wan)
            flat = model_scores(final_model, x_test)
            preds, score_map = predict_from_candidate_scores(test_records, owners, flat)
            for local_i, rec_idx in enumerate(test_idx):
                target[int(rec_idx)] = int(preds[local_i])
                if anchor == "normal":
                    normal_score_map[records[int(rec_idx)]["key"]] = score_map[test_records[local_i]["key"]]

    truth = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    return {
        "predictions": normal_pred.tolist(),
        "shuffle_predictions": shuffle_pred.tolist(),
        "reverse_predictions": reverse_pred.tolist(),
        "score_map": normal_score_map,
        "accuracy": float((normal_pred == truth).mean()),
        "shuffle_accuracy": float((shuffle_pred == truth).mean()),
        "reverse_accuracy": float((reverse_pred == truth).mean()),
    }


def direct_score_map(records: list[dict[str, Any]], scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]], family: str, anchor: str = "normal") -> dict[tuple[int, str], np.ndarray]:
    return {rec["key"]: scores[family][anchor][rec["key"]] for rec in records}


def direct_predictions(records: list[dict[str, Any]], score_map: dict[tuple[int, str], np.ndarray]) -> list[int]:
    return [pred(score_map[rec["key"]]) for rec in records]


def combined_score_map(records: list[dict[str, Any]], maps: list[dict[tuple[int, str], np.ndarray]], weights: list[float] | None = None) -> dict[tuple[int, str], np.ndarray]:
    weights = weights or [1.0] * len(maps)
    out = {}
    for rec in records:
        key = rec["key"]
        total = np.zeros_like(maps[0][key], dtype=np.float64)
        for weight, mp in zip(weights, maps):
            total += float(weight) * normalize_scores(mp[key], "zscore")
        out[key] = total
    return out


def metrics_from_predictions(records: list[dict[str, Any]], predictions: list[int], mask: np.ndarray | None = None) -> dict[str, Any]:
    use = np.ones((len(records),), dtype=bool) if mask is None else mask.astype(bool)
    truth = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    pred_arr = np.asarray(predictions, dtype=np.int64)
    correct = pred_arr == truth
    by_type: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        if use[i]:
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
    }


def nested_wps_gate(
    records: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[tuple[int, str], np.ndarray]]],
    distances: dict[str, dict[str, dict[tuple[int, str], float]]],
    base_score_map: dict[tuple[int, str], np.ndarray],
    wps_result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = np.asarray([rec["question_type"] for rec in records])
    enc = LabelEncoder()
    y_strat = enc.fit_transform(labels)
    truth = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    counts = np.bincount(y_strat)
    folds = max(2, min(args.folds, int(counts[counts > 0].min())))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed + 77)
    switched = np.zeros((len(records),), dtype=np.int64)
    gate_flags = np.zeros((len(records),), dtype=bool)
    base_preds_all = np.asarray(direct_predictions(records, base_score_map), dtype=np.int64)
    wps_preds_all = np.asarray(wps_result["predictions"], dtype=np.int64)
    sample_features = np.asarray([sample_wps_features(rec, scores, "wan", "normal", distances) for rec in records], dtype=np.float64)
    base_margins = np.asarray([margin(base_score_map[rec["key"]]) for rec in records], dtype=np.float64).reshape(-1, 1)
    gate_x = np.concatenate([sample_features, base_margins], axis=1)
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(records)), y_strat)):
        # Inner held-out target on train split.
        inner_labels = y_strat[train_idx]
        inner_counts = np.bincount(inner_labels)
        inner_folds = max(2, min(3, int(inner_counts[inner_counts > 0].min())))
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=args.seed + 100 + split_id)
        inner_target = np.zeros((len(train_idx),), dtype=np.int64)
        for _, inner_val in inner.split(np.arange(len(train_idx)), inner_labels):
            global_val = train_idx[inner_val]
            inner_target[inner_val] = ((wps_preds_all[global_val] == truth[global_val]) & (base_preds_all[global_val] != truth[global_val])).astype(np.int64)
        if len(np.unique(inner_target)) < 2:
            gate_score = np.zeros((len(test_idx),), dtype=np.float64)
        else:
            gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed + split_id))
            gate.fit(gate_x[train_idx], inner_target)
            prob = gate.predict_proba(gate_x[test_idx])
            pos = list(gate.steps[-1][1].classes_).index(1)
            gate_score = prob[:, pos]
        threshold = float(np.quantile(gate_score, max(0.0, 1.0 - args.gate_coverage))) if len(gate_score) else 1.0
        flags = gate_score >= threshold
        gate_flags[test_idx] = flags
        switched[test_idx] = np.where(flags, wps_preds_all[test_idx], base_preds_all[test_idx])
    return {
        "predictions": switched.tolist(),
        "gate_flags": gate_flags.tolist(),
        "gate_coverage": float(gate_flags.mean()),
        "switch_helps": int(((switched == truth) & (base_preds_all != truth)).sum()),
        "switch_hurts": int(((switched != truth) & (base_preds_all == truth)).sum()),
        "base_accuracy": float((base_preds_all == truth).mean()),
        "wps_accuracy": float((wps_preds_all == truth).mean()),
        "switch_accuracy": float((switched == truth).mean()),
    }


def short_type(item: dict[str, Any], qtype: str) -> str:
    stats = item.get("per_question_type", {}).get(qtype, {})
    if not stats:
        return "0.0000"
    return f"{stats['accuracy']:.4f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench WPS Experiment", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Acc | Temporal-sensitive acc | Correct/total | Shuffle gap | Reverse gap |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, item in payload["models"].items():
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['temporal_sensitive_accuracy']:.4f} | "
            f"{item['correct']}/{item['total']} | {item.get('shuffle_gap', 0.0):.4f} | {item.get('reverse_gap', 0.0):.4f} |"
        )
    lines.append("")
    lines.append("## Question Type")
    qtypes = payload["question_types"]
    lines.append("| Model | " + " | ".join(qtypes) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(qtypes)) + "|")
    for name, item in payload["models"].items():
        lines.append("| " + name + " | " + " | ".join(short_type(item, q) for q in qtypes) + " |")
    lines.append("")
    lines.append("## WPS Gate")
    gate = payload["gate"]
    lines.append("| Gate | Value |")
    lines.append("|---|---:|")
    for key in ["gate_coverage", "base_accuracy", "wps_accuracy", "switch_accuracy", "switch_helps", "switch_hurts"]:
        lines.append(f"| {key} | {gate[key]:.4f} |" if isinstance(gate[key], float) else f"| {key} | {gate[key]} |")
    lines.append("")
    lines.append("## Success Criteria")
    lines.append("| Criterion | Result | Pass |")
    lines.append("|---|---|---:|")
    for row in payload["success_criteria"]:
        lines.append(f"| {row['criterion']} | {row['result']} | {row['pass']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-scores-jsonl", required=True)
    parser.add_argument("--shuffle-scores-jsonl", required=True)
    parser.add_argument("--reverse-scores-jsonl", required=True)
    parser.add_argument("--first-scores-jsonl", required=True)
    parser.add_argument("--timeavg-scores-jsonl", required=True)
    parser.add_argument("--features-h5", default="")
    parser.add_argument("--metadata-jsonl", default="")
    parser.add_argument("--mode", default="high_motion+camera_comp")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--classifier", default="ridge", choices=["ridge", "logistic"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gate-coverage", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, scores = load_all_scores(args)
    distances = feature_distances(args, records)
    truth = np.asarray([rec["truth"] for rec in records], dtype=np.int64)
    qtypes = sorted({rec["question_type"] for rec in records})

    score_maps = {
        "text": direct_score_map(records, scores, "text"),
        "pixel": direct_score_map(records, scores, "pixel"),
        "flow": direct_score_map(records, scores, "flow"),
        "raw_wan": direct_score_map(records, scores, "wan"),
    }
    score_maps["text_pixel_flow"] = combined_score_map(records, [score_maps["text"], score_maps["pixel"], score_maps["flow"]])

    model_outputs: dict[str, dict[str, Any]] = {}
    for name in ["text", "pixel", "flow", "raw_wan", "text_pixel_flow"]:
        preds = direct_predictions(records, score_maps[name])
        metric = metrics_from_predictions(records, preds)
        if name in {"raw_wan", "pixel", "flow"}:
            family = {"raw_wan": "wan", "pixel": "pixel", "flow": "flow"}[name]
            shuffle_preds = direct_predictions(records, direct_score_map(records, scores, family, "shuffle"))
            reverse_preds = direct_predictions(records, direct_score_map(records, scores, family, "reverse"))
            metric["shuffle_gap"] = metric["accuracy"] - float((np.asarray(shuffle_preds) == truth).mean())
            metric["reverse_gap"] = metric["accuracy"] - float((np.asarray(reverse_preds) == truth).mean())
        else:
            metric["shuffle_gap"] = 0.0
            metric["reverse_gap"] = 0.0
        model_outputs[name] = {"predictions": preds, "score_map": score_maps[name], "metric": metric}

    wps_specs = {
        "wan_wps": ("wan", False, []),
        "wan_raw_plus_wps": ("wan", True, []),
        "pixel_wps": ("pixel", False, []),
        "flow_wps": ("flow", False, []),
        "text_pixel_flow_wps": ("wan", False, ["text", "pixel", "flow"]),
    }
    for name, (family, include_anchor, extras) in wps_specs.items():
        result = cv_candidate_model(records, scores, distances, args, family, include_anchor_score=include_anchor, extra_score_families=extras)
        preds = result["predictions"]
        metric = metrics_from_predictions(records, preds)
        metric["shuffle_gap"] = metric["accuracy"] - result["shuffle_accuracy"]
        metric["reverse_gap"] = metric["accuracy"] - result["reverse_accuracy"]
        model_outputs[name] = {"predictions": preds, "score_map": result["score_map"], "metric": metric, "cv": result}

    score_final_specs = {
        "score_final_tpf_wps": False,
        "score_final_tpf_raw_wan_wps": True,
    }
    for name, include_raw_wan in score_final_specs.items():
        result = cv_score_final_model(records, scores, distances, args, include_raw_wan=include_raw_wan)
        preds = result["predictions"]
        metric = metrics_from_predictions(records, preds)
        metric["shuffle_gap"] = metric["accuracy"] - result["shuffle_accuracy"]
        metric["reverse_gap"] = metric["accuracy"] - result["reverse_accuracy"]
        model_outputs[name] = {"predictions": preds, "score_map": result["score_map"], "metric": metric, "cv": result}

    raw_pred = np.asarray(model_outputs["raw_wan"]["predictions"], dtype=np.int64)
    raw_shuffle_pred = np.asarray(direct_predictions(records, direct_score_map(records, scores, "wan", "shuffle")), dtype=np.int64)
    raw_reverse_pred = np.asarray(direct_predictions(records, direct_score_map(records, scores, "wan", "reverse")), dtype=np.int64)
    temporal_sensitive = (raw_pred != raw_shuffle_pred) | (raw_pred != raw_reverse_pred)
    for name, output in model_outputs.items():
        subset_metric = metrics_from_predictions(records, output["predictions"], temporal_sensitive)
        output["metric"]["temporal_sensitive_accuracy"] = subset_metric["accuracy"]
        output["metric"]["temporal_sensitive_correct"] = subset_metric["correct"]
        output["metric"]["temporal_sensitive_total"] = subset_metric["total"]

    gate = nested_wps_gate(records, scores, distances, model_outputs["text_pixel_flow"]["score_map"], model_outputs["wan_wps"]["cv"], args)
    gate_metric = metrics_from_predictions(records, gate["predictions"])
    gate_temporal = metrics_from_predictions(records, gate["predictions"], temporal_sensitive)
    gate.update({"metric": gate_metric, "temporal_sensitive_accuracy": gate_temporal["accuracy"]})

    models_payload = {}
    for name, output in model_outputs.items():
        models_payload[name] = output["metric"]
    models_payload["wps_gate_switch"] = {
        **gate_metric,
        "temporal_sensitive_accuracy": gate["temporal_sensitive_accuracy"],
        "shuffle_gap": 0.0,
        "reverse_gap": 0.0,
    }

    base_ts = models_payload["text_pixel_flow"]["temporal_sensitive_accuracy"]
    raw_acc = models_payload["raw_wan"]["accuracy"]
    wps_acc = models_payload["wan_wps"]["accuracy"]
    raw_shuffle_gap = models_payload["raw_wan"]["shuffle_gap"]
    raw_reverse_gap = models_payload["raw_wan"]["reverse_gap"]
    wps_shuffle_gap = models_payload["wan_wps"]["shuffle_gap"]
    wps_reverse_gap = models_payload["wan_wps"]["reverse_gap"]
    criteria = [
        {
            "criterion": "WPS > raw Wan overall",
            "result": f"{wps_acc:.4f} vs {raw_acc:.4f}",
            "pass": bool(wps_acc > raw_acc),
        },
        {
            "criterion": "temporal-sensitive WPS > text+pixel+flow by >=5%p",
            "result": f"{models_payload['wan_wps']['temporal_sensitive_accuracy']:.4f} vs {base_ts:.4f}",
            "pass": bool(models_payload["wan_wps"]["temporal_sensitive_accuracy"] - base_ts >= 0.05),
        },
        {
            "criterion": "held-out WPS gate > raw Wan everywhere",
            "result": f"{gate['switch_accuracy']:.4f} vs {raw_acc:.4f}",
            "pass": bool(gate["switch_accuracy"] > raw_acc),
        },
        {
            "criterion": "WPS normal-shuffle/reverse gap > raw Wan",
            "result": f"shuffle {wps_shuffle_gap:.4f} vs {raw_shuffle_gap:.4f}; reverse {wps_reverse_gap:.4f} vs {raw_reverse_gap:.4f}",
            "pass": bool(wps_shuffle_gap > raw_shuffle_gap and wps_reverse_gap > raw_reverse_gap),
        },
    ]

    payload = {
        "config": vars(args),
        "rows": len(records),
        "temporal_sensitive_rows": int(temporal_sensitive.sum()),
        "question_types": qtypes,
        "models": models_payload,
        "gate": gate,
        "success_criteria": criteria,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(Path(args.output_md), payload)
    print(json.dumps({"rows": len(records), "temporal_sensitive_rows": int(temporal_sensitive.sum()), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
