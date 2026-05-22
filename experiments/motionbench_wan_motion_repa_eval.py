#!/usr/bin/env python3
"""Evaluate Wan-Motion-REPA targets on MotionBench candidate reranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import (  # noqa: E402
    TextEmbedder,
    candidate_feature_matrix,
    flatten_dataset,
    parse_candidates,
    qa_candidate_text,
    random_project,
    row_id,
)
from motionbench_repa_common import aligned_rows_by_mode, candidate_bundle, safe_name, select_h5_rows, vectorize_feature_array, write_jsonl  # noqa: E402


def make_cv(stratify_labels: np.ndarray, folds: int, seed: int):
    enc = LabelEncoder()
    y = enc.fit_transform([str(x) for x in stratify_labels])
    counts = np.bincount(y)
    max_folds = int(counts[counts > 0].min()) if np.any(counts > 0) else 2
    folds = max(2, min(folds, max_folds))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(y)), y)


def reduce_video_embeddings(x: np.ndarray, joint_dim: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if joint_dim <= 0 or x.shape[1] == joint_dim:
        return x
    if x.shape[1] < joint_dim:
        pad = np.zeros((x.shape[0], joint_dim - x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)
    return random_project(x, joint_dim, seed).astype(np.float32)


def train_scorer(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, seed: int):
    if args.classifier == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, C=args.logistic_c)
    else:
        clf = RidgeClassifier(alpha=args.ridge_alpha)
    model = make_pipeline(StandardScaler(), clf)
    model.fit(x, y)
    return model


def scorer_scores(model, x: np.ndarray) -> np.ndarray:
    final = model.steps[-1][1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    else:
        prob = model.predict_proba(x)
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, prob.shape[1] - 1)
        scores = prob[:, pos]
    if np.asarray(scores).ndim == 2:
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, scores.shape[1] - 1)
        scores = scores[:, pos]
    return np.asarray(scores, dtype=np.float64)


def pad_width(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == width:
        return x
    if x.shape[1] > width:
        return x[:, :width]
    pad = np.zeros((x.shape[0], width - x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=1)


def residualize_fold(target: np.ndarray, basis: np.ndarray, train_indices: list[int], test_indices: list[int], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    scaler_b = StandardScaler()
    scaler_t = StandardScaler()
    b_train = scaler_b.fit_transform(basis[train_indices])
    t_train = scaler_t.fit_transform(target[train_indices])
    model = Ridge(alpha=alpha)
    model.fit(b_train, t_train)
    pred_train = scaler_t.inverse_transform(model.predict(b_train))
    pred_test = scaler_t.inverse_transform(model.predict(scaler_b.transform(basis[test_indices])))
    return (target[train_indices] - pred_train).astype(np.float32), (target[test_indices] - pred_test).astype(np.float32)


def load_source_feature(h5: h5py.File, rows: list[dict[str, Any]], mode: str, feature: str, seed: int) -> np.ndarray:
    data = select_h5_rows(h5[mode][feature][:], rows)
    return flatten_dataset(data, rows, feature, transform="full", seed=seed).astype(np.float32)


def load_target_feature(h5: h5py.File, rows: list[dict[str, Any]], mode: str, feature: str) -> np.ndarray:
    data = select_h5_rows(h5[mode][feature][:], rows)
    return data.reshape(data.shape[0], -1).astype(np.float32)


def build_residual_basis(source_h5: h5py.File, rows: list[dict[str, Any]], mode: str, wan_feature: str, seed: int) -> np.ndarray:
    pieces = []
    for feature in ["pixel_grid_sequence", "flow_grid_sequence"]:
        if feature in source_h5[mode]:
            data = select_h5_rows(source_h5[mode][feature][:], rows)
            pieces.append(flatten_dataset(data, rows, feature, transform="full", seed=seed))
    if wan_feature in source_h5[mode]:
        data = select_h5_rows(source_h5[mode][wan_feature][:], rows)
        for transform in ["first_frame_only", "time_average"]:
            pieces.append(vectorize_feature_array(data, rows, wan_feature, transform=transform, seed=seed))
    width = max(piece.shape[1] for piece in pieces)
    return np.concatenate([pad_width(piece, width) for piece in pieces], axis=1).astype(np.float32)


def text_embeddings(rows: list[dict[str, Any]], args: argparse.Namespace):
    candidates, labels, texts, offsets = candidate_bundle(rows)
    valid = [i for i, cands in enumerate(candidates) if cands and labels[i] >= 0]
    embedder = TextEmbedder(kind=args.text_encoder, model_name=args.text_model, dim=args.text_dim, batch_size=args.text_batch_size, device=args.text_device, seed=args.seed).fit(texts)
    emb = embedder.transform(texts).astype(np.float32)
    emb = random_project(emb, args.joint_dim, args.seed + 17).astype(np.float32)
    return candidates, labels, offsets, valid, emb


def run_feature(
    name: str,
    video_emb: np.ndarray | None,
    residual_target: np.ndarray | None,
    residual_basis: np.ndarray | None,
    rows: list[dict[str, Any]],
    candidates: list[list[dict[str, Any]]],
    labels: np.ndarray,
    offsets: list[list[int]],
    valid: list[int],
    text_emb: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stratify = np.asarray([rows[i].get(args.stratify_column, labels[i]) for i in valid])
    all_true: list[int] = []
    all_pred: list[int] = []
    split_rows = []
    score_rows: list[dict[str, Any]] = []
    per_type: dict[str, list[int]] = {}

    for split_id, (train_local, test_local) in enumerate(make_cv(stratify, args.folds, args.seed)):
        train_indices = [valid[int(i)] for i in train_local]
        test_indices = [valid[int(i)] for i in test_local]
        if residual_target is not None and residual_basis is not None:
            x_train_video, x_test_video = residualize_fold(residual_target, residual_basis, train_indices, test_indices, args.residual_alpha)
            x_train_video = reduce_video_embeddings(x_train_video, args.joint_dim, args.seed + 1000 + split_id)
            x_test_video = reduce_video_embeddings(x_test_video, args.joint_dim, args.seed + 1000 + split_id)
            row_video = {idx: arr for idx, arr in zip(train_indices, x_train_video)}
            row_video.update({idx: arr for idx, arr in zip(test_indices, x_test_video)})
        else:
            assert video_emb is not None
            reduced = reduce_video_embeddings(video_emb, args.joint_dim, args.seed + 2000)
            row_video = {idx: reduced[idx] for idx in train_indices + test_indices}

        x_chunks = []
        y_train = []
        for row_idx in train_indices:
            cand_offsets = offsets[row_idx]
            repeated_video = np.repeat(row_video[row_idx][None], len(cand_offsets), axis=0)
            x_chunks.append(candidate_feature_matrix(repeated_video, text_emb[cand_offsets]))
            y_train.extend([1 if j == labels[row_idx] else 0 for j in range(len(cand_offsets))])
        model = train_scorer(np.concatenate(x_chunks, axis=0), np.asarray(y_train, dtype=np.int64), args, args.seed + split_id)

        split_correct = 0
        for row_idx in test_indices:
            cand_offsets = offsets[row_idx]
            repeated_video = np.repeat(row_video[row_idx][None], len(cand_offsets), axis=0)
            x_test = candidate_feature_matrix(repeated_video, text_emb[cand_offsets])
            scores = scorer_scores(model, x_test)
            pred = int(np.argmax(scores))
            truth = int(labels[row_idx])
            correct = int(pred == truth)
            split_correct += correct
            all_true.append(truth)
            all_pred.append(pred)
            qtype = str(rows[row_idx].get("question_type", "all"))
            per_type.setdefault(qtype, []).append(correct)
            score_rows.append(
                {
                    "split": split_id,
                    "mode": args.mode,
                    "feature": name,
                    "classifier": args.classifier,
                    "row_index": int(row_idx),
                    "video_id": row_id(rows[row_idx], row_idx),
                    "question_type": qtype,
                    "answer": rows[row_idx].get("answer"),
                    "answer_index": truth,
                    "prediction_index": pred,
                    "correct": bool(correct),
                    "candidates": [
                        {
                            "index": int(j),
                            "letter": candidates[row_idx][j]["letter"],
                            "text": candidates[row_idx][j]["text"],
                            "score": float(scores[j]),
                        }
                        for j in range(len(candidates[row_idx]))
                    ],
                }
            )
        split_rows.append({"split": split_id, "accuracy": float(split_correct / max(1, len(test_indices))), "correct": int(split_correct), "total": int(len(test_indices))})

    truth_arr = np.asarray(all_true, dtype=np.int64)
    pred_arr = np.asarray(all_pred, dtype=np.int64)
    correct_arr = truth_arr == pred_arr
    label_count = max(max(all_true + all_pred) + 1 if all_true else 0, 1)
    summary = {
        "feature": name,
        "accuracy_mean": float(np.mean([row["accuracy"] for row in split_rows])),
        "accuracy_std": float(np.std([row["accuracy"] for row in split_rows], ddof=0)),
        "correct": int(correct_arr.sum()),
        "total": int(correct_arr.size),
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(per_type.items())
        },
        "confusion_matrix": confusion_matrix(truth_arr, pred_arr, labels=list(range(label_count))).tolist(),
        "splits": split_rows,
    }
    return summary, score_rows


def short_per_type(item: dict[str, Any], qtype: str) -> str:
    stats = item.get("per_question_type", {}).get(qtype, {})
    return f"{stats.get('accuracy', 0.0):.4f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    qtypes = payload["question_types"]
    lines = ["# MotionBench Wan-Motion-REPA Evaluation", ""]
    lines.append("| Step | Feature | Acc | Correct/total | " + " | ".join(qtypes) + " |")
    lines.append("|---:|---|---:|---:|" + "|".join(["---:"] * len(qtypes)) + "|")
    for idx, item in enumerate(payload["results"], start=1):
        lines.append(
            f"| {idx} | {item['feature']} | {item['accuracy_mean']:.4f} | {item['correct']}/{item['total']} | "
            + " | ".join(short_per_type(item, q) for q in qtypes)
            + " |"
        )
    lines.append("")
    lines.append("## Pairwise Gain vs Baselines")
    base = next((x for x in payload["results"] if x["feature"] == "text_only"), None)
    raw = next((x for x in payload["results"] if x["feature"] == "raw_wan"), None)
    lines.append("| Feature | Gain vs text | Gain vs raw Wan |")
    lines.append("|---|---:|---:|")
    for item in payload["results"]:
        gain_text = item["accuracy_mean"] - (base["accuracy_mean"] if base else 0.0)
        gain_raw = item["accuracy_mean"] - (raw["accuracy_mean"] if raw else 0.0)
        lines.append(f"| {item['feature']} | {gain_text:+.4f} | {gain_raw:+.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features-h5", required=True)
    parser.add_argument("--source-metadata-jsonl", required=True)
    parser.add_argument("--target-features-h5", required=True)
    parser.add_argument("--target-metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-scores-jsonl", required=True)
    parser.add_argument("--mode", default="high_motion+camera_comp")
    parser.add_argument("--wan-feature", default="wan_vae_grid_2x2")
    parser.add_argument("--target-prefix", default="wmrepa")
    parser.add_argument("--classifier", default="ridge", choices=["ridge", "logistic"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--residual-alpha", type=float, default=10.0)
    parser.add_argument("--joint-dim", type=int, default=512)
    parser.add_argument("--text-encoder", default="hash", choices=["hash", "tfidf", "clip", "hf", "wan-t5", "umt5", "sentence-transformer"])
    parser.add_argument("--text-model", default="")
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--stratify-column", default="question_type")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows_by = aligned_rows_by_mode(args.source_metadata_jsonl)
    target_rows_by = aligned_rows_by_mode(args.target_metadata_jsonl)
    rows = target_rows_by.get(args.mode) or source_rows_by.get(args.mode) or []
    if not rows:
        raise ValueError(f"No rows for mode {args.mode}")
    candidates, labels, offsets, valid, text_emb = text_embeddings(rows, args)
    qtypes = sorted({str(rows[i].get("question_type", "all")) for i in valid})
    safe_wan = safe_name(args.wan_feature)
    target_names = {
        "structured_compact": f"{args.target_prefix}_structured_compact_{safe_wan}",
        "dynamics_relation": f"{args.target_prefix}_dynamics_relation_{safe_wan}",
        "relation_only": f"{args.target_prefix}_relation_only_{safe_wan}",
        "equivariance": f"{args.target_prefix}_equivariance_{safe_wan}",
        "multi_target": f"{args.target_prefix}_multi_target_{safe_wan}",
    }

    results = []
    score_rows: list[dict[str, Any]] = []
    with h5py.File(args.source_features_h5, "r") as source_h5, h5py.File(args.target_features_h5, "r") as target_h5:
        zeros = np.zeros((len(rows), args.joint_dim), dtype=np.float32)
        specs: list[tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
        specs.append(("text_only", zeros, None, None))
        specs.append(("raw_wan", load_source_feature(source_h5, rows, args.mode, args.wan_feature, args.seed), None, None))
        for name, feature in target_names.items():
            if args.mode in target_h5 and feature in target_h5[args.mode]:
                specs.append((name, load_target_feature(target_h5, rows, args.mode, feature), None, None))
        if "dynamics_relation" in target_names and target_names["dynamics_relation"] in target_h5[args.mode]:
            target = load_target_feature(target_h5, rows, args.mode, target_names["dynamics_relation"])
            basis = build_residual_basis(source_h5, rows, args.mode, args.wan_feature, args.seed)
            specs.insert(4, ("residualized_dynamics_relation", None, target, basis))

        for name, video_emb, residual_target, residual_basis in specs:
            summary, rows_scores = run_feature(
                name,
                video_emb,
                residual_target,
                residual_basis,
                rows,
                candidates,
                labels,
                offsets,
                valid,
                text_emb,
                args,
            )
            results.append(summary)
            score_rows.extend(rows_scores)

    payload = {"config": vars(args), "question_types": qtypes, "results": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(Path(args.output_md), payload)
    write_jsonl(Path(args.output_scores_jsonl), score_rows)
    print(json.dumps({"results": len(results), "score_rows": len(score_rows), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
