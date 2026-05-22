#!/usr/bin/env python3
"""Candidate-conditioned MotionBench QA reranker for cached Wan/pixel/flow features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import (  # noqa: E402
    TextEmbedder,
    answer_index,
    bootstrap_ci,
    candidate_feature_matrix,
    flatten_dataset,
    parse_candidates,
    qa_candidate_text,
    random_project,
    read_jsonl,
    row_id,
    rows_by_mode,
    write_jsonl,
)


def make_cv(stratify_labels: np.ndarray, folds: int, repeats: int, seed: int):
    encoder = LabelEncoder()
    y = encoder.fit_transform([str(x) for x in stratify_labels])
    counts = np.bincount(y)
    max_folds = int(counts[counts > 0].min()) if np.any(counts > 0) else 2
    folds = max(2, min(folds, max_folds))
    if repeats <= 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(y)), y)
    return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(np.arange(len(y)), y)


def stable_seed(*parts: Any, base: int = 0) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int((int(digest[:12], 16) + int(base)) % 1_000_000_007)


def reduce_video_embeddings(x: np.ndarray, args: argparse.Namespace, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    target = int(args.joint_dim)
    if args.video_reduction == "none" or target <= 0 or x.shape[1] == target:
        return x
    if x.shape[1] < target:
        pad = np.zeros((x.shape[0], target - x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)
    if args.video_reduction == "random":
        return random_project(x, target, seed)
    if args.video_reduction == "pca":
        from sklearn.decomposition import PCA

        n_components = max(1, min(target, x.shape[0] - 1, x.shape[1]))
        pca = PCA(n_components=n_components, random_state=seed)
        reduced = pca.fit_transform(x).astype(np.float32)
        if n_components < target:
            pad = np.zeros((x.shape[0], target - n_components), dtype=np.float32)
            reduced = np.concatenate([reduced, pad], axis=1)
        return reduced
    raise ValueError(f"Unknown video reduction: {args.video_reduction}")


def build_video_embeddings(
    h5: h5py.File,
    mode: str,
    feature: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> np.ndarray | None:
    if feature in {"text_only", "zero_video"}:
        return np.zeros((len(rows), int(args.joint_dim)), dtype=np.float32)
    if mode not in h5 or feature not in h5[mode]:
        return None
    data = h5[mode][feature][:]
    h5_indices = [int(row.get("__h5_index", i)) for i, row in enumerate(rows)]
    if h5_indices != list(range(len(rows))):
        data = data[h5_indices]
    x = flatten_dataset(data, rows, feature, transform=args.temporal_transform, seed=args.seed)
    x = reduce_video_embeddings(
        x,
        args,
        stable_seed(mode, feature, args.temporal_transform, args.video_reduction, base=args.seed),
    )
    return x.astype(np.float32)


def train_scorer(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, seed: int):
    classifier = args.classifier
    if classifier == "logistic":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, C=args.logistic_c)
    elif classifier == "ridge":
        model = RidgeClassifier(alpha=args.ridge_alpha)
    elif classifier == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=1e-4,
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(x, y)
    return pipe


def scorer_scores(model, x: np.ndarray) -> np.ndarray:
    final = model.steps[-1][1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    elif hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, prob.shape[1] - 1)
        scores = prob[:, pos]
    else:
        pred = model.predict(x)
        scores = pred.astype(np.float32)
    if np.asarray(scores).ndim == 2:
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, scores.shape[1] - 1)
        scores = scores[:, pos]
    return np.asarray(scores, dtype=np.float64)


def candidate_rows(rows: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    all_candidates: list[list[dict[str, Any]]] = []
    labels = []
    for row in rows:
        cands = parse_candidates(row)
        idx = answer_index(row)
        if len(cands) < 2 or idx is None or idx >= len(cands):
            all_candidates.append([])
            labels.append(-1)
        else:
            all_candidates.append(cands)
            labels.append(idx)
    return all_candidates, np.asarray(labels, dtype=np.int64)


def make_pair_features(video_emb: np.ndarray, text_emb: np.ndarray, row_indices: list[int], cand_offsets: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    owner = []
    for row_idx in row_indices:
        offsets = cand_offsets[row_idx]
        repeated_video = np.repeat(video_emb[row_idx : row_idx + 1], len(offsets), axis=0)
        xs.append(candidate_feature_matrix(repeated_video, text_emb[offsets]))
        owner.extend([row_idx] * len(offsets))
    return np.concatenate(xs, axis=0), np.asarray(owner, dtype=np.int64)


def run_mode_feature(h5, mode: str, feature: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    video_emb = build_video_embeddings(h5, mode, feature, rows, args)
    if video_emb is None:
        return None, []
    video_emb = video_emb.astype(np.float32)
    candidates, answer_indices = candidate_rows(rows)
    valid_indices = [i for i, cands in enumerate(candidates) if cands]
    if len(valid_indices) < 4:
        return None, []

    texts: list[str] = []
    cand_offsets: list[list[int]] = [[] for _ in rows]
    for i, cands in enumerate(candidates):
        for cand in cands:
            cand_offsets[i].append(len(texts))
            texts.append(qa_candidate_text(rows[i], cand))

    embedder = TextEmbedder(
        kind=args.text_encoder,
        model_name=args.text_model,
        dim=args.text_dim,
        batch_size=args.text_batch_size,
        device=args.text_device,
        seed=args.seed,
    ).fit(texts)
    text_emb = embedder.transform(texts)
    text_emb = random_project(text_emb, args.joint_dim, args.seed + 17).astype(np.float32)

    stratify = np.asarray([rows[i].get(args.stratify_column, answer_indices[i]) for i in valid_indices])
    split_rows = []
    score_rows: list[dict[str, Any]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    margins: list[float] = []
    per_type: dict[str, list[int]] = {}

    for split_id, (train_local, test_local) in enumerate(make_cv(stratify, args.folds, args.repeats, args.seed)):
        train_indices = [valid_indices[int(i)] for i in train_local]
        test_indices = [valid_indices[int(i)] for i in test_local]

        x_chunks = []
        y_train = []
        for row_idx in train_indices:
            offsets = cand_offsets[row_idx]
            repeated_video = np.repeat(video_emb[row_idx : row_idx + 1], len(offsets), axis=0)
            x_chunks.append(candidate_feature_matrix(repeated_video, text_emb[offsets]))
            y_train.extend([1 if j == answer_indices[row_idx] else 0 for j in range(len(offsets))])
        x_train = np.concatenate(x_chunks, axis=0)
        y_train_np = np.asarray(y_train, dtype=np.int64)
        model = train_scorer(x_train, y_train_np, args, args.seed + split_id)

        pred_indices = []
        true_indices = []
        split_correct = 0
        for row_idx in test_indices:
            offsets = cand_offsets[row_idx]
            repeated_video = np.repeat(video_emb[row_idx : row_idx + 1], len(offsets), axis=0)
            x_test = candidate_feature_matrix(repeated_video, text_emb[offsets])
            scores = scorer_scores(model, x_test)
            pred = int(np.argmax(scores))
            true = int(answer_indices[row_idx])
            pred_indices.append(pred)
            true_indices.append(true)
            all_true.append(true)
            all_pred.append(pred)
            is_correct = int(pred == true)
            split_correct += is_correct
            qtype = str(rows[row_idx].get("question_type", rows[row_idx].get("label", "all")))
            per_type.setdefault(qtype, []).append(is_correct)
            sorted_scores = np.sort(scores)
            margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else 0.0
            margins.append(margin)
            score_rows.append(
                {
                    "split": split_id,
                    "mode": mode,
                    "feature": feature,
                    "classifier": args.classifier,
                    "row_index": int(row_idx),
                    "video_id": row_id(rows[row_idx], row_idx),
                    "question_type": qtype,
                    "answer": rows[row_idx].get("answer"),
                    "answer_index": true,
                    "prediction_index": pred,
                    "correct": bool(is_correct),
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
        split_rows.append(
            {
                "split": split_id,
                "accuracy": float(split_correct / max(1, len(test_indices))),
                "correct": int(split_correct),
                "total": int(len(test_indices)),
            }
        )

    correct = (np.asarray(all_true) == np.asarray(all_pred)).astype(np.float32)
    ci = bootstrap_ci(correct, seed=args.seed, n_boot=args.bootstrap)
    label_count = max(max(all_true + all_pred) + 1 if all_true else 0, 1)
    summary = {
        "mode": mode,
        "feature": feature,
        "text_encoder": args.text_encoder,
        "classifier": args.classifier,
        "video_reduction": args.video_reduction,
        "joint_dim": args.joint_dim,
        "ridge_alpha": args.ridge_alpha,
        "logistic_c": args.logistic_c,
        "temporal_transform": args.temporal_transform,
        "accuracy_mean": float(np.mean([x["accuracy"] for x in split_rows])),
        "accuracy_std": float(np.std([x["accuracy"] for x in split_rows], ddof=0)),
        "correct": int(correct.sum()),
        "total": int(correct.size),
        "accuracy_ci95": [ci[0], ci[1]],
        "candidate_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "candidate_margin_std": float(np.std(margins, ddof=0)) if margins else 0.0,
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(per_type.items())
        },
        "confusion_matrix": confusion_matrix(all_true, all_pred, labels=list(range(label_count))).tolist(),
        "splits": split_rows,
    }
    return summary, score_rows


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench Candidate Rerank Probe", ""]
    lines.append("| mode | feature | text | classifier | acc mean | std | CI low | CI high | correct/total | margin |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for item in payload["results"]:
        ci = item["accuracy_ci95"]
        lines.append(
            f"| {item['mode']} | {item['feature']} | {item['text_encoder']} | {item['classifier']} | "
            f"{item['accuracy_mean']:.4f} | {item['accuracy_std']:.4f} | {ci[0]:.4f} | {ci[1]:.4f} | "
            f"{item['correct']}/{item['total']} | {item['candidate_margin_mean']:.4f} |"
        )
    lines.extend(["", "## Per Question Type", ""])
    lines.append("| mode | feature | question_type | acc | correct/total |")
    lines.append("|---|---|---|---:|---:|")
    for item in payload["results"]:
        for qtype, stats in item["per_question_type"].items():
            lines.append(
                f"| {item['mode']} | {item['feature']} | {qtype} | {stats['accuracy']:.4f} | {stats['correct']}/{stats['total']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-scores-jsonl", required=True)
    parser.add_argument("--feature-names", default="wan_vae_grid_sequence,pixel_grid_sequence,flow_grid_sequence")
    parser.add_argument("--modes", default="none")
    parser.add_argument("--temporal-transform", default="full")
    parser.add_argument("--text-encoder", default="hash", choices=["hash", "tfidf", "clip", "hf", "wan-t5", "umt5", "sentence-transformer"])
    parser.add_argument("--text-model", default="")
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--joint-dim", type=int, default=512)
    parser.add_argument("--video-reduction", default="random", choices=["random", "pca", "none"])
    parser.add_argument("--classifier", default="logistic", choices=["logistic", "ridge", "mlp"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--max-rows-per-type", type=int, default=0)
    parser.add_argument("--stratify-column", default="question_type")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.metadata_jsonl))
    rows_by = rows_by_mode(rows)
    for mode_rows in rows_by.values():
        for idx, row in enumerate(mode_rows):
            row["__h5_index"] = idx
    if args.max_rows_per_type > 0:
        rng = np.random.default_rng(args.seed)
        filtered: dict[str, list[dict[str, Any]]] = {}
        for mode, mode_rows in rows_by.items():
            by_type: dict[str, list[dict[str, Any]]] = {}
            for row in mode_rows:
                key = str(row.get(args.stratify_column, row.get("question_type", row.get("label", "all"))))
                by_type.setdefault(key, []).append(row)
            selected: list[dict[str, Any]] = []
            for key in sorted(by_type):
                group = by_type[key]
                if len(group) <= args.max_rows_per_type:
                    selected.extend(group)
                else:
                    take = sorted(rng.choice(np.arange(len(group)), size=args.max_rows_per_type, replace=False).tolist())
                    selected.extend([group[i] for i in take])
            filtered[mode] = selected
        rows_by = filtered
    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    results = []
    score_rows: list[dict[str, Any]] = []

    with h5py.File(args.features_h5, "r") as h5:
        for mode in modes:
            mode_rows = rows_by.get(mode)
            if not mode_rows:
                continue
            for feature in feature_names:
                summary, scores = run_mode_feature(h5, mode, feature, mode_rows, args)
                if summary is not None:
                    results.append(summary)
                    score_rows.extend(scores)

    payload = {"config": vars(args), "results": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    write_jsonl(Path(args.output_scores_jsonl), score_rows)
    print(
        json.dumps(
            {
                "results": len(results),
                "score_rows": len(score_rows),
                "output_json": args.output_json,
                "output_scores_jsonl": args.output_scores_jsonl,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
