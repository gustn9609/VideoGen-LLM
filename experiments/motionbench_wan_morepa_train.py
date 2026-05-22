#!/usr/bin/env python3
"""Train a Wan-MoREPA residual candidate reranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import TextEmbedder, candidate_feature_matrix, parse_candidates, qa_candidate_text, random_project, row_id  # noqa: E402
from motionbench_repa_common import (  # noqa: E402
    aligned_rows_by_mode,
    candidate_bundle,
    l2_normalize_rows,
    random_project_fit,
    safe_name,
    select_h5_rows,
    stable_seed,
    tokens_from_feature_array,
    write_jsonl,
)


class WanMorepaResidualScorer(nn.Module):
    def __init__(self, token_dim: int, text_dim: int, hidden_dim: int, align_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_adapter = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, align_dim),
        )
        self.text_proj = nn.Linear(text_dim, align_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(align_dim, max(8, align_dim // 4)), nn.GELU(), nn.Linear(max(8, align_dim // 4), 1))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def encode_video(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        z_tok = self.token_adapter(tokens)
        mask = token_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        z = (z_tok * mask).sum(dim=1) / denom
        return z

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_video(tokens, token_mask)
        z_norm = F.normalize(z, dim=-1)
        t = F.normalize(self.text_proj(text_emb), dim=-1)
        residual = torch.einsum("bd,bcd->bc", z_norm, t)
        gate = torch.sigmoid(self.gate(z)).clamp(0.0, 1.0)
        residual = residual * gate * self.residual_scale
        return residual, z


def make_cv(stratify_labels: np.ndarray, folds: int, repeats: int, seed: int):
    encoder = LabelEncoder()
    y = encoder.fit_transform([str(x) for x in stratify_labels])
    counts = np.bincount(y)
    max_folds = int(counts[counts > 0].min()) if np.any(counts > 0) else 2
    folds = max(2, min(folds, max_folds))
    if repeats <= 1:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(y)), y)
    return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(np.arange(len(y)), y)


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
        scores = model.predict(x).astype(np.float32)
    if np.asarray(scores).ndim == 2:
        cls = list(getattr(final, "classes_", [0, 1]))
        pos = cls.index(1) if 1 in cls else min(1, scores.shape[1] - 1)
        scores = scores[:, pos]
    return np.asarray(scores, dtype=np.float32)


def train_text_scorer(x: np.ndarray, y: np.ndarray, classifier: str, seed: int, ridge_alpha: float, logistic_c: float):
    if classifier == "ridge":
        clf = RidgeClassifier(alpha=ridge_alpha)
    elif classifier == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, C=logistic_c)
    else:
        raise ValueError(f"Unknown text classifier: {classifier}")
    model = make_pipeline(StandardScaler(), clf)
    model.fit(x, y)
    return model


def text_pair_matrix(text_emb: np.ndarray, offsets: list[int]) -> np.ndarray:
    emb = text_emb[offsets].astype(np.float32)
    zero = np.zeros_like(emb)
    return candidate_feature_matrix(zero, emb)


def prepare_text_scores(
    model,
    text_emb: np.ndarray,
    candidate_offsets: list[list[int]],
    indices: list[int],
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full((len(indices), max_candidates), -1e4, dtype=np.float32)
    mask = np.zeros((len(indices), max_candidates), dtype=np.float32)
    for out_i, row_idx in enumerate(indices):
        offsets = candidate_offsets[row_idx]
        if not offsets:
            continue
        raw = scorer_scores(model, text_pair_matrix(text_emb, offsets))
        scores[out_i, : len(offsets)] = raw
        mask[out_i, : len(offsets)] = 1.0
    return scores, mask


def pack_candidate_texts(text_emb: np.ndarray, candidate_offsets: list[list[int]], indices: list[int], max_candidates: int) -> np.ndarray:
    out = np.zeros((len(indices), max_candidates, text_emb.shape[1]), dtype=np.float32)
    for out_i, row_idx in enumerate(indices):
        offsets = candidate_offsets[row_idx]
        if offsets:
            out[out_i, : len(offsets)] = text_emb[offsets]
    return out


def teacher_global(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        return data
    return data.reshape(data.shape[0], -1)


def standardize_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool)
    if not valid.any():
        return tokens.astype(np.float32)
    flat = tokens[valid]
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    out = tokens.copy()
    out[valid] = (flat - mean) / np.maximum(std, 1e-6)
    return out.astype(np.float32)


def hard_weights_from_text(scores: np.ndarray, labels: np.ndarray, hard_weight: float, margin_quantile: float) -> np.ndarray:
    preds = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2] if scores.shape[1] > 1 else np.zeros((scores.shape[0],), dtype=np.float32)
    threshold = float(np.quantile(margins, margin_quantile)) if len(margins) else 0.0
    hard = (preds != labels) | (margins <= threshold)
    weights = np.ones((scores.shape[0],), dtype=np.float32)
    weights[hard] = float(hard_weight)
    return weights


def eval_split(
    model: WanMorepaResidualScorer,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    text_batch: np.ndarray,
    candidate_mask: np.ndarray,
    base_scores: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    score_rows = []
    with torch.no_grad():
        for start in range(0, len(labels), args.eval_batch_size):
            end = min(len(labels), start + args.eval_batch_size)
            tok = torch.from_numpy(tokens[start:end]).to(device)
            tok_mask = torch.from_numpy(token_mask[start:end]).to(device)
            txt = torch.from_numpy(text_batch[start:end]).to(device)
            base = torch.from_numpy(base_scores[start:end]).to(device)
            cand_mask = torch.from_numpy(candidate_mask[start:end]).to(device)
            residual, _ = model(tok, tok_mask, txt)
            logits = base + float(args.residual_lambda) * residual
            logits = logits.masked_fill(cand_mask <= 0, -1e4)
            score_rows.append(logits.cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist())
    return np.asarray(preds, dtype=np.int64), np.concatenate(score_rows, axis=0)


def train_one_split(
    split_id: int,
    train_indices: list[int],
    test_indices: list[int],
    tokens: np.ndarray,
    token_mask: np.ndarray,
    neg_tokens: np.ndarray,
    neg_mask: np.ndarray,
    teacher_target: np.ndarray,
    text_emb: np.ndarray,
    candidate_offsets: list[list[int]],
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_chunks = []
    y_train = []
    for row_idx in train_indices:
        offsets = candidate_offsets[row_idx]
        if not offsets:
            continue
        x_chunks.append(text_pair_matrix(text_emb, offsets))
        y_train.extend([1 if j == labels[row_idx] else 0 for j in range(len(offsets))])
    text_model = train_text_scorer(
        np.concatenate(x_chunks, axis=0),
        np.asarray(y_train, dtype=np.int64),
        args.text_classifier,
        args.seed + split_id,
        args.ridge_alpha,
        args.logistic_c,
    )
    max_candidates = max(len(candidate_offsets[i]) for i in train_indices + test_indices)
    train_base, train_cand_mask = prepare_text_scores(text_model, text_emb, candidate_offsets, train_indices, max_candidates)
    test_base, test_cand_mask = prepare_text_scores(text_model, text_emb, candidate_offsets, test_indices, max_candidates)
    train_text = pack_candidate_texts(text_emb, candidate_offsets, train_indices, max_candidates)
    test_text = pack_candidate_texts(text_emb, candidate_offsets, test_indices, max_candidates)
    train_labels = labels[train_indices].astype(np.int64)
    test_labels = labels[test_indices].astype(np.int64)
    row_weights = hard_weights_from_text(train_base, train_labels, args.hard_weight, args.hard_margin_quantile)

    model = WanMorepaResidualScorer(
        token_dim=tokens.shape[-1],
        text_dim=text_emb.shape[-1],
        hidden_dim=args.hidden_dim,
        align_dim=args.align_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_t = torch.from_numpy(tokens[train_indices]).to(device)
    train_m = torch.from_numpy(token_mask[train_indices]).to(device)
    train_neg_t = torch.from_numpy(neg_tokens[train_indices]).to(device)
    train_neg_m = torch.from_numpy(neg_mask[train_indices]).to(device)
    train_teacher = torch.from_numpy(teacher_target[train_indices]).to(device)
    train_text_t = torch.from_numpy(train_text).to(device)
    train_base_t = torch.from_numpy(train_base).to(device)
    train_cand_mask_t = torch.from_numpy(train_cand_mask).to(device)
    train_labels_t = torch.from_numpy(train_labels).to(device)
    row_weights_t = torch.from_numpy(row_weights).to(device)

    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_indices), device=device)
        for start in range(0, len(train_indices), args.batch_size):
            idx = order[start : start + args.batch_size]
            residual, z = model(train_t[idx], train_m[idx], train_text_t[idx])
            logits = train_base_t[idx] + float(args.residual_lambda) * residual
            logits = logits.masked_fill(train_cand_mask_t[idx] <= 0, -1e4)
            ce = F.cross_entropy(logits, train_labels_t[idx], reduction="none")
            rank_loss = (ce * row_weights_t[idx]).mean()
            z_norm = F.normalize(z, dim=-1)
            teacher_norm = F.normalize(train_teacher[idx], dim=-1)
            align_loss = 1.0 - (z_norm * teacher_norm).sum(dim=-1).mean()
            neg_z = model.encode_video(train_neg_t[idx], train_neg_m[idx])
            neg_cos = (F.normalize(neg_z, dim=-1) * teacher_norm).sum(dim=-1)
            pos_cos = (z_norm * teacher_norm).sum(dim=-1)
            contrast_loss = F.relu(neg_cos - pos_cos + float(args.contrast_margin)).mean()
            zero_mean = residual.mean(dim=1).pow(2).mean()
            loss = (
                rank_loss
                + float(args.lambda_align) * align_loss
                + float(args.lambda_contrast) * contrast_loss
                + float(args.lambda_zero_mean) * zero_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    test_preds, test_scores = eval_split(
        model,
        tokens[test_indices],
        token_mask[test_indices],
        test_text,
        test_cand_mask,
        test_base,
        test_labels,
        args,
        device,
    )
    text_preds = np.argmax(test_base, axis=1)
    stats = {
        "text_accuracy": float(accuracy_score(test_labels, text_preds)),
        "morepa_accuracy": float(accuracy_score(test_labels, test_preds)),
    }
    return test_preds, test_scores, stats


def run_mode_feature(
    wan_h5: h5py.File,
    teacher_h5: h5py.File,
    mode: str,
    feature: str,
    teacher_feature: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if mode not in wan_h5 or feature not in wan_h5[mode] or mode not in teacher_h5 or teacher_feature not in teacher_h5[mode]:
        return None, []
    wan_data = select_h5_rows(wan_h5[mode][feature][:], rows)
    teacher_data = select_h5_rows(teacher_h5[mode][teacher_feature][:], rows)
    tokens, token_mask = tokens_from_feature_array(wan_data, rows, feature, max_tokens=args.max_tokens)
    neg_tokens, neg_mask = tokens_from_feature_array(wan_data, rows, feature, max_tokens=args.max_tokens)
    rng = np.random.default_rng(stable_seed(mode, feature, "neg", base=args.seed))
    for i in range(neg_tokens.shape[0]):
        valid = int(neg_mask[i].sum())
        if valid > 1:
            order = np.arange(valid)
            rng.shuffle(order)
            neg_tokens[i, :valid] = neg_tokens[i, order]
    tokens = standardize_tokens(tokens, token_mask)
    neg_tokens = standardize_tokens(neg_tokens, neg_mask)
    teacher = teacher_global(teacher_data)
    proj = random_project_fit(teacher.shape[1], args.align_dim, stable_seed(mode, teacher_feature, base=args.seed))
    teacher_target = l2_normalize_rows(teacher @ proj if teacher.shape[1] == proj.shape[0] else teacher[:, : args.align_dim])
    if teacher_target.shape[1] < args.align_dim:
        pad = np.zeros((teacher_target.shape[0], args.align_dim - teacher_target.shape[1]), dtype=np.float32)
        teacher_target = np.concatenate([teacher_target, pad], axis=1)

    candidates, labels, texts, candidate_offsets = candidate_bundle(rows)
    valid_indices = [i for i, cands in enumerate(candidates) if cands and labels[i] >= 0]
    if len(valid_indices) < 4:
        return None, []
    embedder = TextEmbedder(
        kind=args.text_encoder,
        model_name=args.text_model,
        dim=args.text_dim,
        batch_size=args.text_batch_size,
        device=args.text_device,
        seed=args.seed,
    ).fit(texts)
    text_emb = embedder.transform(texts).astype(np.float32)
    text_emb = random_project(text_emb, args.joint_dim, args.seed + 17).astype(np.float32)
    stratify = np.asarray([rows[i].get(args.stratify_column, labels[i]) for i in valid_indices])
    all_true = []
    all_pred = []
    split_rows = []
    score_rows: list[dict[str, Any]] = []
    per_type: dict[str, list[int]] = {}
    text_accs = []

    for split_id, (train_local, test_local) in enumerate(make_cv(stratify, args.folds, args.repeats, args.seed)):
        train_indices = [valid_indices[int(i)] for i in train_local]
        test_indices = [valid_indices[int(i)] for i in test_local]
        preds, scores, stats = train_one_split(
            split_id,
            train_indices,
            test_indices,
            tokens,
            token_mask,
            neg_tokens,
            neg_mask,
            teacher_target,
            text_emb,
            candidate_offsets,
            labels,
            args,
            device,
        )
        text_accs.append(stats["text_accuracy"])
        split_correct = 0
        for local_i, row_idx in enumerate(test_indices):
            pred = int(preds[local_i])
            truth = int(labels[row_idx])
            all_true.append(truth)
            all_pred.append(pred)
            is_correct = int(pred == truth)
            split_correct += is_correct
            qtype = str(rows[row_idx].get("question_type", rows[row_idx].get("label", "all")))
            per_type.setdefault(qtype, []).append(is_correct)
            cands = parse_candidates(rows[row_idx])
            score_rows.append(
                {
                    "split": split_id,
                    "mode": mode,
                    "feature": f"wan_morepa_{safe_name(feature)}",
                    "source_feature": feature,
                    "teacher_feature": teacher_feature,
                    "classifier": "wan_morepa_residual",
                    "row_index": int(row_idx),
                    "video_id": row_id(rows[row_idx], row_idx),
                    "question_type": qtype,
                    "answer": rows[row_idx].get("answer"),
                    "answer_index": truth,
                    "prediction_index": pred,
                    "correct": bool(is_correct),
                    "candidates": [
                        {
                            "index": int(j),
                            "letter": cands[j]["letter"],
                            "text": cands[j]["text"],
                            "score": float(scores[local_i, j]),
                        }
                        for j in range(len(cands))
                    ],
                }
            )
        split_rows.append({"split": split_id, "accuracy": float(split_correct / max(1, len(test_indices))), "correct": split_correct, "total": len(test_indices)})

    correct = (np.asarray(all_true) == np.asarray(all_pred)).astype(np.float32)
    label_count = max(max(all_true + all_pred) + 1 if all_true else 0, 1)
    summary = {
        "mode": mode,
        "feature": f"wan_morepa_{safe_name(feature)}",
        "source_feature": feature,
        "teacher_feature": teacher_feature,
        "text_encoder": args.text_encoder,
        "accuracy_mean": float(np.mean([x["accuracy"] for x in split_rows])),
        "accuracy_std": float(np.std([x["accuracy"] for x in split_rows], ddof=0)),
        "text_only_accuracy_mean": float(np.mean(text_accs)) if text_accs else 0.0,
        "text_only_gain": float(np.mean([x["accuracy"] for x in split_rows]) - np.mean(text_accs)) if text_accs else 0.0,
        "correct": int(correct.sum()),
        "total": int(correct.size),
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(per_type.items())
        },
        "confusion_matrix": confusion_matrix(all_true, all_pred, labels=list(range(label_count))).tolist(),
        "splits": split_rows,
    }
    return summary, score_rows


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench Wan-MoREPA Residual Reranker", ""]
    lines.append("| mode | feature | teacher | acc | text-only | gain | correct/total |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for item in payload["results"]:
        lines.append(
            f"| {item['mode']} | {item['source_feature']} | {item['teacher_feature']} | "
            f"{item['accuracy_mean']:.4f} | {item['text_only_accuracy_mean']:.4f} | "
            f"{item['text_only_gain']:.4f} | {item['correct']}/{item['total']} |"
        )
    lines.extend(["", "## Per Question Type", ""])
    lines.append("| mode | feature | question_type | acc | correct/total |")
    lines.append("|---|---|---|---:|---:|")
    for item in payload["results"]:
        for qtype, stats in item["per_question_type"].items():
            lines.append(
                f"| {item['mode']} | {item['source_feature']} | {qtype} | "
                f"{stats['accuracy']:.4f} | {stats['correct']}/{stats['total']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-features-h5", required=True)
    parser.add_argument("--wan-metadata-jsonl", required=True)
    parser.add_argument("--teacher-h5", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-scores-jsonl", required=True)
    parser.add_argument("--feature-names", default="wan_vae_grid_1x1")
    parser.add_argument("--teacher-feature", default="motion_teacher_summary")
    parser.add_argument("--modes", default="high_motion+camera_comp")
    parser.add_argument("--text-encoder", default="hash", choices=["hash", "tfidf", "clip", "hf", "wan-t5", "umt5", "sentence-transformer"])
    parser.add_argument("--text-model", default="")
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--joint-dim", type=int, default=512)
    parser.add_argument("--text-classifier", default="ridge", choices=["ridge", "logistic"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--align-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-align", type=float, default=0.25)
    parser.add_argument("--lambda-contrast", type=float, default=0.1)
    parser.add_argument("--lambda-zero-mean", type=float, default=0.01)
    parser.add_argument("--residual-lambda", type=float, default=1.0)
    parser.add_argument("--contrast-margin", type=float, default=0.1)
    parser.add_argument("--hard-weight", type=float, default=2.0)
    parser.add_argument("--hard-margin-quantile", type=float, default=0.35)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--stratify-column", default="question_type")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows_by = aligned_rows_by_mode(args.wan_metadata_jsonl)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    features = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    results = []
    score_rows: list[dict[str, Any]] = []
    with h5py.File(args.wan_features_h5, "r") as wan_h5, h5py.File(args.teacher_h5, "r") as teacher_h5:
        for mode in modes:
            rows = rows_by.get(mode, [])
            if not rows:
                continue
            for feature in features:
                summary, scores = run_mode_feature(wan_h5, teacher_h5, mode, feature, args.teacher_feature, rows, args, device)
                if summary is not None:
                    results.append(summary)
                    score_rows.extend(scores)
    payload = {"config": vars(args), "results": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    write_jsonl(Path(args.output_scores_jsonl), score_rows)
    print(json.dumps({"results": len(results), "score_rows": len(score_rows), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
