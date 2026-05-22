#!/usr/bin/env python3
"""Train a MotionBench temporal adapter with Wan-Motion-REPA losses.

This is the executable version of wan_motion_repa_methods.md:

visual tokens -> temporal adapter -> QA residual scorer

loss = L_QA
     + lambda_align * Align(StudentMotion, WanTarget)
     + lambda_relation * AlignRelation(StudentSegments, WanSegments)
     + lambda_equiv * AlignEquivariance(StudentPerturbResponse, WanPerturbResponse)
     + lambda_contrast * TemporalContrast
     + lambda_preserve * PreserveBase

The script uses cached pixel/flow grid features as the frozen VideoLLM visual
token stream when real VideoLLM token dumps are not available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import TextEmbedder, candidate_feature_matrix, parse_candidates, qa_candidate_text, random_project, row_id  # noqa: E402
from motionbench_repa_common import (  # noqa: E402
    aligned_rows_by_mode,
    candidate_bundle,
    reduce_segments,
    safe_name,
    select_h5_rows,
    tokens_from_feature_array,
    vectorize_feature_array,
    write_jsonl,
)


CONTROL_NAMES = ["reverse", "shuffle", "first", "timeavg"]


@dataclass
class BranchSpec:
    name: str
    align_target: str = ""
    residualize: bool = False
    use_relation: bool = False
    use_equivariance: bool = False
    use_contrast: bool = False
    lambda_align_scale: float = 1.0
    lambda_relation_scale: float = 1.0
    lambda_equiv_scale: float = 1.0


class TemporalMotionAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, align_dim: int, segments: int, num_heads: int, num_layers: int, dropout: float, max_tokens: int):
        super().__init__()
        self.segments = int(segments)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.randn(max_tokens, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, align_dim)
        self.seg_out = nn.Linear(hidden_dim, align_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = tokens.shape[1]
        x = self.input_proj(tokens) + self.pos[:length].unsqueeze(0)
        key_padding = mask <= 0
        x = self.encoder(x, src_key_padding_mask=key_padding)
        x = self.norm(x)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        z = self.out(pooled)

        segs = []
        seg_masks = []
        boundaries = torch.linspace(0, length, self.segments + 1, device=tokens.device).round().long()
        for i in range(self.segments):
            start, end = int(boundaries[i].item()), int(boundaries[i + 1].item())
            if end <= start:
                end = min(length, start + 1)
            chunk = x[:, start:end]
            chunk_mask = mask[:, start:end]
            chunk_denom = chunk_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            segs.append((chunk * chunk_mask.unsqueeze(-1)).sum(dim=1) / chunk_denom)
            seg_masks.append((chunk_mask.sum(dim=1) > 0).float())
        seg = self.seg_out(torch.stack(segs, dim=1))
        seg_mask = torch.stack(seg_masks, dim=1)
        return z, seg, seg_mask


class MotionRepaReranker(nn.Module):
    def __init__(self, input_dim: int, text_dim: int, hidden_dim: int, align_dim: int, segments: int, heads: int, layers: int, dropout: float, max_tokens: int):
        super().__init__()
        self.adapter = TemporalMotionAdapter(input_dim, hidden_dim, align_dim, segments, heads, layers, dropout, max_tokens)
        self.text_proj = nn.Linear(text_dim, align_dim, bias=False)
        gate_hidden = max(16, align_dim // 4)
        self.gate = nn.Sequential(nn.Linear(align_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def score(self, tokens: torch.Tensor, mask: torch.Tensor, text: torch.Tensor, base: torch.Tensor, cand_mask: torch.Tensor, residual_lambda: float):
        z, seg, seg_mask = self.adapter(tokens, mask)
        z_norm = F.normalize(z, dim=-1)
        t = F.normalize(self.text_proj(text), dim=-1)
        residual = torch.einsum("bd,bcd->bc", z_norm, t)
        gate = torch.sigmoid(self.gate(z)).clamp(0.0, 1.0)
        logits = base + float(residual_lambda) * self.scale * gate * residual
        logits = logits.masked_fill(cand_mask <= 0, -1e4)
        return logits, residual, z, seg, seg_mask


def make_cv(stratify_labels: np.ndarray, folds: int, seed: int):
    enc = LabelEncoder()
    y = enc.fit_transform([str(x) for x in stratify_labels])
    counts = np.bincount(y)
    max_folds = int(counts[counts > 0].min()) if np.any(counts > 0) else 2
    folds = max(2, min(folds, max_folds))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(y)), y)


def train_text_scorer(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, seed: int):
    if args.text_classifier == "logistic":
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
    return np.asarray(scores, dtype=np.float32)


def text_pair_matrix(text_emb: np.ndarray, offsets: list[int]) -> np.ndarray:
    emb = text_emb[offsets].astype(np.float32)
    return candidate_feature_matrix(np.zeros_like(emb), emb)


def prepare_text_scores(model, text_emb: np.ndarray, offsets: list[list[int]], indices: list[int], max_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full((len(indices), max_candidates), -1e4, dtype=np.float32)
    mask = np.zeros((len(indices), max_candidates), dtype=np.float32)
    for out_i, row_idx in enumerate(indices):
        cand_offsets = offsets[row_idx]
        if not cand_offsets:
            continue
        row_scores = scorer_scores(model, text_pair_matrix(text_emb, cand_offsets))
        scores[out_i, : len(cand_offsets)] = row_scores
        mask[out_i, : len(cand_offsets)] = 1.0
    return scores, mask


def pack_text(text_emb: np.ndarray, offsets: list[list[int]], indices: list[int], max_candidates: int) -> np.ndarray:
    out = np.zeros((len(indices), max_candidates, text_emb.shape[1]), dtype=np.float32)
    for out_i, row_idx in enumerate(indices):
        cand_offsets = offsets[row_idx]
        if cand_offsets:
            out[out_i, : len(cand_offsets)] = text_emb[cand_offsets]
    return out


def load_visual_tokens(h5: h5py.File, rows: list[dict[str, Any]], mode: str, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pieces = []
    masks = []
    for feature in features:
        if mode not in h5 or feature not in h5[mode]:
            continue
        data = select_h5_rows(h5[mode][feature][:], rows)
        tokens, mask = tokens_from_feature_array(data, rows, feature)
        pieces.append(tokens.astype(np.float32))
        masks.append(mask.astype(np.float32))
    if not pieces:
        raise ValueError(f"No visual features found for mode={mode}, features={features}")
    max_len = max(piece.shape[1] for piece in pieces)
    padded = []
    out_mask = np.zeros((pieces[0].shape[0], max_len), dtype=np.float32)
    for piece, mask in zip(pieces, masks):
        if piece.shape[1] < max_len:
            pad = np.zeros((piece.shape[0], max_len - piece.shape[1], piece.shape[2]), dtype=np.float32)
            piece = np.concatenate([piece, pad], axis=1)
            mask = np.concatenate([mask, np.zeros((mask.shape[0], max_len - mask.shape[1]), dtype=np.float32)], axis=1)
        padded.append(piece)
        out_mask = np.maximum(out_mask, mask)
    return np.concatenate(padded, axis=2).astype(np.float32), out_mask.astype(np.float32)


def transform_tokens(tokens: np.ndarray, kind: str, seed: int) -> np.ndarray:
    out = tokens.copy()
    if kind == "reverse":
        return out[:, ::-1].copy()
    if kind == "shuffle":
        rng = np.random.default_rng(seed)
        for i in range(out.shape[0]):
            order = np.arange(out.shape[1])
            rng.shuffle(order)
            out[i] = out[i, order]
        return out
    if kind == "first":
        return np.repeat(out[:, :1], out.shape[1], axis=1)
    if kind == "timeavg":
        return np.repeat(out.mean(axis=1, keepdims=True), out.shape[1], axis=1)
    raise ValueError(kind)


def standardize_by_train(tokens: np.ndarray, train_indices: list[int]) -> np.ndarray:
    out = tokens.copy().astype(np.float32)
    flat = out[train_indices].reshape(-1, out.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    out = (out - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), 1e-6)
    return out.astype(np.float32)


def pad_width(x: np.ndarray, width: int) -> np.ndarray:
    if x.shape[1] == width:
        return x.astype(np.float32)
    if x.shape[1] > width:
        return x[:, :width].astype(np.float32)
    return np.concatenate([x, np.zeros((x.shape[0], width - x.shape[1]), dtype=np.float32)], axis=1).astype(np.float32)


def flatten_source_feature(h5: h5py.File, rows: list[dict[str, Any]], mode: str, feature: str, seed: int) -> np.ndarray:
    data = select_h5_rows(h5[mode][feature][:], rows)
    return vectorize_feature_array(data, rows, feature, transform="full", seed=seed).astype(np.float32)


def load_target(h5: h5py.File, rows: list[dict[str, Any]], mode: str, feature: str) -> np.ndarray:
    data = select_h5_rows(h5[mode][feature][:], rows)
    return data.reshape(data.shape[0], -1).astype(np.float32)


def residual_basis(source_h5: h5py.File, rows: list[dict[str, Any]], mode: str, wan_feature: str, seed: int) -> np.ndarray:
    pieces = []
    for feature in ["pixel_grid_sequence", "flow_grid_sequence"]:
        if feature in source_h5[mode]:
            data = select_h5_rows(source_h5[mode][feature][:], rows)
            pieces.append(vectorize_feature_array(data, rows, feature, transform="full", seed=seed))
    if wan_feature in source_h5[mode]:
        data = select_h5_rows(source_h5[mode][wan_feature][:], rows)
        pieces.append(vectorize_feature_array(data, rows, wan_feature, transform="first_frame_only", seed=seed))
        pieces.append(vectorize_feature_array(data, rows, wan_feature, transform="time_average", seed=seed))
    width = max(piece.shape[1] for piece in pieces)
    return np.concatenate([pad_width(piece, width) for piece in pieces], axis=1).astype(np.float32)


def random_project_to(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == out_dim:
        y = x
    elif x.shape[1] < out_dim:
        y = np.concatenate([x, np.zeros((x.shape[0], out_dim - x.shape[1]), dtype=np.float32)], axis=1)
    else:
        y = random_project(x, out_dim, seed)
    denom = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.maximum(denom, 1e-8)).astype(np.float32)


def residualize_target(target: np.ndarray, basis: np.ndarray, train_indices: list[int], test_indices: list[int], out_dim: int, alpha: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scaler_b = StandardScaler()
    scaler_t = StandardScaler()
    b_train = scaler_b.fit_transform(basis[train_indices])
    t_train = scaler_t.fit_transform(target[train_indices])
    model = Ridge(alpha=alpha)
    model.fit(b_train, t_train)
    train_res = target[train_indices] - scaler_t.inverse_transform(model.predict(b_train))
    test_res = target[test_indices] - scaler_t.inverse_transform(model.predict(scaler_b.transform(basis[test_indices])))
    return random_project_to(train_res, out_dim, seed), random_project_to(test_res, out_dim, seed)


def prepare_wan_segments(source_h5: h5py.File, rows: list[dict[str, Any]], mode: str, wan_feature: str, segments: int, align_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = select_h5_rows(source_h5[mode][wan_feature][:], rows)
    tokens, mask = tokens_from_feature_array(data, rows, wan_feature)
    seg, seg_mask = reduce_segments(tokens, mask, segments)
    flat = seg.reshape(-1, seg.shape[-1])
    flat_proj = random_project_to(flat, align_dim, seed)
    return flat_proj.reshape(seg.shape[0], seg.shape[1], align_dim).astype(np.float32), seg_mask.astype(np.float32)


def split_equivariance_target(eq: np.ndarray, align_dim: int, seed: int) -> np.ndarray:
    eq = np.asarray(eq, dtype=np.float32)
    chunks = np.array_split(eq, len(CONTROL_NAMES), axis=1)
    projected = [random_project_to(chunk, align_dim, seed + i * 991) for i, chunk in enumerate(chunks)]
    return np.stack(projected, axis=1).astype(np.float32)


def relation_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher, dim=-1)
    rs = torch.matmul(s, s.transpose(1, 2))
    rt = torch.matmul(t, t.transpose(1, 2))
    pair = mask.unsqueeze(1) * mask.unsqueeze(2)
    return ((rs - rt).pow(2) * pair).sum() / pair.sum().clamp_min(1.0)


def align_loss(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - (F.normalize(z, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean()


def equivariance_loss(model: MotionRepaReranker, normal_tokens, mask, controls, target, args) -> torch.Tensor:
    z0, _, _ = model.adapter(normal_tokens, mask)
    losses = []
    for i, name in enumerate(CONTROL_NAMES):
        zc, _, _ = model.adapter(controls[name], mask)
        response = F.normalize(z0 - zc, dim=-1)
        losses.append((response - F.normalize(target[:, i], dim=-1)).pow(2).mean())
    return torch.stack(losses).mean()


def contrast_loss(model: MotionRepaReranker, normal_tokens, mask, controls, margin: float) -> torch.Tensor:
    z0, _, _ = model.adapter(normal_tokens, mask)
    z_time, _, _ = model.adapter(controls["timeavg"], mask)
    pos = (F.normalize(z0, dim=-1) * F.normalize(z_time, dim=-1)).sum(dim=-1)
    vals = []
    for name in ["reverse", "shuffle", "first"]:
        zn, _, _ = model.adapter(controls[name], mask)
        neg = (F.normalize(z0, dim=-1) * F.normalize(zn, dim=-1)).sum(dim=-1)
        vals.append(F.relu(neg - pos + float(margin)).mean())
    return torch.stack(vals).mean()


def branch_specs() -> list[BranchSpec]:
    return [
        BranchSpec("qa_only"),
        BranchSpec("raw_wan", align_target="raw_wan"),
        BranchSpec("structured_compact", align_target="structured_compact"),
        BranchSpec("dynamics_relation", align_target="dynamics_relation"),
        BranchSpec("residualized_dynamics_relation", align_target="dynamics_relation", residualize=True),
        BranchSpec("relation_only", use_relation=True),
        BranchSpec("equivariance", use_equivariance=True, use_contrast=True),
        BranchSpec("multi_target", align_target="multi_target", use_relation=True, use_equivariance=True, use_contrast=True),
    ]


def train_split(
    branch: BranchSpec,
    split_id: int,
    train_indices: list[int],
    test_indices: list[int],
    visual_tokens: np.ndarray,
    visual_mask: np.ndarray,
    target_vectors: dict[str, np.ndarray],
    target_basis: np.ndarray,
    teacher_segments: np.ndarray,
    teacher_segment_mask: np.ndarray,
    equiv_targets: np.ndarray,
    text_emb: np.ndarray,
    offsets: list[list[int]],
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_chunks = []
    y_train = []
    for row_idx in train_indices:
        cand_offsets = offsets[row_idx]
        x_chunks.append(text_pair_matrix(text_emb, cand_offsets))
        y_train.extend([1 if j == labels[row_idx] else 0 for j in range(len(cand_offsets))])
    text_model = train_text_scorer(np.concatenate(x_chunks, axis=0), np.asarray(y_train, dtype=np.int64), args, args.seed + split_id)
    max_candidates = max(len(offsets[i]) for i in train_indices + test_indices)
    train_base, train_cmask = prepare_text_scores(text_model, text_emb, offsets, train_indices, max_candidates)
    test_base, test_cmask = prepare_text_scores(text_model, text_emb, offsets, test_indices, max_candidates)
    train_text = pack_text(text_emb, offsets, train_indices, max_candidates)
    test_text = pack_text(text_emb, offsets, test_indices, max_candidates)
    train_labels = labels[train_indices].astype(np.int64)
    test_labels = labels[test_indices].astype(np.int64)

    tokens_std = standardize_by_train(visual_tokens, train_indices)
    control_tokens = {
        name: standardize_by_train(transform_tokens(visual_tokens, name, args.seed + 100 * split_id + i), train_indices)
        for i, name in enumerate(CONTROL_NAMES)
    }

    train_align = test_align = None
    if branch.align_target:
        raw_target = target_vectors[branch.align_target]
        if branch.residualize:
            train_align, test_align = residualize_target(raw_target, target_basis, train_indices, test_indices, args.align_dim, args.residual_alpha, args.seed + split_id + 700)
        else:
            projected = random_project_to(raw_target, args.align_dim, args.seed + split_id + 700)
            train_align = projected[train_indices]
            test_align = projected[test_indices]

    model = MotionRepaReranker(
        input_dim=visual_tokens.shape[-1],
        text_dim=text_emb.shape[-1],
        hidden_dim=args.hidden_dim,
        align_dim=args.align_dim,
        segments=args.segments,
        heads=args.num_heads,
        layers=args.num_layers,
        dropout=args.dropout,
        max_tokens=visual_tokens.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_t = torch.from_numpy(tokens_std[train_indices]).to(device)
    train_m = torch.from_numpy(visual_mask[train_indices]).to(device)
    train_controls = {name: torch.from_numpy(arr[train_indices]).to(device) for name, arr in control_tokens.items()}
    train_text_t = torch.from_numpy(train_text).to(device)
    train_base_t = torch.from_numpy(train_base).to(device)
    train_cmask_t = torch.from_numpy(train_cmask).to(device)
    train_labels_t = torch.from_numpy(train_labels).to(device)
    train_seg_t = torch.from_numpy(teacher_segments[train_indices]).to(device)
    train_seg_m = torch.from_numpy(teacher_segment_mask[train_indices]).to(device)
    train_eq_t = torch.from_numpy(equiv_targets[train_indices]).to(device)
    train_align_t = torch.from_numpy(train_align).to(device) if train_align is not None else None

    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_indices), device=device)
        epoch_losses = []
        for start in range(0, len(train_indices), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits, residual, z, seg, seg_mask = model.score(
                train_t[idx], train_m[idx], train_text_t[idx], train_base_t[idx], train_cmask_t[idx], args.residual_lambda
            )
            qa = F.cross_entropy(logits, train_labels_t[idx])
            loss = qa
            align = torch.zeros((), device=device)
            rel = torch.zeros((), device=device)
            equiv = torch.zeros((), device=device)
            ctr = torch.zeros((), device=device)
            preserve = residual.mean(dim=1).pow(2).mean()
            if train_align_t is not None:
                align = align_loss(z, train_align_t[idx])
                loss = loss + float(args.lambda_align) * branch.lambda_align_scale * align
            if branch.use_relation:
                common_mask = seg_mask * train_seg_m[idx]
                rel = relation_loss(seg, train_seg_t[idx], common_mask)
                loss = loss + float(args.lambda_relation) * branch.lambda_relation_scale * rel
            if branch.use_equivariance:
                equiv = equivariance_loss(model, train_t[idx], train_m[idx], {k: v[idx] for k, v in train_controls.items()}, train_eq_t[idx], args)
                loss = loss + float(args.lambda_equiv) * branch.lambda_equiv_scale * equiv
            if branch.use_contrast and args.lambda_contrast > 0:
                ctr = contrast_loss(model, train_t[idx], train_m[idx], {k: v[idx] for k, v in train_controls.items()}, args.contrast_margin)
                loss = loss + float(args.lambda_contrast) * ctr
            loss = loss + float(args.lambda_preserve) * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append([float(qa.detach().cpu()), float(align.detach().cpu()), float(rel.detach().cpu()), float(equiv.detach().cpu()), float(ctr.detach().cpu())])
        if epoch in {0, args.epochs - 1}:
            arr = np.asarray(epoch_losses, dtype=np.float64)
            history.append(
                {
                    "epoch": int(epoch),
                    "qa_loss": float(arr[:, 0].mean()),
                    "align_loss": float(arr[:, 1].mean()),
                    "relation_loss": float(arr[:, 2].mean()),
                    "equivariance_loss": float(arr[:, 3].mean()),
                    "contrast_loss": float(arr[:, 4].mean()),
                }
            )

    model.eval()
    test_t = torch.from_numpy(tokens_std[test_indices]).to(device)
    test_m = torch.from_numpy(visual_mask[test_indices]).to(device)
    test_text_t = torch.from_numpy(test_text).to(device)
    test_base_t = torch.from_numpy(test_base).to(device)
    test_cmask_t = torch.from_numpy(test_cmask).to(device)
    preds = []
    score_rows = []
    with torch.no_grad():
        for start in range(0, len(test_indices), args.eval_batch_size):
            end = min(len(test_indices), start + args.eval_batch_size)
            logits, _, _, _, _ = model.score(
                test_t[start:end],
                test_m[start:end],
                test_text_t[start:end],
                test_base_t[start:end],
                test_cmask_t[start:end],
                args.residual_lambda,
            )
            score_rows.append(logits.cpu().numpy())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist())
    scores = np.concatenate(score_rows, axis=0)
    text_preds = np.argmax(test_base, axis=1)
    return np.asarray(preds, dtype=np.int64), scores, {"text_accuracy": float((text_preds == test_labels).mean()), "history": history}


def run_branch(
    branch: BranchSpec,
    rows: list[dict[str, Any]],
    valid_indices: list[int],
    candidates: list[list[dict[str, Any]]],
    labels: np.ndarray,
    offsets: list[list[int]],
    text_emb: np.ndarray,
    visual_tokens: np.ndarray,
    visual_mask: np.ndarray,
    target_vectors: dict[str, np.ndarray],
    target_basis: np.ndarray,
    teacher_segments: np.ndarray,
    teacher_segment_mask: np.ndarray,
    equiv_targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stratify = np.asarray([rows[i].get(args.stratify_column, labels[i]) for i in valid_indices])
    all_true = []
    all_pred = []
    text_accs = []
    histories = []
    score_output: list[dict[str, Any]] = []
    per_type: dict[str, list[int]] = {}
    split_rows = []
    for split_id, (train_local, test_local) in enumerate(make_cv(stratify, args.folds, args.seed)):
        train_indices = [valid_indices[int(i)] for i in train_local]
        test_indices = [valid_indices[int(i)] for i in test_local]
        preds, scores, stats = train_split(
            branch,
            split_id,
            train_indices,
            test_indices,
            visual_tokens,
            visual_mask,
            target_vectors,
            target_basis,
            teacher_segments,
            teacher_segment_mask,
            equiv_targets,
            text_emb,
            offsets,
            labels,
            args,
            device,
        )
        text_accs.append(stats["text_accuracy"])
        histories.append({"split": split_id, **stats})
        split_correct = 0
        for local_i, row_idx in enumerate(test_indices):
            pred = int(preds[local_i])
            truth = int(labels[row_idx])
            correct = int(pred == truth)
            split_correct += correct
            all_true.append(truth)
            all_pred.append(pred)
            qtype = str(rows[row_idx].get("question_type", "all"))
            per_type.setdefault(qtype, []).append(correct)
            cands = parse_candidates(rows[row_idx])
            score_output.append(
                {
                    "split": split_id,
                    "mode": args.mode,
                    "feature": f"adapter_{branch.name}",
                    "source_feature": ",".join(args.visual_features.split(",")),
                    "classifier": "wan_motion_repa_adapter",
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
                            "letter": cands[j]["letter"],
                            "text": cands[j]["text"],
                            "score": float(scores[local_i, j]),
                        }
                        for j in range(len(cands))
                    ],
                }
            )
        split_rows.append({"split": split_id, "accuracy": float(split_correct / max(1, len(test_indices))), "correct": split_correct, "total": len(test_indices)})

    truth_arr = np.asarray(all_true, dtype=np.int64)
    pred_arr = np.asarray(all_pred, dtype=np.int64)
    correct_arr = truth_arr == pred_arr
    label_count = max(max(all_true + all_pred) + 1 if all_true else 0, 1)
    summary = {
        "feature": f"adapter_{branch.name}",
        "branch": branch.__dict__,
        "accuracy_mean": float(np.mean([row["accuracy"] for row in split_rows])),
        "accuracy_std": float(np.std([row["accuracy"] for row in split_rows], ddof=0)),
        "text_only_accuracy_mean": float(np.mean(text_accs)) if text_accs else 0.0,
        "text_only_gain": float(np.mean([row["accuracy"] for row in split_rows]) - np.mean(text_accs)) if text_accs else 0.0,
        "correct": int(correct_arr.sum()),
        "total": int(correct_arr.size),
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(per_type.items())
        },
        "confusion_matrix": confusion_matrix(truth_arr, pred_arr, labels=list(range(label_count))).tolist(),
        "splits": split_rows,
        "train_history": histories,
    }
    return summary, score_output


def short_type(item: dict[str, Any], qtype: str) -> str:
    return f"{item.get('per_question_type', {}).get(qtype, {}).get('accuracy', 0.0):.4f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    qtypes = payload["question_types"]
    lines = ["# MotionBench Wan-Motion-REPA Adapter Training", ""]
    lines.append("| Model | Acc | Text-only | Gain | Correct/total | " + " | ".join(qtypes) + " |")
    lines.append("|---|---:|---:|---:|---:|" + "|".join(["---:"] * len(qtypes)) + "|")
    for item in payload["results"]:
        lines.append(
            f"| {item['feature']} | {item['accuracy_mean']:.4f} | {item['text_only_accuracy_mean']:.4f} | "
            f"{item['text_only_gain']:+.4f} | {item['correct']}/{item['total']} | "
            + " | ".join(short_type(item, q) for q in qtypes)
            + " |"
        )
    lines.append("")
    lines.append("## Loss Setup")
    lines.append("")
    lines.append(
        "`L = L_QA + lambda_align*L_align + lambda_relation*L_relation + lambda_equiv*L_equivariance "
        "+ lambda_contrast*L_temporal_contrast + lambda_preserve*L_preserve_base`"
    )
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
    parser.add_argument("--wan-feature", default="wan_vae_grid_1x1")
    parser.add_argument("--target-prefix", default="wmrepa")
    parser.add_argument("--visual-features", default="pixel_grid_sequence,flow_grid_sequence")
    parser.add_argument("--branches", default="qa_only,raw_wan,structured_compact,dynamics_relation,residualized_dynamics_relation,relation_only,equivariance,multi_target")
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
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-lambda", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.25)
    parser.add_argument("--lambda-relation", type=float, default=0.25)
    parser.add_argument("--lambda-equiv", type=float, default=0.1)
    parser.add_argument("--lambda-contrast", type=float, default=0.05)
    parser.add_argument("--lambda-preserve", type=float, default=0.01)
    parser.add_argument("--contrast-margin", type=float, default=0.1)
    parser.add_argument("--residual-alpha", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--stratify-column", default="question_type")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    target_rows_by = aligned_rows_by_mode(args.target_metadata_jsonl)
    source_rows_by = aligned_rows_by_mode(args.source_metadata_jsonl)
    rows = target_rows_by.get(args.mode, [])
    source_rows = source_rows_by.get(args.mode, [])
    if rows and source_rows and len(rows) == len(source_rows):
        rows = [{**source_rows[i], **rows[i], **{k: v for k, v in source_rows[i].items() if k not in rows[i]}} for i in range(len(rows))]
    if not rows:
        rows = source_rows
    if not rows:
        raise ValueError(f"No rows for mode {args.mode}")

    candidates, labels, texts, offsets = candidate_bundle(rows)
    valid = [i for i, cands in enumerate(candidates) if cands and labels[i] >= 0]
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

    safe_wan = safe_name(args.wan_feature)
    target_feature_names = {
        "structured_compact": f"{args.target_prefix}_structured_compact_{safe_wan}",
        "dynamics_relation": f"{args.target_prefix}_dynamics_relation_{safe_wan}",
        "relation_only": f"{args.target_prefix}_relation_only_{safe_wan}",
        "equivariance": f"{args.target_prefix}_equivariance_{safe_wan}",
        "multi_target": f"{args.target_prefix}_multi_target_{safe_wan}",
    }

    with h5py.File(args.source_features_h5, "r") as source_h5, h5py.File(args.target_features_h5, "r") as target_h5:
        visual_features = [x.strip() for x in args.visual_features.split(",") if x.strip()]
        visual_tokens, visual_mask = load_visual_tokens(source_h5, rows, args.mode, visual_features)
        target_vectors = {
            "raw_wan": flatten_source_feature(source_h5, rows, args.mode, args.wan_feature, args.seed),
        }
        for key, feature in target_feature_names.items():
            if args.mode in target_h5 and feature in target_h5[args.mode]:
                target_vectors[key] = load_target(target_h5, rows, args.mode, feature)
        target_basis = residual_basis(source_h5, rows, args.mode, args.wan_feature, args.seed)
        teacher_segments, teacher_segment_mask = prepare_wan_segments(source_h5, rows, args.mode, args.wan_feature, args.segments, args.align_dim, args.seed + 333)
        equiv_raw = target_vectors.get("equivariance")
        if equiv_raw is None:
            raise ValueError("Equivariance target not found")
        equiv_targets = split_equivariance_target(equiv_raw, args.align_dim, args.seed + 444)

        allowed = {spec.name: spec for spec in branch_specs()}
        wanted = [x.strip() for x in args.branches.split(",") if x.strip()]
        results = []
        score_rows: list[dict[str, Any]] = []
        for name in wanted:
            if name not in allowed:
                raise ValueError(f"Unknown branch: {name}")
            summary, scores = run_branch(
                allowed[name],
                rows,
                valid,
                candidates,
                labels,
                offsets,
                text_emb,
                visual_tokens,
                visual_mask,
                target_vectors,
                target_basis,
                teacher_segments,
                teacher_segment_mask,
                equiv_targets,
                args,
                device,
            )
            results.append(summary)
            score_rows.extend(scores)

    qtypes = sorted({str(rows[i].get("question_type", "all")) for i in valid})
    payload = {"config": vars(args), "device": str(device), "question_types": qtypes, "results": results}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(Path(args.output_md), payload)
    write_jsonl(Path(args.output_scores_jsonl), score_rows)
    print(json.dumps({"results": len(results), "score_rows": len(score_rows), "device": str(device), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
