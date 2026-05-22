#!/usr/bin/env python3
"""Shared helpers for Wan REPA-style MotionBench experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from motionbench_common import (
    answer_index,
    feature_to_temporal,
    parse_candidates,
    qa_candidate_text,
    read_jsonl,
    row_id,
    rows_by_mode,
    write_jsonl,
)


def stable_seed(*parts: Any, base: int = 0) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int((int(digest[:12], 16) + int(base)) % 1_000_000_007)


def safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "value"


def copy_h5_attrs(src: h5py.Group | h5py.File, dst: h5py.Group | h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def aligned_rows_by_mode(metadata_jsonl: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(Path(metadata_jsonl))
    grouped = rows_by_mode(rows)
    for mode_rows in grouped.values():
        for idx, row in enumerate(mode_rows):
            row["__h5_index"] = idx
    return grouped


def select_h5_rows(data: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    indices = [int(row.get("__h5_index", i)) for i, row in enumerate(rows)]
    if indices != list(range(len(rows))):
        data = data[indices]
    return data


def temporal_vector(
    value: np.ndarray,
    row: dict[str, Any] | None,
    feature_name: str,
    transform: str = "full",
    seed: int = 0,
) -> np.ndarray:
    temporal = feature_to_temporal(value, row=row, feature_name=feature_name)
    if transform == "full":
        return temporal.reshape(-1)
    if transform == "first_frame_only":
        one = temporal[:1]
        if temporal.shape[0] > 1:
            one = np.repeat(one, temporal.shape[0], axis=0)
        return one.reshape(-1)
    if transform == "time_average":
        avg = temporal.mean(axis=0, keepdims=True)
        if temporal.shape[0] > 1:
            avg = np.repeat(avg, temporal.shape[0], axis=0)
        return avg.reshape(-1)
    if transform == "shuffled":
        rng = np.random.default_rng(seed)
        order = np.arange(len(temporal))
        rng.shuffle(order)
        return temporal[order].reshape(-1)
    if transform == "reversed":
        return temporal[::-1].reshape(-1)
    raise ValueError(f"Unknown transform: {transform}")


def vectorize_feature_array(
    data: np.ndarray,
    rows: list[dict[str, Any]],
    feature_name: str,
    transform: str = "full",
    seed: int = 0,
) -> np.ndarray:
    vectors = [
        temporal_vector(data[i], rows[i] if i < len(rows) else None, feature_name, transform=transform, seed=seed + i)
        for i in range(len(rows))
    ]
    max_len = max((vec.size for vec in vectors), default=0)
    out = np.zeros((len(vectors), max_len), dtype=np.float32)
    for i, vec in enumerate(vectors):
        out[i, : vec.size] = vec.astype(np.float32)
    return out


def tokens_from_feature_array(
    data: np.ndarray,
    rows: list[dict[str, Any]],
    feature_name: str,
    max_tokens: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded [N,S,C] temporal tokens and [N,S] validity mask."""
    tokens = []
    for i in range(len(rows)):
        row = rows[i]
        tok = feature_to_temporal(data[i], row=row, feature_name=feature_name).astype(np.float32)
        length_keys = [f"{feature_name}_length", f"{feature_name.rstrip('s')}_length"]
        for length_key in length_keys:
            if length_key in row:
                tok = tok[: max(0, min(int(row[length_key]), tok.shape[0]))]
                break
        tokens.append(tok)
    if max_tokens > 0:
        tokens = [tok[:max_tokens] for tok in tokens]
    token_count = max((tok.shape[0] for tok in tokens), default=0)
    channel_count = max((tok.shape[1] for tok in tokens), default=0)
    out = np.zeros((len(tokens), token_count, channel_count), dtype=np.float32)
    mask = np.zeros((len(tokens), token_count), dtype=np.float32)
    for i, tok in enumerate(tokens):
        s, c = tok.shape
        out[i, :s, :c] = tok
        mask[i, :s] = 1.0
    return out, mask


def reduce_segments(tokens: np.ndarray, mask: np.ndarray, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool [N,S,C] tokens into a fixed number of ordered segments."""
    n, s, c = tokens.shape
    segments = max(1, int(segments))
    out = np.zeros((n, segments, c), dtype=np.float32)
    out_mask = np.zeros((n, segments), dtype=np.float32)
    if s == 0:
        return out, out_mask
    boundaries = np.linspace(0, s, segments + 1).round().astype(int)
    for seg in range(segments):
        start, end = int(boundaries[seg]), int(boundaries[seg + 1])
        if end <= start:
            end = min(s, start + 1)
        chunk = tokens[:, start:end]
        chunk_mask = mask[:, start:end]
        denom = np.maximum(chunk_mask.sum(axis=1, keepdims=True), 1.0)
        out[:, seg] = (chunk * chunk_mask[..., None]).sum(axis=1) / denom
        out_mask[:, seg] = (chunk_mask.sum(axis=1) > 0).astype(np.float32)
    return out, out_mask


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def random_project_fit(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    if in_dim == out_dim:
        return np.eye(in_dim, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / math.sqrt(max(1, out_dim)), size=(in_dim, out_dim)).astype(np.float32)


def apply_projection(x: np.ndarray, projection: np.ndarray, out_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == out_dim and projection.shape[0] == projection.shape[1] == out_dim:
        return x
    if x.shape[1] == projection.shape[0]:
        return x @ projection
    if x.shape[1] < out_dim:
        pad = np.zeros((x.shape[0], out_dim - x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)
    return x[:, :out_dim]


def candidate_bundle(rows: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], np.ndarray, list[str], list[list[int]]]:
    candidates: list[list[dict[str, Any]]] = []
    labels = []
    texts: list[str] = []
    offsets: list[list[int]] = []
    for row in rows:
        cands = parse_candidates(row)
        label = answer_index(row)
        if label is None or label >= len(cands):
            cands = []
            label = -1
        candidates.append(cands)
        labels.append(int(label))
        row_offsets = []
        for cand in cands:
            row_offsets.append(len(texts))
            texts.append(qa_candidate_text(row, cand))
        offsets.append(row_offsets)
    return candidates, np.asarray(labels, dtype=np.int64), texts, offsets


def candidate_scores_from_row(row: dict[str, Any]) -> np.ndarray:
    if "candidates" in row and isinstance(row["candidates"], list):
        return np.asarray([float(c.get("score", c.get("logprob", 0.0))) for c in row["candidates"]], dtype=np.float64)
    scores = row.get("scores")
    if isinstance(scores, list):
        return np.asarray(scores, dtype=np.float64)
    if isinstance(scores, dict):
        return np.asarray([float(scores[k]) for k in sorted(scores)], dtype=np.float64)
    raise KeyError("Could not find candidate scores")


def score_row_key(row: dict[str, Any], include_split: bool = True) -> tuple[str, ...]:
    key = [
        str(row.get("mode", "")),
        str(row.get("row_index", "")),
        str(row.get("video_id", row.get("uid", row.get("id", row.get("row_index", ""))))),
    ]
    if include_split:
        key.insert(0, str(row.get("split", 0)))
    return tuple(key)


def metric_from_predictions(rows: list[dict[str, Any]], predictions: Iterable[int]) -> dict[str, Any]:
    correct = []
    by_type: dict[str, list[int]] = {}
    preds = list(predictions)
    for row, pred in zip(rows, preds):
        truth = int(row.get("answer_index", answer_index(row) if answer_index(row) is not None else -1))
        val = int(int(pred) == truth)
        correct.append(val)
        by_type.setdefault(str(row.get("question_type", "all")), []).append(val)
    arr = np.asarray(correct, dtype=np.float64)
    return {
        "accuracy": float(arr.mean()) if arr.size else 0.0,
        "correct": int(arr.sum()),
        "total": int(arr.size),
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(by_type.items())
        },
        "correct_vector": correct,
    }


def write_markdown_table(path: Path, title: str, headers: list[str], rows: list[list[Any]]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "aligned_rows_by_mode",
    "apply_projection",
    "candidate_bundle",
    "candidate_scores_from_row",
    "copy_h5_attrs",
    "l2_normalize_rows",
    "metric_from_predictions",
    "random_project_fit",
    "read_jsonl",
    "reduce_segments",
    "row_id",
    "safe_name",
    "score_row_key",
    "select_h5_rows",
    "stable_seed",
    "tokens_from_feature_array",
    "vectorize_feature_array",
    "write_jsonl",
    "write_markdown_table",
]
