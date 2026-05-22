#!/usr/bin/env python3
"""Shared utilities for MotionBench Wan-feature experiments."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHOICE_RE = re.compile(r"^\s*([A-Z])[\.\)]\s*(.+?)\s*$")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidates as [{"letter": "A", "text": "..."}]."""
    raw = row.get("candidates") or row.get("choices")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed
        except Exception:
            raw = None
    if isinstance(raw, list) and raw:
        out = []
        for idx, item in enumerate(raw):
            letter = chr(ord("A") + idx)
            text = str(item)
            match = CHOICE_RE.match(text)
            if match:
                letter, text = match.group(1), match.group(2)
            out.append({"letter": letter, "text": text.strip()})
        return out

    question = str(row.get("question", ""))
    out = []
    seen = set()
    for line in question.splitlines():
        match = CHOICE_RE.match(line)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            out.append({"letter": match.group(1), "text": match.group(2).strip()})
    return out


def answer_letter(row: dict[str, Any]) -> str | None:
    answer = row.get("answer")
    if answer is None:
        return None
    answer = str(answer).strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    for cand in parse_candidates(row):
        if answer == cand["text"] or answer.endswith(cand["text"]):
            return cand["letter"]
    return answer[:1].upper() if answer else None


def answer_index(row: dict[str, Any]) -> int | None:
    letter = answer_letter(row)
    if not letter:
        return None
    candidates = parse_candidates(row)
    for idx, cand in enumerate(candidates):
        if cand["letter"] == letter:
            return idx
    if "A" <= letter <= "Z":
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(candidates):
            return idx
    return None


def question_text_without_choices(row: dict[str, Any]) -> str:
    lines = []
    for line in str(row.get("question", "")).splitlines():
        if not CHOICE_RE.match(line):
            lines.append(line.strip())
    return " ".join(x for x in lines if x)


def qa_candidate_text(row: dict[str, Any], candidate: dict[str, Any]) -> str:
    return f"Question: {question_text_without_choices(row)} Answer: {candidate['text']}"


def row_id(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("uid") or row.get("video_id") or row.get("id") or row.get("key") or fallback)


def rows_by_mode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("lowfps_mode", "none")), []).append(row)
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def random_project(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if out_dim <= 0 or x.shape[1] == out_dim:
        return x
    if x.shape[1] < out_dim:
        pad = np.zeros((x.shape[0], out_dim - x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(out_dim), size=(x.shape[1], out_dim)).astype(np.float32)
    return x @ proj


def feature_to_temporal(value: np.ndarray, row: dict[str, Any] | None = None, feature_name: str = "") -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4:
        return value.reshape(value.shape[0], -1)
    if value.ndim == 3:
        return value.reshape(value.shape[0], -1)
    if value.ndim == 2:
        return value
    if value.ndim == 1:
        if row is not None:
            latent_shape = row.get("vae_latent_shape")
            if feature_name.startswith("wan_vae_global") and latent_shape and len(latent_shape) >= 2:
                temporal = int(latent_shape[1])
                if temporal > 0 and value.size % temporal == 0:
                    return value.reshape(temporal, -1)
            num_frames = int(row.get("num_frames", 0) or 0)
            if num_frames > 0 and value.size % num_frames == 0:
                return value.reshape(num_frames, -1)
        return value.reshape(1, -1)
    return value.reshape(value.shape[0], -1)


def reduce_temporal_feature(
    value: np.ndarray,
    row: dict[str, Any] | None = None,
    feature_name: str = "",
    transform: str = "full",
    seed: int = 0,
) -> np.ndarray:
    temporal = feature_to_temporal(value, row=row, feature_name=feature_name)
    if transform == "full":
        return temporal.reshape(-1)
    if transform == "first_frame_only":
        return temporal[:1].reshape(-1)
    if transform == "time_average":
        return temporal.mean(axis=0).reshape(-1)
    if transform == "reversed":
        return temporal[::-1].reshape(-1)
    if transform == "shuffled":
        rng = np.random.default_rng(seed)
        order = np.arange(len(temporal))
        rng.shuffle(order)
        return temporal[order].reshape(-1)
    raise ValueError(f"Unknown temporal transform: {transform}")


def flatten_dataset(data: np.ndarray, rows: list[dict[str, Any]], feature_name: str, transform: str = "full", seed: int = 0) -> np.ndarray:
    vectors = [
        reduce_temporal_feature(data[i], row=rows[i], feature_name=feature_name, transform=transform, seed=seed + i)
        for i in range(len(rows))
    ]
    max_len = max(vec.size for vec in vectors)
    out = np.zeros((len(vectors), max_len), dtype=np.float32)
    for i, vec in enumerate(vectors):
        out[i, : vec.size] = vec.astype(np.float32)
    return out


def metadata_features(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    numeric_names = ["fps", "duration", "original_fps", "original_frame_count", "num_frames"]
    categorical_names = ["question_type", "video_type", "lowfps_mode"]
    cat_values = {name: sorted({str(row.get(name, "")) for row in rows}) for name in categorical_names}
    names = list(numeric_names)
    for name, values in cat_values.items():
        names.extend([f"{name}={value}" for value in values])
    x = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, row in enumerate(rows):
        col = 0
        for name in numeric_names:
            value = row.get(name)
            try:
                x[i, col] = float(value) if value not in (None, "") else 0.0
            except Exception:
                x[i, col] = 0.0
            col += 1
        for name in categorical_names:
            value = str(row.get(name, ""))
            values = cat_values[name]
            if value in values:
                x[i, col + values.index(value)] = 1.0
            col += len(values)
    return x, names


def bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boots[i] = sample.mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1.0 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, seed: int = 0, n_boot: int = 2000) -> dict[str, float]:
    correct_a = np.asarray(correct_a, dtype=np.float64)
    correct_b = np.asarray(correct_b, dtype=np.float64)
    n = min(correct_a.size, correct_b.size)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diff = correct_a[:n] - correct_b[:n]
    lo, hi = bootstrap_ci(diff, seed=seed, n_boot=n_boot)
    return {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)
    n = min(correct_a.size, correct_b.size)
    a = correct_a[:n]
    b = correct_b[:n]
    b_only = int((~a & b).sum())
    a_only = int((a & ~b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        prob = sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant)
        p = min(1.0, 2.0 * prob)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": float(p)}


def normalize_scores(scores: np.ndarray, method: str, temperature: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if method == "raw":
        return scores
    if method == "zscore":
        std = scores.std()
        return (scores - scores.mean()) / (std if std > 1e-8 else 1.0)
    if method == "rank":
        order = np.argsort(np.argsort(scores))
        return order.astype(np.float64) / max(1, len(scores) - 1)
    if method == "softmax":
        temp = max(float(temperature), 1e-6)
        x = scores / temp
        x = x - x.max()
        exp = np.exp(x)
        return exp / max(float(exp.sum()), 1e-12)
    raise ValueError(f"Unknown score normalization: {method}")


@dataclass
class TextEmbedder:
    kind: str = "hash"
    model_name: str = ""
    dim: int = 512
    batch_size: int = 32
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        self.kind = self.kind.lower().strip()
        self.model = None
        self.tokenizer = None
        self.vectorizer = None

    def fit(self, texts: list[str]) -> "TextEmbedder":
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(max_features=self.dim, ngram_range=(1, 2), min_df=1)
            self.vectorizer.fit(texts)
        elif self.kind == "hash":
            from sklearn.feature_extraction.text import HashingVectorizer

            self.vectorizer = HashingVectorizer(n_features=self.dim, alternate_sign=False, norm=None, ngram_range=(1, 2))
        elif self.kind in {"clip", "hf", "wan-t5", "umt5", "sentence-transformer"}:
            self._load_neural_model()
        else:
            raise ValueError(f"Unknown text encoder kind: {self.kind}")
        return self

    def _load_neural_model(self) -> None:
        if self.model is not None:
            return
        if self.kind == "sentence-transformer":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name or "sentence-transformers/all-MiniLM-L6-v2", device=self.device)
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

        if self.kind == "clip":
            name = self.model_name or "openai/clip-vit-base-patch32"
            self.tokenizer = CLIPTokenizer.from_pretrained(name)
            self.model = CLIPTextModel.from_pretrained(name).to(self.device)
        else:
            name = self.model_name
            if not name:
                if self.kind in {"wan-t5", "umt5"}:
                    name = "google/umt5-small"
                else:
                    name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name).to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is not None:
            x = self.vectorizer.transform(texts).astype(np.float32)
            arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x, dtype=np.float32)
            return l2_normalize(arr.astype(np.float32))
        if self.kind == "sentence-transformer":
            arr = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)
            return arr.astype(np.float32)
        self._load_neural_model()
        import torch

        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**tokens)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output
                else:
                    mask = tokens["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
                    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(emb.float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs, axis=0).astype(np.float32))


def candidate_feature_matrix(video_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    if video_emb.shape[0] != text_emb.shape[0]:
        raise ValueError("video_emb and text_emb must have same row count")
    if video_emb.shape[1] != text_emb.shape[1]:
        return np.concatenate([video_emb, text_emb], axis=1).astype(np.float32)
    return np.concatenate([video_emb, text_emb, video_emb * text_emb, np.abs(video_emb - text_emb)], axis=1).astype(np.float32)
