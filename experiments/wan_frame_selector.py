#!/usr/bin/env python3
"""Select compact frame sets from cached Wan/pixel temporal features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def temporal_vectors(feature: np.ndarray, row: dict[str, Any], feature_name: str) -> np.ndarray:
    value = np.asarray(feature, dtype=np.float32)
    if value.ndim >= 2:
        return value.reshape(value.shape[0], -1)
    latent_shape = row.get("vae_latent_shape")
    if feature_name.startswith("wan_vae_global") and latent_shape and len(latent_shape) >= 2:
        temporal = int(latent_shape[1])
        if temporal > 0 and value.size % temporal == 0:
            return value.reshape(temporal, -1)
    num_frames = int(row.get("num_frames", 0) or 0)
    if num_frames > 0 and value.size % num_frames == 0:
        return value.reshape(num_frames, -1)
    return value.reshape(1, -1)


def motion_scores(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] <= 1:
        return np.ones(vectors.shape[0], dtype=np.float32)
    delta = np.diff(vectors, axis=0, prepend=vectors[:1])
    scores = np.linalg.norm(delta, axis=1)
    scores[0] = max(scores[0], float(np.percentile(scores, 50)))
    if np.allclose(scores.sum(), 0.0):
        return np.ones_like(scores, dtype=np.float32)
    return scores.astype(np.float32)


def map_temporal_to_source(row: dict[str, Any], temporal_len: int) -> tuple[np.ndarray, np.ndarray]:
    sampled = row.get("sampled_frame_indices") or list(range(int(row.get("num_frames", temporal_len) or temporal_len)))
    sampled = np.asarray(sampled, dtype=np.int64)
    if len(sampled) == 0:
        sampled = np.arange(temporal_len, dtype=np.int64)
    positions = np.linspace(0, len(sampled) - 1, temporal_len).round().astype(np.int64)
    return positions, sampled[positions]


def select_positions(scores: np.ndarray, budget: int, keep_endpoints: bool) -> list[int]:
    budget = max(1, int(budget))
    temporal_len = len(scores)
    if budget >= temporal_len:
        return list(range(temporal_len))

    chosen: set[int] = set()
    if keep_endpoints and temporal_len > 1:
        chosen.update([0, temporal_len - 1])
    ranked = np.argsort(-scores)
    for idx in ranked.tolist():
        chosen.add(int(idx))
        if len(chosen) >= budget:
            break
    if len(chosen) < budget:
        for idx in np.linspace(0, temporal_len - 1, budget).round().astype(np.int64).tolist():
            chosen.add(int(idx))
            if len(chosen) >= budget:
                break
    return sorted(chosen)


def interpolate_to_budget(source_indices: np.ndarray, budget: int) -> list[int]:
    if len(source_indices) == 0:
        return []
    if budget <= len(source_indices):
        return source_indices[:budget].astype(int).tolist()
    uniform_positions = np.linspace(0, len(source_indices) - 1, budget).round().astype(np.int64)
    return source_indices[uniform_positions].astype(int).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--feature-name", default="wan_vae_grid_sequence")
    parser.add_argument("--modes", default="")
    parser.add_argument("--budgets", default="8,16,32")
    parser.add_argument("--keep-endpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.metadata_jsonl))
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_mode.setdefault(str(row.get("lowfps_mode", "none")), []).append(row)
    budgets = [int(item.strip()) for item in args.budgets.split(",") if item.strip()]
    requested_modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    output_rows: list[dict[str, Any]] = []

    with h5py.File(args.features_h5, "r") as h5:
        modes = requested_modes or list(h5.keys())
        for mode in modes:
            if mode not in h5 or mode not in rows_by_mode or args.feature_name not in h5[mode]:
                continue
            data = h5[mode][args.feature_name]
            for idx, row in enumerate(rows_by_mode[mode]):
                vectors = temporal_vectors(data[idx], row, args.feature_name)
                scores = motion_scores(vectors)
                _, source_by_temporal = map_temporal_to_source(row, len(scores))
                selected: dict[str, Any] = {}
                for budget in budgets:
                    temporal_positions = select_positions(scores, budget, args.keep_endpoints)
                    source_indices = source_by_temporal[temporal_positions]
                    selected[str(budget)] = {
                        "temporal_positions": [int(x) for x in temporal_positions],
                        "source_frame_indices": interpolate_to_budget(source_indices, budget),
                    }
                out_row = {
                    "video_id": row.get("video_id", row.get("id", idx)),
                    "lowfps_mode": mode,
                    "feature_name": args.feature_name,
                    "scores": [float(x) for x in scores.tolist()],
                    "selected": selected,
                }
                output_rows.append(out_row)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(args.output_jsonl), output_rows)
    print(json.dumps({"rows": len(output_rows), "output_jsonl": args.output_jsonl}, indent=2))


if __name__ == "__main__":
    main()
