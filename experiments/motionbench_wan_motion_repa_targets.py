#!/usr/bin/env python3
"""Build Wan-Motion-REPA transformed targets from cached Wan features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_repa_common import aligned_rows_by_mode, copy_h5_attrs, safe_name, select_h5_rows, write_jsonl  # noqa: E402


def l2(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def safe_stats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros((4,), dtype=np.float32)
    flat = x.reshape(-1)
    return np.asarray([float(flat.mean()), float(flat.std()), float(flat.min()), float(flat.max())], dtype=np.float32)


def autocorr(values: np.ndarray, lags: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros((lags,), dtype=np.float32)
    if values.size < 3 or float(values.std()) < 1e-8:
        return out
    centered = values - float(values.mean())
    denom = float(np.dot(centered, centered))
    if denom <= 1e-8:
        return out
    for lag in range(1, lags + 1):
        if lag < values.size:
            out[lag - 1] = float(np.dot(centered[:-lag], centered[lag:]) / denom)
    return out


def fft_summary(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 3 or float(values.std()) < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    centered = values - float(values.mean())
    spec = np.abs(np.fft.rfft(centered))[1:]
    if spec.size == 0 or float(spec.sum()) <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    peak = int(np.argmax(spec))
    prob = spec / max(float(spec.sum()), 1e-8)
    entropy = -float((prob * np.log(np.maximum(prob, 1e-8))).sum() / math.log(max(2, spec.size)))
    return np.asarray([peak / max(1, spec.size - 1), float(spec[peak] / max(float(spec.sum()), 1e-8)), entropy], dtype=np.float32)


def structured_tokens(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4:
        # [T,H,W,C] -> global, left-right, top-bottom, and compact spatial cells.
        t, h, w, c = value.shape
        global_pool = value.mean(axis=(1, 2))
        if w >= 2:
            left = value[:, :, : max(1, w // 2), :].mean(axis=(1, 2))
            right = value[:, :, max(1, w // 2) :, :].mean(axis=(1, 2))
            lr = left - right
        else:
            lr = np.zeros((t, c), dtype=np.float32)
        if h >= 2:
            top = value[:, : max(1, h // 2), :, :].mean(axis=(1, 2))
            bottom = value[:, max(1, h // 2) :, :, :].mean(axis=(1, 2))
            tb = top - bottom
        else:
            tb = np.zeros((t, c), dtype=np.float32)
        cells = value.reshape(t, h * w * c)
        return np.concatenate([global_pool, lr, tb, cells], axis=1).astype(np.float32)
    if value.ndim == 3:
        return value.reshape(value.shape[0], -1).astype(np.float32)
    if value.ndim == 2:
        return value.astype(np.float32)
    return value.reshape(1, -1).astype(np.float32)


def transform_tokens(tokens: np.ndarray, transform: str, seed: int) -> np.ndarray:
    if transform == "full":
        return tokens
    if transform == "reverse":
        return tokens[::-1]
    if transform == "shuffle":
        rng = np.random.default_rng(seed)
        order = np.arange(tokens.shape[0])
        rng.shuffle(order)
        return tokens[order]
    if transform == "first":
        return np.repeat(tokens[:1], tokens.shape[0], axis=0)
    if transform == "timeavg":
        return np.repeat(tokens.mean(axis=0, keepdims=True), tokens.shape[0], axis=0)
    raise ValueError(transform)


def relation(tokens: np.ndarray) -> np.ndarray:
    z = l2(tokens.astype(np.float32), axis=1)
    return (z @ z.T).astype(np.float32)


def relation_upper(tokens: np.ndarray) -> np.ndarray:
    rel = relation(tokens)
    if rel.shape[0] <= 1:
        return rel.reshape(-1)
    idx = np.triu_indices(rel.shape[0], k=1)
    return rel[idx].astype(np.float32)


def dynamics_core(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.shape[0] <= 1:
        dh = np.zeros_like(tokens)
    else:
        dh = np.diff(tokens, axis=0)
    if dh.shape[0] <= 1:
        ddh = np.zeros_like(dh)
    else:
        ddh = np.diff(dh, axis=0)
    energy = np.linalg.norm(dh, axis=1) if dh.size else np.zeros((1,), dtype=np.float32)
    argmax = float(np.argmax(energy) / max(1, energy.size - 1)) if energy.size else 0.0
    return np.concatenate(
        [
            tokens.mean(axis=0),
            tokens.std(axis=0),
            dh.mean(axis=0) if dh.size else np.zeros((tokens.shape[1],), dtype=np.float32),
            np.abs(dh).mean(axis=0) if dh.size else np.zeros((tokens.shape[1],), dtype=np.float32),
            ddh.mean(axis=0) if ddh.size else np.zeros((tokens.shape[1],), dtype=np.float32),
            np.abs(ddh).mean(axis=0) if ddh.size else np.zeros((tokens.shape[1],), dtype=np.float32),
            safe_stats(energy),
            np.asarray([argmax], dtype=np.float32),
            autocorr(energy, 4),
            fft_summary(energy),
            relation_upper(tokens),
        ],
        axis=0,
    ).astype(np.float32)


def pad_stack(vectors: list[np.ndarray]) -> np.ndarray:
    width = max((int(v.size) for v in vectors), default=0)
    out = np.zeros((len(vectors), width), dtype=np.float32)
    for i, vec in enumerate(vectors):
        out[i, : vec.size] = vec.astype(np.float32)
    return out


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def targets_for_video(value: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    tokens = structured_tokens(value)
    full = transform_tokens(tokens, "full", seed)
    controls = {
        name: transform_tokens(tokens, name, seed + idx * 997)
        for idx, name in enumerate(["reverse", "shuffle", "first", "timeavg"], start=1)
    }
    raw_pooled = np.concatenate([full.mean(axis=0), full.std(axis=0)], axis=0).astype(np.float32)
    dyn = dynamics_core(full)
    rel = relation(full).reshape(-1).astype(np.float32)
    control_stats = []
    response = []
    for name, ctrl in controls.items():
        ctrl_dyn = dynamics_core(ctrl)
        limit = min(dyn.size, ctrl_dyn.size)
        diff = dyn[:limit] - ctrl_dyn[:limit]
        response.append(diff)
        control_stats.extend(
            [
                float(np.linalg.norm(diff)),
                float(np.linalg.norm(relation(full) - relation(ctrl))),
                float(np.linalg.norm(full.reshape(-1) - ctrl.reshape(-1))),
            ]
        )
    control_arr = np.asarray(control_stats, dtype=np.float32)
    dynamics_relation = np.concatenate([dyn, control_arr], axis=0).astype(np.float32)
    relation_only = np.concatenate([rel, np.asarray([float(np.linalg.norm(relation(full) - relation(ctrl))) for ctrl in controls.values()], dtype=np.float32)], axis=0)
    equivariance = np.concatenate([l2(vec.reshape(1, -1)).reshape(-1) for vec in response], axis=0).astype(np.float32)
    structured_compact = np.concatenate([raw_pooled, control_arr], axis=0).astype(np.float32)
    multi = np.concatenate([raw_pooled, dynamics_relation, relation_only, equivariance], axis=0).astype(np.float32)
    return {
        "raw_pooled": raw_pooled,
        "structured_compact": structured_compact,
        "dynamics_relation": dynamics_relation,
        "relation_only": relation_only,
        "equivariance": equivariance,
        "multi_target": multi,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-metadata-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--feature-names", default="wan_vae_grid_2x2")
    parser.add_argument("--modes", default="high_motion+camera_comp")
    parser.add_argument("--output-prefix", default="wmrepa")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    features = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    rows_by = aligned_rows_by_mode(args.metadata_jsonl)
    output_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"config": vars(args), "features": []}
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.features_h5, "r") as src, h5py.File(args.output_h5, "w") as dst:
        copy_h5_attrs(src, dst)
        dst.attrs["feature_extractor_version"] = "motionbench_wan_motion_repa_targets_v1"
        for mode in modes:
            if mode not in src:
                continue
            mode_rows = [dict(row) for row in rows_by.get(mode, [])]
            if not mode_rows:
                continue
            group = dst.create_group(mode)
            copy_h5_attrs(src[mode], group)
            feature_shapes: dict[str, list[int]] = {}
            for feature in features:
                if feature not in src[mode]:
                    continue
                data = select_h5_rows(src[mode][feature][:], mode_rows)
                buckets: dict[str, list[np.ndarray]] = {
                    "raw_pooled": [],
                    "structured_compact": [],
                    "dynamics_relation": [],
                    "relation_only": [],
                    "equivariance": [],
                    "multi_target": [],
                }
                for idx in range(data.shape[0]):
                    item = targets_for_video(data[idx], args.seed + idx)
                    for key, vec in item.items():
                        buckets[key].append(vec)
                for key, vecs in buckets.items():
                    arr = normalize_rows(pad_stack(vecs))
                    out_name = f"{args.output_prefix}_{key}_{safe_name(feature)}"
                    group.create_dataset(out_name, data=arr, compression="gzip")
                    feature_shapes[out_name] = [int(x) for x in arr.shape[1:]]
                    report["features"].append(
                        {
                            "mode": mode,
                            "source_feature": feature,
                            "output_feature": out_name,
                            "target": key,
                            "rows": int(arr.shape[0]),
                            "dim": int(arr.shape[1]),
                        }
                    )
            group.attrs["feature_shapes"] = json.dumps(feature_shapes)
            for row in mode_rows:
                row["feature_extractor_version"] = "motionbench_wan_motion_repa_targets_v1"
                row["feature_shapes"] = feature_shapes
                output_rows.append(row)

    write_jsonl(Path(args.output_metadata_jsonl), output_rows)
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MotionBench Wan-Motion-REPA Targets", ""]
    lines.append("| mode | source | target | output | rows | dim |")
    lines.append("|---|---|---|---|---:|---:|")
    for item in report["features"]:
        lines.append(
            f"| {item['mode']} | {item['source_feature']} | {item['target']} | {item['output_feature']} | {item['rows']} | {item['dim']} |"
        )
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"features": len(report["features"]), "output_h5": args.output_h5, "report_json": args.report_json}, indent=2))


if __name__ == "__main__":
    main()
