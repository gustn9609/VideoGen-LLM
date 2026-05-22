#!/usr/bin/env python3
"""Build negative-control REPA targets for VideoChat2-HD adapter experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import read_jsonl, write_jsonl  # noqa: E402
from motionbench_repa_common import aligned_rows_by_mode, copy_h5_attrs, safe_name, select_h5_rows  # noqa: E402
from motionbench_wan_motion_repa_targets import normalize_rows, targets_for_video, transform_tokens, structured_tokens  # noqa: E402


def transformed_value(value: np.ndarray, transform: str, seed: int) -> np.ndarray:
    tokens = structured_tokens(value)
    if transform == "first_frame":
        out = transform_tokens(tokens, "first", seed)
    elif transform == "time_average":
        out = transform_tokens(tokens, "timeavg", seed)
    elif transform == "shuffle":
        out = transform_tokens(tokens, "shuffle", seed)
    elif transform == "reverse":
        out = transform_tokens(tokens, "reverse", seed)
    else:
        out = tokens
    return out.astype(np.float32)


def pad_stack(vectors: list[np.ndarray]) -> np.ndarray:
    width = max((int(vec.size) for vec in vectors), default=0)
    out = np.zeros((len(vectors), width), dtype=np.float32)
    for idx, vec in enumerate(vectors):
        out[idx, : vec.size] = vec.astype(np.float32)
    return out


def load_reference_shape(path: str, mode: str, dataset: str) -> tuple[int, int]:
    with h5py.File(path, "r") as h5:
        data = h5[mode][dataset]
        return int(data.shape[0]), int(data.shape[1])


def make_random(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, dim)).astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return x.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, choices=["random", "first_frame", "time_average", "pixel", "flow"])
    parser.add_argument("--source-features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-metadata-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--mode", default="high_motion+camera_comp")
    parser.add_argument("--source-feature", default="")
    parser.add_argument("--output-prefix", default="wmrepa")
    parser.add_argument("--output-feature-base", default="wan_vae_grid_1x1")
    parser.add_argument("--target", default="equivariance")
    parser.add_argument("--reference-target-h5", default="results/motionbench_wan_motion_repa_20260511_pool1f8/wmrepa_targets.h5")
    parser.add_argument("--reference-target-feature", default="wmrepa_equivariance_wan_vae_grid_1x1")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_by = aligned_rows_by_mode(args.metadata_jsonl)
    mode_rows = rows_by.get(args.mode, [])
    if not mode_rows:
        raise ValueError(f"No rows for mode {args.mode} in {args.metadata_jsonl}")
    out_name = f"{args.output_prefix}_{args.target}_{safe_name(args.output_feature_base)}"
    output_rows: list[dict[str, Any]] = [dict(row) for row in mode_rows]
    report: dict[str, Any] = {"config": vars(args), "output_feature": out_name}

    if args.control == "random":
        n, dim = load_reference_shape(args.reference_target_h5, args.mode, args.reference_target_feature)
        if n != len(mode_rows):
            raise ValueError(f"Reference rows {n} != metadata rows {len(mode_rows)}")
        arr = make_random(n, dim, args.seed)
        source_feature = "random_normal"
    else:
        if args.control == "pixel":
            source_feature = args.source_feature or "pixel_grid_sequence"
        elif args.control == "flow":
            source_feature = args.source_feature or "flow_grid_sequence"
        else:
            source_feature = args.source_feature or "wan_vae_grid_1x1"
        with h5py.File(args.source_features_h5, "r") as src:
            if args.mode not in src or source_feature not in src[args.mode]:
                raise KeyError(f"{args.mode}/{source_feature} not found in {args.source_features_h5}")
            data = select_h5_rows(src[args.mode][source_feature][:], mode_rows)
        vecs = []
        for idx in range(data.shape[0]):
            if args.control in {"first_frame", "time_average"}:
                tokens = transformed_value(data[idx], args.control, args.seed + idx)
                item = targets_for_video(tokens, args.seed + idx)
            else:
                item = targets_for_video(data[idx], args.seed + idx)
            vecs.append(item[args.target])
        arr = normalize_rows(pad_stack(vecs))

    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as h5:
        h5.attrs["feature_extractor_version"] = "videochat2_repa_negative_targets_v1"
        h5.attrs["control"] = args.control
        group = h5.create_group(args.mode)
        group.create_dataset(out_name, data=arr.astype(np.float32), compression="gzip")
        group.attrs["feature_shapes"] = json.dumps({out_name: [int(arr.shape[1])]})
    for row in output_rows:
        row["feature_extractor_version"] = "videochat2_repa_negative_targets_v1"
        row["negative_control"] = args.control
        row["negative_control_source_feature"] = source_feature
        row["feature_shapes"] = {out_name: [int(arr.shape[1])]}
    write_jsonl(Path(args.output_metadata_jsonl), output_rows)
    report.update({"rows": int(arr.shape[0]), "dim": int(arr.shape[1]), "source_feature": source_feature})
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"control": args.control, "rows": int(arr.shape[0]), "dim": int(arr.shape[1]), "output_h5": args.output_h5}, indent=2))


if __name__ == "__main__":
    main()
