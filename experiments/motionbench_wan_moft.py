#!/usr/bin/env python3
"""Build Wan-MOFT content-free motion features from cached MotionBench features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_repa_common import (  # noqa: E402
    aligned_rows_by_mode,
    copy_h5_attrs,
    safe_name,
    select_h5_rows,
    stable_seed,
    vectorize_feature_array,
    write_jsonl,
)


def centered_projection_residual(x: np.ndarray, basis: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - mean
    coeff = centered @ basis.T
    projection = coeff @ basis
    return centered - projection, projection + mean


def fit_content_basis(bank: np.ndarray, requested_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bank = np.asarray(bank, dtype=np.float32)
    mean = bank.mean(axis=0, keepdims=True)
    centered = bank - mean
    max_components = max(1, min(requested_components, centered.shape[0] - 1, centered.shape[1]))
    pca = PCA(n_components=max_components, random_state=seed)
    pca.fit(centered)
    info = {
        "requested_components": int(requested_components),
        "components": int(max_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return pca.components_.astype(np.float32), mean.astype(np.float32), info


def corr_with_signal(x: np.ndarray, signal: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.shape[0] != signal.shape[0] or x.shape[0] < 2:
        return np.zeros((x.shape[1],), dtype=np.float32)
    xs = x - x.mean(axis=0, keepdims=True)
    ss = signal - float(signal.mean())
    denom = np.sqrt(np.maximum((xs * xs).sum(axis=0) * float((ss * ss).sum()), 1e-12))
    return ((xs * ss[:, None]).sum(axis=0) / denom).astype(np.float32)


def temporal_energy_from_data(data: np.ndarray) -> np.ndarray:
    flat = data.reshape(data.shape[0], data.shape[1], -1) if data.ndim >= 3 else data.reshape(data.shape[0], 1, -1)
    if flat.shape[1] <= 1:
        return np.zeros((flat.shape[0],), dtype=np.float32)
    return np.abs(np.diff(flat, axis=1)).mean(axis=(1, 2)).astype(np.float32)


def score_motion_channels(
    residual: np.ndarray,
    normal_vec: np.ndarray,
    control_vecs: list[np.ndarray],
    flow_signal: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    temporal_proxy = np.abs(normal_vec - np.mean(control_vecs, axis=0)).mean(axis=1) if control_vecs else np.abs(normal_vec).mean(axis=1)
    motion_corr = np.abs(corr_with_signal(residual, temporal_proxy))
    if flow_signal is not None:
        flow_corr = np.abs(corr_with_signal(residual, flow_signal))
    else:
        flow_corr = np.zeros_like(motion_corr)
    control_corr = np.zeros_like(motion_corr)
    for ctrl in control_vecs:
        limit = min(residual.shape[1], ctrl.shape[1])
        if limit <= 0:
            continue
        diff_signal = np.abs(normal_vec[:, :limit] - ctrl[:, :limit]).mean(axis=1)
        control_corr[:limit] = np.maximum(control_corr[:limit], np.abs(corr_with_signal(residual[:, :limit], diff_signal)))
    score = motion_corr + flow_corr + 0.25 * control_corr
    stats = {
        "motion_corr_mean": float(np.mean(motion_corr)),
        "flow_corr_mean": float(np.mean(flow_corr)),
        "control_corr_mean": float(np.mean(control_corr)),
        "score_mean": float(np.mean(score)),
        "score_max": float(np.max(score)) if score.size else 0.0,
    }
    return score.astype(np.float32), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-metadata-jsonl", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--report-md", default="")
    parser.add_argument("--feature-names", default="wan_vae_grid_1x1")
    parser.add_argument("--modes", default="high_motion+camera_comp")
    parser.add_argument("--output-prefix", default="wan_moft")
    parser.add_argument("--content-transforms", default="first_frame_only,time_average,shuffled")
    parser.add_argument("--content-components", type=int, default=32)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--keep-residual-full", action="store_true")
    parser.add_argument("--flow-feature", default="flow_grid_sequence")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    transforms = [x.strip() for x in args.content_transforms.split(",") if x.strip()]
    rows_by = aligned_rows_by_mode(args.metadata_jsonl)
    report: dict[str, Any] = {"config": vars(args), "features": []}
    output_rows: list[dict[str, Any]] = []

    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.features_h5, "r") as src, h5py.File(args.output_h5, "w") as dst:
        copy_h5_attrs(src, dst)
        dst.attrs["feature_extractor_version"] = "motionbench_wan_moft_v1"
        for mode in modes:
            if mode not in src:
                continue
            mode_rows = [dict(row) for row in rows_by.get(mode, [])]
            if not mode_rows:
                continue
            group = dst.create_group(mode)
            copy_h5_attrs(src[mode], group)
            mode_feature_shapes: dict[str, list[int]] = {}
            for feature in feature_names:
                if feature not in src[mode]:
                    continue
                data = select_h5_rows(src[mode][feature][:], mode_rows)
                normal = vectorize_feature_array(data, mode_rows, feature, transform="full", seed=args.seed)
                controls = [
                    vectorize_feature_array(data, mode_rows, feature, transform=transform, seed=stable_seed(mode, feature, transform, base=args.seed))
                    for transform in transforms
                ]
                bank = np.concatenate(controls, axis=0) if controls else normal
                basis, mean, pca_info = fit_content_basis(bank, args.content_components, stable_seed(mode, feature, base=args.seed))
                residual, projection = centered_projection_residual(normal, basis, mean)

                flow_signal = None
                if args.flow_feature and args.flow_feature in src[mode]:
                    flow_data = select_h5_rows(src[mode][args.flow_feature][:], mode_rows)
                    flow_signal = temporal_energy_from_data(flow_data)
                scores, score_stats = score_motion_channels(residual, normal, controls, flow_signal)
                keep = np.argsort(scores)[::-1]
                if args.topk > 0:
                    keep = keep[: min(args.topk, residual.shape[1])]
                keep = np.sort(keep)
                filtered = residual[:, keep].astype(np.float32)
                base_name = f"{args.output_prefix}_{safe_name(feature)}"
                group.create_dataset(base_name, data=filtered, compression="gzip")
                mode_feature_shapes[base_name] = [int(x) for x in filtered.shape[1:]]
                if args.keep_residual_full:
                    residual_name = f"{base_name}_residual_full"
                    group.create_dataset(residual_name, data=residual.astype(np.float32), compression="gzip")
                    mode_feature_shapes[residual_name] = [int(x) for x in residual.shape[1:]]
                info = {
                    "mode": mode,
                    "source_feature": feature,
                    "output_feature": base_name,
                    "rows": int(filtered.shape[0]),
                    "source_dim": int(normal.shape[1]),
                    "output_dim": int(filtered.shape[1]),
                    "content_transforms": transforms,
                    "pca": pca_info,
                    "channel_score": score_stats,
                    "kept_indices_preview": [int(x) for x in keep[: min(16, len(keep))]],
                }
                report["features"].append(info)
            for row in mode_rows:
                row["feature_extractor_version"] = "motionbench_wan_moft_v1"
                row["feature_shapes"] = mode_feature_shapes
                output_rows.append(row)
            group.attrs["feature_shapes"] = json.dumps(mode_feature_shapes)

    write_jsonl(Path(args.output_metadata_jsonl), output_rows)
    report_json = Path(args.report_json) if args.report_json else Path(args.output_h5).with_suffix(".report.json")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md = Path(args.report_md) if args.report_md else Path(args.output_h5).with_suffix(".report.md")
    lines = ["# MotionBench Wan-MOFT Features", ""]
    lines.append("| mode | source | output | rows | source dim | output dim | PCA var | score mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for item in report["features"]:
        lines.append(
            f"| {item['mode']} | {item['source_feature']} | {item['output_feature']} | {item['rows']} | "
            f"{item['source_dim']} | {item['output_dim']} | {item['pca']['explained_variance_ratio_sum']:.4f} | "
            f"{item['channel_score']['score_mean']:.4f} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"features": len(report["features"]), "output_h5": args.output_h5, "report_json": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()
