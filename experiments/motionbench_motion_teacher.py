#!/usr/bin/env python3
"""Create lightweight motion-teacher targets for Wan-MoREPA experiments."""

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
from extract_wan_features import (  # noqa: E402
    estimate_integer_shift,
    grayscale,
    load_video,
    pixel_grid_sequence,
)
from motionbench_common import read_jsonl, write_jsonl  # noqa: E402


def diff_sequence(video_01: np.ndarray) -> np.ndarray:
    gray = grayscale(video_01)
    if gray.shape[0] <= 1:
        return np.zeros((gray.shape[0], *gray.shape[1:]), dtype=np.float32)
    diff = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    return diff.astype(np.float32)


def segment_bounds(length: int, segments: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, max(1, length), max(1, segments) + 1).round().astype(int)
    out = []
    for i in range(len(boundaries) - 1):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        if end <= start:
            end = min(length, start + 1)
        out.append((start, end))
    return out


def periodicity_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 4 or float(values.std()) < 1e-8:
        return 0.0
    centered = values - float(values.mean())
    corr = np.correlate(centered, centered, mode="full")[values.size - 1 :]
    corr = corr / max(float(corr[0]), 1e-8)
    if corr.size <= 2:
        return 0.0
    return float(np.max(corr[2:]))


def direction_histogram(shifts: np.ndarray, bins: int = 8) -> np.ndarray:
    if shifts.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    dy = shifts[:, 0].astype(np.float32)
    dx = shifts[:, 1].astype(np.float32)
    mag = np.sqrt(dx * dx + dy * dy)
    if float(mag.sum()) <= 1e-8:
        return np.zeros((bins,), dtype=np.float32)
    angle = (np.arctan2(dy, dx) + 2.0 * math.pi) % (2.0 * math.pi)
    idx = np.floor(angle / (2.0 * math.pi) * bins).astype(int).clip(0, bins - 1)
    hist = np.zeros((bins,), dtype=np.float32)
    for i, weight in zip(idx, mag):
        hist[int(i)] += float(weight)
    hist /= max(float(hist.sum()), 1e-8)
    return hist


def grid_motion_descriptor(video_01: np.ndarray, grid_hw: int) -> np.ndarray:
    grid = pixel_grid_sequence(video_01[None], grid_hw)[0]
    delta = np.abs(np.diff(grid, axis=0, prepend=grid[:1])).mean(axis=-1)
    return delta.astype(np.float32)


def frame_shifts(video_01: np.ndarray, max_shift: int) -> np.ndarray:
    gray = grayscale(video_01)
    shifts = []
    for i in range(1, gray.shape[0]):
        dy, dx = estimate_integer_shift(gray[i - 1], gray[i], max_shift=max_shift)
        shifts.append([dy, dx])
    if not shifts:
        shifts = [[0, 0]]
    return np.asarray(shifts, dtype=np.float32)


def summarize_motion(video_01: np.ndarray, grid_hw: int, max_shift: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = diff_sequence(video_01)
    energy = diff.mean(axis=(1, 2)).astype(np.float32)
    grid = grid_motion_descriptor(video_01, grid_hw)
    shifts = frame_shifts(video_01, max_shift=max_shift)
    shift_mag = np.sqrt((shifts * shifts).sum(axis=1))
    high_idx = int(np.argmax(energy)) if energy.size else 0
    summary = np.concatenate(
        [
            np.asarray(
                [
                    float(energy.mean()) if energy.size else 0.0,
                    float(energy.std()) if energy.size else 0.0,
                    float(energy.max()) if energy.size else 0.0,
                    float(high_idx / max(1, energy.size - 1)),
                    periodicity_score(energy),
                    float(shift_mag.mean()) if shift_mag.size else 0.0,
                    float(shift_mag.std()) if shift_mag.size else 0.0,
                    float(shifts[:, 0].mean()) if shifts.size else 0.0,
                    float(shifts[:, 1].mean()) if shifts.size else 0.0,
                    float(shifts[:, 0].std()) if shifts.size else 0.0,
                    float(shifts[:, 1].std()) if shifts.size else 0.0,
                    float(grid.mean()) if grid.size else 0.0,
                    float(grid.std()) if grid.size else 0.0,
                    float(grid.max()) if grid.size else 0.0,
                ],
                dtype=np.float32,
            ),
            direction_histogram(shifts),
        ],
        axis=0,
    )
    return summary.astype(np.float32), energy.astype(np.float32), grid.astype(np.float32)


def segment_features(video_01: np.ndarray, segments: int, grid_hw: int, max_shift: int) -> np.ndarray:
    feats = []
    for start, end in segment_bounds(video_01.shape[0], segments):
        seg = video_01[start:end]
        summary, _, _ = summarize_motion(seg, grid_hw=grid_hw, max_shift=max_shift)
        feats.append(summary)
    return np.stack(feats, axis=0).astype(np.float32)


def cached_summary_from_flow(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 1:
        flow = flow.reshape(1, 1, 1, -1)
    if flow.ndim == 2:
        flow = flow.reshape(flow.shape[0], 1, 1, flow.shape[1])
    if flow.ndim == 3:
        flow = flow[..., None]
    energy_grid = np.abs(flow).mean(axis=-1).astype(np.float32)
    energy = energy_grid.reshape(energy_grid.shape[0], -1).mean(axis=1).astype(np.float32)
    channel_mean = flow.reshape(flow.shape[0], -1, flow.shape[-1]).mean(axis=1)
    if channel_mean.shape[1] >= 2:
        shifts = channel_mean[:, :2]
    elif channel_mean.shape[1] == 1:
        shifts = np.concatenate([channel_mean, np.zeros_like(channel_mean)], axis=1)
    else:
        shifts = np.zeros((flow.shape[0], 2), dtype=np.float32)
    shift_mag = np.sqrt((shifts * shifts).sum(axis=1))
    high_idx = int(np.argmax(energy)) if energy.size else 0
    summary = np.concatenate(
        [
            np.asarray(
                [
                    float(energy.mean()) if energy.size else 0.0,
                    float(energy.std()) if energy.size else 0.0,
                    float(energy.max()) if energy.size else 0.0,
                    float(high_idx / max(1, energy.size - 1)),
                    periodicity_score(energy),
                    float(shift_mag.mean()) if shift_mag.size else 0.0,
                    float(shift_mag.std()) if shift_mag.size else 0.0,
                    float(shifts[:, 0].mean()) if shifts.size else 0.0,
                    float(shifts[:, 1].mean()) if shifts.size else 0.0,
                    float(shifts[:, 0].std()) if shifts.size else 0.0,
                    float(shifts[:, 1].std()) if shifts.size else 0.0,
                    float(energy_grid.mean()) if energy_grid.size else 0.0,
                    float(energy_grid.std()) if energy_grid.size else 0.0,
                    float(energy_grid.max()) if energy_grid.size else 0.0,
                ],
                dtype=np.float32,
            ),
            direction_histogram(shifts),
        ],
        axis=0,
    )
    return summary.astype(np.float32), energy.astype(np.float32), energy_grid.astype(np.float32)


def cached_segment_features(flow: np.ndarray, segments: int) -> np.ndarray:
    feats = []
    for start, end in segment_bounds(flow.shape[0], segments):
        summary, _, _ = cached_summary_from_flow(flow[start:end])
        feats.append(summary)
    return np.stack(feats, axis=0).astype(np.float32)


def run_cached_features(args: argparse.Namespace, rows: list[dict[str, Any]], modes: list[str]) -> None:
    output_rows: list[dict[str, Any]] = []
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cached_features_h5, "r") as src, h5py.File(args.output_h5, "w") as h5:
        h5.attrs["feature_extractor_version"] = "motionbench_motion_teacher_cached_v1"
        h5.attrs["source_features_h5"] = args.cached_features_h5
        h5.attrs["segments"] = int(args.segments)
        for mode in modes:
            if mode not in src or args.flow_feature not in src[mode]:
                continue
            mode_rows = [dict(row) for row in rows if not args.input_mode_filter or str(row.get("lowfps_mode", "")) == args.input_mode_filter]
            flow_all = src[mode][args.flow_feature][:]
            flow_all = flow_all[: len(mode_rows)]
            summaries = []
            energies = []
            grids = []
            segments = []
            for idx, row in enumerate(mode_rows):
                summary, energy, grid = cached_summary_from_flow(flow_all[idx])
                seg = cached_segment_features(flow_all[idx], args.segments)
                summaries.append(summary)
                energies.append(energy)
                grids.append(grid)
                segments.append(seg)
                out_row = dict(row)
                out_row["lowfps_mode"] = mode
                out_row["feature_extractor_version"] = "motionbench_motion_teacher_cached_v1"
                output_rows.append(out_row)
            group = h5.create_group(mode)
            summary_arr = np.stack(summaries, axis=0).astype(np.float32)
            energy_arr = np.stack(energies, axis=0).astype(np.float32)
            grid_arr = np.stack(grids, axis=0).astype(np.float32)
            segment_arr = np.stack(segments, axis=0).astype(np.float32)
            group.create_dataset("motion_teacher_summary", data=summary_arr, compression="gzip")
            group.create_dataset("motion_teacher_energy", data=energy_arr, compression="gzip")
            group.create_dataset("motion_teacher_grid", data=grid_arr, compression="gzip")
            group.create_dataset("motion_teacher_segments", data=segment_arr, compression="gzip")
            feature_shapes = {
                "motion_teacher_summary": [int(x) for x in summary_arr.shape[1:]],
                "motion_teacher_energy": [int(x) for x in energy_arr.shape[1:]],
                "motion_teacher_grid": [int(x) for x in grid_arr.shape[1:]],
                "motion_teacher_segments": [int(x) for x in segment_arr.shape[1:]],
            }
            group.attrs["feature_shapes"] = json.dumps(feature_shapes)
            for row in output_rows[-len(mode_rows) :]:
                row["feature_shapes"] = feature_shapes

    write_jsonl(Path(args.metadata_output), output_rows)
    print(
        json.dumps(
            {
                "videos": len(output_rows),
                "modes": modes,
                "source": args.cached_features_h5,
                "output_h5": args.output_h5,
                "metadata_output": args.metadata_output,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--input-mode-filter", default="")
    parser.add_argument("--cached-features-h5", default="")
    parser.add_argument("--flow-feature", default="flow_grid_sequence")
    parser.add_argument("--lowfps-modes", default="high_motion+camera_comp")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--grid-hw", type=int, default=4)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.video_jsonl))
    if args.input_mode_filter:
        rows = [row for row in rows if str(row.get("lowfps_mode", "")) == args.input_mode_filter]
    modes = [x.strip() for x in args.lowfps_modes.split(",") if x.strip()]
    if args.cached_features_h5:
        run_cached_features(args, rows, modes)
        return
    output_rows: list[dict[str, Any]] = []

    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as h5:
        h5.attrs["feature_extractor_version"] = "motionbench_motion_teacher_v1"
        h5.attrs["num_frames"] = int(args.num_frames)
        h5.attrs["resize"] = json.dumps([int(args.height), int(args.width)])
        h5.attrs["segments"] = int(args.segments)
        h5.attrs["grid_hw"] = int(args.grid_hw)
        for mode in modes:
            summaries = []
            energies = []
            grids = []
            segments = []
            mode_rows = []
            for row_idx, row in enumerate(rows):
                fps_value = row.get("fps", row.get("original_fps", None))
                fps = float(fps_value) if fps_value not in (None, "") else None
                video_m11, indices, timestamps, transform_meta = load_video(
                    Path(row["path"]),
                    args.num_frames,
                    args.height,
                    args.width,
                    mode,
                    args.seed + row_idx,
                    fps,
                    row.get("start"),
                    row.get("end"),
                )
                video_01 = np.clip((video_m11 + 1.0) / 2.0, 0.0, 1.0)
                summary, energy, grid = summarize_motion(video_01, grid_hw=args.grid_hw, max_shift=args.max_shift)
                seg = segment_features(video_01, segments=args.segments, grid_hw=args.grid_hw, max_shift=args.max_shift)
                summaries.append(summary)
                energies.append(energy)
                grids.append(grid)
                segments.append(seg)
                out_row = dict(row)
                original_frame_count = transform_meta.get("original_frame_count")
                duration = row.get("duration")
                if duration is None and fps is not None and fps > 0 and original_frame_count:
                    duration = float(original_frame_count) / fps
                out_row.update(
                    {
                        "lowfps_mode": mode,
                        "sampled_frame_indices": indices.tolist(),
                        "sampled_timestamps": timestamps,
                        "original_fps": fps,
                        "original_frame_count": original_frame_count,
                        "duration": duration,
                        "num_frames": args.num_frames,
                        "resize": [args.height, args.width],
                        "augmentation": transform_meta,
                        "feature_extractor_version": "motionbench_motion_teacher_v1",
                    }
                )
                mode_rows.append(out_row)
            group = h5.create_group(mode)
            summary_arr = np.stack(summaries, axis=0).astype(np.float32)
            energy_arr = np.stack(energies, axis=0).astype(np.float32)
            grid_arr = np.stack(grids, axis=0).astype(np.float32)
            segment_arr = np.stack(segments, axis=0).astype(np.float32)
            group.create_dataset("motion_teacher_summary", data=summary_arr, compression="gzip")
            group.create_dataset("motion_teacher_energy", data=energy_arr, compression="gzip")
            group.create_dataset("motion_teacher_grid", data=grid_arr, compression="gzip")
            group.create_dataset("motion_teacher_segments", data=segment_arr, compression="gzip")
            feature_shapes = {
                "motion_teacher_summary": [int(x) for x in summary_arr.shape[1:]],
                "motion_teacher_energy": [int(x) for x in energy_arr.shape[1:]],
                "motion_teacher_grid": [int(x) for x in grid_arr.shape[1:]],
                "motion_teacher_segments": [int(x) for x in segment_arr.shape[1:]],
            }
            group.attrs["feature_shapes"] = json.dumps(feature_shapes)
            for row in mode_rows:
                row["feature_shapes"] = feature_shapes
                output_rows.append(row)

    write_jsonl(Path(args.metadata_output), output_rows)
    print(
        json.dumps(
            {
                "videos": len(rows),
                "modes": modes,
                "output_h5": args.output_h5,
                "metadata_output": args.metadata_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
