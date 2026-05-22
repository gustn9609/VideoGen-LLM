#!/usr/bin/env python3
"""Wan feature probes for temporal reasoning tasks closer to VideoLLM failures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_feature_sanity import (  # noqa: E402
    MODEL_ID,
    TaskData,
    background,
    draw_disk,
    encode_wan_vae,
    load_vae,
    make_direction_task,
    to_model_range,
)
from wan_next_experiments import (  # noqa: E402
    encode_dit_features,
    pixel_pooling_features,
    vae_pooling_features,
)


@dataclass
class FeatureBundle:
    train: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    labels: list[str]
    y_train: np.ndarray
    y_test: np.ndarray


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_dim": int(x_train.shape[1]),
    }


def before_after_video(
    rng: np.random.Generator,
    red_first: bool,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.92, 0.15, 0.12], dtype=np.float32)
    blue = np.array([0.10, 0.38, 0.95], dtype=np.float32)
    radius = float(rng.uniform(5.5, 8.0))
    x0 = rng.uniform(width * 0.18, width * 0.28)
    x1 = rng.uniform(width * 0.72, width * 0.82)
    y_red = rng.uniform(height * 0.30, height * 0.42)
    y_blue = rng.uniform(height * 0.58, height * 0.70)
    split = frames // 2

    def progress(i: int, first_event: bool) -> float:
        if first_event:
            return float(np.clip(i / split, 0.0, 1.0))
        return float(np.clip((i - split) / (frames - 1 - split), 0.0, 1.0))

    for i in range(frames):
        red_p = progress(i, red_first)
        blue_p = progress(i, not red_first)
        frame = draw_disk(video[i], x0 + (x1 - x0) * red_p, y_red, radius, red)
        frame = draw_disk(frame, x0 + (x1 - x0) * blue_p, y_blue, radius, blue)
        video[i] = frame
    return to_model_range(video)


def before_after_cycle_video(
    rng: np.random.Generator,
    red_first: bool,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.92, 0.15, 0.12], dtype=np.float32)
    blue = np.array([0.10, 0.38, 0.95], dtype=np.float32)
    radius = float(rng.uniform(5.5, 8.0))
    home_x = rng.uniform(width * 0.20, width * 0.30)
    span = rng.uniform(width * 0.42, width * 0.52)
    y_red = rng.uniform(height * 0.30, height * 0.42)
    y_blue = rng.uniform(height * 0.58, height * 0.70)
    split = frames // 2

    def cycle_position(i: int, first_event: bool) -> float:
        if first_event:
            if i > split:
                return home_x
            local = i / split
        else:
            if i < split:
                return home_x
            local = (i - split) / (frames - 1 - split)
        return home_x + span * np.sin(np.pi * local)

    for i in range(frames):
        red_x = cycle_position(i, red_first)
        blue_x = cycle_position(i, not red_first)
        frame = draw_disk(video[i], red_x, y_red, radius, red)
        frame = draw_disk(frame, blue_x, y_blue, radius, blue)
        video[i] = frame
    return to_model_range(video)


def interaction_video(
    rng: np.random.Generator,
    touching: bool,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.94, 0.18, 0.10], dtype=np.float32)
    green = np.array([0.18, 0.80, 0.25], dtype=np.float32)
    radius = float(rng.uniform(5.5, 7.5))
    y_mid = rng.uniform(height * 0.40, height * 0.60)
    gap = 0.0 if touching else rng.uniform(radius * 2.2, radius * 3.0)
    y_red = y_mid - gap * 0.5
    y_green = y_mid + gap * 0.5
    left = rng.uniform(width * 0.16, width * 0.24)
    right = rng.uniform(width * 0.76, width * 0.84)
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        ease = 0.5 - 0.5 * np.cos(np.pi * t)
        x_red = left + (right - left) * ease
        x_green = right + (left - right) * ease
        frame = draw_disk(video[i], x_red, y_red, radius, red)
        frame = draw_disk(frame, x_green, y_green, radius, green)
        if touching and 0.43 <= t <= 0.57:
            flash = np.array([1.0, 0.9, 0.18], dtype=np.float32)
            frame = draw_disk(frame, (x_red + x_green) * 0.5, y_mid, radius * 0.55, flash)
        video[i] = frame
    return to_model_range(video)


def repetition_video(
    rng: np.random.Generator,
    count: int,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    video = background(rng, frames, height, width)
    color = rng.uniform(0.45, 0.95, size=3).astype(np.float32)
    radius = float(rng.uniform(5.5, 8.0))
    x = rng.uniform(width * 0.35, width * 0.65)
    y_base = rng.uniform(height * 0.42, height * 0.52)
    amplitude = rng.uniform(height * 0.16, height * 0.23)
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        phase = (count * t) % 1.0
        tap = np.sin(np.pi * phase) ** 4
        y = y_base + amplitude * tap
        frame = draw_disk(video[i], x, y, radius, color)
        if tap > 0.80:
            frame = draw_disk(frame, x, y_base + amplitude + radius * 1.25, radius * 0.35, np.array([0.95, 0.95, 0.95]))
        video[i] = frame
    return to_model_range(video)


def make_before_after_task(
    rng: np.random.Generator,
    train_base: int,
    test_base: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["red_before_blue", "blue_before_red"]

    def build(n_base: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for _ in range(n_base):
            videos.append(before_after_video(rng, True, frames, height, width))
            ys.append(0)
            videos.append(before_after_video(rng, False, frames, height, width))
            ys.append(1)
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def make_before_after_cycle_task(
    rng: np.random.Generator,
    train_base: int,
    test_base: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["red_before_blue_cycle", "blue_before_red_cycle"]

    def build(n_base: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for _ in range(n_base):
            videos.append(before_after_cycle_video(rng, True, frames, height, width))
            ys.append(0)
            videos.append(before_after_cycle_video(rng, False, frames, height, width))
            ys.append(1)
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def make_interaction_task(
    rng: np.random.Generator,
    train_base: int,
    test_base: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["touch", "near_miss"]

    def build(n_base: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for _ in range(n_base):
            videos.append(interaction_video(rng, True, frames, height, width))
            ys.append(0)
            videos.append(interaction_video(rng, False, frames, height, width))
            ys.append(1)
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def make_repetition_task(
    rng: np.random.Generator,
    train_per_class: int,
    test_per_class: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["one_tap", "two_taps", "three_taps", "four_taps"]

    def build(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for label, count in enumerate([1, 2, 3, 4]):
            for _ in range(n_per_class):
                videos.append(repetition_video(rng, count, frames, height, width))
                ys.append(label)
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_per_class)
    x_test, y_test = build(test_per_class)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def degrade_low_fps(videos: np.ndarray, keyframes: int) -> np.ndarray:
    frames = videos.shape[1]
    key_idx = np.linspace(0, frames - 1, keyframes).round().astype(np.int64)
    target_idx = np.linspace(0, keyframes - 1, frames).round().astype(np.int64)
    return videos[:, key_idx[target_idx]].copy()


def build_temporal_tasks(args: argparse.Namespace, seed_offset: int = 0) -> dict[str, TaskData]:
    rng = np.random.default_rng(args.seed + seed_offset)
    return {
        "before_after": make_before_after_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
        "before_after_cycle": make_before_after_cycle_task(
            rng, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
        "interaction": make_interaction_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
        "repetition4": make_repetition_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
    }


def extract_features(
    videos_train: np.ndarray,
    videos_test: np.ndarray,
    vae,
    transformer,
    scheduler,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    dit_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pixel_kinds = [
        "pixel_rgb_clip_like_avg",
        "pixel_grid_time_avg",
        "pixel_grid_sequence",
        "pixel_grid_delta",
    ]
    vae_kinds = [
        "wan_vae_channel_avg",
        "wan_vae_global_sequence",
        "wan_vae_grid_sequence",
        "wan_vae_global_delta",
    ]
    train_features: dict[str, np.ndarray] = {}
    test_features: dict[str, np.ndarray] = {}

    for kind in pixel_kinds:
        train_features[kind] = pixel_pooling_features(videos_train, kind, args.baseline_hw)
        test_features[kind] = pixel_pooling_features(videos_test, kind, args.baseline_hw)

    z_train = encode_wan_vae(vae, videos_train, args.batch_size, device, dtype)
    z_test = encode_wan_vae(vae, videos_test, args.batch_size, device, dtype)
    for kind in vae_kinds:
        train_features[kind] = vae_pooling_features(z_train, kind)
        test_features[kind] = vae_pooling_features(z_test, kind)

    if transformer is not None and scheduler is not None:
        dit_train = encode_dit_features(
            transformer,
            scheduler,
            z_train,
            [args.dit_layer],
            [args.dit_timestep],
            [args.dit_pooling],
            args.batch_size,
            device,
            dtype,
            dit_seed,
        )
        dit_test = encode_dit_features(
            transformer,
            scheduler,
            z_test,
            [args.dit_layer],
            [args.dit_timestep],
            [args.dit_pooling],
            args.batch_size,
            device,
            dtype,
            dit_seed + 1,
        )
        for key, value in dit_train.items():
            train_features[key] = value
            test_features[key] = dit_test[key]

    return train_features, test_features


def run_temporal_reasoning(args: argparse.Namespace, vae, transformer, scheduler, device, dtype) -> dict:
    results: dict[str, dict] = {}
    for seed_idx in range(args.seeds):
        seed_key = f"seed_{args.seed + seed_idx}"
        results[seed_key] = {}
        for task_name, task in build_temporal_tasks(args, seed_idx).items():
            train_features, test_features = extract_features(
                task.x_train,
                task.x_test,
                vae,
                transformer,
                scheduler,
                args,
                device,
                dtype,
                args.seed + seed_idx * 100 + 5000,
            )
            task_result = {"labels": task.labels, "features": {}}
            for feature, x_train in train_features.items():
                task_result["features"][feature] = train_probe(x_train, task.y_train, test_features[feature], task.y_test)
            results[seed_key][task_name] = task_result
    return results


def run_lowfps_robustness(args: argparse.Namespace, vae, transformer, scheduler, device, dtype) -> dict:
    results: dict[str, dict] = {}
    for seed_idx in range(args.seeds):
        seed_key = f"seed_{args.seed + seed_idx}"
        rng = np.random.default_rng(args.seed + seed_idx + 200)
        tasks = {
            "direction4": make_direction_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
            "before_after_cycle": make_before_after_cycle_task(
                rng, args.train_base, args.test_base, args.frames, args.height, args.width
            ),
        }
        results[seed_key] = {}
        for task_name, task in tasks.items():
            low_test = degrade_low_fps(task.x_test, args.lowfps_keyframes)
            train_features, full_test_features = extract_features(
                task.x_train,
                task.x_test,
                vae,
                transformer,
                scheduler,
                args,
                device,
                dtype,
                args.seed + seed_idx * 100 + 7000,
            )
            _, low_test_features = extract_features(
                task.x_train[:1],
                low_test,
                vae,
                transformer,
                scheduler,
                args,
                device,
                dtype,
                args.seed + seed_idx * 100 + 9000,
            )
            task_result = {"labels": task.labels, "features": {}}
            for feature, x_train in train_features.items():
                full = train_probe(x_train, task.y_train, full_test_features[feature], task.y_test)
                low = train_probe(x_train, task.y_train, low_test_features[feature], task.y_test)
                task_result["features"][feature] = {
                    "full_test_balanced_accuracy": full["balanced_accuracy"],
                    "lowfps_test_balanced_accuracy": low["balanced_accuracy"],
                    "drop": full["balanced_accuracy"] - low["balanced_accuracy"],
                    "feature_dim": full["feature_dim"],
                }
            results[seed_key][task_name] = task_result
    return results


def summarize_probe_results(results: dict) -> dict:
    aggregate: dict[str, dict[str, list[float]]] = {}
    for seed_result in results.values():
        for task_name, task_result in seed_result.items():
            aggregate.setdefault(task_name, {})
            for feature, result in task_result["features"].items():
                aggregate[task_name].setdefault(feature, []).append(result["balanced_accuracy"])
    summary: dict[str, dict] = {}
    for task_name, features in aggregate.items():
        summary[task_name] = {}
        for feature, values in features.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[task_name][feature] = {
                "balanced_accuracy_mean": float(arr.mean()),
                "balanced_accuracy_std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
    return summary


def summarize_lowfps(results: dict) -> dict:
    aggregate: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed_result in results.values():
        for task_name, task_result in seed_result.items():
            aggregate.setdefault(task_name, {})
            for feature, result in task_result["features"].items():
                aggregate[task_name].setdefault(feature, {"full": [], "low": [], "drop": []})
                aggregate[task_name][feature]["full"].append(result["full_test_balanced_accuracy"])
                aggregate[task_name][feature]["low"].append(result["lowfps_test_balanced_accuracy"])
                aggregate[task_name][feature]["drop"].append(result["drop"])
    summary: dict[str, dict] = {}
    for task_name, features in aggregate.items():
        summary[task_name] = {}
        for feature, values in features.items():
            summary[task_name][feature] = {
                "full_mean": float(np.mean(values["full"])),
                "lowfps_mean": float(np.mean(values["low"])),
                "drop_mean": float(np.mean(values["drop"])),
                "n": int(len(values["full"])),
            }
    return summary


def write_markdown(path: Path, payload: dict) -> None:
    lines = ["# Wan temporal reasoning experiment results", ""]
    lines.extend(["## Temporal reasoning probes", ""])
    lines.append("| task | feature | balanced_accuracy_mean | std | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for task_name, features in payload["temporal_reasoning_summary"].items():
        rows = sorted(features.items(), key=lambda kv: kv[1]["balanced_accuracy_mean"], reverse=True)
        for feature, result in rows:
            lines.append(
                f"| {task_name} | {feature} | {result['balanced_accuracy_mean']:.4f} | "
                f"{result['balanced_accuracy_std']:.4f} | {result['n']} |"
            )
    lines.extend(["", "## Low-frame-rate robustness", ""])
    lines.append("| task | feature | full | lowfps | drop | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task_name, features in payload["lowfps_summary"].items():
        rows = sorted(features.items(), key=lambda kv: kv[1]["lowfps_mean"], reverse=True)
        for feature, result in rows:
            lines.append(
                f"| {task_name} | {feature} | {result['full_mean']:.4f} | {result['lowfps_mean']:.4f} | "
                f"{result['drop_mean']:.4f} | {result['n']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-base", type=int, default=32)
    parser.add_argument("--test-base", type=int, default=16)
    parser.add_argument("--baseline-hw", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-dit", action="store_true")
    parser.add_argument("--dit-layer", type=int, default=14)
    parser.add_argument("--dit-timestep", type=int, default=900)
    parser.add_argument("--dit-pooling", default="token_mean")
    parser.add_argument("--lowfps-keyframes", type=int, default=5)
    parser.add_argument("--output-json", default="results/wan_temporal_reasoning_results.json")
    parser.add_argument("--output-md", default="results/wan_temporal_reasoning_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = load_vae(device, dtype)

    transformer = None
    scheduler = None
    if not args.skip_dit:
        from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

        transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
        transformer.to(device)
        transformer.eval()
        scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    temporal = run_temporal_reasoning(args, vae, transformer, scheduler, device, dtype)
    lowfps = run_lowfps_robustness(args, vae, transformer, scheduler, device, dtype)
    payload = {
        "model_id": MODEL_ID,
        "config": vars(args),
        "device": str(device),
        "temporal_reasoning": temporal,
        "temporal_reasoning_summary": summarize_probe_results(temporal),
        "lowfps_robustness": lowfps,
        "lowfps_summary": summarize_lowfps(lowfps),
        "seconds": round(time.time() - t0, 2),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"seconds": payload["seconds"], "temporal_reasoning_summary": payload["temporal_reasoning_summary"], "lowfps_summary": payload["lowfps_summary"]}, indent=2))


if __name__ == "__main__":
    main()
