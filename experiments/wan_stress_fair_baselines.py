#!/usr/bin/env python3
"""Hard synthetic stress tasks with same-token fair baselines."""

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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_feature_sanity import MODEL_ID, TaskData, background, draw_disk, encode_wan_vae, load_vae, to_model_range  # noqa: E402
from wan_next_experiments import encode_dit_features, vae_pooling_features  # noqa: E402


@dataclass
class ProbeResult:
    accuracy: float
    balanced_accuracy: float
    feature_dim: int
    confusion_matrix: list


def draw_square(frame: np.ndarray, cx: float, cy: float, size: float, color: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    x0 = max(0, int(round(cx - size / 2)))
    x1 = min(w, int(round(cx + size / 2)))
    y0 = max(0, int(round(cy - size / 2)))
    y1 = min(h, int(round(cy + size / 2)))
    out = frame.copy()
    out[y0:y1, x0:x1] = color
    return out


def same_first_last_path_video(rng, upper_path: bool, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    color = np.array([0.95, 0.25, 0.10], dtype=np.float32)
    radius = float(rng.uniform(5.5, 8.0))
    x0, x1 = width * 0.22, width * 0.78
    y_mid = rng.uniform(height * 0.43, height * 0.57)
    amp = rng.uniform(height * 0.18, height * 0.26)
    sign = -1.0 if upper_path else 1.0
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        x = x0 + (x1 - x0) * t
        y = y_mid + sign * amp * np.sin(np.pi * t)
        video[i] = draw_disk(video[i], x, y, radius, color)
    return to_model_range(video)


def camera_vs_object_video(rng, object_moves_right: bool, frames: int, height: int, width: int) -> np.ndarray:
    canvas_w = width * 2
    base = background(rng, frames, height, canvas_w)
    color = np.array([0.15, 0.78, 0.95], dtype=np.float32)
    radius = float(rng.uniform(5.5, 8.0))
    y = rng.uniform(height * 0.38, height * 0.62)
    obj_start = canvas_w * 0.38
    obj_delta = width * (0.22 if object_moves_right else -0.22)
    cam_delta = width * 0.22
    out = []
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        canvas = base[i]
        obj_x = obj_start + obj_delta * t
        canvas = draw_disk(canvas, obj_x, y, radius, color)
        for gx in np.linspace(0, canvas_w, 9):
            canvas = draw_square(canvas, gx, height * 0.85, 3, np.array([0.7, 0.7, 0.7], dtype=np.float32))
        cam_x = int(round(width * 0.25 + cam_delta * t))
        out.append(canvas[:, cam_x : cam_x + width])
    return to_model_range(np.stack(out))


def crossing_identity_video(rng, red_ends_right: bool, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.94, 0.12, 0.08], dtype=np.float32)
    blue = np.array([0.10, 0.25, 0.95], dtype=np.float32)
    radius = float(rng.uniform(5.5, 7.5))
    y = rng.uniform(height * 0.42, height * 0.58)
    left, right = width * 0.22, width * 0.78
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        red_x = left + (right - left) * t if red_ends_right else right + (left - right) * t
        blue_x = right + (left - right) * t if red_ends_right else left + (right - left) * t
        frame = draw_disk(video[i], red_x, y - radius * 0.2, radius, red)
        if 0.44 <= t <= 0.56:
            frame = draw_square(frame, width * 0.5, y, radius * 4.0, np.array([0.20, 0.20, 0.20], dtype=np.float32))
        frame = draw_disk(frame, blue_x, y + radius * 0.2, radius, blue)
        video[i] = frame
    return to_model_range(video)


def target_count_distractor_video(rng, count: int, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.94, 0.10, 0.08], dtype=np.float32)
    blue = np.array([0.10, 0.25, 0.95], dtype=np.float32)
    radius = float(rng.uniform(5.0, 7.0))
    red_x = rng.uniform(width * 0.25, width * 0.42)
    blue_x = rng.uniform(width * 0.58, width * 0.75)
    base_y = height * 0.55
    amp = rng.uniform(height * 0.12, height * 0.18)
    distractor_count = int(rng.choice([1, 2, 3, 4]))
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        red_tap = np.sin(np.pi * ((count * t) % 1.0)) ** 4
        blue_tap = np.sin(np.pi * ((distractor_count * t + 0.37) % 1.0)) ** 4
        frame = draw_disk(video[i], red_x, base_y + amp * red_tap, radius, red)
        frame = draw_disk(frame, blue_x, base_y + amp * blue_tap, radius, blue)
        video[i] = frame
    return to_model_range(video)


def causality_push_video(rng, causal: bool, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.94, 0.14, 0.08], dtype=np.float32)
    green = np.array([0.16, 0.80, 0.22], dtype=np.float32)
    radius = float(rng.uniform(5.5, 8.0))
    y = rng.uniform(height * 0.42, height * 0.58)
    red_start = width * 0.20
    green_start = width * 0.53
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        red_x = red_start + width * 0.28 * min(t / 0.55, 1.0)
        if causal:
            green_x = green_start + width * 0.22 * max((t - 0.52) / 0.48, 0.0)
        else:
            green_x = green_start + width * 0.22 * min(t / 0.48, 1.0)
        frame = draw_disk(video[i], red_x, y, radius, red)
        frame = draw_disk(frame, green_x, y, radius, green)
        video[i] = frame
    return to_model_range(video)


def make_binary_task(rng, labels, fn, train_base, test_base, frames, height, width) -> TaskData:
    def build(n):
        xs, ys = [], []
        for _ in range(n):
            xs.append(fn(rng, True, frames, height, width))
            ys.append(0)
            xs.append(fn(rng, False, frames, height, width))
            ys.append(1)
        order = rng.permutation(len(xs))
        return np.stack(xs)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def make_count_task(rng, train_base, test_base, frames, height, width) -> TaskData:
    labels = ["red_1", "red_2", "red_3", "red_4"]

    def build(n):
        xs, ys = [], []
        for label, count in enumerate([1, 2, 3, 4]):
            for _ in range(n):
                xs.append(target_count_distractor_video(rng, count, frames, height, width))
                ys.append(label)
        order = rng.permutation(len(xs))
        return np.stack(xs)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def build_tasks(args, seed_offset) -> dict[str, TaskData]:
    rng = np.random.default_rng(args.seed + seed_offset)
    return {
        "same_first_last_path": make_binary_task(
            rng, ["upper_path", "lower_path"], same_first_last_path_video, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
        "camera_vs_object_motion": make_binary_task(
            rng, ["object_right", "object_left"], camera_vs_object_video, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
        "crossing_identity": make_binary_task(
            rng, ["red_ends_right", "red_ends_left"], crossing_identity_video, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
        "target_count_distractor": make_count_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
        "contact_causality_push": make_binary_task(
            rng, ["causal_push", "independent_motion"], causality_push_video, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
    }


def average_pool_video(videos: np.ndarray, out_hw: int) -> np.ndarray:
    n, frames, height, width, channels = videos.shape
    ph = height // out_hw
    pw = width // out_hw
    cropped = videos[:, :, : out_hw * ph, : out_hw * pw, :]
    return cropped.reshape(n, frames, out_hw, ph, out_hw, pw, channels).mean(axis=(3, 5))


def compress_tokens(tokens: np.ndarray, token_budget: int) -> np.ndarray:
    n, seq, channels = tokens.shape
    if token_budget >= seq:
        pad = np.repeat(tokens[:, -1:], token_budget - seq, axis=1) if token_budget > seq else np.empty((n, 0, channels), dtype=tokens.dtype)
        return np.concatenate([tokens, pad], axis=1).reshape(n, -1)
    edges = np.linspace(0, seq, token_budget + 1).round().astype(int)
    pooled = []
    for i in range(token_budget):
        start, end = edges[i], max(edges[i + 1], edges[i] + 1)
        pooled.append(tokens[:, start:end].mean(axis=1))
    return np.stack(pooled, axis=1).reshape(n, -1)


def pixel_tokens(videos: np.ndarray, out_hw: int) -> np.ndarray:
    grid = average_pool_video(videos, out_hw)
    return grid.reshape(grid.shape[0], -1, grid.shape[-1]).astype(np.float32)


def grid_tokens(grid: np.ndarray) -> np.ndarray:
    return grid.reshape(grid.shape[0], -1, grid.shape[-1]).astype(np.float32)


def spatial_mean_tokens(grid: np.ndarray) -> np.ndarray:
    return grid.mean(axis=(2, 3)).reshape(grid.shape[0], grid.shape[1], grid.shape[-1]).astype(np.float32)


def temporal_mean_tokens(grid: np.ndarray) -> np.ndarray:
    return grid.mean(axis=1).reshape(grid.shape[0], -1, grid.shape[-1]).astype(np.float32)


def reversed_time_tokens(grid: np.ndarray) -> np.ndarray:
    return grid[:, ::-1].reshape(grid.shape[0], -1, grid.shape[-1]).astype(np.float32)


def shuffled_time_tokens(grid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(grid.shape[1])
    return grid[:, perm].reshape(grid.shape[0], -1, grid.shape[-1]).astype(np.float32)


def flow_tokens(videos: np.ndarray, out_hw: int) -> np.ndarray:
    grid = average_pool_video(videos, out_hw)
    delta = np.diff(grid, axis=1, prepend=grid[:, :1])
    return delta.reshape(delta.shape[0], -1, delta.shape[-1]).astype(np.float32)


def vae_grid_tokens(latents: np.ndarray) -> np.ndarray:
    grid = latents.transpose(0, 2, 3, 4, 1)
    return grid.reshape(grid.shape[0], -1, grid.shape[-1]).astype(np.float32)


def extract_features(task: TaskData, vae, transformer, scheduler, args, device, dtype, seed) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}

    pixel_grid_train = average_pool_video(task.x_train, args.pixel_grid_hw)
    pixel_grid_test = average_pool_video(task.x_test, args.pixel_grid_hw)
    flow_grid_train = np.diff(pixel_grid_train, axis=1, prepend=pixel_grid_train[:, :1])
    flow_grid_test = np.diff(pixel_grid_test, axis=1, prepend=pixel_grid_test[:, :1])
    token_sources_train = {
        "pixel_grid_same_token": grid_tokens(pixel_grid_train),
        "flow_grid_same_token": grid_tokens(flow_grid_train),
    }
    token_sources_test = {
        "pixel_grid_same_token": grid_tokens(pixel_grid_test),
        "flow_grid_same_token": grid_tokens(flow_grid_test),
    }
    if not args.no_ablations:
        token_sources_train.update(
            {
                "pixel_grid_spatial_mean_same_token": spatial_mean_tokens(pixel_grid_train),
                "pixel_grid_temporal_mean_same_token": temporal_mean_tokens(pixel_grid_train),
                "pixel_grid_reversed_time_same_token": reversed_time_tokens(pixel_grid_train),
                "pixel_grid_shuffled_time_same_token": shuffled_time_tokens(pixel_grid_train, seed + 1),
                "flow_grid_spatial_mean_same_token": spatial_mean_tokens(flow_grid_train),
                "flow_grid_temporal_mean_same_token": temporal_mean_tokens(flow_grid_train),
            }
        )
        token_sources_test.update(
            {
                "pixel_grid_spatial_mean_same_token": spatial_mean_tokens(pixel_grid_test),
                "pixel_grid_temporal_mean_same_token": temporal_mean_tokens(pixel_grid_test),
                "pixel_grid_reversed_time_same_token": reversed_time_tokens(pixel_grid_test),
                "pixel_grid_shuffled_time_same_token": shuffled_time_tokens(pixel_grid_test, seed + 1),
                "flow_grid_spatial_mean_same_token": spatial_mean_tokens(flow_grid_test),
                "flow_grid_temporal_mean_same_token": temporal_mean_tokens(flow_grid_test),
            }
        )

    z_train = encode_wan_vae(vae, task.x_train, args.batch_size, device, dtype)
    z_test = encode_wan_vae(vae, task.x_test, args.batch_size, device, dtype)
    wan_grid_train = z_train.transpose(0, 2, 3, 4, 1)
    wan_grid_test = z_test.transpose(0, 2, 3, 4, 1)
    token_sources_train["wan_vae_grid_same_token"] = grid_tokens(wan_grid_train)
    token_sources_test["wan_vae_grid_same_token"] = grid_tokens(wan_grid_test)
    if not args.no_ablations:
        token_sources_train.update(
            {
                "wan_vae_grid_spatial_mean_same_token": spatial_mean_tokens(wan_grid_train),
                "wan_vae_grid_temporal_mean_same_token": temporal_mean_tokens(wan_grid_train),
                "wan_vae_grid_reversed_time_same_token": reversed_time_tokens(wan_grid_train),
                "wan_vae_grid_shuffled_time_same_token": shuffled_time_tokens(wan_grid_train, seed + 2),
            }
        )
        token_sources_test.update(
            {
                "wan_vae_grid_spatial_mean_same_token": spatial_mean_tokens(wan_grid_test),
                "wan_vae_grid_temporal_mean_same_token": temporal_mean_tokens(wan_grid_test),
                "wan_vae_grid_reversed_time_same_token": reversed_time_tokens(wan_grid_test),
                "wan_vae_grid_shuffled_time_same_token": shuffled_time_tokens(wan_grid_test, seed + 2),
            }
        )
    train["wan_vae_global_sequence"] = vae_pooling_features(z_train, "wan_vae_global_sequence")
    test["wan_vae_global_sequence"] = vae_pooling_features(z_test, "wan_vae_global_sequence")

    for token_budget in args.token_budgets:
        for name, value in token_sources_train.items():
            train[f"{name}_{token_budget}"] = compress_tokens(value, token_budget)
        for name, value in token_sources_test.items():
            test[f"{name}_{token_budget}"] = compress_tokens(value, token_budget)

    if transformer is not None and scheduler is not None:
        dit_train = encode_dit_features(
            transformer, scheduler, z_train, [14], [900], ["token_mean"], args.batch_size, device, dtype, seed
        )
        dit_test = encode_dit_features(
            transformer, scheduler, z_test, [14], [900], ["token_mean"], args.batch_size, device, dtype, seed + 1
        )
        train["dit_l14_t900_token_mean"] = dit_train["dit_l14_t900_token_mean"]
        test["dit_l14_t900_token_mean"] = dit_test["dit_l14_t900_token_mean"]
    return train, test


def fit_eval(classifier: str, x_train, y_train, x_test, y_test) -> ProbeResult:
    if classifier == "linear":
        model = RidgeClassifier(alpha=1.0)
    elif classifier == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-4, max_iter=300, random_state=0, early_stopping=True)
    else:
        raise ValueError(classifier)
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    return ProbeResult(
        accuracy=float(accuracy_score(y_test, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        feature_dim=int(x_train.shape[1]),
        confusion_matrix=confusion_matrix(y_test, pred).tolist(),
    )


def run(args, vae, transformer, scheduler, device, dtype) -> dict:
    results = {}
    for seed_idx in range(args.seeds):
        seed_key = f"seed_{args.seed + seed_idx}"
        results[seed_key] = {}
        for task_name, task in build_tasks(args, seed_idx).items():
            train_features, test_features = extract_features(
                task, vae, transformer, scheduler, args, device, dtype, args.seed + seed_idx * 1000
            )
            task_result = {"labels": task.labels, "features": {}}
            for feature_name, x_train in train_features.items():
                task_result["features"][feature_name] = {}
                for classifier in args.classifiers:
                    result = fit_eval(classifier, x_train, task.y_train, test_features[feature_name], task.y_test)
                    task_result["features"][feature_name][classifier] = result.__dict__
            results[seed_key][task_name] = task_result
    return results


def summarize(results: dict) -> dict:
    aggregate: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed_result in results.values():
        for task, task_result in seed_result.items():
            aggregate.setdefault(task, {})
            for feature, feature_result in task_result["features"].items():
                aggregate[task].setdefault(feature, {})
                for classifier, result in feature_result.items():
                    aggregate[task][feature].setdefault(classifier, []).append(result["balanced_accuracy"])
    summary = {}
    for task, features in aggregate.items():
        summary[task] = {}
        for feature, classifiers in features.items():
            summary[task][feature] = {}
            for classifier, values in classifiers.items():
                arr = np.asarray(values, dtype=np.float64)
                summary[task][feature][classifier] = {
                    "balanced_accuracy_mean": float(arr.mean()),
                    "balanced_accuracy_std": float(arr.std(ddof=0)),
                    "n": int(arr.size),
                }
    return summary


def write_markdown(path: Path, payload: dict) -> None:
    lines = ["# Wan stress fair-baseline results", ""]
    lines.append("| task | feature | classifier | balanced_accuracy_mean | std | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task, features in payload["summary"].items():
        rows = []
        for feature, classifiers in features.items():
            for classifier, result in classifiers.items():
                rows.append((feature, classifier, result))
        rows.sort(key=lambda item: item[2]["balanced_accuracy_mean"], reverse=True)
        for feature, classifier, result in rows:
            lines.append(
                f"| {task} | {feature} | {classifier} | {result['balanced_accuracy_mean']:.4f} | "
                f"{result['balanced_accuracy_std']:.4f} | {result['n']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-base", type=int, default=24)
    parser.add_argument("--test-base", type=int, default=12)
    parser.add_argument("--pixel-grid-hw", type=int, default=16)
    parser.add_argument("--token-budgets", default="16,32")
    parser.add_argument("--classifiers", default="linear")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-dit", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--output-json", default="results/wan_stress_fair_baselines_results.json")
    parser.add_argument("--output-md", default="results/wan_stress_fair_baselines_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.token_budgets = [int(x) for x in args.token_budgets.split(",") if x.strip()]
    args.classifiers = [x.strip() for x in args.classifiers.split(",") if x.strip()]
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

    results = run(args, vae, transformer, scheduler, device, dtype)
    payload = {
        "model_id": MODEL_ID,
        "config": vars(args),
        "results": results,
        "summary": summarize(results),
        "seconds": round(time.time() - t0, 2),
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps({"seconds": payload["seconds"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
