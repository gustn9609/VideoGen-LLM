#!/usr/bin/env python3
"""Sanity checks for whether Wan features carry temporal/motion signal.

The script runs two experiments:

1. Wan-VAE frozen-feature probing on synthetic video tasks.
2. Optional Wan-DiT denoising-loss comparison for normal vs reversed/shuffled
   videos using zero text conditioning.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class TaskData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: list[str]


def draw_disk(frame: np.ndarray, cx: float, cy: float, radius: float, color: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(radius + 1.5 - dist, 0.0, 1.0)[..., None]
    return frame * (1.0 - alpha) + color.reshape(1, 1, 3) * alpha


def background(rng: np.random.Generator, frames: int, height: int, width: int) -> np.ndarray:
    base = rng.uniform(0.05, 0.25, size=(1, 1, 1, 3)).astype(np.float32)
    y_grad = np.linspace(0.0, 0.08, height, dtype=np.float32).reshape(1, height, 1, 1)
    x_grad = np.linspace(0.0, 0.05, width, dtype=np.float32).reshape(1, 1, width, 1)
    noise = rng.normal(0.0, 0.008, size=(frames, height, width, 3)).astype(np.float32)
    return np.clip(base + y_grad + x_grad + noise, 0.0, 1.0)


def to_model_range(video_01: np.ndarray) -> np.ndarray:
    return (video_01 * 2.0 - 1.0).astype(np.float32)


def linear_motion_video(
    rng: np.random.Generator,
    direction: int,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    video = background(rng, frames, height, width)
    color = rng.uniform(0.45, 0.95, size=3).astype(np.float32)
    radius = float(rng.uniform(5.0, 9.0))
    margin = radius + 6.0
    span_x = width - 2 * margin
    span_y = height - 2 * margin
    start = rng.uniform(0.15, 0.35)
    end = rng.uniform(0.65, 0.85)
    fixed_x = rng.uniform(margin + 0.2 * span_x, margin + 0.8 * span_x)
    fixed_y = rng.uniform(margin + 0.2 * span_y, margin + 0.8 * span_y)
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)

    for i, t in enumerate(ts):
        if direction == 0:
            cx, cy = margin + span_x * (start + (end - start) * t), fixed_y
        elif direction == 1:
            cx, cy = margin + span_x * (end + (start - end) * t), fixed_y
        elif direction == 2:
            cx, cy = fixed_x, margin + span_y * (start + (end - start) * t)
        else:
            cx, cy = fixed_x, margin + span_y * (end + (start - end) * t)
        video[i] = draw_disk(video[i], cx, cy, radius, color)

    return to_model_range(video)


def gravity_video(
    rng: np.random.Generator,
    frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    video = background(rng, frames, height, width)
    color = rng.uniform(0.50, 0.95, size=3).astype(np.float32)
    radius = float(rng.uniform(5.5, 8.5))
    x0 = rng.uniform(width * 0.25, width * 0.75)
    drift = rng.uniform(-width * 0.10, width * 0.10)
    y0 = rng.uniform(height * 0.15, height * 0.25)
    y_span = rng.uniform(height * 0.47, height * 0.58)
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)

    for i, t in enumerate(ts):
        cx = x0 + drift * t
        cy = y0 + y_span * (t**2)
        impact = max(0.0, (t - 0.82) / 0.18)
        r = radius * (1.0 + 0.45 * impact)
        frame = draw_disk(video[i], cx, cy, r, color)
        if impact > 0:
            shadow_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
            frame = draw_disk(frame, cx, height * 0.84, radius * 0.45 * impact, shadow_color)
        video[i] = frame

    return to_model_range(video)


def make_direction_task(
    rng: np.random.Generator,
    train_per_class: int,
    test_per_class: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]

    def build(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for label in range(4):
            for _ in range(n_per_class):
                videos.append(linear_motion_video(rng, label, frames, height, width))
                ys.append(label)
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_per_class)
    x_test, y_test = build(test_per_class)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def make_reversal_task(
    rng: np.random.Generator,
    train_base: int,
    test_base: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["forward_gravity", "reversed_gravity"]

    def build(n_base: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for _ in range(n_base):
            normal = gravity_video(rng, frames, height, width)
            videos.extend([normal, normal[::-1].copy()])
            ys.extend([0, 1])
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def make_shuffle_task(
    rng: np.random.Generator,
    train_base: int,
    test_base: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    labels = ["smooth_order", "shuffled_order"]

    def build(n_base: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for _ in range(n_base):
            direction = int(rng.integers(0, 4))
            normal = linear_motion_video(rng, direction, frames, height, width)
            perm = rng.permutation(frames)
            while np.all(perm == np.arange(frames)):
                perm = rng.permutation(frames)
            shuffled = normal[perm].copy()
            videos.extend([normal, shuffled])
            ys.extend([0, 1])
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_base)
    x_test, y_test = build(test_base)
    return TaskData(x_train, y_train, x_test, y_test, labels)


def average_pool_video(videos: np.ndarray, out_hw: int) -> np.ndarray:
    n, f, h, w, c = videos.shape
    ph = h // out_hw
    pw = w // out_hw
    cropped = videos[:, :, : out_hw * ph, : out_hw * pw, :]
    pooled = cropped.reshape(n, f, out_hw, ph, out_hw, pw, c).mean(axis=(3, 5))
    return pooled


def baseline_features(videos: np.ndarray, kind: str, out_hw: int) -> np.ndarray:
    if kind == "frame_rgb_mean":
        return videos.mean(axis=(2, 3)).reshape(videos.shape[0], -1)
    pooled = average_pool_video(videos, out_hw)
    if kind == "pixel_grid":
        return pooled.reshape(videos.shape[0], -1)
    if kind == "pixel_grid_delta":
        return np.diff(pooled, axis=1).reshape(videos.shape[0], -1)
    raise ValueError(f"Unknown baseline feature: {kind}")


def load_vae(device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    return vae


def encode_wan_vae(
    vae,
    videos: np.ndarray,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    means = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    stds = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    latents: list[np.ndarray] = []
    for start in range(0, len(videos), batch_size):
        batch_np = videos[start : start + batch_size]
        batch = torch.from_numpy(batch_np).permute(0, 4, 1, 2, 3).contiguous().to(device=device, dtype=dtype)
        with torch.no_grad():
            posterior = vae.encode(batch).latent_dist
            z = posterior.mode()
            z = (z - means) / stds
        latents.append(z.float().cpu().numpy())
    return np.concatenate(latents, axis=0)


def wan_latent_features(latents: np.ndarray, kind: str) -> np.ndarray:
    if kind == "wan_vae_global":
        return latents.mean(axis=(3, 4)).reshape(latents.shape[0], -1)
    if kind == "wan_vae_grid":
        return latents.reshape(latents.shape[0], -1)
    if kind == "wan_vae_delta_grid":
        delta = np.diff(latents, axis=2)
        return np.concatenate(
            [latents.reshape(latents.shape[0], -1), delta.reshape(delta.shape[0], -1)],
            axis=1,
        )
    raise ValueError(f"Unknown Wan feature: {kind}")


def train_probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_dim": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def run_vae_probes(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = load_vae(device, dtype)

    task_builders: dict[str, Callable[[], TaskData]] = {
        "direction4": lambda: make_direction_task(
            rng, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
        "reversal": lambda: make_reversal_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
        "shuffle": lambda: make_shuffle_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
    }

    baseline_kinds = ["frame_rgb_mean", "pixel_grid", "pixel_grid_delta"]
    wan_kinds = ["wan_vae_global", "wan_vae_grid", "wan_vae_delta_grid"]
    results: dict[str, dict] = {}

    for task_name, builder in task_builders.items():
        t0 = time.time()
        task = builder()
        task_result: dict[str, dict] = {"labels": task.labels, "features": {}}

        for kind in baseline_kinds:
            x_train = baseline_features(task.x_train, kind, args.baseline_hw)
            x_test = baseline_features(task.x_test, kind, args.baseline_hw)
            task_result["features"][kind] = train_probe(x_train, task.y_train, x_test, task.y_test)

        train_latents = encode_wan_vae(vae, task.x_train, args.batch_size, device, dtype)
        test_latents = encode_wan_vae(vae, task.x_test, args.batch_size, device, dtype)
        task_result["latent_shape"] = list(train_latents.shape[1:])

        for kind in wan_kinds:
            x_train = wan_latent_features(train_latents, kind)
            x_test = wan_latent_features(test_latents, kind)
            task_result["features"][kind] = train_probe(x_train, task.y_train, x_test, task.y_test)

        task_result["seconds"] = round(time.time() - t0, 2)
        results[task_name] = task_result

    return results


def pair_videos_for_dit(rng: np.random.Generator, n_pairs: int, frames: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals: list[np.ndarray] = []
    reverseds: list[np.ndarray] = []
    shuffleds: list[np.ndarray] = []
    for i in range(n_pairs):
        if i % 2 == 0:
            normal = gravity_video(rng, frames, height, width)
        else:
            normal = linear_motion_video(rng, int(rng.integers(0, 4)), frames, height, width)
        perm = rng.permutation(frames)
        while np.all(perm == np.arange(frames)):
            perm = rng.permutation(frames)
        normals.append(normal)
        reverseds.append(normal[::-1].copy())
        shuffleds.append(normal[perm].copy())
    return np.stack(normals), np.stack(reverseds), np.stack(shuffleds)


def denoise_losses(
    transformer,
    scheduler,
    latents: torch.Tensor,
    timesteps: list[int],
    seed: int,
    text_dim: int,
) -> dict[str, list[float]]:
    losses: dict[str, list[float]] = {}
    batch = latents.shape[0]
    device = latents.device
    dtype = latents.dtype
    prompt_embeds = torch.zeros(batch, 1, text_dim, device=device, dtype=dtype)
    target_dims = tuple(range(1, latents.ndim))
    generator = torch.Generator(device=device).manual_seed(seed)

    for timestep in timesteps:
        t = torch.full((batch,), int(timestep), device=device, dtype=torch.long)
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
        noisy = scheduler.add_noise(latents, noise, t)
        with torch.no_grad():
            pred = transformer(
                hidden_states=noisy,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        target = noise - latents
        loss = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=target_dims)
        losses[str(timestep)] = [float(x) for x in loss.cpu()]
    return losses


def run_dit_loss(args: argparse.Namespace) -> dict:
    from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

    rng = np.random.default_rng(args.seed + 1000)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    vae = load_vae(device, dtype)
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.to(device)
    transformer.eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    normal, reversed_video, shuffled = pair_videos_for_dit(
        rng, args.dit_pairs, args.frames, args.height, args.width
    )
    all_videos = np.concatenate([normal, reversed_video, shuffled], axis=0)
    z = encode_wan_vae(vae, all_videos, args.batch_size, device, dtype)
    z_torch = torch.from_numpy(z).to(device=device, dtype=dtype)

    raw_timesteps = [int(x) for x in args.dit_timesteps.split(",")]
    losses = denoise_losses(
        transformer,
        scheduler,
        z_torch,
        raw_timesteps,
        args.seed + 2000,
        int(transformer.config.text_dim),
    )

    n = args.dit_pairs
    summary: dict[str, dict] = {}
    for timestep, values in losses.items():
        normal_loss = np.asarray(values[:n], dtype=np.float64)
        reversed_loss = np.asarray(values[n : 2 * n], dtype=np.float64)
        shuffled_loss = np.asarray(values[2 * n :], dtype=np.float64)
        summary[timestep] = {
            "normal_mean": float(normal_loss.mean()),
            "reversed_mean": float(reversed_loss.mean()),
            "shuffled_mean": float(shuffled_loss.mean()),
            "normal_lt_reversed_rate": float((normal_loss < reversed_loss).mean()),
            "normal_lt_shuffled_rate": float((normal_loss < shuffled_loss).mean()),
            "normal_minus_reversed_mean": float((normal_loss - reversed_loss).mean()),
            "normal_minus_shuffled_mean": float((normal_loss - shuffled_loss).mean()),
            "normal": normal_loss.tolist(),
            "reversed": reversed_loss.tolist(),
            "shuffled": shuffled_loss.tolist(),
        }
    return {
        "n_pairs": n,
        "latent_shape": list(z_torch.shape[1:]),
        "zero_text_conditioning": True,
        "summary": summary,
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# Wan Feature Sanity Results")
    lines.append("")
    lines.append("## VAE linear probes")
    lines.append("")
    lines.append("| task | feature | accuracy | balanced_accuracy | dim |")
    lines.append("|---|---:|---:|---:|---:|")
    for task, task_result in payload.get("vae_probes", {}).items():
        for feature, result in task_result["features"].items():
            lines.append(
                f"| {task} | {feature} | {result['accuracy']:.4f} | "
                f"{result['balanced_accuracy']:.4f} | {result['feature_dim']} |"
            )
    if "dit_loss" in payload:
        lines.append("")
        lines.append("## DiT denoising loss")
        lines.append("")
        lines.append("| timestep | normal | reversed | shuffled | normal<reversed | normal<shuffled |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for timestep, result in payload["dit_loss"]["summary"].items():
            lines.append(
                f"| {timestep} | {result['normal_mean']:.6f} | {result['reversed_mean']:.6f} | "
                f"{result['shuffled_mean']:.6f} | {result['normal_lt_reversed_rate']:.3f} | "
                f"{result['normal_lt_shuffled_rate']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-base", type=int, default=64)
    parser.add_argument("--test-base", type=int, default=32)
    parser.add_argument("--baseline-hw", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dit", action="store_true")
    parser.add_argument("--dit-pairs", type=int, default=8)
    parser.add_argument("--dit-timesteps", default="900,500,100")
    parser.add_argument("--output-json", default="results/wan_feature_sanity_results.json")
    parser.add_argument("--output-md", default="results/wan_feature_sanity_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": MODEL_ID,
        "config": vars(args),
        "vae_probes": run_vae_probes(args),
    }
    if args.run_dit:
        payload["dit_loss"] = run_dit_loss(args)

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
