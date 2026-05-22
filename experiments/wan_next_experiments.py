#!/usr/bin/env python3
"""Follow-up Wan feature experiments.

Experiments:

1. Pooling ablation for Wan-VAE and low-level pixel features.
2. Wan-DiT hidden-state layer/timestep sweep with linear probes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    average_pool_video,
    encode_wan_vae,
    load_vae,
    make_direction_task,
    make_reversal_task,
    make_shuffle_task,
)


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


def pixel_pooling_features(videos: np.ndarray, kind: str, out_hw: int) -> np.ndarray:
    pooled = average_pool_video(videos, out_hw)
    if kind == "pixel_rgb_clip_like_avg":
        return videos.mean(axis=(1, 2, 3))
    if kind == "pixel_grid_time_avg":
        return pooled.mean(axis=1).reshape(videos.shape[0], -1)
    if kind == "pixel_grid_sequence":
        return pooled.reshape(videos.shape[0], -1)
    if kind == "pixel_grid_delta":
        return np.diff(pooled, axis=1).reshape(videos.shape[0], -1)
    raise ValueError(kind)


def vae_pooling_features(latents: np.ndarray, kind: str) -> np.ndarray:
    if kind == "wan_vae_channel_avg":
        return latents.mean(axis=(2, 3, 4))
    if kind == "wan_vae_grid_time_avg":
        return latents.mean(axis=2).reshape(latents.shape[0], -1)
    if kind == "wan_vae_global_sequence":
        return latents.mean(axis=(3, 4)).reshape(latents.shape[0], -1)
    if kind == "wan_vae_grid_sequence":
        return latents.reshape(latents.shape[0], -1)
    if kind == "wan_vae_global_delta":
        seq = latents.mean(axis=(3, 4))
        return np.diff(seq, axis=2).reshape(latents.shape[0], -1)
    raise ValueError(kind)


def build_tasks(args: argparse.Namespace, seed_offset: int = 0) -> dict[str, TaskData]:
    rng = np.random.default_rng(args.seed + seed_offset)
    return {
        "direction4": make_direction_task(
            rng, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
        "reversal": make_reversal_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
        "shuffle": make_shuffle_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
    }


def run_pooling_ablation(args: argparse.Namespace, vae, device: torch.device, dtype: torch.dtype) -> dict:
    pixel_kinds = [
        "pixel_rgb_clip_like_avg",
        "pixel_grid_time_avg",
        "pixel_grid_sequence",
        "pixel_grid_delta",
    ]
    vae_kinds = [
        "wan_vae_channel_avg",
        "wan_vae_grid_time_avg",
        "wan_vae_global_sequence",
        "wan_vae_grid_sequence",
        "wan_vae_global_delta",
    ]
    results: dict[str, dict] = {}

    for seed_idx in range(args.pooling_seeds):
        seed_key = f"seed_{args.seed + seed_idx}"
        results[seed_key] = {}
        for task_name, task in build_tasks(args, seed_idx).items():
            task_result = {"labels": task.labels, "features": {}}
            for kind in pixel_kinds:
                x_train = pixel_pooling_features(task.x_train, kind, args.baseline_hw)
                x_test = pixel_pooling_features(task.x_test, kind, args.baseline_hw)
                task_result["features"][kind] = train_probe(x_train, task.y_train, x_test, task.y_test)

            z_train = encode_wan_vae(vae, task.x_train, args.batch_size, device, dtype)
            z_test = encode_wan_vae(vae, task.x_test, args.batch_size, device, dtype)
            task_result["latent_shape"] = list(z_train.shape[1:])
            for kind in vae_kinds:
                x_train = vae_pooling_features(z_train, kind)
                x_test = vae_pooling_features(z_test, kind)
                task_result["features"][kind] = train_probe(x_train, task.y_train, x_test, task.y_test)
            results[seed_key][task_name] = task_result
    return results


def summarize_pooling(pooling: dict) -> dict:
    aggregate: dict[str, dict[str, list[float]]] = {}
    for seed_result in pooling.values():
        for task, task_result in seed_result.items():
            aggregate.setdefault(task, {})
            for feature, result in task_result["features"].items():
                aggregate[task].setdefault(feature, []).append(result["balanced_accuracy"])

    summary: dict[str, dict] = {}
    for task, feature_values in aggregate.items():
        summary[task] = {}
        for feature, values in feature_values.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[task][feature] = {
                "balanced_accuracy_mean": float(arr.mean()),
                "balanced_accuracy_std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
    return summary


def register_hooks(transformer, layers: list[int], store: dict[int, torch.Tensor]) -> list:
    handles = []

    def make_hook(layer_id: int):
        def hook(_module, _inputs, output):
            store[layer_id] = output.detach()

        return hook

    for layer in layers:
        handles.append(transformer.blocks[layer].register_forward_hook(make_hook(layer)))
    return handles


def pool_dit_hidden(hidden: torch.Tensor, latent_shape: tuple[int, int, int, int], kind: str) -> np.ndarray:
    _, latent_frames, latent_height, latent_width = latent_shape
    patch_t, patch_h, patch_w = (1, 2, 2)
    post_t = latent_frames // patch_t
    post_h = latent_height // patch_h
    post_w = latent_width // patch_w
    bsz, tokens, channels = hidden.shape
    if tokens != post_t * post_h * post_w:
        raise ValueError(f"Unexpected token count {tokens}; expected {post_t * post_h * post_w}")
    x = hidden.float().reshape(bsz, post_t, post_h, post_w, channels)
    if kind == "token_mean":
        return x.mean(dim=(1, 2, 3)).cpu().numpy()
    if kind == "temporal_sequence":
        return x.mean(dim=(2, 3)).reshape(bsz, -1).cpu().numpy()
    if kind == "temporal_delta":
        seq = x.mean(dim=(2, 3))
        return torch.diff(seq, dim=1).reshape(bsz, -1).cpu().numpy()
    if kind == "spatial_tokens":
        return x.reshape(bsz, post_t * post_h * post_w, channels).cpu().numpy()
    raise ValueError(kind)


def encode_dit_features(
    transformer,
    scheduler,
    latents_np: np.ndarray,
    layers: list[int],
    timesteps: list[int],
    pooling_kinds: list[str],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, np.ndarray]:
    features: dict[str, list[np.ndarray]] = {}
    latent_shape = tuple(latents_np.shape[1:])
    generator = torch.Generator(device=device).manual_seed(seed)

    for timestep in timesteps:
        for layer in layers:
            for kind in pooling_kinds:
                features[f"dit_l{layer:02d}_t{timestep}_{kind}"] = []

    for start in range(0, len(latents_np), batch_size):
        batch_np = latents_np[start : start + batch_size]
        latents = torch.from_numpy(batch_np).to(device=device, dtype=dtype)
        bsz = latents.shape[0]
        prompt_embeds = torch.zeros(bsz, 1, int(transformer.config.text_dim), device=device, dtype=dtype)

        for timestep in timesteps:
            store: dict[int, torch.Tensor] = {}
            handles = register_hooks(transformer, layers, store)
            t = torch.full((bsz,), int(timestep), device=device, dtype=torch.long)
            noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
            noisy = scheduler.add_noise(latents, noise, t)
            with torch.no_grad():
                transformer(
                    hidden_states=noisy,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
            for handle in handles:
                handle.remove()
            for layer, hidden in store.items():
                for kind in pooling_kinds:
                    key = f"dit_l{layer:02d}_t{timestep}_{kind}"
                    features[key].append(pool_dit_hidden(hidden, latent_shape, kind))

    return {key: np.concatenate(value, axis=0) for key, value in features.items()}


def run_dit_hidden_sweep(args: argparse.Namespace, vae, device: torch.device, dtype: torch.dtype) -> dict:
    from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
    transformer.to(device)
    transformer.eval()
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    layers = [int(x) for x in args.dit_layers.split(",")]
    timesteps = [int(x) for x in args.dit_timesteps.split(",")]
    pooling_kinds = [x.strip() for x in args.dit_pooling.split(",") if x.strip()]
    tasks = build_tasks(args, seed_offset=100)
    results: dict[str, dict] = {}

    for task_name in args.dit_tasks.split(","):
        task_name = task_name.strip()
        task = tasks[task_name]
        z_train = encode_wan_vae(vae, task.x_train, args.batch_size, device, dtype)
        z_test = encode_wan_vae(vae, task.x_test, args.batch_size, device, dtype)

        train_features = encode_dit_features(
            transformer,
            scheduler,
            z_train,
            layers,
            timesteps,
            pooling_kinds,
            args.batch_size,
            device,
            dtype,
            args.seed + 3000,
        )
        test_features = encode_dit_features(
            transformer,
            scheduler,
            z_test,
            layers,
            timesteps,
            pooling_kinds,
            args.batch_size,
            device,
            dtype,
            args.seed + 4000,
        )

        task_result = {"labels": task.labels, "latent_shape": list(z_train.shape[1:]), "features": {}}
        for key, x_train in train_features.items():
            task_result["features"][key] = train_probe(x_train, task.y_train, test_features[key], task.y_test)
        results[task_name] = task_result

    return results


def summarize_dit(dit: dict) -> dict:
    summary: dict[str, dict] = {}
    for task, task_result in dit.items():
        rows = []
        for feature, result in task_result["features"].items():
            parts = feature.split("_")
            layer = parts[1][1:]
            timestep = parts[2][1:]
            pooling = "_".join(parts[3:])
            rows.append(
                {
                    "feature": feature,
                    "layer": int(layer),
                    "timestep": int(timestep),
                    "pooling": pooling,
                    "balanced_accuracy": result["balanced_accuracy"],
                    "accuracy": result["accuracy"],
                    "feature_dim": result["feature_dim"],
                }
            )
        rows.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
        summary[task] = {
            "best": rows[:10],
            "all": rows,
        }
    return summary


def write_markdown(path: Path, payload: dict) -> None:
    lines = ["# Wan next experiment results", ""]
    if "pooling_summary" in payload:
        lines.extend(["## VAE/pixel pooling ablation", ""])
        lines.append("| task | feature | balanced_accuracy_mean | std | n |")
        lines.append("|---|---:|---:|---:|---:|")
        for task, features in payload["pooling_summary"].items():
            for feature, result in sorted(
                features.items(), key=lambda kv: kv[1]["balanced_accuracy_mean"], reverse=True
            ):
                lines.append(
                    f"| {task} | {feature} | {result['balanced_accuracy_mean']:.4f} | "
                    f"{result['balanced_accuracy_std']:.4f} | {result['n']} |"
                )
    if "dit_summary" in payload:
        lines.extend(["", "## DiT hidden-state sweep: top features", ""])
        lines.append("| task | feature | layer | timestep | pooling | balanced_accuracy | dim |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for task, result in payload["dit_summary"].items():
            for row in result["best"][:10]:
                lines.append(
                    f"| {task} | {row['feature']} | {row['layer']} | {row['timestep']} | "
                    f"{row['pooling']} | {row['balanced_accuracy']:.4f} | {row['feature_dim']} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-base", type=int, default=32)
    parser.add_argument("--test-base", type=int, default=16)
    parser.add_argument("--baseline-hw", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-pooling", action="store_true")
    parser.add_argument("--pooling-seeds", type=int, default=3)
    parser.add_argument("--skip-dit", action="store_true")
    parser.add_argument("--dit-tasks", default="direction4,shuffle")
    parser.add_argument("--dit-layers", default="0,14,29")
    parser.add_argument("--dit-timesteps", default="900,500,100")
    parser.add_argument("--dit-pooling", default="token_mean,temporal_sequence,temporal_delta")
    parser.add_argument("--output-json", default="results/wan_next_experiments_results.json")
    parser.add_argument("--output-md", default="results/wan_next_experiments_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = load_vae(device, dtype)

    payload = {
        "model_id": MODEL_ID,
        "config": vars(args),
        "device": str(device),
    }
    if not args.skip_pooling:
        payload["pooling_ablation"] = run_pooling_ablation(args, vae, device, dtype)
        payload["pooling_summary"] = summarize_pooling(payload["pooling_ablation"])
    if not args.skip_dit:
        payload["dit_hidden_sweep"] = run_dit_hidden_sweep(args, vae, device, dtype)
        payload["dit_summary"] = summarize_dit(payload["dit_hidden_sweep"])

    payload["seconds"] = round(time.time() - t0, 2)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(out_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
