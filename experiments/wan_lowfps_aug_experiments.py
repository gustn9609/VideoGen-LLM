#!/usr/bin/env python3
"""Low-frame-rate augmentation probes for Wan features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_feature_sanity import MODEL_ID, encode_wan_vae, load_vae, make_direction_task  # noqa: E402
from extract_wan_features import apply_mode_transforms, resize_frames, sample_indices  # noqa: E402
from wan_next_experiments import encode_dit_features, pixel_pooling_features, vae_pooling_features  # noqa: E402
from wan_temporal_reasoning_experiments import degrade_low_fps, make_before_after_cycle_task  # noqa: E402


FEATURES_PIXEL = [
    "pixel_grid_sequence",
    "pixel_grid_delta",
]
FEATURES_VAE = [
    "wan_vae_global_sequence",
    "wan_vae_grid_sequence",
    "wan_vae_global_delta",
]


def train_eval(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "feature_dim": int(x_train.shape[1]),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def extract_feature_set(
    videos: np.ndarray,
    vae,
    transformer,
    scheduler,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    dit_seed: int,
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for feature in FEATURES_PIXEL:
        features[feature] = pixel_pooling_features(videos, feature, args.baseline_hw)

    z = encode_wan_vae(vae, videos, args.batch_size, device, dtype)
    for feature in FEATURES_VAE:
        features[feature] = vae_pooling_features(z, feature)

    if transformer is not None and scheduler is not None:
        dit_features = encode_dit_features(
            transformer,
            scheduler,
            z,
            [args.dit_layer],
            [args.dit_timestep],
            [args.dit_pooling],
            args.batch_size,
            device,
            dtype,
            dit_seed,
        )
        features.update(dit_features)
    return features


def build_tasks(args: argparse.Namespace, seed_offset: int) -> dict:
    rng = np.random.default_rng(args.seed + seed_offset)
    return {
        "direction4": make_direction_task(rng, args.train_base, args.test_base, args.frames, args.height, args.width),
        "before_after_cycle": make_before_after_cycle_task(
            rng, args.train_base, args.test_base, args.frames, args.height, args.width
        ),
    }


def augment_videos(videos: np.ndarray, mode: str, seed: int, lowfps_keyframes: int) -> np.ndarray:
    if mode in {"none", "full"}:
        return videos.copy()
    if mode == "uniform":
        mode = f"uniform{lowfps_keyframes}"
    if mode == "nonuniform":
        mode = f"nonuniform{lowfps_keyframes}"

    out = []
    frames = videos.shape[1]
    for idx, video in enumerate(videos):
        unit = np.clip((video + 1.0) / 2.0, 0.0, 1.0)
        sampled_idx = sample_indices(frames, frames, mode, seed + idx)
        sampled = unit[sampled_idx]
        transformed, _ = apply_mode_transforms(sampled, mode, seed + idx)
        if transformed.shape[1] != videos.shape[2] or transformed.shape[2] != videos.shape[3]:
            transformed = resize_frames(transformed, videos.shape[2], videos.shape[3])
        out.append(np.clip(transformed * 2.0 - 1.0, -1.0, 1.0).astype(np.float32))
    return np.stack(out, axis=0)


def run(args: argparse.Namespace, vae, transformer, scheduler, device: torch.device, dtype: torch.dtype) -> dict:
    results: dict[str, dict] = {}
    aug_modes = [item.strip() for item in args.augmentation_modes.split(",") if item.strip()]
    for seed_idx in range(args.seeds):
        seed_key = f"seed_{args.seed + seed_idx}"
        results[seed_key] = {}
        for task_name, task in build_tasks(args, seed_idx).items():
            full_train_features = extract_feature_set(
                task.x_train,
                vae,
                transformer,
                scheduler,
                args,
                device,
                dtype,
                args.seed + seed_idx * 1000 + 10,
            )
            full_test_features = extract_feature_set(
                task.x_test,
                vae,
                transformer,
                scheduler,
                args,
                device,
                dtype,
                args.seed + seed_idx * 1000 + 30,
            )

            task_result = {"labels": task.labels, "augmentations": {}}
            for aug_idx, aug_mode in enumerate(aug_modes):
                if aug_mode in {"none", "full"}:
                    aug_train_features = full_train_features
                    aug_test_features = full_test_features
                else:
                    aug_train = augment_videos(task.x_train, aug_mode, args.seed + seed_idx * 10000 + aug_idx * 100, args.lowfps_keyframes)
                    aug_test = augment_videos(
                        task.x_test,
                        aug_mode,
                        args.seed + seed_idx * 10000 + aug_idx * 100 + 5000,
                        args.lowfps_keyframes,
                    )
                    aug_train_features = extract_feature_set(
                        aug_train,
                        vae,
                        transformer,
                        scheduler,
                        args,
                        device,
                        dtype,
                        args.seed + seed_idx * 1000 + aug_idx * 20 + 40,
                    )
                    aug_test_features = extract_feature_set(
                        aug_test,
                        vae,
                        transformer,
                        scheduler,
                        args,
                        device,
                        dtype,
                        args.seed + seed_idx * 1000 + aug_idx * 20 + 50,
                    )

                mode_result = {"features": {}}
                for feature in full_train_features:
                    y_aug = np.concatenate([task.y_train, task.y_train], axis=0)
                    x_aug = np.concatenate([full_train_features[feature], aug_train_features[feature]], axis=0)
                    mode_result["features"][feature] = {
                        "train_full_test_full": train_eval(
                            full_train_features[feature], task.y_train, full_test_features[feature], task.y_test
                        ),
                        "train_full_test_aug": train_eval(
                            full_train_features[feature], task.y_train, aug_test_features[feature], task.y_test
                        ),
                        "train_aug_test_aug": train_eval(
                            aug_train_features[feature], task.y_train, aug_test_features[feature], task.y_test
                        ),
                        "train_full_plus_aug_test_aug": train_eval(x_aug, y_aug, aug_test_features[feature], task.y_test),
                        "train_full_plus_aug_test_full": train_eval(x_aug, y_aug, full_test_features[feature], task.y_test),
                    }
                task_result["augmentations"][aug_mode] = mode_result
            results[seed_key][task_name] = task_result
    return results


def summarize(results: dict) -> dict:
    fields = [
        "train_full_test_full",
        "train_full_test_aug",
        "train_aug_test_aug",
        "train_full_plus_aug_test_aug",
        "train_full_plus_aug_test_full",
    ]
    aggregate: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for seed_result in results.values():
        for task_name, task_result in seed_result.items():
            aggregate.setdefault(task_name, {})
            for aug_mode, aug_result in task_result["augmentations"].items():
                aggregate[task_name].setdefault(aug_mode, {})
                for feature, feature_result in aug_result["features"].items():
                    aggregate[task_name][aug_mode].setdefault(feature, {field: [] for field in fields})
                    for field in fields:
                        aggregate[task_name][aug_mode][feature][field].append(feature_result[field]["balanced_accuracy"])

    summary: dict[str, dict] = {}
    for task_name, aug_modes in aggregate.items():
        summary[task_name] = {}
        for aug_mode, features in aug_modes.items():
            summary[task_name][aug_mode] = {}
            for feature, values in features.items():
                row = {}
                for field, nums in values.items():
                    arr = np.asarray(nums, dtype=np.float64)
                    row[field] = {
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=0)),
                        "n": int(arr.size),
                    }
                row["aug_gain"] = row["train_full_plus_aug_test_aug"]["mean"] - row["train_full_test_aug"]["mean"]
                row["aug_only_gain"] = row["train_aug_test_aug"]["mean"] - row["train_full_test_aug"]["mean"]
                summary[task_name][aug_mode][feature] = row
    return summary


def write_markdown(path: Path, payload: dict) -> None:
    lines = ["# Wan low-fps augmentation experiment results", ""]
    lines.append("| task | aug_mode | feature | full->full | full->aug | aug->aug | full+aug->aug | aug gain |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for task_name, aug_modes in payload["summary"].items():
        rows = []
        for aug_mode, features in aug_modes.items():
            for feature, result in features.items():
                rows.append((aug_mode, feature, result))
        rows.sort(key=lambda kv: kv[2]["aug_gain"], reverse=True)
        for aug_mode, feature, result in rows:
            lines.append(
                f"| {task_name} | {aug_mode} | {feature} | "
                f"{result['train_full_test_full']['mean']:.4f} | "
                f"{result['train_full_test_aug']['mean']:.4f} | "
                f"{result['train_aug_test_aug']['mean']:.4f} | "
                f"{result['train_full_plus_aug_test_aug']['mean']:.4f} | "
                f"{result['aug_gain']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
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
    parser.add_argument(
        "--augmentation-modes",
        default="uniform5,nonuniform5,speed0.5,speed2,repeat8,none+blur,none+jpeg45,none+crop_shift,reverse,shuffle",
    )
    parser.add_argument("--output-json", default="results/wan_lowfps_aug_results.json")
    parser.add_argument("--output-md", default="results/wan_lowfps_aug_results.md")
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
