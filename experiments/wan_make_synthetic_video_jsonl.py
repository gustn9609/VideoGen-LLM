#!/usr/bin/env python3
"""Create a small JSONL video set for extractor/probe smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def background(rng: np.random.Generator, frames: int, height: int, width: int) -> np.ndarray:
    base = rng.uniform(0.04, 0.12, size=(height, width, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    texture = 0.04 * np.sin(xx / 7.0 + rng.uniform(0, 2 * np.pi)) + 0.03 * np.cos(yy / 9.0)
    image = np.clip(base + texture[..., None], 0.0, 1.0)
    return np.repeat(image[None], frames, axis=0)


def draw_disk(frame: np.ndarray, cx: float, cy: float, radius: float, color: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    out = frame.copy()
    out[mask] = color
    return out


def make_video(rng: np.random.Generator, label: str, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    radius = float(rng.uniform(5.0, 8.0))
    color = np.array([0.9, 0.2, 0.1], dtype=np.float32)
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for frame_idx, t in enumerate(ts):
        if label == "right":
            x = width * (0.18 + 0.64 * t)
            y = height * 0.45
        elif label == "left":
            x = width * (0.82 - 0.64 * t)
            y = height * 0.55
        elif label == "upper_arc":
            x = width * (0.18 + 0.64 * t)
            y = height * (0.58 - 0.25 * np.sin(np.pi * t))
        else:
            x = width * (0.18 + 0.64 * t)
            y = height * (0.42 + 0.25 * np.sin(np.pi * t))
        video[frame_idx] = draw_disk(video[frame_idx], x, y, radius, color)
    return video.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--test-per-class", type=int, default=4)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    labels = ["right", "left", "upper_arc", "lower_arc"]
    rows = []
    rng = np.random.default_rng(args.seed)
    for split, per_class in [("train", args.train_per_class), ("test", args.test_per_class)]:
        for label in labels:
            for item_idx in range(per_class):
                video = make_video(rng, label, args.frames, args.height, args.width)
                video_id = f"{split}_{label}_{item_idx:03d}"
                path = video_dir / f"{video_id}.npy"
                np.save(path, video)
                rows.append(
                    {
                        "video_id": video_id,
                        "path": str(path),
                        "label": label,
                        "task": "motion_direction_path",
                        "split": split,
                        "fps": args.fps,
                        "duration": float(args.frames) / float(args.fps),
                    }
                )
    jsonl_path = out_dir / "videos.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "jsonl": str(jsonl_path), "video_dir": str(video_dir)}, indent=2))


if __name__ == "__main__":
    main()
