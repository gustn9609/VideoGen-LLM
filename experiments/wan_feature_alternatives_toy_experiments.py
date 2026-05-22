#!/usr/bin/env python3
"""Toy experiments for alternative Wan feature usages.

This script implements the minimal experiment set from
`wan_feature_alternatives_toy_experiments.md`:

1. Wan Perturbation Signature (WPS) gate.
2. VAE latent trajectory features.
3. Structured compact spatial features.
4. Training-free relation descriptors.
5. Prototype/feature-space CFG and future-consistency proxies.
6. Segment localization toy.

The expensive DiT CFG and future generation ideas are exposed as lightweight
feature-space proxies here so the full toy suite can run without invoking video
generation. Wan-VAE features are real Wan features when `--feature-source wan`
is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_feature_sanity import MODEL_ID, draw_disk, encode_wan_vae, load_vae  # noqa: E402


@dataclass
class TaskData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: list[str]
    task_type: str


@dataclass
class FeatureBundle:
    videos: np.ndarray
    latents: np.ndarray | None


def background(rng: np.random.Generator, frames: int, height: int, width: int, camera_dx: float = 0.0) -> np.ndarray:
    base = rng.uniform(0.05, 0.20, size=(1, height, width, 3)).astype(np.float32)
    yy = np.linspace(0.0, 0.10, height, dtype=np.float32).reshape(height, 1, 1)
    xx = np.linspace(0.0, 0.07, width, dtype=np.float32).reshape(1, width, 1)
    pattern = base[0] + yy + xx
    frames_out = []
    for t in range(frames):
        shift = int(round(camera_dx * (t / max(1, frames - 1))))
        frame = np.roll(pattern, shift=shift, axis=1)
        noise = rng.normal(0.0, 0.004, size=(height, width, 3)).astype(np.float32)
        frames_out.append(np.clip(frame + noise, 0.0, 1.0))
    return np.stack(frames_out, axis=0).astype(np.float32)


def to_model_range(video_01: np.ndarray) -> np.ndarray:
    return (np.clip(video_01, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)


def from_model_range(video: np.ndarray) -> np.ndarray:
    return np.clip((video.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)


def draw_rect(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w, _ = out.shape
    xa, xb = int(max(0, round(min(x0, x1)))), int(min(w, round(max(x0, x1))))
    ya, yb = int(max(0, round(min(y0, y1)))), int(min(h, round(max(y0, y1))))
    out[ya:yb, xa:xb] = color.reshape(1, 1, 3)
    return out


def linear_motion_video(
    rng: np.random.Generator,
    direction: int,
    frames: int,
    height: int,
    width: int,
    camera_dx: float = 0.0,
    color: np.ndarray | None = None,
) -> np.ndarray:
    video = background(rng, frames, height, width, camera_dx=camera_dx)
    color = rng.uniform(0.45, 0.95, size=3).astype(np.float32) if color is None else color.astype(np.float32)
    radius = float(rng.uniform(4.5, 7.0))
    margin = radius + 5.0
    span_x = width - 2 * margin
    span_y = height - 2 * margin
    fixed_x = rng.uniform(margin + 0.25 * span_x, margin + 0.75 * span_x)
    fixed_y = rng.uniform(margin + 0.25 * span_y, margin + 0.75 * span_y)
    ts = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    for i, t in enumerate(ts):
        if direction == 0:
            cx, cy = margin + span_x * t, fixed_y
        elif direction == 1:
            cx, cy = margin + span_x * (1.0 - t), fixed_y
        elif direction == 2:
            cx, cy = fixed_x, margin + span_y * t
        else:
            cx, cy = fixed_x, margin + span_y * (1.0 - t)
        video[i] = draw_disk(video[i], cx + camera_dx * t, cy, radius, color)
    return to_model_range(video)


def event_order_video(rng: np.random.Generator, label: int, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.95, 0.12, 0.10], dtype=np.float32)
    blue = np.array([0.10, 0.25, 0.95], dtype=np.float32)
    positions = [(width * 0.30, height * 0.52), (width * 0.70, height * 0.52)]
    first, second = (0, 1) if label == 0 else (1, 0)
    event_frames = [frames // 3, 2 * frames // 3]
    for t in range(frames):
        frame = video[t]
        for obj, (cx, cy) in enumerate(positions):
            color = red if obj == 0 else blue
            boost = 1.0
            if obj == first and abs(t - event_frames[0]) <= 2:
                boost = 1.6
            if obj == second and abs(t - event_frames[1]) <= 2:
                boost = 1.6
            frame = draw_disk(frame, cx, cy, 6.0 * boost, np.clip(color * boost, 0.0, 1.0))
        video[t] = frame
    return to_model_range(video)


def repetition_video(rng: np.random.Generator, count: int, frames: int, height: int, width: int, rhythm: str = "regular") -> np.ndarray:
    video = background(rng, frames, height, width)
    color = np.array([0.92, 0.25, 0.16], dtype=np.float32)
    cx = width * 0.50 + rng.uniform(-3.0, 3.0)
    base_y = height * 0.50
    if rhythm == "bursty":
        centers = np.linspace(0.25, 0.55, count)
    elif rhythm == "irregular":
        centers = np.sort(rng.uniform(0.15, 0.85, size=count))
    else:
        centers = np.linspace(0.18, 0.82, count)
    ts = np.linspace(0.0, 1.0, frames)
    for i, t in enumerate(ts):
        pulse = 0.0
        for c in centers:
            pulse += math.exp(-((float(t) - float(c)) ** 2) / (2.0 * 0.018**2))
        y = base_y - 16.0 * min(1.0, pulse)
        video[i] = draw_disk(video[i], cx, y, 6.0, color)
        video[i] = draw_rect(video[i], cx - 8, base_y + 12, cx + 8, base_y + 14, np.array([0.85, 0.85, 0.85], dtype=np.float32))
    return to_model_range(video)


def middle_path_video(rng: np.random.Generator, label: int, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    color = np.array([0.85, 0.25, 0.95], dtype=np.float32)
    start = np.array([width * 0.20, height * 0.50])
    end = np.array([width * 0.80, height * 0.50])
    control_y = height * (0.25 if label == 0 else 0.75)
    control = np.array([width * 0.50, control_y])
    obstacle = np.array([0.15, 0.15, 0.15], dtype=np.float32)
    ts = np.linspace(0.0, 1.0, frames)
    for i, t in enumerate(ts):
        p = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end
        frame = draw_rect(video[i], width * 0.45, height * 0.42, width * 0.55, height * 0.58, obstacle)
        video[i] = draw_disk(frame, float(p[0]), float(p[1]), 5.5, color)
    return to_model_range(video)


def contact_video(rng: np.random.Generator, label: int, frames: int, height: int, width: int) -> np.ndarray:
    video = background(rng, frames, height, width)
    red = np.array([0.95, 0.10, 0.10], dtype=np.float32)
    green = np.array([0.10, 0.90, 0.25], dtype=np.float32)
    ts = np.linspace(0.0, 1.0, frames)
    for i, t in enumerate(ts):
        if label == 0:  # hit: B moves after contact
            ax = width * (0.20 + 0.35 * min(t / 0.55, 1.0))
            bx = width * (0.55 + 0.25 * max(0.0, (t - 0.55) / 0.45))
            ay = by = height * 0.50
        elif label == 1:  # miss: A passes above, B static
            ax = width * (0.20 + 0.65 * t)
            ay = height * 0.35
            bx, by = width * 0.58, height * 0.58
        else:  # B moves before A arrives
            ax = width * (0.20 + 0.35 * min(t / 0.70, 1.0))
            bx = width * (0.55 + 0.25 * min(t / 0.45, 1.0))
            ay = by = height * 0.50
        frame = draw_disk(video[i], ax, ay, 5.5, red)
        video[i] = draw_disk(frame, bx, by, 5.5, green)
    return to_model_range(video)


def camera_object_video(rng: np.random.Generator, label: int, frames: int, height: int, width: int) -> np.ndarray:
    camera_dx = {-1: 0.0, 0: -14.0, 1: 0.0, 2: -10.0}[label if label in {0, 1, 2} else -1]
    video = background(rng, frames, height, width, camera_dx=camera_dx)
    color = np.array([0.95, 0.72, 0.15], dtype=np.float32)
    ts = np.linspace(0.0, 1.0, frames)
    for i, t in enumerate(ts):
        if label == 0:  # camera pans, object static in world
            cx = width * 0.50 + camera_dx * t
        elif label == 1:  # object moves, camera static
            cx = width * (0.25 + 0.50 * t)
        else:  # both
            cx = width * (0.25 + 0.50 * t) + camera_dx * t
        cy = height * 0.52
        video[i] = draw_disk(video[i], cx, cy, 6.0, color)
    return to_model_range(video)


def shuffle_detection_video(rng: np.random.Generator, label: int, frames: int, height: int, width: int) -> np.ndarray:
    normal = linear_motion_video(rng, int(rng.integers(0, 4)), frames, height, width)
    if label == 0:
        return normal
    perm = rng.permutation(frames)
    while np.all(perm == np.arange(frames)):
        perm = rng.permutation(frames)
    return normal[perm].copy()


def make_task(
    rng: np.random.Generator,
    name: str,
    train_per_class: int,
    test_per_class: int,
    frames: int,
    height: int,
    width: int,
) -> TaskData:
    specs: dict[str, tuple[list[str], Callable[[np.random.Generator, int, int, int, int], np.ndarray], str]] = {
        "direction4": (
            ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"],
            lambda r, y, f, h, w: linear_motion_video(r, y, f, h, w),
            "direction",
        ),
        "action_order": (
            ["red_before_blue", "blue_before_red"],
            event_order_video,
            "order",
        ),
        "repetition_count": (
            ["two", "three", "four"],
            lambda r, y, f, h, w: repetition_video(r, [2, 3, 4][y], f, h, w),
            "count",
        ),
        "same_first_last_path": (
            ["above_obstacle", "below_obstacle"],
            middle_path_video,
            "path",
        ),
        "contact_causality": (
            ["hit_then_moves", "miss_static", "moves_before_contact"],
            contact_video,
            "causality",
        ),
        "camera_object_motion": (
            ["camera_pan_static_object", "object_moves_static_camera", "both_camera_and_object"],
            camera_object_video,
            "camera",
        ),
        "shuffle_detection": (
            ["smooth_order", "shuffled_order"],
            shuffle_detection_video,
            "shuffle",
        ),
        "rhythm": (
            ["regular", "irregular", "bursty"],
            lambda r, y, f, h, w: repetition_video(r, 3, f, h, w, rhythm=["regular", "irregular", "bursty"][y]),
            "rhythm",
        ),
    }
    labels, builder, task_type = specs[name]

    def build(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        videos: list[np.ndarray] = []
        ys: list[int] = []
        for label_id in range(len(labels)):
            for _ in range(n_per_class):
                videos.append(builder(rng, label_id, frames, height, width))
                ys.append(label_id)
        order = rng.permutation(len(videos))
        return np.stack(videos)[order], np.asarray(ys, dtype=np.int64)[order]

    x_train, y_train = build(train_per_class)
    x_test, y_test = build(test_per_class)
    return TaskData(x_train, y_train, x_test, y_test, labels, task_type)


def transform_videos(videos: np.ndarray, transform: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if transform == "normal":
        return videos.copy()
    if transform == "reverse":
        return videos[:, ::-1].copy()
    if transform == "shuffle":
        out = videos.copy()
        for i in range(len(out)):
            order = rng.permutation(out.shape[1])
            while np.all(order == np.arange(out.shape[1])):
                order = rng.permutation(out.shape[1])
            out[i] = out[i, order]
        return out
    if transform == "timeavg":
        avg = videos.mean(axis=1, keepdims=True)
        return np.repeat(avg, videos.shape[1], axis=1).astype(np.float32)
    if transform == "first_repeat":
        return np.repeat(videos[:, :1], videos.shape[1], axis=1).astype(np.float32)
    if transform == "lowfps":
        frames = videos.shape[1]
        key = np.linspace(0, frames - 1, max(2, frames // 3)).round().astype(int)
        repeat = np.linspace(0, len(key) - 1, frames).round().astype(int)
        return videos[:, key[repeat]].copy()
    if transform == "hflip":
        return videos[:, :, :, ::-1].copy()
    raise ValueError(transform)


def average_pool_video(videos: np.ndarray, out_h: int, out_w: int | None = None) -> np.ndarray:
    out_w = out_h if out_w is None else out_w
    videos_01 = from_model_range(videos)
    n, f, h, w, c = videos_01.shape
    ph = h // out_h
    pw = w // out_w
    cropped = videos_01[:, :, : out_h * ph, : out_w * pw, :]
    return cropped.reshape(n, f, out_h, ph, out_w, pw, c).mean(axis=(3, 5)).astype(np.float32)


def pool_latents(latents: np.ndarray, out_h: int, out_w: int | None = None) -> np.ndarray:
    out_w = out_h if out_w is None else out_w
    n, c, t, h, w = latents.shape
    if h < out_h or w < out_w:
        out_h, out_w = min(out_h, h), min(out_w, w)
    if h % out_h != 0 or w % out_w != 0:
        # Crop to a divisible window. Toy images are small; this keeps the code robust.
        h2 = (h // out_h) * out_h
        w2 = (w // out_w) * out_w
        latents = latents[:, :, :, :h2, :w2]
        h, w = h2, w2
    ph = h // out_h
    pw = w // out_w
    pooled = latents.reshape(n, c, t, out_h, ph, out_w, pw).mean(axis=(4, 6))
    return pooled.transpose(0, 2, 3, 4, 1).astype(np.float32)


def flatten_seq(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32)


def seq_global_from_latents(latents: np.ndarray) -> np.ndarray:
    return latents.mean(axis=(3, 4)).transpose(0, 2, 1).astype(np.float32)


def seq_global_from_videos(videos: np.ndarray) -> np.ndarray:
    return average_pool_video(videos, 1).reshape(videos.shape[0], videos.shape[1], -1).astype(np.float32)


def energy_stats(seq: np.ndarray) -> np.ndarray:
    if seq.shape[1] <= 1:
        return np.zeros((seq.shape[0], 8), dtype=np.float32)
    delta = np.diff(seq, axis=1)
    accel = np.diff(delta, axis=1) if delta.shape[1] > 1 else np.zeros_like(delta[:, :1])
    energy = np.linalg.norm(delta, axis=2)
    total = energy.sum(axis=1, keepdims=True)
    prob = energy / np.maximum(total, 1e-8)
    entropy = -(prob * np.log(np.maximum(prob, 1e-8))).sum(axis=1, keepdims=True) / math.log(max(2, energy.shape[1]))
    centered = energy - energy.mean(axis=1, keepdims=True)
    autocorr = []
    fft_peak = []
    for row in centered:
        denom = float((row * row).sum())
        if denom <= 1e-8 or len(row) < 3:
            autocorr.append(0.0)
        else:
            corr = np.correlate(row, row, mode="full")[len(row) - 1 :] / denom
            autocorr.append(float(np.max(corr[1:])) if len(corr) > 1 else 0.0)
        spec = np.abs(np.fft.rfft(row))
        fft_peak.append(float(np.max(spec[1:]) / max(np.sum(spec[1:]), 1e-8)) if len(spec) > 2 else 0.0)
    curvature = []
    if delta.shape[1] > 1:
        a = delta[:, :-1]
        b = delta[:, 1:]
        cos = (a * b).sum(axis=2) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-8)
        curvature = [cos.mean(axis=1, keepdims=True), cos.std(axis=1, keepdims=True)]
    else:
        curvature = [np.zeros((seq.shape[0], 1)), np.zeros((seq.shape[0], 1))]
    return np.concatenate(
        [
            energy.mean(axis=1, keepdims=True),
            energy.std(axis=1, keepdims=True),
            energy.max(axis=1, keepdims=True),
            np.linalg.norm(accel, axis=2).mean(axis=1, keepdims=True),
            np.asarray(autocorr, dtype=np.float32).reshape(-1, 1),
            np.asarray(fft_peak, dtype=np.float32).reshape(-1, 1),
            entropy.astype(np.float32),
            *[x.astype(np.float32) for x in curvature],
        ],
        axis=1,
    ).astype(np.float32)


def relation_descriptor(seq: np.ndarray) -> np.ndarray:
    n, t, d = seq.shape
    if t <= 1:
        return np.zeros((n, 8), dtype=np.float32)
    x = seq / np.maximum(np.linalg.norm(seq, axis=2, keepdims=True), 1e-8)
    feats = []
    for i in range(n):
        r = x[i] @ x[i].T
        tri = r[np.triu_indices(t, k=1)]
        near = np.diag(r, k=1)
        far = r[np.abs(np.subtract.outer(np.arange(t), np.arange(t))) >= max(2, t // 2)]
        vals = (tri + 1.0) / 2.0
        hist, _ = np.histogram(vals, bins=8, range=(0.0, 1.0), density=False)
        prob = hist.astype(np.float32) / max(1, hist.sum())
        entropy = float(-(prob * np.log(np.maximum(prob, 1e-8))).sum() / math.log(8))
        decay = []
        for lag in range(1, min(5, t)):
            decay.append(float(np.diag(r, k=lag).mean()))
        while len(decay) < 4:
            decay.append(0.0)
        feats.append(
            [
                float(tri.mean()),
                float(tri.std()),
                float(near.mean()) if near.size else 0.0,
                float(far.mean()) if far.size else 0.0,
                entropy,
                *decay[:4],
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def structured_compact(seq_grid: np.ndarray) -> np.ndarray:
    # seq_grid: [N,T,H,W,C]
    global_seq = seq_grid.mean(axis=(2, 3))
    feats = [energy_stats(global_seq)]
    h = seq_grid.shape[2]
    w = seq_grid.shape[3]
    if w >= 2:
        left = seq_grid[:, :, :, : w // 2].mean(axis=(2, 3))
        right = seq_grid[:, :, :, w // 2 :].mean(axis=(2, 3))
        feats.append(energy_stats(left) - energy_stats(right))
    if h >= 2:
        top = seq_grid[:, :, : h // 2].mean(axis=(2, 3))
        bottom = seq_grid[:, :, h // 2 :].mean(axis=(2, 3))
        feats.append(energy_stats(top) - energy_stats(bottom))
    if h >= 2 and w >= 2:
        quads = [
            seq_grid[:, :, : h // 2, : w // 2].mean(axis=(2, 3)),
            seq_grid[:, :, : h // 2, w // 2 :].mean(axis=(2, 3)),
            seq_grid[:, :, h // 2 :, : w // 2].mean(axis=(2, 3)),
            seq_grid[:, :, h // 2 :, w // 2 :].mean(axis=(2, 3)),
        ]
        quad_energy = [energy_stats(q) for q in quads]
        feats.append(np.concatenate(quad_energy, axis=1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def dynamics_features(seq: np.ndarray) -> dict[str, np.ndarray]:
    delta = np.diff(seq, axis=1) if seq.shape[1] > 1 else np.zeros_like(seq)
    accel = np.diff(delta, axis=1) if delta.shape[1] > 1 else np.zeros_like(delta)
    return {
        "delta": flatten_seq(delta),
        "accel": flatten_seq(accel),
        "energy_autocorr": energy_stats(seq),
        "relation": relation_descriptor(seq),
        "dynamics_all": np.concatenate([flatten_seq(delta), flatten_seq(accel), energy_stats(seq), relation_descriptor(seq)], axis=1),
    }


def pixel_feature_set(videos: np.ndarray) -> dict[str, np.ndarray]:
    seq1 = average_pool_video(videos, 1).reshape(videos.shape[0], videos.shape[1], -1)
    seq2 = average_pool_video(videos, 2).reshape(videos.shape[0], videos.shape[1], -1)
    seq4_grid = average_pool_video(videos, 4)
    seq4 = seq4_grid.reshape(videos.shape[0], videos.shape[1], -1)
    feats = {
        "pixel_raw_1x1": flatten_seq(seq1),
        "pixel_raw_2x2": flatten_seq(seq2),
        "pixel_raw_4x4": flatten_seq(seq4),
        "pixel_delta_2x2": flatten_seq(np.diff(seq2, axis=1)),
        "pixel_dynamics_2x2": dynamics_features(seq2)["dynamics_all"],
        "pixel_structured_compact": structured_compact(seq4_grid),
    }
    return feats


def wan_feature_set(latents: np.ndarray | None, videos: np.ndarray) -> dict[str, np.ndarray]:
    if latents is None:
        # Fallback makes smoke tests possible without Wan weights.
        feats = pixel_feature_set(videos)
        return {key.replace("pixel", "wan_proxy"): value for key, value in feats.items()}
    seq1 = pool_latents(latents, 1).reshape(latents.shape[0], latents.shape[2], -1)
    seq2_grid = pool_latents(latents, 2)
    seq2 = seq2_grid.reshape(latents.shape[0], latents.shape[2], -1)
    max_hw = min(4, latents.shape[3], latents.shape[4])
    seq4_grid = pool_latents(latents, max_hw)
    seq4 = seq4_grid.reshape(latents.shape[0], latents.shape[2], -1)
    dyn = dynamics_features(seq1)
    feats = {
        "wan_raw_1x1": flatten_seq(seq1),
        "wan_raw_2x2": flatten_seq(seq2),
        "wan_raw_4x4": flatten_seq(seq4),
        "wan_delta_1x1": dyn["delta"],
        "wan_accel_1x1": dyn["accel"],
        "wan_energy_autocorr": dyn["energy_autocorr"],
        "wan_relation": dyn["relation"],
        "wan_dynamics_1x1": dyn["dynamics_all"],
        "wan_raw_plus_dynamics": np.concatenate([flatten_seq(seq1), dyn["dynamics_all"]], axis=1),
        "wan_structured_compact": structured_compact(seq4_grid),
    }
    return feats


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int = 0):
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(x_train, y_train)
    return clf


def decision_scores(model, x: np.ndarray, n_classes: int) -> np.ndarray:
    scores = model.decision_function(x)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    if scores.shape[1] < n_classes:
        padded = np.zeros((scores.shape[0], n_classes), dtype=np.float32)
        classes = list(model.steps[-1][1].classes_)
        for j, cls in enumerate(classes):
            padded[:, int(cls)] = scores[:, j]
        return padded
    return scores


def evaluate_feature(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, labels: list[str]) -> dict[str, Any]:
    model = fit_probe(x_train, y_train)
    pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "feature_dim": int(x_train.shape[1]),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(len(labels)))).tolist(),
        "predictions": pred.astype(int).tolist(),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(np.maximum(p, 1e-8)) - np.log(np.maximum(m, 1e-8)))).sum(axis=1)
    kl_qm = (q * (np.log(np.maximum(q, 1e-8)) - np.log(np.maximum(m, 1e-8)))).sum(axis=1)
    return (0.5 * (kl_pm + kl_qm)).astype(np.float32)


def margins(scores: np.ndarray) -> np.ndarray:
    s = np.sort(scores, axis=1)
    return (s[:, -1] - s[:, -2]).astype(np.float32) if scores.shape[1] > 1 else np.zeros((scores.shape[0],), dtype=np.float32)


def wps_from_model(model, normal_x: np.ndarray, corrupt: dict[str, np.ndarray], n_classes: int) -> np.ndarray:
    normal_scores = decision_scores(model, normal_x, n_classes)
    normal_prob = softmax(normal_scores)
    normal_margin = margins(normal_scores)
    pieces = []
    for name in ["shuffle", "reverse", "timeavg", "lowfps", "first_repeat"]:
        if name not in corrupt:
            continue
        c_scores = decision_scores(model, corrupt[name], n_classes)
        c_prob = softmax(c_scores)
        pieces.append(normal_scores - c_scores)
        pieces.append((normal_margin - margins(c_scores)).reshape(-1, 1))
        pieces.append(js_divergence(normal_prob, c_prob).reshape(-1, 1))
        pieces.append((np.argmax(normal_scores, axis=1) != np.argmax(c_scores, axis=1)).astype(np.float32).reshape(-1, 1))
        dist = np.linalg.norm(normal_x - corrupt[name], axis=1, keepdims=True)
        pieces.append(dist.astype(np.float32))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def oof_switch_gate(
    base_train: np.ndarray,
    wan_train: np.ndarray,
    wps_train_by_fold: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    base_test: np.ndarray,
    wan_test: np.ndarray,
    wps_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    seed: int,
) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=min(4, np.bincount(y_train).min()), shuffle=True, random_state=seed)
    train_wps_parts = []
    target_parts = []
    for fit_idx, val_idx in skf.split(base_train, y_train):
        base_model = fit_probe(base_train[fit_idx], y_train[fit_idx], seed)
        wan_model = fit_probe(wan_train[fit_idx], y_train[fit_idx], seed + 1)
        wps_val, base_pred, wan_pred = wps_train_by_fold(fit_idx, val_idx)
        target = (wan_pred == y_train[val_idx]) & (base_pred != y_train[val_idx])
        train_wps_parts.append(wps_val)
        target_parts.append(target.astype(np.int64))
    x_gate = np.concatenate(train_wps_parts, axis=0)
    y_gate = np.concatenate(target_parts, axis=0)
    if len(np.unique(y_gate)) < 2:
        gate_pred = np.zeros((len(y_test),), dtype=bool)
        gate_score = np.zeros((len(y_test),), dtype=np.float32)
        gate_kind = "constant_zero"
    else:
        gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
        gate.fit(x_gate, y_gate)
        prob = gate.predict_proba(wps_test)
        pos = list(gate.steps[-1][1].classes_).index(1)
        gate_score = prob[:, pos]
        # Keep the gate selective as the document recommends.
        threshold = float(np.quantile(gate_score, 0.80))
        gate_pred = gate_score >= threshold
        gate_kind = "logistic_top20"

    base_model_full = fit_probe(base_train, y_train, seed)
    wan_model_full = fit_probe(wan_train, y_train, seed + 1)
    base_scores = decision_scores(base_model_full, base_test, len(labels))
    wan_scores = decision_scores(wan_model_full, wan_test, len(labels))
    base_pred = np.argmax(base_scores, axis=1)
    wan_pred = np.argmax(wan_scores, axis=1)
    ensemble_pred = np.argmax(base_scores + wan_scores, axis=1)
    switch_pred = np.where(gate_pred, wan_pred, base_pred)
    helps = int(((switch_pred == y_test) & (base_pred != y_test)).sum())
    hurts = int(((switch_pred != y_test) & (base_pred == y_test)).sum())
    gate_mask = gate_pred.astype(bool)
    return {
        "gate_kind": gate_kind,
        "gate_coverage": float(gate_mask.mean()),
        "gate_count": int(gate_mask.sum()),
        "base_accuracy": float(accuracy_score(y_test, base_pred)),
        "raw_wan_everywhere_accuracy": float(accuracy_score(y_test, wan_pred)),
        "base_raw_wan_score_ensemble_accuracy": float(accuracy_score(y_test, ensemble_pred)),
        "wps_gate_switch_accuracy": float(accuracy_score(y_test, switch_pred)),
        "gate1_base_accuracy": float(accuracy_score(y_test[gate_mask], base_pred[gate_mask])) if gate_mask.any() else 0.0,
        "gate1_wan_accuracy": float(accuracy_score(y_test[gate_mask], wan_pred[gate_mask])) if gate_mask.any() else 0.0,
        "gate0_base_accuracy": float(accuracy_score(y_test[~gate_mask], base_pred[~gate_mask])) if (~gate_mask).any() else 0.0,
        "gate0_wan_accuracy": float(accuracy_score(y_test[~gate_mask], wan_pred[~gate_mask])) if (~gate_mask).any() else 0.0,
        "switch_helps": helps,
        "switch_hurts": hurts,
        "gate_target_positive_train": int(y_gate.sum()),
        "gate_target_total_train": int(len(y_gate)),
        "gate_f1_train_oof_target_proxy": float(f1_score(y_gate, (np.concatenate([np.zeros(len(x)) for x in train_wps_parts]) > 0).astype(int), zero_division=0)),
    }


def prototype_alignment(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, labels: list[str]) -> dict[str, Any]:
    classes = list(range(len(labels)))
    proto = []
    for cls in classes:
        proto.append(x_train[y_train == cls].mean(axis=0))
    proto = np.stack(proto, axis=0)
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8)
    x = x_test / np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True), 1e-8)
    scores = x @ proto.T
    pred = np.argmax(scores, axis=1)
    return {"accuracy": float(accuracy_score(y_test, pred)), "balanced_accuracy": float(balanced_accuracy_score(y_test, pred))}


def future_consistency_proxy(seq_train: np.ndarray, y_train: np.ndarray, seq_test: np.ndarray, y_test: np.ndarray, labels: list[str]) -> dict[str, Any]:
    # Compare observed second-half dynamics with class prototypes from training.
    cut_train = max(1, seq_train.shape[1] // 2)
    cut_test = max(1, seq_test.shape[1] // 2)
    future_train = energy_stats(seq_train[:, cut_train:])
    future_test = energy_stats(seq_test[:, cut_test:])
    return prototype_alignment(future_train, y_train, future_test, y_test, labels)


def make_bundles(task: TaskData, vae, device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> dict[str, tuple[FeatureBundle, FeatureBundle]]:
    transforms = ["normal", "shuffle", "reverse", "timeavg", "lowfps", "first_repeat", "hflip"]
    out: dict[str, tuple[FeatureBundle, FeatureBundle]] = {}
    for tr in transforms:
        train_v = transform_videos(task.x_train, tr, args.seed + len(tr))
        test_v = transform_videos(task.x_test, tr, args.seed + 100 + len(tr))
        train_z = test_z = None
        if args.feature_source == "wan":
            train_z = encode_wan_vae(vae, train_v, args.batch_size, device, dtype)
            test_z = encode_wan_vae(vae, test_v, args.batch_size, device, dtype)
        out[tr] = (FeatureBundle(train_v, train_z), FeatureBundle(test_v, test_z))
    return out


def run_task(task_name: str, task: TaskData, vae, device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    bundles = make_bundles(task, vae, device, dtype, args)
    normal_train, normal_test = bundles["normal"]
    train_pixel = pixel_feature_set(normal_train.videos)
    test_pixel = pixel_feature_set(normal_test.videos)
    train_wan = wan_feature_set(normal_train.latents, normal_train.videos)
    test_wan = wan_feature_set(normal_test.latents, normal_test.videos)
    feature_results: dict[str, Any] = {}
    selected_features = {
        **{k: (train_pixel[k], test_pixel[k]) for k in train_pixel},
        **{k: (train_wan[k], test_wan[k]) for k in train_wan},
    }
    for name, (xtr, xte) in selected_features.items():
        feature_results[name] = evaluate_feature(xtr, task.y_train, xte, task.y_test, task.labels)

    # Score contrast and WPS from raw Wan 1x1. Base is a strong non-Wan temporal feature.
    base_feature = "pixel_dynamics_2x2"
    raw_wan_feature = "wan_raw_1x1" if "wan_raw_1x1" in train_wan else next(iter(train_wan))
    raw_model = fit_probe(train_wan[raw_wan_feature], task.y_train)
    corrupt_train_features = {}
    corrupt_test_features = {}
    for tr in ["shuffle", "reverse", "timeavg", "lowfps", "first_repeat"]:
        tr_train, tr_test = bundles[tr]
        tr_wan_train = wan_feature_set(tr_train.latents, tr_train.videos)[raw_wan_feature]
        tr_wan_test = wan_feature_set(tr_test.latents, tr_test.videos)[raw_wan_feature]
        corrupt_train_features[tr] = tr_wan_train
        corrupt_test_features[tr] = tr_wan_test

    wps_train_full = wps_from_model(raw_model, train_wan[raw_wan_feature], corrupt_train_features, len(task.labels))
    wps_test = wps_from_model(raw_model, test_wan[raw_wan_feature], corrupt_test_features, len(task.labels))
    feature_results["wan_wps_signature"] = evaluate_feature(wps_train_full, task.y_train, wps_test, task.y_test, task.labels)
    feature_results["wan_raw_plus_wps"] = evaluate_feature(
        np.concatenate([train_wan[raw_wan_feature], wps_train_full], axis=1),
        task.y_train,
        np.concatenate([test_wan[raw_wan_feature], wps_test], axis=1),
        task.y_test,
        task.labels,
    )

    score_contrast_results = {}
    normal_scores_test = decision_scores(raw_model, test_wan[raw_wan_feature], len(task.labels))
    for tr, xte in corrupt_test_features.items():
        c_scores = decision_scores(raw_model, xte, len(task.labels))
        pred = np.argmax(normal_scores_test - c_scores, axis=1)
        score_contrast_results[tr] = {
            "accuracy": float(accuracy_score(task.y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(task.y_test, pred)),
        }

    def wps_train_fold(fit_idx: np.ndarray, val_idx: np.ndarray):
        model = fit_probe(train_wan[raw_wan_feature][fit_idx], task.y_train[fit_idx])
        fold_corrupt = {tr: feat[val_idx] for tr, feat in corrupt_train_features.items()}
        wps_val = wps_from_model(model, train_wan[raw_wan_feature][val_idx], fold_corrupt, len(task.labels))
        base_m = fit_probe(train_pixel[base_feature][fit_idx], task.y_train[fit_idx])
        wan_m = model
        base_pred = base_m.predict(train_pixel[base_feature][val_idx])
        wan_pred = wan_m.predict(train_wan[raw_wan_feature][val_idx])
        return wps_val, base_pred, wan_pred

    wps_gate = oof_switch_gate(
        train_pixel[base_feature],
        train_wan[raw_wan_feature],
        wps_train_fold,
        task.y_train,
        test_pixel[base_feature],
        test_wan[raw_wan_feature],
        wps_test,
        task.y_test,
        task.labels,
        args.seed,
    )

    seq_train = seq_global_from_latents(normal_train.latents) if normal_train.latents is not None else seq_global_from_videos(normal_train.videos)
    seq_test = seq_global_from_latents(normal_test.latents) if normal_test.latents is not None else seq_global_from_videos(normal_test.videos)
    cfg_proxy = prototype_alignment(dynamics_features(seq_train)["delta"], task.y_train, dynamics_features(seq_test)["delta"], task.y_test, task.labels)
    future_proxy = future_consistency_proxy(seq_train, task.y_train, seq_test, task.y_test, task.labels)

    temporal_gaps = {}
    normal_raw = feature_results[raw_wan_feature]["balanced_accuracy"]
    for tr in ["shuffle", "reverse", "timeavg", "lowfps", "first_repeat"]:
        tr_train, tr_test = bundles[tr]
        tr_wan_train = wan_feature_set(tr_train.latents, tr_train.videos)[raw_wan_feature]
        tr_wan_test = wan_feature_set(tr_test.latents, tr_test.videos)[raw_wan_feature]
        res = evaluate_feature(tr_wan_train, task.y_train, tr_wan_test, task.y_test, task.labels)
        temporal_gaps[tr] = {
            "control_balanced_accuracy": res["balanced_accuracy"],
            "normal_minus_control": float(normal_raw - res["balanced_accuracy"]),
        }

    return {
        "labels": task.labels,
        "task_type": task.task_type,
        "n_train": int(len(task.y_train)),
        "n_test": int(len(task.y_test)),
        "feature_results": feature_results,
        "score_contrast": score_contrast_results,
        "wps_gate": wps_gate,
        "cfg_alignment_proxy": cfg_proxy,
        "future_consistency_proxy": future_proxy,
        "temporal_gaps": temporal_gaps,
        "seconds": round(time.time() - t0, 2),
    }


def segment_localization_toy(rng: np.random.Generator, vae, device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> dict[str, Any]:
    n = args.segment_videos
    frames = args.segment_frames
    segments = args.segment_count
    height = args.height
    width = args.width
    videos = []
    targets = []
    for _ in range(n):
        target = int(rng.integers(1, segments - 1))
        distractors = [int((target + 2) % segments), int((target + 4) % segments)]
        video = background(rng, frames, height, width)
        seg_len = frames // segments
        for seg in range(segments):
            start = seg * seg_len
            end = frames if seg == segments - 1 else (seg + 1) * seg_len
            if seg == target:
                # Relevant bounce/count event.
                snippet = repetition_video(rng, 3, end - start, height, width)
                video[start:end] = from_model_range(snippet)
            elif seg in distractors:
                snippet = linear_motion_video(rng, int(rng.integers(0, 4)), end - start, height, width)
                video[start:end] = from_model_range(snippet)
        videos.append(to_model_range(video))
        targets.append(target)
    videos_np = np.stack(videos)
    targets_np = np.asarray(targets, dtype=int)

    pixel_grid = average_pool_video(videos_np, 4).reshape(n, frames, -1)
    pixel_energy = np.linalg.norm(np.diff(pixel_grid, axis=1, prepend=pixel_grid[:, :1]), axis=2)
    pixel_seg = segment_mean(pixel_energy, segments)
    flow_seg = pixel_seg.copy()
    if args.feature_source == "wan":
        latents = encode_wan_vae(vae, videos_np, args.batch_size, device, dtype)
        seq = seq_global_from_latents(latents)
    else:
        seq = seq_global_from_videos(videos_np)
    wan_energy = np.linalg.norm(np.diff(seq, axis=1, prepend=seq[:, :1]), axis=2)
    wan_seg = segment_mean(wan_energy, segments)
    # Temporal sensitivity proxy: distance from local time-average in each segment.
    sens = wan_seg / np.maximum(wan_seg.sum(axis=1, keepdims=True), 1e-8)
    uniform = np.ones_like(wan_seg)

    def recall(scores: np.ndarray, k: int) -> float:
        top = np.argsort(scores, axis=1)[:, -k:]
        return float(np.mean([targets_np[i] in top[i] for i in range(n)]))

    return {
        "n": int(n),
        "segments": int(segments),
        "top1_recall": {
            "uniform": recall(uniform, 1),
            "pixel_motion": recall(pixel_seg, 1),
            "flow_motion": recall(flow_seg, 1),
            "wan_temporal_sensitivity": recall(sens, 1),
        },
        "top3_recall": {
            "uniform": recall(uniform, min(3, segments)),
            "pixel_motion": recall(pixel_seg, min(3, segments)),
            "flow_motion": recall(flow_seg, min(3, segments)),
            "wan_temporal_sensitivity": recall(sens, min(3, segments)),
        },
    }


def segment_mean(values: np.ndarray, segments: int) -> np.ndarray:
    n, t = values.shape
    out = np.zeros((n, segments), dtype=np.float32)
    bounds = np.linspace(0, t, segments + 1).round().astype(int)
    for i in range(segments):
        start, end = int(bounds[i]), int(bounds[i + 1])
        if end <= start:
            end = min(t, start + 1)
        out[:, i] = values[:, start:end].mean(axis=1)
    return out


def summarize_results(payload: dict[str, Any]) -> str:
    lines = ["# Wan Feature Alternatives Toy Experiments", ""]
    lines.append("## Feature Probe Summary")
    lines.append("")
    lines.append("| Task | Best feature | Best bal acc | raw Wan 1x1 | Wan dynamics | Structured compact | WPS | raw+WPS | Pixel dynamics |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for task, result in payload["tasks"].items():
        features = result["feature_results"]
        best_name, best_res = max(features.items(), key=lambda kv: kv[1]["balanced_accuracy"])
        def acc(name: str) -> float:
            return features.get(name, {}).get("balanced_accuracy", float("nan"))
        lines.append(
            f"| {task} | {best_name} | {best_res['balanced_accuracy']:.4f} | "
            f"{acc('wan_raw_1x1'):.4f} | {acc('wan_dynamics_1x1'):.4f} | {acc('wan_structured_compact'):.4f} | "
            f"{acc('wan_wps_signature'):.4f} | {acc('wan_raw_plus_wps'):.4f} | {acc('pixel_dynamics_2x2'):.4f} |"
        )
    lines.append("")
    lines.append("## WPS Gate")
    lines.append("")
    lines.append("| Task | Coverage | Base | Raw Wan everywhere | Score ensemble | WPS gate switch | Helps | Hurts | Gate=1 base | Gate=1 Wan |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for task, result in payload["tasks"].items():
        g = result["wps_gate"]
        lines.append(
            f"| {task} | {g['gate_coverage']:.4f} | {g['base_accuracy']:.4f} | {g['raw_wan_everywhere_accuracy']:.4f} | "
            f"{g['base_raw_wan_score_ensemble_accuracy']:.4f} | {g['wps_gate_switch_accuracy']:.4f} | "
            f"{g['switch_helps']} | {g['switch_hurts']} | {g['gate1_base_accuracy']:.4f} | {g['gate1_wan_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Score Contrast")
    lines.append("")
    lines.append("| Task | shuffle | reverse | timeavg | lowfps | first_repeat |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task, result in payload["tasks"].items():
        sc = result["score_contrast"]
        lines.append(
            f"| {task} | {sc['shuffle']['balanced_accuracy']:.4f} | {sc['reverse']['balanced_accuracy']:.4f} | "
            f"{sc['timeavg']['balanced_accuracy']:.4f} | {sc['lowfps']['balanced_accuracy']:.4f} | {sc['first_repeat']['balanced_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## CFG / Future Consistency Proxies")
    lines.append("")
    lines.append("| Task | CFG alignment proxy | Future consistency proxy |")
    lines.append("|---|---:|---:|")
    for task, result in payload["tasks"].items():
        lines.append(
            f"| {task} | {result['cfg_alignment_proxy']['balanced_accuracy']:.4f} | "
            f"{result['future_consistency_proxy']['balanced_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Segment Localization")
    lines.append("")
    seg = payload["segment_localization"]
    lines.append("| Selector | Top-1 recall | Top-3 recall |")
    lines.append("|---|---:|---:|")
    for name in seg["top1_recall"]:
        lines.append(f"| {name} | {seg['top1_recall'][name]:.4f} | {seg['top3_recall'][name]:.4f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- CFG and future-consistency entries are feature-space toy proxies, not Wan generation runs.")
    lines.append("- All Wan rows use real Wan-VAE features when `feature_source=wan`.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="results/wan_feature_alternatives_toy_results.json")
    parser.add_argument("--output-md", default="results/wan_feature_alternatives_toy_results.md")
    parser.add_argument("--tasks", default="direction4,action_order,repetition_count,same_first_last_path,contact_causality,camera_object_motion,shuffle_detection,rhythm")
    parser.add_argument("--feature-source", default="wan", choices=["wan", "pixel-proxy"])
    parser.add_argument("--train-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=8)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-videos", type=int, default=48)
    parser.add_argument("--segment-frames", type=int, default=33)
    parser.add_argument("--segment-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.feature_source == "wan" else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = None
    if args.feature_source == "wan":
        vae = load_vae(device, dtype)
    task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    payload: dict[str, Any] = {
        "config": vars(args),
        "model_id": MODEL_ID if args.feature_source == "wan" else None,
        "tasks": {},
    }
    for offset, task_name in enumerate(task_names):
        task_rng = np.random.default_rng(args.seed + 1000 * offset)
        task = make_task(task_rng, task_name, args.train_per_class, args.test_per_class, args.frames, args.height, args.width)
        payload["tasks"][task_name] = run_task(task_name, task, vae, device, dtype, args)
        print(json.dumps({"task": task_name, "seconds": payload["tasks"][task_name]["seconds"]}, ensure_ascii=False))
    payload["segment_localization"] = segment_localization_toy(rng, vae, device, dtype, args)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(summarize_results(payload), encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_md": args.output_md, "tasks": len(payload["tasks"])}, indent=2))


if __name__ == "__main__":
    main()
