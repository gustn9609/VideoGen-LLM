#!/usr/bin/env python3
"""Dataset-facing Wan feature extractor.

Input JSONL rows should contain at least:

{"video_id": "...", "path": "/path/to/video.mp4"}

For smoke tests and synthetic pipelines, `path` can also point to `.npy` files
with shape [T, H, W, 3] in either [0, 1], [-1, 1], or uint8 [0, 255].
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_feature_sanity import MODEL_ID, encode_wan_vae, load_vae  # noqa: E402
from wan_next_experiments import encode_dit_features, vae_pooling_features  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_video_array(video: np.ndarray) -> np.ndarray:
    video = np.asarray(video)
    if video.ndim != 4 or video.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Expected video [T,H,W,C], got {video.shape}")
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    if video.shape[-1] == 4:
        video = video[..., :3]
    if video.dtype == np.uint8:
        video = video.astype(np.float32) / 255.0
    else:
        video = video.astype(np.float32)
        if video.min() < -0.1:
            video = (video + 1.0) / 2.0
        if video.max() > 1.5:
            video = video / 255.0
    return np.clip(video, 0.0, 1.0)


def resize_frames(video: np.ndarray, height: int, width: int) -> np.ndarray:
    frames = []
    for frame in video:
        image = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        frames.append(np.asarray(image).astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def center_crop_video(video: np.ndarray, crop_fraction: float = 0.875) -> tuple[np.ndarray, dict[str, Any]]:
    _, height, width, _ = video.shape
    crop_h = max(2, int(round(height * crop_fraction)))
    crop_w = max(2, int(round(width * crop_fraction)))
    y = max(0, (height - crop_h) // 2)
    x = max(0, (width - crop_w) // 2)
    return video[:, y : y + crop_h, x : x + crop_w], {
        "crop_fraction": crop_fraction,
        "crop_size": [crop_h, crop_w],
        "offset": [int(y), int(x)],
    }


def mode_parts(mode: str) -> list[str]:
    return [part.strip().lower() for part in str(mode).split("+") if part.strip()]


def grayscale(video: np.ndarray) -> np.ndarray:
    return (0.299 * video[..., 0] + 0.587 * video[..., 1] + 0.114 * video[..., 2]).astype(np.float32)


def motion_energy(video: np.ndarray) -> np.ndarray:
    gray = grayscale(video)
    if gray.shape[0] <= 1:
        return np.zeros((gray.shape[0],), dtype=np.float32)
    diff = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[float(diff[0])], diff.astype(np.float32)], axis=0)


def high_motion_indices(video: np.ndarray, num_frames: int, seed: int) -> np.ndarray:
    total_frames = int(video.shape[0])
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    if total_frames <= num_frames:
        return sample_indices(total_frames, num_frames, "none", seed)

    energy = motion_energy(video)
    window = min(total_frames, max(2, int(round(total_frames * 0.5)), num_frames))
    if window >= total_frames:
        start = 0
    else:
        kernel = np.ones((window,), dtype=np.float32)
        scores = np.convolve(energy, kernel, mode="valid")
        start = int(np.argmax(scores))
    end = min(total_frames - 1, start + window - 1)
    return np.linspace(start, end, num_frames).round().clip(0, total_frames - 1).astype(np.int64)


def bbox_from_mask(mask: np.ndarray, pad_fraction: float = 0.15) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    height, width = mask.shape
    pad_y = int(round((y1 - y0) * pad_fraction))
    pad_x = int(round((x1 - x0) * pad_fraction))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(height, y1 + pad_y)
    x1 = min(width, x1 + pad_x)
    return y0, y1, x0, x1


def object_crop_video(video: np.ndarray, min_fraction: float = 0.5) -> tuple[np.ndarray, dict[str, Any]]:
    gray = grayscale(video)
    temporal_std = gray.std(axis=0)
    mean_frame = gray.mean(axis=0)
    dy = np.abs(np.diff(mean_frame, axis=0, prepend=mean_frame[:1]))
    dx = np.abs(np.diff(mean_frame, axis=1, prepend=mean_frame[:, :1]))
    saliency = temporal_std + 0.25 * (dx + dy)
    threshold = float(np.quantile(saliency, 0.80))
    bbox = bbox_from_mask(saliency >= threshold)
    _, height, width, _ = video.shape
    if bbox is None:
        cropped, meta = center_crop_video(video, crop_fraction=0.875)
        meta["detector"] = "motion_saliency_fallback_center"
        return cropped, meta

    y0, y1, x0, x1 = bbox
    min_h = max(2, int(round(height * min_fraction)))
    min_w = max(2, int(round(width * min_fraction)))
    box_h = max(y1 - y0, min_h)
    box_w = max(x1 - x0, min_w)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = int(np.clip(cy - box_h // 2, 0, max(0, height - box_h)))
    x0 = int(np.clip(cx - box_w // 2, 0, max(0, width - box_w)))
    y1 = min(height, y0 + box_h)
    x1 = min(width, x0 + box_w)
    return video[:, y0:y1, x0:x1], {
        "detector": "motion_saliency",
        "bbox_yxyx": [int(y0), int(y1), int(x0), int(x1)],
        "crop_size": [int(y1 - y0), int(x1 - x0)],
    }


def downsample_for_shift(image: np.ndarray, max_size: int = 64) -> tuple[np.ndarray, int]:
    height, width = image.shape
    step = max(1, int(math.ceil(max(height, width) / max_size)))
    return image[::step, ::step].astype(np.float32), step


def estimate_integer_shift(reference: np.ndarray, moving: np.ndarray, max_shift: int = 8) -> tuple[int, int]:
    ref, step = downsample_for_shift(reference, max_size=64)
    mov, _ = downsample_for_shift(moving, max_size=64)
    ref = ref - float(ref.mean())
    mov = mov - float(mov.mean())
    height, width = ref.shape
    if height < 2 or width < 2:
        return 0, 0
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    ref_fft = np.fft.fft2(ref * window)
    mov_fft = np.fft.fft2(mov * window)
    cross = ref_fft * np.conj(mov_fft)
    cross /= np.maximum(np.abs(cross), 1e-8)
    corr = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if peak_y > height // 2:
        peak_y -= height
    if peak_x > width // 2:
        peak_x -= width
    dy = int(np.clip(round(peak_y * step), -max_shift, max_shift))
    dx = int(np.clip(round(peak_x * step), -max_shift, max_shift))
    return dy, dx


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(frame)
    if dy >= 0:
        src_y = slice(None, -dy or None)
        dst_y = slice(dy, None)
    else:
        src_y = slice(-dy, None)
        dst_y = slice(None, dy)
    if dx >= 0:
        src_x = slice(None, -dx or None)
        dst_x = slice(dx, None)
    else:
        src_x = slice(-dx, None)
        dst_x = slice(None, dx)
    out[dst_y, dst_x] = frame[src_y, src_x]
    return out


def camera_compensate_video(video: np.ndarray, max_shift: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    if video.shape[0] <= 1:
        return video, {"method": "integer_translation", "shifts": [[0, 0]]}
    gray = grayscale(video)
    compensated = [video[0]]
    shifts = [[0, 0]]
    cumulative_y = 0
    cumulative_x = 0
    reference = gray[0]
    for frame_idx in range(1, video.shape[0]):
        dy, dx = estimate_integer_shift(reference, gray[frame_idx], max_shift=max_shift)
        cumulative_y += dy
        cumulative_x += dx
        compensated.append(shift_frame(video[frame_idx], cumulative_y, cumulative_x))
        shifts.append([int(cumulative_y), int(cumulative_x)])
    return np.stack(compensated, axis=0), {"method": "integer_translation", "shifts": shifts, "max_shift": max_shift}


def sample_indices(total_frames: int, num_frames: int, mode: str, seed: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Video has no frames")
    if num_frames <= 1:
        return np.array([0], dtype=np.int64)

    parts = mode_parts(mode) or ["none"]
    sampling_part = "none"
    for candidate in parts:
        if (
            candidate == "none"
            or candidate.startswith("uniform")
            or candidate.startswith("nonuniform")
            or candidate.startswith("speed")
            or candidate.startswith("repeat")
        ):
            sampling_part = candidate
            break

    rng = np.random.default_rng(seed)
    if sampling_part == "none":
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)
    elif sampling_part.startswith("uniform"):
        keyframes = int(sampling_part.replace("uniform", "") or "5")
        key_idx = np.linspace(0, total_frames - 1, keyframes).round().astype(np.int64)
        repeat_idx = np.linspace(0, keyframes - 1, num_frames).round().astype(np.int64)
        indices = key_idx[repeat_idx]
    elif sampling_part.startswith("nonuniform"):
        keyframes = int(sampling_part.replace("nonuniform", "") or "5")
        if keyframes >= total_frames:
            key_idx = np.arange(total_frames)
        else:
            middle = rng.choice(np.arange(1, max(total_frames - 1, 2)), size=max(keyframes - 2, 0), replace=False)
            key_idx = np.sort(np.concatenate([[0], middle, [total_frames - 1]])).astype(np.int64)
        repeat_idx = np.linspace(0, len(key_idx) - 1, num_frames).round().astype(np.int64)
        indices = key_idx[repeat_idx]
    elif sampling_part.startswith("speed"):
        factor_text = sampling_part.replace("speed", "") or "1.0"
        factor = max(0.1, float(factor_text))
        if factor < 1.0:
            positions = np.linspace(0, (total_frames - 1) * factor, num_frames)
        else:
            active = max(2, int(math.ceil(num_frames / factor)))
            active_positions = np.linspace(0, total_frames - 1, active)
            pad = np.full(max(0, num_frames - active), total_frames - 1, dtype=np.float32)
            positions = np.concatenate([active_positions, pad], axis=0)[:num_frames]
        indices = positions.round().clip(0, total_frames - 1).astype(np.int64)
    elif sampling_part.startswith("repeat"):
        keep_text = sampling_part.replace("repeat", "")
        keyframes = int(keep_text or str(max(2, num_frames // 2)))
        keyframes = max(2, min(keyframes, total_frames, num_frames))
        key_idx = np.linspace(0, total_frames - 1, keyframes).round().astype(np.int64)
        repeat_idx = np.linspace(0, keyframes - 1, num_frames).round().astype(np.int64)
        indices = key_idx[repeat_idx]
        if num_frames > keyframes:
            repeat_locs = rng.choice(np.arange(num_frames), size=num_frames - keyframes, replace=True)
            indices[repeat_locs] = indices[np.maximum(0, repeat_locs - 1)]
    else:
        raise ValueError(f"Unknown lowfps mode: {mode}")

    if "reverse" in parts:
        indices = indices[::-1].copy()
    if "shuffle" in parts:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        indices = shuffled
    return indices.astype(np.int64)


def sampled_timestamps(indices: np.ndarray, fps: float | None) -> list[float | None]:
    if fps is None or fps <= 0:
        return [None for _ in indices.tolist()]
    return [float(index) / float(fps) for index in indices.tolist()]


def temporal_blur(video: np.ndarray) -> np.ndarray:
    if video.shape[0] < 3:
        return video
    blurred = video.copy()
    blurred[1:-1] = 0.25 * video[:-2] + 0.5 * video[1:-1] + 0.25 * video[2:]
    return blurred.astype(np.float32)


def jpeg_compress(video: np.ndarray, quality: int) -> np.ndarray:
    quality = int(max(5, min(100, quality)))
    output = []
    for frame in video:
        image = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        output.append(np.asarray(Image.open(buffer).convert("RGB")).astype(np.float32) / 255.0)
    return np.stack(output, axis=0)


def crop_shift(video: np.ndarray, seed: int, crop_fraction: float = 0.875) -> tuple[np.ndarray, dict[str, Any]]:
    if video.ndim != 4:
        raise ValueError(f"Expected video [T,H,W,C], got {video.shape}")
    rng = np.random.default_rng(seed)
    _, height, width, _ = video.shape
    crop_h = max(2, int(round(height * crop_fraction)))
    crop_w = max(2, int(round(width * crop_fraction)))
    max_y = max(0, height - crop_h)
    max_x = max(0, width - crop_w)
    start_y = int(rng.integers(0, max_y + 1)) if max_y else 0
    end_y = int(rng.integers(0, max_y + 1)) if max_y else 0
    start_x = int(rng.integers(0, max_x + 1)) if max_x else 0
    end_x = int(rng.integers(0, max_x + 1)) if max_x else 0

    crops = []
    offsets: list[list[int]] = []
    denom = max(1, video.shape[0] - 1)
    for frame_index, frame in enumerate(video):
        alpha = frame_index / denom
        y = int(round((1.0 - alpha) * start_y + alpha * end_y))
        x = int(round((1.0 - alpha) * start_x + alpha * end_x))
        crops.append(frame[y : y + crop_h, x : x + crop_w])
        offsets.append([y, x])

    return np.stack(crops, axis=0), {
        "crop_fraction": crop_fraction,
        "crop_size": [crop_h, crop_w],
        "offsets": offsets,
    }


def apply_mode_transforms(video: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    transform_meta: dict[str, Any] = {"transforms": []}
    transformed = video
    for part in mode_parts(mode):
        if part in {"none", "reverse", "shuffle", "high_motion"}:
            continue
        if part.startswith("uniform") or part.startswith("nonuniform") or part.startswith("speed") or part.startswith("repeat"):
            continue
        if part == "blur":
            transformed = temporal_blur(transformed)
            transform_meta["transforms"].append({"name": "temporal_blur"})
        elif part.startswith("jpeg"):
            quality = int(part.replace("jpeg", "") or "55")
            transformed = jpeg_compress(transformed, quality)
            transform_meta["transforms"].append({"name": "jpeg", "quality": quality})
        elif part == "crop_shift":
            transformed, crop_meta = crop_shift(transformed, seed=seed)
            transform_meta["transforms"].append({"name": "crop_shift", **crop_meta})
        elif part == "center_crop":
            transformed, crop_meta = center_crop_video(transformed)
            transform_meta["transforms"].append({"name": "center_crop", **crop_meta})
        elif part in {"object_crop", "detected_object_crop"}:
            transformed, crop_meta = object_crop_video(transformed)
            transform_meta["transforms"].append({"name": "object_crop", **crop_meta})
        elif part in {"camera_comp", "camera_compensated"}:
            transformed, comp_meta = camera_compensate_video(transformed)
            transform_meta["transforms"].append({"name": "camera_comp", **comp_meta})
        else:
            raise ValueError(f"Unknown lowfps transform: {part}")
    return transformed, transform_meta


def load_video(
    path: Path,
    num_frames: int,
    height: int,
    width: int,
    mode: str,
    seed: int,
    fps: float | None,
    start: float | None = None,
    end: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float | None], dict[str, Any]]:
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
        video = normalize_video_array(raw)
        if fps is not None and fps > 0 and (start is not None or end is not None):
            start_frame = int(max(0, round(float(start or 0.0) * fps)))
            end_frame = int(round(float(end) * fps)) if end is not None else int(video.shape[0])
            end_frame = max(start_frame + 1, min(int(video.shape[0]), end_frame))
            video = video[start_frame:end_frame]
        original_frame_count = int(video.shape[0])
        if "high_motion" in mode_parts(mode):
            indices = high_motion_indices(video, num_frames, seed)
        else:
            indices = sample_indices(video.shape[0], num_frames, mode, seed)
        sampled = video[indices]
    else:
        try:
            vr = VideoReader(str(path), ctx=cpu(0), num_threads=4)
            total = int(len(vr))
            if total <= 0:
                raise ValueError(f"Video has no frames: {path}")
            parts = mode_parts(mode)
            if "high_motion" in parts and total > num_frames:
                # Avoid decoding long source clips in full.  A sparse preview is
                # sufficient to choose a high-motion half-window, then only the
                # final sampled frames are decoded at full resolution.
                preview_count = min(total, max(16, num_frames * 2))
                preview_indices = np.linspace(0, total - 1, preview_count).round().astype(np.int64)
                preview = normalize_video_array(vr.get_batch(preview_indices).asnumpy())
                energy = motion_energy(preview)
                window = min(preview_count, max(2, int(round(preview_count * 0.5)), num_frames))
                if window >= preview_count:
                    start_pos = 0
                else:
                    scores = np.convolve(energy, np.ones((window,), dtype=np.float32), mode="valid")
                    start_pos = int(np.argmax(scores))
                end_pos = min(preview_count - 1, start_pos + window - 1)
                raw_start = int(preview_indices[start_pos])
                raw_end = int(preview_indices[end_pos])
                indices = np.linspace(raw_start, raw_end, num_frames).round().clip(0, total - 1).astype(np.int64)
            else:
                indices = sample_indices(total, num_frames, mode, seed)
            sampled = normalize_video_array(vr.get_batch(indices).asnumpy())
            original_frame_count = total
        except Exception:
            raw = iio.imread(path)
            video = normalize_video_array(raw)
            if fps is not None and fps > 0 and (start is not None or end is not None):
                start_frame = int(max(0, round(float(start or 0.0) * fps)))
                end_frame = int(round(float(end) * fps)) if end is not None else int(video.shape[0])
                end_frame = max(start_frame + 1, min(int(video.shape[0]), end_frame))
                video = video[start_frame:end_frame]
            original_frame_count = int(video.shape[0])
            if "high_motion" in mode_parts(mode):
                indices = high_motion_indices(video, num_frames, seed)
            else:
                indices = sample_indices(video.shape[0], num_frames, mode, seed)
            sampled = video[indices]
    sampled, transform_meta = apply_mode_transforms(sampled, mode, seed)
    resized = resize_frames(sampled, height, width)
    transform_meta["original_frame_count"] = original_frame_count
    return (resized * 2.0 - 1.0).astype(np.float32), indices, sampled_timestamps(indices, fps), transform_meta


def pixel_grid_sequence(videos: np.ndarray, out_hw: int) -> np.ndarray:
    n, frames, height, width, channels = videos.shape
    ph = height // out_hw
    pw = width // out_hw
    cropped = videos[:, :, : out_hw * ph, : out_hw * pw, :]
    pooled = cropped.reshape(n, frames, out_hw, ph, out_hw, pw, channels).mean(axis=(3, 5))
    return pooled.astype(np.float32)


def flow_grid_sequence(videos: np.ndarray, out_hw: int) -> np.ndarray:
    grid = pixel_grid_sequence(videos, out_hw)
    delta = np.diff(grid, axis=1, prepend=grid[:, :1])
    return delta.astype(np.float32)


def parse_wan_grid_pool(name: str) -> int | None:
    prefix = "wan_vae_grid_"
    suffix = "x"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    if not rest.endswith(suffix + rest.split(suffix)[-1]):
        return None
    if "x" not in rest:
        return None
    left, right = rest.split("x", 1)
    if not left.isdigit() or not right.isdigit() or left != right:
        return None
    value = int(left)
    if value not in {1, 2, 4, 8, 16}:
        raise ValueError(f"Unsupported Wan grid resolution in feature name: {name}")
    return value


def pool_wan_grid(latents: np.ndarray, grid_hw: int) -> np.ndarray:
    n, channels, frames, height, width = latents.shape
    if grid_hw == height and grid_hw == width:
        pooled = latents
    else:
        if height % grid_hw != 0 or width % grid_hw != 0:
            raise ValueError(f"Cannot pool latent grid {height}x{width} to {grid_hw}x{grid_hw}")
        ph = height // grid_hw
        pw = width // grid_hw
        pooled = latents.reshape(n, channels, frames, grid_hw, ph, grid_hw, pw).mean(axis=(4, 6))
    return pooled.transpose(0, 2, 3, 4, 1).astype(np.float32)


def make_feature_arrays(
    videos: np.ndarray,
    feature_types: list[str],
    vae,
    transformer,
    scheduler,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], list[int] | None]:
    out: dict[str, np.ndarray] = {}
    latent_shape = None
    if "pixel_grid_sequence" in feature_types:
        out["pixel_grid_sequence"] = pixel_grid_sequence(videos, args.pixel_grid_hw)
    if "flow_grid_sequence" in feature_types:
        out["flow_grid_sequence"] = flow_grid_sequence(videos, args.pixel_grid_hw)

    wan_needed = any(name.startswith("wan_vae_") or name.startswith("dit_") for name in feature_types)
    latents = None
    if wan_needed:
        latents = encode_wan_vae(vae, videos, args.batch_size, device, dtype)
        latent_shape = [int(x) for x in latents.shape[1:]]
        if "wan_vae_grid_sequence" in feature_types:
            out["wan_vae_grid_sequence"] = latents.transpose(0, 2, 3, 4, 1).astype(np.float32)
        for feature_type in feature_types:
            grid_hw = parse_wan_grid_pool(feature_type)
            if grid_hw is not None:
                out[feature_type] = pool_wan_grid(latents, grid_hw)
        if "wan_vae_global_sequence" in feature_types:
            out["wan_vae_global_sequence"] = vae_pooling_features(latents, "wan_vae_global_sequence").astype(np.float32)
        if "wan_vae_global_delta" in feature_types:
            out["wan_vae_global_delta"] = vae_pooling_features(latents, "wan_vae_global_delta").astype(np.float32)

    dit_requested = [name for name in feature_types if name.startswith("dit_l14_t900_")]
    if dit_requested:
        if transformer is None or scheduler is None:
            raise ValueError("DiT feature requested but transformer/scheduler were not loaded")
        pooling_kinds = []
        for name in dit_requested:
            pooling = name.replace("dit_l14_t900_", "")
            pooling_kinds.append(pooling)
        dit = encode_dit_features(
            transformer,
            scheduler,
            latents,
            [14],
            [900],
            sorted(set(pooling_kinds)),
            args.batch_size,
            device,
            dtype,
            args.seed + 1000,
        )
        for name in dit_requested:
            out[name] = dit[name].astype(np.float32)
    return out, latent_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument(
        "--feature-types",
        default="wan_vae_grid_sequence,wan_vae_global_sequence,wan_vae_global_delta,dit_l14_t900_token_mean,pixel_grid_sequence,flow_grid_sequence",
    )
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--pixel-grid-hw", type=int, default=16)
    parser.add_argument("--lowfps-modes", default="none")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_types = [x.strip() for x in args.feature_types.split(",") if x.strip()]
    modes = [x.strip() for x in args.lowfps_modes.split(",") if x.strip()]
    rows = read_jsonl(Path(args.video_jsonl))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = None
    transformer = None
    scheduler = None
    if any(name.startswith("wan_vae_") or name.startswith("dit_") for name in feature_types):
        vae = load_vae(device, dtype)
    if any(name.startswith("dit_") for name in feature_types):
        from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

        transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype)
        transformer.to(device)
        transformer.eval()
        scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    aligned_rows = []
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as h5:
        h5.attrs["feature_extractor_version"] = "extract_wan_features_v2"
        h5.attrs["wan_checkpoint"] = MODEL_ID
        h5.attrs["num_frames"] = args.num_frames
        h5.attrs["resize"] = json.dumps([args.height, args.width])
        for mode in modes:
            videos = []
            mode_rows = []
            for row_idx, row in enumerate(rows):
                fps_value = row.get("fps", row.get("original_fps", None))
                fps = float(fps_value) if fps_value not in (None, "") else None
                video, indices, timestamps, transform_meta = load_video(
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
                videos.append(video)
                original_frame_count = transform_meta.get("original_frame_count")
                duration = row.get("duration", None)
                if duration is None and fps is not None and fps > 0 and original_frame_count:
                    duration = float(original_frame_count) / fps
                out_row = dict(row)
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
                        "wan_checkpoint": MODEL_ID,
                        "feature_extractor_version": "extract_wan_features_v2",
                    }
                )
                mode_rows.append(out_row)
            videos_np = np.stack(videos, axis=0)
            features, latent_shape = make_feature_arrays(videos_np, feature_types, vae, transformer, scheduler, args, device, dtype)
            feature_shapes = {name: [int(x) for x in value.shape[1:]] for name, value in features.items()}
            for row in mode_rows:
                row["feature_shapes"] = feature_shapes
                row["vae_latent_shape"] = latent_shape
            group = h5.create_group(mode)
            group.attrs["feature_shapes"] = json.dumps(feature_shapes)
            if latent_shape is not None:
                group.attrs["vae_latent_shape"] = json.dumps(latent_shape)
            for name, value in features.items():
                group.create_dataset(name, data=value, compression="gzip")
            aligned_rows.extend(mode_rows)

    write_jsonl(Path(args.metadata_output), aligned_rows)
    print(json.dumps({"videos": len(rows), "modes": modes, "features": feature_types, "output_h5": args.output_h5}, indent=2))


if __name__ == "__main__":
    main()
