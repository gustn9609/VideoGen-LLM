#!/usr/bin/env python3
"""Dump VideoChat2 projected video tokens for MotionBench/MVBench rows.

The dumped feature is the tensor that VideoChat2-Mistral inserts into the LLM:

    inputs_mistral = mistral_proj(QFormer(video_tokens))

This is the right hidden-token stream for Wan-Motion-REPA adapter experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[1]
VC2_ROOT = ROOT / "Ask-Anything" / "video_chat2"
sys.path.insert(0, str(VC2_ROOT))

from models import VideoChat2_it_mistral  # noqa: E402
from utils.config import Config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import parse_candidates, read_jsonl, rows_by_mode, write_jsonl  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def answer_letter(row: dict[str, Any]) -> str:
    ans = str(row.get("answer", "")).strip()
    if len(ans) == 1 and ans.isalpha():
        return ans.upper()
    cands = parse_candidates(row)
    for idx, cand in enumerate(cands):
        if ans == cand.get("text") or str(row.get("answer_text", "")) == cand.get("text"):
            return chr(ord("A") + idx)
    return ans[:1].upper() if ans else "A"


def question_instruction(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    cands = parse_candidates(row)
    if cands and not any(f"{cand['letter']}." in question or f"{cand['letter']})" in question for cand in cands):
        options = "\n".join(f"{cand['letter']}. {cand['text']}" for cand in cands)
        question = f"{question}\n{options}"
    return question + "\nOnly give the option letter."


def frame_indices(row: dict[str, Any], total: int, num_frames: int) -> list[int]:
    sampled = row.get("sampled_frame_indices")
    if isinstance(sampled, list) and len(sampled) >= num_frames:
        return [min(max(int(x), 0), total - 1) for x in sampled[:num_frames]]
    if total <= 0:
        return [0] * num_frames
    if total < num_frames:
        base = list(range(total))
        return base + [base[-1]] * (num_frames - total)
    return np.linspace(0, total - 1, num_frames).round().astype(int).tolist()


def load_video_tensor(path: str, row: dict[str, Any], num_frames: int, image_size: int) -> torch.Tensor:
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idx = frame_indices(row, len(vr), num_frames)
    frames = vr.get_batch(idx).asnumpy()
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    short = int(round(image_size * 256 / 224))
    x = F.interpolate(x, size=(short, short), mode="bicubic", align_corners=False)
    top = max(0, (x.shape[-2] - image_size) // 2)
    left = max(0, (x.shape[-1] - image_size) // 2)
    x = x[:, :, top : top + image_size, left : left + image_size]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)


def load_model(args: argparse.Namespace):
    cfg = Config.from_file(str(VC2_ROOT / "configs" / "config_mistral.json"))
    cfg.model.mistral_model_path = args.mistral_model_path
    cfg.model.videochat2_model_path = args.videochat2_model_path
    cfg.model.vit_blip_model_path = args.vit_blip_model_path
    cfg.model.use_flash_attention = bool(args.use_flash_attention)
    cfg.model.freeze_vit = True
    cfg.model.freeze_qformer = True
    cfg.model.vision_encoder.num_frames = int(args.num_frames)
    model = VideoChat2_it_mistral(config=cfg.model)
    model.eval().to(args.device)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--mistral-model-path", default="")
    parser.add_argument("--videochat2-model-path", default="")
    parser.add_argument("--vit-blip-model-path", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--feature-name", default="videochat2_mistral_tokens")
    parser.add_argument("--pooled-feature-name", default="videochat2_mistral_pooled")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        args.mistral_model_path = args.mistral_model_path or manifest["mistral_model_path"]
        args.videochat2_model_path = args.videochat2_model_path or manifest["videochat2_model_path"]
        args.vit_blip_model_path = args.vit_blip_model_path or manifest["vit_blip_model_path"]
    missing = [name for name in ["mistral_model_path", "videochat2_model_path", "vit_blip_model_path"] if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing model paths: {missing}")

    rows = read_jsonl(args.video_jsonl)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    grouped = rows_by_mode(rows)
    if args.modes:
        modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    else:
        modes = sorted(grouped)
    model = load_model(args)

    output_rows: list[dict[str, Any]] = []
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as h5:
        h5.attrs["feature_extractor_version"] = "videochat2_mistral_hidden_tokens_v1"
        h5.attrs["mistral_model_path"] = args.mistral_model_path
        h5.attrs["videochat2_model_path"] = args.videochat2_model_path
        h5.attrs["vit_blip_model_path"] = args.vit_blip_model_path
        for mode in modes:
            mode_rows = grouped.get(mode, [])
            if not mode_rows:
                continue
            tokens_out = []
            pooled_out = []
            for row in mode_rows:
                video = load_video_tensor(str(row["path"]), row, args.num_frames, args.image_size).to(args.device)
                instruction = [question_instruction(row)]
                with torch.no_grad():
                    tokens, _ = model.encode_img(video, instruction)
                tokens_np = tokens.float().cpu().numpy()[0]
                tokens_out.append(tokens_np.astype(np.float32))
                pooled_out.append(tokens_np.mean(axis=0).astype(np.float32))
                out_row = dict(row)
                out_row["feature_extractor_version"] = "videochat2_mistral_hidden_tokens_v1"
                output_rows.append(out_row)
            group = h5.create_group(mode)
            tokens_arr = np.stack(tokens_out, axis=0).astype(np.float32)
            pooled_arr = np.stack(pooled_out, axis=0).astype(np.float32)
            group.create_dataset(args.feature_name, data=tokens_arr, compression="gzip")
            group.create_dataset(args.pooled_feature_name, data=pooled_arr, compression="gzip")
            shapes = {
                args.feature_name: [int(x) for x in tokens_arr.shape[1:]],
                args.pooled_feature_name: [int(x) for x in pooled_arr.shape[1:]],
            }
            group.attrs["feature_shapes"] = json.dumps(shapes)
            for row in output_rows[-len(mode_rows) :]:
                row["feature_shapes"] = shapes
    write_jsonl(args.metadata_output, output_rows)
    print(json.dumps({"rows": len(output_rows), "modes": modes, "output_h5": args.output_h5, "metadata_output": args.metadata_output}, indent=2))


if __name__ == "__main__":
    main()
