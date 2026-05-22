#!/usr/bin/env python3
"""Dump VideoChat2-HD projected video tokens for MotionBench/MVBench rows.

The dumped tensor is the hidden video-token stream returned by
VideoChat2_it_hd_mistral.encode_img():

    output_imgs = mistral_proj(QFormer(UMT-video-tokens))

Those are the visual embeddings inserted between <Video> and </Video> before
the Mistral LLM consumes the prompt.  This is the full-VideoLLM token dump used
by the Wan-Motion-REPA adapter experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoConfig, AutoModel


def patch_transformers_remote_code_compat() -> None:
    """Expose old modeling_utils symbols expected by VideoChat2 HF code."""
    import transformers.modeling_utils as modeling_utils
    from transformers import PreTrainedModel
    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - already_pruned_heads
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not getattr(PreTrainedModel.resize_token_embeddings, "_videochat2_compat", False):
        original_resize = PreTrainedModel.resize_token_embeddings

        def resize_token_embeddings_no_mean(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=False):
            return original_resize(
                self,
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=False,
            )

        resize_token_embeddings_no_mean._videochat2_compat = True
        PreTrainedModel.resize_token_embeddings = resize_token_embeddings_no_mean
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

        def get_all_tied_weights_keys(self):
            return getattr(self, "_videochat2_compat_all_tied_weights_keys", {})

        def set_all_tied_weights_keys(self, value):
            object.__setattr__(self, "_videochat2_compat_all_tied_weights_keys", value)

        PreTrainedModel.all_tied_weights_keys = property(get_all_tied_weights_keys, set_all_tied_weights_keys)
    if not hasattr(PreTrainedModel, "get_head_mask"):

        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if head_mask.dim() != 5:
                raise AssertionError(f"head_mask.dim != 5, got {head_mask.dim()}")
            dtype = getattr(self, "dtype", None)
            if dtype is None:
                try:
                    dtype = next(self.parameters()).dtype
                except StopIteration:
                    dtype = torch.float32
            return head_mask.to(dtype=dtype)

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask
        PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import parse_candidates, read_jsonl, rows_by_mode, write_jsonl  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def hd_transform_no_padding(frames: torch.Tensor, image_size: int = 224, hd_num: int = 6) -> torch.Tensor:
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    return F.interpolate(frames, size=(target_height, target_width), mode="bicubic", align_corners=False)


def hd_transform_padding(frames: torch.Tensor, image_size: int = 224, hd_num: int = 6) -> torch.Tensor:
    def pad_to_block(x: torch.Tensor) -> torch.Tensor:
        _, _, height, _ = x.shape
        target = int(np.ceil(height / image_size) * image_size)
        top = (target - height) // 2
        bottom = target - height - top
        return F.pad(x, pad=[0, 0, top, bottom], mode="constant", value=255)

    _, _, height, width = frames.shape
    transposed = False
    if width < height:
        frames = frames.flip(-2, -1)
        transposed = True
        width, height = height, width
    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)
    resized = F.interpolate(frames, size=(new_h, new_w), mode="bicubic", align_corners=False)
    padded = pad_to_block(resized)
    if transposed:
        padded = padded.flip(-2, -1)
    return padded


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
    has_inline_choices = any(
        f"{cand['letter']}." in question or f"{cand['letter']})" in question for cand in cands
    )
    if cands and not has_inline_choices:
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


def load_video_tensor(
    path: str,
    row: dict[str, Any],
    num_frames: int,
    image_size: int,
    hd_num: int,
    padding: bool,
) -> torch.Tensor:
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idx = frame_indices(row, len(vr), num_frames)
    frames = vr.get_batch(idx).asnumpy()
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().contiguous()
    if padding:
        x = hd_transform_padding(x, image_size=image_size, hd_num=hd_num)
    else:
        x = hd_transform_no_padding(x, image_size=image_size, hd_num=hd_num)
    x = x / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


def set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    setattr(cfg, key, value)
    if hasattr(cfg, "cfg"):
        cfg.cfg[key] = value


def set_nested_cfg_value(cfg: Any, path: tuple[str, ...], value: Any) -> None:
    if hasattr(cfg, "cfg"):
        node = cfg.cfg
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value


def materialize_vit_position_embeddings(model: Any, cfg: Any, device: str, dtype: torch.dtype) -> None:
    encoder = model.vision_encoder.encoder
    vit_module = sys.modules[encoder.__class__.__module__]
    vision_cfg = cfg.cfg.vision_encoder if hasattr(cfg, "cfg") else cfg.vision_encoder
    tubelet_size = int(getattr(encoder.patch_embed, "tubelet_size", 1))
    num_frames = int(vision_cfg.num_frames)
    cur_frame = max(1, num_frames // tubelet_size)
    ckpt_num_frame = int(getattr(vision_cfg, "ckpt_num_frame", -1))
    num_patches = int(encoder.patch_embed.num_patches)
    embed_dim = int(encoder.embed_dim)
    img_size = int(getattr(vision_cfg, "img_size", 224))

    needs_pos = getattr(encoder.pos_embed, "is_meta", False) or str(encoder.pos_embed.device) != device
    needs_img_pos = getattr(encoder.img_pos_embed, "is_meta", False) or str(encoder.img_pos_embed.device) != device
    if not (needs_pos or needs_img_pos):
        return
    if img_size != 224:
        pos_embed = vit_module.get_sinusoid_encoding_table2(
            num_patches,
            embed_dim,
            ckpt_num_frame=ckpt_num_frame,
            cur_frame=cur_frame,
        )
        img_pos_embed = vit_module.get_sinusoid_encoding_table2(
            num_patches // cur_frame,
            embed_dim,
            cur_frame=1,
            ckpt_num_frame=1,
            pre_n_position=14 * 14,
        )
    else:
        pos_embed = vit_module.get_sinusoid_encoding_table(
            num_patches,
            embed_dim,
            ckpt_num_frame=ckpt_num_frame,
            cur_frame=cur_frame,
        )
        img_pos_embed = vit_module.get_sinusoid_encoding_table(num_patches // cur_frame, embed_dim)
    encoder.pos_embed = pos_embed.to(device=device, dtype=dtype)
    encoder.img_pos_embed = img_pos_embed.to(device=device, dtype=dtype)


def load_model(args: argparse.Namespace) -> Any:
    patch_transformers_remote_code_compat()
    cfg = AutoConfig.from_pretrained(args.model_id_or_path, trust_remote_code=True)
    if args.mistral_model_path:
        set_cfg_value(cfg, "mistral_model_path", str(Path(args.mistral_model_path).resolve()))
    set_cfg_value(cfg, "use_flash_attention", bool(args.use_flash_attention))
    set_cfg_value(cfg, "freeze_vit", True)
    set_cfg_value(cfg, "freeze_qformer", True)
    set_cfg_value(cfg, "freeze_llm", True)
    set_cfg_value(cfg, "use_lora", False)
    set_nested_cfg_value(cfg, ("vision_encoder", "num_frames"), int(args.num_frames))
    set_nested_cfg_value(cfg, ("vision_encoder", "use_checkpoint"), False)
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModel.from_pretrained(
        args.model_id_or_path,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    model.eval().to(args.device)
    materialize_vit_position_embeddings(model, cfg, args.device, torch_dtype)
    for param in model.parameters():
        param.requires_grad = False
    return model


def encode_batch(model: Any, videos: list[torch.Tensor], instructions: list[str], device: str) -> list[np.ndarray]:
    videos = [video.to(device, non_blocking=True) for video in videos]
    with torch.no_grad():
        output_imgs, output_lens, _ = model.encode_img(videos, instructions)
    out = []
    for tokens, length in zip(output_imgs, output_lens):
        item = tokens[0, : int(length)].float().cpu().numpy().astype(np.float32)
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-jsonl", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--model-id-or-path", default="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf")
    parser.add_argument(
        "--mistral-model-path",
        default=str(ROOT / "models" / "videochat2_mistral" / "Mistral-7B-Instruct-v0.2"),
    )
    parser.add_argument("--modes", default="")
    parser.add_argument("--feature-name", default="videochat2_hd_tokens")
    parser.add_argument("--pooled-feature-name", default="videochat2_hd_pooled")
    parser.add_argument("--mask-feature-name", default="videochat2_hd_token_mask")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hd-num", type=int, default=6)
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.video_jsonl)
    grouped = rows_by_mode(rows)
    if args.modes:
        modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    else:
        modes = sorted(grouped)

    model = load_model(args)
    output_rows: list[dict[str, Any]] = []
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output_h5, "w") as h5:
        h5.attrs["feature_extractor_version"] = "videochat2_hd_hidden_tokens_v1"
        h5.attrs["model_id_or_path"] = args.model_id_or_path
        h5.attrs["mistral_model_path"] = args.mistral_model_path
        h5.attrs["num_frames"] = int(args.num_frames)
        h5.attrs["hd_num"] = int(args.hd_num)
        h5.attrs["padding"] = bool(args.padding)
        for mode in modes:
            mode_rows = grouped.get(mode, [])
            if args.max_rows > 0:
                mode_rows = mode_rows[: args.max_rows]
            if not mode_rows:
                continue

            token_items: list[np.ndarray] = []
            pooled_items: list[np.ndarray] = []
            mask_items: list[np.ndarray] = []
            row_items: list[dict[str, Any]] = []
            for start in range(0, len(mode_rows), args.batch_size):
                batch_rows = mode_rows[start : start + args.batch_size]
                videos = [
                    load_video_tensor(
                        str(row["path"]),
                        row,
                        args.num_frames,
                        args.image_size,
                        args.hd_num,
                        args.padding,
                    )
                    for row in batch_rows
                ]
                instructions = [question_instruction(row) for row in batch_rows]
                batch_tokens = encode_batch(model, videos, instructions, args.device)
                for row, tokens in zip(batch_rows, batch_tokens):
                    token_items.append(tokens)
                    pooled_items.append(tokens.mean(axis=0).astype(np.float32))
                    mask_items.append(np.ones(tokens.shape[0], dtype=np.float32))
                    out_row = dict(row)
                    out_row["feature_extractor_version"] = "videochat2_hd_hidden_tokens_v1"
                    out_row["videochat2_answer_letter"] = answer_letter(row)
                    out_row["videochat2_hd_token_length"] = int(tokens.shape[0])
                    row_items.append(out_row)

            max_len = max(item.shape[0] for item in token_items)
            dim = token_items[0].shape[1]
            token_arr = np.zeros((len(token_items), max_len, dim), dtype=np.float32)
            mask_arr = np.zeros((len(token_items), max_len), dtype=np.float32)
            for idx, tokens in enumerate(token_items):
                token_arr[idx, : tokens.shape[0]] = tokens
                mask_arr[idx, : tokens.shape[0]] = 1.0
            pooled_arr = np.stack(pooled_items, axis=0).astype(np.float32)

            group = h5.create_group(mode)
            group.create_dataset(args.feature_name, data=token_arr, compression="gzip")
            group.create_dataset(args.mask_feature_name, data=mask_arr, compression="gzip")
            group.create_dataset(args.pooled_feature_name, data=pooled_arr, compression="gzip")
            shapes = {
                args.feature_name: [int(x) for x in token_arr.shape[1:]],
                args.mask_feature_name: [int(x) for x in mask_arr.shape[1:]],
                args.pooled_feature_name: [int(x) for x in pooled_arr.shape[1:]],
            }
            group.attrs["feature_shapes"] = json.dumps(shapes)
            for row in row_items:
                row["feature_shapes"] = shapes
            output_rows.extend(row_items)

    write_jsonl(args.metadata_output, output_rows)
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "modes": modes,
                "output_h5": args.output_h5,
                "metadata_output": args.metadata_output,
                "feature_name": args.feature_name,
                "pooled_feature_name": args.pooled_feature_name,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
