#!/usr/bin/env python3
"""Fine-tune VideoChat2-HD with multiple-choice CE and Wan-REPA auxiliary loss.

This is the next step after hidden-token dumping:

1. Load the real VideoChat2-HD checkpoint.
2. Feed VideoLLM visual tokens into the frozen Mistral checkpoint.
3. Train a small residual video-token adapter, optionally with Mistral LoRA.
4. Add a Wan-Motion-REPA auxiliary alignment loss on the adapted video tokens.
5. Evaluate by candidate-letter negative log-likelihood.

The default run uses cached VideoChat2 hidden tokens.  That keeps the experiment
faithful to the full LLM checkpoint while avoiding repeated frozen UMT/QFormer
encoding.  Use --hidden-features-h5 "" to encode videos online through
VideoChat2.encode_img().
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_videochat2_hd_hidden_tokens import load_model, load_video_tensor, question_instruction  # noqa: E402
from motionbench_common import answer_index, parse_candidates, read_jsonl, row_id, rows_by_mode, write_jsonl  # noqa: E402
from motionbench_repa_common import aligned_rows_by_mode, safe_name, select_h5_rows  # noqa: E402


@dataclass
class Example:
    row_index: int
    row: dict[str, Any]
    label: int
    letters: list[str]


class VideoResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, dropout: float, init_scale: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(F.gelu(self.down(self.norm(tokens)))))
        return tokens + self.scale * residual * mask.unsqueeze(-1)


class BranchSeparatedAdapter(nn.Module):
    """Separate residual branches for equivariance and relation alignment."""

    def __init__(self, dim: int, bottleneck: int, dropout: float, init_scale: float) -> None:
        super().__init__()
        self.eq_norm = nn.LayerNorm(dim)
        self.eq_down = nn.Linear(dim, bottleneck)
        self.eq_up = nn.Linear(bottleneck, dim)
        self.rel_norm = nn.LayerNorm(dim)
        self.rel_down = nn.Linear(dim, bottleneck)
        self.rel_up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.eq_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.rel_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def branch_residual(self, tokens: torch.Tensor, norm: nn.LayerNorm, down: nn.Linear, up: nn.Linear) -> torch.Tensor:
        return up(self.dropout(F.gelu(down(norm(tokens)))))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_exp = mask.unsqueeze(-1)
        eq_residual = self.branch_residual(tokens, self.eq_norm, self.eq_down, self.eq_up) * mask_exp
        rel_residual = self.branch_residual(tokens, self.rel_norm, self.rel_down, self.rel_up) * mask_exp
        eq_tokens = tokens + self.eq_scale * eq_residual
        rel_tokens = tokens + self.rel_scale * rel_residual
        combined = tokens + self.eq_scale * eq_residual + self.rel_scale * rel_residual
        return {"combined": combined, "eq": eq_tokens, "rel": rel_tokens}


class RepaHead(nn.Module):
    def __init__(self, dim: int, align_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, align_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.net(pooled)


def make_adapter(args: argparse.Namespace, token_dim: int, device: torch.device) -> nn.Module:
    if args.adapter_type == "branch_separated":
        return BranchSeparatedAdapter(token_dim, args.adapter_dim, args.adapter_dropout, args.adapter_init_scale).to(device=device)
    return VideoResidualAdapter(token_dim, args.adapter_dim, args.adapter_dropout, args.adapter_init_scale).to(device=device)


def apply_adapter(adapter: nn.Module, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = adapter(tokens, mask)
    if isinstance(out, dict):
        return out["combined"], out
    return out, {"combined": out, "eq": out, "rel": out}


def random_project_to(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] == out_dim:
        y = x
    elif x.shape[1] < out_dim:
        y = np.concatenate([x, np.zeros((x.shape[0], out_dim - x.shape[1]), dtype=np.float32)], axis=1)
    else:
        rng = np.random.default_rng(seed)
        proj = rng.normal(0.0, 1.0 / math.sqrt(out_dim), size=(x.shape[1], out_dim)).astype(np.float32)
        y = x @ proj
    denom = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.maximum(denom, 1e-8)).astype(np.float32)


def load_examples(metadata_jsonl: str, mode: str, max_rows: int) -> list[Example]:
    grouped = rows_by_mode(read_jsonl(metadata_jsonl))
    rows = grouped.get(mode, [])
    if max_rows > 0:
        rows = rows[:max_rows]
    examples = []
    for idx, row in enumerate(rows):
        cands = parse_candidates(row)
        label = answer_index(row)
        if label is None or label < 0 or label >= len(cands):
            continue
        examples.append(
            Example(
                row_index=idx,
                row=row,
                label=int(label),
                letters=[str(cand["letter"]) for cand in cands],
            )
        )
    return examples


def split_examples(examples: list[Example], test_size: float, seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(len(examples))
    strat = [str(ex.row.get("question_type", "")) for ex in examples]
    try:
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=strat)
    except ValueError:
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    return sorted(train_idx.tolist()), sorted(test_idx.tolist())


def cv_splits(examples: list[Example], folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    indices = np.arange(len(examples))
    strat = np.asarray([str(ex.row.get("question_type", "")) for ex in examples])
    if folds <= 1:
        return []
    counts = {label: int((strat == label).sum()) for label in sorted(set(strat))}
    max_folds = min(counts.values()) if counts else folds
    folds = max(2, min(int(folds), int(max_folds)))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(sorted(train.tolist()), sorted(test.tolist())) for train, test in splitter.split(indices, strat)]


def token_embedder(videochat2: Any):
    lm = videochat2.mistral_model
    if hasattr(lm, "base_model") and hasattr(lm.base_model, "model") and hasattr(lm.base_model.model, "model"):
        return lm.base_model.model.model.embed_tokens
    if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
        return lm.model.embed_tokens
    if hasattr(lm, "base_model") and hasattr(lm.base_model, "embed_tokens"):
        return lm.base_model.embed_tokens
    return lm.model.embed_tokens


def get_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters())
    return param.device, param.dtype


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    question = question_instruction(row)
    before = "[INST] <Video>"
    after = f"</Video> [/INST] [INST] {question} [/INST] Best option: ("
    return before, after


def embed_text(videochat2: Any, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = videochat2.mistral_tokenizer
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return token_embedder(videochat2)(ids), ids


def build_lm_batch(
    videochat2: Any,
    video_tokens: list[torch.Tensor],
    video_masks: list[torch.Tensor],
    rows: list[dict[str, Any]],
    answer_letters: list[str],
    adapter: nn.Module,
    repa_head: RepaHead,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = get_device_dtype(videochat2.mistral_model)
    tokenizer = videochat2.mistral_tokenizer
    bos_id = tokenizer.bos_token_id
    bos_ids = torch.tensor([[bos_id]], device=device)
    bos_embed = token_embedder(videochat2)(bos_ids)

    input_items: list[torch.Tensor] = []
    label_items: list[torch.Tensor] = []
    z_items: list[torch.Tensor] = []
    for tokens, mask, row, letter in zip(video_tokens, video_masks, rows, answer_letters):
        tokens = tokens.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.float32)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            mask = mask.unsqueeze(0)
        adapted, aux_tokens = apply_adapter(adapter, tokens, mask)
        z_items.append(repa_head(aux_tokens["eq"], mask)[0])
        adapted_lm = adapted.to(dtype=dtype)

        before, after = build_prompt(row)
        before_emb, _ = embed_text(videochat2, before, device)
        after_emb, _ = embed_text(videochat2, after, device)
        answer_text = f"{letter})"
        answer_emb, answer_ids = embed_text(videochat2, answer_text, device)
        inputs = torch.cat([bos_embed, before_emb, adapted_lm, after_emb, answer_emb], dim=1)[0]
        labels = torch.full((inputs.shape[0],), -100, dtype=torch.long, device=device)
        labels[-answer_ids.shape[1] :] = answer_ids[0]
        input_items.append(inputs)
        label_items.append(labels)

    max_len = max(item.shape[0] for item in input_items)
    hidden_dim = input_items[0].shape[-1]
    inputs_embeds = torch.zeros((len(input_items), max_len, hidden_dim), dtype=dtype, device=device)
    attention_mask = torch.zeros((len(input_items), max_len), dtype=torch.long, device=device)
    labels = torch.full((len(input_items), max_len), -100, dtype=torch.long, device=device)
    for i, (inputs, lbl) in enumerate(zip(input_items, label_items)):
        length = inputs.shape[0]
        inputs_embeds[i, :length] = inputs
        attention_mask[i, :length] = 1
        labels[i, :length] = lbl
    return inputs_embeds, attention_mask, labels, torch.stack(z_items, dim=0)


def lm_loss(videochat2: Any, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    outputs = videochat2.mistral_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss


def lm_outputs(videochat2: Any, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    return videochat2.mistral_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )


def shifted_label_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels[:, 1:] != -100


def kl_to_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    mask = shifted_label_mask(labels)
    if not bool(mask.any()):
        return student_logits.new_zeros(())
    s = student_logits[:, :-1][mask] / float(temperature)
    t = teacher_logits[:, :-1][mask] / float(temperature)
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean") * (float(temperature) ** 2)


def segment_pool(tokens: torch.Tensor, mask: torch.Tensor, segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = tokens.shape[1]
    boundaries = torch.linspace(0, length, int(segments) + 1, device=tokens.device).round().long()
    pooled = []
    pooled_mask = []
    for idx in range(int(segments)):
        start = int(boundaries[idx].item())
        end = int(boundaries[idx + 1].item())
        if end <= start:
            end = min(length, start + 1)
        chunk = tokens[:, start:end]
        chunk_mask = mask[:, start:end]
        denom = chunk_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled.append((chunk * chunk_mask.unsqueeze(-1)).sum(dim=1) / denom)
        pooled_mask.append((chunk_mask.sum(dim=1) > 0).float())
    return torch.stack(pooled, dim=1), torch.stack(pooled_mask, dim=1)


def relation_matrix(x: torch.Tensor) -> torch.Tensor:
    z = F.normalize(x, dim=-1)
    return torch.matmul(z, z.transpose(1, 2))


def relation_repa_loss(tokens: torch.Tensor, mask: torch.Tensor, teacher_relation: torch.Tensor, segments: int) -> torch.Tensor:
    pooled, pooled_mask = segment_pool(tokens, mask, segments)
    student_rel = relation_matrix(pooled)
    pair_mask = pooled_mask.unsqueeze(1) * pooled_mask.unsqueeze(2)
    return ((student_rel - teacher_relation.to(student_rel.device, student_rel.dtype)).pow(2) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def candidate_nll(
    videochat2: Any,
    video_token: torch.Tensor,
    video_mask: torch.Tensor,
    row: dict[str, Any],
    letter: str,
    adapter: nn.Module,
    repa_head: RepaHead,
) -> float:
    inputs_embeds, attention_mask, labels, _ = build_lm_batch(
        videochat2,
        [video_token],
        [video_mask],
        [row],
        [letter],
        adapter,
        repa_head,
    )
    with torch.no_grad():
        loss = lm_loss(videochat2, inputs_embeds, attention_mask, labels)
    count = int((labels != -100).sum().item())
    return float(loss.detach().cpu()) * max(1, count)


def load_cached_video_tokens(args: argparse.Namespace, examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(args.hidden_features_h5, "r") as h5:
        group = h5[args.mode]
        tokens = group[args.hidden_feature][:]
        if args.hidden_mask_feature and args.hidden_mask_feature in group:
            masks = group[args.hidden_mask_feature][:]
        else:
            masks = np.ones(tokens.shape[:2], dtype=np.float32)
    ids = [ex.row_index for ex in examples]
    return torch.from_numpy(tokens[ids].astype(np.float32)), torch.from_numpy(masks[ids].astype(np.float32))


def encode_video_online(videochat2: Any, args: argparse.Namespace, examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = get_device_dtype(videochat2.mistral_model)
    token_items = []
    mask_items = []
    for ex in examples:
        video = load_video_tensor(
            str(ex.row["path"]),
            ex.row,
            args.num_frames,
            args.image_size,
            args.hd_num,
            args.padding,
        ).to(device)
        with torch.no_grad():
            output_imgs, output_lens, _ = videochat2.encode_img([video], [question_instruction(ex.row)])
        tokens = output_imgs[0][0, : int(output_lens[0])].detach().to(dtype=torch.float32).cpu()
        token_items.append(tokens)
        mask_items.append(torch.ones(tokens.shape[0], dtype=torch.float32))
    max_len = max(x.shape[0] for x in token_items)
    dim = token_items[0].shape[1]
    tokens = torch.zeros((len(token_items), max_len, dim), dtype=torch.float32)
    masks = torch.zeros((len(token_items), max_len), dtype=torch.float32)
    for i, item in enumerate(token_items):
        tokens[i, : item.shape[0]] = item
        masks[i, : item.shape[0]] = 1.0
    return tokens, masks


def load_repa_targets(args: argparse.Namespace, examples: list[Example]) -> np.ndarray:
    safe_wan = safe_name(args.wan_feature)
    feature = f"{args.target_prefix}_{args.repa_target}_{safe_wan}"
    target_rows = aligned_rows_by_mode(args.target_metadata_jsonl).get(args.mode, [])
    with h5py.File(args.target_features_h5, "r") as h5:
        data = select_h5_rows(h5[args.mode][feature][:], target_rows)
    ids = [ex.row_index for ex in examples]
    target = data[ids].reshape(len(ids), -1).astype(np.float32)
    return random_project_to(target, args.align_dim, args.seed + 991)


def temporal_tokens_np(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 4:
        return value.reshape(value.shape[0], -1)
    if value.ndim == 3:
        return value.reshape(value.shape[0], -1)
    if value.ndim == 2:
        return value
    return value.reshape(1, -1)


def segment_pool_np(tokens: np.ndarray, segments: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(tokens, dtype=np.float32)
    segments = int(segments)
    out = np.zeros((segments, tokens.shape[1]), dtype=np.float32)
    mask = np.zeros((segments,), dtype=np.float32)
    if tokens.shape[0] == 0:
        return out, mask
    boundaries = np.linspace(0, tokens.shape[0], segments + 1).round().astype(int)
    for idx in range(segments):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        if end <= start:
            end = min(tokens.shape[0], start + 1)
        chunk = tokens[start:end]
        if chunk.size:
            out[idx] = chunk.mean(axis=0)
            mask[idx] = 1.0
    return out, mask


def load_wan_relation_targets(args: argparse.Namespace, examples: list[Example]) -> torch.Tensor | None:
    if not args.source_features_h5 or float(args.lambda_relation) <= 0:
        return None
    with h5py.File(args.source_features_h5, "r") as h5:
        data = h5[args.mode][args.wan_feature][:]
    rels = []
    for ex in examples:
        tokens = temporal_tokens_np(data[ex.row_index])
        seg, _ = segment_pool_np(tokens, args.segments)
        norm = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-8)
        rels.append((norm @ norm.T).astype(np.float32))
    return torch.from_numpy(np.stack(rels, axis=0))


def parse_type_lambdas(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    return {str(key): float(value) for key, value in json.loads(raw).items()}


def batch_type_weights(examples: list[Example], batch_idx: list[int], type_lambdas: dict[str, float], device: torch.device) -> torch.Tensor:
    values = [float(type_lambdas.get(str(examples[idx].row.get("question_type", "")), 1.0)) for idx in batch_idx]
    return torch.tensor(values, dtype=torch.float32, device=device)


def lambda_schedule_multiplier(args: argparse.Namespace, epoch: int) -> float:
    """Return auxiliary-loss multiplier for a zero-based epoch id.

    Cross-validation passes `fold_id * 1000 + epoch` as the RNG epoch, so the
    modulo keeps schedules identical across folds.
    """

    local_epoch = int(epoch) % 1000
    total = max(1, int(args.epochs))
    schedule = str(getattr(args, "lambda_schedule", "constant"))
    if schedule == "constant":
        return 1.0
    if schedule == "warmup_ce_polish":
        if total == 1:
            return 1.0
        if local_epoch == total - 1:
            return 0.0
        if local_epoch == 0:
            return 0.5
        return 1.0
    if schedule == "late_start":
        return 0.0 if local_epoch == 0 and total > 1 else 1.0
    if schedule == "cosine_decay":
        if total == 1:
            return 1.0
        return float(0.5 * (1.0 + math.cos(math.pi * local_epoch / max(1, total - 1))))
    raise ValueError(f"Unknown lambda schedule: {schedule}")


def configure_trainables(videochat2: Any, args: argparse.Namespace) -> None:
    for param in videochat2.parameters():
        param.requires_grad = False
    if args.use_lora:
        try:
            import peft.import_utils as peft_import_utils
            import peft.tuners.lora.torchao as peft_lora_torchao

            peft_import_utils.is_torchao_available = lambda: False
            peft_lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        videochat2.mistral_model = get_peft_model(videochat2.mistral_model, peft_cfg)
        videochat2.mistral_model.train()


def train_epoch(
    videochat2: Any,
    adapter: nn.Module,
    repa_head: RepaHead,
    optimizer: torch.optim.Optimizer,
    examples: list[Example],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    repa_targets: torch.Tensor,
    relation_targets: torch.Tensor | None,
    train_idx: list[int],
    args: argparse.Namespace,
    type_lambdas: dict[str, float],
    teacher_adapter: nn.Module | None,
    teacher_repa_head: RepaHead | None,
    epoch: int,
) -> dict[str, float]:
    adapter.train()
    repa_head.train()
    if args.use_lora:
        videochat2.mistral_model.train()
    else:
        videochat2.mistral_model.eval()
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(train_idx)
    ce_losses = []
    repa_losses = []
    relation_losses = []
    kl_losses = []
    for start in range(0, len(order), args.batch_size):
        batch_idx = [int(x) for x in order[start : start + args.batch_size]]
        batch_examples = [examples[i] for i in batch_idx]
        letters = [ex.letters[ex.label] for ex in batch_examples]
        type_weights = batch_type_weights(examples, batch_idx, type_lambdas, tokens.device).to(dtype=torch.float32)
        input_embeds, attention_mask, labels, z = build_lm_batch(
            videochat2,
            [tokens[i] for i in batch_idx],
            [masks[i] for i in batch_idx],
            [ex.row for ex in batch_examples],
            letters,
            adapter,
            repa_head,
        )
        outputs = lm_outputs(videochat2, input_embeds, attention_mask, labels)
        ce = outputs.loss
        target = repa_targets[batch_idx].to(z.device, dtype=torch.float32)
        repa_per = 1.0 - (F.normalize(z.float(), dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1)
        repa = (repa_per * type_weights).mean()
        relation = z.new_zeros(())
        if relation_targets is not None and float(args.lambda_relation) > 0:
            _, aux_tokens = apply_adapter(adapter, tokens[batch_idx], masks[batch_idx])
            relation = relation_repa_loss(aux_tokens["rel"], masks[batch_idx], relation_targets[batch_idx], args.segments)
        kl = z.new_zeros(())
        if teacher_adapter is not None and teacher_repa_head is not None and float(args.lambda_kl) > 0:
            with torch.no_grad():
                teacher_inputs, teacher_attention, teacher_labels, _ = build_lm_batch(
                    videochat2,
                    [tokens[i] for i in batch_idx],
                    [masks[i] for i in batch_idx],
                    [ex.row for ex in batch_examples],
                    letters,
                    teacher_adapter,
                    teacher_repa_head,
                )
                teacher_outputs = lm_outputs(videochat2, teacher_inputs, teacher_attention, teacher_labels)
            kl = kl_to_teacher(outputs.logits, teacher_outputs.logits, labels, args.kl_temperature)
        aux_mult = lambda_schedule_multiplier(args, epoch)
        eff_lambda_repa = float(args.lambda_repa) * aux_mult
        eff_lambda_relation = float(args.lambda_relation) * aux_mult
        loss = ce + eff_lambda_repa * repa + eff_lambda_relation * relation + float(args.lambda_kl) * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + list(repa_head.parameters()) + [p for p in videochat2.mistral_model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        ce_losses.append(float(ce.detach().cpu()))
        repa_losses.append(float(repa.detach().cpu()))
        relation_losses.append(float(relation.detach().cpu()))
        kl_losses.append(float(kl.detach().cpu()))
    return {
        "ce_loss": float(np.mean(ce_losses)),
        "repa_loss": float(np.mean(repa_losses)),
        "relation_loss": float(np.mean(relation_losses)),
        "kl_loss": float(np.mean(kl_losses)),
        "lambda_repa_effective": float(args.lambda_repa) * lambda_schedule_multiplier(args, epoch),
        "lambda_relation_effective": float(args.lambda_relation) * lambda_schedule_multiplier(args, epoch),
    }


def evaluate(
    videochat2: Any,
    adapter: nn.Module,
    repa_head: RepaHead,
    examples: list[Example],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    indices: list[int],
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    adapter.eval()
    repa_head.eval()
    videochat2.mistral_model.eval()
    correct = 0
    rows = []
    per_type: dict[str, list[int]] = {}
    for idx in indices:
        ex = examples[idx]
        scores = []
        for letter in ex.letters:
            nll = candidate_nll(videochat2, tokens[idx], masks[idx], ex.row, letter, adapter, repa_head)
            scores.append(-nll)
        pred = int(np.argmax(scores))
        ok = int(pred == ex.label)
        correct += ok
        qtype = str(ex.row.get("question_type", ""))
        per_type.setdefault(qtype, []).append(ok)
        cands = parse_candidates(ex.row)
        rows.append(
            {
                "split": split_name,
                "row_index": int(ex.row_index),
                "video_id": row_id(ex.row, ex.row_index),
                "question_type": qtype,
                "answer_index": int(ex.label),
                "prediction_index": pred,
                "correct": bool(ok),
                "candidates": [
                    {
                        "index": int(i),
                        "letter": ex.letters[i],
                        "text": cands[i]["text"] if i < len(cands) else "",
                        "score": float(scores[i]),
                    }
                    for i in range(len(ex.letters))
                ],
            }
        )
    summary = {
        "split": split_name,
        "accuracy": float(correct / max(1, len(indices))),
        "correct": int(correct),
        "total": int(len(indices)),
        "per_question_type": {
            key: {"accuracy": float(np.mean(vals)), "correct": int(np.sum(vals)), "total": int(len(vals))}
            for key, vals in sorted(per_type.items())
        },
    }
    return summary, rows


def load_teacher_modules(args: argparse.Namespace, fold_id: int, token_dim: int, device: torch.device) -> tuple[nn.Module | None, RepaHead | None]:
    if not args.teacher_checkpoint:
        return None, None
    ckpt = torch.load(args.teacher_checkpoint, map_location="cpu")
    adapter_type = str(ckpt.get("config", {}).get("adapter_type", "residual"))
    adapter_args = argparse.Namespace(**{**vars(args), "adapter_type": adapter_type})
    adapter = make_adapter(adapter_args, token_dim, device)
    repa_head = RepaHead(token_dim, args.align_dim).to(device=device)
    if "folds" in ckpt:
        fold_state = next(item for item in ckpt["folds"] if int(item["fold"]) == int(fold_id))
        adapter.load_state_dict(fold_state["adapter"])
        repa_head.load_state_dict(fold_state["repa_head"])
    else:
        adapter.load_state_dict(ckpt["adapter"])
        repa_head.load_state_dict(ckpt["repa_head"])
    adapter.eval()
    repa_head.eval()
    for param in adapter.parameters():
        param.requires_grad = False
    for param in repa_head.parameters():
        param.requires_grad = False
    return adapter, repa_head


def pretrain_repa(
    adapter: nn.Module,
    repa_head: RepaHead,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    repa_targets: torch.Tensor,
    relation_targets: torch.Tensor | None,
    indices: list[int],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    history = []
    if int(args.pretrain_epochs) <= 0:
        return history
    for epoch in range(int(args.pretrain_epochs)):
        adapter.train()
        repa_head.train()
        rng = np.random.default_rng(args.seed + 10000 + epoch)
        order = rng.permutation(indices)
        losses = []
        rel_losses = []
        for start in range(0, len(order), args.batch_size):
            batch_idx = [int(x) for x in order[start : start + args.batch_size]]
            adapted, aux_tokens = apply_adapter(adapter, tokens[batch_idx], masks[batch_idx])
            z = repa_head(aux_tokens["eq"], masks[batch_idx])
            target = repa_targets[batch_idx].to(z.device, dtype=torch.float32)
            repa = 1.0 - (F.normalize(z.float(), dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean()
            relation = z.new_zeros(())
            if relation_targets is not None and float(args.lambda_relation) > 0:
                relation = relation_repa_loss(aux_tokens["rel"], masks[batch_idx], relation_targets[batch_idx], args.segments)
            loss = float(args.lambda_repa) * repa + float(args.lambda_relation) * relation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(repa_head.parameters()), args.grad_clip)
            optimizer.step()
            losses.append(float(repa.detach().cpu()))
            rel_losses.append(float(relation.detach().cpu()))
        history.append({"pretrain_epoch": epoch + 1, "repa_loss": float(np.mean(losses)), "relation_loss": float(np.mean(rel_losses))})
    return history


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# VideoChat2-HD Wan-REPA Fine-tuning", ""]
    lines.append("| Fold | Stage | Split | Acc | Correct/total |")
    lines.append("|---|---|---|---:|---:|")
    for item in payload["eval"]:
        lines.append(
            f"| {item.get('fold', '')} | {item['stage']} | {item['split']} | "
            f"{item['accuracy']:.4f} | {item['correct']}/{item['total']} |"
        )
    lines.append("")
    lines.append("## Training")
    lines.append("")
    lines.append("| Fold | Epoch | CE loss | REPA loss | Relation loss | KL loss | Eff REPA λ | Eff Rel λ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["history"]:
        if "epoch" not in row:
            continue
        lines.append(
            f"| {row.get('fold', '')} | {row['epoch']} | {row['ce_loss']:.4f} | {row['repa_loss']:.4f} | "
            f"{row.get('relation_loss', 0.0):.4f} | {row.get('kl_loss', 0.0):.4f} | "
            f"{row.get('lambda_repa_effective', 0.0):.4f} | {row.get('lambda_relation_effective', 0.0):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_stage(records: list[dict[str, Any]], stage: str, split: str) -> dict[str, Any]:
    selected = [row for row in records if row.get("stage") == stage and row.get("split") == split and row.get("fold") != "all"]
    correct = sum(int(row["correct"]) for row in selected)
    total = sum(int(row["total"]) for row in selected)
    per_type: dict[str, dict[str, int]] = {}
    for row in selected:
        for qtype, stats in row.get("per_question_type", {}).items():
            bucket = per_type.setdefault(qtype, {"correct": 0, "total": 0})
            bucket["correct"] += int(stats["correct"])
            bucket["total"] += int(stats["total"])
    return {
        "fold": "all",
        "stage": stage,
        "split": split,
        "accuracy": float(correct / max(1, total)),
        "correct": int(correct),
        "total": int(total),
        "per_question_type": {
            key: {
                "accuracy": float(value["correct"] / max(1, value["total"])),
                "correct": int(value["correct"]),
                "total": int(value["total"]),
            }
            for key, value in sorted(per_type.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--hidden-features-h5", default="")
    parser.add_argument("--hidden-feature", default="videochat2_hd_tokens")
    parser.add_argument("--hidden-mask-feature", default="videochat2_hd_token_mask")
    parser.add_argument("--target-features-h5", required=True)
    parser.add_argument("--target-metadata-jsonl", required=True)
    parser.add_argument("--source-features-h5", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="high_motion+camera_comp")
    parser.add_argument("--model-id-or-path", default="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf")
    parser.add_argument("--mistral-model-path", default=str(ROOT / "models" / "videochat2_mistral" / "Mistral-7B-Instruct-v0.2"))
    parser.add_argument("--wan-feature", default="wan_vae_grid_1x1")
    parser.add_argument("--target-prefix", default="wmrepa")
    parser.add_argument("--repa-target", default="equivariance", choices=["structured_compact", "dynamics_relation", "relation_only", "equivariance", "multi_target"])
    parser.add_argument("--lambda-repa", type=float, default=0.1)
    parser.add_argument("--lambda-relation", type=float, default=0.0)
    parser.add_argument("--lambda-kl", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--type-lambda-json", default="")
    parser.add_argument("--align-dim", type=int, default=128)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--adapter-type", choices=["residual", "branch_separated"], default="residual")
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--adapter-dropout", type=float, default=0.05)
    parser.add_argument("--adapter-init-scale", type=float, default=0.1)
    parser.add_argument(
        "--lambda-schedule",
        choices=["constant", "warmup_ce_polish", "late_start", "cosine_decay"],
        default="constant",
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-scope", choices=["train", "all"], default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hd-num", type=int, default=6)
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.metadata_jsonl, args.mode, args.max_rows)
    if len(examples) < 8:
        raise ValueError(f"Need at least 8 valid examples, got {len(examples)}")

    videochat2 = load_model(args)
    configure_trainables(videochat2, args)
    device, dtype = get_device_dtype(videochat2.mistral_model)
    if args.hidden_features_h5:
        tokens, masks = load_cached_video_tokens(args, examples)
    else:
        tokens, masks = encode_video_online(videochat2, args, examples)
    tokens = tokens.to(device=device, dtype=torch.float32)
    masks = masks.to(device=device, dtype=torch.float32)

    repa_targets = torch.from_numpy(load_repa_targets(args, examples)).to(device=device, dtype=torch.float32)
    relation_targets = load_wan_relation_targets(args, examples)
    if relation_targets is not None:
        relation_targets = relation_targets.to(device=device, dtype=torch.float32)
    type_lambdas = parse_type_lambdas(args.type_lambda_json)

    eval_payload = []
    all_score_rows: list[dict[str, Any]] = []
    history = []
    fold_checkpoints = []
    if args.folds > 1 and args.use_lora:
        raise ValueError("Cross-validation with --use-lora is not supported in one process; run folds separately.")
    split_seed = args.seed if int(args.split_seed) < 0 else int(args.split_seed)
    splits = cv_splits(examples, args.folds, split_seed) if args.folds > 1 else [split_examples(examples, args.test_size, split_seed)]
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(args.seed + fold_id)
        adapter = make_adapter(args, tokens.shape[-1], device)
        repa_head = RepaHead(tokens.shape[-1], args.align_dim).to(device=device)
        teacher_adapter, teacher_repa_head = load_teacher_modules(args, fold_id, tokens.shape[-1], device)
        trainable = list(adapter.parameters()) + list(repa_head.parameters()) + [p for p in videochat2.mistral_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        pretrain_indices = list(range(len(examples))) if args.pretrain_scope == "all" else train_idx
        pretrain_history = pretrain_repa(adapter, repa_head, optimizer, tokens, masks, repa_targets, relation_targets, pretrain_indices, args)
        for item in pretrain_history:
            history.append({"fold": fold_id, **item})

        initial_summary, score_rows = evaluate(videochat2, adapter, repa_head, examples, tokens, masks, test_idx, "test")
        initial_summary["stage"] = "initial"
        initial_summary["fold"] = fold_id
        eval_payload.append(initial_summary)
        for row in score_rows:
            row["stage"] = "initial"
            row["fold"] = fold_id
        all_score_rows.extend(score_rows)

        for epoch in range(args.epochs):
            stats = train_epoch(
                videochat2,
                adapter,
                repa_head,
                optimizer,
                examples,
                tokens,
                masks,
                repa_targets,
                relation_targets,
                train_idx,
                args,
                type_lambdas,
                teacher_adapter,
                teacher_repa_head,
                epoch + fold_id * 1000,
            )
            history.append({"fold": fold_id, "epoch": epoch + 1, **stats})
            summary, epoch_scores = evaluate(videochat2, adapter, repa_head, examples, tokens, masks, test_idx, "test")
            summary["stage"] = f"epoch_{epoch + 1}"
            summary["fold"] = fold_id
            eval_payload.append(summary)
            for row in epoch_scores:
                row["stage"] = f"epoch_{epoch + 1}"
                row["fold"] = fold_id
            all_score_rows.extend(epoch_scores)

        train_summary, train_scores = evaluate(videochat2, adapter, repa_head, examples, tokens, masks, train_idx, "train")
        train_summary["stage"] = "final"
        train_summary["fold"] = fold_id
        final_summary, final_scores = evaluate(videochat2, adapter, repa_head, examples, tokens, masks, test_idx, "test")
        final_summary["stage"] = "final"
        final_summary["fold"] = fold_id
        eval_payload.extend([train_summary, final_summary])
        for split_scores, stage in [(train_scores, "final"), (final_scores, "final")]:
            for row in split_scores:
                row["stage"] = stage
                row["fold"] = fold_id
        all_score_rows.extend(train_scores + final_scores)
        fold_checkpoints.append(
            {
                "fold": fold_id,
                "train_indices": train_idx,
                "test_indices": test_idx,
                "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
                "repa_head": {key: value.detach().cpu() for key, value in repa_head.state_dict().items()},
            }
        )

    if args.folds > 1:
        stage_order = ["initial"] + [f"epoch_{epoch + 1}" for epoch in range(args.epochs)] + ["final"]
        for stage in stage_order:
            eval_payload.append(aggregate_stage(eval_payload, stage, "test"))

    payload = {
        "config": vars(args),
        "num_examples": len(examples),
        "folds": [{"fold": i, "train_indices": train, "test_indices": test} for i, (train, test) in enumerate(splits)],
        "eval": eval_payload,
        "history": history,
    }
    (out_dir / "finetune_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir / "finetune_eval.md", payload)
    write_jsonl(out_dir / "finetune_scores.jsonl", all_score_rows)
    checkpoint = {
        "folds": fold_checkpoints,
        "config": vars(args),
    }
    if args.use_lora:
        checkpoint["mistral_lora"] = get_peft_model_state_dict(videochat2.mistral_model)
    torch.save(checkpoint, out_dir / "adapter_checkpoint.pt")
    final_overall = next((row for row in reversed(eval_payload) if row.get("stage") == "final" and row.get("fold") == "all"), None)
    if final_overall is None:
        final_overall = next(row for row in reversed(eval_payload) if row.get("stage") == "final" and row.get("split") == "test")
    print(json.dumps({"output_dir": str(out_dir), "final_test": final_overall, "history": history}, indent=2))


if __name__ == "__main__":
    main()
