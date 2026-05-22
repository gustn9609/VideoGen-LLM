#!/usr/bin/env python3
"""Wan-TRD token relation distillation for segment-level Wan features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_repa_common import (  # noqa: E402
    aligned_rows_by_mode,
    copy_h5_attrs,
    l2_normalize_rows,
    random_project_fit,
    reduce_segments,
    safe_name,
    select_h5_rows,
    stable_seed,
    tokens_from_feature_array,
    write_jsonl,
)


class RelationAdapter(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, align_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, align_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def relation_matrix(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    rel = torch.matmul(x, x.transpose(-1, -2))
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    return rel * pair_mask


def relation_loss(z: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rz = relation_matrix(z, mask)
    rt = relation_matrix(teacher, mask)
    pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    denom = pair_mask.sum().clamp_min(1.0)
    rel = ((rz - rt).pow(2) * pair_mask).sum() / denom
    direct = 1.0 - (F.normalize(z, dim=-1) * F.normalize(teacher, dim=-1)).sum(dim=-1)
    direct = (direct * mask).sum() / mask.sum().clamp_min(1.0)
    return rel, direct


def prepare_teacher_segments(data: np.ndarray, segments: int, align_dim: int, seed: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = np.repeat(data[:, None, :], segments, axis=1)
    elif data.ndim > 3:
        data = data.reshape(data.shape[0], data.shape[1], -1)
    if data.shape[1] != segments:
        old = data
        boundaries = np.linspace(0, old.shape[1], segments + 1).round().astype(int)
        pooled = np.zeros((old.shape[0], segments, old.shape[2]), dtype=np.float32)
        for i in range(segments):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            if end <= start:
                end = min(old.shape[1], start + 1)
            pooled[:, i] = old[:, start:end].mean(axis=1)
        data = pooled
    flat = data.reshape(-1, data.shape[-1])
    proj = random_project_fit(flat.shape[1], align_dim, seed)
    projected = flat @ proj if flat.shape[1] == proj.shape[0] else flat[:, :align_dim]
    if projected.shape[1] < align_dim:
        pad = np.zeros((projected.shape[0], align_dim - projected.shape[1]), dtype=np.float32)
        projected = np.concatenate([projected, pad], axis=1)
    projected = l2_normalize_rows(projected).reshape(data.shape[0], data.shape[1], align_dim)
    return projected.astype(np.float32)


def train_relation_adapter(
    wan_segments: np.ndarray,
    wan_mask: np.ndarray,
    teacher_segments: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model = RelationAdapter(wan_segments.shape[-1], args.hidden_dim, args.align_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x = torch.from_numpy(wan_segments).to(device)
    mask = torch.from_numpy(wan_mask).to(device)
    teacher = torch.from_numpy(teacher_segments).to(device)
    n = x.shape[0]
    initial_rel = None
    final_rel = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n, device=device)
        epoch_rel = []
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            z = model(x[idx])
            rel, direct = relation_loss(z, teacher[idx], mask[idx])
            loss = rel + float(args.lambda_direct) * direct
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_rel.append(float(rel.detach().cpu()))
        if epoch == 0:
            initial_rel = float(np.mean(epoch_rel))
        final_rel = float(np.mean(epoch_rel))
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            outs.append(model(x[start : start + args.batch_size]).cpu().numpy())
    aligned = np.concatenate(outs, axis=0).astype(np.float32)
    stats = {"initial_relation_loss": float(initial_rel or 0.0), "final_relation_loss": float(final_rel or 0.0)}
    return aligned, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-features-h5", required=True)
    parser.add_argument("--wan-metadata-jsonl", required=True)
    parser.add_argument("--teacher-h5", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-metadata-jsonl", required=True)
    parser.add_argument("--report-json", default="")
    parser.add_argument("--report-md", default="")
    parser.add_argument("--feature-names", default="wan_vae_grid_1x1,wan_vae_grid_2x2")
    parser.add_argument("--teacher-feature", default="motion_teacher_segments")
    parser.add_argument("--modes", default="high_motion+camera_comp")
    parser.add_argument("--output-prefix", default="wan_trd")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--align-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-direct", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows_by = aligned_rows_by_mode(args.wan_metadata_jsonl)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    features = [x.strip() for x in args.feature_names.split(",") if x.strip()]
    output_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"config": vars(args), "results": []}

    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.wan_features_h5, "r") as wan_h5, h5py.File(args.teacher_h5, "r") as teacher_h5, h5py.File(args.output_h5, "w") as out_h5:
        copy_h5_attrs(wan_h5, out_h5)
        out_h5.attrs["feature_extractor_version"] = "motionbench_relation_distill_v1"
        for mode in modes:
            if mode not in wan_h5 or mode not in teacher_h5:
                continue
            rows = [dict(row) for row in rows_by.get(mode, [])]
            if not rows or args.teacher_feature not in teacher_h5[mode]:
                continue
            group = out_h5.create_group(mode)
            copy_h5_attrs(wan_h5[mode], group)
            feature_shapes: dict[str, list[int]] = {}
            teacher_data = select_h5_rows(teacher_h5[mode][args.teacher_feature][:], rows)
            teacher = prepare_teacher_segments(
                teacher_data,
                args.segments,
                args.align_dim,
                stable_seed(mode, args.teacher_feature, base=args.seed),
            )
            for feature in features:
                if feature not in wan_h5[mode]:
                    continue
                wan_data = select_h5_rows(wan_h5[mode][feature][:], rows)
                tokens, token_mask = tokens_from_feature_array(wan_data, rows, feature)
                segments, segment_mask = reduce_segments(tokens, token_mask, args.segments)
                aligned, stats = train_relation_adapter(segments, segment_mask, teacher, args, device)
                out_name = f"{args.output_prefix}_{safe_name(feature)}"
                group.create_dataset(out_name, data=aligned, compression="gzip")
                pooled_name = f"{out_name}_pooled"
                pooled = (aligned * segment_mask[..., None]).sum(axis=1) / np.maximum(segment_mask.sum(axis=1, keepdims=True), 1.0)
                group.create_dataset(pooled_name, data=pooled.astype(np.float32), compression="gzip")
                feature_shapes[out_name] = [int(x) for x in aligned.shape[1:]]
                feature_shapes[pooled_name] = [int(x) for x in pooled.shape[1:]]
                report["results"].append(
                    {
                        "mode": mode,
                        "source_feature": feature,
                        "output_feature": out_name,
                        "pooled_feature": pooled_name,
                        "rows": int(aligned.shape[0]),
                        "segments": int(aligned.shape[1]),
                        "align_dim": int(aligned.shape[2]),
                        **stats,
                    }
                )
            group.attrs["feature_shapes"] = json.dumps(feature_shapes)
            for row in rows:
                row["feature_extractor_version"] = "motionbench_relation_distill_v1"
                row["feature_shapes"] = feature_shapes
                output_rows.append(row)

    write_jsonl(Path(args.output_metadata_jsonl), output_rows)
    report_json = Path(args.report_json) if args.report_json else Path(args.output_h5).with_suffix(".report.json")
    report_md = Path(args.report_md) if args.report_md else Path(args.output_h5).with_suffix(".report.md")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# MotionBench Wan-TRD Relation Distillation", ""]
    lines.append("| mode | source | output | rows | segments | dim | initial rel loss | final rel loss |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for item in report["results"]:
        lines.append(
            f"| {item['mode']} | {item['source_feature']} | {item['output_feature']} | {item['rows']} | "
            f"{item['segments']} | {item['align_dim']} | {item['initial_relation_loss']:.4f} | {item['final_relation_loss']:.4f} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results": len(report["results"]), "output_h5": args.output_h5, "report_json": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()
