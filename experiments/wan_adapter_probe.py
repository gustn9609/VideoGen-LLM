#!/usr/bin/env python3
"""Train a small factorized Wan motion-token adapter on cached grid features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_motion_adapter import FactorizedMotionResampler  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_grid(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim == 5:
        return x
    if x.ndim == 3:
        return x[:, :, None, None, :]
    raise ValueError(f"Expected [N,T,H,W,C] or [N,T,C], got {x.shape}")


def make_split(rows: list[dict[str, Any]], labels: np.ndarray, args: argparse.Namespace, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if args.split_column and args.split_column in rows[0]:
        train_idx = np.asarray([i for i, row in enumerate(rows) if str(row.get(args.split_column)) == args.train_split], dtype=np.int64)
        test_idx = np.asarray([i for i, row in enumerate(rows) if str(row.get(args.split_column)) == args.test_split], dtype=np.int64)
        if len(train_idx) and len(test_idx):
            return train_idx, test_idx

    all_idx = np.arange(len(rows), dtype=np.int64)
    class_counts = np.bincount(labels)
    stratify = labels if len(class_counts) > 1 and np.all(class_counts[class_counts > 0] >= 2) else None
    train_idx, test_idx = train_test_split(all_idx, test_size=args.test_fraction, random_state=seed, stratify=stratify)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


class AdapterClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        tokens_per_frame: int,
        output_tokens: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.adapter = FactorizedMotionResampler(
            input_dim=input_dim,
            output_dim=hidden_dim,
            tokens_per_frame=tokens_per_frame,
            output_tokens=output_tokens,
            hidden_dim=hidden_dim,
            num_heads=heads,
            num_layers=layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * output_tokens),
            nn.Linear(hidden_dim * output_tokens, num_classes),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        tokens = self.adapter(grid)
        return self.head(tokens.flatten(1))


def run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for x, _ in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.int64)


def evaluate_mode(h5: h5py.File, mode: str, rows: list[dict[str, Any]], args: argparse.Namespace, seed: int) -> dict[str, Any] | None:
    if args.feature_name not in h5[mode]:
        return None
    if args.label_column not in rows[0]:
        raise KeyError(f"Label column '{args.label_column}' missing from metadata")

    grid = normalize_grid(h5[mode][args.feature_name][:])
    labels_text = [str(row[args.label_column]) for row in rows]
    labels = LabelEncoder().fit_transform(labels_text).astype(np.int64)
    if len(np.unique(labels)) < 2:
        return None

    if args.max_samples > 0 and len(labels) > args.max_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(np.arange(len(labels)), size=args.max_samples, replace=False)
        keep.sort()
        grid = grid[keep]
        labels = labels[keep]
        rows = [rows[int(i)] for i in keep]

    train_idx, test_idx = make_split(rows, labels, args, seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = AdapterClassifier(
        input_dim=int(grid.shape[-1]),
        num_classes=int(len(np.unique(labels))),
        hidden_dim=args.hidden_dim,
        tokens_per_frame=args.tokens_per_frame,
        output_tokens=args.output_tokens,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    train_ds = TensorDataset(torch.from_numpy(grid[train_idx]), torch.from_numpy(labels[train_idx]))
    test_ds = TensorDataset(torch.from_numpy(grid[test_idx]), torch.from_numpy(labels[test_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = 0.0
    for _ in range(args.epochs):
        last_loss = run_epoch(model, train_loader, optimizer, device)
    pred = predict(model, test_loader, device)
    y_test = labels[test_idx]
    return {
        "mode": mode,
        "feature": args.feature_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "classes": int(len(np.unique(labels))),
        "feature_shape": [int(x) for x in grid.shape[1:]],
        "output_tokens": int(args.output_tokens),
        "tokens_per_frame": int(args.tokens_per_frame),
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "last_train_loss": float(last_loss),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Wan Adapter Probe Results",
        "",
        "| mode | feature | acc | bal_acc | train | test | tokens | hidden |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {mode} | {feature} | {accuracy:.3f} | {balanced_accuracy:.3f} | {train_size} | {test_size} | {output_tokens} | {hidden_dim} |".format(
                **item
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--feature-name", default="wan_vae_grid_sequence")
    parser.add_argument("--modes", default="")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tokens-per-frame", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = read_jsonl(Path(args.metadata_jsonl))
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_mode.setdefault(str(row.get("lowfps_mode", "none")), []).append(row)

    results = []
    requested_modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    with h5py.File(args.features_h5, "r") as h5:
        modes = requested_modes or list(h5.keys())
        for mode in modes:
            if mode not in h5 or mode not in rows_by_mode:
                continue
            result = evaluate_mode(h5, mode, rows_by_mode[mode], args, args.seed)
            if result is not None:
                results.append(result)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), results)
    print(json.dumps({"results": len(results), "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
