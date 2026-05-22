#!/usr/bin/env python3
"""Aggregate coarse pooling diagnostic ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POOLS = [1, 2, 4, 8, 16]
MODES = ["none", "high_motion"]
MODE_LABELS = {"none": "all", "high_motion": "high-motion"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def idx(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(item["mode"], item["feature"]): item for item in payload.get("results", [])}


def acc(item: dict[str, Any] | None) -> float | None:
    return None if not item else float(item["accuracy_mean"])


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def load_probe(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    return idx(read_json(path))


def parse_safe_float(text: str) -> float:
    return float(text.replace("p", "."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    random_probe = load_probe(result_dir / "controlled_random_same_dim.json")
    pca_probe = load_probe(result_dir / "controlled_pca_same_dim.json")
    raw_probe = load_probe(result_dir / "controlled_raw_ridge.json")

    ridge: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for path in sorted(result_dir.glob("ridge_alpha_*.json")):
        alpha = path.stem.replace("ridge_alpha_", "")
        ridge[alpha] = load_probe(path)

    sample: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for path in sorted(result_dir.glob("sample_per_type_*.json")):
        size = path.stem.replace("sample_per_type_", "")
        sample[size] = load_probe(path)

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for pool in POOLS:
            feature = f"wan_vae_grid_{pool}x{pool}"
            rows.append(
                {
                    "mode": mode,
                    "pool": f"{pool}x{pool}",
                    "random_same_dim": acc(random_probe.get((mode, feature))),
                    "pca_same_dim": acc(pca_probe.get((mode, feature))),
                    "raw_ridge": acc(raw_probe.get((mode, feature))),
                    "text_only": acc(pca_probe.get((mode, "text_only"))),
                }
            )

    ridge_rows: list[dict[str, Any]] = []
    for alpha, probe in sorted(ridge.items(), key=lambda x: parse_safe_float(x[0])):
        for mode in MODES:
            for pool in POOLS:
                feature = f"wan_vae_grid_{pool}x{pool}"
                ridge_rows.append({"alpha": alpha, "mode": mode, "pool": f"{pool}x{pool}", "acc": acc(probe.get((mode, feature)))})

    sample_rows: list[dict[str, Any]] = []
    for size, probe in sorted(sample.items(), key=lambda x: int(x[0])):
        for mode in MODES:
            for pool in POOLS:
                feature = f"wan_vae_grid_{pool}x{pool}"
                sample_rows.append({"per_type": size, "mode": mode, "pool": f"{pool}x{pool}", "acc": acc(probe.get((mode, feature)))})

    per_type_rows: list[dict[str, Any]] = []
    for mode in MODES:
        for pool in POOLS:
            feature = f"wan_vae_grid_{pool}x{pool}"
            item = pca_probe.get((mode, feature))
            if not item:
                continue
            for qtype, stats in item.get("per_question_type", {}).items():
                per_type_rows.append(
                    {
                        "mode": mode,
                        "pool": f"{pool}x{pool}",
                        "question_type": qtype,
                        "acc": float(stats["accuracy"]),
                        "correct": int(stats["correct"]),
                        "total": int(stats["total"]),
                    }
                )

    payload = {"summary": rows, "ridge": ridge_rows, "sample_size": sample_rows, "per_type": per_type_rows}
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# MotionBench Coarse Pooling Diagnostics", ""]
    lines.append("| Mode | Pool | Random same-dim | PCA same-dim | Raw ridge | Text-only |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {MODE_LABELS.get(row['mode'], row['mode'])} | {row['pool']} | {fmt(row['random_same_dim'])} | "
            f"{fmt(row['pca_same_dim'])} | {fmt(row['raw_ridge'])} | {fmt(row['text_only'])} |"
        )

    lines.extend(["", "## Ridge Alpha Sweep", ""])
    lines.append("| Alpha | Mode | Pool | Acc |")
    lines.append("|---:|---|---:|---:|")
    for row in ridge_rows:
        lines.append(f"| {row['alpha']} | {MODE_LABELS.get(row['mode'], row['mode'])} | {row['pool']} | {fmt(row['acc'])} |")

    lines.extend(["", "## Sample Size Sweep", ""])
    lines.append("| Samples/type | Mode | Pool | Acc |")
    lines.append("|---:|---|---:|---:|")
    for row in sample_rows:
        lines.append(f"| {row['per_type']} | {MODE_LABELS.get(row['mode'], row['mode'])} | {row['pool']} | {fmt(row['acc'])} |")

    lines.extend(["", "## PCA Same-Dim Per-Type", ""])
    lines.append("| Mode | Pool | Question type | Acc | Correct/total |")
    lines.append("|---|---:|---|---:|---:|")
    for row in per_type_rows:
        lines.append(
            f"| {MODE_LABELS.get(row['mode'], row['mode'])} | {row['pool']} | {row['question_type']} | "
            f"{fmt(row['acc'])} | {row['correct']}/{row['total']} |"
        )

    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary_rows": len(rows), "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
