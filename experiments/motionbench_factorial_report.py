#!/usr/bin/env python3
"""Aggregate MotionBench factorial mini-sweep probe outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SAMPLING_LABELS = {
    "none": "all",
    "high_motion": "high-motion",
    "camera_comp": "camera-comp",
    "high_motion+camera_comp": "high-motion+camera-comp",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def result_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(item["mode"], item["feature"]): item for item in payload.get("results", [])}


def acc(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    return float(item["accuracy_mean"])


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def zscore(scores: np.ndarray) -> np.ndarray:
    std = float(scores.std())
    if std <= 1e-8:
        return scores - float(scores.mean())
    return (scores - float(scores.mean())) / std


def ensemble_accuracy(scores_path: Path, mode: str, features: list[str]) -> float | None:
    rows = read_jsonl(scores_path)
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("mode") != mode or row.get("feature") not in features:
            continue
        grouped[(int(row["split"]), int(row["row_index"]))][str(row["feature"])] = row
    correct = []
    for feature_rows in grouped.values():
        if not all(feature in feature_rows for feature in features):
            continue
        candidate_scores = []
        answer = None
        for feature in features:
            row = feature_rows[feature]
            answer = int(row["answer_index"])
            scores = np.asarray([float(x["score"]) for x in row["candidates"]], dtype=np.float64)
            candidate_scores.append(zscore(scores))
        mean_scores = np.mean(np.stack(candidate_scores, axis=0), axis=0)
        correct.append(int(np.argmax(mean_scores) == answer))
    if not correct:
        return None
    return float(np.mean(correct))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--pools", default="1,2,4")
    parser.add_argument("--frames", default="8,16")
    parser.add_argument("--samplings", default="none,high_motion,camera_comp,high_motion+camera_comp")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    pools = [int(x) for x in args.pools.split(",") if x.strip()]
    frames_list = [int(x) for x in args.frames.split(",") if x.strip()]
    samplings = [x.strip() for x in args.samplings.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    for pool in pools:
        wan_feature = f"wan_vae_grid_{pool}x{pool}"
        for frames in frames_list:
            job_dir = result_dir / f"pool_{pool}_frames_{frames}"
            full_path = job_dir / "probe_full.json"
            if not full_path.exists():
                continue
            full = result_index(read_json(full_path))
            shuffled = result_index(read_json(job_dir / "probe_shuffled.json")) if (job_dir / "probe_shuffled.json").exists() else {}
            reversed_payload = result_index(read_json(job_dir / "probe_reversed.json")) if (job_dir / "probe_reversed.json").exists() else {}
            first = result_index(read_json(job_dir / "probe_first_frame_only.json")) if (job_dir / "probe_first_frame_only.json").exists() else {}
            for sampling in samplings:
                sampling_label = SAMPLING_LABELS.get(sampling, sampling)
                wan = acc(full.get((sampling, wan_feature)))
                text = acc(full.get((sampling, "text_only")))
                pixel = acc(full.get((sampling, "pixel_grid_sequence")))
                flow = acc(full.get((sampling, "flow_grid_sequence")))
                wan_shuffle = acc(shuffled.get((sampling, wan_feature)))
                wan_reverse = acc(reversed_payload.get((sampling, wan_feature)))
                wan_first = acc(first.get((sampling, wan_feature)))
                row = {
                    "pool": f"{pool}x{pool}",
                    "frames": frames,
                    "sampling": sampling_label,
                    "mode": sampling,
                    "wan_acc": wan,
                    "text_only_acc": text,
                    "pixel_acc": pixel,
                    "flow_acc": flow,
                    "text_only_gain": diff(wan, text),
                    "pixel_gap": diff(wan, pixel),
                    "flow_gap": diff(wan, flow),
                    "normal_shuffle_gap": diff(wan, wan_shuffle),
                    "normal_reverse_gap": diff(wan, wan_reverse),
                    "normal_first_frame_gap": diff(wan, wan_first),
                }
                rows.append(row)

                score_path = job_dir / "probe_full_scores.jsonl"
                ensemble_rows.append(
                    {
                        "pool": f"{pool}x{pool}",
                        "frames": frames,
                        "sampling": sampling_label,
                        "wan_pixel_ensemble_acc": ensemble_accuracy(score_path, sampling, [wan_feature, "pixel_grid_sequence"]),
                        "wan_flow_ensemble_acc": ensemble_accuracy(score_path, sampling, [wan_feature, "flow_grid_sequence"]),
                        "wan_pixel_flow_ensemble_acc": ensemble_accuracy(
                            score_path,
                            sampling,
                            [wan_feature, "pixel_grid_sequence", "flow_grid_sequence"],
                        ),
                    }
                )

                wan_item = full.get((sampling, wan_feature))
                text_item = full.get((sampling, "text_only"))
                pixel_item = full.get((sampling, "pixel_grid_sequence"))
                flow_item = full.get((sampling, "flow_grid_sequence"))
                if wan_item:
                    qtypes = sorted(wan_item.get("per_question_type", {}).keys())
                    for qtype in qtypes:
                        wan_q = wan_item["per_question_type"].get(qtype, {}).get("accuracy")
                        text_q = text_item.get("per_question_type", {}).get(qtype, {}).get("accuracy") if text_item else None
                        pixel_q = pixel_item.get("per_question_type", {}).get(qtype, {}).get("accuracy") if pixel_item else None
                        flow_q = flow_item.get("per_question_type", {}).get(qtype, {}).get("accuracy") if flow_item else None
                        per_type_rows.append(
                            {
                                "pool": f"{pool}x{pool}",
                                "frames": frames,
                                "sampling": sampling_label,
                                "question_type": qtype,
                                "wan_acc": wan_q,
                                "text_only_gain": diff(wan_q, text_q),
                                "pixel_gap": diff(wan_q, pixel_q),
                                "flow_gap": diff(wan_q, flow_q),
                            }
                        )

    out_payload = {
        "rows": rows,
        "ensemble_rows": ensemble_rows,
        "per_type_rows": per_type_rows,
    }
    Path(args.output_json).write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "pool",
        "frames",
        "sampling",
        "wan_acc",
        "text_only_gain",
        "pixel_gap",
        "flow_gap",
        "normal_shuffle_gap",
        "normal_reverse_gap",
        "normal_first_frame_gap",
        "text_only_acc",
        "pixel_acc",
        "flow_acc",
    ]
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = ["# MotionBench Factorial Mini-Sweep", ""]
    lines.append("| Pool | Frames | Sampling | Acc | Text-only gain | Pixel gap | Flow gap | Normal-shuffle gap |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['pool']} | {row['frames']} | {row['sampling']} | {fmt(row['wan_acc'])} | "
            f"{fmt(row['text_only_gain'])} | {fmt(row['pixel_gap'])} | {fmt(row['flow_gap'])} | "
            f"{fmt(row['normal_shuffle_gap'])} |"
        )

    lines.extend(["", "## Temporal Robustness", ""])
    lines.append("| Pool | Frames | Sampling | Normal-reverse gap | Normal-first-frame gap |")
    lines.append("|---:|---:|---|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['pool']} | {row['frames']} | {row['sampling']} | "
            f"{fmt(row['normal_reverse_gap'])} | {fmt(row['normal_first_frame_gap'])} |"
        )

    lines.extend(["", "## Wan + Pixel/Flow Ensembles", ""])
    lines.append("| Pool | Frames | Sampling | Wan+Pixel | Wan+Flow | Wan+Pixel+Flow |")
    lines.append("|---:|---:|---|---:|---:|---:|")
    for row in ensemble_rows:
        lines.append(
            f"| {row['pool']} | {row['frames']} | {row['sampling']} | "
            f"{fmt(row['wan_pixel_ensemble_acc'])} | {fmt(row['wan_flow_ensemble_acc'])} | "
            f"{fmt(row['wan_pixel_flow_ensemble_acc'])} |"
        )

    lines.extend(["", "## Per-Type Wan Incremental View", ""])
    lines.append("| Pool | Frames | Sampling | Type | Wan Acc | Text-only gain | Pixel gap | Flow gap |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---:|")
    for row in per_type_rows:
        lines.append(
            f"| {row['pool']} | {row['frames']} | {row['sampling']} | {row['question_type']} | "
            f"{fmt(row['wan_acc'])} | {fmt(row['text_only_gain'])} | {fmt(row['pixel_gap'])} | {fmt(row['flow_gap'])} |"
        )

    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "per_type_rows": len(per_type_rows), "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
