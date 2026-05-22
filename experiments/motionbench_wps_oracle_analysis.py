#!/usr/bin/env python3
"""Oracle upper-bound analysis for the real MotionBench WPS gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_wps_experiment import (  # noqa: E402
    combined_score_map,
    cv_candidate_model,
    direct_predictions,
    direct_score_map,
    feature_distances,
    load_all_scores,
    metrics_from_predictions,
)


def acc(preds: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    use = np.ones_like(truth, dtype=bool) if mask is None else mask.astype(bool)
    correct = preds[use] == truth[use]
    return {
        "accuracy": float(correct.mean()) if correct.size else 0.0,
        "correct": int(correct.sum()),
        "total": int(correct.size),
    }


def helps_hurts(switched: np.ndarray, base: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    use = np.ones_like(truth, dtype=bool) if mask is None else mask.astype(bool)
    return {
        "helps": int(((switched == truth) & (base != truth) & use).sum()),
        "hurts": int(((switched != truth) & (base == truth) & use).sum()),
        "net": int(((switched == truth).astype(int) - (base == truth).astype(int))[use].sum()),
    }


def rel_path(repo_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else repo_root / path)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(title for title, _ in columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(payload: dict[str, Any]) -> str:
    lines = ["# MotionBench WPS Oracle Upper Bound", ""]
    lines.append("## Gate Subset")
    lines.append("")
    lines.extend(
        markdown_table(
            [payload["gate_subset"]],
            [
                ("Coverage", "coverage"),
                ("N", "total"),
                ("Base acc", "base_accuracy"),
                ("Raw Wan acc", "raw_wan_accuracy"),
                ("WPS acc", "wps_accuracy"),
            ],
        )
    )
    lines.append("")
    lines.append("## Switch Comparison")
    lines.append("")
    lines.extend(
        markdown_table(
            payload["switches"],
            [
                ("Switch", "name"),
                ("Acc", "accuracy"),
                ("Correct/Total", "correct_total"),
                ("Gain vs base", "gain_vs_base"),
                ("Helps", "helps"),
                ("Hurts", "hurts"),
                ("Net", "net"),
            ],
        )
    )
    lines.append("")
    lines.append("## Temporal-Sensitive Oracle")
    lines.append("")
    lines.extend(
        markdown_table(
            [payload["temporal_sensitive"]],
            [
                ("Coverage", "coverage"),
                ("N", "total"),
                ("Base acc", "base_accuracy"),
                ("Raw Wan acc", "raw_wan_accuracy"),
                ("Raw switch acc", "raw_switch_accuracy"),
                ("Raw switch gain", "raw_switch_gain"),
                ("Max(base, raw) oracle acc", "base_raw_oracle_accuracy"),
                ("Oracle gain", "base_raw_oracle_gain"),
            ],
        )
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(payload["verdict"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--wps-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    wps_payload = json.loads(Path(args.wps_json).read_text(encoding="utf-8"))
    cfg = dict(wps_payload["config"])
    for key in ["normal_scores_jsonl", "shuffle_scores_jsonl", "reverse_scores_jsonl", "first_scores_jsonl", "timeavg_scores_jsonl", "features_h5", "metadata_jsonl"]:
        if cfg.get(key):
            cfg[key] = rel_path(repo_root, str(cfg[key]))
    cfg_args = argparse.Namespace(**cfg)

    records, scores = load_all_scores(cfg_args)
    distances = feature_distances(cfg_args, records)
    truth = np.asarray([rec["truth"] for rec in records], dtype=np.int64)

    text_map = direct_score_map(records, scores, "text")
    pixel_map = direct_score_map(records, scores, "pixel")
    flow_map = direct_score_map(records, scores, "flow")
    raw_wan_map = direct_score_map(records, scores, "wan")
    base_map = combined_score_map(records, [text_map, pixel_map, flow_map])

    base_preds = np.asarray(direct_predictions(records, base_map), dtype=np.int64)
    raw_preds = np.asarray(direct_predictions(records, raw_wan_map), dtype=np.int64)
    wps_result = cv_candidate_model(records, scores, distances, cfg_args, "wan", include_anchor_score=False, extra_score_families=[])
    wps_preds = np.asarray(wps_result["predictions"], dtype=np.int64)

    raw_shuffle_preds = np.asarray(direct_predictions(records, direct_score_map(records, scores, "wan", "shuffle")), dtype=np.int64)
    raw_reverse_preds = np.asarray(direct_predictions(records, direct_score_map(records, scores, "wan", "reverse")), dtype=np.int64)
    temporal_sensitive = (raw_preds != raw_shuffle_preds) | (raw_preds != raw_reverse_preds)

    gate_flags = np.asarray(wps_payload["gate"]["gate_flags"], dtype=bool)
    if len(gate_flags) != len(records):
        raise ValueError(f"Gate flag length mismatch: {len(gate_flags)} vs {len(records)}")

    raw_gate_switch = np.where(gate_flags, raw_preds, base_preds)
    wps_gate_switch = np.where(gate_flags, wps_preds, base_preds)
    oracle_gate_correct = (base_preds == truth).copy()
    oracle_gate_correct[gate_flags] = (raw_preds[gate_flags] == truth[gate_flags]) | (wps_preds[gate_flags] == truth[gate_flags])

    ts_raw_switch = np.where(temporal_sensitive, raw_preds, base_preds)
    ts_oracle_correct = (base_preds == truth).copy()
    ts_oracle_correct[temporal_sensitive] = (base_preds[temporal_sensitive] == truth[temporal_sensitive]) | (raw_preds[temporal_sensitive] == truth[temporal_sensitive])

    base_all = acc(base_preds, truth)
    raw_all = acc(raw_preds, truth)
    wps_all = acc(wps_preds, truth)
    raw_switch_all = acc(raw_gate_switch, truth)
    wps_switch_all = acc(wps_gate_switch, truth)
    oracle_switch_acc = float(oracle_gate_correct.mean())
    oracle_switch_correct = int(oracle_gate_correct.sum())
    n = len(records)

    gate_base = acc(base_preds, truth, gate_flags)
    gate_raw = acc(raw_preds, truth, gate_flags)
    gate_wps = acc(wps_preds, truth, gate_flags)
    raw_hh = helps_hurts(raw_gate_switch, base_preds, truth, gate_flags)
    wps_hh = helps_hurts(wps_gate_switch, base_preds, truth, gate_flags)

    switches = [
        {
            "name": "base everywhere",
            "accuracy": base_all["accuracy"],
            "correct_total": f"{base_all['correct']}/{base_all['total']}",
            "gain_vs_base": 0.0,
            "helps": 0,
            "hurts": 0,
            "net": 0,
        },
        {
            "name": "raw Wan everywhere",
            "accuracy": raw_all["accuracy"],
            "correct_total": f"{raw_all['correct']}/{raw_all['total']}",
            "gain_vs_base": raw_all["accuracy"] - base_all["accuracy"],
            **helps_hurts(raw_preds, base_preds, truth),
        },
        {
            "name": "WPS everywhere",
            "accuracy": wps_all["accuracy"],
            "correct_total": f"{wps_all['correct']}/{wps_all['total']}",
            "gain_vs_base": wps_all["accuracy"] - base_all["accuracy"],
            **helps_hurts(wps_preds, base_preds, truth),
        },
        {
            "name": "gate -> raw Wan",
            "accuracy": raw_switch_all["accuracy"],
            "correct_total": f"{raw_switch_all['correct']}/{raw_switch_all['total']}",
            "gain_vs_base": raw_switch_all["accuracy"] - base_all["accuracy"],
            **raw_hh,
        },
        {
            "name": "gate -> WPS",
            "accuracy": wps_switch_all["accuracy"],
            "correct_total": f"{wps_switch_all['correct']}/{wps_switch_all['total']}",
            "gain_vs_base": wps_switch_all["accuracy"] - base_all["accuracy"],
            **wps_hh,
        },
        {
            "name": "gate -> max(raw Wan, WPS) oracle",
            "accuracy": oracle_switch_acc,
            "correct_total": f"{oracle_switch_correct}/{n}",
            "gain_vs_base": oracle_switch_acc - base_all["accuracy"],
            "helps": int((oracle_gate_correct & (base_preds != truth)).sum()),
            "hurts": int(((~oracle_gate_correct) & (base_preds == truth)).sum()),
            "net": oracle_switch_correct - base_all["correct"],
        },
    ]

    ts_base = acc(base_preds, truth, temporal_sensitive)
    ts_raw = acc(raw_preds, truth, temporal_sensitive)
    ts_switch = acc(ts_raw_switch, truth)
    ts_oracle_acc = float(ts_oracle_correct.mean())
    ts_hh = helps_hurts(ts_raw_switch, base_preds, truth, temporal_sensitive)

    gate_subset = {
        "coverage": float(gate_flags.mean()),
        "total": int(gate_flags.sum()),
        "base_accuracy": gate_base["accuracy"],
        "raw_wan_accuracy": gate_raw["accuracy"],
        "wps_accuracy": gate_wps["accuracy"],
        "base_correct": gate_base["correct"],
        "raw_wan_correct": gate_raw["correct"],
        "wps_correct": gate_wps["correct"],
    }
    temporal_payload = {
        "coverage": float(temporal_sensitive.mean()),
        "total": int(temporal_sensitive.sum()),
        "base_accuracy": ts_base["accuracy"],
        "raw_wan_accuracy": ts_raw["accuracy"],
        "raw_switch_accuracy": ts_switch["accuracy"],
        "raw_switch_gain": ts_switch["accuracy"] - base_all["accuracy"],
        "raw_switch_helps": ts_hh["helps"],
        "raw_switch_hurts": ts_hh["hurts"],
        "base_raw_oracle_accuracy": ts_oracle_acc,
        "base_raw_oracle_gain": ts_oracle_acc - base_all["accuracy"],
        "base_raw_oracle_correct": int(ts_oracle_correct.sum()),
    }

    verdict = (
        "Gate direction has limited headroom: using the existing gate to switch to raw Wan is "
        f"{raw_switch_all['accuracy']:.4f}, while the gate-only max(raw Wan, WPS) oracle is "
        f"{oracle_switch_acc:.4f}. The temporal-sensitive perfect detector raw-Wan switch is "
        f"{ts_switch['accuracy']:.4f}, a {ts_switch['accuracy'] - base_all['accuracy']:+.4f} gain over base."
    )

    out = {
        "source_wps_json": str(args.wps_json),
        "rows": n,
        "base_everywhere": base_all,
        "raw_wan_everywhere": raw_all,
        "wps_everywhere": wps_all,
        "gate_subset": gate_subset,
        "switches": switches,
        "temporal_sensitive": temporal_payload,
        "per_question_type": {
            "base": metrics_from_predictions(records, base_preds.tolist())["per_question_type"],
            "raw_wan": metrics_from_predictions(records, raw_preds.tolist())["per_question_type"],
            "wps": metrics_from_predictions(records, wps_preds.tolist())["per_question_type"],
            "gate_raw_switch": metrics_from_predictions(records, raw_gate_switch.tolist())["per_question_type"],
            "gate_wps_switch": metrics_from_predictions(records, wps_gate_switch.tolist())["per_question_type"],
        },
        "verdict": verdict,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(build_report(out), encoding="utf-8")
    print(json.dumps({"rows": n, "gate_n": gate_subset["total"], "temporal_sensitive_n": temporal_payload["total"], "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
