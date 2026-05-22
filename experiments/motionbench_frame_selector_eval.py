#!/usr/bin/env python3
"""Compare frame selection policies on cached MotionBench features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import TextEmbedder, feature_to_temporal, qa_candidate_text, parse_candidates, random_project, read_jsonl, write_jsonl  # noqa: E402


def temporal_scores(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] <= 1:
        return np.ones((vectors.shape[0],), dtype=np.float32)
    delta = np.diff(vectors, axis=0, prepend=vectors[:1])
    score = np.linalg.norm(delta, axis=1)
    if np.allclose(score.sum(), 0.0):
        score = np.ones_like(score)
    return score.astype(np.float32)


def source_indices(row: dict[str, Any], temporal_len: int) -> np.ndarray:
    sampled = np.asarray(row.get("sampled_frame_indices") or list(range(int(row.get("num_frames", temporal_len) or temporal_len))), dtype=np.int64)
    if len(sampled) == temporal_len:
        return sampled
    positions = np.linspace(0, len(sampled) - 1, temporal_len).round().astype(np.int64)
    return sampled[positions]


def select_by_scores(scores: np.ndarray, budget: int, keep_endpoints: bool) -> list[int]:
    n = len(scores)
    if budget >= n:
        return list(range(n))
    chosen = set()
    if keep_endpoints and n > 1:
        chosen.update([0, n - 1])
    for idx in np.argsort(-scores).tolist():
        chosen.add(int(idx))
        if len(chosen) >= budget:
            break
    return sorted(chosen)


def uniform_select(n: int, budget: int) -> list[int]:
    if budget >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, budget).round().astype(np.int64).tolist()))


def random_select(n: int, budget: int, seed: int) -> list[int]:
    if budget >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.arange(n), size=budget, replace=False).astype(int).tolist())


def coverage(scores: np.ndarray, selected: list[int]) -> float:
    denom = float(scores.sum())
    if denom <= 1e-8:
        return 1.0
    return float(scores[selected].sum() / denom)


def coverage_by_source(ref_scores: np.ndarray, ref_source: np.ndarray, selected_source: list[int]) -> float:
    if len(ref_scores) == 0:
        return 0.0
    selected_positions = set()
    for src in selected_source:
        nearest = int(np.argmin(np.abs(ref_source.astype(np.int64) - int(src))))
        selected_positions.add(nearest)
    return coverage(ref_scores, sorted(selected_positions))


def question_scores(vectors: np.ndarray, text_vec: np.ndarray, seed: int) -> np.ndarray:
    x = random_project(vectors, text_vec.shape[0], seed)
    denom = np.linalg.norm(x, axis=1) * max(float(np.linalg.norm(text_vec)), 1e-8)
    return ((x @ text_vec) / np.maximum(denom, 1e-8)).astype(np.float32)


def optional_accuracy_by_selector(path: str) -> dict[tuple[str, int], list[int]]:
    if not path:
        return {}
    rows = read_jsonl(Path(path))
    out: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        key = (str(row.get("selector")), int(row.get("budget", 0)))
        out.setdefault(key, []).append(int(bool(row.get("correct"))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--saliency-feature", default="wan_vae_grid_sequence")
    parser.add_argument("--pixel-feature", default="pixel_grid_sequence")
    parser.add_argument("--flow-feature", default="flow_grid_sequence")
    parser.add_argument("--budgets", default="4,8,16,32,64")
    parser.add_argument("--selectors", default="uniform,random,pixel_motion,flow_motion,wan_saliency,question_text,candidate_text")
    parser.add_argument("--keep-endpoints", action="store_true")
    parser.add_argument("--text-encoder", default="hash")
    parser.add_argument("--text-model", default="")
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--videollm-frame-results-jsonl", default="")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows_all = read_jsonl(Path(args.metadata_jsonl))
    rows = [row for row in rows_all if str(row.get("lowfps_mode", "none")) == args.mode]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    selectors = [x.strip() for x in args.selectors.split(",") if x.strip()]
    qa_texts = []
    for row in rows:
        cands = parse_candidates(row)
        if cands:
            qa_texts.append(" ".join(qa_candidate_text(row, cand) for cand in cands))
        else:
            qa_texts.append(str(row.get("question", "")))
    embedder = TextEmbedder(kind=args.text_encoder, model_name=args.text_model, dim=args.text_dim, seed=args.seed).fit(qa_texts)
    text_emb = embedder.transform(qa_texts)
    external_acc = optional_accuracy_by_selector(args.videollm_frame_results_jsonl)

    output_rows = []
    metric_rows: dict[tuple[str, int, str], list[float]] = {}
    with h5py.File(args.features_h5, "r") as h5:
        if args.mode not in h5:
            raise KeyError(f"Mode not found: {args.mode}")
        data = {name: h5[args.mode][name] for name in [args.saliency_feature, args.pixel_feature, args.flow_feature] if name in h5[args.mode]}
        for i, row in enumerate(rows):
            wan_vec = feature_to_temporal(data[args.saliency_feature][i], row, args.saliency_feature)
            ref_scores = temporal_scores(wan_vec)
            temporal_len = len(ref_scores)
            ref_src = source_indices(row, temporal_len)
            selector_scores: dict[str, np.ndarray] = {}
            if "pixel_motion" in selectors and args.pixel_feature in data:
                selector_scores["pixel_motion"] = temporal_scores(feature_to_temporal(data[args.pixel_feature][i], row, args.pixel_feature))
            if "flow_motion" in selectors and args.flow_feature in data:
                selector_scores["flow_motion"] = temporal_scores(feature_to_temporal(data[args.flow_feature][i], row, args.flow_feature))
            if "wan_saliency" in selectors:
                selector_scores["wan_saliency"] = ref_scores
            if "question_text" in selectors:
                selector_scores["question_text"] = question_scores(wan_vec, text_emb[i], args.seed + i)
            if "candidate_text" in selectors:
                selector_scores["candidate_text"] = question_scores(wan_vec, text_emb[i], args.seed + 1000 + i)

            for budget in budgets:
                for selector in selectors:
                    if selector == "uniform":
                        score_used = ref_scores
                        src = ref_src
                        positions = uniform_select(len(score_used), budget)
                    elif selector == "random":
                        score_used = ref_scores
                        src = ref_src
                        positions = random_select(len(score_used), budget, args.seed + i + budget)
                    elif selector in selector_scores:
                        score_used = selector_scores[selector]
                        src = source_indices(row, len(score_used))
                        positions = select_by_scores(score_used, budget, args.keep_endpoints)
                    else:
                        continue
                    frame_indices = src[positions].astype(int).tolist()
                    row_out = {
                        "video_id": row.get("video_id", row.get("id", i)),
                        "question_type": row.get("question_type", row.get("label")),
                        "selector": selector,
                        "budget": int(budget),
                        "temporal_positions": [int(x) for x in positions],
                        "source_frame_indices": frame_indices,
                        "wan_motion_coverage": coverage_by_source(ref_scores, ref_src, frame_indices),
                        "temporal_span": int(max(positions) - min(positions)) if positions else 0,
                    }
                    output_rows.append(row_out)
                    key = (selector, int(budget), str(row_out["question_type"]))
                    metric_rows.setdefault(key, []).append(row_out["wan_motion_coverage"])

    write_jsonl(Path(args.output_jsonl), output_rows)
    lines = ["# MotionBench Frame Selector Evaluation", ""]
    lines.append("| selector | budget | question_type | wan motion coverage | external acc |")
    lines.append("|---|---:|---|---:|---:|")
    for (selector, budget, qtype), vals in sorted(metric_rows.items()):
        acc_vals = external_acc.get((selector, budget), [])
        acc = "" if not acc_vals else f"{float(np.mean(acc_vals)):.4f}"
        lines.append(f"| {selector} | {budget} | {qtype} | {float(np.mean(vals)):.4f} | {acc} |")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output_rows), "output_jsonl": args.output_jsonl, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
