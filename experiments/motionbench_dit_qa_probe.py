#!/usr/bin/env python3
"""Question/candidate-conditioned Wan-DiT hidden-feature reranking probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motionbench_common import (  # noqa: E402
    TextEmbedder,
    answer_index,
    bootstrap_ci,
    parse_candidates,
    qa_candidate_text,
    random_project,
    read_jsonl,
    row_id,
    rows_by_mode,
    write_jsonl,
)
from wan_feature_sanity import MODEL_ID  # noqa: E402
from wan_next_experiments import pool_dit_hidden, register_hooks  # noqa: E402


def valid_rows(rows: list[dict[str, Any]]) -> list[int]:
    keep = []
    for i, row in enumerate(rows):
        if len(parse_candidates(row)) >= 2 and answer_index(row) is not None:
            keep.append(i)
    return keep


def latent_from_grid(grid: np.ndarray) -> np.ndarray:
    # H5 grid layout is [T, H, W, C]; transformer expects [C, T, H, W].
    return np.asarray(grid, dtype=np.float32).transpose(3, 0, 1, 2)


def encode_candidate_dit(
    transformer,
    scheduler,
    latents: np.ndarray,
    prompt_embeds: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    outputs = []
    latent_shape = tuple(latents.shape[1:])
    for start in range(0, len(latents), args.batch_size):
        batch_latents = torch.from_numpy(latents[start : start + args.batch_size]).to(device=device, dtype=dtype)
        batch_text = torch.from_numpy(prompt_embeds[start : start + args.batch_size]).to(device=device, dtype=dtype)
        if batch_text.ndim == 2:
            batch_text = batch_text[:, None, :]
        bsz = batch_latents.shape[0]
        t = torch.full((bsz,), int(args.timestep), device=device, dtype=torch.long)
        noise = torch.randn(batch_latents.shape, generator=generator, device=device, dtype=dtype)
        noisy = scheduler.add_noise(batch_latents, noise, t)
        store: dict[int, torch.Tensor] = {}
        handles = register_hooks(transformer, [args.layer], store)
        with torch.no_grad():
            transformer(hidden_states=noisy, timestep=t, encoder_hidden_states=batch_text, return_dict=False)
        for handle in handles:
            handle.remove()
        hidden = store[args.layer]
        pooled = pool_dit_hidden(hidden, latent_shape, args.pooling)
        outputs.append(np.asarray(pooled, dtype=np.float32).reshape(bsz, -1))
    return np.concatenate(outputs, axis=0)


def make_candidate_arrays(h5, rows, mode, feature_name, args):
    grid = h5[mode][feature_name]
    keep = valid_rows(rows)
    latents = []
    texts = []
    owners = []
    y_binary = []
    answer_by_row = {}
    for row_idx in keep:
        cands = parse_candidates(rows[row_idx])
        ans = answer_index(rows[row_idx])
        answer_by_row[row_idx] = ans
        latent = latent_from_grid(grid[row_idx])
        for cand_idx, cand in enumerate(cands):
            latents.append(latent)
            texts.append(qa_candidate_text(rows[row_idx], cand))
            owners.append(row_idx)
            y_binary.append(1 if cand_idx == ans else 0)
    embedder = TextEmbedder(
        kind=args.text_encoder,
        model_name=args.text_model,
        dim=args.text_dim,
        batch_size=args.text_batch_size,
        device=args.text_device,
        seed=args.seed,
    ).fit(texts)
    text = embedder.transform(texts)
    text = random_project(text, args.text_projection_dim, args.seed + 900).astype(np.float32)
    return np.stack(latents), text, np.asarray(owners), np.asarray(y_binary), answer_by_row


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# MotionBench QA-Conditioned DiT Probe", ""]
    lines.append("| mode | feature | layer | timestep | pooling | acc | CI low | CI high | correct/total |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for item in payload["results"]:
        ci = item["accuracy_ci95"]
        lines.append(
            f"| {item['mode']} | {item['feature']} | {item['layer']} | {item['timestep']} | {item['pooling']} | "
            f"{item['accuracy']:.4f} | {ci[0]:.4f} | {ci[1]:.4f} | {item['correct']}/{item['total']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-h5", required=True)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-scores-jsonl", required=True)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--feature-name", default="wan_vae_grid_sequence")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--timestep", type=int, default=900)
    parser.add_argument("--pooling", default="token_mean", choices=["token_mean", "temporal_sequence", "temporal_delta", "spatial_tokens"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--text-encoder", default="hash")
    parser.add_argument("--text-model", default="")
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--text-projection-dim", type=int, default=4096)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows_all = read_jsonl(Path(args.metadata_jsonl))
    rows = rows_by_mode(rows_all).get(args.mode, [])
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=dtype).to(device)
    transformer.eval()
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    with h5py.File(args.features_h5, "r") as h5:
        if args.mode not in h5 or args.feature_name not in h5[args.mode]:
            raise KeyError(f"Missing {args.mode}/{args.feature_name}")
        latents, text, owners, y_binary, answer_by_row = make_candidate_arrays(h5, rows, args.mode, args.feature_name, args)
        dit_x = encode_candidate_dit(transformer, scheduler, latents, text, args, device, dtype)

    unique_rows = np.asarray(sorted(set(owners.tolist())), dtype=np.int64)
    stratify = np.asarray([str(rows[int(i)].get("question_type", "all")) for i in unique_rows])
    folds = max(2, min(args.folds, np.min(np.unique(stratify, return_counts=True)[1])))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    score_rows = []
    correct = []
    all_true = []
    all_pred = []
    for split_id, (train_rows_local, test_rows_local) in enumerate(cv.split(unique_rows, stratify)):
        train_rows = set(unique_rows[train_rows_local].tolist())
        test_rows = set(unique_rows[test_rows_local].tolist())
        train_mask = np.asarray([owner in train_rows for owner in owners])
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed + split_id))
        model.fit(dit_x[train_mask], y_binary[train_mask])
        for row_idx in sorted(test_rows):
            cand_mask = owners == row_idx
            scores = model.decision_function(dit_x[cand_mask])
            pred = int(np.argmax(scores))
            truth = int(answer_by_row[row_idx])
            correct.append(int(pred == truth))
            all_true.append(truth)
            all_pred.append(pred)
            cands = parse_candidates(rows[int(row_idx)])
            score_rows.append(
                {
                    "split": split_id,
                    "mode": args.mode,
                    "feature": f"dit_l{args.layer:02d}_t{args.timestep}_{args.pooling}_qa_conditioned",
                    "video_id": row_id(rows[int(row_idx)], int(row_idx)),
                    "row_index": int(row_idx),
                    "question_type": rows[int(row_idx)].get("question_type", rows[int(row_idx)].get("label")),
                    "answer_index": truth,
                    "prediction_index": pred,
                    "correct": bool(pred == truth),
                    "candidates": [
                        {"index": i, "letter": cands[i]["letter"], "text": cands[i]["text"], "score": float(scores[i])}
                        for i in range(len(cands))
                    ],
                }
            )
    arr = np.asarray(correct, dtype=np.float32)
    ci = bootstrap_ci(arr, seed=args.seed)
    result = {
        "mode": args.mode,
        "feature": args.feature_name,
        "layer": args.layer,
        "timestep": args.timestep,
        "pooling": args.pooling,
        "accuracy": float(arr.mean()) if arr.size else 0.0,
        "correct": int(arr.sum()),
        "total": int(arr.size),
        "accuracy_ci95": [ci[0], ci[1]],
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
    }
    payload = {"config": vars(args), "results": [result]}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    write_jsonl(Path(args.output_scores_jsonl), score_rows)
    print(json.dumps({"results": 1, "scores": len(score_rows), "output_json": args.output_json}, indent=2))


if __name__ == "__main__":
    main()
