#!/usr/bin/env python3
"""Build a local MVBench subset JSONL compatible with extract_wan_features.py.

The Hugging Face OpenGVLab/MVBench repository contains annotation JSON files.
Actual videos are from the source benchmarks and must exist locally. This script
resolves each annotation's video filename under one or more video roots and
writes a JSONL subset plus a missing-video report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_JSON_DIR = (
    Path.home()
    / ".cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots"
    / "230a2d4fac8900333c61754641c7a13e069ac9c6/json"
)


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_letter(candidates: list[str], answer: str) -> str:
    answer = str(answer).strip()
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()
    for idx, candidate in enumerate(candidates):
        if answer == str(candidate).strip():
            return chr(ord("A") + idx)
    for idx, candidate in enumerate(candidates):
        if answer in str(candidate) or str(candidate) in answer:
            return chr(ord("A") + idx)
    return "A"


def build_video_index(roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    suffixes = {".mp4", ".avi", ".webm", ".mkv", ".mov"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                index[path.name].append(path)
                index[path.stem].append(path)
    return index


def resolve_video(video_name: str, roots: list[Path], index: dict[str, list[Path]]) -> Path | None:
    raw = Path(video_name)
    if raw.is_absolute() and raw.exists():
        return raw
    for root in roots:
        direct = root / video_name
        if direct.exists():
            return direct
    matches = index.get(raw.name) or index.get(raw.stem) or []
    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR))
    parser.add_argument("--video-root", action="append", default=[])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--missing-jsonl", required=True)
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    import numpy as np

    json_dir = Path(args.json_dir).expanduser()
    roots = [Path(x).expanduser() for x in args.video_root if x]
    task_filter = {x.strip() for x in args.tasks.split(",") if x.strip()}
    rng = np.random.default_rng(args.seed)
    index = build_video_index(roots)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        task = json_path.stem
        if task_filter and task not in task_filter:
            continue
        raw_rows = read_json(json_path)
        if not raw_rows:
            continue
        order = rng.permutation(len(raw_rows)).tolist()
        selected = 0
        task_missing = 0
        for idx in order:
            if selected >= args.per_task:
                break
            item = raw_rows[idx]
            video_name = str(item.get("video", ""))
            video_path = resolve_video(video_name, roots, index)
            candidates = [str(x) for x in item.get("candidates", [])]
            if not video_path or not candidates:
                task_missing += 1
                missing.append({"task": task, "video": video_name, "question": item.get("question"), "reason": "missing_video_or_candidates"})
                continue
            letter = answer_letter(candidates, str(item.get("answer", "")))
            answer_idx = max(0, min(len(candidates) - 1, ord(letter) - ord("A")))
            rows.append(
                {
                    "video_id": f"mvbench_{task}_{Path(video_name).stem}_{selected}",
                    "path": str(video_path),
                    "label": task,
                    "question_type": task,
                    "task": task,
                    "question": item.get("question", ""),
                    "candidates": candidates,
                    "answer": letter,
                    "answer_text": candidates[answer_idx],
                    "source_dataset": "MVBench",
                    "source_annotation": str(json_path),
                    "source_video": video_name,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "split": "train" if selected < int(round(args.per_task * 0.7)) else "test",
                }
            )
            selected += 1
        task_counts[task] = selected
        if selected == 0 and task_missing == 0:
            missing.append({"task": task, "reason": "no_rows"})

    write_jsonl(Path(args.output_jsonl), rows)
    write_jsonl(Path(args.missing_jsonl), missing)
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "missing_jsonl": args.missing_jsonl,
                "rows": len(rows),
                "task_counts": task_counts,
                "video_roots": [str(x) for x in roots],
                "json_dir": str(json_dir),
                "missing": len(missing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
