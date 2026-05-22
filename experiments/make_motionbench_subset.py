#!/usr/bin/env python3
"""Build a small local MotionBench subset JSONL from Hugging Face files.

The output JSONL is compatible with `extract_wan_features.py`.
It downloads only the selected mp4 files, instead of cloning the whole dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "zai-org/MotionBench"
META_FILE = "MotionBench/video_info.meta.jsonl"
VIDEO_PREFIX = "MotionBench/public-dataset"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_text(question: str, answer: str) -> str:
    prefix = f"{answer}."
    for line in question.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--per-question-type", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()

    import numpy as np

    rng = np.random.default_rng(args.seed)
    meta_path = Path(hf_hub_download(args.repo_id, META_FILE, repo_type="dataset"))
    raw_rows = read_jsonl(meta_path)
    available_files = set(HfApi().list_repo_files(args.repo_id, repo_type="dataset"))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        qa = (row.get("qa") or [{}])[0]
        answer = qa.get("answer")
        if answer in (None, "NA", ""):
            continue
        qtype = row.get("question_type")
        video_path = row.get("video_path")
        if not qtype or not video_path:
            continue
        repo_video_path = f"{VIDEO_PREFIX}/{video_path}"
        if repo_video_path not in available_files:
            continue
        buckets[str(qtype)].append(row)

    out_rows = []
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    for qtype in sorted(buckets):
        candidates = buckets[qtype]
        take = min(args.per_question_type, len(candidates))
        selected_indices = rng.choice(np.arange(len(candidates)), size=take, replace=False)
        for local_idx, idx in enumerate(selected_indices.tolist()):
            row = candidates[idx]
            qa = row["qa"][0]
            repo_video_path = f"{VIDEO_PREFIX}/{row['video_path']}"
            downloaded = Path(
                hf_hub_download(
                    args.repo_id,
                    repo_video_path,
                    repo_type="dataset",
                    local_dir=download_dir,
                )
            )
            split = "train" if local_idx < int(round(take * args.train_fraction)) else "test"
            vinfo = row.get("video_info") or {}
            fps = vinfo.get("fps")
            duration = vinfo.get("duration")
            out_rows.append(
                {
                    "video_id": row.get("key", Path(row["video_path"]).stem),
                    "path": str(downloaded),
                    "label": str(qtype),
                    "question_type": str(qtype),
                    "task": str(qtype),
                    "answer": str(qa.get("answer")),
                    "answer_text": answer_text(str(qa.get("question", "")), str(qa.get("answer"))),
                    "question": qa.get("question"),
                    "video_type": row.get("video_type"),
                    "fps": fps,
                    "duration": duration,
                    "split": split,
                    "source_repo": args.repo_id,
                    "source_video_path": repo_video_path,
                }
            )

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output_jsonl": str(output),
                "videos": len(out_rows),
                "question_types": {key: min(args.per_question_type, len(value)) for key, value in sorted(buckets.items())},
                "download_dir": str(download_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
