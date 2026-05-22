#!/usr/bin/env python3
"""Prepare a local MVBench video subset for Wan ablation experiments.

The OpenGVLab/MVBench Hugging Face dataset stores annotations and source
benchmark videos as large zip files. This script mirrors the sampling order in
make_mvbench_subset.py, downloads the source zip files that are needed, and
extracts only the selected video files into a compact local root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_REPO_ID = "OpenGVLab/MVBench"
DEFAULT_JSON_DIR = (
    Path.home()
    / ".cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots"
    / "230a2d4fac8900333c61754641c7a13e069ac9c6/json"
)

TASK_TO_ZIP = {
    "action_antonym": "video/ssv2_video.zip",
    "action_count": "video/perception.zip",
    "action_localization": "video/sta.zip",
    "action_prediction": "video/star.zip",
    "action_sequence": "video/star.zip",
    "character_order": "video/perception.zip",
    "counterfactual_inference": "video/clevrer.zip",
    "egocentric_navigation": "video/vlnqa.zip",
    "fine_grained_action": "video/Moments_in_Time_Raw.zip",
    "moving_attribute": "video/clevrer.zip",
    "moving_count": "video/clevrer.zip",
    "moving_direction": "video/clevrer.zip",
    "object_existence": "video/clevrer.zip",
    "object_interaction": "video/star.zip",
    "object_shuffle": "video/perception.zip",
    "scene_transition": "video/scene_qa.zip",
    "state_change": "video/perception.zip",
    "unexpected_action": "video/FunQA_test.zip",
}

VIDEO_SUFFIXES = {".mp4", ".avi", ".webm", ".mkv", ".mov"}


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_zip_index(zip_path: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            path = Path(member)
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            normalized = member.lstrip("./")
            index[normalized].append(member)
            index[path.name].append(member)
            index[path.stem].append(member)
    return index


def resolve_member(video_name: str, index: dict[str, list[str]]) -> str | None:
    raw = Path(video_name)
    keys = [video_name.lstrip("./"), raw.name, raw.stem]
    for key in keys:
        matches = index.get(key)
        if matches:
            return sorted(matches, key=len)[0]
    suffix_matches = [
        member
        for member in index.get(raw.name, [])
        if member.endswith(video_name.lstrip("./"))
    ]
    if suffix_matches:
        return sorted(suffix_matches, key=len)[0]
    return None


def extract_member(zip_path: Path, member: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def ensure_json_dir(repo_id: str, json_dir: Path | None) -> Path:
    if json_dir and json_dir.exists():
        return json_dir
    snapshot = snapshot_download(repo_id, repo_type="dataset", allow_patterns=["json/*.json"])
    return Path(snapshot) / "json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    import numpy as np

    json_dir = ensure_json_dir(args.repo_id, Path(args.json_dir).expanduser())
    output_root = Path(args.output_root).expanduser()
    task_filter = {x.strip() for x in args.tasks.split(",") if x.strip()}
    rng = np.random.default_rng(args.seed)

    zip_cache: dict[str, Path] = {}
    zip_indexes: dict[str, dict[str, list[str]]] = {}
    extracted: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    task_counts: dict[str, int] = {}

    for json_path in sorted(json_dir.glob("*.json")):
        task = json_path.stem
        if task_filter and task not in task_filter:
            continue

        rows = read_json(json_path)
        order = rng.permutation(len(rows)).tolist()
        zip_name = TASK_TO_ZIP.get(task)
        selected = 0

        if zip_name is None:
            task_counts[task] = 0
            missing.append({"task": task, "reason": "unsupported_or_external_video_source"})
            continue

        if zip_name not in zip_cache:
            if args.skip_download:
                local = Path.home() / ".cache/huggingface/hub" / zip_name
                if not local.exists():
                    raise FileNotFoundError(f"{zip_name} not found and --skip-download was set")
            else:
                local = Path(
                    hf_hub_download(args.repo_id, filename=zip_name, repo_type="dataset")
                )
            zip_cache[zip_name] = local
            zip_indexes[zip_name] = build_zip_index(local)

        for idx in order:
            if selected >= args.per_task:
                break
            item = rows[idx]
            video_name = str(item.get("video", ""))
            if not video_name or Path(video_name).suffix.lower() not in VIDEO_SUFFIXES:
                missing.append({"task": task, "video": video_name, "reason": "not_a_video_file"})
                continue
            member = resolve_member(video_name, zip_indexes[zip_name])
            if member is None:
                missing.append({"task": task, "video": video_name, "zip": zip_name, "reason": "not_in_zip"})
                continue
            output_path = output_root / video_name
            extract_member(zip_cache[zip_name], member, output_path)
            extracted.append(
                {
                    "task": task,
                    "video": video_name,
                    "zip": zip_name,
                    "member": member,
                    "path": str(output_path),
                }
            )
            selected += 1

        task_counts[task] = selected

    report = {
        "repo_id": args.repo_id,
        "json_dir": str(json_dir),
        "output_root": str(output_root),
        "per_task": args.per_task,
        "task_counts": task_counts,
        "extracted": len(extracted),
        "missing": len(missing),
        "zip_files": {name: str(path) for name, path in zip_cache.items()},
        "missing_examples": missing[:50],
    }
    write_json(Path(args.report_json).expanduser(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
