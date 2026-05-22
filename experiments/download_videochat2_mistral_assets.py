#!/usr/bin/env python3
"""Download VideoChat2-Mistral assets needed for hidden-token extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/videochat2_mistral")
    parser.add_argument("--mistral-repo", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--videochat2-repo", default="OpenGVLab/VideoChat2_stage3_Mistral_7B")
    parser.add_argument("--videochat2-file", default="videochat2_mistral_7b_stage3.pth")
    parser.add_argument("--vit-repo", default="OpenGVLab/videochat")
    parser.add_argument("--vit-file", default="umt_l16_qformer.pth")
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mistral_dir = snapshot_download(
        repo_id=args.mistral_repo,
        local_dir=str(out / "Mistral-7B-Instruct-v0.2"),
        local_dir_use_symlinks=False,
    )
    videochat2_path = hf_hub_download(
        repo_id=args.videochat2_repo,
        filename=args.videochat2_file,
        local_dir=str(out / "VideoChat2_stage3_Mistral_7B"),
        local_dir_use_symlinks=False,
    )
    vit_path = hf_hub_download(
        repo_id=args.vit_repo,
        filename=args.vit_file,
        local_dir=str(out / "videochat"),
        local_dir_use_symlinks=False,
    )
    manifest = {
        "mistral_model_path": str(Path(mistral_dir).resolve()),
        "videochat2_model_path": str(Path(videochat2_path).resolve()),
        "vit_blip_model_path": str(Path(vit_path).resolve()),
    }
    manifest_path = Path(args.manifest) if args.manifest else out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
