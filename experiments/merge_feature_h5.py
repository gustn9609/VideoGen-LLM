#!/usr/bin/env python3
"""Merge feature groups from multiple HDF5 files into one HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def copy_group(src_group: h5py.Group, dst_group: h5py.Group, overwrite: bool) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)
    for name, item in src_group.items():
        if name in dst_group:
            if not overwrite:
                raise ValueError(f"Dataset already exists: {dst_group.name}/{name}")
            del dst_group[name]
        if isinstance(item, h5py.Dataset):
            dst_group.create_dataset(name, data=item[:], compression="gzip")
        elif isinstance(item, h5py.Group):
            child = dst_group.require_group(name)
            copy_group(item, child, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated input HDF5 files in copy order.")
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--modes", default="", help="Comma-separated group names. Empty copies every group.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as out:
        out.attrs["feature_extractor_version"] = "merged_feature_h5_v1"
        for path in inputs:
            with h5py.File(path, "r") as src:
                copy_attrs(src.attrs, out.attrs)
                group_names = modes or list(src.keys())
                for mode in group_names:
                    if mode not in src:
                        continue
                    dst_group = out.require_group(mode)
                    copy_group(src[mode], dst_group, overwrite=args.overwrite)
    print({"output_h5": args.output_h5, "inputs": [str(x) for x in inputs], "modes": modes or "all"})


if __name__ == "__main__":
    main()
