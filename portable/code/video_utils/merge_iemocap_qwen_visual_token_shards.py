#!/usr/bin/env python3
"""Merge sharded qwen_visual_temporal_tokens_v1 caches."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    merged = {}
    metadata = None
    for path in args.inputs:
        payload = pickle.load(path.open("rb"))
        if payload.get("schema") != "qwen_visual_temporal_tokens_v1":
            raise ValueError(f"Unexpected schema in {path}")
        if metadata is None:
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {
                    "sample_features",
                    "num_sample_features",
                    "total_visual_tokens",
                    "min_visual_tokens",
                    "max_visual_tokens",
                    "shard_index",
                    "num_shards",
                }
            }
        for key, value in payload["sample_features"].items():
            if key in merged:
                raise ValueError(f"Duplicate visual feature key: {key}")
            merged[key] = value

    token_counts = [int(entry["features"].shape[0]) for entry in merged.values()]
    output = {
        **metadata,
        "num_sample_features": len(merged),
        "total_visual_tokens": sum(token_counts),
        "min_visual_tokens": min(token_counts),
        "max_visual_tokens": max(token_counts),
        "sample_features": dict(sorted(merged.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.output)
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "sample_features"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
