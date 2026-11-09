#!/usr/bin/env python3
"""Merge sharded qwen_audio_temporal_tokens_v1 caches."""

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
        if payload.get("schema") != "qwen_audio_temporal_tokens_v1":
            raise ValueError(f"Unexpected schema in {path}")
        current_metadata = {
            key: value
            for key, value in payload.items()
            if key not in {
                "sample_features",
                "num_sample_features",
                "num_missing_audio",
                "total_source_audio_tokens",
                "total_audio_tokens",
                "min_audio_tokens",
                "max_audio_tokens",
                "compression_ratio",
                "shard_index",
                "num_shards",
            }
        }
        if metadata is None:
            metadata = current_metadata
        elif current_metadata != metadata:
            raise ValueError(f"Incompatible audio shard metadata: {path}")
        for key, value in payload["sample_features"].items():
            if key in merged:
                raise ValueError(f"Duplicate audio feature key: {key}")
            merged[key] = value

    source_counts = [
        int(entry.get("source_audio_tokens", 0)) for entry in merged.values()
    ]
    token_counts = [int(entry["features"].shape[0]) for entry in merged.values()]
    source_total = sum(source_counts)
    output = {
        **(metadata or {}),
        "num_sample_features": len(merged),
        "num_missing_audio": sum(
            bool(entry.get("missing_audio")) for entry in merged.values()
        ),
        "total_source_audio_tokens": source_total,
        "total_audio_tokens": sum(token_counts),
        "min_audio_tokens": min(token_counts),
        "max_audio_tokens": max(token_counts),
        "compression_ratio": (
            float(sum(token_counts)) / float(source_total)
            if source_total > 0
            else None
        ),
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
