#!/usr/bin/env python3
"""Derive fixed-K Qwen audio-token caches without rerunning the audio encoder."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def adaptive_mean(features: np.ndarray, output_tokens: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(f"Expected non-empty [tokens, dim] features, got {features.shape}")
    length = int(features.shape[0])
    rows = []
    for index in range(output_tokens):
        start = int(np.floor(index * length / output_tokens))
        end = int(np.ceil((index + 1) * length / output_tokens))
        end = max(start + 1, min(end, length))
        rows.append(features[start:end].astype(np.float32).mean(axis=0))
    return np.stack(rows).astype(features.dtype, copy=False)


def build(source: Path, output: Path, token_count: int) -> None:
    if output.exists():
        with output.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            cached.get("pooling_mode") == "fixed"
            and int(cached.get("fixed_temporal_tokens", -1)) == token_count
        ):
            print(f"[CACHE SKIP] validated {output}", flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite incompatible cache: {output}")

    with source.open("rb") as handle:
        payload = pickle.load(handle)
    entries = payload.get("sample_features")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"Invalid source audio-token cache: {source}")

    converted = {}
    total_source = 0
    total_output = 0
    for sample_id, entry in entries.items():
        item = dict(entry)
        features = np.asarray(item["features"])
        pooled = adaptive_mean(features, token_count)
        item["features"] = pooled
        item["compressed_audio_tokens"] = int(pooled.shape[0])
        converted[sample_id] = item
        total_source += int(item.get("source_audio_tokens", features.shape[0]))
        total_output += int(pooled.shape[0])

    result = dict(payload)
    result.update(
        source=f"Fixed-K adaptive mean pooling derived from {source}",
        pooling="adaptive_avg_pool1d",
        pooling_mode="fixed",
        fixed_temporal_tokens=token_count,
        min_temporal_tokens=token_count,
        max_temporal_tokens=token_count,
        total_source_audio_tokens=total_source,
        total_audio_tokens=total_output,
        min_audio_tokens=token_count,
        max_audio_tokens=token_count,
        compression_ratio=(total_output / total_source if total_source else 0.0),
        sample_features=converted,
    )
    result.pop("compression_factor", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    print(
        f"[CACHE OK] K={token_count} samples={len(converted)} tokens={total_output} -> {output}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    args = parser.parse_args()
    for token_count in args.tokens:
        if token_count < 1:
            raise ValueError("Fixed token counts must be positive")
        build(
            args.source,
            args.output_dir / f"iemocap_qwen_audio_fixed{token_count}.pkl",
            token_count,
        )


if __name__ == "__main__":
    main()
