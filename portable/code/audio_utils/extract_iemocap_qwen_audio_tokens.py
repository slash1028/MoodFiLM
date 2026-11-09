#!/usr/bin/env python3
"""Extract compact utterance-level Qwen2.5-Omni audio tokens for IEMOCAP."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen2_5OmniThinkerForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-mel-pkl",
        type=Path,
        default=Path(
            "original_data/iemocap/iemocap_qwen_native_mel_features.pkl"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("LLM_bases/Qwen2.5-Omni-7B"),
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path(
            "original_data/iemocap/iemocap_qwen_audio_temporal_tokens.pkl"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--pooling-mode",
        choices=("fixed", "dynamic"),
        default="fixed",
        help=(
            "fixed always pools to max-temporal-tokens; dynamic keeps roughly "
            "one token per compression-factor source tokens within the configured bounds."
        ),
    )
    parser.add_argument("--min-temporal-tokens", type=int, default=4)
    parser.add_argument("--max-temporal-tokens", type=int, default=4)
    parser.add_argument("--compression-factor", type=float, default=8.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extraction_config(args: argparse.Namespace) -> dict:
    return {
        "pooling_mode": args.pooling_mode,
        "min_temporal_tokens": int(args.min_temporal_tokens),
        "max_temporal_tokens": int(args.max_temporal_tokens),
        "compression_factor": float(args.compression_factor),
    }


def load_partial(path: Path, expected_config: dict) -> dict:
    if not path.is_file():
        return {}
    payload = pickle.load(path.open("rb"))
    if payload.get("schema") != "qwen_audio_temporal_tokens_partial_v1":
        raise ValueError(f"Unexpected partial cache schema: {path}")
    partial_config = payload.get("extraction_config")
    if partial_config is None:
        legacy_config = {
            "pooling_mode": "fixed",
            "min_temporal_tokens": 4,
            "max_temporal_tokens": 4,
            "compression_factor": 8.0,
        }
        partial_config = legacy_config
    if partial_config != expected_config:
        raise ValueError(
            f"Partial cache configuration mismatch for {path}: "
            f"{partial_config} != {expected_config}. "
            "Use a different output path or pass --overwrite."
        )
    return dict(payload.get("sample_features") or {})


def save_partial(
    path: Path,
    sample_features: dict,
    config: dict,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(
            {
                "schema": "qwen_audio_temporal_tokens_partial_v1",
                "extraction_config": config,
                "sample_features": sample_features,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)


def resolve_temporal_token_count(
    source_tokens: int,
    pooling_mode: str,
    min_temporal_tokens: int,
    max_temporal_tokens: int,
    compression_factor: float,
) -> int:
    source_tokens = int(source_tokens)
    if source_tokens <= 0:
        return 1
    if pooling_mode == "fixed":
        target_tokens = max_temporal_tokens
    elif pooling_mode == "dynamic":
        target_tokens = int(math.ceil(source_tokens / compression_factor))
        target_tokens = max(min_temporal_tokens, target_tokens)
        target_tokens = min(max_temporal_tokens, target_tokens)
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling_mode}")
    return max(1, min(source_tokens, int(target_tokens)))


def temporal_pool(
    audio_features: torch.Tensor,
    target_temporal_tokens: int,
) -> torch.Tensor:
    if audio_features.ndim != 2 or audio_features.shape[0] <= 0:
        raise ValueError(f"Invalid audio feature shape: {tuple(audio_features.shape)}")
    if audio_features.shape[0] <= target_temporal_tokens:
        return audio_features
    return F.adaptive_avg_pool1d(
        audio_features.transpose(0, 1).unsqueeze(0),
        target_temporal_tokens,
    ).squeeze(0).transpose(0, 1)


def build_audio_batch(
    sample_ids: list[str],
    sample_features: dict,
    sample_lengths: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = []
    lengths = []
    for sample_id in sample_ids:
        array = np.asarray(sample_features[sample_id], dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"Expected mel feature [feature_dim, frames] for {sample_id}, "
                f"got {array.shape}"
            )
        length = int(sample_lengths.get(sample_id, array.shape[-1]))
        length = max(1, min(length, int(array.shape[-1])))
        arrays.append(array[:, :length])
        lengths.append(length)

    feature_dim = int(arrays[0].shape[0])
    max_length = max(lengths)
    input_features = torch.zeros(
        (len(arrays), feature_dim, max_length),
        dtype=torch.float32,
    )
    feature_attention_mask = torch.zeros(
        (len(arrays), max_length),
        dtype=torch.long,
    )
    for index, (array, length) in enumerate(zip(arrays, lengths)):
        if int(array.shape[0]) != feature_dim:
            raise ValueError(
                f"Inconsistent mel feature dimension for {sample_ids[index]}: "
                f"{array.shape[0]} != {feature_dim}"
            )
        input_features[index, :, :length] = torch.from_numpy(array)
        feature_attention_mask[index, :length] = 1
    return input_features, feature_attention_mask


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_temporal_tokens <= 0:
        raise ValueError("--min-temporal-tokens must be positive")
    if args.max_temporal_tokens <= 0:
        raise ValueError("--max-temporal-tokens must be positive")
    if args.min_temporal_tokens > args.max_temporal_tokens:
        raise ValueError(
            "--min-temporal-tokens cannot exceed --max-temporal-tokens"
        )
    if args.compression_factor <= 0:
        raise ValueError("--compression-factor must be positive")
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Require 0 <= shard-index < num-shards")
    config = extraction_config(args)

    mel_payload = pickle.load(args.input_mel_pkl.open("rb"))
    if mel_payload.get("schema") != "qwen_native_mel_v1":
        raise ValueError(
            f"Expected qwen_native_mel_v1 cache: {args.input_mel_pkl}"
        )
    mel_features = mel_payload.get("sample_features")
    mel_lengths = mel_payload.get("sample_feature_lengths") or {}
    if not isinstance(mel_features, dict) or not mel_features:
        raise ValueError("Input mel cache has no sample_features")

    all_sample_ids = sorted(
        str(sample_id)
        for sample_id in (
            (mel_payload.get("conv_turn_to_uttid") or {}).keys()
            or mel_features.keys()
        )
    )
    shard_sample_ids = [
        sample_id
        for index, sample_id in enumerate(all_sample_ids)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        shard_sample_ids = shard_sample_ids[: args.limit]

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_pkl.with_suffix(args.output_pkl.suffix + ".partial")
    compact_features = (
        {} if args.overwrite else load_partial(partial_path, config)
    )

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_dim = int(model.audio_tower.config.output_dim)
    pending = [
        sample_id
        for sample_id in shard_sample_ids
        if sample_id not in compact_features
    ]
    processed_since_save = 0
    for batch_start in tqdm(
        range(0, len(pending), args.batch_size),
        desc="Extract compact Qwen audio tokens",
    ):
        batch_ids = pending[batch_start : batch_start + args.batch_size]
        valid_ids = [sample_id for sample_id in batch_ids if sample_id in mel_features]
        if valid_ids:
            input_features, feature_attention_mask = build_audio_batch(
                valid_ids,
                mel_features,
                mel_lengths,
            )
            feature_lengths = feature_attention_mask.sum(dim=1)
            _, output_lengths = model.audio_tower._get_feat_extract_output_lengths(
                feature_lengths.to(args.device)
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(args.device).startswith("cuda"),
            ):
                outputs = model.get_audio_features(
                    input_features=input_features.to(args.device),
                    feature_attention_mask=feature_attention_mask.to(args.device),
                    audio_feature_lengths=feature_lengths.to(args.device),
                    return_dict=True,
                )
            split_features = outputs.last_hidden_state.split(
                [int(length) for length in output_lengths.tolist()],
                dim=0,
            )
            for sample_id, features in zip(valid_ids, split_features):
                target_tokens = resolve_temporal_token_count(
                    source_tokens=int(features.shape[0]),
                    pooling_mode=args.pooling_mode,
                    min_temporal_tokens=args.min_temporal_tokens,
                    max_temporal_tokens=args.max_temporal_tokens,
                    compression_factor=args.compression_factor,
                )
                pooled = temporal_pool(features, target_tokens)
                compact_features[sample_id] = {
                    "features": pooled.detach().to(torch.float16).cpu().numpy(),
                    "source_audio_tokens": int(features.shape[0]),
                    "compressed_audio_tokens": int(pooled.shape[0]),
                    "missing_audio": False,
                }

        for sample_id in batch_ids:
            if sample_id not in mel_features:
                compact_features[sample_id] = {
                    "features": np.zeros((1, output_dim), dtype=np.float16),
                    "source_audio_tokens": 0,
                    "compressed_audio_tokens": 1,
                    "missing_audio": True,
                }

        processed_since_save += len(batch_ids)
        if processed_since_save >= args.save_every:
            save_partial(partial_path, compact_features, config)
            processed_since_save = 0

    save_partial(partial_path, compact_features, config)
    missing = sorted(set(shard_sample_ids) - set(compact_features))
    if missing:
        raise RuntimeError(f"Audio cache is incomplete; missing {len(missing)} keys")

    selected = {
        sample_id: compact_features[sample_id]
        for sample_id in shard_sample_ids
    }
    source_counts = [
        int(entry["source_audio_tokens"]) for entry in selected.values()
    ]
    compact_counts = [
        int(entry["features"].shape[0]) for entry in selected.values()
    ]
    valid_source_total = sum(source_counts)
    payload = {
        "schema": "qwen_audio_temporal_tokens_v1",
        "source": (
            "Qwen2.5-Omni audio tower last_hidden_state with adaptive temporal "
            "average pooling"
        ),
        "model_path": str(args.model_path.resolve()),
        "input_mel_pkl": str(args.input_mel_pkl.resolve()),
        "pooling": "adaptive_avg_pool1d",
        "pooling_mode": args.pooling_mode,
        "min_temporal_tokens": args.min_temporal_tokens,
        "max_temporal_tokens": args.max_temporal_tokens,
        "compression_factor": args.compression_factor,
        "feature_dim": output_dim,
        "feature_dtype": "float16",
        "num_sample_features": len(selected),
        "num_missing_audio": sum(
            bool(entry.get("missing_audio")) for entry in selected.values()
        ),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_source_audio_tokens": valid_source_total,
        "total_audio_tokens": sum(compact_counts),
        "min_audio_tokens": min(compact_counts),
        "max_audio_tokens": max(compact_counts),
        "compression_ratio": (
            float(sum(compact_counts)) / float(valid_source_total)
            if valid_source_total > 0
            else None
        ),
        "sample_features": dict(sorted(selected.items())),
    }
    tmp = args.output_pkl.with_suffix(args.output_pkl.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.output_pkl)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "sample_features"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved: {args.output_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
