#!/usr/bin/env python3
"""Extract frozen Qwen2.5-Omni utterance-level text features for IEMOCAP."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iemocap-pkl",
        type=Path,
        default=Path("original_data/iemocap/iemocap.pkl"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("LLM_bases/Qwen2.5-Omni-7B"),
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path("original_data/iemocap/text/iemocap_qwen_text_utterance_features.pkl"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_partial(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = pickle.load(path.open("rb"))
    if payload.get("schema") != "qwen_text_utterance_features_partial_v1":
        raise ValueError(f"Unexpected partial cache schema: {path}")
    return dict(payload.get("sample_features") or {})


def save_partial(path: Path, sample_features: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(
            {
                "schema": "qwen_text_utterance_features_partial_v1",
                "sample_features": sample_features,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)


def build_rows(iemocap_pkl: Path) -> list[tuple[str, str, str, int]]:
    payload = pickle.load(iemocap_pkl.open("rb"))
    if not isinstance(payload, (list, tuple)) or len(payload) < 6:
        raise ValueError(f"Unexpected IEMOCAP pkl structure: {iemocap_pkl}")
    sentence_dict = payload[2]
    conv_ids = list(payload[3]) + list(payload[4]) + list(payload[5])
    rows = []
    for conv_id in conv_ids:
        utterances = sentence_dict.get(conv_id, [])
        for turn_idx, text in enumerate(utterances):
            turn_key = f"{conv_id}_{turn_idx}"
            rows.append((turn_key, str(text), str(conv_id), int(turn_idx)))
    return rows


def extract_batch_features(model, tokenizer, texts, device) -> np.ndarray:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=str(device).startswith("cuda"),
    ):
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        hidden = getattr(outputs, "last_hidden_state", None)
    else:
        hidden = hidden_states[-1]
    if hidden is None:
        raise RuntimeError("Model output did not contain hidden states.")
    mask = encoded["attention_mask"].to(hidden.device).bool()
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    return pooled.detach().to(torch.float16).cpu().numpy()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    rows = build_rows(args.iemocap_pkl)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_pkl.with_suffix(args.output_pkl.suffix + ".partial")
    sample_features = {} if args.overwrite else load_partial(partial_path)

    config = AutoConfig.from_pretrained(str(args.model_path), trust_remote_code=True)
    if str(getattr(config, "model_type", "")).lower() != "qwen2_5_omni":
        raise ValueError(f"Expected Qwen2.5-Omni model: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_path),
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_path),
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    pending = [row for row in rows if row[0] not in sample_features]
    processed_since_save = 0
    for batch_start in tqdm(
        range(0, len(pending), args.batch_size),
        desc="Extract Qwen utterance text features",
    ):
        batch_rows = pending[batch_start : batch_start + args.batch_size]
        texts = [row[1] for row in batch_rows]
        features = extract_batch_features(model, tokenizer, texts, args.device)
        if not np.isfinite(features).all():
            raise RuntimeError(f"Non-finite text features in batch starting at {batch_start}")
        for row, feature in zip(batch_rows, features):
            turn_key, text, conv_id, turn_idx = row
            sample_features[turn_key] = {
                "features": feature.reshape(1, -1),
                "text": text,
                "dialog_id": conv_id,
                "turn_idx": turn_idx,
            }
        processed_since_save += len(batch_rows)
        if processed_since_save >= args.save_every:
            save_partial(partial_path, sample_features)
            processed_since_save = 0

    save_partial(partial_path, sample_features)
    expected_keys = {row[0] for row in rows}
    missing = sorted(expected_keys - set(sample_features))
    if missing:
        raise RuntimeError(f"Text feature cache is incomplete; missing {len(missing)} keys.")

    selected = {key: sample_features[key] for key in sorted(expected_keys)}
    feature_dim = int(next(iter(selected.values()))["features"].shape[-1])
    payload = {
        "schema": "qwen_text_utterance_features_v1",
        "source": "Qwen2.5-Omni thinker final hidden state mean over utterance text",
        "model_path": str(args.model_path.resolve()),
        "iemocap_pkl": str(args.iemocap_pkl.resolve()),
        "feature_dim": feature_dim,
        "feature_dtype": "float16",
        "num_sample_features": len(selected),
        "sample_features": selected,
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
