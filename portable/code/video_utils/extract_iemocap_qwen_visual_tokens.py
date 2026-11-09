#!/usr/bin/env python3
"""Extract compact Qwen2.5-Omni visual tokens for aligned IEMOCAP clips."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("original_data/iemocap/video_clips/aligned_complete/manifest.jsonl"),
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
            "original_data/iemocap/iemocap_qwen_visual_temporal_tokens.pkl"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=0.25)
    parser.add_argument("--resized-height", type=int, default=160)
    parser.add_argument("--resized-width", type=int, default=0)
    parser.add_argument("--max-temporal-tokens", type=int, default=4)
    parser.add_argument(
        "--speaker-crop",
        choices=("none", "target"),
        default="none",
        help=(
            "For IEMOCAP split-screen videos, crop the target speaker camera. "
            "The side is resolved from a manually verified per-dialog map."
        ),
    )
    parser.add_argument(
        "--speaker-side-map",
        type=Path,
        default=None,
        help=(
            "Manually verified JSON containing dialog_speaker_sides. Required "
            "when --speaker-crop=target; no gender/side layout is assumed."
        ),
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_target_speaker_crop(
    row: dict,
    output_path: Path,
    speaker_side_map: dict,
) -> tuple[Path, str]:
    speaker = str(row.get("pkl_speaker", "")).strip().upper()
    dialog_id = str(row.get("dialog_id", "")).strip()
    if speaker not in {"F", "M"}:
        raise ValueError(
            f"speaker-crop=target requires pkl_speaker M/F: {row.get('turn_key')}"
        )
    dialog_mapping = speaker_side_map.get(dialog_id)
    if not isinstance(dialog_mapping, dict):
        raise ValueError(f"No verified speaker-side mapping for {dialog_id}")
    crop_side = str(dialog_mapping.get(speaker, "")).strip().lower()
    if crop_side == "left":
        crop_filter = "crop=iw/2:ih:0:0"
    elif crop_side == "right":
        crop_filter = "crop=iw/2:ih:iw/2:0"
    else:
        raise ValueError(
            f"Invalid speaker-side mapping for {dialog_id}/{speaker}: {crop_side!r}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(row["video_path"]),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            crop_filter,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    return output_path, crop_side


def load_partial(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = pickle.load(path.open("rb"))
    if payload.get("schema") != "qwen_visual_temporal_tokens_partial_v1":
        raise ValueError(f"Unexpected partial cache schema: {path}")
    return dict(payload.get("sample_features") or {})


def save_partial(path: Path, sample_features: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(
            {
                "schema": "qwen_visual_temporal_tokens_partial_v1",
                "sample_features": sample_features,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, path)


def spatial_pool_temporal_tokens(
    video_features: torch.Tensor,
    video_grid_thw: torch.Tensor,
    max_temporal_tokens: int,
) -> torch.Tensor:
    if video_grid_thw.shape[0] != 1:
        raise ValueError(f"Expected one video grid, got {video_grid_thw.tolist()}")
    grid_t, grid_h, grid_w = [int(x) for x in video_grid_thw[0].tolist()]
    if grid_t <= 0:
        raise ValueError(f"Invalid temporal grid: {video_grid_thw.tolist()}")
    if video_features.shape[0] % grid_t != 0:
        raise ValueError(
            f"Visual token count {video_features.shape[0]} is not divisible by grid_t={grid_t}"
        )
    spatial_tokens = video_features.shape[0] // grid_t
    expected_spatial = (grid_h * grid_w) // 4
    if spatial_tokens != expected_spatial:
        raise ValueError(
            f"Unexpected spatial token count: actual={spatial_tokens}, "
            f"expected={expected_spatial}, grid={video_grid_thw.tolist()}"
        )
    temporal = video_features.reshape(grid_t, spatial_tokens, -1).mean(dim=1)
    if max_temporal_tokens > 0 and temporal.shape[0] > max_temporal_tokens:
        temporal = F.adaptive_avg_pool1d(
            temporal.transpose(0, 1).unsqueeze(0),
            max_temporal_tokens,
        ).squeeze(0).transpose(0, 1)
    return temporal


def main() -> int:
    args = parse_args()
    if args.max_temporal_tokens <= 0:
        raise ValueError("--max-temporal-tokens must be positive")
    default_mixed_output = Path(
        "original_data/iemocap/iemocap_qwen_visual_temporal_tokens.pkl"
    ).resolve()
    if (
        args.speaker_crop == "target"
        and args.output_pkl.resolve() == default_mixed_output
    ):
        raise ValueError(
            "Target-speaker extraction must use a new --output-pkl; refusing "
            "to overwrite the existing full-frame visual cache. Suggested: "
            "original_data/iemocap/"
            "iemocap_qwen_visual_target_speaker_temporal_tokens.pkl"
        )
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Require 0 <= shard-index < num-shards")
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if len({row["turn_key"] for row in rows}) != len(rows):
        raise ValueError("Manifest turn_key values must be unique")
    speaker_side_map = {}
    if args.speaker_crop == "target":
        if args.speaker_side_map is None or not args.speaker_side_map.is_file():
            raise ValueError(
                "--speaker-crop=target requires an existing --speaker-side-map JSON"
            )
        side_payload = json.loads(args.speaker_side_map.read_text(encoding="utf-8"))
        speaker_side_map = side_payload.get("dialog_speaker_sides", side_payload)
        if not isinstance(speaker_side_map, dict):
            raise ValueError("speaker-side map must contain dialog_speaker_sides")
        expected_dialogs = {str(row["dialog_id"]) for row in rows}
        missing_dialogs = sorted(expected_dialogs - set(speaker_side_map))
        if missing_dialogs:
            raise ValueError(
                f"speaker-side map is missing {len(missing_dialogs)} dialogs: "
                f"{missing_dialogs[:5]}"
            )
        invalid_dialogs = []
        for dialog_id in sorted(expected_dialogs):
            mapping = speaker_side_map.get(dialog_id)
            if (
                not isinstance(mapping, dict)
                or set(mapping) != {"F", "M"}
                or {mapping.get("F"), mapping.get("M")} != {"left", "right"}
            ):
                invalid_dialogs.append(dialog_id)
        if invalid_dialogs:
            raise ValueError(
                f"speaker-side map has {len(invalid_dialogs)} invalid dialogs: "
                f"{invalid_dialogs[:5]}"
            )
        uncertain_dialogs = sorted(
            dialog_id
            for dialog_id, detail in (side_payload.get("details") or {}).items()
            if isinstance(detail, dict) and detail.get("status") == "uncertain"
        )
        if uncertain_dialogs:
            raise ValueError(
                f"speaker-side map still has {len(uncertain_dialogs)} uncertain "
                f"dialogs: {uncertain_dialogs[:5]}"
            )

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    partial_suffix = (
        ".speaker_target.partial"
        if args.speaker_crop == "target"
        else ".partial"
    )
    partial_path = args.output_pkl.with_suffix(args.output_pkl.suffix + partial_suffix)
    sample_features = {} if args.overwrite else load_partial(partial_path)

    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
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

    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise ImportError("qwen_omni_utils is required for video extraction") from exc

    processed_since_save = 0
    with tempfile.TemporaryDirectory(prefix="iemocap_speaker_crop_") as temp_dir:
        temp_root = Path(temp_dir)
        for row in tqdm(rows, desc="Extract compact Qwen visual tokens"):
            turn_key = str(row["turn_key"])
            if turn_key in sample_features and not args.overwrite:
                continue

            encoded_video_path = Path(str(row["video_path"]))
            crop_side = "none"
            if args.speaker_crop == "target":
                encoded_video_path, crop_side = build_target_speaker_crop(
                    row,
                    temp_root / f"{turn_key}.mp4",
                    speaker_side_map,
                )

            video_item = {
                "type": "video",
                "video": str(encoded_video_path),
                "fps": args.fps,
                "resized_height": args.resized_height,
            }
            if args.resized_width > 0:
                video_item["resized_width"] = args.resized_width
            messages = [[{"role": "user", "content": [video_item, {"type": "text", "text": " "}]}]]
            prompt_texts = [
                processor.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for message in messages
            ]
            _, images, videos = process_mm_info(messages, use_audio_in_video=False)
            processor_out = processor(
                text=prompt_texts,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            pixel_values = processor_out["pixel_values_videos"].to(args.device)
            video_grid_thw = processor_out["video_grid_thw"].to(args.device)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(args.device).startswith("cuda"),
            ):
                video_features = model.get_video_features(
                    pixel_values_videos=pixel_values,
                    video_grid_thw=video_grid_thw,
                    return_dict=True,
                ).pooler_output
                temporal_tokens = spatial_pool_temporal_tokens(
                    video_features=video_features,
                    video_grid_thw=video_grid_thw,
                    max_temporal_tokens=args.max_temporal_tokens,
                )
            sample_features[turn_key] = {
                "features": temporal_tokens.detach().to(torch.float16).cpu().numpy(),
                "utterance_id": str(row["utterance_id"]),
                "video_grid_thw": processor_out["video_grid_thw"][0].cpu().tolist(),
                "video_path": str(row["video_path"]),
                "visual_speaker": str(row.get("pkl_speaker", "")).strip().upper(),
                "crop_side": crop_side,
            }
            if encoded_video_path.parent == temp_root:
                encoded_video_path.unlink(missing_ok=True)
            processed_since_save += 1
            if processed_since_save >= args.save_every:
                save_partial(partial_path, sample_features)
                processed_since_save = 0

    save_partial(partial_path, sample_features)
    expected_keys = {str(row["turn_key"]) for row in rows}
    missing = sorted(expected_keys - set(sample_features))
    if missing:
        raise RuntimeError(f"Visual cache is incomplete; missing {len(missing)} keys")

    token_counts = [int(sample_features[key]["features"].shape[0]) for key in expected_keys]
    feature_dim = int(next(iter(sample_features.values()))["features"].shape[1])
    payload = {
        "schema": "qwen_visual_temporal_tokens_v1",
        "source": (
            "Qwen2.5-Omni visual tower pooler_output, spatial mean pooled, "
            f"speaker_crop={args.speaker_crop}"
        ),
        "model_path": str(args.model_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "speaker_crop": args.speaker_crop,
        "speaker_side_map": (
            str(args.speaker_side_map.resolve())
            if args.speaker_side_map is not None
            else None
        ),
        "fps": args.fps,
        "resized_height": args.resized_height,
        "resized_width": args.resized_width,
        "max_temporal_tokens": args.max_temporal_tokens,
        "feature_dim": feature_dim,
        "feature_dtype": "float16",
        "num_sample_features": len(expected_keys),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_visual_tokens": sum(token_counts),
        "min_visual_tokens": min(token_counts),
        "max_visual_tokens": max(token_counts),
        "sample_features": {key: sample_features[key] for key in sorted(expected_keys)},
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
