#!/usr/bin/env python3
"""Generate utterance-level visual cue text from IEMOCAP video clips.

This script uses Qwen2.5-Omni to describe utterance videos. By default it is
visual-only; pass --use-audio-in-video to let Qwen also read the audio track
inside the mp4. The output is a JSONL file keyed by utterance_id / sample_id,
intended for later ERC prompt conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from qwen_omni_utils import process_mm_info


DEFAULT_MODEL_PATH = Path(
    "/n/work6/yiwang/MoodFilm/portable/LLM_bases/Qwen2.5-Omni-7B"
)
DEFAULT_MANIFEST = Path(
    "/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/video_clips/"
    "video_with_audio/manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/video_cues/"
    "qwen25_omni_visual_cues.jsonl"
)


PROMPT = (
    "You are analyzing the visible non-verbal behavior of the current speaker "
    "for emotion recognition. Use only the video frames, not audio and not any "
    "transcript. In 1-2 concise English sentences, describe emotion-relevant "
    "visual cues such as facial expression, gaze, head movement, body posture, "
    "and apparent arousal. Do not choose an emotion label."
)

AUDIO_VIDEO_PROMPT = (
    "You are analyzing the current speaker for emotion recognition using the "
    "provided video clip and its audio track, but not any transcript. In 1-2 "
    "concise English sentences, describe emotion-relevant cues such as facial "
    "expression, gaze, head movement, body posture, speaking tone, intensity, "
    "pauses, and apparent arousal. Do not choose an emotion label."
)


def target_speaker_from_utterance_id(utterance_id: str) -> str:
    marker = ""
    parts = str(utterance_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1]:
        marker = parts[1][0].upper()
    if marker == "F":
        return "female speaker"
    if marker == "M":
        return "male speaker"
    return "current speaker"


def prompt_for_row(row: dict[str, Any], use_audio_in_video: bool) -> str:
    target_speaker = target_speaker_from_utterance_id(str(row.get("utterance_id", "")))
    if use_audio_in_video:
        return (
            f"The target speaker is the {target_speaker}; when audio is available, "
            "focus on the person speaking in the audio track. "
            + AUDIO_VIDEO_PROMPT
        )
    return (
        f"The target speaker is the {target_speaker}. "
        "If multiple people are visible, focus on this target speaker rather than the listener. "
        + PROMPT
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual cue descriptions for IEMOCAP utterance videos."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--resized-height", type=int, default=224)
    parser.add_argument("--resized-width", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--sessions",
        default="",
        help="Optional comma-separated session filter, e.g. Session5 or Session1,Session2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Read audio from the mp4 together with visual frames.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to from_pretrained. Use 'auto' for one or more visible GPUs.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="",
        help="Optional attention backend, e.g. flash_attention_2. Empty keeps HF default.",
    )
    return parser.parse_args()


def load_manifest(path: Path, sessions: set[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if sessions is not None and row.get("session") not in sessions:
                continue
            rows.append(row)
    return rows


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("utterance_id")
            if key:
                done.add(str(key))
    return done


def first_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_messages(row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    video_path = row.get("video_path") or row.get("output_video")
    if not video_path:
        raise KeyError("manifest row must contain video_path or output_video")
    video_item: dict[str, Any] = {
        "type": "video",
        "video": str(video_path),
        "fps": float(args.fps),
        "resized_height": int(args.resized_height),
    }
    if int(args.resized_width) > 0:
        video_item["resized_width"] = int(args.resized_width)
    prompt = prompt_for_row(row, args.use_audio_in_video)
    return [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]


def strip_prompt(output_ids: torch.Tensor, input_ids: torch.Tensor, input_len: int) -> torch.Tensor:
    output_ids = output_ids.detach().cpu()
    input_ids = input_ids.detach().cpu()
    prefix = output_ids[: input_ids.numel()]
    if prefix.numel() == input_ids.numel() and torch.equal(prefix, input_ids):
        return output_ids[input_ids.numel() :]
    return output_ids[int(input_len) :]


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.manifest_jsonl.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_jsonl}")

    sessions = None
    if args.sessions.strip():
        sessions = {x.strip() for x in args.sessions.split(",") if x.strip()}

    rows = load_manifest(args.manifest_jsonl, sessions)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if not rows:
        raise ValueError("No manifest rows selected.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()
    done = load_done(args.output_jsonl)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    model.eval()
    device = first_model_device(model)

    mode = "a" if args.output_jsonl.exists() and not args.overwrite else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as out_f:
        iterator = tqdm(rows, desc="Generating visual cues", dynamic_ncols=True)
        for row in iterator:
            utterance_id = str(row.get("utterance_id", ""))
            if utterance_id in done:
                continue
            video_path = Path(row.get("video_path") or row.get("output_video") or "")
            if not video_path.is_file():
                result = {
                    "utterance_id": utterance_id,
                    "sample_id": row.get("sample_id"),
                    "video_path": str(video_path),
                    "status": "missing_video",
                    "visual_cue": "",
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            messages = build_messages(row, args)
            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = process_mm_info(
                [messages],
                use_audio_in_video=args.use_audio_in_video,
            )
            processor_kwargs = {
                "text": [prompt_text],
                "images": images,
                "videos": videos,
                "return_tensors": "pt",
                "padding": True,
                "use_audio_in_video": args.use_audio_in_video,
            }
            if audios is not None:
                processor_kwargs["audio"] = audios
                processor_kwargs["sampling_rate"] = int(
                    getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
                    or 16000
                )
            inputs = processor(**processor_kwargs)
            inputs = move_to_device(inputs, device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    num_beams=1,
                    do_sample=False,
                    repetition_penalty=1.0,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            input_len = int(inputs["attention_mask"][0].sum().item())
            gen_ids = strip_prompt(outputs[0], inputs["input_ids"][0], input_len)
            text = processor.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            result = {
                "utterance_id": utterance_id,
                "sample_id": row.get("sample_id"),
                "session": row.get("session"),
                "dialog_id": row.get("dialog_id"),
                "emotion": row.get("emotion"),
                "video_path": str(video_path),
                "fps": float(args.fps),
                "resized_height": int(args.resized_height),
                "resized_width": int(args.resized_width),
                "use_audio_in_video": bool(args.use_audio_in_video),
                "target_speaker": target_speaker_from_utterance_id(utterance_id),
                "num_shards": int(args.num_shards),
                "shard_index": int(args.shard_index),
                "status": "ok",
                "visual_cue": text,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    print(args.output_jsonl)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
