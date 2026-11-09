#!/usr/bin/env python3
"""Permanently crop aligned IEMOCAP clips to the target speaker's half."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/aligned_complete/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--speaker-side-map",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/aligned_complete/"
            "speaker_side_manual.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/target_speaker_half"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_ROOT/manifest.jsonl.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--drop-audio",
        action="store_true",
        help="Remove audio from cropped files. Audio is preserved by default.",
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if len({str(row["turn_key"]) for row in rows}) != len(rows):
        raise ValueError("Manifest turn_key values must be unique")

    payload = json.loads(args.speaker_side_map.read_text(encoding="utf-8"))
    mapping = payload.get("dialog_speaker_sides", payload)
    if not isinstance(mapping, dict):
        raise ValueError("Speaker-side map must contain dialog_speaker_sides")

    expected_dialogs = {str(row["dialog_id"]) for row in rows}
    invalid = []
    for dialog_id in sorted(expected_dialogs):
        sides = mapping.get(dialog_id)
        if (
            not isinstance(sides, dict)
            or set(sides) != {"F", "M"}
            or {sides.get("F"), sides.get("M")} != {"left", "right"}
        ):
            invalid.append(dialog_id)
    if invalid:
        raise ValueError(
            f"Missing or invalid speaker-side mappings for {len(invalid)} dialogs: "
            f"{invalid[:5]}"
        )
    return rows, mapping


def output_path_for(row: dict, output_root: Path) -> Path:
    return (
        output_root
        / str(row["session"])
        / str(row["dialog_id"])
        / f"{row['utterance_id']}.mp4"
    )


def crop_one(
    row: dict,
    speaker_side_map: dict,
    output_root: Path,
    overwrite: bool,
    drop_audio: bool,
) -> dict:
    source = Path(str(row["video_path"]))
    if not source.is_file():
        raise FileNotFoundError(source)

    speaker = str(row.get("pkl_speaker", "")).strip().upper()
    if speaker not in {"F", "M"}:
        raise ValueError(f"Invalid target speaker for {row.get('turn_key')}: {speaker!r}")
    crop_side = str(speaker_side_map[str(row["dialog_id"])][speaker])
    crop_filter = (
        "crop=trunc(iw/2/2)*2:trunc(ih/2)*2:0:0"
        if crop_side == "left"
        else "crop=trunc(iw/2/2)*2:trunc(ih/2)*2:iw-trunc(iw/2/2)*2:0"
    )

    output = output_path_for(row, output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not output.is_file() or output.stat().st_size == 0:
        temporary = output.with_name(output.stem + ".tmp.mp4")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
        ]
        if drop_audio:
            command.append("-an")
        else:
            command.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"])
        command.extend(
            [
                "-vf",
                crop_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        try:
            subprocess.run(command, check=True)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    result = dict(row)
    result["original_video_path"] = str(source.resolve())
    result["video_path"] = str(output.resolve())
    result["target_speaker"] = speaker
    result["crop_side"] = crop_side
    result["speaker_crop"] = "target_half"
    return result


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    rows, speaker_side_map = load_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_manifest or args.output_root / "manifest.jsonl"

    completed: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                crop_one,
                row,
                speaker_side_map,
                args.output_root,
                args.overwrite,
                args.drop_audio,
            ): str(row["turn_key"])
            for row in rows
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Crop target-speaker videos",
        ):
            turn_key = futures[future]
            completed[turn_key] = future.result()

    ordered_rows = [completed[str(row["turn_key"])] for row in rows]
    write_manifest(output_manifest, ordered_rows)
    print(
        json.dumps(
            {
                "samples": len(ordered_rows),
                "output_root": str(args.output_root.resolve()),
                "output_manifest": str(output_manifest.resolve()),
                "audio_preserved": not args.drop_audio,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
