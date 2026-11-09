#!/usr/bin/env python3
"""Prepare representative contact sheets for manual IEMOCAP side annotation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


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
        "--output-dir",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/speaker_side_annotation_frames"
        ),
    )
    parser.add_argument("--frames-per-dialog", type=int, default=5)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--jpeg-quality", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dialogs(manifest: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped[str(row["dialog_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (float(row["start"]), int(row["pkl_turn"])))
    return dict(grouped)


def representative_rows(rows: list[dict], count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("--frames-per-dialog must be positive")
    if len(rows) <= count:
        return rows
    indexes = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[int(index)] for index in indexes]


def extract_frame(
    source_video: Path,
    timestamp: float,
    output: Path,
    width: int,
) -> float:
    # A few forced-alignment endpoints extend slightly beyond the decodable
    # video duration. Step backwards rather than losing the whole dialog.
    candidates = [max(0.0, timestamp - offset) for offset in (0.0, 0.5, 1.0, 2.0, 4.0)]
    for candidate in dict.fromkeys(candidates):
        output.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{candidate:.4f}",
                "-i",
                str(source_video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:-2",
                "-q:v",
                "2",
                str(output),
            ],
            check=False,
        )
        if result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
            return candidate
    raise RuntimeError(
        f"Could not decode a frame near {timestamp:.4f}s from {source_video}"
    )


def make_contact_sheet(
    frames: list[Path],
    output: Path,
    dialog_id: str,
    jpeg_quality: int,
) -> None:
    if not frames:
        raise ValueError(f"No frames supplied for {dialog_id}")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for frame in frames:
        command.extend(["-i", str(frame)])
    inputs = "".join(f"[{index}:v]" for index in range(len(frames)))
    safe_label = dialog_id.replace("'", "")
    filter_complex = (
        f"{inputs}hstack=inputs={len(frames)},"
        "pad=iw:ih+42:0:42:color=black,"
        f"drawtext=text='{safe_label}':x=12:y=10:"
        "fontsize=24:fontcolor=white"
    )
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-frames:v",
            "1",
            "-q:v",
            str(jpeg_quality),
            str(output),
        ]
    )
    subprocess.run(command, check=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    dialogs = load_dialogs(args.manifest)
    dialog_ids = sorted(dialogs)
    if args.limit > 0:
        dialog_ids = dialog_ids[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, dialog_id in enumerate(dialog_ids, start=1):
        rows = dialogs[dialog_id]
        selected = representative_rows(rows, args.frames_per_dialog)
        sheet_path = args.output_dir / f"{dialog_id}.jpg"
        timestamps = [
            (float(row["start"]) + float(row["end"])) / 2.0 for row in selected
        ]
        if args.overwrite or not sheet_path.is_file():
            with tempfile.TemporaryDirectory(
                prefix=f"{dialog_id}_", dir=args.output_dir
            ) as temp_dir:
                frame_paths = []
                actual_timestamps = []
                for frame_index, (row, timestamp) in enumerate(
                    zip(selected, timestamps)
                ):
                    frame_path = Path(temp_dir) / f"frame_{frame_index:02d}.jpg"
                    actual_timestamp = extract_frame(
                        Path(row["source_video"]),
                        timestamp,
                        frame_path,
                        args.frame_width,
                    )
                    actual_timestamps.append(actual_timestamp)
                    frame_paths.append(frame_path)
                make_contact_sheet(
                    frame_paths,
                    sheet_path,
                    dialog_id,
                    args.jpeg_quality,
                )
                timestamps = actual_timestamps
        entries.append(
            {
                "dialog_id": dialog_id,
                "contact_sheet": str(sheet_path.resolve()),
                "source_video": str(Path(rows[0]["source_video"]).resolve()),
                "frame_timestamps": [round(value, 4) for value in timestamps],
                "utterance_ids": [str(row["utterance_id"]) for row in selected],
            }
        )
        print(f"[{index:03d}/{len(dialog_ids):03d}] {dialog_id}: {sheet_path}")

    payload = {
        "schema": "iemocap_speaker_side_annotation_frames_v1",
        "manifest": str(args.manifest.resolve()),
        "dialog_count": len(entries),
        "frames_per_dialog": args.frames_per_dialog,
        "entries": entries,
    }
    index_path = args.output_dir / "index.json"
    atomic_write_json(index_path, payload)
    print(f"saved index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
