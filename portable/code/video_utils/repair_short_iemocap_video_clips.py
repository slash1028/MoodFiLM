#!/usr/bin/env python3
"""Repair very short IEMOCAP utterance clips for Qwen2.5-Omni video input.

Qwen's video reader requires at least two frames. Some utterance clips cut at
2 FPS are shorter than one second and can contain only one video frame. This
script creates a derived manifest where only those short clips are re-encoded
with cloned tail frames; all normal clips keep their original path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument(
        "--duration-threshold",
        type=float,
        default=1.0,
        help="Repair rows whose manifest duration is below this many seconds.",
    )
    parser.add_argument(
        "--scan-decord",
        action="store_true",
        help="Also repair clips that decord can read as fewer than --min-frames frames.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_video_frames(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    streams = json.loads(out).get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    value = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_decord_frames(path: Path) -> int | None:
    try:
        from decord import VideoReader, cpu

        return len(VideoReader(str(path), ctx=cpu(0)))
    except Exception:  # noqa: BLE001
        return None


def repair_clip(src: Path, dst: Path, fps: float, height: int, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return

    scale = f"scale=-2:{height}" if height > 0 else "scale=iw:ih"
    vf = f"tpad=stop_mode=clone:stop_duration=1,fps={fps:g},{scale}"
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    run(cmd)


def main() -> None:
    args = parse_args()
    args.manifest_jsonl = args.manifest_jsonl.resolve()
    args.output_root = args.output_root.resolve()
    rows = [json.loads(line) for line in args.manifest_jsonl.open()]
    output_manifest = args.output_root / "manifest.jsonl"
    repaired_root = args.output_root / "repaired_clips"
    args.output_root.mkdir(parents=True, exist_ok=True)

    repaired = 0
    kept = 0
    failed: list[dict] = []
    output_rows = []

    for row in rows:
        src = Path(row["video_path"])
        duration = float(row.get("duration") or 0.0)
        decord_frames = probe_decord_frames(src) if args.scan_decord else None
        needs_repair = duration < args.duration_threshold
        if args.scan_decord and (decord_frames is None or decord_frames < args.min_frames):
            needs_repair = True

        out_row = dict(row)
        if needs_repair:
            rel_parts = src.parts[-3:]
            dst = repaired_root.joinpath(*rel_parts)
            try:
                repair_clip(src, dst, args.fps, args.height, args.overwrite)
                frames = probe_video_frames(dst)
                if frames is None or frames < args.min_frames:
                    raise RuntimeError(f"repaired clip has {frames} frames")
                out_row["original_video_path"] = str(src)
                out_row["video_path"] = str(dst)
                reasons = []
                if duration < args.duration_threshold:
                    reasons.append(f"duration_lt_{args.duration_threshold:g}s")
                if args.scan_decord and (decord_frames is None or decord_frames < args.min_frames):
                    reasons.append(f"decord_frames_{decord_frames}")
                out_row["repair_reason"] = ",".join(reasons) or "short_clip"
                out_row["repaired_video_frames"] = frames
                repaired += 1
            except Exception as exc:  # noqa: BLE001
                failed.append(
                    {
                        "utterance_id": row.get("utterance_id"),
                        "video_path": str(src),
                        "error": repr(exc),
                    }
                )
        else:
            kept += 1
        output_rows.append(out_row)

    with output_manifest.open("w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_manifest": str(args.manifest_jsonl),
        "output_manifest": str(output_manifest),
        "total": len(rows),
        "kept_original": kept,
        "repaired": repaired,
        "failed": len(failed),
        "failed_items": failed,
        "fps": args.fps,
        "height": args.height,
        "duration_threshold": args.duration_threshold,
        "min_frames": args.min_frames,
    }
    (args.output_root / "repair_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
