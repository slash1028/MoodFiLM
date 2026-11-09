#!/usr/bin/env python3
"""Cut IEMOCAP dialog-level videos into utterance-level clips.

IEMOCAP ships dialog videos under:

    Session*/dialog/avi/DivX/<dialog_id>.avi

and utterance timestamps under:

    Session*/dialog/EmoEvaluation/<dialog_id>.txt

This script parses the timestamp files and uses ffmpeg to create one clip per
utterance, plus a JSONL manifest that can later be joined with ERC samples by
`utterance_id` / `sample_id`.

The default output is visual-only mp4, capped to a low fps and height to keep
storage manageable. Use `--include-audio` only if you intentionally want the
video file to carry audio as well.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_IEMOCAP_ROOT = Path("/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release")
DEFAULT_OUTPUT_ROOT = Path(
    "/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/video_clips"
)

TIMESTAMP_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<emotion>\S+)\s+\[(?P<vad>[^\]]+)\]"
)


@dataclass(frozen=True)
class ClipJob:
    session: str
    dialog_id: str
    utterance_id: str
    emotion: str
    start: float
    end: float
    vad: list[float]
    source_video: Path
    output_video: Path

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut IEMOCAP dialog-level AVI files into utterance clips."
    )
    parser.add_argument("--iemocap-root", type=Path, default=DEFAULT_IEMOCAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["Session1", "Session2", "Session3", "Session4", "Session5"],
        help="Sessions to process, e.g. Session1 Session5.",
    )
    parser.add_argument(
        "--emotion-filter",
        default="hap,sad,neu,ang,exc,fru",
        help=(
            "Comma-separated labels to keep. Use 'all' to cut every labeled "
            "utterance, including xxx/oth/etc."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Output fps. Use 0 to keep original frame rate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=336,
        help="Output height with aspect ratio preserved. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Keep audio stream in the output clips. Default is visual-only.",
    )
    parser.add_argument(
        "--video-codec",
        default="libx264",
        help="ffmpeg video codec. Default: libx264.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="x264 CRF when using libx264. Lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset. Default: veryfast.",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and write planned manifest, but do not run ffmpeg.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for debugging. 0 means no limit.",
    )
    return parser.parse_args()


def parse_vad(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def iter_eval_files(iemocap_root: Path, sessions: Iterable[str]) -> Iterable[Path]:
    for session in sessions:
        eval_dir = iemocap_root / session / "dialog" / "EmoEvaluation"
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"EmoEvaluation directory not found: {eval_dir}")
        yield from sorted(eval_dir.glob("*.txt"))


def source_video_for_eval_file(eval_file: Path) -> Path:
    # .../Session1/dialog/EmoEvaluation/Ses01F_impro01.txt
    session_dir = eval_file.parents[2]
    dialog_id = eval_file.stem
    return session_dir / "dialog" / "avi" / "DivX" / f"{dialog_id}.avi"


def collect_jobs(args: argparse.Namespace) -> list[ClipJob]:
    if not args.iemocap_root.is_dir():
        raise FileNotFoundError(f"IEMOCAP root not found: {args.iemocap_root}")

    if args.emotion_filter.strip().lower() == "all":
        allowed: set[str] | None = None
    else:
        allowed = {x.strip() for x in args.emotion_filter.split(",") if x.strip()}

    jobs: list[ClipJob] = []
    missing_videos: list[Path] = []

    mode_dir = "video_with_audio" if args.include_audio else "visual_only"
    for eval_file in iter_eval_files(args.iemocap_root, args.sessions):
        source_video = source_video_for_eval_file(eval_file)
        if not source_video.is_file():
            missing_videos.append(source_video)
            continue

        session = eval_file.parents[2].name
        dialog_id = eval_file.stem
        for line in eval_file.read_text(errors="ignore").splitlines():
            match = TIMESTAMP_RE.match(line)
            if not match:
                continue
            emotion = match.group("emotion")
            if allowed is not None and emotion not in allowed:
                continue

            start = float(match.group("start"))
            end = float(match.group("end"))
            if end <= start:
                continue

            utterance_id = match.group("utt")
            output_video = (
                args.output_root
                / mode_dir
                / session
                / dialog_id
                / f"{utterance_id}.mp4"
            )
            jobs.append(
                ClipJob(
                    session=session,
                    dialog_id=dialog_id,
                    utterance_id=utterance_id,
                    emotion=emotion,
                    start=start,
                    end=end,
                    vad=parse_vad(match.group("vad")),
                    source_video=source_video,
                    output_video=output_video,
                )
            )
            if args.limit and len(jobs) >= args.limit:
                break
        if args.limit and len(jobs) >= args.limit:
            break

    if missing_videos:
        preview = "\n".join(str(x) for x in missing_videos[:10])
        raise FileNotFoundError(
            f"{len(missing_videos)} dialog videos are missing. First entries:\n{preview}"
        )

    return jobs


def ffmpeg_command(job: ClipJob, args: argparse.Namespace) -> list[str]:
    duration = f"{job.duration:.6f}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
        "-ss",
        f"{job.start:.6f}",
        "-i",
        str(job.source_video),
        "-t",
        duration,
        "-map",
        "0:v:0",
    ]
    if args.include_audio:
        cmd.extend(["-map", "0:a?"])
    else:
        cmd.append("-an")

    filters: list[str] = []
    if args.fps and args.fps > 0:
        filters.append(f"fps={args.fps:g}")
    if args.height and args.height > 0:
        filters.append(f"scale=-2:{args.height}")
    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    cmd.extend(
        [
            "-c:v",
            args.video_codec,
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if args.include_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "96k"])
    cmd.extend(["-movflags", "+faststart", str(job.output_video)])
    return cmd


def cut_one(job: ClipJob, args: argparse.Namespace) -> tuple[str, str | None]:
    if job.output_video.exists() and not args.overwrite:
        return ("skipped", None)

    job.output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = ffmpeg_command(job, args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return ("failed", proc.stderr.strip() or "ffmpeg failed")
    return ("ok", None)


def manifest_record(job: ClipJob, args: argparse.Namespace) -> dict:
    rec = asdict(job)
    rec["sample_id"] = job.utterance_id
    rec["duration"] = job.duration
    rec["source_video"] = str(job.source_video)
    rec["video_path"] = str(job.output_video)
    rec.pop("output_video", None)
    rec["include_audio"] = bool(args.include_audio)
    rec["fps"] = args.fps
    rec["height"] = args.height
    return rec


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    jobs = collect_jobs(args)
    mode_dir = "video_with_audio" if args.include_audio else "visual_only"
    manifest_path = args.output_root / mode_dir / "manifest.jsonl"
    stats_path = args.output_root / mode_dir / "stats.json"

    print(f"iemocap_root: {args.iemocap_root}")
    print(f"output_root: {args.output_root / mode_dir}")
    print(f"sessions: {args.sessions}")
    print(f"emotion_filter: {args.emotion_filter}")
    print(f"include_audio: {args.include_audio}")
    print(f"fps: {args.fps}, height: {args.height}")
    print(f"jobs: {len(jobs)}")
    print(f"manifest: {manifest_path}")

    write_jsonl(manifest_path, (manifest_record(job, args) for job in jobs))

    counts = {"planned": len(jobs), "ok": 0, "skipped": 0, "failed": 0}
    failures: list[dict] = []

    if not args.dry_run:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            future_to_job = {ex.submit(cut_one, job, args): job for job in jobs}
            done = 0
            for fut in futures.as_completed(future_to_job):
                job = future_to_job[fut]
                status, error = fut.result()
                counts[status] = counts.get(status, 0) + 1
                done += 1
                if error:
                    failures.append(
                        {
                            "utterance_id": job.utterance_id,
                            "source_video": str(job.source_video),
                            "output_video": str(job.output_video),
                            "error": error,
                        }
                    )
                if done % 100 == 0 or done == len(jobs):
                    print(
                        f"progress {done}/{len(jobs)} "
                        f"ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}",
                        flush=True,
                    )
    else:
        print("dry_run=True; no ffmpeg commands executed.")

    stats = {
        **counts,
        "dry_run": args.dry_run,
        "include_audio": args.include_audio,
        "fps": args.fps,
        "height": args.height,
        "emotion_filter": args.emotion_filter,
        "sessions": args.sessions,
        "manifest": str(manifest_path),
        "failures": failures[:50],
        "num_failures_recorded": len(failures),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"stats: {stats_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
