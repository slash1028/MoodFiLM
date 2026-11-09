#!/usr/bin/env python3
"""Build a complete, turn-aligned IEMOCAP video manifest.

The project PKL contains a filtered subsequence of the original transcripts, so
PKL turn indexes must not be treated as original transcript indexes. This tool
aligns by normalized text, speaker, and monotonic dialogue order. Missing clips
are cut from the original dialogue video.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
from pathlib import Path

from audit_iemocap_video_alignment import (
    EVAL_RE,
    TRANSCRIPT_RE,
    UTTERANCE_RE,
    align_pkl_turns,
    session_for_dialog,
)


TRANSCRIPT_TIME_RE = re.compile(
    r"^(?P<utt>\S+)\s*\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s*:\s*(?P<text>.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iemocap-pkl",
        type=Path,
        default=Path("original_data/iemocap/iemocap.pkl"),
    )
    parser.add_argument(
        "--iemocap-root",
        type=Path,
        default=Path("/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release"),
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/"
            "video_with_audio_min2frames/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("original_data/iemocap/video_clips/aligned_complete"),
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_transcript_with_times(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRANSCRIPT_TIME_RE.match(line.strip())
        if not match:
            continue
        rows.append(
            {
                "utterance_id": match.group("utt"),
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "text": match.group("text"),
            }
        )
    return rows


def read_eval_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EVAL_RE.match(line.strip())
        if not match:
            continue
        rows[match.group("utt")] = {
            "start": float(match.group("start")),
            "end": float(match.group("end")),
            "emotion": match.group("emotion"),
            "vad": [float(value.strip()) for value in match.group("vad").split(",")],
        }
    return rows


def cut_clip(
    source: Path,
    output: Path,
    start: float,
    end: float,
    fps: float,
    height: int,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        try:
            if probe_frames(output) >= 2:
                return
        except Exception:
            pass
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    filters = [
        f"trim=duration={duration:.6f}",
        "setpts=PTS-STARTPTS",
        "tpad=stop_mode=clone:stop_duration=1",
    ]
    if fps > 0:
        filters.append(f"fps={fps:g}")
    if height > 0:
        filters.append(f"scale=-2:{height}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{max(duration, 1.0):.6f}",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def probe_frames(path: Path) -> int:
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
    streams = json.loads(subprocess.check_output(cmd, text=True)).get("streams") or []
    if not streams:
        return 0
    value = streams[0].get("nb_read_frames") or streams[0].get("nb_frames") or 0
    return int(value)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    data = pickle.load(args.iemocap_pkl.open("rb"))
    speaker_dict = data[0]
    label_dict = data[1]
    sentence_dict = data[2]
    dialog_ids = list(data[3]) + list(data[4]) + list(data[5])

    base_rows = {}
    for line in args.base_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        base_rows[str(row.get("utterance_id") or row.get("sample_id"))] = row

    output_rows = []
    alignment_warnings = []
    newly_cut = 0
    for dialog_id in dialog_ids:
        session = session_for_dialog(dialog_id)
        transcript_path = (
            args.iemocap_root
            / session
            / "dialog"
            / "transcriptions"
            / f"{dialog_id}.txt"
        )
        eval_path = (
            args.iemocap_root
            / session
            / "dialog"
            / "EmoEvaluation"
            / f"{dialog_id}.txt"
        )
        transcript_timed = read_transcript_with_times(transcript_path)
        transcript_rows = [
            (row["utterance_id"], row["text"]) for row in transcript_timed
        ]
        transcript_by_id = {row["utterance_id"]: row for row in transcript_timed}
        eval_by_id = read_eval_rows(eval_path)
        dialog_warnings = []
        aligned = align_pkl_turns(
            dialog_id=dialog_id,
            sentences=sentence_dict[dialog_id],
            speakers=speaker_dict[dialog_id],
            transcript_rows=transcript_rows,
            errors=dialog_warnings,
        )
        fatal = [
            warning
            for warning in dialog_warnings
            if warning["kind"] not in {"ambiguous_text_speaker_match"}
        ]
        if fatal:
            raise RuntimeError(f"Fatal alignment errors for {dialog_id}: {fatal[:3]}")
        alignment_warnings.extend(dialog_warnings)

        if len(aligned) != len(sentence_dict[dialog_id]):
            raise RuntimeError(
                f"Incomplete alignment for {dialog_id}: "
                f"{len(aligned)} / {len(sentence_dict[dialog_id])}"
            )

        source_video = (
            args.iemocap_root
            / session
            / "dialog"
            / "avi"
            / "DivX"
            / f"{dialog_id}.avi"
        )
        for turn, utt_id, transcript_text in aligned:
            parsed = UTTERANCE_RE.match(utt_id)
            if parsed is None or parsed.group("dialog") != dialog_id:
                raise RuntimeError(f"Invalid aligned utterance id: {dialog_id} -> {utt_id}")
            expected_speaker = str(speaker_dict[dialog_id][turn]).strip().upper()
            if parsed.group("speaker") != expected_speaker:
                raise RuntimeError(
                    f"Speaker mismatch: {dialog_id}_{turn} -> {utt_id}, "
                    f"expected {expected_speaker}"
                )

            row = dict(base_rows.get(utt_id, {}))
            if not row:
                timing = eval_by_id.get(utt_id) or transcript_by_id.get(utt_id)
                if timing is None:
                    raise RuntimeError(f"No timestamps available for {utt_id}")
                output_video = (
                    args.output_root
                    / "missing_clips"
                    / session
                    / dialog_id
                    / f"{utt_id}.mp4"
                )
                cut_clip(
                    source=source_video,
                    output=output_video,
                    start=float(timing["start"]),
                    end=float(timing["end"]),
                    fps=args.fps,
                    height=args.height,
                    overwrite=args.overwrite,
                )
                frames = probe_frames(output_video)
                if frames < 2:
                    raise RuntimeError(f"New clip has fewer than two frames: {utt_id} ({frames})")
                evaluation = eval_by_id.get(utt_id, {})
                row = {
                    "session": session,
                    "dialog_id": dialog_id,
                    "utterance_id": utt_id,
                    "emotion": evaluation.get("emotion", "unlabeled"),
                    "start": float(timing["start"]),
                    "end": float(timing["end"]),
                    "vad": evaluation.get("vad", []),
                    "source_video": str(source_video),
                    "sample_id": utt_id,
                    "duration": float(timing["end"]) - float(timing["start"]),
                    "video_path": str(output_video.resolve()),
                    "include_audio": False,
                    "fps": args.fps,
                    "height": args.height,
                    "alignment_completion_clip": True,
                    "video_frames": frames,
                }
                newly_cut += 1

            video_path = Path(str(row["video_path"]))
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            row["turn_key"] = f"{dialog_id}_{turn}"
            row["pkl_turn"] = turn
            row["pkl_speaker"] = expected_speaker
            row["pkl_label"] = int(label_dict[dialog_id][turn])
            row["aligned_transcript_text"] = transcript_text
            output_rows.append(row)

    turn_keys = [row["turn_key"] for row in output_rows]
    utterance_ids = [row["utterance_id"] for row in output_rows]
    if len(turn_keys) != len(set(turn_keys)):
        raise RuntimeError("Duplicate turn_key in aligned manifest")
    if len(utterance_ids) != len(set(utterance_ids)):
        raise RuntimeError("Duplicate utterance_id in aligned manifest")

    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "schema": "iemocap_turn_aligned_video_manifest_v1",
        "iemocap_pkl": str(args.iemocap_pkl.resolve()),
        "base_manifest": str(args.base_manifest.resolve()),
        "manifest": str(manifest_path.resolve()),
        "rows": len(output_rows),
        "unique_turn_keys": len(set(turn_keys)),
        "unique_utterance_ids": len(set(utterance_ids)),
        "reused_clips": len(output_rows) - newly_cut,
        "newly_cut_clips": newly_cut,
        "alignment_ambiguity_warnings": len(alignment_warnings),
        "fps": args.fps,
        "height": args.height,
    }
    (args.output_root / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "alignment_warnings.json").write_text(
        json.dumps(alignment_warnings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
