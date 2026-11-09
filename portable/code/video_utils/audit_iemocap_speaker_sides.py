#!/usr/bin/env python3
"""Infer and audit IEMOCAP split-screen speaker sides from motion evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
        "--output",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/aligned_complete/"
            "speaker_side_audit.json"
        ),
    )
    parser.add_argument("--samples-per-speaker", type=int, default=6)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.05,
        help="Minimum normalized separation for automatic acceptance.",
    )
    return parser.parse_args()


def decode_gray_frames(path: str, fps: float, width: int) -> np.ndarray:
    height = max(2, int(round(width * 2.0 / 3.0)) // 2 * 2)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        f"fps={fps:g},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_size = width * height
    if frame_size <= 0 or len(raw) < frame_size * 2:
        return np.empty((0, height, width), dtype=np.uint8)
    usable = len(raw) // frame_size * frame_size
    return np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, height, width)


def side_motion_scores(path: str, fps: float, width: int) -> tuple[float, float]:
    frames = decode_gray_frames(path, fps=fps, width=width)
    if frames.shape[0] < 2:
        return 0.0, 0.0
    differences = np.abs(np.diff(frames.astype(np.float32), axis=0))
    half = differences.shape[2] // 2
    # Ignore borders and the bottom edge; actor face/upper body dominates here.
    y0 = int(differences.shape[1] * 0.08)
    y1 = int(differences.shape[1] * 0.88)
    margin = max(1, int(half * 0.04))
    left = differences[:, y0:y1, margin : half - margin]
    right = differences[:, y0:y1, half + margin : -margin]
    # Pixel-wise medians are usually zero because most of each frame is static.
    # Mean absolute change retains sparse mouth/head/body motion.
    return float(np.mean(left)), float(np.mean(right))


def evenly_sample(rows: list[dict], count: int, min_duration: float) -> list[dict]:
    candidates = [row for row in rows if float(row.get("duration", 0.0)) >= min_duration]
    if not candidates:
        candidates = rows
    candidates = sorted(candidates, key=lambda row: int(row.get("pkl_turn", 0)))
    if len(candidates) <= count:
        return candidates
    indexes = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [candidates[index] for index in indexes]


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"F": [], "M": []}
    )
    for row in rows:
        speaker = str(row.get("pkl_speaker", "")).strip().upper()
        if speaker in {"F", "M"}:
            grouped[str(row["dialog_id"])][speaker].append(row)

    dialog_speaker_sides = {}
    details = {}
    review_required = []
    for dialog_id in sorted(grouped):
        evidence = {}
        for speaker in ("F", "M"):
            selected = evenly_sample(
                grouped[dialog_id][speaker],
                args.samples_per_speaker,
                args.min_duration,
            )
            deltas = []
            samples = []
            for row in selected:
                left, right = side_motion_scores(
                    str(row["video_path"]),
                    fps=args.fps,
                    width=args.width,
                )
                delta = left - right
                deltas.append(delta)
                samples.append(
                    {
                        "turn_key": row["turn_key"],
                        "utterance_id": row["utterance_id"],
                        "left_motion": round(left, 6),
                        "right_motion": round(right, 6),
                        "left_minus_right": round(delta, 6),
                    }
                )
            evidence[speaker] = {
                "median_left_minus_right": (
                    float(np.median(deltas)) if deltas else 0.0
                ),
                "samples": samples,
            }

        female_delta = evidence["F"]["median_left_minus_right"]
        male_delta = evidence["M"]["median_left_minus_right"]
        separation = female_delta - male_delta
        scale = max(abs(female_delta), abs(male_delta), 1e-6)
        confidence = abs(separation) / scale
        if separation > 0:
            mapping = {"F": "left", "M": "right"}
        else:
            mapping = {"F": "right", "M": "left"}

        mocap_match = re.match(r"^Ses\d{2}([FM])_", dialog_id)
        mocap_speaker = mocap_match.group(1) if mocap_match else None
        inferred_mocap_side = mapping.get(mocap_speaker)
        accepted = confidence >= args.min_confidence
        if not accepted:
            review_required.append(dialog_id)
        dialog_speaker_sides[dialog_id] = mapping
        details[dialog_id] = {
            "mapping": mapping,
            "motion_separation": round(separation, 6),
            "confidence": round(confidence, 6),
            "accepted": accepted,
            "mocap_speaker": mocap_speaker,
            "inferred_mocap_side": inferred_mocap_side,
            "evidence": evidence,
        }

    payload = {
        "schema": "iemocap_dialog_speaker_side_audit_v1",
        "method": (
            "Per-dialog left/right temporal motion comparison over aligned "
            "female and male utterances. Low-confidence dialogs require review."
        ),
        "manifest": str(args.manifest.resolve()),
        "dialog_count": len(grouped),
        "samples_per_speaker": args.samples_per_speaker,
        "min_confidence": args.min_confidence,
        "review_required": review_required,
        "dialog_speaker_sides": dialog_speaker_sides,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dialog_count": len(grouped),
                "review_required_count": len(review_required),
                "review_required": review_required,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
