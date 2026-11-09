#!/usr/bin/env python3
"""Validate a manual IEMOCAP dialog speaker-side map before feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/video_clips/aligned_complete/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--speaker-side-map",
        type=Path,
        default=Path(
            "/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/video_clips/aligned_complete/"
            "speaker_side_manual.json"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report incomplete annotations without returning a failure status.",
    )
    return parser.parse_args()


def validate_map(manifest: Path, speaker_side_map: Path) -> dict:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_dialogs = {str(row["dialog_id"]) for row in rows}
    payload = json.loads(speaker_side_map.read_text(encoding="utf-8"))
    mappings = payload.get("dialog_speaker_sides")
    details = payload.get("details") or {}
    if not isinstance(mappings, dict):
        raise ValueError("Map must contain a dialog_speaker_sides object")

    invalid = {}
    for dialog_id, mapping in mappings.items():
        if not isinstance(mapping, dict):
            invalid[dialog_id] = "mapping is not an object"
            continue
        if set(mapping) != {"F", "M"}:
            invalid[dialog_id] = "mapping must contain exactly F and M"
            continue
        if {mapping["F"], mapping["M"]} != {"left", "right"}:
            invalid[dialog_id] = "F/M must map to opposite left/right values"

    uncertain = sorted(
        dialog_id
        for dialog_id, detail in details.items()
        if isinstance(detail, dict) and detail.get("status") == "uncertain"
    )
    missing = sorted(expected_dialogs - set(mappings))
    unexpected = sorted(set(mappings) - expected_dialogs)
    return {
        "expected_dialog_count": len(expected_dialogs),
        "mapped_dialog_count": len(mappings),
        "missing_count": len(missing),
        "uncertain_count": len(uncertain),
        "invalid_count": len(invalid),
        "unexpected_count": len(unexpected),
        "missing": missing,
        "uncertain": uncertain,
        "invalid": invalid,
        "unexpected": unexpected,
        "valid_for_extraction": not (missing or uncertain or invalid or unexpected),
    }


def main() -> int:
    args = parse_args()
    report = validate_map(args.manifest, args.speaker_side_map)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["valid_for_extraction"] or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
