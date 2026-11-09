#!/usr/bin/env python
"""Export dialogue predictions and leakage-free train-only speaker aggregates."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_stage1_predicted_personality import load_eval_file, parse_personality_continuous

TRAITS = ("O", "C", "E", "A", "N")
PATTERN = re.compile(r"^MELDPersonality_(train|valid|test)_dia(\d+)_(.+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, records = load_eval_file(args.prediction_file)
    rows, failures = [], []
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        match = PATTERN.match(sample_id)
        if match is None:
            continue
        parsed = parse_personality_continuous(record.get("output_raw") or record.get("output") or "")
        if parsed is None:
            failures.append(sample_id)
            continue
        split, dialogue_id, speaker = match.groups()
        rows.append({"split": split, "dialogue_id": int(dialogue_id), "speaker": speaker,
                     **{trait: round(float(parsed[trait]), 3) for trait in TRAITS}})
    if not rows:
        raise RuntimeError("No valid MELD dialogue-personality predictions found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dialogue_personality.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "dialogue_id", "speaker", *TRAITS])
        writer.writeheader(); writer.writerows(rows)
    train_by_speaker = {}
    for row in rows:
        if row["split"] == "train":
            train_by_speaker.setdefault(row["speaker"], []).append(row)
    aggregates = {}
    for speaker, speaker_rows in sorted(train_by_speaker.items()):
        aggregates[speaker] = {
            "num_dialogues": len(speaker_rows),
            "mean": {t: round(statistics.mean(r[t] for r in speaker_rows), 3) for t in TRAITS},
            "median": {t: round(statistics.median(r[t] for r in speaker_rows), 3) for t in TRAITS},
            "std": {t: round(statistics.stdev(r[t] for r in speaker_rows), 3) if len(speaker_rows) > 1 else 0.0 for t in TRAITS},
        }
    payload = {"aggregation_source": "train only", "num_predictions": len(rows),
               "num_parse_failures": len(failures), "parse_failures": failures[:50],
               "speakers": aggregates}
    (args.output_dir / "train_speaker_aggregates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
