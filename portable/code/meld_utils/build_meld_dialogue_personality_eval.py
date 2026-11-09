#!/usr/bin/env python
"""Build full-dialogue, target-speaker MELD samples for personality inference."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MAIN_SPEAKERS = {"Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"}
INSTRUCTION = (
    "Predict the OCEAN personality vector [O, C, E, A, N] of this speaker. "
    "Use the Big Five meanings: O=Openness, C=Conscientiousness, "
    "E=Extraversion, A=Agreeableness, N=Neuroticism. Use a 1.0-7.0 Likert "
    "scale where 1.0 is very low, 4.0 is medium, and 7.0 is very high. Each "
    "output value must be a number within [1.0, 7.0]. Output only a valid JSON "
    "array with exactly 5 numbers in this exact order"
)


def read_dialogues(path: Path):
    dialogues = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dialogue_id = int(row["Dialogue_ID"])
            dialogues[dialogue_id].append({
                "utterance_id": int(row["Utterance_ID"]),
                "speaker": row["Speaker"].strip(),
                "utterance": row["Utterance"].strip(),
            })
    for turns in dialogues.values():
        turns.sort(key=lambda item: item["utterance_id"])
    return dialogues


def build_row(split, dialogue_id, turns, speaker, anonymize_speakers=False, audio_markers=False):
    rendered, target_utterances, audio_ids = [], [], []
    speaker_aliases = {
        name: f"Speaker_{index}"
        for index, name in enumerate(dict.fromkeys(turn["speaker"] for turn in turns))
    }
    target_tag = speaker_aliases[speaker] if anonymize_speakers else speaker
    for turn in turns:
        marked = "*" if turn["speaker"] == speaker else ""
        display_speaker = speaker_aliases[turn["speaker"]] if anonymize_speakers else turn["speaker"]
        suffix = " [AUDIO_UTTERANCE]" if audio_markers else ""
        rendered.append(f'{display_speaker}{marked}: {json.dumps(turn["utterance"], ensure_ascii=False)}{suffix}')
        audio_ids.append(f'{split}_dia{dialogue_id}_utt{turn["utterance_id"]}')
        if marked:
            target_utterances.append(turn["utterance"])
    prompt = (
        "Now you are expert of personality analysis. Given the full conversation "
        f"where target speaker turns are marked as <{target_tag}*>: "
        f"<{' '.join(rendered)}> {INSTRUCTION}"
    )
    return {
        "input": prompt,
        "target": "[4.0, 4.0, 4.0, 4.0, 4.0]",
        "sample_id": f"MELDPersonality_{split}_dia{dialogue_id}_{speaker}",
        "speaker_name": speaker,
        "dialogue_id": f"{split}_dia{dialogue_id}",
        "split": split,
        "personality_labels_available": False,
        "audio_turn_sample_ids": audio_ids,
        "debug_target_utterance": target_utterances,
        "speaker_alias": target_tag,
        "speaker_aliases": speaker_aliases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld-raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-speakers-only", action="store_true")
    parser.add_argument("--anonymize-speakers", action="store_true")
    parser.add_argument("--audio-markers", action="store_true",
                        help="Add one ordered audio insertion marker per utterance.")
    args = parser.parse_args()
    split_files = {
        "train": args.meld_raw_root / "train_splits/train_sent_emo.csv",
        "valid": args.meld_raw_root / "dev_sent_emo.csv",
        "test": args.meld_raw_root / "test_sent_emo.csv",
    }
    rows, counts = [], {}
    for split, path in split_files.items():
        split_rows = []
        for dialogue_id, turns in sorted(read_dialogues(path).items()):
            speakers = list(dict.fromkeys(turn["speaker"] for turn in turns))
            if args.main_speakers_only:
                speakers = [speaker for speaker in speakers if speaker in MAIN_SPEAKERS]
            split_rows.extend(
                build_row(split, dialogue_id, turns, speaker, args.anonymize_speakers, args.audio_markers)
                for speaker in speakers
            )
        counts[split] = len(split_rows)
        rows.extend(split_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test.json").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("train", "valid"):
        (args.output_dir / f"{split}.json").write_text("", encoding="utf-8")
    manifest = {"counts": counts, "total": len(rows),
                "main_speakers_only": args.main_speakers_only,
                "anonymize_speakers": args.anonymize_speakers,
                "audio_markers": args.audio_markers,
                "personality_labels_available": False,
                "warning": "Targets are placeholders, not GT. Do not report CCC/MAE against them."}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
