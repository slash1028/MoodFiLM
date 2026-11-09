#!/usr/bin/env python3
"""Build MELD Stage2 emotion JSONL with PELD GT personality metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TRAITS = ("O", "C", "E", "A", "N")
EMOTION_MAP = {
    "neutral": "neutral",
    "surprise": "surprise",
    "fear": "fear",
    "sadness": "sad",
    "joy": "joyful",
    "disgust": "disgust",
    "anger": "angry",
}
EMOTION_CHOICES = "<neutral, surprise, fear, sad, joyful, disgust, angry>"
SPLIT_FILES = {
    "train": "train_splits/train_sent_emo.csv",
    "valid": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}
DEFAULT_PELD = {
    "Chandler": [0.648, 0.375, 0.386, 0.58, 0.477],
    "Joey": [0.574, 0.614, 0.297, 0.545, 0.455],
    "Monica": [0.713, 0.457, 0.457, 0.66, 0.511],
    "Phoebe": [0.6, 0.48, 0.31, 0.46, 0.56],
    "Rachel": [0.635, 0.354, 0.521, 0.552, 0.469],
    "Ross": [0.722, 0.489, 0.6, 0.533, 0.356],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld-raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peld-personality-json", type=Path, required=True)
    parser.add_argument(
        "--dialogue-personality-csv",
        type=Path,
        default=None,
        help="Optional 1-7 OCEAN predictions keyed by split, dialogue_id, speaker.",
    )
    parser.add_argument("--context-window", type=int, default=12)
    parser.add_argument("--include-prompt-personality", action="store_true")
    parser.add_argument(
        "--prompt-personality-mode",
        choices=("none", "legacy", "target_iemocap"),
        default="none",
        help=(
            "none: no personality text; legacy: original MELD target plus aggregated "
            "context personality text; target_iemocap: target speaker only in the "
            "IEMOCAP OCEAN prompt style."
        ),
    )
    parser.add_argument("--include-demo", action="store_true")
    parser.add_argument("--include-audio-markers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.include_prompt_personality and args.prompt_personality_mode == "none":
        args.prompt_personality_mode = "legacy"
    return args


def canonical_speaker(name: str) -> str:
    text = str(name or "").strip()
    for speaker in DEFAULT_PELD:
        if text.lower() == speaker.lower():
            return speaker
    return text


def load_or_write_peld(path: Path) -> Dict[str, List[float]]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = DEFAULT_PELD
        with path.open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2, ensure_ascii=False)
    result: Dict[str, List[float]] = {}
    for speaker, values in raw.items():
        if not isinstance(values, (list, tuple)) or len(values) != 5:
            continue
        result[canonical_speaker(speaker)] = [float(item) for item in values]
    return result


def peld_to_prefix(values: Optional[List[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    return [1.0 + 6.0 * float(item) for item in values]


def load_dialogue_personality(path: Optional[Path]) -> Dict[Tuple[str, int, str], List[float]]:
    """Load predicted 1-7 profiles and normalize them to the builder's 0-1 scale."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    result: Dict[Tuple[str, int, str], List[float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = str(row["split"]).strip().lower()
            if split == "dev":
                split = "valid"
            dialogue_id = int(row["dialogue_id"])
            speaker = canonical_speaker(row["speaker"])
            values = [float(row[trait]) for trait in TRAITS]
            if not all(1.0 <= value <= 7.0 for value in values):
                raise ValueError(f"Out-of-range dialogue personality: {split}/{dialogue_id}/{speaker}")
            result[(split, dialogue_id, speaker)] = [(value - 1.0) / 6.0 for value in values]
    return result


def trait_level(value: float) -> str:
    if value < 0.4:
        return "low"
    if value > 0.6:
        return "high"
    return "medium"


def personality_text(values: Optional[List[float]]) -> str:
    if values is None:
        return "unknown"
    names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return ", ".join(f"{trait_level(v)} {name}" for v, name in zip(values, names))


def likert_trait_level(value: float) -> str:
    if value < 3.0:
        return "low"
    if value > 5.0:
        return "high"
    return "medium"


def format_likert(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def iemocap_style_personality_text(values: Optional[List[float]]) -> str:
    if values is None:
        return "Target speaker personality profile is unknown."
    scores = peld_to_prefix(values)
    score_text = ", ".join(
        f"{trait}:{format_likert(score)}"
        for trait, score in zip(TRAITS, scores)
    )
    names = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    tendencies = ", ".join(
        f"{likert_trait_level(score)} {name}"
        for score, name in zip(scores, names)
    )
    return (
        "Target speaker personality profile (OCEAN, 1.0-7.0 where higher means stronger trait) is "
        f"<{score_text}>. "
        f"Interpreted tendencies: {tendencies}. "
        "Use this profile to disambiguate subtle emotions in tone and wording."
    )


def read_split(root: Path, split: str) -> Dict[int, List[dict]]:
    path = root / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(path)
    dialogues: Dict[int, List[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])
            speaker = canonical_speaker(row.get("Speaker", ""))
            media_sample_id = f"dia{dialogue_id}_utt{utterance_id}"
            dialogues.setdefault(dialogue_id, []).append(
                {
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "sample_id": f"{split}_{media_sample_id}",
                    "media_sample_id": media_sample_id,
                    "speaker": speaker,
                    "utterance": str(row.get("Utterance", "")).strip(),
                    "emotion": EMOTION_MAP.get(str(row.get("Emotion", "")).strip().lower(), str(row.get("Emotion", "")).strip().lower()),
                }
            )
    for turns in dialogues.values():
        turns.sort(key=lambda item: item["utterance_id"])
    return dialogues


def speaker_aliases(turns: List[dict]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for turn in turns:
        speaker = turn["speaker"]
        if speaker not in aliases:
            aliases[speaker] = f"Speaker_{len(aliases)}"
    return aliases


def weighted_context_personality(
    history: List[dict],
    target_speaker: str,
    peld: Dict[str, List[float]],
) -> Optional[List[float]]:
    weighted = [0.0] * 5
    total = 0.0
    for distance, turn in enumerate(reversed(history), start=1):
        speaker = turn["speaker"]
        values = peld.get(speaker)
        if speaker == target_speaker or values is None:
            continue
        weight = 1.0 / float(distance)
        for idx, value in enumerate(values):
            weighted[idx] += weight * float(value)
        total += weight
    if total <= 0:
        return None
    return [value / total for value in weighted]


def speaker_context_set(
    history: List[dict],
    target_speaker: str,
    peld: Dict[str, List[float]],
) -> Tuple[List[str], List[Optional[List[float]]], List[List[float]], List[bool]]:
    """Build target-aware context speaker set without pre-averaging personalities."""
    stats: Dict[str, dict] = {}
    for distance, turn in enumerate(reversed(history), start=1):
        speaker = turn["speaker"]
        if speaker == target_speaker:
            continue
        entry = stats.setdefault(
            speaker,
            {
                "min_distance": distance,
                "turn_count": 0,
                "is_previous": 0.0,
            },
        )
        entry["min_distance"] = min(entry["min_distance"], distance)
        entry["turn_count"] += 1
        if distance == 1:
            entry["is_previous"] = 1.0

    ordered = sorted(
        stats.items(),
        key=lambda item: (float(item[1]["min_distance"]), -int(item[1]["turn_count"]), item[0]),
    )
    speakers: List[str] = []
    personalities: List[Optional[List[float]]] = []
    relation_features: List[List[float]] = []
    known: List[bool] = []
    history_len = max(1, len(history))
    for speaker, item in ordered:
        values = peld.get(speaker)
        available = values is not None
        distance = float(item["min_distance"])
        turn_count = float(item["turn_count"])
        speakers.append(speaker)
        personalities.append(peld_to_prefix(values) if available else None)
        relation_features.append(
            [
                1.0 / max(distance, 1.0),
                min(turn_count / float(history_len), 1.0),
                float(item["is_previous"]),
                1.0 if available else 0.0,
            ]
        )
        known.append(available)
    return speakers, personalities, relation_features, known


def demo_examples(train_dialogues: Dict[int, List[dict]]) -> Dict[str, Tuple[str, str]]:
    examples: Dict[str, Tuple[str, str]] = {}
    for dialogue_id in sorted(train_dialogues):
        for turn in train_dialogues[dialogue_id]:
            emotion = turn["emotion"]
            if emotion and emotion not in examples:
                examples[emotion] = (turn["utterance"], emotion)
    return examples


def build_prompt(
    context: List[dict],
    target: dict,
    aliases: Dict[str, str],
    peld: Dict[str, List[float]],
    prompt_personality_mode: str,
    include_demo: bool,
    demos: Dict[str, Tuple[str, str]],
    include_audio_markers: bool,
) -> str:
    pieces = [
        "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ###"
    ]
    for turn in context:
        marker = " [AUDIO_UTTERANCE]" if include_audio_markers else ""
        pieces.append(f'\t {aliases[turn["speaker"]]}:"{turn["utterance"]}"{marker}')
    if prompt_personality_mode == "target_iemocap":
        pieces.append(iemocap_style_personality_text(peld.get(target["speaker"])))
    elif prompt_personality_mode == "legacy":
        target_values = peld.get(target["speaker"])
        pieces.append(
            "The target speaker personality traits: "
            f"{personality_text(target_values)}."
        )
        context_values = weighted_context_personality(
            context[:-1],
            target["speaker"],
            peld,
        )
        if context_values is not None:
            pieces.append(
                "The context speakers' aggregated personality traits: "
                f"{personality_text(context_values)}."
            )
    pieces.append("###")
    pieces.append(
        f'Please select the emotional label of < {aliases[target["speaker"]]}:"{target["utterance"]}"> '
        f"from {EMOTION_CHOICES}:"
    )
    if include_demo:
        utterance, emotion = demos.get(target["emotion"], ("I understand.", "neutral"))
        pieces.append(
            f' A similar utterance and its emotion label are given for reference: '
            f'<Utterance: "{utterance}"; Emotion label: {emotion}>.'
        )
    return " ".join(pieces)


def build_records(
    split: str,
    dialogues: Dict[int, List[dict]],
    peld: Dict[str, List[float]],
    context_window: int,
    prompt_personality_mode: str,
    include_demo: bool,
    demos: Dict[str, Tuple[str, str]],
    include_audio_markers: bool,
    dialogue_personalities: Optional[Dict[Tuple[str, int, str], List[float]]] = None,
) -> List[dict]:
    records: List[dict] = []
    for dialogue_id in sorted(dialogues):
        turns = dialogues[dialogue_id]
        active_personality = peld
        if dialogue_personalities:
            active_personality = {
                speaker: dialogue_personalities[(split, dialogue_id, speaker)]
                for speaker in {turn["speaker"] for turn in turns}
                if (split, dialogue_id, speaker) in dialogue_personalities
            }
        aliases = speaker_aliases(turns)
        for index, target in enumerate(turns):
            start = max(0, index + 1 - context_window) if context_window > 0 else 0
            context = turns[start : index + 1]
            target_values = active_personality.get(target["speaker"])
            context_values = weighted_context_personality(context[:-1], target["speaker"], active_personality)
            (
                context_speakers,
                context_personalities,
                context_relation_features,
                context_personality_known,
            ) = speaker_context_set(context[:-1], target["speaker"], active_personality)
            audio_turn_sample_ids = [turn["sample_id"] for turn in context]
            record = {
                "input": build_prompt(
                    context=context,
                    target=target,
                    aliases=aliases,
                    peld=active_personality,
                    prompt_personality_mode=prompt_personality_mode,
                    include_demo=include_demo,
                    demos=demos,
                    include_audio_markers=include_audio_markers,
                ),
                "target": target["emotion"],
                "sample_id": target["sample_id"],
                "utterance_id": target["sample_id"],
                "audio_turn_sample_ids": audio_turn_sample_ids,
                "speaker_id": aliases[target["speaker"]],
                "speaker_name": target["speaker"],
                "split": split,
                "gt_personality": peld_to_prefix(target_values),
                "personality": peld_to_prefix(target_values) if dialogue_personalities else None,
                "interlocutor_personality": peld_to_prefix(context_values),
                "gt_personality_raw": target_values,
                "interlocutor_personality_raw": context_values,
                "target_personality_known": target_values is not None,
                "context_personality_known": context_values is not None,
                "speaker_context_speakers": context_speakers,
                "speaker_context_personalities": context_personalities,
                "speaker_context_relation_features": context_relation_features,
                "speaker_context_personality_known": context_personality_known,
                "personality_source": "dialogue_predicted" if dialogue_personalities else "global_profile",
            }
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.context_window <= 0:
        raise ValueError("--context-window must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        print(f"[INFO] Reusing existing data dir: {args.output_dir}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peld = load_or_write_peld(args.peld_personality_json)
    dialogue_personalities = load_dialogue_personality(args.dialogue_personality_csv)
    dialogues_by_split = {
        split: read_split(args.meld_raw_root, split)
        for split in ("train", "valid", "test")
    }
    demos = demo_examples(dialogues_by_split["train"])
    summary = {
        "meld_raw_root": str(args.meld_raw_root.resolve()),
        "peld_personality_json": str(args.peld_personality_json.resolve()),
        "dialogue_personality_csv": (
            str(args.dialogue_personality_csv.resolve())
            if args.dialogue_personality_csv is not None else None
        ),
        "num_dialogue_personalities": len(dialogue_personalities),
        "context_window": args.context_window,
        "include_prompt_personality": bool(args.include_prompt_personality),
        "prompt_personality_mode": args.prompt_personality_mode,
        "include_demo": bool(args.include_demo),
        "include_audio_markers": bool(args.include_audio_markers),
        "splits": {},
    }
    for split, dialogues in dialogues_by_split.items():
        records = build_records(
            split=split,
            dialogues=dialogues,
            peld=peld,
            context_window=args.context_window,
            prompt_personality_mode=args.prompt_personality_mode,
            include_demo=args.include_demo,
            demos=demos,
            include_audio_markers=args.include_audio_markers,
            dialogue_personalities=dialogue_personalities,
        )
        out_name = "valid.json" if split == "valid" else f"{split}.json"
        write_jsonl(args.output_dir / out_name, records)
        known_target = sum(1 for item in records if item["target_personality_known"])
        known_context = sum(1 for item in records if item["context_personality_known"])
        summary["splits"][split] = {
            "num_records": len(records),
            "num_dialogues": len(dialogues),
            "target_personality_known": known_target,
            "target_personality_known_pct": round(100.0 * known_target / max(1, len(records)), 3),
            "context_personality_known": known_context,
            "context_personality_known_pct": round(100.0 * known_context / max(1, len(records)), 3),
        }
    with (args.output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
