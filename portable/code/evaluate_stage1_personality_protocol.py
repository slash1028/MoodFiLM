#!/usr/bin/env python
"""Evaluate utterance-level personality predictions before/after speaker averaging."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TRAITS = ("O", "C", "E", "A", "N")


def parse_vector(value):
    text = str(value or "")
    try:
        values = json.loads(text)
        if isinstance(values, list) and len(values) == 5:
            return [float(v) for v in values]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    found = re.findall(r"([OCEAN])\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    mapping = {key.upper(): float(number) for key, number in found}
    if all(key in mapping for key in TRAITS):
        return [mapping[key] for key in TRAITS]
    return None


def ccc(gold, pred):
    count = len(gold)
    if count < 2:
        return None
    mean_gold = sum(gold) / count
    mean_pred = sum(pred) / count
    covariance = sum((x - mean_gold) * (y - mean_pred) for x, y in zip(gold, pred)) / count
    var_gold = sum((x - mean_gold) ** 2 for x in gold) / count
    var_pred = sum((y - mean_pred) ** 2 for y in pred) / count
    denom = var_gold + var_pred + (mean_gold - mean_pred) ** 2
    return None if denom <= 0 else 2.0 * covariance / denom


def report(items):
    per_trait = {}
    for trait_index, trait in enumerate(TRAITS):
        per_trait[trait] = ccc(
            [item[0][trait_index] for item in items],
            [item[1][trait_index] for item in items],
        )
    valid = [value for value in per_trait.values() if value is not None]
    return {
        "samples": len(items),
        "per_trait_ccc": {key: None if value is None else round(value, 4) for key, value in per_trait.items()},
        "avg_ccc": None if not valid else round(sum(valid) / len(valid), 4),
    }


def read_records(path):
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text.lstrip())
    records, _ = decoder.raw_decode(text.lstrip()[end:].lstrip())
    return records


def group_key(record):
    sample_id = str(record.get("sample_id", ""))
    prefix = "Personality_utterance_"
    if not sample_id.startswith(prefix):
        return None
    payload = sample_id[len(prefix):]
    if "_" not in payload:
        return None
    dialogue_id, _ = payload.rsplit("_", 1)
    match = re.search(r"Given this utterance from Speaker_(\d+)", str(record.get("input", "")))
    return None if match is None else (dialogue_id, int(match.group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    utterance_items = []
    grouped = defaultdict(list)
    for record in read_records(args.predictions):
        key = group_key(record)
        gold = parse_vector(record.get("target"))
        pred = parse_vector(record.get("output"))
        if key is None or gold is None or pred is None:
            continue
        utterance_items.append((gold, pred))
        grouped[key].append((gold, pred))

    speaker_items = []
    for items in grouped.values():
        gold = items[0][0]
        pred = [sum(item[1][idx] for item in items) / len(items) for idx in range(5)]
        speaker_items.append((gold, pred))
    result = {
        "protocol": {"training": "utterance", "inference": "utterance_or_speaker_mean"},
        "source_predictions": str(Path(args.predictions).resolve()),
        "utterance_to_utterance": report(utterance_items),
        "utterance_to_conversation": report(speaker_items),
        "valid_speaker_profiles": len(speaker_items),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
