"""Prepare controlled IEMOCAP stage-1 pair-task data from the audited A+T set.

Only pair-task supervision changes. The speaker and continuous-personality rows,
full-conversation audio IDs, and valid/test rows remain identical in every arm.
"""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path


TRAITS = "OCEAN"
RANK_RE = re.compile(r"^TraitRank_([OCEAN]): Speaker_([01])$")
INSTRUCTION_RE = re.compile(
    r"Output exactly TraitRank_([OCEAN]): Speaker_0 or TraitRank_\1: Speaker_1\."
)


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def personality_key(row):
    sample_id = str(row.get("sample_id", ""))
    if not sample_id.startswith("Personality_full_"):
        return None
    payload = sample_id.removeprefix("Personality_full_")
    conv, speaker = payload.rsplit("_", 1)
    return conv, int(speaker)


def make_ternary(row, scores, threshold):
    result = copy.deepcopy(row)
    match = RANK_RE.fullmatch(str(row["target"]))
    if match is None:
        raise ValueError(f"Not a binary pair row: {row['target']!r}")
    trait = match.group(1)
    conv, _ = personality_key(row)
    score0, score1 = scores[conv][0][TRAITS.index(trait)], scores[conv][1][TRAITS.index(trait)]
    delta = score0 - score1
    label = "Similar" if abs(delta) <= threshold else (
        "Speaker_0_higher" if delta > 0 else "Speaker_1_higher"
    )
    result["target"] = f"TraitCompare_{trait}: {label}"
    result["input"], replacements = INSTRUCTION_RE.subn(
        f"Output exactly TraitCompare_{trait}: Speaker_0_higher, "
        f"TraitCompare_{trait}: Similar, or TraitCompare_{trait}: Speaker_1_higher. "
        f"Use Similar when the two speakers' {trait} levels are close.",
        row["input"], count=1,
    )
    if replacements != 1:
        raise ValueError(f"Missing pair instruction for {row['sample_id']} / {trait}")
    result["pairwise_gold_abs_diff"] = round(abs(delta), 4)
    return result


def validate_audio(rows, kind):
    for row in rows:
        target = str(row.get("target", ""))
        if not target.startswith(("TraitRank_", "TraitCompare_")):
            continue
        ids = row.get("audio_turn_sample_ids")
        markers = row["input"].count("[AUDIO_UTTERANCE]")
        if not isinstance(ids, list) or not ids or len(ids) != markers:
            raise ValueError(f"{kind}: audio marker/id mismatch: {row['sample_id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--strong-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.threshold != 0.5:
        raise ValueError("This preregistered comparison fixes Similar at |difference| <= 0.5")

    source = {split: read_rows(args.source / f"{split}.json") for split in ("train", "valid", "test")}
    strong = read_rows(args.strong_source / "train.json")
    train = source["train"]
    base = [row for row in train if not RANK_RE.fullmatch(str(row.get("target", "")))]
    all_pairs = [row for row in train if RANK_RE.fullmatch(str(row.get("target", "")))]
    strong_pairs = [row for row in strong if RANK_RE.fullmatch(str(row.get("target", "")))]
    if (len(base), len(all_pairs), len(strong_pairs)) != (7755, 1080, 736):
        raise ValueError("Unexpected source counts; refusing to build a non-comparable grid")
    if [row for row in strong if not RANK_RE.fullmatch(str(row.get("target", "")))] != base:
        raise ValueError("Strong-pair source changed its non-pair training rows")

    scores = {}
    for row in base:
        key = personality_key(row)
        if key is None:
            continue
        try:
            values = json.loads(row["target"])
        except (TypeError, ValueError):
            continue
        if isinstance(values, list) and len(values) >= 5:
            conv, speaker = key
            scores.setdefault(conv, {})[speaker] = tuple(float(value) for value in values[:5])
    if len(scores) != 108 or any(set(by_speaker) != {0, 1} for by_speaker in scores.values()):
        raise ValueError("Expected 108 two-speaker training conversations")

    ternary = [make_ternary(row, scores, args.threshold) for row in all_pairs]
    labels = Counter(row["target"].split(": ", 1)[1] for row in ternary)
    if labels["Similar"] != 344:
        raise ValueError(f"Expected 344 repeated Similar rows, got {labels['Similar']}")
    variants = {
        "none": base,
        "binary_all": train,
        "binary_strong05": base + strong_pairs,
        "ternary05": base + ternary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, train_rows in variants.items():
        validate_audio(train_rows, name)
        directory = args.out / f"data_{name}"
        if directory.exists():
            manifest_path = directory / "grid_manifest.json"
            if not manifest_path.exists():
                raise FileExistsError(f"Existing unverified data directory: {directory}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("train_rows") != len(train_rows) or manifest.get("source") != str(args.source.resolve()):
                raise ValueError(f"Existing data directory is from a different grid: {directory}")
            print(f"[DATA REUSE] {name}: train={len(train_rows)}", flush=True)
            continue
        directory.mkdir()
        write_rows(directory / "train.json", train_rows)
        for split in ("valid", "test"):
            write_rows(directory / f"{split}.json", source[split])
        manifest = {
            "source": str(args.source.resolve()), "strong_source": str(args.strong_source.resolve()),
            "variant": name, "threshold": args.threshold, "train_rows": len(train_rows),
            "train_pair_rows": len(train_rows) - len(base), "valid_rows": len(source["valid"]),
            "test_rows": len(source["test"]), "ternary_labels": dict(labels) if name == "ternary05" else None,
        }
        (directory / "grid_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[DATA BUILT] {name}: train={len(train_rows)} pair={len(train_rows) - len(base)}", flush=True)


if __name__ == "__main__":
    main()
