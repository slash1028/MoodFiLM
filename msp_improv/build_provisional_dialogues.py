"""Build explicitly approximate MSP scene sequences for pilot experiments.

This does not assert that filename order is recording chronology.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
NAME = re.compile(r"MSP-IMPROV-(S\d{2}[A-Z])-([FM]\d+)-([ST])-([FM]{2})(\d+)$")


def main():
    source = ROOT / "utterances.jsonl"
    if not source.is_file():
        raise SystemExit("Run: python msp_improv/audit.py --output msp_improv/utterances.jsonl")
    groups = defaultdict(list)
    for line in source.open(encoding="utf-8"):
        row = json.loads(line)
        match = NAME.fullmatch(row["id"])
        if not match:
            continue  # P: preparation; R: read sentence
        scene, actor, kind, role, index = match.groups()
        key = (row["session"], scene, actor)
        row["candidate_dialogue_id"] = "_".join(key)
        row["turn_number_local"] = int(index)
        row["speaker_from_readme"] = row["session"] + "_" + role[1]
        row["speaker_label_confidence"] = "requires_audio_validation"
        row["kind"] = kind
        row["role_code"] = role
        groups[key].append(row)

    output = ROOT / "provisional_dialogues.jsonl"
    flags = Counter()
    with output.open("w", encoding="utf-8") as stream:
        for key, turns in sorted(groups.items()):
            # Tie policy is deterministic for reproducibility, not a time claim.
            turns.sort(key=lambda r: (r["turn_number_local"], r["role_code"], r["kind"], r["id"]))
            counts = Counter(r["turn_number_local"] for r in turns)
            ties = sorted(n for n, count in counts.items() if count > 1)
            flags["dialogues"] += 1
            flags["utterances"] += len(turns)
            flags["dialogues_with_ties"] += bool(ties)
            flags["ambiguous_turn_positions"] += sum(counts[n] for n in ties)
            for pos, row in enumerate(turns):
                row["provisional_position"] = pos
                row["order_uncertain"] = row["turn_number_local"] in ties
            stream.write(json.dumps({"dialogue_id": "_".join(key), "split": turns[0]["split"],
                                     "protocol": "approximate_filename_order_v1",
                                     "same_number_ties": ties, "turns": turns}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(flags), indent=2))
    print(output)


if __name__ == "__main__":
    main()
