#!/usr/bin/env python
"""Inject a continuous five-factor speaker-profile prompt into JSONL splits."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

TRAITS = ("O", "C", "E", "A", "N")
TRAIT_NAMES = {
    "O": "openness", "C": "conscientiousness", "E": "extraversion",
    "A": "agreeableness", "N": "neuroticism",
}


def coerce_profile(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, dict):
        value = [value[key] for key in TRAITS]
    if not isinstance(value, (tuple, list)) or len(value) != 5:
        raise ValueError(f"Invalid five-factor profile: {value!r}")
    return [float(item) for item in value]


def prompt_clause(values):
    ocean = ", ".join(f"{trait}:{value:.3f}" for trait, value in zip(TRAITS, values))
    tendencies = []
    for trait, value in zip(TRAITS, values):
        level = "low" if value < 3.0 else "high" if value > 5.0 else "medium"
        tendencies.append(f"{level} {TRAIT_NAMES[trait]}")
    return (
        "Target speaker personality profile (OCEAN, 1.0-7.0 where higher means stronger trait) is "
        f"<{ocean}>. Interpreted tendencies: {', '.join(tendencies)}. "
        "Use this profile to disambiguate subtle emotions in tone and wording. "
    )


def inject(text, clause):
    if "Target speaker personality profile" in text:
        return text
    updated, count = re.subn(r"(?=Please select the emotional label)", clause, text, count=1)
    if count != 1:
        raise RuntimeError("Could not locate emotion-label instruction in prompt.")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", default="gt_personality")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    injected = 0
    unavailable = 0
    for split in ("train.json", "valid.json", "test.json"):
        source = args.source / split
        if not source.is_file():
            continue
        with source.open(encoding="utf-8") as reader, (args.output / split).open("w", encoding="utf-8") as writer:
            for line in reader:
                row = json.loads(line)
                # MELD does not provide a profile for every named character.
                # Leave such prompts untouched rather than introducing an
                # invented neutral profile as an additional confound.
                value = row.get(args.field)
                if value is None:
                    unavailable += 1
                else:
                    row["input"] = inject(row["input"], prompt_clause(coerce_profile(value)))
                    injected += 1
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / ".profile_prompt_complete").write_text(
        json.dumps({"field": args.field, "injected": injected, "unavailable": unavailable}) + "\n",
        encoding="utf-8",
    )
    print(f"[profile-prompt] injected={injected}, unavailable={unavailable}")


if __name__ == "__main__":
    main()
