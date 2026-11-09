"""Reassemble the three archived lightweight checkpoints and verify their hashes."""

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = {
    "stage1": (
        "portable/dialogue_personality/checkpoint_07648/trainable_model.safetensors",
        43_306_016,
        "c9dfa86a121d3db6a6c1f4f97448f57637acfa790472b0de62995606a83a303e",
    ),
    "iemocap": (
        "portable/adapters/iemocap/trainable_model.safetensors",
        24_281_730,
        "128042c975ab66b482eb313f20f82ac87f847eed5a226c0c416027a60382c621",
    ),
    "meld": (
        "portable/adapters/meld/trainable_model.safetensors",
        29_811_026,
        "d2e624fd68206b8e128f986b8fe0ff798b96acd10b97aadeeff34059d6ce1312",
    ),
}


def digest(path: Path) -> tuple[int, str]:
    h = sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            size += len(block)
            h.update(block)
    return size, h.hexdigest()


def main() -> None:
    for name, (target_name, expected_size, expected_sha) in WEIGHTS.items():
        target = ROOT / target_name
        if target.exists():
            if digest(target) != (expected_size, expected_sha):
                raise SystemExit(f"Existing file has wrong size/hash: {target}")
            print(f"VERIFIED {target.relative_to(ROOT)}")
            continue
        parts = sorted((ROOT / "weights_parts" / name).glob("part[0-9][0-9]"))
        if not parts:
            raise SystemExit(f"Missing parts for {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".partial")
        with temp.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        output.write(block)
        if digest(temp) != (expected_size, expected_sha):
            temp.unlink()
            raise SystemExit(f"Parts failed size/hash check for {name}")
        temp.replace(target)
        print(f"RESTORED {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
