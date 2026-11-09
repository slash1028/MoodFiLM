"""Extract a text/audio-free index from the archived licensed IEMOCAP files."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Authorized source portable directory with data/iemocap and original_data/iemocap")
    parser.add_argument("--output", type=Path, default=ROOT / "repro/iemocap_split_index.json")
    args = parser.parse_args()
    source = args.source_root.resolve()
    original = source / "original_data/iemocap/iemocap.pkl"
    with original.open("rb") as stream:
        data = pickle.load(stream)
    speaker_by_dialogue = data[0]
    result = {
        "schema": "iemocap_split_index_v1",
        "note": "IDs, speaker tags, order and hashes only; no transcript, emotion label or audio data",
        "source_hashes": {"iemocap.pkl": file_sha(original)},
        "dialogue_splits": {"train": data[3], "test": data[4], "valid": data[5]},
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        path = source / "data/iemocap" / f"{split}.json"
        entries = []
        seen = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row["sample_id"]
                dialogue, turn = sample_id.rsplit("_", 1)
                gender = speaker_by_dialogue[dialogue][int(turn)]
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample ID: {sample_id}")
                seen.add(sample_id)
                entries.append([sample_id, row["utterance_id"], gender])
        result["splits"][split] = entries
        result["source_hashes"][f"{split}.json"] = file_sha(path)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("Index output must remain inside this project")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(output, {split: len(rows) for split, rows in result["splits"].items()})


if __name__ == "__main__":
    main()
