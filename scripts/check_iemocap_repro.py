"""Check archived IEMOCAP split order and public PA-IEMOCAP GT mapping."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "repro/iemocap_split_index.json"
OCEAN = ROOT / "portable/datasets/PA-IEMOCAP/ocean.csv"
OCEAN_URL = "https://raw.githubusercontent.com/Kyoto-University-Speech-and-Audio/PA-IEMOCAP/main/ocean.csv"
OCEAN_SHA = "4772bdc9f013a3d6c4e8b038e559c5633092ce68e5edb3fed0aef99e4f9689ec"


def sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized(dialogue: str) -> str:
    return re.sub(r"(script)(\d)_", r"\g<1>0\2_", dialogue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-ocean", action="store_true")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "portable",
                        help="Portable directory holding the licensed prepared data and audio cache")
    parser.add_argument("--check-cache-hash", action="store_true", help="Read the 1.7 GB mel cache")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    if args.download_ocean and not OCEAN.is_file():
        OCEAN.parent.mkdir(parents=True, exist_ok=True)
        data = urlopen(OCEAN_URL, timeout=30).read()
        if sha256(data).hexdigest() != OCEAN_SHA:
            raise SystemExit("Official PA-IEMOCAP file changed; inspect it before use")
        OCEAN.write_bytes(data)
    if not OCEAN.is_file() or sha(OCEAN) != OCEAN_SHA:
        raise SystemExit("Missing or changed official ocean.csv; use --download-ocean")
    if not INDEX.is_file():
        raise SystemExit("Official ocean.csv verified. Local split index missing; restore authorized old data, then run scripts/make_iemocap_split_index.py --source-root portable")
    index = json.loads(INDEX.read_text())
    ocean = {}
    with OCEAN.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ocean[(normalized(row["dialogue_id"]), row["gender"].upper())] = {
                "O": float(row["OP"]), "C": float(row["CO"]),
                "E": float(row["EX"]), "A": float(row["AG"]), "N": float(row["NE"]),
            }
    missing = []
    for split, expected in index["splits"].items():
        path = asset_root / "data/iemocap" / f"{split}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != len(expected):
            raise SystemExit(f"{split}: expected {len(expected)} records, found {len(rows)}")
        for row, (sample_id, utterance_id, gender) in zip(rows, expected):
            if row["sample_id"] != sample_id or row["utterance_id"] != utterance_id:
                raise SystemExit(f"{split}: sample/utterance order mismatch at {sample_id}")
            dialogue = sample_id.rsplit("_", 1)[0]
            gt = ocean.get((normalized(dialogue), gender))
            if gt is None or any(abs(gt[t] - float(row["gt_personality"][t])) > 0.001 for t in "OCEAN"):
                raise SystemExit(f"{split}: GT OCEAN mismatch at {sample_id}")
        expected_hash = index["source_hashes"][f"{split}.json"]
        exact = sha(path) == expected_hash
        print(f"{split}: {len(rows)} records, IDs/order/GT match; original file hash match: {exact}")
    source_pkl = asset_root / "original_data/iemocap/iemocap.pkl"
    if source_pkl.is_file():
        print("iemocap.pkl original hash match:", sha(source_pkl) == index["source_hashes"]["iemocap.pkl"])
    else:
        missing.append(str(source_pkl))
    cache = asset_root / "original_data/iemocap/iemocap_qwen_native_mel_features.pkl"
    if cache.is_file():
        print("native mel cache bytes:", cache.stat().st_size)
        if args.check_cache_hash:
            print("native mel cache original hash match:", sha(cache) == "ca9edbf84d82ade227cb08dded54b1f34d1927882455e383dfb41d856ce75c2e")
    else:
        missing.append(str(cache))
    if missing:
        print("Missing licensed local assets:", *missing, sep="\n  ")
        if args.require_data:
            raise SystemExit(1)
    else:
        print("IEMOCAP reproduction assets available; proceed with a real-batch smoke.")


if __name__ == "__main__":
    main()
