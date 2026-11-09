import argparse
import json
import os
from pathlib import Path


def iter_jsonl(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Build an eval-only personality dataset covering all unique Stage1 personality samples."
    )
    parser.add_argument("--stage1_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--setting", default="full", choices=["full", "conversation", "utterance", "all"])
    args = parser.parse_args()

    stage1_data_dir = Path(args.stage1_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []
    prefix = None if args.setting == "all" else f"Personality_{args.setting}_"
    for split in ("train", "test", "valid"):
        for item in iter_jsonl(stage1_data_dir / f"{split}.json"):
            sample_id = str(item.get("sample_id", ""))
            if not sample_id.startswith("Personality_"):
                continue
            if prefix is not None and not sample_id.startswith(prefix):
                continue
            if sample_id in seen:
                continue
            seen.add(sample_id)
            rows.append(item)

    out_test = output_dir / "test.json"
    with out_test.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Keep train/valid present for tooling that expects a data directory shape.
    for split in ("train", "valid"):
        (output_dir / f"{split}.json").write_text("", encoding="utf-8")

    manifest = {
        "stage1_data_dir": str(stage1_data_dir),
        "setting": args.setting,
        "num_unique_personality_samples": len(rows),
        "test_json": str(out_test),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    main()
