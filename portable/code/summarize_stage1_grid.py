#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


def read_report(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    report, _ = decoder.raw_decode(text)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    rows = []
    for output in sorted(root.glob("stage1_*")):
        if not output.is_dir():
            continue
        for path in sorted(output.glob("preds_for_eval_*.text")):
            try:
                report = read_report(path)
            except Exception:
                continue
            cont = report.get("personality_continuous") or {}
            spk = report.get("speaker") or {}
            pair = ((report.get("personality_pair_from_continuous") or {}).get("pairwise") or {})
            rows.append({
                "setting": output.name.replace("stage1_", ""),
                "epoch": int(path.stem.rsplit("_", 1)[-1]),
                "gen_ccc": cont.get("avg_ccc"),
                "gen_mae": cont.get("avg_mae"),
                "gen_mse": cont.get("avg_mse"),
                "speaker_acc": spk.get("acc"),
                "derived_pairwise_acc": pair.get("accuracy"),
                "derived_pairwise_macro_f1": pair.get("macro_f1"),
                "path": str(path),
            })

    fields = [
        "setting",
        "epoch",
        "gen_ccc",
        "gen_mae",
        "gen_mse",
        "speaker_acc",
        "derived_pairwise_acc",
        "derived_pairwise_macro_f1",
        "path",
    ]
    all_path = root / "stage1_grid_all_epochs.tsv"
    with all_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    best = {}
    for row in rows:
        ccc = row.get("gen_ccc")
        if ccc is None:
            continue
        setting = row["setting"]
        if setting not in best or float(ccc) > float(best[setting]["gen_ccc"]):
            best[setting] = row
    best_path = root / "stage1_grid_best_gen_ccc.tsv"
    with best_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(best.values(), key=lambda x: str(x["setting"])):
            writer.writerow(row)

    print(f"[OK] wrote {all_path}")
    print(f"[OK] wrote {best_path}")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
