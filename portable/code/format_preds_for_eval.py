#!/usr/bin/env python3
"""Pretty-print streamed preds_for_eval files without changing their content."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def split_report_and_records(text: str) -> tuple[dict | None, list]:
    if "\n\n" in text:
        report_text, records_text = text.split("\n\n", 1)
        try:
            report = json.loads(report_text)
            records = json.loads(records_text)
            if isinstance(report, dict) and isinstance(records, list):
                return report, records
        except json.JSONDecodeError:
            pass

    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list, optionally preceded by a JSON report.")
    return None, records


def format_file(path: Path) -> None:
    report, records = split_report_and_records(path.read_text(encoding="utf-8"))
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        if report is not None:
            handle.write(json.dumps(report, indent=2, ensure_ascii=False))
            handle.write("\n\n")
        handle.write("[\n")
        for index, record in enumerate(records):
            if index:
                handle.write(",\n")
            record_text = json.dumps(record, indent=5, ensure_ascii=False)
            handle.write("     " + record_text.replace("\n", "\n     "))
        handle.write("\n]\n")
    os.replace(temp_path, path)
    print(f"formatted: {path} ({len(records)} records)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        format_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
