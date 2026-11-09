import argparse
import ast
import json
import re
from pathlib import Path


TRAITS = ["O", "C", "E", "A", "N"]


def valid_values(values):
    if len(values) != 5:
        return False
    return all(1.0 <= float(v) <= 7.0 for v in values)


def parse_personality_continuous(text):
    raw = str(text or "").strip()
    candidates = [raw]

    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL):
        body = match.group(1).strip()
        if body:
            candidates.append(body)

    for match in re.finditer(r"\[[^\[\]]+\]", raw, flags=re.DOTALL):
        snippet = match.group(0).strip()
        if snippet:
            candidates.append(snippet)

    for cand in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(cand)
                if isinstance(parsed, list) and len(parsed) >= 5:
                    values = [float(parsed[i]) for i in range(5)]
                    if valid_values(values):
                        return dict(zip(TRAITS, values))
            except Exception:
                pass

    trait_values = {}
    for trait in TRAITS:
        match = re.search(
            rf"(?:^|[<>\s,;]){trait}\s*[:=]\s*([+-]?\d+(?:\.\d+)?)",
            raw,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        trait_values[trait] = float(match.group(1))
    values = [trait_values[t] for t in TRAITS]
    if valid_values(values):
        return trait_values
    return None


def load_eval_file(path):
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    report, idx = decoder.raw_decode(text)
    rows = []
    rest = text[idx:].strip()
    if rest:
        rows, _ = decoder.raw_decode(rest)
    return report, rows


def metric_value(report):
    metric = report.get("selection_metric") if isinstance(report, dict) else None
    if isinstance(metric, dict) and metric.get("value") is not None:
        return float(metric["value"])
    pc = report.get("personality_continuous") if isinstance(report, dict) else None
    if isinstance(pc, dict):
        if pc.get("avg_ccc") is not None:
            return 100.0 * float(pc["avg_ccc"])
        if pc.get("avg_mae") is not None:
            return 100.0 - float(pc["avg_mae"])
    return float("-inf")


def parse_personality_sample_id(sample_id, setting):
    sample_id = str(sample_id)
    match = re.match(r"^Personality_([^_]+)_(.+)$", sample_id)
    if not match:
        return None
    sample_setting, key = match.groups()
    if setting != "all" and sample_setting != setting:
        return None
    if sample_setting == "utterance":
        return None
    try:
        conv_id, speaker_idx = key.rsplit("_", 1)
    except ValueError:
        return None
    if not speaker_idx.isdigit():
        return None
    return f"{conv_id}|Speaker_{int(speaker_idx)}"


def coerce_regression_personality(value):
    if isinstance(value, dict) and all(trait in value for trait in TRAITS):
        try:
            vals = [float(value[trait]) for trait in TRAITS]
        except Exception:
            return None
        if valid_values(vals):
            return dict(zip(TRAITS, vals))
    if isinstance(value, (list, tuple)) and len(value) >= 5:
        try:
            vals = [float(value[i]) for i in range(5)]
        except Exception:
            return None
        if valid_values(vals):
            return dict(zip(TRAITS, vals))
    return None


def main():
    parser = argparse.ArgumentParser(description="Export Stage1 personality predictions as a Stage2 pseudo-personality map.")
    parser.add_argument("--stage1_eval_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--setting", default="full", choices=["full", "conversation", "all"])
    parser.add_argument("--fallback", default="medium", choices=["medium", "skip"])
    parser.add_argument("--source", default="auto", choices=["generation", "regression", "auto"])
    args = parser.parse_args()

    eval_dir = Path(args.stage1_eval_dir)
    files = sorted(eval_dir.glob("preds_for_eval_*.text"))
    plain_eval_file = eval_dir / "preds_for_eval.text"
    if plain_eval_file.exists():
        files.append(plain_eval_file)
    if not files:
        raise FileNotFoundError(f"No preds_for_eval_*.text found in {eval_dir}")

    best = None
    best_metric = float("-inf")
    for path in files:
        try:
            report, rows = load_eval_file(path)
            value = metric_value(report)
        except Exception:
            continue
        if value > best_metric:
            best_metric = value
            best = (path, report, rows)

    if best is None:
        raise RuntimeError(f"Could not parse any eval file in {eval_dir}")

    best_path, _, rows = best
    result = {}
    parse_failures = []
    for row in rows:
        key = parse_personality_sample_id(row.get("sample_id"), args.setting)
        if key is None:
            continue
        parsed = None
        if args.source in ("regression", "auto"):
            parsed = coerce_regression_personality(row.get("personality_regression"))
        if parsed is None and args.source in ("generation", "auto"):
            raw = row.get("output_raw") or row.get("output") or ""
            parsed = parse_personality_continuous(raw)
        if parsed is None and args.fallback == "medium":
            parsed = {trait: 4.0 for trait in TRAITS}
            parse_failures.append(row.get("sample_id"))
        elif parsed is None:
            parse_failures.append(row.get("sample_id"))
            continue
        result[key] = {trait: round(float(parsed[trait]), 3) for trait in TRAITS}

    if not result:
        raise RuntimeError(f"No personality predictions exported from {best_path}")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    info = {
        "stage1_eval_dir": str(eval_dir),
        "selected_eval_file": str(best_path),
        "selected_metric": best_metric,
        "setting": args.setting,
        "source": args.source,
        "num_exported": len(result),
        "num_parse_failures": len(parse_failures),
        "parse_failures": parse_failures[:50],
        "output_json": str(output_json),
    }
    output_json.with_suffix(".info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
