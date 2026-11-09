#!/usr/bin/env python
"""Export speaker mood dynamics case-study tables and heatmaps."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


def _json_docs(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    docs: list[Any] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        docs.append(obj)
        idx = end
    return docs


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for path in paths:
        for doc in _json_docs(path):
            if isinstance(doc, list):
                for item in doc:
                    if isinstance(item, dict):
                        item = dict(item)
                        item["_source_file"] = str(path)
                        records.append(item)
            elif isinstance(doc, dict):
                if "emotion" in doc or "task_counts" in doc:
                    item = dict(doc)
                    item["_source_file"] = str(path)
                    metrics.append(item)
                elif "records" in doc and isinstance(doc["records"], list):
                    for item in doc["records"]:
                        if isinstance(item, dict):
                            item = dict(item)
                            item["_source_file"] = str(path)
                            records.append(item)
    return records, metrics


def load_data_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load JSON/JSONL data rows and index them by sample/utterance id."""
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        rows: list[Any]
        try:
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in (row.get("sample_id"), row.get("utterance_id"), row.get("id")):
                if key:
                    lookup[str(key)] = row
    return lookup


def attach_data_metadata(records: list[dict[str, Any]], lookup: dict[str, dict[str, Any]]) -> None:
    for record in records:
        meta = None
        for key in (record.get("sample_id"), record.get("utterance_id"), record.get("id")):
            if key and str(key) in lookup:
                meta = lookup[str(key)]
                break
        if not meta:
            continue
        record["_data_meta"] = meta
        speaker_id_to_name: dict[str, str] = {}
        speaker_name_to_ids: dict[str, list[str]] = {}
        turn_ids = meta.get("audio_turn_sample_ids") or []
        for turn_id in turn_ids:
            turn = lookup.get(str(turn_id))
            if not isinstance(turn, dict):
                continue
            spk_id = str(turn.get("speaker_id") or "")
            spk_name = str(turn.get("speaker_name") or "")
            if not spk_id or not spk_name:
                continue
            speaker_id_to_name[spk_id] = spk_name
            speaker_name_to_ids.setdefault(spk_name, [])
            if spk_id not in speaker_name_to_ids[spk_name]:
                speaker_name_to_ids[spk_name].append(spk_id)
        spk_id = str(meta.get("speaker_id") or "")
        spk_name = str(meta.get("speaker_name") or "")
        if spk_id and spk_name:
            speaker_id_to_name[spk_id] = spk_name
            speaker_name_to_ids.setdefault(spk_name, [])
            if spk_id not in speaker_name_to_ids[spk_name]:
                speaker_name_to_ids[spk_name].append(spk_id)
        record["_speaker_id_to_name"] = speaker_id_to_name
        record["_speaker_name_to_ids"] = speaker_name_to_ids


UTT_RE = re.compile(r'(Speaker_\d+):"([^"]*)"')
TARGET_RE = re.compile(r'<\s*(Speaker_\d+):"([^"]*)"\s*>')


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "joyful": "joy",
        "happy": "hap",
        "happiness": "hap",
        "excited": "exc",
        "frustrated": "fru",
        "neutral": "neu",
        "angry": "ang",
        "sad": "sad",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
    }
    return aliases.get(text, text)


def truncate(text: Any, limit: int = 120) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def extract_utterances(prompt: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in UTT_RE.finditer(prompt or "")]


def extract_target_from_prompt(prompt: str) -> tuple[str | None, str | None]:
    match = TARGET_RE.search(prompt or "")
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _candidate_prompt_speakers(record: dict[str, Any], speaker: str, target_speaker: str | None) -> list[str]:
    speaker = str(speaker or "")
    if not speaker:
        return []
    if speaker.startswith("Speaker_"):
        return [speaker]
    utterance_speakers = [spk for spk, _ in extract_utterances(record.get("input", ""))]
    if speaker == "interlocutor":
        return [spk for spk in utterance_speakers if spk != target_speaker]
    name_to_ids = record.get("_speaker_name_to_ids") or {}
    ids = name_to_ids.get(speaker) or []
    if ids:
        return list(ids)
    meta = record.get("_data_meta") or {}
    context_speakers = meta.get("speaker_context_speakers") or []
    if len(context_speakers) == 1 and str(context_speakers[0]) == speaker:
        return [spk for spk in utterance_speakers if spk != target_speaker]
    return []


def last_context_utterance(record: dict[str, Any], speaker: str) -> str:
    utterances = extract_utterances(record.get("input", ""))
    target_speaker, target_text = extract_target_from_prompt(record.get("input", ""))
    dyn = record.get("speaker_mood_dynamics") or {}
    if not target_speaker and str(dyn.get("target_speaker") or "").startswith("Speaker_"):
        target_speaker = str(dyn.get("target_speaker"))
    candidate_speakers = set(_candidate_prompt_speakers(record, speaker, target_speaker))
    cutoff = len(utterances)
    if target_speaker and target_text:
        for idx, (spk, utt) in enumerate(utterances):
            if spk == target_speaker and utt.strip() == target_text.strip():
                cutoff = idx
                break
    for spk, utt in reversed(utterances[:cutoff]):
        if spk in candidate_speakers:
            return utt
    for spk, utt in reversed(utterances):
        if spk in candidate_speakers:
            return utt
    return ""


def dialogue_window(record: dict[str, Any]) -> list[dict[str, Any]]:
    utterances = extract_utterances(record.get("input", ""))
    meta = record.get("_data_meta") or {}
    max_turns = len(meta.get("audio_turn_sample_ids") or [])
    if max_turns <= 0:
        target_speaker, target_text = extract_target_from_prompt(record.get("input", ""))
        max_turns = len(utterances)
        if target_speaker and target_text:
            for idx, (spk, utt) in enumerate(utterances):
                if spk == target_speaker and utt.strip() == target_text.strip():
                    max_turns = idx + 1
                    break
    id_to_name = record.get("_speaker_id_to_name") or {}
    target_speaker, target_text = extract_target_from_prompt(record.get("input", ""))
    rows: list[dict[str, Any]] = []
    for spk, utt in utterances[:max_turns]:
        rows.append(
            {
                "speaker_id": spk,
                "speaker_name": id_to_name.get(spk, spk),
                "utterance": utt,
                "is_target": bool(target_speaker and target_text and spk == target_speaker and utt.strip() == target_text.strip()),
            }
        )
    return rows


def get_gold(record: dict[str, Any]) -> str:
    return normalize_label(record.get("gold_label") or record.get("target"))


def get_pred(record: dict[str, Any]) -> str:
    return normalize_label(record.get("pred_label") or record.get("output"))


def sorted_attention(dyn: dict[str, Any]) -> list[dict[str, Any]]:
    rows = dyn.get("target_to_context_attention") or []
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            weight = float(row.get("weight", 0.0))
        except Exception:
            weight = 0.0
        cleaned.append({"speaker": str(row.get("speaker", "")), "weight": weight})
    return sorted(cleaned, key=lambda item: item["weight"], reverse=True)


def entropy(weights: list[float]) -> float:
    vals = [max(0.0, float(v)) for v in weights]
    total = sum(vals)
    if total <= 0:
        return 0.0
    probs = [v / total for v in vals if v > 0]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def emotion_score(mapping: dict[str, Any], label: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    if label in mapping:
        return float(mapping[label])
    for key, value in mapping.items():
        if normalize_label(key) == label:
            return float(value)
    return None


def _softmax_from_values(values: list[float | None]) -> list[float | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return [None for _ in values]
    max_value = max(usable)
    exps: list[float | None] = [
        math.exp(float(value) - max_value) if value is not None else None
        for value in values
    ]
    denom = sum(value for value in exps if value is not None)
    if denom <= 0:
        return [None for _ in values]
    return [value / denom if value is not None else None for value in exps]


def _label_values_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    base_logits = row.get("_base_logits") or {}
    final_logits = row.get("_final_logits") or {}
    final_delta = row.get("_final_delta") or {}
    labels = list(base_logits.keys()) or list(final_logits.keys()) or list(final_delta.keys())
    before_probs = _softmax_from_values([base_logits.get(label) for label in labels])
    after_probs = _softmax_from_values([final_logits.get(label) for label in labels])
    values: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        before = base_logits.get(label)
        after = final_logits.get(label)
        requested_delta = final_delta.get(label)
        effective_delta = None
        if before is not None and after is not None:
            effective_delta = float(after) - float(before)
        norm_label = normalize_label(label)
        roles = []
        if norm_label == row.get("gold"):
            roles.append("gold")
        if norm_label == row.get("pred_before_prior"):
            roles.append("before")
        if norm_label == row.get("pred_after_prior"):
            roles.append("after")
        values.append(
            {
                "label": str(label),
                "norm_label": norm_label,
                "roles": roles,
                "before": float(before) if before is not None else None,
                "before_prob": float(before_probs[idx]) if before_probs[idx] is not None else None,
                "requested_delta": float(requested_delta) if requested_delta is not None else None,
                "after": float(after) if after is not None else None,
                "after_prob": float(after_probs[idx]) if after_probs[idx] is not None else None,
                "prob_delta": (
                    float(after_probs[idx]) - float(before_probs[idx])
                    if before_probs[idx] is not None and after_probs[idx] is not None
                    else None
                ),
                "effective_delta": effective_delta,
            }
        )
    return values


def _top_margin(values: list[dict[str, Any]], field: str) -> tuple[str, float, str, float, float] | None:
    usable = [item for item in values if item.get(field) is not None]
    if not usable:
        return None
    ranked = sorted(usable, key=lambda item: float(item[field]), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {"label": "", field: float("nan")}
    margin = float(top[field]) - float(second[field]) if len(ranked) > 1 else float("nan")
    return str(top["label"]), float(top[field]), str(second["label"]), float(second[field]), margin


def flatten_record(record: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    dyn = record.get("speaker_mood_dynamics")
    if not isinstance(dyn, dict):
        return None
    gold = get_gold(record)
    pred = get_pred(record)
    pred_before = normalize_label(dyn.get("pred_before_prior"))
    pred_after = normalize_label(dyn.get("pred_after_prior"))
    attn = sorted_attention(dyn)
    top = attn[0] if attn else {"speaker": "", "weight": 0.0}
    second = attn[1] if len(attn) > 1 else {"speaker": "", "weight": 0.0}
    top_weight = float(top["weight"])
    second_weight = float(second["weight"])
    focus_gap = top_weight - second_weight
    focus_ratio = top_weight / max(second_weight, 1e-6) if second_weight > 0 else (999.0 if top_weight > 0 else 0.0)
    target_speaker = str(dyn.get("target_speaker") or "")
    target_speaker_prompt, target_text_prompt = extract_target_from_prompt(record.get("input", ""))
    if not target_speaker:
        target_speaker = target_speaker_prompt or ""
    target_utt = record.get("debug_target_utterance") or target_text_prompt or ""
    transition_prob = dyn.get("transition_prob") or {}
    transition_prior = dyn.get("transition_prior") or {}
    final_delta = dyn.get("final_logits_delta") or {}
    base_logits = dyn.get("base_label_logits") or {}
    final_logits = dyn.get("final_label_logits") or {}
    row = {
        "dataset": dataset,
        "index": record.get("index"),
        "sample_id": record.get("sample_id"),
        "utterance_id": record.get("utterance_id"),
        "source_file": record.get("_source_file"),
        "target_speaker": target_speaker,
        "target_utterance": target_utt,
        "gold": gold,
        "output": pred,
        "pred_before_prior": pred_before,
        "pred_after_prior": pred_after,
        "correct": pred == gold,
        "prior_flip_to_gold": pred_before != gold and pred_after == gold,
        "prior_changed_prediction": pred_before != pred_after,
        "mood_shift_index": float(dyn.get("mood_shift_index", 0.0) or 0.0),
        "top_attention_speaker": top["speaker"],
        "top_attention_weight": top_weight,
        "second_attention_speaker": second["speaker"],
        "second_attention_weight": second_weight,
        "attention_focus_gap": focus_gap,
        "attention_focus_ratio": focus_ratio,
        "attention_entropy": entropy([item["weight"] for item in attn]),
        "top_context_utterance": last_context_utterance(record, top["speaker"]),
        "second_context_utterance": last_context_utterance(record, second["speaker"]),
        "dialogue_window": json.dumps(dialogue_window(record), ensure_ascii=False),
        "gold_transition_prob": emotion_score(transition_prob, gold),
        "gold_transition_prior": emotion_score(transition_prior, gold),
        "gold_final_logits_delta": emotion_score(final_delta, gold),
        "gold_base_logit": emotion_score(base_logits, gold),
        "gold_final_logit": emotion_score(final_logits, gold),
        "_record": record,
        "_attention": attn,
        "_transition_prob": transition_prob,
        "_final_delta": final_delta,
        "_base_logits": base_logits,
        "_final_logits": final_logits,
    }
    label_values = _label_values_for_row(row)
    before_margin = _top_margin(label_values, "before")
    after_margin = _top_margin(label_values, "after")
    if before_margin:
        row.update(
            {
                "before_top_label": before_margin[0],
                "before_top_logit": before_margin[1],
                "before_second_label": before_margin[2],
                "before_second_logit": before_margin[3],
                "before_top_margin": before_margin[4],
            }
        )
    if after_margin:
        row.update(
            {
                "after_top_label": after_margin[0],
                "after_top_logit": after_margin[1],
                "after_second_label": after_margin[2],
                "after_second_logit": after_margin[3],
                "after_top_margin": after_margin[4],
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_case(row: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attn = row.get("_attention") or []
    attn = sorted(attn, key=lambda item: float(item.get("weight", 0.0)), reverse=True)[:8]
    speakers = [str(item["speaker"]) for item in attn]
    weights = [float(item["weight"]) for item in attn]
    snippets = [truncate(last_context_utterance(row["_record"], spk), 58) for spk in speakers]
    speaker_labels = [f"{spk}: {snip}" if snip else spk for spk, snip in zip(speakers, snippets)]

    label_values = _label_values_for_row(row)
    emotions = [item["label"] for item in label_values]
    before_logits = [item["before"] if item["before"] is not None else 0.0 for item in label_values]
    after_logits = [item["after"] if item["after"] is not None else 0.0 for item in label_values]
    before_probs = [item["before_prob"] if item["before_prob"] is not None else 0.0 for item in label_values]
    after_probs = [item["after_prob"] if item["after_prob"] is not None else 0.0 for item in label_values]
    deltas = [item["requested_delta"] if item["requested_delta"] is not None else 0.0 for item in label_values]

    fig = plt.figure(figsize=(11.5, 8.6))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.2, 1.0], hspace=0.62)
    ax0 = fig.add_subplot(grid[0])
    if weights:
        y_pos = list(range(len(weights)))[::-1]
        ax0.barh(y_pos, weights, color="#4c78a8")
        ax0.set_yticks(y_pos)
        ax0.set_yticklabels(speaker_labels, fontsize=8)
        ax0.set_xlim(0, max(1.0, max(weights) * 1.08))
        ax0.set_xlabel("attention weight")
        for y, weight in zip(y_pos, weights):
            ax0.text(weight + 0.01, y, f"{weight:.3f}", va="center", fontsize=8)
    else:
        ax0.text(0.5, 0.5, "No context attention saved", ha="center", va="center")
        ax0.set_axis_off()
    ax0.set_title("Target-aware context speaker attention")

    ax1 = fig.add_subplot(grid[1])
    if label_values:
        x = list(range(len(emotions)))
        width = 0.36
        before_colors = ["#d55e00" if item["norm_label"] == row["pred_before_prior"] else "#9ecae9" for item in label_values]
        after_colors = ["#009e73" if item["norm_label"] == row["pred_after_prior"] else "#bdbdbd" for item in label_values]
        ax1.bar([v - width / 2 for v in x], before_probs, width=width, color=before_colors, label="before prior")
        ax1.bar([v + width / 2 for v in x], after_probs, width=width, color=after_colors, label="after prior")
        ax1.set_ylim(0.0, min(1.0, max(before_probs + after_probs + [0.01]) * 1.20))
        ax1.set_xticks(x)
        ax1.set_xticklabels(emotions, rotation=20, ha="right")
        ax1.set_ylabel("probability")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_title("Before/after emotion-label probabilities")
    else:
        ax1.text(0.5, 0.5, "No label logits saved", ha="center", va="center")
        ax1.set_axis_off()

    ax2 = fig.add_subplot(grid[2])
    if label_values:
        colors = ["#009e73" if normalize_label(e) == row["gold"] else "#999999" for e in emotions]
        ax2.bar(emotions, deltas, color=colors)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("logit delta")
        ax2.set_title("Mood transition prior contribution to label logits")
    else:
        ax2.set_axis_off()

    title = (
        f"{row['dataset']} idx={row['index']} gold={row['gold']} output={row['output']} "
        f"before={row['pred_before_prior']} after={row['pred_after_prior']} "
        f"shift={row['mood_shift_index']:.3f}"
    )
    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_attention_matrix(rows: list[dict[str, Any]], out_path: Path, max_rank: int = 6) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix: list[list[float]] = []
    ylabels: list[str] = []
    annotations: list[list[str]] = []
    for row in rows:
        attn = sorted(row.get("_attention") or [], key=lambda item: float(item.get("weight", 0.0)), reverse=True)
        attn = attn[:max_rank]
        weights = [float(item.get("weight", 0.0)) for item in attn]
        speakers = [str(item.get("speaker") or "") for item in attn]
        if len(weights) < max_rank:
            weights += [0.0] * (max_rank - len(weights))
            speakers += [""] * (max_rank - len(speakers))
        matrix.append(weights)
        annotations.append(speakers)
        ylabels.append(f"{row.get('case_id')} idx{row.get('index')} {row.get('gold')}")

    fig_h = max(4.0, min(18.0, 0.32 * len(matrix) + 1.8))
    fig, ax = plt.subplots(figsize=(9.0, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(list(range(max_rank)))
    ax.set_xticklabels([f"rank {idx + 1}" for idx in range(max_rank)])
    ax.set_yticks(list(range(len(ylabels))))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title("Target-aware attention over context speakers")
    for row_idx, speaker_row in enumerate(annotations):
        for col_idx, speaker in enumerate(speaker_row):
            if speaker:
                ax.text(col_idx, row_idx, truncate(speaker, 12), ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="attention weight")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def case_markdown(row: dict[str, Any], image_rel: str) -> str:
    top_utt = truncate(row.get("top_context_utterance"), 260)
    second_utt = truncate(row.get("second_context_utterance"), 220)
    target_utt = truncate(row.get("target_utterance"), 260)
    label_values = _label_values_for_row(row)
    before_margin = _top_margin(label_values, "before")
    after_margin = _top_margin(label_values, "after")
    fields = [
        f"# Case {row.get('case_id')}: {row['dataset']} index={row['index']}",
        "",
        f"![attention heatmap]({image_rel})",
        "",
        "## Prediction",
        "",
        f"- Gold: `{row['gold']}`",
        f"- Output: `{row['output']}`",
        f"- Before prior: `{row['pred_before_prior']}`",
        f"- After prior: `{row['pred_after_prior']}`",
        f"- Mood shift index: `{row['mood_shift_index']:.6f}`",
        f"- Gold logit delta: `{row.get('gold_final_logits_delta')}`",
    ]
    if before_margin:
        fields.append(
            f"- Before top margin: `{before_margin[0]}` {before_margin[1]:.6f} vs "
            f"`{before_margin[2]}` {before_margin[3]:.6f} (margin {before_margin[4]:+.6f})"
        )
    if after_margin:
        fields.append(
            f"- After top margin: `{after_margin[0]}` {after_margin[1]:.6f} vs "
            f"`{after_margin[2]}` {after_margin[3]:.6f} (margin {after_margin[4]:+.6f})"
        )
    if label_values:
        fields.extend(
            [
                "",
                "## Label Probabilities and First-Token Logits",
                "",
                "Probabilities are softmax-normalized over the candidate emotion labels using their first-token logits.",
                "The logits are kept as diagnostics for the transition-prior injection.",
                "",
                "| Label | Role | Before prob | After prob | Prob delta | Before first-token logit | Requested logit delta | Effective logit change | After first-token logit |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in label_values:
            before_prob_text = f"{item['before_prob']:.4f}" if item["before_prob"] is not None else ""
            after_prob_text = f"{item['after_prob']:.4f}" if item["after_prob"] is not None else ""
            prob_delta_text = f"{item['prob_delta']:+.4f}" if item["prob_delta"] is not None else ""
            before_text = f"{item['before']:.6f}" if item["before"] is not None else ""
            delta_text = f"{item['requested_delta']:+.6f}" if item["requested_delta"] is not None else ""
            effective_delta_text = f"{item['effective_delta']:+.6f}" if item["effective_delta"] is not None else ""
            after_text = f"{item['after']:.6f}" if item["after"] is not None else ""
            fields.append(
                f"| `{item['label']}` | {', '.join(item['roles'])} | {before_prob_text} | {after_prob_text} | {prob_delta_text} | {before_text} | {delta_text} | {effective_delta_text} | {after_text} |"
            )
    fields.extend(
        [
        "",
        "## Dialogue Evidence",
        "",
        f"- Target speaker: `{row.get('target_speaker')}`",
        f"- Target utterance: {target_utt}",
        f"- Top-attended speaker: `{row.get('top_attention_speaker')}` weight `{row.get('top_attention_weight'):.6f}`",
        f"- Top context utterance: {top_utt}",
        f"- Second speaker: `{row.get('second_attention_speaker')}` weight `{row.get('second_attention_weight'):.6f}`",
        f"- Second context utterance: {second_utt}",
        ]
    )
    try:
        window = json.loads(row.get("dialogue_window") or "[]")
    except Exception:
        window = []
    if window:
        fields.extend(["", "## Dialogue Window", ""])
        for item in window:
            marker = " **[target]**" if item.get("is_target") else ""
            speaker = item.get("speaker_name") or item.get("speaker_id")
            fields.append(f"- `{speaker}`: {truncate(item.get('utterance'), 220)}{marker}")
    return "\n".join(fields) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Prediction files or directories.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-json", nargs="*", default=[], help="Optional source JSON/JSONL files for speaker-id/name mapping.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--attention-threshold", type=float, default=0.40)
    parser.add_argument("--strong-prior-margin", type=float, default=0.125)
    args = parser.parse_args()

    inputs: list[Path] = []
    for item in args.input:
        path = Path(item)
        if path.is_dir():
            inputs.extend(sorted(path.glob("preds_for_eval*.text")))
        else:
            inputs.append(path)
    inputs = [path for path in inputs if path.exists()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, metrics = load_records(inputs)
    data_lookup = load_data_rows([Path(item) for item in args.data_json])
    if data_lookup:
        attach_data_metadata(records, data_lookup)
    rows = [flatten_record(record, args.dataset) for record in records]
    rows = [row for row in rows if row is not None]

    csv_fields = [
        "dataset",
        "case_id",
        "category",
        "index",
        "sample_id",
        "utterance_id",
        "target_speaker",
        "gold",
        "output",
        "pred_before_prior",
        "pred_after_prior",
        "prior_flip_to_gold",
        "prior_changed_prediction",
        "mood_shift_index",
        "top_attention_speaker",
        "top_attention_weight",
        "second_attention_speaker",
        "second_attention_weight",
        "attention_focus_gap",
        "attention_focus_ratio",
        "attention_entropy",
        "gold_transition_prob",
        "gold_final_logits_delta",
        "gold_base_logit",
        "gold_final_logit",
        "before_top_label",
        "before_top_logit",
        "before_second_label",
        "before_second_logit",
        "before_top_margin",
        "after_top_label",
        "after_top_logit",
        "after_second_label",
        "after_second_logit",
        "after_top_margin",
        "target_utterance",
        "top_context_utterance",
        "second_context_utterance",
        "dialogue_window",
        "source_file",
    ]

    prior_flip = sorted(
        [row for row in rows if row["prior_flip_to_gold"]],
        key=lambda r: (
            float(r.get("top_attention_weight") or 0.0),
            float(r.get("mood_shift_index") or 0.0),
            float(r.get("gold_final_logits_delta") or 0.0),
        ),
        reverse=True,
    )
    strong_prior_flip = [
        row
        for row in prior_flip
        if float(row.get("before_top_margin") or 0.0) >= float(args.strong_prior_margin)
    ]
    tie_prior_flip = [
        row
        for row in prior_flip
        if abs(float(row.get("before_top_margin") or 0.0)) < 1e-9
    ]
    weak_prior_flip = [
        row
        for row in prior_flip
        if row not in strong_prior_flip and row not in tie_prior_flip
    ]
    attention_focus = sorted(
        [
            row
            for row in rows
            if float(row.get("top_attention_weight") or 0.0) >= args.attention_threshold
        ],
        key=lambda r: (
            float(r.get("top_attention_weight") or 0.0),
            float(r.get("attention_focus_gap") or 0.0),
            float(r.get("mood_shift_index") or 0.0),
        ),
        reverse=True,
    )
    multi_speaker_attention = sorted(
        [
            row
            for row in rows
            if row.get("correct")
            and float(row.get("top_attention_weight") or 0.0) >= args.attention_threshold
            and float(row.get("top_attention_weight") or 0.0) < 0.98
            and float(row.get("second_attention_weight") or 0.0) > 0.02
            and str(row.get("target_utterance") or "").strip()
            and str(row.get("top_context_utterance") or "").strip()
        ],
        key=lambda r: (
            float(r.get("attention_focus_gap") or 0.0),
            float(r.get("top_attention_weight") or 0.0),
            float(r.get("mood_shift_index") or 0.0),
        ),
        reverse=True,
    )
    balanced_multi_speaker_attention = sorted(
        [
            row
            for row in rows
            if row.get("correct")
            and 0.55 <= float(row.get("top_attention_weight") or 0.0) <= 0.90
            and float(row.get("second_attention_weight") or 0.0) >= 0.08
            and str(row.get("target_utterance") or "").strip()
            and str(row.get("top_context_utterance") or "").strip()
        ],
        key=lambda r: (
            float(r.get("mood_shift_index") or 0.0),
            float(r.get("attention_focus_gap") or 0.0),
            float(r.get("top_attention_weight") or 0.0),
        ),
        reverse=True,
    )
    mood_shift = sorted(rows, key=lambda r: float(r.get("mood_shift_index") or 0.0), reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for category, group in [
        ("strong_prior_flip", strong_prior_flip[: args.top_k]),
        ("weak_prior_flip", weak_prior_flip[: args.top_k]),
        ("tie_prior_flip", tie_prior_flip[: args.top_k]),
        ("balanced_multi_speaker_attention", balanced_multi_speaker_attention[: args.top_k]),
        ("multi_speaker_attention", multi_speaker_attention[: args.top_k]),
        ("attention_focus", attention_focus[: args.top_k]),
        ("mood_shift", mood_shift[: args.top_k]),
    ]:
        for row in group:
            key = (str(row.get("source_file")), row.get("index"))
            if key in seen:
                continue
            row = dict(row)
            row["category"] = category
            row["case_id"] = f"{len(selected) + 1:03d}"
            selected.append(row)
            seen.add(key)

    write_csv(out_dir / "all_speaker_mood_cases.csv", [{**row, "category": ""} for row in rows], csv_fields)
    write_csv(out_dir / "prior_flip_cases.csv", [dict(row, category="prior_flip", case_id=f"P{idx+1:03d}") for idx, row in enumerate(prior_flip)], csv_fields)
    write_csv(out_dir / "strong_prior_flip_cases.csv", [dict(row, category="strong_prior_flip", case_id=f"SP{idx+1:03d}") for idx, row in enumerate(strong_prior_flip)], csv_fields)
    write_csv(out_dir / "weak_prior_flip_cases.csv", [dict(row, category="weak_prior_flip", case_id=f"WP{idx+1:03d}") for idx, row in enumerate(weak_prior_flip)], csv_fields)
    write_csv(out_dir / "tie_prior_flip_cases.csv", [dict(row, category="tie_prior_flip", case_id=f"TP{idx+1:03d}") for idx, row in enumerate(tie_prior_flip)], csv_fields)
    write_csv(out_dir / "balanced_multi_speaker_attention_cases.csv", [dict(row, category="balanced_multi_speaker_attention", case_id=f"BMS{idx+1:03d}") for idx, row in enumerate(balanced_multi_speaker_attention)], csv_fields)
    write_csv(out_dir / "multi_speaker_attention_cases.csv", [dict(row, category="multi_speaker_attention", case_id=f"MS{idx+1:03d}") for idx, row in enumerate(multi_speaker_attention)], csv_fields)
    write_csv(out_dir / "attention_focus_cases.csv", [dict(row, category="attention_focus", case_id=f"A{idx+1:03d}") for idx, row in enumerate(attention_focus)], csv_fields)
    write_csv(out_dir / "mood_shift_cases.csv", [dict(row, category="mood_shift", case_id=f"M{idx+1:03d}") for idx, row in enumerate(mood_shift)], csv_fields)
    write_csv(out_dir / "selected_cases.csv", selected, csv_fields)

    serializable = []
    for row in selected:
        item = {field: row.get(field) for field in csv_fields}
        item["attention"] = row.get("_attention")
        item["transition_prob"] = row.get("_transition_prob")
        item["final_logits_delta"] = row.get("_final_delta")
        item["base_label_logits"] = row.get("_base_logits")
        item["final_label_logits"] = row.get("_final_logits")
        serializable.append(item)
    write_json(out_dir / "selected_cases.json", serializable)
    write_json(out_dir / "metrics.json", metrics)

    heatmap_dir = out_dir / "heatmaps"
    case_dir = out_dir / "case_cards"
    for directory in (heatmap_dir, case_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    for row in selected:
        png_name = f"case_{row['case_id']}_{row['category']}_idx{row['index']}.png"
        plot_case(row, heatmap_dir / png_name)
        md = case_markdown(row, f"../heatmaps/{png_name}")
        (case_dir / f"case_{row['case_id']}_{row['category']}_idx{row['index']}.md").parent.mkdir(parents=True, exist_ok=True)
        (case_dir / f"case_{row['case_id']}_{row['category']}_idx{row['index']}.md").write_text(md, encoding="utf-8")
    plot_attention_matrix(selected, out_dir / "selected_attention_matrix.png")

    summary = [
        f"# Speaker Mood Case Study Summary: {args.dataset}",
        "",
        f"- Input files: {len(inputs)}",
        f"- Records with speaker mood dynamics: {len(rows)}",
        f"- Prior-flip-to-gold cases: {len(prior_flip)}",
        f"- Strong prior-flip-to-gold cases (before margin >= {args.strong_prior_margin:g}): {len(strong_prior_flip)}",
        f"- Weak prior-flip-to-gold cases: {len(weak_prior_flip)}",
        f"- Tie prior-flip-to-gold cases: {len(tie_prior_flip)}",
        f"- Balanced multi-speaker attention cases: {len(balanced_multi_speaker_attention)}",
        f"- Multi-speaker attention cases: {len(multi_speaker_attention)}",
        f"- Attention-focused cases (top weight >= {args.attention_threshold}): {len(attention_focus)}",
        f"- Selected cases: {len(selected)}",
        "",
        "## Recommended First Cases",
        "",
    ]
    for row in selected[: min(12, len(selected))]:
        summary.extend(
            [
                f"### Case {row['case_id']} ({row['category']})",
                f"- idx={row['index']}, sample_id={row.get('sample_id')}, gold={row['gold']}, output={row['output']}",
                f"- before={row['pred_before_prior']}, after={row['pred_after_prior']}, shift={row['mood_shift_index']:.4f}",
                f"- top speaker={row['top_attention_speaker']} weight={row['top_attention_weight']:.4f}",
                f"- target: {truncate(row.get('target_utterance'), 180)}",
                f"- top context: {truncate(row.get('top_context_utterance'), 180)}",
                "",
            ]
        )
    (out_dir / "case_study_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"[OK] wrote case-study artifacts to {out_dir}")
    print(f"[OK] records={len(rows)} prior_flip={len(prior_flip)} attention_focus={len(attention_focus)} selected={len(selected)}")


if __name__ == "__main__":
    main()
