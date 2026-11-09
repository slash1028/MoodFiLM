#!/usr/bin/env python3
"""Jupyter UI and persistence helpers for IEMOCAP speaker-side annotation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "iemocap_dialog_speaker_side_manual_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_frame_index(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "iemocap_speaker_side_annotation_frames_v1":
        raise ValueError(f"Unexpected frame index schema: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Frame index contains no entries: {path}")
    return payload


def new_annotation_payload(frame_index: Path, entries: list[dict]) -> dict:
    return {
        "schema": SCHEMA,
        "frame_index": str(frame_index.resolve()),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "dialog_count": len(entries),
        "dialog_speaker_sides": {},
        "details": {},
    }


def load_or_create_annotations(
    output_path: Path,
    frame_index: Path,
    entries: list[dict],
) -> dict:
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"Unexpected annotation schema: {output_path}")
        return payload
    payload = new_annotation_payload(frame_index, entries)
    atomic_write_json(output_path, payload)
    return payload


def record_annotation(
    payload: dict,
    output_path: Path,
    entry: dict,
    left_gender: str,
) -> None:
    left_gender = left_gender.upper()
    dialog_id = str(entry["dialog_id"])
    if left_gender == "M":
        mapping = {"F": "right", "M": "left"}
        status = "confirmed"
    elif left_gender == "F":
        mapping = {"F": "left", "M": "right"}
        status = "confirmed"
    elif left_gender == "U":
        mapping = None
        status = "uncertain"
    else:
        raise ValueError(f"left_gender must be M, F, or U, got {left_gender!r}")

    if mapping is None:
        payload["dialog_speaker_sides"].pop(dialog_id, None)
    else:
        payload["dialog_speaker_sides"][dialog_id] = mapping
    payload["details"][dialog_id] = {
        "left_gender": left_gender,
        "right_gender": (
            {"M": "F", "F": "M"}.get(left_gender)
        ),
        "status": status,
        "contact_sheet": str(entry["contact_sheet"]),
        "frame_timestamps": list(entry.get("frame_timestamps") or []),
        "updated_at": utc_now(),
    }
    payload["updated_at"] = utc_now()
    atomic_write_json(output_path, payload)


def annotation_counts(entries: list[dict], payload: dict) -> dict[str, int]:
    details = payload.get("details") or {}
    confirmed = sum(
        1
        for entry in entries
        if (details.get(str(entry["dialog_id"])) or {}).get("status") == "confirmed"
    )
    uncertain = sum(
        1
        for entry in entries
        if (details.get(str(entry["dialog_id"])) or {}).get("status") == "uncertain"
    )
    return {
        "total": len(entries),
        "confirmed": confirmed,
        "uncertain": uncertain,
        "remaining": len(entries) - confirmed - uncertain,
    }


def first_unfinished_index(entries: list[dict], payload: dict) -> int:
    details = payload.get("details") or {}
    for index, entry in enumerate(entries):
        if str(entry["dialog_id"]) not in details:
            return index
    for index, entry in enumerate(entries):
        detail = details.get(str(entry["dialog_id"])) or {}
        if detail.get("status") == "uncertain":
            return index
    return max(0, len(entries) - 1)


def launch_annotation_ui(
    frame_index: str | Path,
    output_path: str | Path,
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "This UI requires ipywidgets and IPython. In the Jupyter kernel run: "
            "%pip install ipywidgets"
        ) from exc

    frame_index = Path(frame_index)
    output_path = Path(output_path)
    frame_payload = load_frame_index(frame_index)
    entries = frame_payload["entries"]
    payload = load_or_create_annotations(output_path, frame_index, entries)
    state = {"index": first_unfinished_index(entries, payload)}

    image_widget = widgets.Image(
        format="jpg",
        layout=widgets.Layout(width="100%", max_width="1600px"),
    )
    status = widgets.HTML()
    message = widgets.HTML()
    previous_button = widgets.Button(description="上一条")
    male_button = widgets.Button(
        description="左边是男", button_style="primary", icon="male"
    )
    female_button = widgets.Button(
        description="左边是女", button_style="success", icon="female"
    )
    uncertain_button = widgets.Button(
        description="不确定", button_style="warning", icon="question"
    )

    def render(note: str = "") -> None:
        entry = entries[state["index"]]
        counts = annotation_counts(entries, payload)
        detail = (payload.get("details") or {}).get(str(entry["dialog_id"])) or {}
        current = detail.get("left_gender", "未标注")
        status.value = (
            f"<b>{state['index'] + 1}/{len(entries)} "
            f"{entry['dialog_id']}</b><br>"
            f"已确认 {counts['confirmed']}，不确定 {counts['uncertain']}，"
            f"未处理 {counts['remaining']}；当前记录：{current}"
        )
        message.value = note
        image_path = Path(entry["contact_sheet"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Contact sheet does not exist: {image_path}")
        image_widget.value = image_path.read_bytes()

    def annotate(left_gender: str) -> None:
        entry = entries[state["index"]]
        record_annotation(payload, output_path, entry, left_gender)
        note = (
            f"<span style='color:green'>已保存 {entry['dialog_id']} "
            f"到 {output_path}</span>"
        )
        if state["index"] < len(entries) - 1:
            state["index"] += 1
        render(note)

    def previous(_button) -> None:
        if state["index"] > 0:
            state["index"] -= 1
        render()

    male_button.on_click(lambda _button: annotate("M"))
    female_button.on_click(lambda _button: annotate("F"))
    uncertain_button.on_click(lambda _button: annotate("U"))
    previous_button.on_click(previous)

    controls = widgets.HBox(
        [previous_button, male_button, female_button, uncertain_button]
    )
    ui = widgets.VBox([status, image_widget, controls, message])
    render()
    display(ui)
    return ui
