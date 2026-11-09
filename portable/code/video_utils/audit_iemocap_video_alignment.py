#!/usr/bin/env python3
"""Audit IEMOCAP metadata, transcripts, labels, and utterance video clips."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path


UTTERANCE_RE = re.compile(
    r"^(?P<dialog>Ses\d\d[FM]_.+)_(?P<speaker>[FM])(?P<number>\d+|XX\d+)$"
)
TRANSCRIPT_RE = re.compile(r"^(?P<utt>\S+)\s*\[[^\]]*\]\s*:\s*(?P<text>.*)$")
EVAL_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<emotion>\S+)\s+\[(?P<vad>[^\]]+)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iemocap-pkl",
        type=Path,
        default=Path("original_data/iemocap/iemocap.pkl"),
    )
    parser.add_argument(
        "--iemocap-root",
        type=Path,
        default=Path("/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/"
            "video_with_audio_min2frames/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "original_data/iemocap/video_clips/"
            "video_with_audio_min2frames/alignment_audit.json"
        ),
    )
    parser.add_argument("--max-error-examples", type=int, default=100)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9'\s]", "", text)


def session_for_dialog(dialog_id: str) -> str:
    return f"Session{int(dialog_id[3:5])}"


def read_transcript(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRANSCRIPT_RE.match(line.strip())
        if match:
            rows.append((match.group("utt"), match.group("text")))
    return rows


def read_evaluations(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EVAL_RE.match(line.strip())
        if not match:
            continue
        rows[match.group("utt")] = {
            "start": float(match.group("start")),
            "end": float(match.group("end")),
            "emotion": match.group("emotion"),
        }
    return rows


def add_error(errors: list[dict], kind: str, **details) -> None:
    errors.append({"kind": kind, **details})


def align_pkl_turns(
    dialog_id: str,
    sentences: list[str],
    speakers: list[str],
    transcript_rows: list[tuple[str, str]],
    errors: list[dict],
) -> list[tuple[int, str, str]]:
    """Align a filtered PKL dialogue to the complete transcript as a subsequence."""
    normalized_transcript = [normalize_text(text) for _, text in transcript_rows]
    aligned = []
    search_start = 0
    for turn, sentence in enumerate(sentences):
        normalized_sentence = normalize_text(sentence)
        expected_speaker = str(speakers[turn]).strip().upper() if turn < len(speakers) else ""
        text_candidates = [
            idx
            for idx in range(search_start, len(transcript_rows))
            if normalized_transcript[idx] == normalized_sentence
        ]
        speaker_candidates = []
        for idx in text_candidates:
            parsed = UTTERANCE_RE.match(transcript_rows[idx][0])
            actual_speaker = parsed.group("speaker") if parsed else ""
            if not expected_speaker or actual_speaker == expected_speaker:
                speaker_candidates.append(idx)

        if speaker_candidates:
            selected = speaker_candidates[0]
            if len(speaker_candidates) > 1:
                add_error(
                    errors,
                    "ambiguous_text_speaker_match",
                    dialog_id=dialog_id,
                    turn=turn,
                    pkl_text=sentence,
                    candidates=[transcript_rows[idx][0] for idx in speaker_candidates],
                    selected=transcript_rows[selected][0],
                )
        elif text_candidates:
            selected = text_candidates[0]
            add_error(
                errors,
                "alignment_speaker_fallback",
                dialog_id=dialog_id,
                turn=turn,
                pkl_text=sentence,
                pkl_speaker=expected_speaker,
                selected=transcript_rows[selected][0],
            )
        else:
            add_error(
                errors,
                "unmatched_pkl_turn",
                dialog_id=dialog_id,
                turn=turn,
                pkl_text=sentence,
                pkl_speaker=expected_speaker,
                search_start=search_start,
            )
            continue

        utt_id, transcript_text = transcript_rows[selected]
        aligned.append((turn, utt_id, transcript_text))
        search_start = selected + 1
    return aligned


def main() -> int:
    args = parse_args()
    data = pickle.load(args.iemocap_pkl.open("rb"))
    speaker_dict = data[0]
    sentence_dict = data[2]
    dialog_ids = list(data[3]) + list(data[4]) + list(data[5])

    manifest_rows = []
    for line_number, line in enumerate(
        args.manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_number"] = line_number
        manifest_rows.append(row)

    errors: list[dict] = []
    manifest_by_utt: dict[str, dict] = {}
    for row in manifest_rows:
        utt_id = str(row.get("utterance_id") or row.get("sample_id") or "")
        if not utt_id:
            add_error(errors, "manifest_missing_utterance_id", line=row["_line_number"])
            continue
        if utt_id in manifest_by_utt:
            add_error(
                errors,
                "manifest_duplicate_utterance_id",
                utterance_id=utt_id,
                lines=[manifest_by_utt[utt_id]["_line_number"], row["_line_number"]],
            )
            continue
        manifest_by_utt[utt_id] = row

    mapped_turns = {}
    transcript_ids = set()
    evaluation_ids = set()
    pkl_turn_count = 0

    for dialog_id in dialog_ids:
        session = session_for_dialog(dialog_id)
        transcript_path = (
            args.iemocap_root
            / session
            / "dialog"
            / "transcriptions"
            / f"{dialog_id}.txt"
        )
        eval_path = (
            args.iemocap_root
            / session
            / "dialog"
            / "EmoEvaluation"
            / f"{dialog_id}.txt"
        )
        if not transcript_path.is_file():
            add_error(errors, "missing_transcript", dialog_id=dialog_id)
            continue
        if not eval_path.is_file():
            add_error(errors, "missing_evaluation", dialog_id=dialog_id)
            continue

        transcript_rows = read_transcript(transcript_path)
        evaluations = read_evaluations(eval_path)
        transcript_ids.update(utt_id for utt_id, _ in transcript_rows)
        evaluation_ids.update(evaluations)

        sentences = sentence_dict.get(dialog_id, [])
        speakers = speaker_dict.get(dialog_id, [])
        pkl_turn_count += len(sentences)
        if len(sentences) != len(speakers):
            add_error(
                errors,
                "pkl_sentence_speaker_count_mismatch",
                dialog_id=dialog_id,
                sentences=len(sentences),
                speakers=len(speakers),
            )

        aligned_rows = align_pkl_turns(
            dialog_id=dialog_id,
            sentences=sentences,
            speakers=speakers,
            transcript_rows=transcript_rows,
            errors=errors,
        )
        for turn, utt_id, transcript_text in aligned_rows:
            sentence = sentences[turn]
            turn_key = f"{dialog_id}_{turn}"
            mapped_turns[turn_key] = utt_id

            parsed = UTTERANCE_RE.match(utt_id)
            if parsed is None or parsed.group("dialog") != dialog_id:
                add_error(
                    errors,
                    "utterance_dialog_mismatch",
                    dialog_id=dialog_id,
                    turn=turn,
                    utterance_id=utt_id,
                )
            expected_speaker = str(speakers[turn]).strip().upper() if turn < len(speakers) else ""
            actual_speaker = parsed.group("speaker") if parsed else ""
            if expected_speaker and actual_speaker != expected_speaker:
                add_error(
                    errors,
                    "speaker_mismatch",
                    dialog_id=dialog_id,
                    turn=turn,
                    utterance_id=utt_id,
                    pkl_speaker=expected_speaker,
                    utterance_speaker=actual_speaker,
                )
            if normalize_text(sentence) != normalize_text(transcript_text):
                add_error(
                    errors,
                    "text_mismatch",
                    dialog_id=dialog_id,
                    turn=turn,
                    utterance_id=utt_id,
                    pkl_text=sentence,
                    transcript_text=transcript_text,
                )

            row = manifest_by_utt.get(utt_id)
            if row is None:
                add_error(
                    errors,
                    "mapped_turn_missing_from_manifest",
                    dialog_id=dialog_id,
                    turn=turn,
                    utterance_id=utt_id,
                    has_evaluation=utt_id in evaluations,
                    emotion=evaluations.get(utt_id, {}).get("emotion"),
                )
                continue

            if str(row.get("dialog_id")) != dialog_id:
                add_error(
                    errors,
                    "manifest_dialog_mismatch",
                    utterance_id=utt_id,
                    expected=dialog_id,
                    actual=row.get("dialog_id"),
                )
            if str(row.get("session")) != session:
                add_error(
                    errors,
                    "manifest_session_mismatch",
                    utterance_id=utt_id,
                    expected=session,
                    actual=row.get("session"),
                )

            video_path = Path(str(row.get("video_path", "")))
            if not video_path.is_file():
                add_error(
                    errors,
                    "manifest_video_missing",
                    utterance_id=utt_id,
                    video_path=str(video_path),
                )
            elif video_path.stem != utt_id:
                add_error(
                    errors,
                    "video_basename_mismatch",
                    utterance_id=utt_id,
                    video_path=str(video_path),
                )

            evaluation = evaluations.get(utt_id)
            if evaluation is None:
                if not row.get("alignment_completion_clip"):
                    add_error(errors, "manifest_utterance_missing_evaluation", utterance_id=utt_id)
            else:
                for field in ("start", "end"):
                    actual = float(row.get(field, -1))
                    expected = float(evaluation[field])
                    if abs(actual - expected) > 1e-4:
                        add_error(
                            errors,
                            f"manifest_{field}_mismatch",
                            utterance_id=utt_id,
                            expected=expected,
                            actual=actual,
                        )
                if str(row.get("emotion")) != evaluation["emotion"]:
                    add_error(
                        errors,
                        "manifest_emotion_mismatch",
                        utterance_id=utt_id,
                        expected=evaluation["emotion"],
                        actual=row.get("emotion"),
                    )

    mapped_utterance_ids = set(mapped_turns.values())
    for utt_id, row in manifest_by_utt.items():
        parsed = UTTERANCE_RE.match(utt_id)
        if parsed is None:
            add_error(errors, "invalid_manifest_utterance_id", utterance_id=utt_id)
            continue
        if parsed.group("dialog") != str(row.get("dialog_id")):
            add_error(
                errors,
                "manifest_id_dialog_disagreement",
                utterance_id=utt_id,
                dialog_id=row.get("dialog_id"),
            )
        if utt_id not in transcript_ids:
            add_error(errors, "manifest_id_not_in_transcripts", utterance_id=utt_id)
        if utt_id not in evaluation_ids and not row.get("alignment_completion_clip"):
            add_error(errors, "manifest_id_not_in_evaluations", utterance_id=utt_id)

    error_counts = Counter(error["kind"] for error in errors)
    warning_kinds = {"ambiguous_text_speaker_match"}
    fatal_kinds = {kind for kind in error_counts if kind not in warning_kinds}
    report = {
        "status": "pass" if not fatal_kinds else "fail",
        "iemocap_pkl": str(args.iemocap_pkl.resolve()),
        "iemocap_root": str(args.iemocap_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "dialog_count": len(dialog_ids),
        "pkl_turn_count": pkl_turn_count,
        "mapped_turn_count": len(mapped_turns),
        "mapped_unique_utterance_count": len(mapped_utterance_ids),
        "transcript_utterance_count": len(transcript_ids),
        "evaluation_utterance_count": len(evaluation_ids),
        "manifest_row_count": len(manifest_rows),
        "manifest_unique_utterance_count": len(manifest_by_utt),
        "mapped_turns_with_video": len(mapped_utterance_ids & set(manifest_by_utt)),
        "mapped_turns_without_video": len(mapped_utterance_ids - set(manifest_by_utt)),
        "error_counts": dict(sorted(error_counts.items())),
        "warning_kinds": sorted(kind for kind in error_counts if kind in warning_kinds),
        "fatal_error_kinds": sorted(fatal_kinds),
        "error_examples": errors[: args.max_error_examples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "error_examples"}, indent=2))
    print(f"report: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
