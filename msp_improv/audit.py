"""Read-only MSP-IMPROV alignment and dialogue-reconstruction audit.

The archive identifies scene, speaker and take, but does not document a
globally ordered turn index. This audit deliberately does not guess one.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import re


NAME = re.compile(r"MSP-IMPROV-(S\d{2}[A-Z])-([FM])(\d+)-([PRST])-([FM]{2})(\d+)\.wav$")
LABELS = {"A": "angry", "H": "happy", "S": "sad", "N": "neutral"}


def audit(root: Path) -> dict:
    audio = root / "Audio"
    transcripts = root / "All_human_transcriptions"
    evaluation = root / "Evaluation.txt"
    labels = {}
    for line in evaluation.read_text(errors="replace").splitlines():
        if ".avi;" not in line:
            continue
        head, code, *_ = (part.strip() for part in line.split(";"))
        key = head.removesuffix(".avi").replace("UTD-IMPROV-", "MSP-IMPROV-", 1)
        if key in labels:
            raise ValueError(f"Duplicate evaluation ID: {key}")
        labels[key] = code

    wavs = list(audio.glob("session*/*/*/*.wav"))
    groups = defaultdict(list)
    errors = []
    sessions = Counter()
    scenarios = Counter()
    for wav in wavs:
        match = NAME.fullmatch(wav.name)
        if not match:
            errors.append(wav.name)
            continue
        scene, gender, actor, scenario, pair, turn = match.groups()
        session = wav.parts[-4]
        if wav.parent.name != scenario or wav.parent.parent.name != scene:
            errors.append(wav.name)
        sessions[session] += 1
        scenarios[scenario] += 1
        # Candidate grouping only. The filename alone does not prove chronology.
        groups[(session, scene, scenario, pair, turn)].append((gender, actor, wav.stem))

    stems = {p.stem for p in wavs}
    text_stems = {p.stem for p in transcripts.glob("*.txt")}
    pair_sizes = Counter(len(v) for v in groups.values())
    codes = Counter(labels.values())
    return {
        "wav": len(wavs), "transcripts": len(text_stems), "evaluations": len(labels),
        "missing_transcript": len(stems - text_stems),
        "missing_evaluation": len(stems - set(labels)),
        "evaluation_without_audio": len(set(labels) - stems),
        "filename_errors": errors[:20],
        "sessions": dict(sorted(sessions.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "emotion_codes": dict(sorted(codes.items())),
        "four_class_count": sum(codes[k] for k in LABELS),
        "candidate_group_sizes": dict(sorted(pair_sizes.items())),
        "candidate_group_example": [list(key) + [v] for key, v in list(groups.items())[:3]],
        "chronological_order_verified": False,
        "stage1_ready": False,
        "reason": "No verified within-scene chronology in the supplied filename/README metadata.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "dataset/MSP-IMPROV")
    parser.add_argument("--output", type=Path, help="Write aligned utterance JSONL inside this project")
    args = parser.parse_args()
    report = audit(args.dataset)
    print(json.dumps(report, indent=2))
    if args.output:
        if any(report[k] for k in ("missing_transcript", "missing_evaluation", "evaluation_without_audio", "filename_errors")):
            raise SystemExit("Alignment audit failed; manifest not written")
        root = Path(__file__).resolve().parent
        out = args.output.resolve()
        if not out.is_relative_to(root):
            raise SystemExit("Output must stay inside msp_improv/")
        labels = {}
        for line in (args.dataset / "Evaluation.txt").read_text(errors="replace").splitlines():
            if ".avi;" in line:
                name, code, *_ = (part.strip() for part in line.split(";"))
                labels[name.removesuffix(".avi").replace("UTD-IMPROV-", "MSP-IMPROV-", 1)] = code
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as stream:
            for wav in sorted((args.dataset / "Audio").glob("session*/*/*/*.wav")):
                match = NAME.fullmatch(wav.name)
                scene, gender, actor, scenario, pair, turn = match.groups()
                session = wav.parts[-4]
                code = labels[wav.stem]
                row = {
                    "id": wav.stem, "session": session,
                    "split": "test" if session == "session6" else "valid" if session == "session5" else "train",
                    "scene": scene, "speaker": f"{session}_{gender}{actor}",
                    "scenario": scenario, "take": f"{pair}{turn}",
                    "text": (args.dataset / "All_human_transcriptions" / (wav.stem + ".txt")).read_text(errors="replace").strip(),
                    "audio_path": str(wav.resolve()), "label_code": code,
                    "emotion": LABELS.get(code), "eligible_four_class": code in LABELS,
                    "dialogue_id": None, "turn_index": None,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {report['wav']} aligned utterances to {out}")


if __name__ == "__main__":
    main()
