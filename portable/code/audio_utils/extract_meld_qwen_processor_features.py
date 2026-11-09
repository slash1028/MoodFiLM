#!/usr/bin/env python3
"""Extract MELD utterance mel features for Qwen2.5-Omni audio inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor


SPLIT_CONFIG = {
    "train": ("train_splits/train_sent_emo.csv", "train_splits/train_splits"),
    "valid": ("dev_sent_emo.csv", "dev_splits/dev_splits_complete"),
    "test": ("test_sent_emo.csv", "test_splits/output_repeated_splits_test"),
}
DEFAULT_MELD_RAW_ROOT = Path("/n/work6/yiwang/MoodFilm/portable/datasets/MELD.Raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld-raw-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")),
        help="Local directory for staging pickle writes before copying to NFS.",
    )
    return parser.parse_args()


def read_meld_rows(root: Path) -> List[dict]:
    root = root.expanduser()
    if not root.is_absolute():
        root = root.resolve()
    if not (root / "dev_sent_emo.csv").is_file() and DEFAULT_MELD_RAW_ROOT.is_dir():
        print(
            f"[WARN] MELD raw root {root} does not look valid; "
            f"falling back to {DEFAULT_MELD_RAW_ROOT}",
            flush=True,
        )
        root = DEFAULT_MELD_RAW_ROOT
    rows: List[dict] = []
    for split, (csv_name, media_dir) in SPLIT_CONFIG.items():
        csv_path = root / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                dialogue_id = int(row["Dialogue_ID"])
                utterance_id = int(row["Utterance_ID"])
                media_sample_id = f"dia{dialogue_id}_utt{utterance_id}"
                sample_id = f"{split}_{media_sample_id}"
                media_path = root / media_dir / f"{media_sample_id}.mp4"
                rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "dialogue_id": dialogue_id,
                        "utterance_id": utterance_id,
                        "speaker": row.get("Speaker", ""),
                        "utterance": row.get("Utterance", ""),
                        "emotion": row.get("Emotion", ""),
                        "path": str(media_path),
                    }
                )
    rows.sort(key=lambda item: (item["split"], item["dialogue_id"], item["utterance_id"]))
    return rows


def decode_mp4_audio(path: str, target_sr: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {stderr}")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        audio = np.zeros(target_sr // 10, dtype=np.float32)
    return audio


def prepare_waveform(processor, path: str, target_sr: int) -> np.ndarray:
    waveform = torch.from_numpy(decode_mp4_audio(path, target_sr)).to(torch.float32)
    hop_length = int(getattr(processor.feature_extractor, "hop_length", 160))
    max_model_samples = int(getattr(processor.feature_extractor, "n_samples", waveform.numel()))
    if waveform.numel() > max_model_samples:
        waveform = waveform[:max_model_samples]
    padded_samples = int(math.ceil(float(waveform.numel()) / float(max(1, hop_length))) * hop_length)
    padded_samples = max(hop_length, min(padded_samples, max_model_samples))
    if waveform.numel() < padded_samples:
        waveform = torch.cat(
            [waveform, torch.zeros(padded_samples - waveform.numel(), dtype=waveform.dtype)],
            dim=0,
        )
    return waveform.numpy()


def extract_batch(processor, paths: List[str], target_sr: int) -> List[Tuple[np.ndarray, int]]:
    waveforms = [prepare_waveform(processor, path, target_sr) for path in paths]
    proc_out = processor.feature_extractor(
        waveforms,
        sampling_rate=target_sr,
        return_attention_mask=True,
        padding="longest",
        return_tensors="np",
    )
    input_features = proc_out["input_features"].astype(np.float32)
    attention_mask = proc_out.get("attention_mask", None)
    results: List[Tuple[np.ndarray, int]] = []
    for idx in range(input_features.shape[0]):
        if attention_mask is None:
            feature_length = int(input_features.shape[-1])
        else:
            feature_length = int(attention_mask[idx].sum())
        feature_length = max(1, min(feature_length, int(input_features.shape[-1])))
        results.append((np.ascontiguousarray(input_features[idx, :, :feature_length]), feature_length))
    return results


def save_partial(path: Path, payload: dict, tmp_dir: Path, retries: int = 3) -> None:
    """Write pickle robustly when the output path is on NFS.

    Directly writing large pickle files to ``path.tmp`` can fail with stale file
    handles on the cluster filesystem. Stage the pickle on local storage first,
    then copy the completed file to the destination directory and atomically
    replace the target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest_tmp = path.with_suffix(path.suffix + ".tmp")
    last_error = None
    for attempt in range(1, retries + 1):
        local_name = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{path.name}.",
                suffix=".tmp",
                dir=str(tmp_dir),
                delete=False,
            ) as handle:
                local_name = handle.name
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            shutil.copyfile(local_name, dest_tmp)
            os.replace(dest_tmp, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if dest_tmp.exists():
                    dest_tmp.unlink()
            except OSError:
                pass
            time.sleep(min(10, 2 * attempt))
        finally:
            if local_name is not None:
                try:
                    os.unlink(local_name)
                except OSError:
                    pass
    raise OSError(f"Failed to save {path} after {retries} attempts") from last_error


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output_pkl.is_file() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output_pkl}")
    rows = read_meld_rows(args.meld_raw_root)
    if args.limit > 0:
        rows = rows[: args.limit]

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_pkl.with_suffix(args.output_pkl.suffix + ".partial")
    sample_features: Dict[str, np.ndarray] = {}
    sample_lengths: Dict[str, int] = {}
    missing: List[dict] = []
    manifest: List[dict] = []
    if partial_path.is_file() and not args.overwrite:
        partial = pickle.load(partial_path.open("rb"))
        sample_features.update(partial.get("sample_features") or {})
        sample_lengths.update(partial.get("sample_feature_lengths") or {})
        missing.extend(partial.get("missing") or [])

    pending = [row for row in rows if row["sample_id"] not in sample_features]
    processed_since_save = 0
    for start in tqdm(range(0, len(pending), args.batch_size), desc="Extract MELD Qwen mel"):
        batch = pending[start : start + args.batch_size]
        valid_rows = [row for row in batch if os.path.exists(row["path"])]
        for row in batch:
            if not os.path.exists(row["path"]):
                missing.append({"sample_id": row["sample_id"], "path": row["path"], "reason": "missing_mp4"})
        if valid_rows:
            try:
                extracted = extract_batch(processor, [row["path"] for row in valid_rows], target_sr)
            except Exception:
                extracted = []
                for row in valid_rows:
                    try:
                        one = extract_batch(processor, [row["path"]], target_sr)[0]
                    except Exception as exc:
                        missing.append({"sample_id": row["sample_id"], "path": row["path"], "reason": str(exc)})
                        continue
                    sample_features[row["sample_id"]] = one[0]
                    sample_lengths[row["sample_id"]] = one[1]
            for row, (features, length) in zip(valid_rows, extracted):
                sample_features[row["sample_id"]] = features
                sample_lengths[row["sample_id"]] = length
        processed_since_save += len(batch)
        if args.save_every > 0 and processed_since_save >= args.save_every:
            save_partial(
                partial_path,
                {
                    "schema": "qwen_native_mel_partial_v1",
                    "sample_features": sample_features,
                    "sample_feature_lengths": sample_lengths,
                    "missing": missing,
                },
                args.tmp_dir,
            )
            processed_since_save = 0

    by_dialogue: Dict[str, List[str]] = {}
    for row in rows:
        by_dialogue.setdefault(f"{row['split']}_dia{row['dialogue_id']}", [])
        by_dialogue[f"{row['split']}_dia{row['dialogue_id']}"].append(row["sample_id"])
        manifest.append({key: row[key] for key in ("split", "sample_id", "dialogue_id", "utterance_id", "speaker", "emotion", "path")})

    payload = {
        "schema": "qwen_native_mel_v1",
        "dataset": "meld",
        "model_path": str(args.model_path.resolve()),
        "meld_raw_root": str(args.meld_raw_root.resolve()),
        "sampling_rate": target_sr,
        "feature_dim": int(next(iter(sample_features.values())).shape[0]) if sample_features else 0,
        "num_rows": len(rows),
        "num_sample_features": len(sample_features),
        "num_missing_audio": len(missing),
        "sample_features": dict(sorted(sample_features.items())),
        "sample_feature_lengths": dict(sorted(sample_lengths.items())),
        "conv_turn_to_uttid": {row["sample_id"]: row["sample_id"] for row in rows},
        "dialogue_features": by_dialogue,
        "missing_sample_ids": [item["sample_id"] for item in missing],
        "missing_audio": missing,
        "manifest": manifest,
    }
    save_partial(args.output_pkl, payload, args.tmp_dir)
    manifest_path = args.output_pkl.with_suffix(".manifest.jsonl")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("sample_features", "sample_feature_lengths", "manifest")}, indent=2, ensure_ascii=False))
    print(f"saved: {args.output_pkl}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
