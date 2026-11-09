import argparse
import math
import os
import pickle
import re
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor


def load_wav_mono(path: str) -> Tuple[torch.Tensor, int]:
    """Load PCM WAV without relying on torchaudio backends like torchcodec."""
    with wave.open(path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)

    waveform = torch.from_numpy(data)
    return waveform, sample_rate


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9'\s]", "", s)
    return s


def parse_transcript_lines(transcript_path: str) -> List[Tuple[str, str]]:
    utts: List[Tuple[str, str]] = []
    if not os.path.exists(transcript_path):
        return utts
    with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\S+)\s*\[[^\]]*\]\s*:\s*(.*)$", line)
            if m:
                utt_id = m.group(1)
                text = m.group(2)
                utts.append((utt_id, text))
    return utts


def dialogue_to_session(conv_id: str) -> Optional[str]:
    m = re.match(r"^Ses(\d\d)", conv_id)
    if m is None:
        return None
    return f"Session{int(m.group(1))}"


def build_conv_turn_to_uttid_map(iemocap_pkl: str, iemocap_root: str):
    data = pickle.load(open(iemocap_pkl, "rb"))
    sentence_dict = data[2]
    all_ids = data[3] + data[4] + data[5]

    mapping = {}
    for conv_id in tqdm(all_ids, desc="Align transcripts"):
        session = dialogue_to_session(conv_id)
        if session is None:
            continue

        transcript_path = os.path.join(iemocap_root, session, "dialog", "transcriptions", f"{conv_id}.txt")
        transcript_utts = parse_transcript_lines(transcript_path)
        if len(transcript_utts) == 0:
            continue

        sent_list = sentence_dict[conv_id]
        utt_ids: List[Optional[str]] = []

        if len(transcript_utts) == len(sent_list):
            utt_ids = [u[0] for u in transcript_utts]
        else:
            norm_trans = [normalize_text(t) for _, t in transcript_utts]
            used = set()
            for s in sent_list:
                ns = normalize_text(s)
                found_idx = None
                for i, nt in enumerate(norm_trans):
                    if i in used:
                        continue
                    if nt == ns:
                        found_idx = i
                        break
                if found_idx is None:
                    utt_ids.append(None)
                else:
                    used.add(found_idx)
                    utt_ids.append(transcript_utts[found_idx][0])

        for turn, utt_id in enumerate(utt_ids):
            mapping[f"{conv_id}_{turn}"] = utt_id

    return mapping, data


def _load_and_prepare_qwen_waveform(processor, wav_path: str, target_sr: int) -> np.ndarray:
    waveform, sr = load_wav_mono(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.to(torch.float32)
    if waveform.numel() == 0:
        waveform = torch.zeros(target_sr // 10, dtype=torch.float32)

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


def extract_qwen_native_mel_features(
    processor,
    wav_paths: List[str],
    target_sr: int,
) -> List[Tuple[np.ndarray, int]]:
    if len(wav_paths) == 0:
        return []

    waveforms = [
        _load_and_prepare_qwen_waveform(processor=processor, wav_path=wav_path, target_sr=target_sr)
        for wav_path in wav_paths
    ]

    proc_out = processor.feature_extractor(
        waveforms,
        sampling_rate=target_sr,
        return_attention_mask=True,
        padding="longest",
        return_tensors="np",
    )
    input_features = proc_out["input_features"].astype(np.float32)  # [batch, n_mels, frames]
    attention_mask = proc_out.get("attention_mask", None)

    results = []
    for idx in range(input_features.shape[0]):
        if attention_mask is None:
            feature_length = int(input_features.shape[-1])
        else:
            feature_length = int(attention_mask[idx].sum())
        feature_length = max(1, min(feature_length, int(input_features.shape[-1])))
        results.append((np.ascontiguousarray(input_features[idx, :, :feature_length]), feature_length))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract IEMOCAP utterance mel features for Qwen2.5-Omni native audio inputs."
    )
    parser.add_argument(
        "--iemocap_pkl",
        type=str,
        default="/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/iemocap.pkl",
    )
    parser.add_argument("--iemocap_root", type=str, default="/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release")
    parser.add_argument(
        "--qwen_processor_path",
        type=str,
        default="/n/work6/yiwang/MoodFilm/portable/LLM_bases/Qwen2.5-Omni-7B",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/iemocap_qwen_native_mel_features.pkl",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    conv_turn_to_uttid, data = build_conv_turn_to_uttid_map(args.iemocap_pkl, args.iemocap_root)

    processor = AutoProcessor.from_pretrained(args.qwen_processor_path, trust_remote_code=True)
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))

    sentence_dict = data[2]
    all_ids = data[3] + data[4] + data[5]

    sample_features: Dict[str, np.ndarray] = {}
    sample_feature_lengths: Dict[str, int] = {}
    dialogue_features: Dict[str, List[Optional[str]]] = {}
    missing = []
    total_feature_frames = 0

    for conv_id in tqdm(all_ids, desc="Extract Qwen native mel features"):
        session = dialogue_to_session(conv_id)
        if session is None:
            continue

        seq_refs: List[Optional[str]] = [None] * len(sentence_dict[conv_id])
        valid_turns: List[int] = []
        valid_sample_ids: List[str] = []
        valid_wav_paths: List[str] = []
        for turn in range(len(sentence_dict[conv_id])):
            sample_id = f"{conv_id}_{turn}"
            utt_id = conv_turn_to_uttid.get(sample_id, None)
            if utt_id is None:
                missing.append(sample_id)
                continue

            wav_path = os.path.join(args.iemocap_root, session, "sentences", "wav", conv_id, f"{utt_id}.wav")
            if not os.path.exists(wav_path):
                missing.append(sample_id)
                continue

            valid_turns.append(turn)
            valid_sample_ids.append(sample_id)
            valid_wav_paths.append(wav_path)

        extracted = extract_qwen_native_mel_features(
            processor=processor,
            wav_paths=valid_wav_paths,
            target_sr=target_sr,
        )
        for turn, sample_id, (frame_feat, feature_length) in zip(valid_turns, valid_sample_ids, extracted):
            sample_features[sample_id] = frame_feat
            sample_feature_lengths[sample_id] = feature_length
            total_feature_frames += feature_length
            seq_refs[turn] = sample_id

        dialogue_features[conv_id] = seq_refs

    feature_dim = 0
    if len(sample_features) > 0:
        feature_dim = int(next(iter(sample_features.values())).shape[0])

    payload = {
        "schema": "qwen_native_mel_v1",
        "source": "qwen2_5_omni_processor_whisper_feature_extractor_native_mel",
        "qwen_processor_path": args.qwen_processor_path,
        "sampling_rate": target_sr,
        "hop_length": int(getattr(processor.feature_extractor, "hop_length", 160)),
        "feature_dim": feature_dim,
        "feature_dtype": "float32",
        "sample_features": sample_features,
        "sample_feature_lengths": sample_feature_lengths,
        "dialogue_features": dialogue_features,
        "conv_turn_to_uttid": conv_turn_to_uttid,
        "missing_sample_ids": missing,
        "num_sample_features": len(sample_features),
        "total_feature_frames": total_feature_frames,
    }

    with open(args.output_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {args.output_pkl}")
    print(f"feature_dim: {feature_dim}")
    print(f"feature_dtype: float32")
    print(f"num_sample_features: {len(sample_features)}")
    print(f"num_dialogue_features: {len(dialogue_features)}")
    print(f"total_feature_frames: {total_feature_frames}")
    print(f"missing_sample_ids: {len(missing)}")


if __name__ == "__main__":
    main()
