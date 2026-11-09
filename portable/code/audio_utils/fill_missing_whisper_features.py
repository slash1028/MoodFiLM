import argparse
import os
import pickle
import re

import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor


def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9'\s]", "", s)
    return s


def parse_transcript_lines(transcript_path):
    utts = []
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


def dialogue_to_session(conv_id):
    m = re.match(r"^Ses(\d\d)", conv_id)
    if m is None:
        return None
    return f"Session{int(m.group(1))}"


def extract_whisper_feature(model, processor, wav_path, device):
    waveform, sr = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    input_features = inputs.input_features.to(device=device, dtype=model_dtype)

    with torch.no_grad():
        enc = model.encoder(input_features)
        feat = enc.last_hidden_state.mean(dim=1).squeeze(0).float().cpu()
    return feat


def pick_same_text_candidate(conv_id, turn, text, iemocap_root):
    session = dialogue_to_session(conv_id)
    if session is None:
        return None, None
    transcript_path = os.path.join(iemocap_root, session, "dialog", "transcriptions", f"{conv_id}.txt")
    utts = parse_transcript_lines(transcript_path)
    if len(utts) == 0:
        return None, None

    target = normalize_text(text)
    candidates = []
    for i, (utt_id, utt_text) in enumerate(utts):
        if normalize_text(utt_text) != target:
            continue
        wav_path = os.path.join(iemocap_root, session, "sentences", "wav", conv_id, f"{utt_id}.wav")
        if not os.path.exists(wav_path):
            continue
        candidates.append((abs(i - turn), i, utt_id, wav_path))

    if len(candidates) == 0:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, best_utt_id, best_wav = candidates[0]
    return best_utt_id, best_wav


def main():
    parser = argparse.ArgumentParser(description="Fill missing IEMOCAP Whisper features using same-text utterance audio.")
    parser.add_argument("--feature_pkl", type=str, required=True)
    parser.add_argument("--iemocap_pkl", type=str, default="/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/iemocap.pkl")
    parser.add_argument("--iemocap_root", type=str, default="/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--output_pkl", type=str, default=None)
    args = parser.parse_args()

    payload = pickle.load(open(args.feature_pkl, "rb"))
    data = pickle.load(open(args.iemocap_pkl, "rb"))
    sentence_dict = data[2]

    missing = list(payload.get("missing_sample_ids", []))
    if len(missing) == 0:
        print("No missing sample ids. Nothing to fill.")
        return

    sample_features = payload.get("sample_features", {})
    dialogue_features = payload.get("dialogue_features", {})
    conv_turn_to_uttid = payload.get("conv_turn_to_uttid", {})

    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    model = WhisperModel.from_pretrained(args.whisper_model)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)
    if args.dtype == "float16" and device.type == "cuda":
        model = model.half()
    model.eval()

    filled = []
    still_missing = []
    replacement_map = payload.get("replacement_map", {})

    for sid in tqdm(missing, desc="Fill missing with same-text audio"):
        if "_" not in str(sid):
            still_missing.append(sid)
            continue
        conv_id, turn_str = str(sid).rsplit("_", 1)
        try:
            turn = int(turn_str)
        except ValueError:
            still_missing.append(sid)
            continue

        if conv_id not in sentence_dict or turn >= len(sentence_dict[conv_id]):
            still_missing.append(sid)
            continue

        text = sentence_dict[conv_id][turn]
        replacement_utt_id, replacement_wav = pick_same_text_candidate(conv_id, turn, text, args.iemocap_root)
        if replacement_wav is None:
            still_missing.append(sid)
            continue

        feat = extract_whisper_feature(model, processor, replacement_wav, device)
        feat_list = feat.tolist()
        sample_features[sid] = feat_list

        if conv_id in dialogue_features and turn < len(dialogue_features[conv_id]):
            dialogue_features[conv_id][turn] = feat_list

        replacement_map[sid] = {
            "replacement_utt_id": replacement_utt_id,
            "replacement_wav": replacement_wav,
            "original_missing_utt_id": conv_turn_to_uttid.get(sid),
        }
        filled.append(sid)

    payload["sample_features"] = sample_features
    payload["dialogue_features"] = dialogue_features
    payload["replacement_map"] = replacement_map
    payload["missing_sample_ids"] = still_missing

    out_path = args.feature_pkl if args.inplace else args.output_pkl
    if out_path is None:
        raise ValueError("Set --inplace or provide --output_pkl")

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved: {out_path}")
    print(f"filled_count: {len(filled)}")
    print(f"remaining_missing_count: {len(still_missing)}")
    if len(filled) > 0:
        print("filled_examples:", filled[:10])


if __name__ == "__main__":
    main()
