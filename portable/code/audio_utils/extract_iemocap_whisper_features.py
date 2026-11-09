import argparse
import os
import pickle
import re
from collections import defaultdict

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


def build_conv_turn_to_uttid_map(iemocap_pkl, iemocap_root):
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
		utt_ids = []

		# Fast path: one-to-one by order.
		if len(transcript_utts) == len(sent_list):
			utt_ids = [u[0] for u in transcript_utts]
		else:
			# Fallback: greedy text match.
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


def main():
	parser = argparse.ArgumentParser(description="Extract IEMOCAP utterance Whisper features for multimodal InstructERC.")
	parser.add_argument("--iemocap_pkl", type=str, default="/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/iemocap.pkl")
	parser.add_argument("--iemocap_root", type=str, default="/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release")
	parser.add_argument("--whisper_model", type=str, default="openai/whisper-small")
	parser.add_argument("--output_pkl", type=str, default="/n/work6/yiwang/MoodFilm/portable/original_data/iemocap/iemocap_whisper_features.pkl")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

	conv_turn_to_uttid, data = build_conv_turn_to_uttid_map(args.iemocap_pkl, args.iemocap_root)

	processor = WhisperProcessor.from_pretrained(args.whisper_model)
	model = WhisperModel.from_pretrained(args.whisper_model)
	device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
	model = model.to(device)
	if args.dtype == "float16" and device.type == "cuda":
		model = model.half()
	model.eval()

	sentence_dict = data[2]
	all_ids = data[3] + data[4] + data[5]

	sample_features = {}
	dialogue_features = {}
	missing = []

	for conv_id in tqdm(all_ids, desc="Extract audio features"):
		session = dialogue_to_session(conv_id)
		if session is None:
			continue
		seq_feats = []
		for turn in range(len(sentence_dict[conv_id])):
			sample_id = f"{conv_id}_{turn}"
			utt_id = conv_turn_to_uttid.get(sample_id, None)
			if utt_id is None:
				seq_feats.append(None)
				missing.append(sample_id)
				continue

			wav_path = os.path.join(args.iemocap_root, session, "sentences", "wav", conv_id, f"{utt_id}.wav")
			if not os.path.exists(wav_path):
				seq_feats.append(None)
				missing.append(sample_id)
				continue

			feat = extract_whisper_feature(model, processor, wav_path, device)
			sample_features[sample_id] = feat.tolist()
			seq_feats.append(feat.tolist())

		dialogue_features[conv_id] = seq_feats

	feature_dim = 0
	if len(sample_features) > 0:
		feature_dim = len(next(iter(sample_features.values())))

	payload = {
		"whisper_model": args.whisper_model,
		"feature_dim": feature_dim,
		"sample_features": sample_features,
		"dialogue_features": dialogue_features,
		"conv_turn_to_uttid": conv_turn_to_uttid,
		"missing_sample_ids": missing,
	}
	with open(args.output_pkl, "wb") as f:
		pickle.dump(payload, f)

	print(f"Saved: {args.output_pkl}")
	print(f"feature_dim: {feature_dim}")
	print(f"num_sample_features: {len(sample_features)}")
	print(f"missing_sample_ids: {len(missing)}")


if __name__ == "__main__":
	main()
