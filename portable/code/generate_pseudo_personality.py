import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAITS = ["O", "C", "E", "A", "N"]


def parse_discrete_personality(text):
    raw = str(text).strip()

    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and all(k in parsed for k in TRAITS):
            out = {}
            for k in TRAITS:
                v = str(parsed[k]).strip().lower()
                if v not in ("low", "medium", "high"):
                    return None
                out[k] = v
            return out
    except Exception:
        pass

    lowered = raw.lower()

    # Pattern like "low openness, medium conscientiousness, ..."
    trait_alias = {
        "O": "openness",
        "C": "conscientiousness",
        "E": "extraversion",
        "A": "agreeableness",
        "N": "neuroticism",
    }
    out = {}
    for trait, alias in trait_alias.items():
        m1 = re.search(rf"(low|medium|high)\s+{alias}", lowered)
        m2 = re.search(rf"{trait.lower()}\s*[:=]\s*(low|medium|high)", lowered)
        if m1:
            out[trait] = m1.group(1)
        elif m2:
            out[trait] = m2.group(1)
    if len(out) == 5:
        return out
    return None


def build_prompts_from_dataset_pickle(dataset, pkl_path):
    data = pickle.load(open(pkl_path, "rb"))

    dataset = str(dataset)
    if dataset == "iemocap":
        all_conv_ids = data[3] + data[4] + data[5]
        utterance_dict = data[2]

        def get_speaker_idx(conv_id, turn):
            spk = str(data[0][conv_id][turn]).strip().upper()
            return 0 if spk == "M" else 1

    elif dataset == "meld":
        all_conv_ids = data[4] + data[5] + data[6]
        utterance_dict = data[3]

        def get_speaker_idx(conv_id, turn):
            return data[0][conv_id][turn].index(1)

    elif dataset == "EmoryNLP":
        all_conv_ids = data[3] + data[4] + data[5]
        utterance_dict = data[2]

        def get_speaker_idx(conv_id, turn):
            return data[0][conv_id][turn].index(1)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    samples = []
    for conv_id in all_conv_ids:
        utterances = utterance_dict[conv_id]

        utt_by_spk = defaultdict(list)
        for turn, utt in enumerate(utterances):
            spk_idx = get_speaker_idx(conv_id, turn)
            utt_by_spk[spk_idx].append(str(utt))

        for spk_idx, utts in utt_by_spk.items():
            if not utts:
                continue
            utt_block = " ".join([f'[{i + 1}] "{u}"' for i, u in enumerate(utts)])
            prompt = (
                "Now you are expert of personality analysis. "
                f"Given all utterances from Speaker_{spk_idx} in this conversation: {utt_block} "
                "predict the trait level of OCEAN [O, C, E, A, N] using low/medium/high."
            )
            samples.append({
                "key": f"{conv_id}|Speaker_{spk_idx}",
                "prompt": prompt,
            })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo personality map from a stage1 model.")
    parser.add_argument("--dataset", type=str, default="meld")
    parser.add_argument("--dataset_pkl", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--stage1_checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset_pkl = args.dataset_pkl or f"/n/work6/yiwang/MoodFilm/portable/original_data/{args.dataset}/{args.dataset}.pkl"
    if not os.path.exists(dataset_pkl):
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_pkl}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        output_hidden_states=True,
    ).half().cuda()
    model = PeftModel.from_pretrained(model, args.stage1_checkpoint_dir)
    model.eval()

    samples = build_prompts_from_dataset_pickle(args.dataset, dataset_pkl)
    result = {}

    bs = max(1, int(args.batch_size))
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        prompts = [x["prompt"] for x in chunk]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.0,
            )

        for j, sample in enumerate(chunk):
            input_len = int(inputs["attention_mask"][j].sum().item())
            pred_ids = out_ids[j][input_len:]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            parsed = parse_discrete_personality(pred_text)
            if parsed is None:
                parsed = {"O": "medium", "C": "medium", "E": "medium", "A": "medium", "N": "medium"}
            result[sample["key"]] = parsed

        if (i // bs) % 50 == 0:
            print(f"Processed {min(i + bs, len(samples))}/{len(samples)} speaker prompts")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(args.output_json)


if __name__ == "__main__":
    main()
