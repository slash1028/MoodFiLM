"""CPU-only, end-to-end collator smoke test for the stage-1 token/mel comparison."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_utils.data_utils import Seq2SeqCollator  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def load_examples(path):
    pair = None
    full = None
    pair_count = 0
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        target = str(row.get("target", ""))
        if target.startswith("TraitRank_"):
            pair_count += 1
            ids = row.get("audio_turn_sample_ids") or row.get("meta", {}).get("audio_turn_sample_ids", [])
            markers = row["input"].count("[AUDIO_UTTERANCE]")
            if not ids or len(ids) != markers:
                raise RuntimeError(f"Unaligned pairwise audio: {row['sample_id']}")
            if pair is None or len(ids) < len(pair.get("audio_turn_sample_ids") or pair.get("meta", {}).get("audio_turn_sample_ids", [])):
                pair = row
        elif str(row.get("sample_id", "")).startswith("Personality_full_"):
            if full is None or len(row["input"]) < len(full["input"]):
                full = row
    if pair is None or full is None or pair_count < 1000:
        raise RuntimeError(f"Expected corrected pairwise and full-personality rows; pairs={pair_count}")
    return pair, full, pair_count


def check_mode(label, cache, model, iemocap_pkl, examples, tokenizer):
    source = "precomputed_tokens" if label == "token" else "precomputed_pkl"
    args = SimpleNamespace(
        dataset="iemocap", model_name_or_path=str(model),
        audio_qwen_processor_path=str(model), audio_iemocap_pkl=str(iemocap_pkl),
        audio_enable=True, text_enable=True, video_enable=False,
        audio_embedding_source=source, audio_feature_pkl=str(cache),
        audio_input_mode="text_audio", audio_embedding_usage="direct",
        audio_context_window=8, personality_media_scope="target",
        personality_pair_mode="none", enable_stage1_context_prefix=False,
    )
    collator = Seq2SeqCollator(args, tokenizer, mode="dev")
    if collator.personality_media_scope != "target" or not collator.audio_direct_enable:
        raise RuntimeError(f"{label}: effective audio configuration is wrong")
    for name, row in examples.items():
        record = {
            "text": row["input"], "target": row["target"],
            "sample_id": row["sample_id"], "meta": row.get("meta", {}),
            "audio_turn_sample_ids": row.get("audio_turn_sample_ids"),
        }
        ids = (collator._record_audio_turn_sample_ids(record)
               or collator._selected_qwen_turn_sample_ids(record["sample_id"]))
        prompts, refs, visuals, counts, messages = collator._processor_inputs_from_records(
            [record], [ids]
        )
        if not refs or counts != [len(refs)]:
            raise RuntimeError(f"{label}/{name}: no audio attached")
        if name == "pair":
            if len(refs) != len(ids):
                raise RuntimeError(f"{label}/pair: dropped audio turns")
            speakers = {int(ref["sample_id"].rsplit("_", 1)[-1]) for ref in refs}
            conv = refs[0]["sample_id"].rsplit("_", 1)[0]
            actual_speakers = {collator.iemocap_speaker_ids_by_conv[conv][i] for i in speakers}
            if actual_speakers != {0, 1}:
                raise RuntimeError(f"{label}/pair: both speakers are not present")
        output = collator._call_qwen_processor(prompts, refs, visuals, messages)
        collator._validate_qwen_audio_token_alignment(output, [record], name)
        audio_key = "precomputed_audio_features" if label == "token" else "input_features"
        if audio_key not in output or output[audio_key].numel() == 0:
            raise RuntimeError(f"{label}/{name}: {audio_key} missing from model inputs")
        print(
            f"[AUDIO SMOKE OK] {label}/{name}: refs={len(refs)} "
            f"input_tokens={output['input_ids'].shape[-1]} "
            f"{audio_key}={tuple(output[audio_key].shape)}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--iemocap-pkl", type=Path, required=True)
    parser.add_argument("--token-pkl", type=Path, required=True)
    parser.add_argument("--mel-pkl", type=Path, required=True)
    args = parser.parse_args()
    pair, full, pair_count = load_examples(args.data_dir / "train.json")
    print(f"[DATA CHECK OK] corrected prompt-pair rows={pair_count}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    examples = {"pair": pair, "full": full}
    check_mode("token", args.token_pkl, args.model_dir, args.iemocap_pkl, examples, tokenizer)
    check_mode("mel", args.mel_pkl, args.model_dir, args.iemocap_pkl, examples, tokenizer)


if __name__ == "__main__":
    main()
