"""Read-only real-cache preflight for IEMOCAP utterance feature alignment.

This deliberately does not load the 7B model or launch training. GPU model
forward/generation is checked by the runner's one-step smoke runs.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils.data_utils import Seq2SeqCollator


def first_row(path, prefix=None, target_prefix=None, min_turn=None):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if (prefix is None or str(row['sample_id']).startswith(prefix)) and (
                target_prefix is None or str(row['target']).startswith(target_prefix)
            ) and (min_turn is None or int(str(row['sample_id']).rsplit('_', 1)[-1]) >= min_turn):
                return row
    raise ValueError(f'No matching row: {path} / {prefix}')


def to_example(row):
    meta = dict(row)
    return (row['input'], row['target'], row['sample_id'], meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--audio-cache', type=Path, required=True)
    parser.add_argument('--text-cache', type=Path, required=True)
    parser.add_argument('--iemocap-pkl', type=Path, required=True)
    parser.add_argument('--stage1-none', type=Path, required=True)
    parser.add_argument('--stage1-ternary', type=Path, required=True)
    parser.add_argument('--stage2', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = SimpleNamespace(
        dataset='iemocap', model_name_or_path=str(args.model),
        audio_qwen_processor_path=str(args.model), audio_iemocap_pkl=str(args.iemocap_pkl),
        audio_enable=True, audio_embedding_source='precomputed_tokens',
        audio_feature_pkl=str(args.audio_cache), audio_input_mode='text_audio',
        audio_embedding_usage='direct', audio_context_window=13,
        text_enable=True, text_feature_pkl=str(args.text_cache),
        utterance_feature_text=True, utterance_feature_audio=True,
        utterance_feature_history_window=12,
        personality_media_scope='full', max_length=16384,
        emotion_prediction=False, open_ended=False,
    )
    collator = Seq2SeqCollator(config, tokenizer, mode='train')
    full = first_row(args.stage1_none, 'Personality_full_')
    speaker = first_row(args.stage1_none, 'Speaker_')
    pair = first_row(args.stage1_ternary, target_prefix='TraitCompare_')
    erc = first_row(args.stage2, min_turn=12)
    for label, row in (('full', full), ('pair', pair), ('speaker', speaker), ('erc', erc)):
        batch = collator([to_example(row)])
        ids = batch['utterance_turn_ids'][0]
        used_text = int(batch['utterance_text_mask'].sum())
        used_audio = int(batch['utterance_audio_mask'].sum())
        direct = int(batch.get('precomputed_audio_features', torch.empty(0)).shape[0])
        if label == 'speaker':
            assert used_text == used_audio == 0, 'Speaker auxiliary task got feature tokens'
            assert direct > 0, 'Speaker auxiliary task lost direct audio'
        else:
            assert used_text == len(ids) > 0, (label, used_text, len(ids))
            assert direct == 0, f'{label} unexpectedly retained direct audio'
            assert used_audio <= len(ids)
            assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)
        if label in ('full', 'pair'):
            assert len(ids) > 18, f'{label} unexpectedly truncated to a fixed short context'
        if label == 'erc':
            assert len(ids) == 13, f'w12 ERC should use history+target: {len(ids)}'
        print(f'[REAL DATA OK] {label}: turns={len(ids)} text={used_text} pooled_audio={used_audio} direct_audio={direct}', flush=True)
    collator.utterance_feature_enable = False
    collator.utterance_feature_text = False
    collator.utterance_feature_audio = False
    collator.stage1_speaker_audio_only = True
    prompt_personality = collator([to_example(full)])
    prompt_speaker = collator([to_example(speaker)])
    prompt_erc = collator([to_example(erc)])
    assert int(prompt_personality['direct_audio_utterance_count'].sum()) == 0
    assert int(prompt_speaker['direct_audio_utterance_count'].sum()) == 1
    assert int(prompt_erc['direct_audio_utterance_count'].sum()) == 0
    collator.stage1_speaker_audio_only = False
    raw_personality = collator([to_example(full)])
    assert int(raw_personality['direct_audio_utterance_count'].sum()) == 53
    collator.utterance_direct_audio_once = True
    raw_erc = collator([to_example(erc)])
    raw_erc_count = int(raw_erc['direct_audio_utterance_count'].sum())
    assert raw_erc_count == 13, f'raw w12 direct utterances={raw_erc_count}'
    print('[MATCHED BASELINES OK] prompt main=0/speaker=1; raw full=53; raw w12=13 direct utterances', flush=True)
    print('[PREFLIGHT DATA OK] Real IEMOCAP cache and prompt alignment', flush=True)


if __name__ == '__main__':
    main()
