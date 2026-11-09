"""Run frozen IEMOCAP personality adapter on provisional MSP scene groups."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PORTABLE = ROOT.parent / 'portable'
sys.path.insert(0, str(PORTABLE / 'dialogue_personality'))
sys.path.insert(0, str(PORTABLE / 'code'))
import infer as existing
from export_stage1_predicted_personality import load_eval_file, parse_personality_continuous
from prepare_stage1 import prepare


def export(predictions, destination, mapping):
    _, records = load_eval_file(predictions)
    lookup = {m['sample_id']: m for m in mapping}
    destination.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for record in records:
        sid = str(record.get('sample_id') or '')
        parsed = parse_personality_continuous(record.get('output_raw') or record.get('output') or '')
        if sid not in lookup or parsed is None:
            failed.append(sid)
            continue
        rows.append({**lookup[sid], **{t: round(float(parsed[t]), 3) for t in 'OCEAN'}})
    if len(rows) != len(mapping) or failed:
        raise RuntimeError(f'Only {len(rows)}/{len(mapping)} valid predictions; failures={failed[:10]}')
    with (destination / 'dialogue_personality.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id','dialogue_id','speaker','split',*'OCEAN'])
        writer.writeheader(); writer.writerows(rows)
    print(f'Exported {len(rows)} OCEAN predictions: {destination}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, default=29765)
    p.add_argument('--run', action='store_true')
    opt = p.parse_args()
    os.chdir(PORTABLE)
    rows = prepare()
    cfg = json.loads((PORTABLE / 'dialogue_personality/stage1_original_args.json').read_text())
    cfg.update(dataset='meld', model_name_or_path=str(PORTABLE/'LLM_bases/Qwen2.5-Omni-7B'),
               checkpoint_dir=str(PORTABLE/'dialogue_personality/checkpoint_07648'),
               data_dir=str(ROOT/'stage1_input'), audio_feature_pkl=str(ROOT/'cache/qwen_native_mel.pkl'),
               audio_qwen_processor_path=str(PORTABLE/'LLM_bases/Qwen2.5-Omni-7B'),
               audio_iemocap_root=None, audio_iemocap_pkl=None, video_manifest_jsonl=None,
               do_train=False, do_eval=True, audio_enable=True, text_enable=True, video_enable=False,
               audio_embedding_source='precomputed_pkl', audio_input_mode='text_audio',
               personality_media_scope='full', audio_context_window=-1,
               gradient_checkpointing=False, eval_batch_size=1, gradient_accumulation_steps=1,
               max_length=32768, max_new_tokens=128, attn_implementation='sdpa',
               save_best_checkpoint=False, disable_deepspeed_checkpoint_save=True)
    for key in ('warmup_steps','scheduler_num_training_steps','output_dir','deepspeed_config'):
        cfg.pop(key, None)
    if not opt.run:
        print(f'Prepared {len(rows)} predictions. Use --run to extract mel and run GPU inference.')
        return
    for path in (Path(cfg['model_name_or_path']), Path(cfg['checkpoint_dir'])/'trainable_model.safetensors'):
        if not path.exists():
            raise FileNotFoundError(path)
    if not Path(cfg['audio_feature_pkl']).is_file():
        from extract_mel import main as extract
        extract([])
    output = ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    existing.audit(cfg, rows, output)
    smoke = output/'smoke_input'; smoke.mkdir()
    mapping = json.loads((ROOT/'stage1_mapping.json').read_text())
    smoke_rows = [min(rows, key=lambda r: len(r['audio_turn_sample_ids'])),
                  max(rows, key=lambda r: len(r['audio_turn_sample_ids']))]
    (smoke/'test.json').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke_rows))
    smoke_ids = {r['sample_id'] for r in smoke_rows}
    smoke_mapping = [m for m in mapping if m['sample_id'] in smoke_ids]
    pred = existing.launch(dict(cfg,data_dir=str(smoke)), str(opt.gpu), opt.port, output/'smoke')
    export(pred, output/'smoke_export', smoke_mapping)
    pred = existing.launch(cfg, str(opt.gpu), opt.port, output/'inference')
    export(pred, output/'export', mapping)
    print(f'FINISHED: {output}/export/dialogue_personality.csv')


if __name__ == '__main__':
    main()
