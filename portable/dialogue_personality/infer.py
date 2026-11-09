"""Check by default; --run performs CPU audio audit, GPU smoke, then full inference."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'code'))
from run import training_parser

def argv_from(cfg):
    parser = training_parser()
    actions = {a.dest: a for a in parser._actions}
    result = []
    for key, value in cfg.items():
        if value is None: continue
        if key not in actions:
            if key not in {'model_type', 'do_sample', 'preprocess_inputs', 'open_ended'}:
                raise ValueError(f'Unknown configuration key: {key}')
            continue
        action = actions[key]
        if isinstance(action, argparse._StoreTrueAction):
            if value: result.append(action.option_strings[0])
        else: result.extend([action.option_strings[0], str(value)])
    args = parser.parse_args(result)
    # Match main_new.py's normalization for the collator audit.
    for key, value in vars(args).items():
        if value in ('True', 'False') and key not in ('statistic_mode',):
            setattr(args, key, value == 'True')
    return result, args

def audit(cfg, rows, output, skip_missing_audio_samples=False):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from data_utils.data_utils import Seq2SeqCollator
    _, args = argv_from(dict(cfg, output_dir=str(output)))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name_or_path'], local_files_only=True)
    collator = Seq2SeqCollator(args, tokenizer, mode='dev')
    requested = {sid for r in rows for sid in r['audio_turn_sample_ids']}
    missing = sorted(requested - set(collator.qwen_mel_features))
    excluded = [r['sample_id'] for r in rows if set(missing).intersection(r['audio_turn_sample_ids'])]
    missing_report = {'missing_audio_ids': missing, 'affected_sample_ids': excluded,
                      'policy': 'exclude_entire_sample' if skip_missing_audio_samples else 'error',
                      'original_sample_count': len(rows)}
    (output/'missing_audio_report.json').write_text(json.dumps(missing_report, indent=2)+'\n')
    if missing and not skip_missing_audio_samples:
        raise RuntimeError(f'Missing {len(missing)} audio IDs: {missing[:20]}. '
                           'To explicitly exclude affected dialogue-character samples, '
                           'use --skip-missing-audio-samples; see missing_audio_report.json.')
    if missing:
        excluded_set = set(excluded)
        rows[:] = [r for r in rows if r['sample_id'] not in excluded_set]
        if len(rows) < 2: raise RuntimeError('Fewer than two complete samples remain.')
        print(f'Explicitly excluded {len(excluded)} incomplete samples; retained {len(rows)}.', flush=True)
        filtered = output/'complete_audio_input'
        filtered.mkdir()
        (filtered/'test.json').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        cfg['data_dir'] = str(filtered)
        requested = {sid for r in rows for sid in r['audio_turn_sample_ids']}
    for sid in requested:
        a = collator.qwen_mel_features[sid]
        if a.shape[0] != 128 or a.shape[1] == 0 or not np.isfinite(a).all():
            raise RuntimeError(f'Invalid mel feature: {sid}')
    # Test actual collated batches for the shortest and longest estimated audio streams.
    ordered = sorted(rows, key=lambda r: sum(collator.qwen_mel_lengths[s] for s in r['audio_turn_sample_ids']))
    inspected = []
    for row in [ordered[0], ordered[-1]]:
        record = {'text': row['input'], 'sample_id': row['sample_id'],
                  'meta': {'audio_turn_sample_ids': row['audio_turn_sample_ids']}}
        selected = collator._effective_qwen_turn_sample_ids(record)
        _, refs, _ = collator._build_qwen_content_items(row['input'], row['sample_id'], selected)
        assert [r['sample_id'] for r in refs] == row['audio_turn_sample_ids'], 'Audio order mismatch'
        batch = collator([[row['input'], row['target'], row['sample_id'], record['meta']]])
        features = batch.get('input_features')
        assert features is not None and features.numel() and torch.isfinite(features).all()
        assert features.abs().sum().item() > 0, 'All-zero mel input'
        assert features.shape[0] == len(selected), 'Audio turns lost'
        inspected.append({'sample_id': row['sample_id'], 'turns': len(selected),
                          'tokens': int(batch['attention_mask'].sum()),
                          'mel_shape': list(features.shape)})
    report = {'checkpoint': cfg['checkpoint_dir'], 'requested_audio_ids': len(requested),
              'missing_audio_ids': len(missing), 'retained_missing_audio_ids': 0,
              'excluded_sample_ids': excluded, 'retained_sample_count': len(rows),
              'inspected_batches': inspected,
              'no_personality_ground_truth': True,
              'note': 'Batch audit proves audio attachment, not that historical training learned to use audio.'}
    (output/'audio_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)

def launch(cfg, gpus, port, directory):
    cfg = dict(cfg, output_dir=str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    cfg['deepspeed_config'] = str(directory/'deepspeed.json')
    (directory/'deepspeed.json').write_text(json.dumps({
        'train_batch_size': len(gpus.split(',')), 'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 1, 'bf16': {'enabled': True},
        'zero_optimization': {'stage': 0}}))
    cfg['batch_size'] = len(gpus.split(','))
    argv, _ = argv_from(cfg)
    (directory/'resolved_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    cmd = [sys.executable, '-m', 'deepspeed.launcher.runner', '--include',
           'localhost:'+gpus, '--master_port', str(port), 'code/main_new.py']+argv
    with (directory/'terminal.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            if process.wait(): raise RuntimeError(f'Inference failed; see {directory}/terminal.log')
        except BaseException:
            process.terminate()
            raise
    return directory/'preds_for_eval.text'

def export(prediction, output):
    subprocess.run([sys.executable, str(ROOT/'code/meld_utils/export_meld_dialogue_personality.py'),
                    '--prediction-file', str(prediction), '--output-dir', str(output)], check=True)
    subprocess.run([sys.executable, str(ROOT/'code/meld_utils/summarize_dialogue_personality.py'),
                    str(output/'dialogue_personality.csv'), '--output', str(output/'dispersion.json')], check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', help='Physical GPU IDs; required for --run')
    p.add_argument('--run', action='store_true')
    p.add_argument('--audit', action='store_true', help='CPU cache and real collator check; no GPU forward')
    p.add_argument('--skip-missing-audio-samples', action='store_true',
                   help='Explicitly exclude entire dialogue-character samples with missing audio; save exclusion manifest.')
    p.add_argument('--max-length', type=int, default=32768)
    p.add_argument('--port', type=int, default=29627)
    opt = p.parse_args()
    os.chdir(ROOT)
    cfg = json.loads((ROOT/'dialogue_personality/stage1_original_args.json').read_text())
    cfg.update(dataset='meld', model_name_or_path=str(ROOT/'LLM_bases/Qwen2.5-Omni-7B'),
               checkpoint_dir=str(ROOT/'dialogue_personality/checkpoint_07648'),
               audio_qwen_processor_path=str(ROOT/'LLM_bases/Qwen2.5-Omni-7B'),
               data_dir=str(ROOT/'dialogue_personality/inputs_anon_full_audio'),
               audio_feature_pkl=str(ROOT/'original_data/meld/meld_qwen_native_mel_features_splitkey.pkl'),
               audio_iemocap_root=None, audio_iemocap_pkl=None, video_manifest_jsonl=None,
               do_train=False, do_eval=True, audio_enable=True, text_enable=True, video_enable=False,
               audio_embedding_source='precomputed_pkl', audio_input_mode='text_audio',
               personality_media_scope='full', audio_context_window=-1,
               gradient_checkpointing=False, eval_batch_size=1, gradient_accumulation_steps=1,
               max_length=opt.max_length, max_new_tokens=128, attn_implementation='sdpa',
               save_best_checkpoint=False, disable_deepspeed_checkpoint_save=True)
    for key in ['warmup_steps', 'scheduler_num_training_steps', 'output_dir', 'deepspeed_config']:
        cfg.pop(key, None)
    required = [Path(cfg[k]) for k in ['model_name_or_path','checkpoint_dir','audio_feature_pkl','data_dir']]
    required += [Path(cfg['checkpoint_dir'])/'trainable_model.safetensors', Path(cfg['data_dir'])/'test.json']
    missing = [str(f) for f in required if not f.exists()]
    if missing: raise SystemExit('Missing files; migration is not complete:\n'+'\n'.join(missing))
    model = Path(cfg['model_name_or_path'])
    index = json.loads((model/'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (model/shard).is_file(): raise SystemExit(f'Missing model shard {shard}')
    # JSONL boundaries are physical newlines, not Unicode NEL inside strings.
    # MELD transcripts contain U+0085; str.splitlines() incorrectly splits those.
    with (Path(cfg['data_dir'])/'test.json').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    argv_from(dict(cfg, output_dir=str(ROOT/'dialogue_personality/results/check')))
    print(f'Path/CLI check passed: {len(rows)} samples. No training requested.')
    if not (opt.run or opt.audit): return
    if opt.run and (not opt.gpus or not all(x.isdigit() for x in opt.gpus.split(',')) or len(set(opt.gpus.split(','))) != len(opt.gpus.split(','))):
        p.error('--run requires unique explicit physical --gpus IDs')
    output = ROOT/'dialogue_personality/results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    (output/'request.json').write_text(json.dumps(vars(opt), indent=2)+'\n')
    audit(cfg, rows, output, opt.skip_missing_audio_samples)
    if not opt.run: return
    smoke = output/'smoke_input'; smoke.mkdir()
    (smoke/'test.json').write_text('\n'.join(json.dumps(r) for r in rows[:2])+'\n')
    smoke_prediction = launch(dict(cfg, data_dir=str(smoke)), opt.gpus, opt.port, output/'smoke')
    export(smoke_prediction, output/'smoke_export')
    smoke_report = json.loads((output/'smoke_export/train_speaker_aggregates.json').read_text())
    if smoke_report['num_predictions'] != 2 or smoke_report['num_parse_failures']:
        raise RuntimeError('Smoke outputs are not two valid OCEAN vectors; full inference not started.')
    prediction = launch(cfg, opt.gpus, opt.port, output/'inference')
    export(prediction, output/'export')
    print(f'FINISHED. Results: {output}/export. Do not use placeholder-label CCC/MAE.')

if __name__ == '__main__': main()
