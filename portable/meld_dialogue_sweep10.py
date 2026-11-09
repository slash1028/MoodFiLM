"""Prepare/launch ten single-GPU MELD ERC comparisons; never falls back to old predictions."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
from run import training_parser
from meld_utils import build_meld_gtperson_stage2_data as builder
from case_study.add_personality_prompt import inject, prompt_clause

TRAITS = ('O', 'C', 'E', 'A', 'N')
ALLOWED_MISSING = {('train', 125, 'Ross'), ('valid', 110, 'Phoebe'), ('valid', 110, 'Rachel')}
SETTINGS = [
    ('01_peld_s42', 'peld', 42, True, True, {}),
    ('02_dialogue_s42', 'dialogue', 42, True, True, {}),
    ('03_peld_s43', 'peld', 43, True, True, {}),
    ('04_dialogue_s43', 'dialogue', 43, True, True, {}),
    ('05_trainmean_s42', 'trainmean', 42, True, True, {}),
    ('06_dialogue_no_prompt', 'dialogue', 42, False, True, {}),
    ('07_dialogue_prompt_only', 'dialogue', 42, True, False, {}),
    ('08_no_personality_no_mood', 'none', 42, False, False, {}),
    ('09_dialogue_lr7e5', 'dialogue', 42, True, True, {'learning_rate': 7e-5}),
    ('10_dialogue_lambda030', 'dialogue', 42, True, True, {'speaker_mood_lambda': .30}),
]

def load_predictions(path, expected):
    predictions = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['split'], int(row['dialogue_id']), row['speaker'])
            if key in predictions: raise ValueError(f'Duplicate prediction: {key}')
            values = [float(row[t]) for t in TRAITS]
            if not all(math.isfinite(x) and 1 <= x <= 7 for x in values):
                raise ValueError(f'Invalid OCEAN: {key}')
            predictions[key] = [(x-1)/6 for x in values]
    missing = expected-set(predictions)
    extra = set(predictions)-expected
    if extra or missing-ALLOWED_MISSING:
        raise ValueError(f'Incomplete/mismatched predictions: missing={len(missing)}, extra={len(extra)}; '
                         f'examples={sorted(missing-ALLOWED_MISSING)[:5]}')
    return predictions, missing

def profiles(predictions, peld):
    by_speaker = {}
    for (split, _, speaker), value in predictions.items():
        if split == 'train': by_speaker.setdefault(speaker, []).append(value)
    means = {s: [statistics.mean(v[t] for v in values) for t in range(5)]
             for s, values in by_speaker.items()}
    return {'dialogue': predictions,
            'peld': {key: peld[key[2]] for key in predictions},
            'trainmean': {key: means[key[2]] for key in predictions}, 'none': {}}

def make_rows(split, dialogues, values, with_prompt, source):
    # Empty fallback prevents builder from silently using global PELD.
    rows = builder.build_records(split, dialogues, {}, 6, 'none', False, {}, True, values)
    for row in rows:
        value = row['gt_personality']
        row['personality'] = value
        row['personality_source'] = source
        # These profiles are predicted/PELD proxies, not MELD ground-truth personality.
        # Keep historical field for builder compatibility; runner selects 'predicted'.
        row['personality_labels_available'] = False
        if with_prompt and value is not None:
            row['input'] = inject(row['input'], prompt_clause(value))
    return rows

def cli_args(cfg):
    parser = training_parser()
    actions = {a.dest: a for a in parser._actions}
    argv = []
    for key, value in cfg.items():
        if value is None: continue
        if key not in actions:
            if key not in {'model_type', 'do_sample', 'preprocess_inputs', 'open_ended'}:
                raise ValueError(f'Unknown CLI key: {key}')
            continue
        action = actions[key]
        if isinstance(action, argparse._StoreTrueAction):
            if value: argv.append(action.option_strings[0])
        else: argv.extend([action.option_strings[0], str(value)])
    parser.parse_args(argv)
    return argv

def summarize(output):
    rows = []
    for name, *_ in SETTINGS:
        for path in sorted((output/name).glob('preds_for_eval_[0-9]*.text')):
            with path.open() as stream:
                report, _ = json.JSONDecoder().raw_decode(stream.read(65536))
            score = report['emotion']['score']
            rows.append({'setting': name, 'epoch': int(path.stem.rsplit('_',1)[1])+1,
                         'WF1': score['F1_SA'], 'ACC': score['Acc_SA']})
    with (output/'epoch_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['setting','epoch','WF1','ACC'])
        writer.writeheader(); writer.writerows(rows)
    best = {name: max([r for r in rows if r['setting']==name], key=lambda r:r['WF1'])
            for name, *_ in SETTINGS if any(r['setting']==name for r in rows)}
    (output/'best_test_metrics.json').write_text(json.dumps(best, indent=2)+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-csv', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--run-id', default='meld_dialogue10_1027')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--port', type=int, default=29647)
    parser.add_argument('--zero-stage', type=int, choices=[0, 2], default=0,
                        help='Single GPU defaults to stage 0; stage 2 is optional.')
    opt = parser.parse_args()
    if not opt.gpu.isdigit(): parser.error('Exactly one physical GPU number is required')
    if opt.epochs < 1: parser.error('epochs must be positive')
    if Path(opt.run_id).name != opt.run_id or opt.run_id in ('.','..'): parser.error('Invalid run-id')
    os.chdir(ROOT)
    raw = ROOT/'dialogue_personality/csv'
    dialogues = {split: builder.read_split(raw, split) for split in builder.SPLIT_FILES}
    expected = {(split,did,t['speaker']) for split,ds in dialogues.items()
                for did,turns in ds.items() for t in turns if t['speaker'] in builder.DEFAULT_PELD}
    candidates = sorted((ROOT/'dialogue_personality/results').glob('*/export/dialogue_personality.csv'))
    path = opt.prediction_csv.resolve() if opt.prediction_csv else (candidates[-1] if candidates else None)
    if path is None or not path.is_file():
        raise SystemExit('STOP: no completed dialogue-personality export. Finish inference first; no training started.')
    predictions, missing = load_predictions(path, expected)
    peld = json.loads((ROOT/'original_data/meld/peld_personality_map.json').read_text())
    maps = profiles(predictions, peld)
    base = json.loads((ROOT/'configs/meld.json').read_text())
    for key in ['model_name_or_path','audio_qwen_processor_path','audio_feature_pkl']:
        base[key] = str((ROOT/base[key]).resolve())
        if not Path(base[key]).exists(): raise FileNotFoundError(base[key])
    index = json.loads((Path(base['model_name_or_path'])/'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        if not (Path(base['model_name_or_path'])/shard).is_file(): raise FileNotFoundError(shard)
    base.update(checkpoint_dir=None, batch_size=40, gradient_accumulation_steps=40,
                eval_batch_size=1, num_train_epochs=opt.epochs, do_train=True, do_eval=True,
                speaker_mood_source='predicted', audio_embedding_source='precomputed_tokens',
                audio_context_window=6, audio_adapter_mode='residual',
                enable_latent_mood_prefix=False, save_best_checkpoint=False,
                disable_deepspeed_checkpoint_save=True, save_lightweight_adapter_epochs='all')
    output = ROOT/'meld_dialogue_sweeps'/opt.run_id
    signature = {'zero_stage': opt.zero_stage, 'prediction_csv': str(path), 'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'base': base, 'settings': SETTINGS,
                 'csv_hashes': {s:hashlib.sha256((raw/f).read_bytes()).hexdigest() for s,f in builder.SPLIT_FILES.items()},
                 'peld': peld, 'runner_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    encoded = json.dumps(signature, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output/'running.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = output/'sweep_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(encoded):
        raise RuntimeError('Existing run has different inputs/config/code. Choose a NEW --run-id.')
    manifest.write_text(encoded+'\n')
    coverage = {'expected_dialogue_profiles':len(expected), 'available':len(predictions),
                'excluded_personality_keys':sorted(missing), 'policy':'same unknown mask in all source comparisons; no fallback',
                'erc_counts':{s:sum(len(t) for t in ds.values()) for s,ds in dialogues.items()},
                'protocol':'OFFLINE full-dialogue personality may use future turns; not strictly causal ERC',
                'reference':'simplified b050 epoch4 test WF1 70.232 ACC71.188; reruns use matched masks and current code'}
    (output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    runs = []
    ids_by_split = {}
    for name, source, seed, prompt, mood, overrides in SETTINGS:
        dataset_dir = output/'data'/f'{source}_prompt{int(prompt)}'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for split, ds in dialogues.items():
            rows = make_rows(split, ds, maps[source], prompt, source)
            ids = [r['sample_id'] for r in rows]
            if split in ids_by_split: assert ids == ids_by_split[split]
            ids_by_split[split] = ids
            builder.write_jsonl(dataset_dir/f'{split}.json', rows)
        folder = output/name
        folder.mkdir(exist_ok=True)
        ds_file = folder/'deepspeed.json'
        ds_file.write_text(json.dumps({'train_batch_size':40,'train_micro_batch_size_per_gpu':1,
            'gradient_accumulation_steps':40,'gradient_clipping':1.0,'bf16':{'enabled':True},
            'zero_optimization':({'stage':0} if opt.zero_stage == 0 else
                                {'stage':2,'allgather_partitions':True,'allgather_bucket_size':500000000,
                                 'overlap_comm':True,'reduce_scatter':True,'reduce_bucket_size':500000000,
                                 'contiguous_gradients':True})},indent=2)+'\n')
        cfg = dict(base, data_dir=str(dataset_dir), output_dir=str(folder), run_id=name,
                   deepspeed_config=str(ds_file), seed=seed, use_speaker_mood_dynamics=mood,
                   use_speaker_mood_prior_loss=mood, use_speaker_mood_dynamics_consistency_loss=mood,
                   **overrides)
        argv = cli_args(cfg)
        (folder/'resolved_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        runs.append((name, folder, argv))
    print(json.dumps(coverage, indent=2), flush=True)
    print(f'Prepared 10 settings, {opt.epochs} epochs each. Prediction source: {path}', flush=True)
    if not opt.run:
        print('No training launched. Add --run to execute.'); return
    env = dict(os.environ)
    env['PYTHONFAULTHANDLER'] = '1'
    env.pop('CUDA_VISIBLE_DEVICES',None)
    for name, folder, argv in runs:
        if (folder/'DONE.json').exists():
            print(f'SKIP completed: {name}',flush=True); continue
        # A completed setting is skipped; a partial setting is never overwritten.
        if (folder/'model_args.json').exists() or list(folder.glob('preds_for_eval*.text')):
            raise RuntimeError(f'{name} contains an incomplete previous run. Use a new --run-id; no files overwritten.')
        attempts = folder/'attempts'; attempts.mkdir(exist_ok=True)
        attempt = len(list(attempts.glob('*.log')))+1
        cmd = [sys.executable,'-m','deepspeed.launcher.runner','--include',f'localhost:{opt.gpu}',
               '--master_port',str(opt.port),'code/main_new.py']+argv
        print(f'START {name} (GPU {opt.gpu}, attempt {attempt})',flush=True)
        with (attempts/f'{attempt:03d}.log').open('w') as log:
            proc = subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in proc.stdout: print(line,end='',flush=True); log.write(line); log.flush()
                if proc.wait(): raise RuntimeError(f'{name} failed; no later settings launched. See {log.name}')
            except BaseException:
                proc.terminate(); raise
        required = [folder/f'preds_for_eval_{i}.text' for i in range(opt.epochs)]
        if not all(f.is_file() for f in required): raise RuntimeError(f'Missing epoch results: {name}')
        summarize(output)
        (folder/'DONE.json').write_text(json.dumps({'epochs':opt.epochs,'prediction_sha256':signature['prediction_sha256']})+'\n')
    print(f'ALL DONE: {output}/epoch_metrics.csv',flush=True)

if __name__ == '__main__': main()
