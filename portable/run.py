"""Portable launcher. Default: CPU-only preflight. --execute explicitly starts work."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def training_parser():
    tree = ast.parse((ROOT/'code/main_new.py').read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'ArgumentParser':
                nodes.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 'parser' and f.attr == 'add_argument':
                nodes.append(node)
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<argument definitions>', 'exec'), scope)
    return scope['parser']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['iemocap', 'meld'], required=True)
    p.add_argument('--gpus', help='Explicit physical GPU IDs, e.g. 2,3')
    p.add_argument('--mode', choices=['train', 'eval'], default='train')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--check-imports', action='store_true')
    p.add_argument('--port', type=int, default=29613)
    opt = p.parse_args()
    os.chdir(ROOT)
    cfg = json.loads((ROOT/'configs'/f'{opt.dataset}.json').read_text())
    errors = []
    for key in ['model_name_or_path', 'audio_qwen_processor_path', 'data_dir',
                'audio_feature_pkl', 'audio_iemocap_pkl', 'checkpoint_dir']:
        if cfg.get(key):
            path = (ROOT/cfg[key]).resolve()
            if not path.exists():
                errors.append(f'{key}: missing {path}')
            cfg[key] = str(path)
    for split in ['train', 'valid', 'test']:
        path = Path(cfg['data_dir'])/f'{split}.json'
        if not path.is_file():
            errors.append(f'Missing split: {path}')
            continue
        try:
            with path.open(encoding='utf-8') as stream:
                count = sum(1 for line in stream if line.strip() and json.loads(line))
            print(f'{opt.dataset} {split}: {count} JSONL records')
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f'Invalid JSONL split {path}: {exc}')
    for folder in [ROOT/'adapters'/opt.dataset] + ([Path(cfg['checkpoint_dir'])] if cfg.get('checkpoint_dir') else []):
        if not any((folder/filename).is_file() for filename in ('trainable_model.safetensors', 'adapter_model.safetensors')):
            errors.append(f'Missing lightweight checkpoint in {folder}')
    model = Path(cfg['model_name_or_path'])
    index_path = model/'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        for shard in set(index['weight_map'].values()):
            if not (model/shard).is_file(): errors.append(f'Missing base-model shard: {shard}')
    else:
        errors.append(f'Missing base-model index: {index_path}')
    ids = opt.gpus.split(',') if opt.gpus else ['0']
    if len(set(ids)) != len(ids) or not all(x.isdigit() for x in ids):
        errors.append('GPU IDs must be unique nonnegative integers')
    global_batch = cfg['batch_size']
    # Keep original effective batch: 12 (IEMOCAP), 40 (MELD).
    if global_batch % len(ids):
        errors.append(f'Original global batch {global_batch} not divisible by {len(ids)} GPUs')
    accumulation = global_batch // len(ids)
    cfg.update(gradient_accumulation_steps=accumulation, num_train_epochs=opt.epochs,
               do_train=opt.mode == 'train', do_eval=True)
    if opt.mode == 'eval': cfg['checkpoint_dir'] = str(ROOT/'adapters'/opt.dataset)
    run_id = f'{opt.dataset}_{opt.mode}_{datetime.now():%Y%m%d_%H%M%S}'
    output = ROOT/'runs'/run_id
    cfg.update(output_dir=str(output), run_id=run_id,
               deepspeed_config=str(output/'deepspeed.json'))
    parser = training_parser()
    actions = {x.dest: x for x in parser._actions}
    argv = []
    ignored = []
    for key, value in cfg.items():
        if value is None: continue
        action = actions.get(key)
        if action is None:
            ignored.append(key)
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value: argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value: argv.append(flag)
        else: argv.extend([flag, str(value)])
    parser.parse_args(argv)
    # These are saved metadata, not accepted command-line options.
    allowed_ignored = {'model_type', 'max_seq_length', 'do_sample', 'preprocess_inputs', 'clip_norm', 'open_ended'}
    unexpected = set(ignored)-allowed_ignored
    if unexpected: errors.append(f'Unmapped config keys: {sorted(unexpected)}')
    print('Metadata-only fields:', ', '.join(ignored))
    cache_env = {**os.environ, 'PYTHONPATH': str(ROOT/'code'),
                 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HOME': str(ROOT/'.cache/huggingface'),
                 'TORCH_HOME': str(ROOT/'.cache/torch'),
                 'TORCH_EXTENSIONS_DIR': str(ROOT/'.cache/torch_extensions'),
                 'TRITON_CACHE_DIR': str(ROOT/'.cache/triton')}
    if opt.check_imports or opt.execute:
        subprocess.run([sys.executable, '-c',
            'import torch, torchaudio, transformers, peft, deepspeed, pandas, sklearn; '
            'from transformers import Qwen2_5OmniForConditionalGeneration; '
            'import data_utils.data_utils, personality_pair_utils, speaker_mood_recurrence, utterance_feature_prefix; '
            'print("Runtime imports OK; torch", torch.__version__)'],
            env=cache_env, check=True)
    if errors: raise SystemExit('\n'.join(errors))
    print(f'PREFLIGHT OK: local paths, data splits, weights, CLI; effective batch={global_batch}')
    if not opt.execute:
        print('No training or GPU allocation started. Add --gpus PHYSICAL_IDS --execute to launch.')
        return
    if not opt.gpus: p.error('--execute requires explicit --gpus')
    output.mkdir(parents=True, exist_ok=False)
    ds = {'train_batch_size': global_batch, 'train_micro_batch_size_per_gpu': 1,
          'gradient_accumulation_steps': accumulation, 'gradient_clipping': 1.0,
          'bf16': {'enabled': True}, 'zero_optimization': {'stage': 2,
          'allgather_partitions': True, 'allgather_bucket_size': 500000000,
          'overlap_comm': True, 'reduce_scatter': True,
          'reduce_bucket_size': 500000000, 'contiguous_gradients': True}}
    (output/'deepspeed.json').write_text(json.dumps(ds, indent=2)+'\n')
    (output/'launch_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    env = cache_env
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(HF_HOME=str(ROOT/'.cache/huggingface'), TORCH_HOME=str(ROOT/'.cache/torch'),
               TORCH_EXTENSIONS_DIR=str(ROOT/'.cache/torch_extensions'),
               TRITON_CACHE_DIR=str(ROOT/'.cache/triton'), PYTHONUNBUFFERED='1')
    command = [sys.executable, '-m', 'deepspeed.launcher.runner', '--include',
               'localhost:'+opt.gpus, '--master_port', str(opt.port), 'code/main_new.py']+argv
    print(shlex.join(command), flush=True)
    subprocess.run(command, env=env, check=True)

if __name__ == '__main__': main()
