#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

BASE = Path('/n/work6/yiwang/MoodFilm/portable/code/experiments')


def detect_setting(run_dir: Path) -> str:
    audio_enable = False
    audio_mode = ''
    for mp in run_dir.rglob('model_args.json'):
        try:
            data = json.loads(mp.read_text(encoding='utf-8'))
        except Exception:
            continue
        ae = str(data.get('audio_enable', 'False')).strip().lower()
        am = str(data.get('audio_input_mode', '')).strip().lower()
        if am:
            audio_mode = am
        if ae in ('true', '1'):
            audio_enable = True
            break

    if not audio_enable:
        return 'text-only'
    if audio_mode == 'audio_only':
        return 'audio-only'
    if audio_mode == 'text_audio':
        return 'audio-text'
    if audio_mode == 'text_only':
        return 'text-only'
    return f'audio-{audio_mode or "unknown"}'


def move_or_merge_run(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        for item in list(src.iterdir()):
            target_item = dst / item.name
            if target_item.exists():
                suffix_idx = 1
                alt = dst / f'{item.name}__legacy'
                while alt.exists():
                    suffix_idx += 1
                    alt = dst / f'{item.name}__legacy{suffix_idx}'
                shutil.move(str(item), str(alt))
            else:
                shutil.move(str(item), str(target_item))
        src.rmdir()
    else:
        shutil.move(str(src), str(dst))


def merge_ablation(dataset: str, run_name: str, target_run_dir: Path) -> bool:
    ablation_run = BASE / 'ablation' / dataset / 'runs' / run_name
    if not (ablation_run.exists() and ablation_run.is_dir()):
        return False

    for item in list(ablation_run.iterdir()):
        dst = target_run_dir / item.name
        if dst.exists():
            idx = 1
            alt = target_run_dir / f'data_{item.name}'
            while alt.exists():
                idx += 1
                alt = target_run_dir / f'data_{item.name}_{idx}'
            shutil.move(str(item), str(alt))
        else:
            shutil.move(str(item), str(dst))

    ablation_run.rmdir()
    runs_parent = ablation_run.parent
    if runs_parent.exists() and not any(runs_parent.iterdir()):
        runs_parent.rmdir()
    return True


def main() -> None:
    legacy_runs = [p for p in BASE.glob('*/*/*/runs/run_*') if p.is_dir()]

    migrated = 0
    merged_ablation = 0
    errors = 0

    for run_dir in sorted(legacy_runs):
        try:
            parts = run_dir.parts
            exp_idx = parts.index('experiments')
            model_name, exp_setting, dataset = parts[exp_idx + 1], parts[exp_idx + 2], parts[exp_idx + 3]
            setting = detect_setting(run_dir)
            target = BASE / model_name / exp_setting / dataset / 'runs' / setting / run_dir.name

            move_or_merge_run(run_dir, target)
            if merge_ablation(dataset, target.name, target):
                merged_ablation += 1
            migrated += 1
        except Exception as exc:
            errors += 1
            print(f'ERROR: {run_dir} -> {exc}')

    print(f'legacy_before={len(legacy_runs)} migrated={migrated} merged_ablation={merged_ablation} errors={errors}')


if __name__ == '__main__':
    main()
