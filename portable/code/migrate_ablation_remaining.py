#!/usr/bin/env python3
import shutil
from pathlib import Path

BASE = Path('/n/work6/yiwang/MoodFilm/portable/code/experiments')


def build_model_run_map():
    run_map = {}
    for p in BASE.glob('*/*/*/runs/*/run_*'):
        if not p.is_dir():
            continue
        # skip ablation tree
        if '/experiments/ablation/' in str(p):
            continue
        run_map[p.name] = p
    return run_map


def merge_dir_contents(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in list(src.iterdir()):
        target = dst / item.name
        if target.exists():
            idx = 1
            alt = dst / f'data_{item.name}'
            while alt.exists():
                idx += 1
                alt = dst / f'data_{item.name}_{idx}'
            shutil.move(str(item), str(alt))
        else:
            shutil.move(str(item), str(target))
    src.rmdir()


def main():
    run_map = build_model_run_map()
    ablation_runs = [p for p in BASE.glob('ablation/*/runs/run_*') if p.is_dir()]

    matched_exact = 0
    matched_base = 0
    unmatched = []

    for src in sorted(ablation_runs):
        name = src.name
        dst = run_map.get(name)

        if dst is not None:
            merge_dir_contents(src, dst)
            matched_exact += 1
            continue

        if '_stage' in name:
            base_name = name.split('_stage', 1)[0]
            base_dst = run_map.get(base_name)
            if base_dst is not None:
                extra_root = base_dst / 'ablation_extra'
                extra_root.mkdir(parents=True, exist_ok=True)
                target = extra_root / name
                if target.exists():
                    suffix = 1
                    while (extra_root / f'{name}_{suffix}').exists():
                        suffix += 1
                    target = extra_root / f'{name}_{suffix}'
                shutil.move(str(src), str(target))
                matched_base += 1
                continue

        unmatched.append(str(src))

    print(f'ablation_total={len(ablation_runs)} matched_exact={matched_exact} matched_base={matched_base} unmatched={len(unmatched)}')
    if unmatched:
        print('--- unmatched ablation runs ---')
        for p in unmatched:
            print(p)


if __name__ == '__main__':
    main()
