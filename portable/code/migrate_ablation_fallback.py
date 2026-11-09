#!/usr/bin/env python3
import shutil
from pathlib import Path

base = Path('/n/work6/yiwang/MoodFilm/portable/code/experiments')
unmatched = [p for p in base.glob('ablation/*/runs/run_*') if p.is_dir()]

moved = 0
for src in sorted(unmatched):
    dataset = src.parts[src.parts.index('ablation') + 1]
    dst = base / 'LLaMA3.1_8B_hf' / 'lora' / dataset / 'runs' / 'text-only' / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        for item in list(src.iterdir()):
            target = dst / item.name
            if target.exists():
                i = 1
                alt = dst / f'data_{item.name}'
                while alt.exists():
                    i += 1
                    alt = dst / f'data_{item.name}_{i}'
                shutil.move(str(item), str(alt))
            else:
                shutil.move(str(item), str(target))
        src.rmdir()
    else:
        shutil.move(str(src), str(dst))
    moved += 1

print(f'fallback_moved={moved}')
