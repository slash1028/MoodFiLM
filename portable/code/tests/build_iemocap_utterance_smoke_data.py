"""Make tiny, recoverable train/eval subsets for a real 7B forward smoke test."""

import argparse
import json
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def choose(rows, stage):
    if stage == 'stage2':
        by_label = {}
        for row in rows:
            suffix = str(row['sample_id']).rsplit('_', 1)[-1]
            if suffix.isdigit() and int(suffix) >= 12:
                by_label.setdefault(str(row['target']), row)
        if len(by_label) < 6:
            raise ValueError('Smoke split lacks all six IEMOCAP ERC labels')
        return list(by_label.values())[:6]
    personalities = [row for row in rows if str(row['sample_id']).startswith('Personality_full_')
                     and not str(row['target']).startswith('TraitCompare_')]
    speakers = [row for row in rows if str(row['sample_id']).startswith('Speaker_')]
    if len(personalities) < 2 or len(speakers) < 2:
        raise ValueError('Smoke split lacks two personality and two speaker rows')
    return personalities[:2] + speakers[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('stage1', 'stage2'), required=True)
    args = parser.parse_args()
    if args.out.exists():
        for split in ('train', 'valid', 'test'):
            if not (args.out / f'{split}.json').is_file():
                raise FileExistsError(f'Partial smoke dataset: {args.out}')
        return
    args.out.mkdir(parents=True)
    for split in ('train', 'valid', 'test'):
        rows = choose(read(args.source / f'{split}.json'), args.stage)
        if not rows:
            raise ValueError(f'No {args.stage} smoke rows for {split}')
        with (args.out / f'{split}.json').open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'[SMOKE DATA] {args.out}', flush=True)


if __name__ == '__main__':
    main()
