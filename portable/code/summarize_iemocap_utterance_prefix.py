"""Summarize completed IEMOCAP utterance-prefix settings without selecting by test."""

import argparse
import json
from pathlib import Path


def parse(path):
    try:
        return json.JSONDecoder().raw_decode(path.read_text(encoding='utf-8', errors='ignore'))[0]
    except (OSError, ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print('setting\tepoch\tCCC\tMAE\tWF1\tACC')
    for setting in sorted(args.root.glob('[0-9][0-9]_*')):
        rows = []
        for path in setting.glob('preds_for_eval_*.text'):
            report = parse(path)
            if report is None:
                continue
            try:
                epoch = int(path.stem.rsplit('_', 1)[-1]) + 1
            except ValueError:
                epoch = '?'
            personality = report.get('personality_continuous') or {}
            emotion = ((report.get('emotion') or {}).get('score') or {})
            rows.append((epoch, personality.get('avg_ccc'), personality.get('avg_mae'),
                         emotion.get('F1_SA'), emotion.get('Acc_SA')))
        for row in sorted(rows, key=lambda value: value[0] if isinstance(value[0], int) else 999):
            print('\t'.join(str(value) if value is not None else '-' for value in (setting.name, *row)))


if __name__ == '__main__':
    main()
