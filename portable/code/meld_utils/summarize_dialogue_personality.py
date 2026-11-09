"""Descriptive dispersion only; MELD personality predictions have no GT here."""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

TRAITS = ('O', 'C', 'E', 'A', 'N')

def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['speaker'])].append(row)
    output = {}
    for (split, speaker), items in sorted(groups.items()):
        output.setdefault(split, {})[speaker] = {
            'num_dialogues': len(items),
            'traits': {t: {
                'mean': st.mean(float(x[t]) for x in items),
                'std_across_dialogues': st.stdev(float(x[t]) for x in items) if len(items)>1 else None,
                'min': min(float(x[t]) for x in items),
                'max': max(float(x[t]) for x in items),
            } for t in TRAITS},
        }
    return {'by_split_and_speaker': output,
            'note': 'Dispersion is not accuracy. No ground-truth personality labels; no CCC/MAE computed.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with args.csv.open(newline='', encoding='utf-8') as f:
        result = summarize(list(csv.DictReader(f)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(args.output)
