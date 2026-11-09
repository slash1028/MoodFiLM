"""Build provisional full-scene speaker personality prompts for MSP-IMPROV."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTRUCTION = (
    'Predict the OCEAN personality vector [O, C, E, A, N] of this speaker. '
    'Use the Big Five meanings: O=Openness, C=Conscientiousness, '
    'E=Extraversion, A=Agreeableness, N=Neuroticism. Use a 1.0-7.0 Likert '
    'scale where 1.0 is very low, 4.0 is medium, and 7.0 is very high. '
    'Output only a valid JSON array with exactly 5 numbers in this exact order.'
)


def prepare():
    source = ROOT / 'provisional_dialogues.jsonl'
    if not source.is_file():
        raise FileNotFoundError(source)
    data = ROOT / 'stage1_input'
    data.mkdir(exist_ok=True)
    rows, mapping = [], []
    for index, line in enumerate(source.open(encoding='utf-8')):
        dialogue = json.loads(line)
        turns = dialogue['turns']
        if not turns:
            continue
        speakers = sorted({t['speaker_from_readme'] for t in turns})
        for speaker in speakers:
            alias = {s: f'Speaker_{i}' for i, s in enumerate(speakers)}
            rendered = []
            for turn in turns:
                name = alias[turn['speaker_from_readme']]
                mark = '*' if turn['speaker_from_readme'] == speaker else ''
                rendered.append(f'{name}{mark}: {json.dumps(turn["text"], ensure_ascii=False)} [AUDIO_UTTERANCE]')
            sample_id = f'MELDPersonality_{dialogue["split"]}_dia{index}_{speaker}'
            rows.append({
                'input': 'Now you are expert of personality analysis. Given the full conversation '
                         f'where target speaker turns are marked as <{alias[speaker]}*>: '
                         f'<{" ".join(rendered)}> {INSTRUCTION}',
                'target': '[4.0, 4.0, 4.0, 4.0, 4.0]',
                'sample_id': sample_id,
                'audio_turn_sample_ids': [t['id'] for t in turns],
                'split': dialogue['split'], 'dialogue_id': dialogue['dialogue_id'],
                'speaker': speaker, 'personality_labels_available': False,
                'protocol': dialogue['protocol'],
            })
            mapping.append({'sample_id': sample_id, 'dialogue_id': dialogue['dialogue_id'],
                            'speaker': speaker, 'split': dialogue['split']})
    (data / 'test.json').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    for split in ('train', 'valid'):
        (data / f'{split}.json').write_text('', encoding='utf-8')
    (ROOT / 'stage1_mapping.json').write_text(json.dumps(mapping, indent=2)+'\n')
    print(f'Prepared {len(rows)} speaker-dialogue predictions from {len(mapping)} IDs')
    return rows


if __name__ == '__main__':
    prepare()
