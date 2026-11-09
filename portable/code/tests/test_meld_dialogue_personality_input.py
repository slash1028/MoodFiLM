import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meld_utils.build_meld_dialogue_personality_eval import build_row

def test_anonymous_full_dialogue_markers():
    turns = [{'utterance_id': 0, 'speaker': 'Ross', 'utterance': 'Hi'},
             {'utterance_id': 1, 'speaker': 'Rachel', 'utterance': 'Hello'},
             {'utterance_id': 2, 'speaker': 'Ross', 'utterance': 'Bye'}]
    row = build_row('test', 7, turns, 'Rachel', True, True)
    assert row['audio_turn_sample_ids'] == ['test_dia7_utt0', 'test_dia7_utt1', 'test_dia7_utt2']
    assert row['input'].count('[AUDIO_UTTERANCE]') == 3
    assert 'Speaker_1*: "Hello"' in row['input']
    assert 'Ross' not in row['input'] and 'Rachel' not in row['input']
    assert not row['personality_labels_available']

def test_legacy_default_keeps_no_markers():
    row = build_row('train', 0, [{'utterance_id': 0, 'speaker': 'Ross', 'utterance': 'Hi'}], 'Ross')
    assert '[AUDIO_UTTERANCE]' not in row['input']
