import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils.data_utils import (  # noqa: E402
    Seq2SeqCollator,
    strip_audio_markers_for_text_only,
)


def _collator(window=1, mode="text_audio"):
    collator = Seq2SeqCollator.__new__(Seq2SeqCollator)
    collator.audio_context_window = window
    collator.audio_embedding_source = "precomputed_tokens"
    collator.audio_enable = True
    collator.audio_direct_enable = True
    collator.video_enable = False
    collator.video_direct_enable = False
    collator.audio_input_mode = mode
    collator.text_enable = mode != "audio_only"
    return collator


def _prompt():
    return (
        "Classify the emotion. ### \t Speaker_0:\"Hello\" [AUDIO_UTTERANCE]"
        " \t Speaker_1:\"Goodbye\" [AUDIO_UTTERANCE] ### "
        "Select the label of < Speaker_1:\"Goodbye\">:"
    )


def _record():
    return {
        "sample_id": "train_dia7_utt1",
        "meta": {"audio_turn_sample_ids": ["train_dia7_utt0", "train_dia7_utt1"]},
    }


def test_meld_metadata_respects_audio_context_window():
    assert _collator(1)._effective_qwen_turn_sample_ids(_record()) == ["train_dia7_utt1"]
    assert _collator(2)._effective_qwen_turn_sample_ids(_record()) == [
        "train_dia7_utt0", "train_dia7_utt1"
    ]


def test_meld_target_audio_is_attached_after_target_utterance():
    collator = _collator()
    items, refs, _ = collator._build_qwen_content_items(
        _prompt(), "train_dia7_utt1", ["train_dia7_utt1"]
    )
    assert [ref["sample_id"] for ref in refs] == ["train_dia7_utt1"]
    assert len([item for item in items if item["text"] == "<|AUDIO_UTTERANCE|>"]) == 1
    assert 'Speaker_1:"Goodbye" [AUDIO_UTTERANCE]' in items[0]["text"]


def test_meld_audio_only_ignores_markers_inside_masked_quotes():
    collator = _collator(mode="audio_only")
    items, refs, _ = collator._build_qwen_content_items(
        _prompt(), "train_dia7_utt1", ["train_dia7_utt1"]
    )
    assert [ref["sample_id"] for ref in refs] == ["train_dia7_utt1"]
    assert 'Speaker_1:"[AUDIO_UTTERANCE]" [AUDIO_UTTERANCE]' in items[0]["text"]


def test_text_only_does_not_read_audio_placeholder_words():
    assert "[AUDIO_UTTERANCE]" not in strip_audio_markers_for_text_only(_prompt())
