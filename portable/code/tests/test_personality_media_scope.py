import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_utils.data_utils import ModelArgs, Seq2SeqCollator  # noqa: E402


def _minimal_collator(scope="target"):
    collator = Seq2SeqCollator.__new__(Seq2SeqCollator)
    collator.personality_media_scope = scope
    collator.audio_context_window = 8
    collator.audio_embedding_source = "precomputed_tokens"
    collator.audio_enable = True
    collator.audio_direct_enable = True
    collator.video_enable = False
    collator.video_direct_enable = False
    collator.text_enable = True
    collator.audio_input_mode = "text_audio"
    collator.iemocap_speaker_ids_by_conv = {
        "Ses01F_impro01": [0, 1, 0, 1, 1, 0],
    }
    collator.iemocap_num_turns_by_conv = {
        "Ses01F_impro01": 6,
    }
    return collator


def test_model_args_preserves_personality_media_scope():
    args = ModelArgs()
    args.update({"personality_media_scope": "target"})
    assert args.personality_media_scope == "target"


def test_target_scope_selects_and_attaches_only_target_speaker_audio():
    collator = _minimal_collator("target")
    sample_id = "Personality_full_Ses01F_impro01_0"
    selected = collator._selected_qwen_turn_sample_ids(sample_id)
    assert selected == [
        "Ses01F_impro01_0",
        "Ses01F_impro01_2",
        "Ses01F_impro01_5",
    ]

    _, audio_refs, _ = collator._build_qwen_content_items(
        prompt_text=(
            'Given the full conversation where target speaker turns are marked '
            'as <Speaker_0*>: <Speaker_0*: "Hello" Speaker_1: "Hi">'
        ),
        sample_id=sample_id,
        turn_sample_ids=selected,
    )
    assert [ref["sample_id"] for ref in audio_refs] == selected


def test_full_scope_selects_all_conversation_audio():
    collator = _minimal_collator("full")
    selected = collator._selected_qwen_turn_sample_ids(
        "Personality_full_Ses01F_impro01_0"
    )
    assert selected == [
        "Ses01F_impro01_0",
        "Ses01F_impro01_1",
        "Ses01F_impro01_2",
        "Ses01F_impro01_3",
        "Ses01F_impro01_4",
        "Ses01F_impro01_5",
    ]


if __name__ == "__main__":
    test_model_args_preserves_personality_media_scope()
    test_target_scope_selects_and_attaches_only_target_speaker_audio()
    test_full_scope_selects_all_conversation_audio()
    print("[OK] personality_media_scope target/full routing checks passed")
