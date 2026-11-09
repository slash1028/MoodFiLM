import sys
import ast
import inspect
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from speaker_mood_recurrence import (  # noqa: E402
    ORDERED_DATASET_INDEX,
    ORDERED_PADDING,
    DialogueOrderedBatchSampler,
    OrderedDatasetView,
    detached_emotion_entries,
    detached_soft_emotion_entries,
    detached_state_entries,
    dialogue_turn_key,
    lookup_previous_emotions,
    lookup_previous_soft_emotions,
    lookup_previous_moods,
    mask_ordered_sample_weights,
    ordered_real_fraction,
    select_recurrent_hidden,
    validate_dialogue_parallel_batch,
)
from data_utils.data_utils import (  # noqa: E402
    _add_personality_prefix_fields,
    _add_source_specific_personality_contexts,
    _add_speaker_mood_dynamics_fields,
)


class _Dataset:
    def __init__(self, sample_ids):
        self.examples = [
            ["prompt", "label", sample_id, {"sample_id": sample_id}]
            for sample_id in sample_ids
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _personality_source_rows():
    return [
        {
            "sample_id": "dialogue_0",
            "speaker_id": "Speaker_A",
            "personality": [1.0, 2.0, 3.0, 4.0, 5.0],
            "gt_personality": [7.0, 6.0, 5.0, 4.0, 3.0],
            "interlocutor_personality": [2.0, 2.0, 2.0, 2.0, 2.0],
        },
        {
            "sample_id": "dialogue_1",
            "speaker_id": "Speaker_B",
            "personality": [2.0, 3.0, 4.0, 5.0, 6.0],
            "gt_personality": [6.0, 5.0, 4.0, 3.0, 2.0],
            "interlocutor_personality": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
    ]


def test_source_specific_contexts_derive_all_gt_without_changing_predicted():
    original = _personality_source_rows()
    enriched = _add_source_specific_personality_contexts(original)

    assert enriched[0]["speaker_context_speakers"] == ["Speaker_B"]
    assert enriched[0]["speaker_context_personalities"] == [
        original[1]["personality"]
    ]
    assert enriched[0]["speaker_context_gt_personalities"] == [
        original[1]["gt_personality"]
    ]
    assert enriched[0]["gt_interlocutor_personality"] == original[1][
        "gt_personality"
    ]
    # Existing predicted interlocutor metadata remains byte-for-byte unchanged.
    assert enriched[0]["interlocutor_personality"] == original[0][
        "interlocutor_personality"
    ]


def test_speaker_mood_gt_source_uses_gt_target_and_gt_context_only():
    metadata = _add_source_specific_personality_contexts(
        _personality_source_rows()
    )[:1]
    gt = _add_speaker_mood_dynamics_fields(
        batch_dict={},
        targets=["neutral"],
        metadata=metadata,
        args=SimpleNamespace(
            use_speaker_mood_dynamics=True,
            speaker_mood_source="gt",
            personality_prefix_source="auto",
            dataset="iemocap",
        ),
    )
    predicted = _add_speaker_mood_dynamics_fields(
        batch_dict={},
        targets=["neutral"],
        metadata=metadata,
        args=SimpleNamespace(
            use_speaker_mood_dynamics=True,
            speaker_mood_source="predicted",
            personality_prefix_source="auto",
            dataset="iemocap",
        ),
    )

    expected_gt_target = (torch.tensor(metadata[0]["gt_personality"]) - 4.0) / 3.0
    expected_gt_context = (
        torch.tensor(metadata[0]["speaker_context_gt_personalities"][0]) - 4.0
    ) / 3.0
    expected_pred_target = (torch.tensor(metadata[0]["personality"]) - 4.0) / 3.0
    expected_pred_context = (
        torch.tensor(metadata[0]["speaker_context_personalities"][0]) - 4.0
    ) / 3.0

    assert torch.allclose(gt["speaker_mood_target_personality"][0], expected_gt_target)
    assert torch.allclose(
        gt["speaker_mood_context_personalities"][0, 0], expected_gt_context
    )
    assert torch.allclose(
        predicted["speaker_mood_target_personality"][0], expected_pred_target
    )
    assert torch.allclose(
        predicted["speaker_mood_context_personalities"][0, 0],
        expected_pred_context,
    )
    assert not torch.allclose(
        gt["speaker_mood_context_personalities"],
        predicted["speaker_mood_context_personalities"],
    )


def test_speaker_mood_gt_source_never_falls_back_to_predicted_profiles():
    metadata = [
        {
            "sample_id": "dialogue_0",
            "speaker_id": "Speaker_A",
            "personality": [7.0] * 5,
            "gt_personality": None,
            "interlocutor_personality": [6.0] * 5,
            "gt_interlocutor_personality": None,
            "speaker_context_speakers": ["Speaker_B"],
            "speaker_context_personalities": [[5.0] * 5],
            "speaker_context_gt_personalities": [None],
            "speaker_context_personality_known": [True],
            "speaker_context_gt_personality_known": [False],
        }
    ]
    result = _add_speaker_mood_dynamics_fields(
        batch_dict={},
        targets=["neutral"],
        metadata=metadata,
        args=SimpleNamespace(
            use_speaker_mood_dynamics=True,
            speaker_mood_source="gt",
            personality_prefix_source="auto",
            dataset="iemocap",
        ),
    )
    assert result["speaker_mood_target_mask"].tolist() == [False]
    assert result["speaker_mood_target_personality"].abs().sum().item() == 0.0
    assert result["speaker_mood_context_personalities"].abs().sum().item() == 0.0
    assert result["speaker_mood_relation_features"][0, 0, 3].item() == 0.0


def test_personality_prefix_gt_source_does_not_reuse_predicted_interlocutor():
    metadata = _add_source_specific_personality_contexts(
        _personality_source_rows()
    )[:1]
    result = _add_personality_prefix_fields(
        {},
        metadata,
        SimpleNamespace(
            enable_personality_prefix=True,
            enable_latent_mood_prefix=False,
            enable_personality_mood_prefix=False,
            personality_prefix_source="gt",
            personality_prefix_mode="target_interlocutor",
        ),
    )
    expected = (
        torch.tensor(metadata[0]["gt_interlocutor_personality"]) - 4.0
    ) / 3.0
    predicted = (
        torch.tensor(metadata[0]["interlocutor_personality"]) - 4.0
    ) / 3.0
    assert result["personality_prefix_mask"].tolist() == [True]
    assert torch.allclose(result["personality_prefix_interlocutor"][0], expected)
    assert not torch.allclose(
        result["personality_prefix_interlocutor"][0], predicted
    )


def test_dialogue_turn_key_supports_both_datasets():
    assert dialogue_turn_key("Ses03F_impro06_12") == ("Ses03F_impro06", 12)
    assert dialogue_turn_key("train_dia17_utt4") == ("train_dia17", 4)


def test_ordered_dataset_view_marks_padding_without_mutation():
    dataset = _Dataset(["dia0_0"])
    view = OrderedDatasetView(dataset)
    item = view[(0, True)]
    assert item[-1][ORDERED_DATASET_INDEX] == 0
    assert item[-1][ORDERED_PADDING] is True
    assert ORDERED_PADDING not in dataset[0][-1]


def test_dialogues_are_rank_isolated_and_chronological():
    # Deliberately shuffled dataset order.
    sample_ids = [
        "A_2",
        "B_1",
        "C_3",
        "A_0",
        "D_0",
        "C_0",
        "E_2",
        "B_0",
        "A_1",
        "C_2",
        "E_0",
        "C_1",
        "E_1",
    ]
    dataset = _Dataset(sample_ids)
    samplers = [
        DialogueOrderedBatchSampler(
            dataset,
            local_batch_size=2,
            num_replicas=2,
            rank=rank,
            shuffle_dialogues=False,
            seed=7,
        )
        for rank in range(2)
    ]

    assert len(samplers[0]) == len(samplers[1])
    seen_indices = []
    dialogue_ranks = defaultdict(set)
    observed_turns = defaultdict(list)
    padding_count = 0

    for rank, sampler in enumerate(samplers):
        for batch in sampler:
            assert len(batch) == 2
            real_dialogues = []
            for dataset_index, is_padding in batch:
                if is_padding:
                    padding_count += 1
                    continue
                sample_id = sample_ids[dataset_index]
                dialogue_id, turn_index = dialogue_turn_key(sample_id)
                real_dialogues.append(dialogue_id)
                dialogue_ranks[dialogue_id].add(rank)
                observed_turns[dialogue_id].append(turn_index)
                seen_indices.append(dataset_index)
            assert len(real_dialogues) == len(set(real_dialogues))

    assert sorted(seen_indices) == list(range(len(dataset)))
    assert padding_count > 0
    assert all(len(ranks) == 1 for ranks in dialogue_ranks.values())
    assert all(turns == sorted(turns) for turns in observed_turns.values())


def test_previous_mood_lookup_and_hidden_selection_are_detached():
    reference = torch.zeros(3, 5, dtype=torch.float32)
    cached = torch.tensor([1.0, 2.0], requires_grad=True)
    previous, valid, expected = lookup_previous_moods(
        reference=reference,
        mood_dim=2,
        previous_sample_ids=["", "utt0", "missing"],
        state_cache={"utt0": cached},
    )
    assert valid.tolist() == [False, True, False]
    assert expected.tolist() == [False, True, True]
    assert previous[1].tolist() == [1.0, 2.0]
    assert previous.requires_grad is False

    initial = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    hidden = select_recurrent_hidden(initial, previous, valid)
    assert torch.allclose(hidden[0], initial[0])
    assert torch.allclose(hidden[1], previous[1])
    assert torch.allclose(hidden[2], initial[2])


def test_detached_state_entries_skip_invalid_and_padding_rows():
    mood_after = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    entries = detached_state_entries(
        mood_after=mood_after,
        sample_ids=["utt0", "utt1", "utt2"],
        valid_mask=torch.tensor([True, False, True]),
        padding_mask=torch.tensor([False, False, True]),
    )
    assert [key for key, _ in entries] == ["utt0"]
    assert entries[0][1].tolist() == [1.0, 2.0]
    assert entries[0][1].requires_grad is False


def test_ordered_padding_has_zero_objective_weight_and_fixed_batch_scale():
    weights = torch.tensor([1.0, 1.0])
    padding = torch.tensor([False, True])
    masked = mask_ordered_sample_weights(weights, padding)
    fraction = ordered_real_fraction(weights.sum(), padding)
    assert masked.tolist() == [1.0, 0.0]
    assert fraction.item() == 0.5


def test_ordered_padding_is_masked_by_speaker_mood_collator_fields():
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    batch = {"labels": labels.clone()}
    metadata = [
        {
            "sample_id": "A_0",
            "speaker_id": "A",
            "personality": [3.0] * 5,
            ORDERED_DATASET_INDEX: 7,
            ORDERED_PADDING: False,
        },
        {
            "sample_id": "B_0",
            "speaker_id": "B",
            "personality": [4.0] * 5,
            ORDERED_DATASET_INDEX: 8,
            ORDERED_PADDING: True,
        },
    ]
    result = _add_speaker_mood_dynamics_fields(
        batch_dict=batch,
        targets=["neutral", "sad"],
        metadata=metadata,
        args=SimpleNamespace(
            use_speaker_mood_dynamics=True,
            speaker_mood_source="auto",
            personality_prefix_source="auto",
            dataset="iemocap",
        ),
    )
    assert result["speaker_mood_dataset_index"].tolist() == [7, 8]
    assert result["speaker_mood_order_padding"].tolist() == [False, True]
    assert result["speaker_mood_target_mask"].tolist() == [True, True]
    assert torch.equal(result["labels"][0], labels[0])
    assert torch.equal(result["labels"][1], labels[1])


def test_predicted_emotion_lookup_is_causal_one_hot_and_detached():
    reference = torch.zeros(4, 3, dtype=torch.float32)
    cached = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
    previous, valid, expected = lookup_previous_emotions(
        reference=reference,
        num_emotions=3,
        previous_sample_ids=["", "utt0", "missing", "wrong_dim"],
        emotion_cache={
            "utt0": cached,
            "wrong_dim": torch.tensor([1.0, 0.0]),
        },
    )
    assert valid.tolist() == [False, True, False, False]
    assert expected.tolist() == [False, True, True, True]
    assert previous[1].tolist() == [0.0, 1.0, 0.0]
    assert previous.requires_grad is False


def test_soft_previous_state_is_detached_and_exact_for_both_label_spaces():
    for num_emotions, first_ids, second_ids in (
        (6, ["A_0", "B_0"], ["A_1", "B_1"]),
        (7, ["train_dia0_utt0", "train_dia1_utt0"], ["train_dia0_utt1", "train_dia1_utt1"]),
    ):
        speakers = ["Speaker_0", "Speaker_0"]
        reference = torch.zeros(2, 4)
        cache = {}
        previous, valid, expected, _ = lookup_previous_soft_emotions(
            reference=reference,
            num_emotions=num_emotions,
            sample_ids=first_ids,
            speaker_ids=speakers,
            previous_sample_ids=["", ""],
            state_cache=cache,
        )
        assert valid.tolist() == [False, False]
        assert expected.tolist() == [False, False]
        assert previous.requires_grad is False

        logits = torch.arange(
            2 * num_emotions, dtype=torch.float32, requires_grad=True
        ).view(2, num_emotions)
        for key, value in detached_soft_emotion_entries(
            final_label_logits=logits,
            sample_ids=first_ids,
            speaker_ids=speakers,
            num_emotions=num_emotions,
        ):
            cache[key] = value

        previous, valid, expected, _ = lookup_previous_soft_emotions(
            reference=reference,
            num_emotions=num_emotions,
            sample_ids=second_ids,
            speaker_ids=speakers,
            previous_sample_ids=first_ids,
            state_cache=cache,
        )
        assert valid.tolist() == [True, True]
        assert expected.tolist() == [True, True]
        assert previous.requires_grad is False
        assert torch.allclose(previous, torch.softmax(logits.detach(), dim=-1))


def test_soft_state_rejects_parallel_rows_from_one_dialogue():
    try:
        validate_dialogue_parallel_batch(
            sample_ids=["train_dia0_utt0", "train_dia0_utt1"],
            speaker_ids=["Speaker_0", "Speaker_1"],
        )
    except RuntimeError as exc:
        assert "more than once" in str(exc)
    else:
        raise AssertionError("Expected duplicate-dialogue batch rejection.")


def test_soft_state_rejects_non_chronological_cached_predecessor():
    cache = {
        ("train_dia0", "Speaker_0"): {
            "sample_id": "train_dia0_utt0",
            "probabilities": torch.tensor([0.7, 0.2, 0.1]),
        }
    }
    try:
        lookup_previous_soft_emotions(
            reference=torch.zeros(1, 2),
            num_emotions=3,
            sample_ids=["train_dia0_utt3"],
            speaker_ids=["Speaker_0"],
            previous_sample_ids=["train_dia0_utt2"],
            state_cache=cache,
        )
    except RuntimeError as exc:
        assert "not chronological" in str(exc)
    else:
        raise AssertionError("Expected stale predecessor rejection.")


def _load_simplified_head_class():
    source_path = CODE_DIR / "main_new.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "SimplifiedSpeakerMoodDynamicsHead"
    )
    namespace = {"torch": torch, "nn": torch.nn, "math": math}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["SimplifiedSpeakerMoodDynamicsHead"]


def _load_main_functions(*names):
    source_path = CODE_DIR / "main_new.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in set(names)
    ]
    namespace = {"torch": torch, "math": math, "hashlib": hashlib}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(source_path), "exec"), namespace)
    return [namespace[name] for name in names]


def test_simplified_mode_is_inactive_when_speaker_mood_is_disabled():
    is_true, architecture_enabled, active = _load_main_functions(
        "_is_true_flag",
        "_speaker_mood_simplified_enabled",
        "_speaker_mood_simplified_active",
    )
    args = SimpleNamespace(
        speaker_mood_architecture="simplified",
        use_speaker_mood_dynamics="False",
    )
    assert architecture_enabled(args)
    assert not active(args)
    args.use_speaker_mood_dynamics = "True"
    assert active(args)


def test_simplified_personality_center_accepts_raw_and_normalized_scales():
    head_class = _load_simplified_head_class()
    raw_midpoint = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        personality_center=4.0,
    )
    normalized_midpoint = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        personality_center=0.0,
    )
    raw_upper_bound = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        personality_center=7.0,
    )

    assert raw_midpoint.personality_center == 0.0
    assert normalized_midpoint.personality_center == 0.0
    assert raw_upper_bound.personality_center == 1.0


def test_simplified_interaction_attention_controls_are_exact():
    head_class = _load_simplified_head_class()
    common = {
        "input_embeds": torch.randn(1, 3, 8),
        "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
        "target_personality": torch.zeros(1, 5),
        "context_personalities": torch.zeros(1, 3, 5),
        "relation_features": torch.tensor(
            [[[0.5, 0.5, 0.0, 1.0], [0.25, 0.5, 1.0, 1.0], [0.0] * 4]]
        ),
        "context_mask": torch.tensor([[True, True, False]]),
    }
    expected = {
        "uniform": torch.tensor([[0.5, 0.5, 0.0]]),
        "recency": torch.tensor([[2.0 / 3.0, 1.0 / 3.0, 0.0]]),
        "previous": torch.tensor([[0.0, 1.0, 0.0]]),
        "nearest": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    for mode, expected_weights in expected.items():
        head = head_class(
            hidden_size=8,
            num_emotions=7,
            speaker_mood_dim=4,
            hidden_dim=6,
            dropout=0.0,
            interaction_attention_mode=mode,
        ).eval()
        with torch.no_grad():
            output = head(**common)
        assert torch.allclose(
            output["attention_weights"], expected_weights, atol=1e-6
        )


def test_simplified_interaction_gate_and_recency_bias():
    head_class = _load_simplified_head_class()
    common = {
        "input_embeds": torch.randn(1, 3, 8),
        "prompt_mask": torch.ones(1, 3, dtype=torch.bool),
        "target_personality": torch.zeros(1, 5),
        "context_personalities": torch.randn(1, 2, 5),
        "relation_features": torch.tensor(
            [[[0.5, 0.5, 0.0, 1.0], [0.25, 0.5, 1.0, 1.0]]]
        ),
        "context_mask": torch.tensor([[True, True]]),
    }
    gated = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        dropout=0.0,
        interaction_min_interlocutors=3,
    ).eval()
    disabled = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        dropout=0.0,
        use_interaction=False,
    ).eval()
    disabled.load_state_dict(gated.state_dict())
    with torch.no_grad():
        gated_output = gated(**common)
        disabled_output = disabled(**common)
    assert torch.allclose(gated_output["mood_after"], disabled_output["mood_after"])

    biased = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        dropout=0.0,
        interaction_recency_bias=1.0,
    ).eval()
    torch.nn.init.zeros_(biased.attention_query.weight)
    torch.nn.init.zeros_(biased.attention_query.bias)
    torch.nn.init.zeros_(biased.attention_key.weight)
    torch.nn.init.zeros_(biased.attention_key.bias)
    with torch.no_grad():
        biased_output = biased(**common)
    assert torch.allclose(
        biased_output["attention_weights"],
        torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]),
        atol=1e-6,
    )


def test_personality_noise_is_profile_stable_clipped_and_eval_scale_correct():
    stable_noise, apply_noise = _load_main_functions(
        "_stable_personality_noise",
        "_apply_speaker_mood_personality_noise",
    )
    target = torch.tensor([[0.0, 0.2, -0.2, 0.9, -0.9]])
    context = target.view(1, 1, 5).clone()
    target_used, context_used = apply_noise(
        target_personality=target,
        context_personalities=context,
        target_available=torch.tensor([True]),
        context_available=torch.tensor([[True]]),
        raw_ocean_std=1.0,
        seed=17,
    )
    # Identical profiles receive one stable perturbation regardless of role.
    assert torch.allclose(target_used[0], context_used[0, 0])
    assert target_used.min().item() >= -1.0
    assert target_used.max().item() <= 1.0
    expected = (target[0] + stable_noise(target[0], 1.0, 17)).clamp(-1.0, 1.0)
    assert torch.allclose(target_used[0], expected)

    normalized_target, _ = apply_noise(
        target,
        context,
        torch.tensor([True]),
        torch.tensor([[True]]),
        0.2,
        17,
        "normalized",
    )
    normalized_expected = (
        target[0] + stable_noise(target[0], 0.2, 17, "normalized")
    ).clamp(-1.0, 1.0)
    assert torch.allclose(normalized_target[0], normalized_expected)

    clean_target, clean_context = apply_noise(
        target, context, torch.tensor([True]), torch.tensor([[True]]), 0.0, 17
    )
    assert clean_target is target
    assert clean_context is context


def test_simplified_head_excludes_gold_target_embeddings_and_supports_6_7_classes():
    head_class = _load_simplified_head_class()
    for num_emotions in (6, 7):
        torch.manual_seed(7)
        head = head_class(
            hidden_size=8,
            num_emotions=num_emotions,
            speaker_mood_dim=4,
            hidden_dim=6,
            dropout=0.0,
        ).eval()
        prompt = torch.randn(2, 3, 8)
        gold_suffix_a = torch.randn(2, 1, 8)
        gold_suffix_b = torch.randn(2, 1, 8) * 100.0
        common = {
            "prompt_mask": torch.tensor([[True, True, True, False]] * 2),
            "target_personality": torch.randn(2, 5),
            "context_personalities": torch.randn(2, 2, 5),
            "relation_features": torch.ones(2, 2, 4),
            "context_mask": torch.tensor([[True, True], [True, False]]),
            "target_mask": torch.ones(2, dtype=torch.bool),
            "previous_emotion_probs": torch.softmax(
                torch.randn(2, num_emotions), dim=-1
            ),
            "previous_emotion_valid": torch.tensor([True, False]),
        }
        with torch.no_grad():
            output_a = head(
                input_embeds=torch.cat([prompt, gold_suffix_a], dim=1),
                **common,
            )
            output_b = head(
                input_embeds=torch.cat([prompt, gold_suffix_b], dim=1),
                **common,
            )
        assert output_a["transition_prior"].shape == (2, num_emotions)
        assert output_a["film_gamma_raw"].shape == (2, num_emotions)
        assert torch.allclose(output_a["mood_after"], output_b["mood_after"])
        assert torch.allclose(
            output_a["transition_prior"], output_b["transition_prior"]
        )
        assert torch.allclose(
            output_a["film_gamma_raw"], output_b["film_gamma_raw"]
        )
        assert "prev_emotion" not in inspect.signature(head.forward).parameters


def test_simplified_head_uses_unknown_personality_without_masking_real_target():
    head_class = _load_simplified_head_class()
    torch.manual_seed(11)
    head = head_class(
        hidden_size=8,
        num_emotions=7,
        speaker_mood_dim=4,
        hidden_dim=6,
        dropout=0.0,
    ).eval()
    common = {
        "input_embeds": torch.randn(2, 3, 8),
        "prompt_mask": torch.ones(2, 3, dtype=torch.bool),
        "context_personalities": torch.randn(2, 1, 5),
        "relation_features": torch.ones(2, 1, 4),
        "context_mask": torch.ones(2, 1, dtype=torch.bool),
        # Row 0 is a real MELD utterance with no mapped personality. Row 1 is
        # ordered padding and must remain excluded from outputs and caches.
        "target_mask": torch.tensor([True, False]),
        "target_personality_available": torch.tensor([False, False]),
        "previous_emotion_probs": torch.zeros(2, 7),
        "previous_emotion_valid": torch.tensor([False, False]),
    }
    with torch.no_grad():
        output_a = head(
            target_personality=torch.zeros(2, 5),
            **common,
        )
        output_b = head(
            target_personality=torch.full((2, 5), 99.0),
            **common,
        )

    assert output_a["target_mask"].tolist() == [True, False]
    assert torch.allclose(output_a["mood_after"][0], output_b["mood_after"][0])
    assert torch.allclose(
        output_a["transition_prior"][0], output_b["transition_prior"][0]
    )
    assert output_a["mood_after"][0].abs().sum().item() > 0.0
    assert output_a["mood_after"][1].abs().sum().item() == 0.0
    assert output_a["transition_prior"][1].abs().sum().item() == 0.0


def test_simplified_head_component_ablation_switches_preserve_output_contract():
    head_class = _load_simplified_head_class()
    common = {
        "input_embeds": torch.randn(2, 3, 8),
        "prompt_mask": torch.ones(2, 3, dtype=torch.bool),
        "target_personality": torch.randn(2, 5),
        "context_personalities": torch.randn(2, 2, 5),
        "relation_features": torch.ones(2, 2, 4),
        "context_mask": torch.tensor([[True, True], [True, False]]),
        "target_mask": torch.ones(2, dtype=torch.bool),
        "previous_emotion_probs": torch.softmax(torch.randn(2, 6), dim=-1),
        "previous_emotion_valid": torch.ones(2, dtype=torch.bool),
    }
    for disabled_flag in (
        "use_personality",
        "use_previous_emotion",
        "use_interaction",
        "use_current_evidence",
    ):
        flags = {
            "use_personality": True,
            "use_previous_emotion": True,
            "use_interaction": True,
            "use_current_evidence": True,
        }
        flags[disabled_flag] = False
        head = head_class(
            hidden_size=8,
            num_emotions=6,
            speaker_mood_dim=4,
            hidden_dim=6,
            dropout=0.0,
            **flags,
        ).eval()
        with torch.no_grad():
            output = head(**common)
        assert output["mood_after"].shape == (2, 4)
        assert output["transition_prior"].shape == (2, 6)
        assert output["film_gamma_raw"].shape == (2, 6)
        assert torch.isfinite(output["mood_after"]).all()


def test_simplified_film_matches_centered_formula_without_gate():
    (apply_modulation,) = _load_main_functions(
        "_apply_speaker_mood_logit_modulation"
    )
    logits = torch.tensor([[[3.0, 9.0, 2.0, 8.0, 1.0]]])
    prior = torch.tensor([[0.4, -0.2, 0.8]])
    gamma_raw = torch.tensor([[0.2, -0.4, 0.6]])
    adjusted = apply_modulation(
        logits=logits,
        transition_prior=prior,
        film_gamma_raw=gamma_raw,
        label_token_ids=[0, 2, 4],
        scale=0.25,
        modulation="film",
        film_norm="center",
        gamma_scale=0.5,
        beta_scale=1.5,
        use_gamma=True,
        use_beta=True,
        beta_gate="none",
    )
    base = logits[0, 0, [0, 2, 4]]
    centered = base - base.mean()
    delta_gamma = 0.5 * torch.tanh(gamma_raw[0])
    expected = base + 0.25 * (delta_gamma * centered + 1.5 * prior[0])
    assert torch.allclose(adjusted[0, 0, [0, 2, 4]], expected)
    assert adjusted[0, 0, 1].item() == logits[0, 0, 1].item()
    assert adjusted[0, 0, 3].item() == logits[0, 0, 3].item()


def test_first_target_cache_logits_do_not_depend_on_current_gold_token():
    _, extract = _load_main_functions(
        "_align_labels_to_logits",
        "_emotion_label_logits_at_first_target",
    )
    torch.manual_seed(11)
    embedding = torch.nn.Embedding(10, 5)
    output = torch.nn.Linear(5, 10, bias=False)

    def causal_logits(token_ids):
        states = embedding(token_ids).cumsum(dim=1)
        return output(states)

    prompt = torch.tensor([[4, 5]])
    full_a = torch.cat([prompt, torch.tensor([[1]])], dim=1)
    full_b = torch.cat([prompt, torch.tensor([[2]])], dim=1)
    labels_a = torch.tensor([[-100, -100, 1]])
    labels_b = torch.tensor([[-100, -100, 2]])
    selected_a = extract(causal_logits(full_a), labels_a, [1, 2, 3])
    selected_b = extract(causal_logits(full_b), labels_b, [1, 2, 3])
    prompt_only = causal_logits(prompt)[:, -1, [1, 2, 3]]
    assert torch.allclose(selected_a, prompt_only)
    assert torch.allclose(selected_b, prompt_only)


def test_generation_processor_captures_final_moodfilm_logits():
    source_path = CODE_DIR / "main_new.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "MoodDynamicsLogitsProcessor",
        "_apply_speaker_mood_logit_modulation",
        "_captured_simplified_generation_label_logits",
    }
    selected = [
        item
        for item in tree.body
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in wanted
    ]
    namespace = {
        "torch": torch,
        "math": math,
        "LogitsProcessor": object,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    processor = namespace["MoodDynamicsLogitsProcessor"](
        transition_prior=torch.tensor([[0.2, 0.5, -0.1], [0.1, -0.3, 0.4]]),
        label_token_ids=[0, 2, 4],
        scale=0.2,
        initial_length=3,
        film_gamma_raw=torch.tensor([[0.2, 0.1, -0.4], [0.3, -0.2, 0.1]]),
        modulation="film",
        capture_final_label_logits=True,
    )
    scores = torch.tensor(
        [[3.0, 8.0, 2.0, 7.0, 1.0], [1.0, 6.0, 2.0, 5.0, 3.0]]
    )
    adjusted = processor(torch.ones(2, 3, dtype=torch.long), scores)
    captured = namespace["_captured_simplified_generation_label_logits"](
        [processor], 2
    )
    assert torch.allclose(captured, adjusted[:, [0, 2, 4]].float())
    assert captured.requires_grad is False


def test_detached_emotion_entries_skip_invalid_and_padding_rows():
    entries = detached_emotion_entries(
        predicted_indices=torch.tensor([2, 1, -1, 0]),
        sample_ids=["utt0", "utt1", "utt2", "utt3"],
        num_emotions=3,
        valid_mask=torch.tensor([True, True, True, False]),
        padding_mask=torch.tensor([False, True, False, False]),
    )
    assert [key for key, _ in entries] == ["utt0"]
    assert entries[0][1].tolist() == [0.0, 0.0, 1.0]
    assert entries[0][1].requires_grad is False
