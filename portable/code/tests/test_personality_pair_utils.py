import sys
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from data_utils.data_utils import (  # noqa: E402
    PERSONALITY_SPEAKER_0_TOKEN,
    PERSONALITY_SPEAKER_1_TOKEN,
    _add_personality_pair_fields,
)
from personality_pair_utils import (  # noqa: E402
    build_personality_pair_report,
    compute_personality_pair_loss,
)


class _Tokenizer:
    token_ids = {
        PERSONALITY_SPEAKER_0_TOKEN: 101,
        PERSONALITY_SPEAKER_1_TOKEN: 102,
    }

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]


def _pair_tensors(gold_difference, predicted_difference):
    targets = torch.full((1, 2, 5), 4.0)
    predictions = torch.full((1, 2, 5), 4.0)
    targets[:, 0, :] += float(gold_difference) / 2.0
    targets[:, 1, :] -= float(gold_difference) / 2.0
    predictions[:, 0, :] += float(predicted_difference) / 2.0
    predictions[:, 1, :] -= float(predicted_difference) / 2.0
    predictions.requires_grad_(True)
    return predictions, targets


def test_pair_collator_fields_and_lm_mask():
    batch = {
        "input_ids": torch.tensor(
            [[1, 101, 2, 102], [3, 4, 5, 6]],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [[-100, -100, 7, 8], [-100, -100, 9, 10]],
            dtype=torch.long,
        ),
        "type_token_ids": torch.tensor(
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            dtype=torch.long,
        ),
    }
    metadata = [
        {"personality_pair_targets": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]},
        {},
    ]
    result = _add_personality_pair_fields(
        batch,
        metadata,
        _Tokenizer(),
        SimpleNamespace(personality_pair_mode="regression"),
    )
    assert result["personality_pair_targets"].shape == (2, 2, 5)
    assert result["personality_pair_query_positions"].tolist() == [
        [1, 3],
        [-1, -1],
    ]
    assert result["personality_pair_mask"].tolist() == [True, False]
    assert torch.all(result["labels"][0].eq(-100))
    assert result["labels"][1, -1].item() == 10


def test_ranknet_gradient_orders_non_similar_pair():
    predictions, targets = _pair_tensors(1.0, -0.2)
    losses = compute_personality_pair_loss(
        predictions,
        targets,
        mode="ranknet",
        auxiliary_weight=1.0,
    )
    losses["auxiliary"].backward()
    assert predictions.grad[0, 0, 0].item() < 0
    assert predictions.grad[0, 1, 0].item() > 0


def test_threshold_boundary_is_similar():
    predictions, targets = _pair_tensors(0.5, 1.0)
    ranknet = compute_personality_pair_loss(
        predictions,
        targets,
        mode="ranknet",
    )
    similarity = compute_personality_pair_loss(
        predictions,
        targets,
        mode="similarity_rank",
    )
    assert ranknet["auxiliary"].item() == 0.0
    assert similarity["auxiliary"].item() > 0.0


def test_report_bins_and_metrics():
    targets = torch.tensor(
        [
            [[4.0] * 5, [4.0] * 5],
            [[4.75] * 5, [4.0] * 5],
            [[5.25] * 5, [4.0] * 5],
            [[6.0] * 5, [4.0] * 5],
        ]
    )
    report = build_personality_pair_report(
        targets.numpy(),
        targets.numpy(),
        threshold=0.5,
    )
    bins = report["pairwise"]["gold_difference_bins"]
    assert [bins[name]["count"] for name in bins] == [5, 5, 5, 5]
    assert report["pairwise"]["accuracy"] == 100.0
    assert report["continuous"]["avg_mse"] == 0.0
