#!/usr/bin/env python
"""CPU smoke test for MoodFiLM diagnostic tables and figures."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "case_study"))

from analyze_moodfilm_diagnostics import (  # noqa: E402
    LABELS,
    analyze_noise,
    analyze_temporal,
    analyze_transitions,
    analyze_uncertainty,
    analyze_variance,
)


def synthetic_samples(dataset: str, mood_scale: float = 1.0) -> list[dict[str, object]]:
    labels = LABELS[dataset]
    samples: list[dict[str, object]] = []
    for dialogue_index in range(3):
        personality = [0.15 * (dialogue_index + 1), -0.2, 0.1, 0.3, -0.1]
        for turn_index in range(8):
            gold_index = (turn_index // 2 + dialogue_index) % len(labels)
            base = np.full(len(labels), 0.08, dtype=np.float64)
            base[gold_index] = 0.60
            base /= base.sum()
            final = base.copy()
            final[gold_index] += 0.08
            final /= final.sum()
            mood = np.asarray(
                [
                    dialogue_index + 0.2 * turn_index,
                    np.sin(turn_index),
                    np.cos(turn_index),
                    gold_index / max(1, len(labels) - 1),
                ],
                dtype=np.float64,
            )
            samples.append(
                {
                    "sample_id": f"dia{dialogue_index}_{turn_index}",
                    "dialogue_id": f"dia{dialogue_index}",
                    "turn_index": turn_index,
                    "speaker_id": f"speaker_{turn_index % 2}",
                    "profile_id": f"profile_{dialogue_index}",
                    "personality_available": True,
                    "gold_label": labels[gold_index],
                    "generated_prediction": labels[gold_index],
                    "base_prediction": labels[int(np.argmax(base))],
                    "final_prediction": labels[int(np.argmax(final))],
                    "personality_clean": personality,
                    "personality_used": personality,
                    "mood": (mood_scale * mood).tolist(),
                    "base_probability": base.tolist(),
                    "final_probability": final.tolist(),
                    "previous_soft_emotion_prob": {},
                    "previous_soft_emotion_valid": turn_index >= 2,
                    "source_index": len(samples),
                }
            )
    return samples


def test_diagnostic_outputs() -> None:
    dataset = "iemocap"
    labels = LABELS[dataset]
    full = synthetic_samples(dataset)
    no_dyn = synthetic_samples(dataset, mood_scale=1.15)
    with tempfile.TemporaryDirectory(prefix="moodfilm_diag_") as directory:
        output = Path(directory)
        analyze_noise({0.0: [full], 0.2: [full, full]}, output / "noise")
        analyze_transitions(full, no_dyn, labels, output / "transition")
        analyze_uncertainty(full, labels, output / "uncertainty")
        analyze_temporal(full, labels, 3, output / "timescale")
        analyze_variance(full, labels, output / "timescale")
        expected = (
            "noise/personality_noise_metrics.csv",
            "noise/personality_noise_robustness.png",
            "noise/personality_noise_robustness.pdf",
            "noise/personality_noise_robustness.svg",
            "transition/transition_summary.csv",
            "transition/emotion_transition_heatmaps.png",
            "uncertainty/difficulty_quartiles.csv",
            "uncertainty/difficulty_stratified_accuracy.png",
            "timescale/temporal_autocorrelation.csv",
            "timescale/temporal_autocorrelation.png",
            "timescale/cross_level_variance.csv",
            "timescale/cross_level_variance.png",
        )
        missing = [relative for relative in expected if not (output / relative).is_file()]
        assert not missing, f"Missing diagnostic outputs: {missing}"


if __name__ == "__main__":
    test_diagnostic_outputs()
    print("MoodFiLM diagnostic smoke test passed.")
