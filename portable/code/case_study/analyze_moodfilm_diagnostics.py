#!/usr/bin/env python
"""Analyze MoodFiLM evaluation traces and export publication-ready source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


LABELS = {
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
}


def json_documents(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    decoder = json.JSONDecoder()
    documents: list[Any] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        document, cursor = decoder.raw_decode(text, cursor)
        documents.append(document)
    return documents


def resolve_eval_file(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidates = sorted(path.glob("preds_for_eval*.text"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one eval trace in {path}, found {len(candidates)}. "
            "Pass the desired preds_for_eval_N.text explicitly."
        )
    return candidates[0]


def load_records(value: str) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    path = resolve_eval_file(value)
    records: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for document in json_documents(path):
        if isinstance(document, list):
            records.extend(item for item in document if isinstance(item, dict))
        elif isinstance(document, dict) and (
            "emotion" in document or "task_counts" in document
        ):
            metrics = document
    return records, metrics, path


def dialogue_turn(sample_id: str, dataset: str) -> tuple[str, int]:
    if dataset == "meld":
        match = re.match(r"^(.*)_utt(\d+)$", sample_id)
    else:
        match = re.match(r"^(.*)_(\d+)$", sample_id)
    if match is None:
        raise ValueError(f"Cannot parse dialogue/turn from sample_id={sample_id!r}")
    return match.group(1), int(match.group(2))


def profile_id(personality: list[float]) -> str:
    payload = ",".join(f"{float(value):.7f}" for value in personality)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def probability_vector(probabilities: dict[str, Any], labels: list[str]) -> list[float]:
    return [float(probabilities.get(label, 0.0)) for label in labels]


def trace_samples(records: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    labels = LABELS[dataset]
    samples: list[dict[str, Any]] = []
    for record in records:
        trace = record.get("speaker_mood_dynamics")
        if not isinstance(trace, dict):
            continue
        mood = trace.get("speaker_mood_after")
        clean_personality = trace.get("target_personality_clean")
        used_personality = trace.get("target_personality_used")
        base_prob = trace.get("base_label_prob")
        final_prob = trace.get("final_label_prob")
        if not (
            isinstance(mood, list)
            and isinstance(clean_personality, list)
            and isinstance(used_personality, list)
            and isinstance(base_prob, dict)
            and isinstance(final_prob, dict)
        ):
            continue
        sample_id = str(record.get("sample_id") or record.get("utterance_id") or "")
        dialogue_id, turn_index = dialogue_turn(sample_id, dataset)
        gold = str(record.get("gold_label") or record.get("target") or "").strip().lower()
        prediction = str(
            record.get("pred_label")
            or trace.get("pred_after_modulation")
            or record.get("output")
            or ""
        ).strip().lower()
        base_vector = probability_vector(base_prob, labels)
        final_vector = probability_vector(final_prob, labels)
        samples.append(
            {
                "sample_id": sample_id,
                "dialogue_id": dialogue_id,
                "turn_index": turn_index,
                "speaker_id": str(trace.get("target_speaker") or "target"),
                "profile_id": profile_id(clean_personality),
                "personality_available": not np.allclose(clean_personality, 0.0),
                "gold_label": gold,
                "generated_prediction": prediction,
                "base_prediction": labels[int(np.argmax(base_vector))],
                "final_prediction": labels[int(np.argmax(final_vector))],
                "personality_clean": [float(value) for value in clean_personality],
                "personality_used": [float(value) for value in used_personality],
                "mood": [float(value) for value in mood],
                "base_probability": base_vector,
                "final_probability": final_vector,
                "previous_soft_emotion_prob": trace.get(
                    "previous_soft_emotion_prob", {}
                ),
                "previous_soft_emotion_valid": bool(
                    trace.get("previous_soft_emotion_valid", False)
                ),
                "source_index": int(record.get("index", -1)),
            }
        )
    if len(samples) != len(records):
        raise RuntimeError(
            f"Complete analysis trace missing: retained {len(samples)}/{len(records)} records. "
            "Re-run evaluation with --save_speaker_mood_analysis True."
        )
    return sorted(samples, key=lambda item: item["source_index"])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def classification_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    gold = [sample["gold_label"] for sample in samples]
    pred = [sample["generated_prediction"] for sample in samples]
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }


def cosine_distance(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.dot(left_array, right_array) / denominator)


def save_figure(fig: Any, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def analyze_noise(
    noise_samples: dict[float, list[list[dict[str, Any]]]],
    output_dir: Path,
    noise_scale: str = "raw",
) -> list[dict[str, Any]]:
    replicate_rows = []
    rows = []
    metric_names = ("accuracy", "weighted_f1", "macro_f1")
    for sigma, replicates in sorted(noise_samples.items()):
        sigma_rows = []
        for replicate, samples in enumerate(replicates, start=1):
            metrics = classification_metrics(samples)
            row = {
                "sigma": sigma,
                "replicate": replicate,
                "num_samples": len(samples),
                **metrics,
            }
            replicate_rows.append(row)
            sigma_rows.append(row)
        aggregate = {
            "sigma": sigma,
            "num_replicates": len(sigma_rows),
            "num_samples_per_replicate": sigma_rows[0]["num_samples"],
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in sigma_rows], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(aggregate)
    write_rows(output_dir / "personality_noise_replicates.csv", replicate_rows)
    write_rows(output_dir / "personality_noise_metrics.csv", rows)

    plt = configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.35, 2.25))
    sigma = [row["sigma"] for row in rows]
    for metric, marker, label in (
        ("weighted_f1", "o", "Weighted F1"),
        ("accuracy", "s", "Accuracy"),
        ("macro_f1", "^", "Macro F1"),
    ):
        axis.errorbar(
            sigma,
            [100 * row[f"{metric}_mean"] for row in rows],
            yerr=[100 * row[f"{metric}_std"] for row in rows],
            marker=marker,
            capsize=2,
            label=label,
        )
    scale_label = "raw OCEAN" if noise_scale == "raw" else "normalized input"
    axis.set_xlabel(f"Personality noise standard deviation ({scale_label} scale)")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(sigma)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save_figure(figure, output_dir / "personality_noise_robustness")
    plt.close(figure)
    return rows


def analyze_personality_controls(
    clean_samples: list[dict[str, Any]],
    control_samples: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = [
        {
            "condition": "clean",
            "num_samples": len(clean_samples),
            **classification_metrics(clean_samples),
        }
    ]
    for condition, samples in sorted(control_samples.items()):
        rows.append(
            {
                "condition": condition,
                "num_samples": len(samples),
                **classification_metrics(samples),
            }
        )
    write_rows(output_dir / "personality_control_metrics.csv", rows)

    plt = configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.35, 2.25))
    positions = np.arange(len(rows))
    width = 0.24
    for offset, (metric, label) in enumerate(
        (
            ("accuracy", "Accuracy"),
            ("weighted_f1", "Weighted F1"),
            ("macro_f1", "Macro F1"),
        )
    ):
        axis.bar(
            positions + (offset - 1) * width,
            [100 * row[metric] for row in rows],
            width=width,
            label=label,
        )
    axis.set_xticks(positions, [row["condition"].capitalize() for row in rows])
    axis.set_ylabel("Score (%)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save_figure(figure, output_dir / "personality_random_control")
    plt.close(figure)
    return rows


def transition_pairs(
    samples: list[dict[str, Any]], checkpoint: str
) -> list[dict[str, Any]]:
    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        trajectories[(sample["dialogue_id"], sample["speaker_id"])].append(sample)
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda item: item["turn_index"])
        for previous, current in zip(trajectory, trajectory[1:]):
            distance = cosine_distance(previous["mood"], current["mood"])
            if not math.isfinite(distance):
                continue
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "dialogue_id": current["dialogue_id"],
                    "speaker_id": current["speaker_id"],
                    "previous_sample_id": previous["sample_id"],
                    "current_sample_id": current["sample_id"],
                    "previous_emotion": previous["gold_label"],
                    "current_emotion": current["gold_label"],
                    "emotion_changed": previous["gold_label"] != current["gold_label"],
                    "mood_shift": distance,
                }
            )
    return rows


def aggregate_transitions(
    pairs: list[dict[str, Any]], labels: list[str], checkpoint: str
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for row_index, previous in enumerate(labels):
        for column_index, current in enumerate(labels):
            values = [
                item["mood_shift"]
                for item in pairs
                if item["previous_emotion"] == previous
                and item["current_emotion"] == current
            ]
            mean = float(np.mean(values)) if values else float("nan")
            matrix[row_index, column_index] = mean
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "previous_emotion": previous,
                    "current_emotion": current,
                    "mean_mood_shift": mean,
                    "num_pairs": len(values),
                }
            )
    unchanged = [item["mood_shift"] for item in pairs if not item["emotion_changed"]]
    changed = [item["mood_shift"] for item in pairs if item["emotion_changed"]]
    unchanged_mean = float(np.mean(unchanged)) if unchanged else float("nan")
    changed_mean = float(np.mean(changed)) if changed else float("nan")
    summary = {
        "checkpoint": checkpoint,
        "unchanged_mean": unchanged_mean,
        "changed_mean": changed_mean,
        "changed_minus_unchanged": changed_mean - unchanged_mean,
        "unchanged_pairs": len(unchanged),
        "changed_pairs": len(changed),
    }
    return rows, summary, matrix


def analyze_transitions(
    full_samples: list[dict[str, Any]],
    no_dyn_samples: list[dict[str, Any]],
    labels: list[str],
    output_dir: Path,
) -> None:
    all_pairs = transition_pairs(full_samples, "with_Ldyn") + transition_pairs(
        no_dyn_samples, "without_Ldyn"
    )
    write_rows(output_dir / "transition_pairs.csv", all_pairs)
    heatmap_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    matrices = []
    for checkpoint in ("with_Ldyn", "without_Ldyn"):
        subset = [item for item in all_pairs if item["checkpoint"] == checkpoint]
        rows, summary, matrix = aggregate_transitions(subset, labels, checkpoint)
        heatmap_rows.extend(rows)
        summaries.append(summary)
        matrices.append(matrix)
    write_rows(output_dir / "transition_heatmap_values.csv", heatmap_rows)
    write_rows(output_dir / "transition_summary.csv", summaries)

    plt = configure_matplotlib()
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, ("With $L_{dyn}$", "Without $L_{dyn}$")):
        image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Current emotion")
        axis.set_ylabel("Previous emotion")
        midpoint = (vmin + vmax) / 2.0
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                value = matrix[row_index, column_index]
                if math.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="white" if value < midpoint else "black",
                    )
    figure.colorbar(image, ax=axes, label="$1 -$ cosine similarity", shrink=0.82)
    save_figure(figure, output_dir / "emotion_transition_heatmaps")
    plt.close(figure)


def analyze_uncertainty(
    samples: list[dict[str, Any]], labels: list[str], output_dir: Path
) -> None:
    rows = []
    for sample in samples:
        gold_index = labels.index(sample["gold_label"])
        base_gold = float(sample["base_probability"][gold_index])
        final_gold = float(sample["final_probability"][gold_index])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "gold_label": sample["gold_label"],
                "difficulty": 1.0 - base_gold,
                "base_gold_probability": base_gold,
                "final_gold_probability": final_gold,
                "delta_gold_probability": final_gold - base_gold,
                "base_correct": sample["base_prediction"] == sample["gold_label"],
                "final_correct": sample["final_prediction"] == sample["gold_label"],
            }
        )
    ordered = sorted(rows, key=lambda item: item["difficulty"], reverse=True)
    quartile_indices = np.array_split(np.arange(len(ordered)), 4)
    summary = []
    for quartile, indices in enumerate(quartile_indices, start=1):
        subset = [ordered[int(index)] for index in indices]
        for item in subset:
            item["quartile"] = f"Q{quartile}"
        base_accuracy = float(np.mean([item["base_correct"] for item in subset]))
        final_accuracy = float(np.mean([item["final_correct"] for item in subset]))
        summary.append(
            {
                "quartile": f"Q{quartile}",
                "difficulty_order": "Q1 hardest; Q4 easiest",
                "num_samples": len(subset),
                "mean_difficulty": float(np.mean([item["difficulty"] for item in subset])),
                "base_accuracy": base_accuracy,
                "final_accuracy": final_accuracy,
                "delta_accuracy": final_accuracy - base_accuracy,
                "mean_delta_gold_probability": float(
                    np.mean([item["delta_gold_probability"] for item in subset])
                ),
            }
        )
    write_rows(output_dir / "uncertainty_samples.csv", ordered)
    write_rows(output_dir / "difficulty_quartiles.csv", summary)

    plt = configure_matplotlib()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 3.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )
    accuracy_axis, change_axis = axes
    positions = np.arange(4)
    width = 0.34
    accuracy_axis.bar(
        positions - width / 2,
        [100 * row["base_accuracy"] for row in summary],
        width,
        label="Base",
    )
    accuracy_axis.bar(
        positions + width / 2,
        [100 * row["final_accuracy"] for row in summary],
        width,
        label="MoodFiLM",
    )
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    change_axis.axhline(0.0, color="0.65", linewidth=0.8)
    change_axis.plot(
        positions,
        [100 * row["delta_accuracy"] for row in summary],
        "o-",
        label=r"$\Delta$Acc",
    )
    change_axis.plot(
        positions,
        [100 * row["mean_delta_gold_probability"] for row in summary],
        "s-",
        label=r"Mean $\Delta p_{gold}$",
    )
    change_axis.set_xticks(positions, [row["quartile"] for row in summary])
    change_axis.set_xlabel("Backbone difficulty quartile (Q1 hardest)")
    change_axis.set_ylabel("Change (pp)")
    change_axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    save_figure(figure, output_dir / "difficulty_stratified_accuracy")
    plt.close(figure)


def standardized_vectors(
    samples: list[dict[str, Any]], key: str, labels: list[str]
) -> dict[str, np.ndarray]:
    if key == "emotion":
        matrix = np.asarray([sample["final_probability"] for sample in samples], dtype=np.float64)
    else:
        matrix = np.asarray([sample[key] for sample in samples], dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - mean) / scale
    return {sample["sample_id"]: normalized[index] for index, sample in enumerate(samples)}


def analyze_temporal(
    samples: list[dict[str, Any]], labels: list[str], max_lag: int, output_dir: Path
) -> None:
    vectors = {
        "personality": standardized_vectors(samples, "personality_clean", labels),
        "mood": standardized_vectors(samples, "mood", labels),
        "emotion": standardized_vectors(samples, "emotion", labels),
    }
    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        trajectories[(sample["dialogue_id"], sample["speaker_id"])].append(sample)
    pair_rows = []
    summary_rows = []
    for representation, lookup in vectors.items():
        for lag in range(1, max_lag + 1):
            trajectory_means = []
            pair_count = 0
            for (dialogue_id, speaker_id), trajectory in trajectories.items():
                trajectory.sort(key=lambda item: item["turn_index"])
                similarities = []
                for left, right in zip(trajectory, trajectory[lag:]):
                    left_vector = lookup[left["sample_id"]]
                    right_vector = lookup[right["sample_id"]]
                    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
                    if denominator <= 1e-12:
                        continue
                    similarity = float(np.dot(left_vector, right_vector) / denominator)
                    similarities.append(similarity)
                    pair_rows.append(
                        {
                            "representation": representation,
                            "lag": lag,
                            "dialogue_id": dialogue_id,
                            "speaker_id": speaker_id,
                            "left_sample_id": left["sample_id"],
                            "right_sample_id": right["sample_id"],
                            "cosine_similarity": similarity,
                        }
                    )
                if similarities:
                    trajectory_means.append(float(np.mean(similarities)))
                    pair_count += len(similarities)
            mean = float(np.mean(trajectory_means)) if trajectory_means else float("nan")
            standard_error = (
                float(np.std(trajectory_means, ddof=1) / math.sqrt(len(trajectory_means)))
                if len(trajectory_means) > 1
                else float("nan")
            )
            summary_rows.append(
                {
                    "representation": representation,
                    "lag": lag,
                    "mean_cosine_similarity": mean,
                    "ci95_low": mean - 1.96 * standard_error,
                    "ci95_high": mean + 1.96 * standard_error,
                    "num_trajectories": len(trajectory_means),
                    "num_pairs": pair_count,
                }
            )
    write_rows(output_dir / "temporal_pairs.csv", pair_rows)
    write_rows(output_dir / "temporal_autocorrelation.csv", summary_rows)

    plt = configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    styles = {"personality": "o-", "mood": "s-", "emotion": "^-"}
    for representation in ("personality", "mood", "emotion"):
        subset = [row for row in summary_rows if row["representation"] == representation]
        axis.plot(
            [row["lag"] for row in subset],
            [row["mean_cosine_similarity"] for row in subset],
            styles[representation],
            label=representation.capitalize(),
        )
        finite_interval = [
            row
            for row in subset
            if math.isfinite(row["ci95_low"]) and math.isfinite(row["ci95_high"])
        ]
        if len(finite_interval) == len(subset):
            axis.fill_between(
                [row["lag"] for row in subset],
                [row["ci95_low"] for row in subset],
                [row["ci95_high"] for row in subset],
                alpha=0.12,
            )
    axis.set_xlabel("Same-speaker occurrence lag")
    axis.set_ylabel("Standardized cosine similarity")
    axis.set_xticks(range(1, max_lag + 1))
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save_figure(figure, output_dir / "temporal_autocorrelation")
    plt.close(figure)


def analyze_variance(
    samples: list[dict[str, Any]], labels: list[str], output_dir: Path
) -> None:
    usable = [sample for sample in samples if sample["personality_available"]]
    rows = []
    for representation, key in (
        ("personality", "personality_clean"),
        ("mood", "mood"),
        ("emotion", "emotion"),
    ):
        lookup = standardized_vectors(usable, key, labels)
        matrix = np.asarray([lookup[sample["sample_id"]] for sample in usable])
        global_mean = matrix.mean(axis=0)
        groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for sample in usable:
            groups[sample["profile_id"]].append(lookup[sample["sample_id"]])
        between = 0.0
        within = 0.0
        for group in groups.values():
            group_matrix = np.asarray(group)
            group_mean = group_matrix.mean(axis=0)
            between += len(group) * float(np.square(group_mean - global_mean).sum())
            within += float(np.square(group_matrix - group_mean).sum())
        total = between + within
        rows.append(
            {
                "representation": representation,
                "between_speaker_variance": between,
                "within_speaker_variance": within,
                "between_fraction": between / total if total > 0 else float("nan"),
                "within_fraction": within / total if total > 0 else float("nan"),
                "num_samples": len(usable),
                "num_profiles": len(groups),
            }
        )
    write_rows(output_dir / "cross_level_variance.csv", rows)

    plt = configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.35, 2.25))
    positions = np.arange(len(rows))
    between = [row["between_fraction"] for row in rows]
    within = [row["within_fraction"] for row in rows]
    axis.bar(positions, between, label="Between speaker")
    axis.bar(positions, within, bottom=between, label="Within speaker")
    axis.set_xticks(positions, [row["representation"].capitalize() for row in rows])
    axis.set_ylabel("Fraction of standardized variance")
    axis.set_ylim(0.0, 1.0)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save_figure(figure, output_dir / "cross_level_variance")
    plt.close(figure)


def parse_noise_specs(values: list[str]) -> dict[float, list[str]]:
    result: dict[float, list[str]] = defaultdict(list)
    for value in values:
        sigma_text, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --noise-eval {value!r}; expected SIGMA=PATH")
        result[float(sigma_text)].append(path)
    if 0.0 not in result:
        raise ValueError("--noise-eval must include sigma 0 for full-model analyses.")
    return result


def parse_control_specs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        condition, separator, path = value.partition("=")
        condition = condition.strip().lower()
        if not separator or not condition or not path:
            raise ValueError(
                f"Invalid --personality-control {value!r}; expected NAME=PATH"
            )
        if condition != "random":
            raise ValueError(f"Unsupported personality control: {condition}")
        if condition in result:
            raise ValueError(f"Duplicate personality control: {condition}")
        result[condition] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(LABELS), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--noise-eval", action="append", default=[], metavar="SIGMA=PATH")
    parser.add_argument(
        "--personality-control", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--dyn-full-eval")
    parser.add_argument("--dyn-no-eval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--noise-scale", choices=["raw", "normalized"], default="raw")
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--dyn-full-checkpoint")
    parser.add_argument("--dyn-no-checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = LABELS[args.dataset]
    noise_specs = parse_noise_specs(args.noise_eval)
    control_specs = parse_control_specs(args.personality_control)
    noise_samples: dict[float, list[list[dict[str, Any]]]] = {}
    source_files: dict[str, str] = {}
    for sigma, values in sorted(noise_specs.items()):
        noise_samples[sigma] = []
        for replicate, value in enumerate(values, start=1):
            records, _, source = load_records(value)
            samples = trace_samples(records, args.dataset)
            noise_samples[sigma].append(samples)
            source_files[f"noise_{sigma:g}_rep{replicate}"] = str(source)
            stem = f"noise_sigma_{sigma:g}_rep{replicate}".replace(".", "p")
            write_jsonl(output_dir / "raw" / f"{stem}.jsonl", samples)
            write_rows(output_dir / "raw" / f"{stem}.csv", samples)

    analyze_noise(noise_samples, output_dir / "noise", args.noise_scale)
    control_samples = {}
    for condition, value in sorted(control_specs.items()):
        records, _, source = load_records(value)
        samples = trace_samples(records, args.dataset)
        control_samples[condition] = samples
        source_files[f"control_{condition}"] = str(source)
        write_jsonl(output_dir / "raw" / f"control_{condition}.jsonl", samples)
        write_rows(output_dir / "raw" / f"control_{condition}.csv", samples)
    if control_samples:
        analyze_personality_controls(
            noise_samples[0.0][0], control_samples, output_dir / "noise"
        )
    if bool(args.dyn_full_eval) != bool(args.dyn_no_eval):
        raise ValueError("Provide both --dyn-full-eval and --dyn-no-eval, or neither.")
    dyn_full_source = None
    dyn_no_source = None
    if args.dyn_full_eval and args.dyn_no_eval:
        dyn_full_records, _, dyn_full_source = load_records(args.dyn_full_eval)
        dyn_no_records, _, dyn_no_source = load_records(args.dyn_no_eval)
        dyn_full_samples = trace_samples(dyn_full_records, args.dataset)
        dyn_no_samples = trace_samples(dyn_no_records, args.dataset)
        write_jsonl(output_dir / "raw" / "dyn_full.jsonl", dyn_full_samples)
        write_jsonl(output_dir / "raw" / "dyn_no.jsonl", dyn_no_samples)
        write_rows(output_dir / "raw" / "dyn_full.csv", dyn_full_samples)
        write_rows(output_dir / "raw" / "dyn_no.csv", dyn_no_samples)
        analyze_transitions(
            dyn_full_samples, dyn_no_samples, labels, output_dir / "transition"
        )
    reference_samples = noise_samples[0.0][0]
    analyze_uncertainty(reference_samples, labels, output_dir / "uncertainty")
    analyze_temporal(
        reference_samples, labels, args.max_lag, output_dir / "timescale"
    )
    analyze_variance(reference_samples, labels, output_dir / "timescale")

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(reference_samples),
        "noise_sigmas": sorted(noise_samples),
        "noise_seed": args.noise_seed,
        "noise_scale": args.noise_scale,
        "personality_controls": sorted(control_samples),
        "final_checkpoint": args.final_checkpoint,
        "dyn_full_checkpoint": args.dyn_full_checkpoint,
        "dyn_no_checkpoint": args.dyn_no_checkpoint,
        "source_files": {
            **source_files,
            "dyn_full": None if dyn_full_source is None else str(dyn_full_source),
            "dyn_no": None if dyn_no_source is None else str(dyn_no_source),
        },
        "difficulty_order": "Q1 hardest; Q4 easiest",
        "temporal_lag": "same-speaker occurrence lag within each dialogue",
        "personality_noise": (
            "One deterministic Gaussian perturbation per unique OCEAN profile; "
            + (
                "raw sigma is divided by 3; "
                if args.noise_scale == "raw"
                else "sigma is applied directly to normalized model inputs; "
            )
            + "normalized values are clipped to [-1, 1]."
        ),
        "personality_random_control": (
            "One deterministic U(1, 7) OCEAN vector per unique input profile; "
            "the same speaker profile receives the same random replacement "
            "throughout evaluation."
        ),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
