import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


TRAITS = ("O", "C", "E", "A", "N")
PAIR_CLASSES = ("Speaker_0_higher", "Similar", "Speaker_1_higher")


def batch_trait_ccc_loss(predictions, targets):
    pred_mean = predictions.mean(dim=0)
    target_mean = targets.mean(dim=0)
    pred_var = predictions.var(dim=0, unbiased=False)
    target_var = targets.var(dim=0, unbiased=False)
    covariance = (
        (predictions - pred_mean) * (targets - target_mean)
    ).mean(dim=0)
    denominator = (
        pred_var + target_var + (pred_mean - target_mean).pow(2)
    )
    ccc = 2.0 * covariance / denominator.clamp(min=1e-8)
    return 1.0 - ccc.mean()


def compute_personality_pair_loss(
    predictions,
    targets,
    mode="regression",
    auxiliary_weight=0.5,
    threshold=0.5,
    margin=0.5,
    ccc_weight=0.5,
):
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError(
            "Paired personality tensors must both have shape [batch, 2, 5], "
            f"got predictions={tuple(predictions.shape)}, targets={tuple(targets.shape)}."
        )
    if predictions.size(1) != 2 or predictions.size(2) != len(TRAITS):
        raise ValueError(
            "Paired personality tensors must have shape [batch, 2, 5], "
            f"got {tuple(predictions.shape)}."
        )

    mode = str(mode).strip().lower()
    valid_modes = {
        "regression",
        "difference",
        "ranknet",
        "margin",
        "similarity_rank",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported personality pair mode: {mode}")

    predictions = predictions.float()
    targets = targets.float()
    flat_predictions = predictions.reshape(-1, len(TRAITS))
    flat_targets = targets.reshape(-1, len(TRAITS))
    mse_loss = F.mse_loss(flat_predictions, flat_targets)
    ccc_loss = batch_trait_ccc_loss(flat_predictions, flat_targets)
    regression_loss = mse_loss + float(ccc_weight) * ccc_loss

    predicted_difference = predictions[:, 0] - predictions[:, 1]
    gold_difference = targets[:, 0] - targets[:, 1]
    auxiliary_loss = regression_loss.new_zeros(())

    if mode == "difference":
        auxiliary_loss = F.mse_loss(predicted_difference, gold_difference)
    elif mode in {"ranknet", "margin", "similarity_rank"}:
        non_similar = gold_difference.abs() > float(threshold)
        direction = gold_difference.sign()
        signed_difference = direction * predicted_difference

        if mode == "ranknet":
            active = F.softplus(-signed_difference[non_similar])
            if active.numel() > 0:
                auxiliary_loss = active.mean()
        elif mode == "margin":
            active = F.relu(float(margin) - signed_difference[non_similar])
            if active.numel() > 0:
                auxiliary_loss = active.mean()
        else:
            losses = []
            ranking = F.softplus(-signed_difference[non_similar])
            if ranking.numel() > 0:
                losses.append(ranking.reshape(-1))
            similar = F.relu(
                predicted_difference[~non_similar].abs() - float(threshold)
            )
            if similar.numel() > 0:
                losses.append(similar.reshape(-1))
            if losses:
                auxiliary_loss = torch.cat(losses).mean()

    total_loss = regression_loss + float(auxiliary_weight) * auxiliary_loss
    return {
        "total": total_loss,
        "regression": regression_loss,
        "mse": mse_loss,
        "ccc": ccc_loss,
        "auxiliary": auxiliary_loss,
    }


def difference_to_class(difference, threshold=0.5):
    difference = float(difference)
    if abs(difference) <= float(threshold):
        return "Similar"
    if difference > 0:
        return "Speaker_0_higher"
    return "Speaker_1_higher"


def _safe_classification_metrics(golds, predictions):
    if not golds:
        return {
            "accuracy": None,
            "macro_f1": None,
            "balanced_accuracy": None,
        }
    labels = list(PAIR_CLASSES)
    return {
        "accuracy": round(100.0 * accuracy_score(golds, predictions), 3),
        "macro_f1": round(
            100.0
            * f1_score(
                golds,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            3,
        ),
        "balanced_accuracy": round(
            100.0 * balanced_accuracy_score(golds, predictions),
            3,
        ),
    }


def _ccc(golds, predictions):
    golds = np.asarray(golds, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if golds.size == 0 or golds.size != predictions.size:
        return None
    gold_mean = golds.mean()
    pred_mean = predictions.mean()
    denominator = (
        golds.var()
        + predictions.var()
        + (gold_mean - pred_mean) ** 2
    )
    if denominator <= 1e-12:
        return 0.0
    covariance = np.mean(
        (golds - gold_mean) * (predictions - pred_mean)
    )
    return float(2.0 * covariance / denominator)


def build_personality_pair_report(targets, predictions, threshold=0.5):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 3:
        raise ValueError(
            "Paired personality arrays must both have shape [samples, 2, 5], "
            f"got targets={targets.shape}, predictions={predictions.shape}."
        )
    if targets.shape[1:] != (2, len(TRAITS)):
        raise ValueError(
            f"Paired personality arrays must have shape [samples, 2, 5], got {targets.shape}."
        )

    continuous = {
        "per_trait_mae": {},
        "per_trait_mse": {},
        "per_trait_ccc": {},
    }
    flat_targets = targets.reshape(-1, len(TRAITS))
    flat_predictions = predictions.reshape(-1, len(TRAITS))
    for trait_index, trait in enumerate(TRAITS):
        gold = flat_targets[:, trait_index]
        pred = flat_predictions[:, trait_index]
        continuous["per_trait_mae"][trait] = round(
            float(np.mean(np.abs(gold - pred))), 4
        )
        continuous["per_trait_mse"][trait] = round(
            float(np.mean((gold - pred) ** 2)), 4
        )
        trait_ccc = _ccc(gold, pred)
        continuous["per_trait_ccc"][trait] = (
            None if trait_ccc is None else round(trait_ccc, 4)
        )
    continuous["avg_mae"] = round(
        float(np.mean(list(continuous["per_trait_mae"].values()))), 4
    )
    continuous["avg_mse"] = round(
        float(np.mean(list(continuous["per_trait_mse"].values()))), 4
    )
    valid_ccc = [
        value
        for value in continuous["per_trait_ccc"].values()
        if value is not None and math.isfinite(value)
    ]
    continuous["avg_ccc"] = (
        None if not valid_ccc else round(float(np.mean(valid_ccc)), 4)
    )
    continuous["evaluated_speakers"] = int(flat_targets.shape[0])

    gold_differences = targets[:, 0] - targets[:, 1]
    predicted_differences = predictions[:, 0] - predictions[:, 1]
    all_golds = []
    all_predictions = []
    all_absolute_gold_differences = []
    per_trait = {}
    for trait_index, trait in enumerate(TRAITS):
        trait_golds = [
            difference_to_class(value, threshold)
            for value in gold_differences[:, trait_index]
        ]
        trait_predictions = [
            difference_to_class(value, threshold)
            for value in predicted_differences[:, trait_index]
        ]
        all_golds.extend(trait_golds)
        all_predictions.extend(trait_predictions)
        all_absolute_gold_differences.extend(
            np.abs(gold_differences[:, trait_index]).tolist()
        )
        per_trait[trait] = {
            "count": len(trait_golds),
            **_safe_classification_metrics(
                trait_golds,
                trait_predictions,
            ),
        }

    bins = (
        ("[0,0.5]", lambda value: value <= 0.5),
        ("(0.5,1.0]", lambda value: 0.5 < value <= 1.0),
        ("(1.0,1.5]", lambda value: 1.0 < value <= 1.5),
        (">1.5", lambda value: value > 1.5),
    )
    difference_bins = {}
    abs_gold = np.asarray(all_absolute_gold_differences, dtype=np.float64)
    gold_classes = np.asarray(all_golds, dtype=object)
    predicted_classes = np.asarray(all_predictions, dtype=object)
    for name, predicate in bins:
        mask = np.asarray([predicate(float(value)) for value in abs_gold])
        count = int(mask.sum())
        accuracy = None
        if count > 0:
            accuracy = round(
                100.0
                * float(np.mean(gold_classes[mask] == predicted_classes[mask])),
                3,
            )
        difference_bins[name] = {"count": count, "accuracy": accuracy}

    return {
        "evaluated_pairs": int(targets.shape[0]),
        "continuous": continuous,
        "pairwise": {
            "threshold": float(threshold),
            "count": len(all_golds),
            **_safe_classification_metrics(all_golds, all_predictions),
            "per_trait": per_trait,
            "gold_difference_bins": difference_bins,
        },
    }
