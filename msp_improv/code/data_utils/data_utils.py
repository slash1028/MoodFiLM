import pandas as pd
import logging
import os
import ast
from os import truncate
from typing import List, Optional, Tuple, Union
import pickle
import wave

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import json
from dataclasses import dataclass
from multiprocessing import Pool
import multiprocessing
import math
from random import sample
import re
import time
import numpy as np
import torchaudio
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    AutoConfig,
    AutoProcessor,
)
from speaker_mood_recurrence import ORDERED_DATASET_INDEX, ORDERED_PADDING


logger = logging.getLogger(__name__)
AUDIO_UTTERANCE_MARKER = "[AUDIO_UTTERANCE]"
AUDIO_UTTERANCE_TOKEN = "<|AUDIO_UTTERANCE|>"
VISUAL_UTTERANCE_TOKEN = "<|VISUAL_UTTERANCE|>"
PERSONALITY_SPEAKER_0_TOKEN = "<|PERSONALITY_SPEAKER_0|>"
PERSONALITY_SPEAKER_1_TOKEN = "<|PERSONALITY_SPEAKER_1|>"
TRAIT_ORDER = ("O", "C", "E", "A", "N")
CONTRASTIVE_LABELS = {
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
    "EmoryNLP": ["Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"],
    "dialydailog": ["happy", "neutral", "angry", "sad", "fear", "surprise", "disgust"],
}


def _as_bool_flag(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


class _SuppressQwenOmniAudioOutputWarning(logging.Filter):
    _MESSAGE = "System prompt modified, audio output may not work as expected."

    def filter(self, record):
        return self._MESSAGE not in record.getMessage()


logging.getLogger().addFilter(_SuppressQwenOmniAudioOutputWarning())


def mask_prompt_content_for_audio_only(text):
    """Mask lexical dialogue content while preserving task instructions and label candidates."""
    if not isinstance(text, str):
        return text

    masked = text
    # Mask dialogue utterances while keeping speaker tags and prompt scaffold.
    masked = re.sub(
        r'(Speaker_\d+\*?\s*:\s*")([^"]*)(")',
        lambda m: f'{m.group(1)}{AUDIO_UTTERANCE_MARKER}{m.group(3)}',
        masked,
    )
    # Mask stage-1 speaker prompts such as <Speaker: raw text>.
    masked = re.sub(
        r'(<Speaker:\s*)([^>]*)(>)',
        lambda m: f'{m.group(1)}{AUDIO_UTTERANCE_MARKER}{m.group(3)}',
        masked,
    )
    # Mask in-context example utterance content.
    masked = re.sub(
        r'(Utterance:\s*")([^"]*)(")',
        lambda m: f'{m.group(1)}{AUDIO_UTTERANCE_MARKER}{m.group(3)}',
        masked,
        flags=re.IGNORECASE,
    )
    # Personality conversation prompts list utterances as bare JSON strings.
    masked = re.sub(
        r'"([^"]*)"',
        lambda m: f'"{AUDIO_UTTERANCE_MARKER}"',
        masked,
    )
    return masked


def strip_audio_markers_for_text_only(text):
    """Remove MELD's audio placeholders from a genuinely text-only prompt."""
    if not isinstance(text, str):
        return text
    return text.replace(AUDIO_UTTERANCE_MARKER, "")


def _is_true_flag(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _extract_tag_value(text, tag):
    match = re.search(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).strip()


def _canonical_contrastive_label(target, dataset):
    labels = CONTRASTIVE_LABELS.get(str(dataset), [])
    label_map = {str(label).strip().lower(): idx for idx, label in enumerate(labels)}
    target_text = str(target or "").strip()
    candidates = [
        _extract_tag_value(target_text, "answer"),
        _extract_tag_value(target_text, "label"),
        target_text.splitlines()[0].strip() if target_text else "",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip().lower()
        if normalized in label_map:
            return label_map[normalized]
        for label_text, label_idx in label_map.items():
            if re.search(rf"(?<![a-zA-Z]){re.escape(label_text)}(?![a-zA-Z])", normalized):
                return label_idx
    return -1


def _coerce_personality_vector(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return None
        return _coerce_personality_vector(parsed)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) < len(TRAIT_ORDER):
            return None
        values = []
        for item in list(value)[: len(TRAIT_ORDER)]:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return None
        return values
    if not isinstance(value, dict):
        return None
    values = []
    for trait in TRAIT_ORDER:
        if trait not in value:
            return None
        try:
            values.append(float(value[trait]))
        except (TypeError, ValueError):
            return None
    return values


def _normalize_personality_vector(values):
    if values is None:
        return None
    normalized = []
    for value in values:
        normalized.append(max(-1.0, min(1.0, (float(value) - 4.0) / 3.0)))
    return normalized


def _parse_personality_regression_target(target_text):
    raw = str(target_text or "").strip()
    if not raw:
        return None

    candidates = [raw]
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL):
        body = match.group(1).strip()
        if body:
            candidates.append(body)
    for match in re.finditer(r"\[[^\[\]]+\]", raw, flags=re.DOTALL):
        snippet = match.group(0).strip()
        if snippet:
            candidates.append(snippet)

    def _valid(vals):
        return len(vals) == 5 and all(1.0 <= float(v) <= 7.0 for v in vals)

    for cand in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(cand)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 5:
                    vals = [float(parsed[i]) for i in range(5)]
                    if _valid(vals):
                        return vals
            except Exception:
                pass

    trait_values = {}
    for trait in TRAIT_ORDER:
        match = re.search(
            rf"(?:^|[<>\s,;]){trait}\s*[:=]\s*([+-]?\d+(?:\.\d+)?)",
            raw,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        trait_values[trait] = float(match.group(1))
    vals = [trait_values[trait] for trait in TRAIT_ORDER]
    if _valid(vals):
        return vals
    return None


def _add_personality_regression_fields(batch_dict, targets, args):
    if not _as_bool_flag(getattr(args, "enable_personality_regression_head", "False")):
        return batch_dict
    rows = []
    masks = []
    for target in targets:
        values = _parse_personality_regression_target(target)
        if values is None:
            rows.append([0.0] * len(TRAIT_ORDER))
            masks.append(False)
        else:
            rows.append(values)
            masks.append(True)
    batch_dict["personality_regression_targets"] = torch.tensor(rows, dtype=torch.float32)
    batch_dict["personality_regression_mask"] = torch.tensor(masks, dtype=torch.bool)
    return batch_dict


def _coerce_personality_pair_targets(value):
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        for loader in (json.loads, ast.literal_eval):
            try:
                return _coerce_personality_pair_targets(loader(raw))
            except Exception:
                pass
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    pair = []
    for speaker_values in value:
        if not isinstance(speaker_values, (list, tuple)) or len(speaker_values) != 5:
            return None
        try:
            values = [float(item) for item in speaker_values]
        except (TypeError, ValueError):
            return None
        if not all(1.0 <= item <= 7.0 for item in values):
            return None
        pair.append(values)
    return pair


def _add_personality_pair_fields(batch_dict, metadata, tokenizer, args):
    mode = str(
        getattr(args, "personality_pair_mode", "none") or "none"
    ).strip().lower()
    if mode == "none":
        return batch_dict

    input_ids = batch_dict.get("input_ids")
    if input_ids is None:
        raise ValueError("personality_pair_mode requires input_ids in the collated batch.")
    query_token_ids = [
        tokenizer.convert_tokens_to_ids(PERSONALITY_SPEAKER_0_TOKEN),
        tokenizer.convert_tokens_to_ids(PERSONALITY_SPEAKER_1_TOKEN),
    ]
    if any(token_id is None or int(token_id) < 0 for token_id in query_token_ids):
        raise ValueError("Personality pair query tokens are not registered in the tokenizer.")

    targets = []
    masks = []
    positions = []
    for row_index, row_metadata in enumerate(metadata):
        pair_targets = _coerce_personality_pair_targets(
            (row_metadata or {}).get("personality_pair_targets")
        )
        is_pair = pair_targets is not None
        masks.append(is_pair)
        targets.append(pair_targets if is_pair else [[0.0] * 5, [0.0] * 5])

        row_positions = []
        for token_id in query_token_ids:
            matches = torch.nonzero(
                input_ids[row_index].eq(int(token_id)),
                as_tuple=False,
            ).flatten()
            if is_pair and matches.numel() != 1:
                raise ValueError(
                    "Each paired personality row must contain exactly one of each "
                    f"query token; row={row_index}, token_id={token_id}, "
                    f"matches={matches.numel()}."
                )
            row_positions.append(
                int(matches.item()) if matches.numel() == 1 else -1
            )
        positions.append(row_positions)

    pair_mask = torch.tensor(masks, dtype=torch.bool)
    batch_dict["personality_pair_targets"] = torch.tensor(
        targets,
        dtype=torch.float32,
    )
    batch_dict["personality_pair_mask"] = pair_mask
    batch_dict["personality_pair_query_positions"] = torch.tensor(
        positions,
        dtype=torch.long,
    )
    if "labels" in batch_dict and torch.any(pair_mask):
        batch_dict["labels"][pair_mask] = -100
        if "type_token_ids" in batch_dict:
            batch_dict["type_token_ids"][pair_mask] = 0
    return batch_dict


def _personality_prefix_source(args):
    if _as_bool_flag(getattr(args, "enable_personality_mood_prefix", "False")) and not (
        _as_bool_flag(getattr(args, "enable_personality_prefix", "False"))
        or _as_bool_flag(getattr(args, "enable_latent_mood_prefix", "False"))
    ):
        source = str(getattr(args, "personality_mood_source", "auto") or "auto").strip().lower()
    elif _as_bool_flag(getattr(args, "enable_latent_mood_prefix", "False")) and not _as_bool_flag(
        getattr(args, "enable_personality_prefix", "False")
    ):
        source = str(getattr(args, "latent_mood_source", "auto") or "auto").strip().lower()
    else:
        source = str(getattr(args, "personality_prefix_source", "auto") or "auto").strip().lower()
    if source not in ("gt", "predicted", "auto"):
        source = "auto"
    return source


def _personality_prefix_mode(args):
    if _as_bool_flag(getattr(args, "enable_personality_mood_prefix", "False")) and not (
        _as_bool_flag(getattr(args, "enable_personality_prefix", "False"))
        or _as_bool_flag(getattr(args, "enable_latent_mood_prefix", "False"))
    ):
        mode = str(getattr(args, "personality_mood_mode", "interaction") or "interaction").strip().lower()
        if mode in ("target_context", "factorized_attention"):
            mode = "interaction"
    elif _as_bool_flag(getattr(args, "enable_latent_mood_prefix", "False")) and not _as_bool_flag(
        getattr(args, "enable_personality_prefix", "False")
    ):
        mode = str(getattr(args, "latent_mood_mode", "interaction") or "interaction").strip().lower()
    else:
        mode = str(getattr(args, "personality_prefix_mode", "interaction") or "interaction").strip().lower()
    if mode not in ("target", "target_interlocutor", "interaction", "attention_interaction", "factorized", "shuffled"):
        mode = "interaction"
    return mode


def _select_target_personality(meta, source):
    meta = meta or {}
    if source == "gt":
        return _coerce_personality_vector(meta.get("gt_personality"))
    if source == "predicted":
        return _coerce_personality_vector(meta.get("personality"))
    return (
        _coerce_personality_vector(meta.get("personality"))
        or _coerce_personality_vector(meta.get("gt_personality"))
    )


def _select_interlocutor_personality(meta, source):
    meta = meta or {}
    if source == "gt":
        return _coerce_personality_vector(meta.get("gt_interlocutor_personality"))
    if source == "predicted":
        return _coerce_personality_vector(meta.get("interlocutor_personality"))
    return (
        _coerce_personality_vector(meta.get("interlocutor_personality"))
        or _coerce_personality_vector(meta.get("gt_interlocutor_personality"))
    )


def _select_speaker_context_personalities(meta, source):
    """Select context profiles without mixing GT and predicted sources."""
    meta = meta or {}
    predicted = _coerce_speaker_context_list(
        meta.get("speaker_context_personalities")
    )
    ground_truth = _coerce_speaker_context_list(
        meta.get("speaker_context_gt_personalities")
    )
    predicted_known = _coerce_speaker_context_list(
        meta.get("speaker_context_personality_known")
    )
    gt_known = _coerce_speaker_context_list(
        meta.get("speaker_context_gt_personality_known")
    )

    if source == "predicted":
        return predicted, predicted_known
    if source == "gt":
        return ground_truth, gt_known

    size = max(len(predicted), len(ground_truth))
    selected = []
    known = []
    for index in range(size):
        predicted_value = predicted[index] if index < len(predicted) else None
        gt_value = ground_truth[index] if index < len(ground_truth) else None
        value = (
            predicted_value
            if _coerce_personality_vector(predicted_value) is not None
            else gt_value
        )
        selected.append(value)
        known.append(_coerce_personality_vector(value) is not None)
    return selected, known


def _add_personality_prefix_fields(batch_dict, metadata, args):
    if not (
        _as_bool_flag(getattr(args, "enable_personality_prefix", "False"))
        or _as_bool_flag(getattr(args, "enable_latent_mood_prefix", "False"))
        or _as_bool_flag(getattr(args, "enable_personality_mood_prefix", "False"))
    ):
        return batch_dict

    source = _personality_prefix_source(args)
    mode = _personality_prefix_mode(args)
    target_rows = []
    interlocutor_rows = []
    masks = []

    for meta in metadata:
        target = _normalize_personality_vector(_select_target_personality(meta, source))
        interlocutor = _normalize_personality_vector(
            _select_interlocutor_personality(meta, source)
        )

        if target is None:
            target_rows.append([0.0] * len(TRAIT_ORDER))
            interlocutor_rows.append([0.0] * len(TRAIT_ORDER))
            masks.append(False)
            continue

        if interlocutor is None or mode == "target":
            interlocutor = [0.0] * len(TRAIT_ORDER)
            valid = mode == "target"
        else:
            valid = True

        target_rows.append(target)
        interlocutor_rows.append(interlocutor)
        masks.append(valid)

    target_tensor = torch.tensor(target_rows, dtype=torch.float32)
    interlocutor_tensor = torch.tensor(interlocutor_rows, dtype=torch.float32)
    if mode == "shuffled" and len(target_rows) > 0:
        if len(target_rows) > 1:
            order = torch.roll(torch.arange(len(target_rows)), shifts=1)
            target_tensor = target_tensor[order]
            interlocutor_tensor = interlocutor_tensor[order]
        else:
            target_tensor, interlocutor_tensor = interlocutor_tensor, target_tensor

    batch_dict["personality_prefix_target"] = target_tensor
    batch_dict["personality_prefix_interlocutor"] = interlocutor_tensor
    batch_dict["personality_prefix_mask"] = torch.tensor(masks, dtype=torch.bool)
    return batch_dict


def _add_mood_dynamics_fields(batch_dict, targets, metadata, args):
    if not _as_bool_flag(getattr(args, "use_mood_dynamics", "False")):
        return batch_dict

    labels = CONTRASTIVE_LABELS.get(
        str(getattr(args, "dataset", "iemocap")),
        CONTRASTIVE_LABELS.get("iemocap", []),
    )
    label_count = len(labels)
    label_map = {str(label).strip().lower(): idx for idx, label in enumerate(labels)}
    rows = []
    masks = []
    for meta in metadata:
        meta = meta or {}
        value = None
        for key in (
            "prev_emotion",
            "previous_emotion",
            "prev_emotion_label",
            "previous_emotion_label",
            "prev_pred_emotion",
            "previous_pred_emotion",
        ):
            if key in meta:
                value = meta.get(key)
                break
        row = [0.0] * label_count
        valid = False
        normalized = str(value or "").strip().lower()
        if normalized in label_map:
            row[label_map[normalized]] = 1.0
            valid = True
        rows.append(row)
        masks.append(valid)

    if not rows:
        rows = [[0.0] * label_count for _ in targets]
        masks = [False for _ in targets]
    batch_dict["mood_prev_emotion"] = torch.tensor(rows, dtype=torch.float32)
    batch_dict["mood_prev_emotion_mask"] = torch.tensor(masks, dtype=torch.bool)
    return batch_dict


def _coerce_relation_features(value):
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except Exception:
            try:
                value = ast.literal_eval(raw)
            except Exception:
                return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    items = list(value)
    if len(items) < 4:
        return None
    features = []
    for item in items[:4]:
        try:
            features.append(float(item))
        except (TypeError, ValueError):
            return None
    return features


def _coerce_speaker_context_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except Exception:
            try:
                value = ast.literal_eval(raw)
            except Exception:
                return []
    if not isinstance(value, (list, tuple)):
        return []
    return list(value)


def _speaker_mood_source(args):
    source = str(getattr(args, "speaker_mood_source", None) or _personality_prefix_source(args)).strip().lower()
    if source not in ("gt", "predicted", "auto"):
        source = "auto"
    return source


def _add_speaker_mood_dynamics_fields(batch_dict, targets, metadata, args):
    if not _as_bool_flag(getattr(args, "use_speaker_mood_dynamics", "False")):
        return batch_dict

    source = _speaker_mood_source(args)
    labels = CONTRASTIVE_LABELS.get(
        str(getattr(args, "dataset", "iemocap")),
        CONTRASTIVE_LABELS.get("iemocap", []),
    )
    label_count = len(labels)
    label_map = {str(label).strip().lower(): idx for idx, label in enumerate(labels)}

    target_rows = []
    context_rows = []
    relation_rows = []
    context_masks = []
    target_masks = []
    prev_rows = []
    prev_masks = []
    target_speakers = []
    target_speaker_ids = []
    sample_ids = []
    previous_sample_ids = []
    context_speakers = []
    ordered_dataset_indices = []
    ordered_padding = []
    has_ordered_metadata = False
    max_context = 1

    parsed_contexts = []
    for meta in metadata:
        meta = meta or {}
        target = _normalize_personality_vector(_select_target_personality(meta, source))
        target_valid = target is not None
        if target is None:
            target = [0.0] * len(TRAIT_ORDER)

        raw_context_values, raw_known_values = (
            _select_speaker_context_personalities(meta, source)
        )
        raw_relation_values = _coerce_speaker_context_list(meta.get("speaker_context_relation_features"))
        raw_context_names = _coerce_speaker_context_list(meta.get("speaker_context_speakers"))

        context_items = []
        for idx, raw_personality in enumerate(raw_context_values):
            values = _normalize_personality_vector(_coerce_personality_vector(raw_personality))
            available = values is not None
            if values is None:
                values = [0.0] * len(TRAIT_ORDER)

            relation = None
            if idx < len(raw_relation_values):
                relation = _coerce_relation_features(raw_relation_values[idx])
            if relation is None:
                relation = [0.0, 0.0, 0.0, 1.0 if available else 0.0]
            if idx < len(raw_known_values):
                relation[3] = 1.0 if bool(raw_known_values[idx]) and available else 0.0

            name = str(raw_context_names[idx]) if idx < len(raw_context_names) else f"context_{idx}"
            context_items.append((values, relation, available, name))

        if not context_items:
            fallback = _normalize_personality_vector(
                _select_interlocutor_personality(meta, source)
            )
            if fallback is not None:
                context_items.append((fallback, [1.0, 1.0, 1.0, 1.0], True, "interlocutor"))

        max_context = max(max_context, len(context_items))
        parsed_contexts.append((target, target_valid, context_items, meta))

    for target, target_valid, context_items, meta in parsed_contexts:
        is_ordered_padding = bool(meta.get(ORDERED_PADDING, False))
        if ORDERED_DATASET_INDEX in meta or ORDERED_PADDING in meta:
            has_ordered_metadata = True
        row_context = []
        row_relation = []
        row_mask = []
        row_names = []
        for values, relation, available, name in context_items:
            row_context.append(values)
            row_relation.append(relation)
            row_mask.append(True)
            row_names.append(name)
        while len(row_context) < max_context:
            row_context.append([0.0] * len(TRAIT_ORDER))
            row_relation.append([0.0, 0.0, 0.0, 0.0])
            row_mask.append(False)
            row_names.append("")

        value = None
        for key in (
            "prev_emotion",
            "previous_emotion",
            "prev_emotion_label",
            "previous_emotion_label",
            "prev_pred_emotion",
            "previous_pred_emotion",
        ):
            if key in meta:
                value = meta.get(key)
                break
        prev = [0.0] * label_count
        prev_valid = False
        normalized = str(value or "").strip().lower()
        if normalized in label_map:
            prev[label_map[normalized]] = 1.0
            prev_valid = True

        target_rows.append(target)
        context_rows.append(row_context)
        relation_rows.append(row_relation)
        context_masks.append(row_mask)
        # Padding rows remain numerically ordinary duplicate forwards. They are
        # excluded explicitly from every objective and state update below.
        target_masks.append(bool(target_valid))
        prev_rows.append(prev)
        prev_masks.append(prev_valid)
        target_speakers.append(str(meta.get("speaker_name") or meta.get("speaker_id") or "target"))
        target_speaker_ids.append(str(meta.get("speaker_id") or meta.get("speaker_name") or "target"))
        sample_ids.append(str(meta.get("sample_id") or meta.get("utterance_id") or ""))
        prev_sample_id = (
            meta.get("previous_same_speaker_sample_id")
            or meta.get("prev_same_speaker_sample_id")
            or meta.get("previous_sample_id")
            or ""
        )
        previous_sample_ids.append(str(prev_sample_id) if prev_sample_id is not None else "")
        context_speakers.append(row_names)
        ordered_dataset_indices.append(int(meta.get(ORDERED_DATASET_INDEX, -1)))
        ordered_padding.append(is_ordered_padding)

    batch_dict["speaker_mood_target_personality"] = torch.tensor(target_rows, dtype=torch.float32)
    batch_dict["speaker_mood_context_personalities"] = torch.tensor(context_rows, dtype=torch.float32)
    batch_dict["speaker_mood_relation_features"] = torch.tensor(relation_rows, dtype=torch.float32)
    batch_dict["speaker_mood_context_mask"] = torch.tensor(context_masks, dtype=torch.bool)
    batch_dict["speaker_mood_target_mask"] = torch.tensor(target_masks, dtype=torch.bool)
    batch_dict["speaker_mood_prev_emotion"] = torch.tensor(prev_rows, dtype=torch.float32)
    batch_dict["speaker_mood_prev_emotion_mask"] = torch.tensor(prev_masks, dtype=torch.bool)
    batch_dict["speaker_mood_target_speaker"] = target_speakers
    batch_dict["speaker_mood_target_speaker_id"] = target_speaker_ids
    batch_dict["speaker_mood_sample_id"] = sample_ids
    batch_dict["speaker_mood_previous_sample_id"] = previous_sample_ids
    batch_dict["speaker_mood_context_speakers"] = context_speakers
    if has_ordered_metadata:
        padding_tensor = torch.tensor(ordered_padding, dtype=torch.bool)
        batch_dict["speaker_mood_dataset_index"] = torch.tensor(
            ordered_dataset_indices,
            dtype=torch.long,
        )
        batch_dict["speaker_mood_order_padding"] = padding_tensor
        for mask_key in (
            "contrastive_task_mask",
            "personality_regression_mask",
            "personality_pair_mask",
        ):
            mask = batch_dict.get(mask_key)
            if torch.is_tensor(mask) and mask.size(0) == padding_tensor.size(0):
                batch_dict[mask_key] = mask.clone()
                batch_dict[mask_key][padding_tensor] = False
    return batch_dict


def _metadata_from_item(item, emotion_prediction):
    if emotion_prediction:
        return item[4] if len(item) > 4 and isinstance(item[4], dict) else {}
    return item[3] if len(item) > 3 and isinstance(item[3], dict) else {}


def _is_missing_metadata_value(value):
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except (TypeError, ValueError):
        return False
    return False


def _add_contrastive_fields(batch_dict, targets, metadata, args):
    if not _is_true_flag(getattr(args, "enable_contrastive", "False")):
        return batch_dict

    dataset = getattr(args, "dataset", "iemocap")
    label_ids = []
    task_mask = []
    personality_rows = []
    personality_mask = []
    for target, meta in zip(targets, metadata):
        label_id = _canonical_contrastive_label(target, dataset)
        label_ids.append(label_id)
        task_mask.append(label_id >= 0)
        personality_vector = _coerce_personality_vector((meta or {}).get("personality"))
        if personality_vector is None:
            personality_rows.append([0.0] * len(TRAIT_ORDER))
            personality_mask.append(False)
        else:
            personality_rows.append(personality_vector)
            personality_mask.append(True)

    batch_dict["contrastive_label_ids"] = torch.tensor(label_ids, dtype=torch.long)
    batch_dict["contrastive_task_mask"] = torch.tensor(task_mask, dtype=torch.bool)
    batch_dict["contrastive_personality"] = torch.tensor(personality_rows, dtype=torch.float32)
    batch_dict["contrastive_personality_mask"] = torch.tensor(personality_mask, dtype=torch.bool)
    return batch_dict


def _normalize_text_for_alignment(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9'\s]", "", s)
    return s


def _parse_transcript_lines(transcript_path: str) -> List[Tuple[str, str]]:
    utts = []
    if not os.path.exists(transcript_path):
        return utts
    with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(\S+)\s*\[[^\]]*\]\s*:\s*(.*)$', line)
            if m:
                utts.append((m.group(1), m.group(2)))
    return utts


def _dialogue_to_session(conv_id: str) -> Optional[str]:
    m = re.match(r'^Ses(\d\d)', str(conv_id))
    if m is None:
        return None
    return f"Session{int(m.group(1))}"


def _load_wav_mono(path: str) -> Tuple[torch.Tensor, int]:
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)

    return torch.from_numpy(data), sample_rate


def _build_conv_turn_to_uttid_map(iemocap_pkl: str, iemocap_root: str):
    data = pickle.load(open(iemocap_pkl, 'rb'))
    sentence_dict = data[2]
    all_ids = data[3] + data[4] + data[5]

    mapping = {}
    for conv_id in all_ids:
        session = _dialogue_to_session(conv_id)
        if session is None:
            continue

        transcript_path = os.path.join(iemocap_root, session, 'dialog', 'transcriptions', f'{conv_id}.txt')
        transcript_utts = _parse_transcript_lines(transcript_path)
        if len(transcript_utts) == 0:
            continue

        sent_list = sentence_dict[conv_id]
        utt_ids = []
        if len(transcript_utts) == len(sent_list):
            utt_ids = [u[0] for u in transcript_utts]
        else:
            norm_trans = [_normalize_text_for_alignment(t) for _, t in transcript_utts]
            search_start = 0
            for s in sent_list:
                ns = _normalize_text_for_alignment(s)
                found_idx = None
                for i in range(search_start, len(norm_trans)):
                    nt = norm_trans[i]
                    if nt == ns:
                        found_idx = i
                        break
                if found_idx is None:
                    utt_ids.append(None)
                else:
                    utt_ids.append(transcript_utts[found_idx][0])
                    search_start = found_idx + 1

        for turn, utt_id in enumerate(utt_ids):
            mapping[f'{conv_id}_{turn}'] = utt_id

    return mapping

def _speaker_mood_dialogue_turn_key(row):
    sample_id = str(row.get("sample_id") or row.get("utterance_id") or "")
    if not sample_id:
        return None, None
    meld_match = re.match(r"(.+_dia\d+)_utt(\d+)$", sample_id)
    if meld_match:
        return meld_match.group(1), int(meld_match.group(2))
    iemocap_match = re.match(r"(.+?)_[FM]?(\d+)$", sample_id)
    if iemocap_match:
        return iemocap_match.group(1), int(iemocap_match.group(2))
    if "_" in sample_id:
        prefix, suffix = sample_id.rsplit("_", 1)
        if suffix.isdigit():
            return prefix, int(suffix)
    return sample_id, None


def _add_same_speaker_previous_emotion(data):
    """Populate previous same-speaker emotion/sample metadata from preceding turns."""
    rows = []
    for index, row in enumerate(data):
        row = dict(row)
        dialogue_id, turn_idx = _speaker_mood_dialogue_turn_key(row)
        speaker = str(row.get("speaker_id") or row.get("speaker_name") or "")
        rows.append((dialogue_id, turn_idx if turn_idx is not None else index, speaker, index, row))

    last_emotion_by_speaker = {}
    last_sample_by_speaker = {}
    for dialogue_id, turn_idx, speaker, _, row in sorted(
        rows,
        key=lambda item: (
            "" if item[0] is None else str(item[0]),
            int(item[1]) if item[1] is not None else 0,
            int(item[3]),
        ),
    ):
        key = (dialogue_id, speaker)
        existing = None
        for field in (
            "prev_emotion",
            "previous_emotion",
            "prev_emotion_label",
            "previous_emotion_label",
            "prev_pred_emotion",
            "previous_pred_emotion",
        ):
            if row.get(field) not in (None, ""):
                existing = row.get(field)
                break
        if speaker and key in last_emotion_by_speaker:
            if existing is None:
                row["previous_emotion_label"] = last_emotion_by_speaker[key]
            if not row.get("previous_same_speaker_sample_id"):
                row["previous_same_speaker_sample_id"] = last_sample_by_speaker.get(key, "")
        elif not row.get("previous_same_speaker_sample_id"):
            row["previous_same_speaker_sample_id"] = ""
        target = row.get("target")
        sample_id = str(row.get("sample_id") or row.get("utterance_id") or "")
        if target not in (None, "") and speaker:
            last_emotion_by_speaker[key] = target
            if sample_id:
                last_sample_by_speaker[key] = sample_id
    return [row for *_, row in sorted(rows, key=lambda item: int(item[3]))]


def _add_source_specific_personality_contexts(data):
    """Materialize predicted and GT context profiles from dialogue speakers."""
    rows = [dict(row) for row in data]
    dialogue_speakers = {}
    profiles = {"predicted": {}, "gt": {}}

    for row in rows:
        dialogue_id, _ = _speaker_mood_dialogue_turn_key(row)
        if dialogue_id is None:
            continue
        speaker_id = str(row.get("speaker_id") or row.get("speaker_name") or "").strip()
        if not speaker_id:
            continue
        speakers = dialogue_speakers.setdefault(str(dialogue_id), [])
        if speaker_id not in speakers:
            speakers.append(speaker_id)

        aliases = {
            speaker_id,
            str(row.get("speaker_id") or "").strip(),
            str(row.get("speaker_name") or "").strip(),
        }
        aliases.discard("")
        for source, field in (("predicted", "personality"), ("gt", "gt_personality")):
            values = _coerce_personality_vector(row.get(field))
            if values is None:
                continue
            for alias in aliases:
                profiles[source].setdefault(
                    (str(dialogue_id), alias),
                    list(values),
                )

    for row in rows:
        dialogue_id, _ = _speaker_mood_dialogue_turn_key(row)
        dialogue_id = None if dialogue_id is None else str(dialogue_id)
        speaker_id = str(row.get("speaker_id") or row.get("speaker_name") or "").strip()
        names_present = (
            "speaker_context_speakers" in row
            and row.get("speaker_context_speakers") is not None
        )
        context_names = [
            str(value).strip()
            for value in _coerce_speaker_context_list(
                row.get("speaker_context_speakers")
            )
            if str(value).strip()
        ]
        if not names_present and dialogue_id is not None:
            other_speakers = [
                value
                for value in dialogue_speakers.get(dialogue_id, [])
                if value != speaker_id
            ]
            # The legacy IEMOCAP rows contain one interlocutor profile but no
            # context identity. Resolve it only when the dialogue is dyadic.
            if len(other_speakers) == 1:
                context_names = other_speakers
                row["speaker_context_speakers"] = list(context_names)

        predicted_context = _coerce_speaker_context_list(
            row.get("speaker_context_personalities")
        )
        gt_context = _coerce_speaker_context_list(
            row.get("speaker_context_gt_personalities")
        )
        if context_names:
            if len(predicted_context) != len(context_names):
                predicted_context = [
                    profiles["predicted"].get((dialogue_id, name))
                    for name in context_names
                ]
            if len(gt_context) != len(context_names):
                gt_context = [
                    profiles["gt"].get((dialogue_id, name))
                    for name in context_names
                ]

            row["speaker_context_personalities"] = predicted_context
            row["speaker_context_gt_personalities"] = gt_context
            row["speaker_context_personality_known"] = [
                _coerce_personality_vector(value) is not None
                for value in predicted_context
            ]
            row["speaker_context_gt_personality_known"] = [
                _coerce_personality_vector(value) is not None
                for value in gt_context
            ]

            if len(context_names) == 1:
                predicted_interlocutor = profiles["predicted"].get(
                    (dialogue_id, context_names[0])
                )
                gt_interlocutor = profiles["gt"].get(
                    (dialogue_id, context_names[0])
                )
                if _coerce_personality_vector(
                    row.get("interlocutor_personality")
                ) is None and predicted_interlocutor is not None:
                    row["interlocutor_personality"] = predicted_interlocutor
                if gt_interlocutor is not None:
                    row["gt_interlocutor_personality"] = gt_interlocutor

    return rows


def read_data(file_name, percent, random_seed):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]
    data = _add_same_speaker_previous_emotion(data)
    data = _add_source_specific_personality_contexts(data)
    inputs = []
    targets = []
    sample_ids = []
    debug_target_utterances = []
    utterance_ids = []
    speaker_ids = []
    personalities = []
    teacher_reasonings = []
    experience_bank_contexts = []
    opcd_gold_contexts = []
    gt_personalities = []
    interlocutor_personalities = []
    gt_interlocutor_personalities = []
    derived_shifts = []
    personality_pair_targets = []
    speaker_context_personalities = []
    speaker_context_gt_personalities = []
    speaker_context_relation_features = []
    speaker_context_speakers = []
    speaker_context_personality_known = []
    speaker_context_gt_personality_known = []
    speaker_names = []
    previous_emotion_labels = []
    previous_same_speaker_sample_ids = []
    audio_turn_sample_ids = []
    for index, d in enumerate(data):
        if pd.isnull(d['target']) or pd.isna(d['target']):
            continue
        inputs.append(d['input'])
        targets.append(d['target'])
        sample_ids.append(d.get('sample_id', None))
        debug_target_utterances.append(d.get('debug_target_utterance', None))
        utterance_ids.append(d.get('utterance_id', None))
        speaker_ids.append(d.get('speaker_id', None))
        personalities.append(d.get('personality', None))
        teacher_reasonings.append(d.get('teacher_reasoning', None))
        experience_bank_contexts.append(d.get('experience_bank_context', None))
        opcd_gold_contexts.append(d.get('opcd_gold_context', None))
        gt_personalities.append(d.get('gt_personality', None))
        interlocutor_personalities.append(d.get('interlocutor_personality', None))
        gt_interlocutor_personalities.append(d.get('gt_interlocutor_personality', None))
        derived_shifts.append(d.get('derived_shift', None))
        personality_pair_targets.append(d.get('personality_pair_targets', None))
        speaker_context_personalities.append(d.get('speaker_context_personalities', None))
        speaker_context_gt_personalities.append(d.get('speaker_context_gt_personalities', None))
        speaker_context_relation_features.append(d.get('speaker_context_relation_features', None))
        speaker_context_speakers.append(d.get('speaker_context_speakers', None))
        speaker_context_personality_known.append(d.get('speaker_context_personality_known', None))
        speaker_context_gt_personality_known.append(d.get('speaker_context_gt_personality_known', None))
        speaker_names.append(d.get('speaker_name', None))
        previous_emotion_labels.append(
            d.get(
                'previous_emotion_label',
                d.get(
                    'prev_emotion_label',
                    d.get('previous_emotion', d.get('prev_emotion', None)),
                ),
            )
        )
        previous_same_speaker_sample_ids.append(
            d.get(
                'previous_same_speaker_sample_id',
                d.get('prev_same_speaker_sample_id', d.get('previous_sample_id', None)),
            )
        )
        audio_turn_sample_ids.append(d.get('audio_turn_sample_ids', None))
    dict_ = {
        'input': inputs,
        'output': targets,
        'sample_id': sample_ids,
        'debug_target_utterance': debug_target_utterances,
        'utterance_id': utterance_ids,
        'speaker_id': speaker_ids,
        'personality': personalities,
        'teacher_reasoning': teacher_reasonings,
        'experience_bank_context': experience_bank_contexts,
        'opcd_gold_context': opcd_gold_contexts,
        'gt_personality': gt_personalities,
        'interlocutor_personality': interlocutor_personalities,
        'gt_interlocutor_personality': gt_interlocutor_personalities,
        'derived_shift': derived_shifts,
        'personality_pair_targets': personality_pair_targets,
        'speaker_context_personalities': speaker_context_personalities,
        'speaker_context_gt_personalities': speaker_context_gt_personalities,
        'speaker_context_relation_features': speaker_context_relation_features,
        'speaker_context_speakers': speaker_context_speakers,
        'speaker_context_personality_known': speaker_context_personality_known,
        'speaker_context_gt_personality_known': speaker_context_gt_personality_known,
        'speaker_name': speaker_names,
        'previous_emotion_label': previous_emotion_labels,
        'previous_same_speaker_sample_id': previous_same_speaker_sample_ids,
        'audio_turn_sample_ids': audio_turn_sample_ids,
    }
    df_data = pd.DataFrame(dict_)
    df_data.dropna(axis=0, how='any')

    # randomly extract *percent of the data
    num_samples = int(len(df_data)*percent)
    rank = os.getenv("RANK", "0")
    verbose_data = os.getenv("INSTRUCTERC_VERBOSE_DATA_LOG", "0") == "1"
    if rank in ("0", "-1") and verbose_data:
        print(f"[INFO] num_samples before sampling: {len(df_data)}")
    df_data = df_data.sample(n=num_samples, random_state=random_seed)
    if rank in ("0", "-1") and verbose_data:
        print(f"[INFO] num_samples after sampling: {len(df_data)}")

    return df_data


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
    
def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


class Seq2SeqDataset(Dataset):
    def __init__(self, args, data, mode):
        sample_ids = list(data['sample_id']) if 'sample_id' in data.columns else [None] * len(data)
        metadata = []
        for idx in range(len(data)):
            item = {}
            for key in (
                "sample_id",
                "utterance_id",
                "speaker_id",
                "speaker_name",
                "personality",
                "teacher_reasoning",
                "experience_bank_context",
                "opcd_gold_context",
                "gt_personality",
                "interlocutor_personality",
                "gt_interlocutor_personality",
                "derived_shift",
                "personality_pair_targets",
                "audio_turn_sample_ids",
                "prev_emotion",
                "previous_emotion",
                "prev_emotion_label",
                "previous_emotion_label",
                "previous_same_speaker_sample_id",
                "prev_same_speaker_sample_id",
                "previous_sample_id",
                "prev_pred_emotion",
                "previous_pred_emotion",
                "speaker_context_personalities",
                "speaker_context_gt_personalities",
                "speaker_context_relation_features",
                "speaker_context_speakers",
                "speaker_context_personality_known",
                "speaker_context_gt_personality_known",
            ):
                if key in data.columns:
                    value = data.iloc[idx][key]
                    if not _is_missing_metadata_value(value):
                        item[key] = value
            metadata.append(item)
        if not args.emotion_prediction:
            inputs = list(data["input"])
            outputs = list(data['output'])
            self.examples = [[i, o, sid, meta] for i, o, sid, meta in zip(inputs, outputs, sample_ids, metadata)]
        elif mode == 'dev':
            inputs = list(data["input"])
            outputs = list(data['output'])
            self.examples = [[i, o, sid, meta] for i, o, sid, meta in zip(inputs, outputs, sample_ids, metadata)]
        else:
            inputs = list(data["input"])
            inputs = [i.split('***') for i in inputs]
            # counter = 0
            # for i in inputs:
            #     if len(i) != 2:
            #         counter += 1
            # print(counter)
            outputs = list(data['output'])
            self.examples = [[i[0], i[1], o, sid, meta] for i, o, sid, meta in zip(inputs, outputs, sample_ids, metadata)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.args = args    
        self.mode = mode
        self.audio_dialogue_features = None
        self.audio_feature_dim = 0
        self.audio_context_window = int(getattr(args, 'audio_context_window', -1))
        self.audio_enable = _as_bool_flag(getattr(args, 'audio_enable', False), default=False)
        self.audio_embedding_source = str(getattr(args, 'audio_embedding_source', 'precomputed_pkl')).strip().lower()
        self.audio_embedding_usage = str(
            getattr(args, 'audio_embedding_usage', 'direct') or 'direct'
        ).strip().lower()
        if self.audio_embedding_usage not in ('direct', 'context', 'both'):
            raise ValueError("Invalid audio_embedding_usage. Use: direct | context | both")
        self.audio_input_mode = str(getattr(args, 'audio_input_mode', 'text_audio')).strip().lower()
        self.audio_token_placement = str(
            getattr(args, 'audio_token_placement', 'aligned') or 'aligned'
        ).strip().lower()
        if self.audio_token_placement not in ('aligned', 'tail'):
            raise ValueError("Invalid audio_token_placement. Use: aligned | tail")
        self.audio_feature_pkl = getattr(args, 'audio_feature_pkl', None)
        self.text_enable = _as_bool_flag(getattr(args, 'text_enable', True), default=True)
        self.video_enable = _as_bool_flag(getattr(args, 'video_enable', False), default=False)
        self.video_embedding_source = str(
            getattr(args, 'video_embedding_source', 'qwen_runtime_extract')
        ).strip().lower()
        self.video_embedding_usage = str(
            getattr(args, 'video_embedding_usage', 'direct') or 'direct'
        ).strip().lower()
        if self.video_embedding_usage not in ('none', 'direct', 'context', 'both'):
            raise ValueError("Invalid video_embedding_usage. Use: none | direct | context | both")
        self.context_prefix_enable = _as_bool_flag(
            getattr(args, 'enable_stage1_context_prefix', False),
            default=False,
        )
        self.audio_direct_enable = (
            self.audio_enable
            and self.audio_embedding_usage in ('direct', 'both')
        )
        self.audio_context_enable = (
            self.audio_enable
            and self.context_prefix_enable
            and self.audio_embedding_usage in ('context', 'both')
        )
        self.video_direct_enable = (
            self.video_enable
            and self.video_embedding_usage in ('direct', 'both')
        )
        self.video_context_enable = (
            self.video_enable
            and self.context_prefix_enable
            and self.video_embedding_usage in ('context', 'both')
        )
        self.text_context_enable = self.context_prefix_enable
        self.text_feature_pkl = getattr(args, 'text_feature_pkl', None)
        self.utterance_feature_text = _as_bool_flag(getattr(args, 'utterance_feature_text', False))
        self.utterance_feature_audio = _as_bool_flag(getattr(args, 'utterance_feature_audio', False))
        self.utterance_feature_enable = self.utterance_feature_text or self.utterance_feature_audio
        self.stage1_speaker_audio_only = _as_bool_flag(getattr(args, 'stage1_speaker_audio_only', False))
        self.utterance_direct_audio_once = _as_bool_flag(getattr(args, 'utterance_direct_audio_once', False))
        self.utterance_feature_history_window = int(getattr(args, 'utterance_feature_history_window', -1))
        self.video_feature_pkl = getattr(args, 'video_feature_pkl', None)
        self.personality_media_scope = str(
            getattr(args, 'personality_media_scope', 'legacy')
        ).strip().lower()
        if self.personality_media_scope not in ('legacy', 'target', 'full'):
            raise ValueError(
                "Invalid personality_media_scope. Use: legacy | target | full"
            )
        self.video_manifest_jsonl = getattr(args, 'video_manifest_jsonl', None)
        self.video_fps = float(getattr(args, 'video_fps', 2.0) or 2.0)
        self.video_resized_height = int(getattr(args, 'video_resized_height', 336) or 336)
        self.video_resized_width = int(getattr(args, 'video_resized_width', 0) or 0)
        self.video_use_audio_in_video = _as_bool_flag(getattr(args, 'video_use_audio_in_video', False), default=False)
        self.qwen_processor = None
        self.qwen_tokenizer = None
        self.qwen_sampling_rate = None
        self.qwen_hop_length = 160
        self.qwen_text_hidden_size = 0
        self.sampleid_to_wavpath = None
        self.sampleid_to_video_path = {}
        self.qwen_text_features = None
        self.qwen_text_feature_dim = 0
        self.qwen_visual_features = None
        self.qwen_visual_feature_dim = 0
        self.qwen_mel_features = None
        self.qwen_mel_lengths = None
        self.qwen_missing_input_features = None
        self.qwen_missing_feature_length = None
        self.qwen_audio_features = None
        self.qwen_audio_feature_dim = 0
        self.iemocap_speaker_ids_by_conv = {}
        self.iemocap_num_turns_by_conv = {}
        if self.audio_enable or self.video_enable or self.context_prefix_enable or self.utterance_feature_enable:
            self._init_iemocap_dialogue_index()
        if (self.context_prefix_enable or self.utterance_feature_enable) and self.qwen_processor is None:
            self._init_qwen_processor()
        if self.text_context_enable or self.utterance_feature_text:
            self._init_qwen_precomputed_text_feature_store()
        if self.video_enable:
            self._init_qwen_processor()
            self._init_qwen_video_store()
            if self.video_embedding_source == 'precomputed_tokens':
                self._init_qwen_precomputed_video_store()
            elif self.video_embedding_source != 'qwen_runtime_extract':
                raise ValueError(
                    f"Invalid video_embedding_source: {self.video_embedding_source}. "
                    "Use qwen_runtime_extract or precomputed_tokens."
                )
        if self.audio_enable:
            if self.audio_embedding_source == 'precomputed_pkl':
                self._init_qwen_precomputed_audio_store()
            elif self.audio_embedding_source == 'precomputed_tokens':
                self._init_qwen_precomputed_audio_token_store()
            elif self.audio_embedding_source == 'qwen_runtime_extract':
                self._init_qwen_runtime_audio_store()
            else:
                raise ValueError(f"Invalid audio_embedding_source: {self.audio_embedding_source}")
        if self.utterance_feature_audio and self.qwen_audio_features is None:
            raise ValueError("Utterance audio features require a precomputed_tokens audio cache")

    def _utterance_feature_row_enabled(self, sample_id):
        # Stage-1 single-utterance speaker supervision must retain its original
        # direct audio path in every comparison arm.
        return bool(getattr(self, 'utterance_feature_enable', False)) and not str(sample_id or '').startswith('Speaker_')

    def _build_utterance_feature_inputs(self, records, turn_ids_by_record):
        if not self.utterance_feature_enable:
            return {}
        selected = [
            ids if self._utterance_feature_row_enabled(record['sample_id']) else []
            for record, ids in zip(records, turn_ids_by_record)
        ]
        max_turns = max((len(ids) for ids in selected), default=0)
        batch_size = len(records)
        text_dim = int(self.qwen_text_feature_dim or self.qwen_text_hidden_size)
        audio_dim = int(self.qwen_audio_feature_dim or self.qwen_text_hidden_size)
        text = torch.zeros(batch_size, max_turns, text_dim, dtype=torch.float32)
        audio = torch.zeros(batch_size, max_turns, audio_dim, dtype=torch.float32)
        text_mask = torch.zeros(batch_size, max_turns, dtype=torch.bool)
        audio_mask = torch.zeros(batch_size, max_turns, dtype=torch.bool)
        for row, turn_ids in enumerate(selected):
            for turn, sample_id in enumerate(turn_ids):
                if self.utterance_feature_text:
                    feature = self._lookup_context_feature(self.qwen_text_features, sample_id, 'text')
                    if feature is None:
                        raise KeyError(f"Missing utterance text feature: {sample_id}")
                    text[row, turn] = torch.as_tensor(feature, dtype=torch.float32)
                    text_mask[row, turn] = True
                if self.utterance_feature_audio:
                    if str(sample_id) in self.qwen_audio_features:
                        feature = self._lookup_context_feature(self.qwen_audio_features, sample_id, 'audio')
                        audio[row, turn] = torch.as_tensor(feature, dtype=torch.float32)
                        audio_mask[row, turn] = True
        return {
            'utterance_text_features': text,
            'utterance_audio_features': audio,
            'utterance_text_mask': text_mask,
            'utterance_audio_mask': audio_mask,
            'utterance_feature_text_count': text_mask.sum(dim=1),
            'utterance_feature_audio_count': audio_mask.sum(dim=1),
            'utterance_turn_ids': selected,
        }

    def _init_iemocap_dialogue_index(self):
        dataset = str(getattr(self.args, 'dataset', '')).strip().lower()
        if dataset != 'iemocap':
            return

        iemocap_pkl = getattr(self.args, 'audio_iemocap_pkl', None)
        if iemocap_pkl is None or not os.path.exists(str(iemocap_pkl)):
            default_pkl = '/n/work3/huang/projects/InstructERC/original_data/iemocap/iemocap.pkl'
            iemocap_pkl = default_pkl if os.path.exists(default_pkl) else None
        if iemocap_pkl is None:
            logger.warning("IEMOCAP metadata pkl not found; stage-1 audio sample ids may not be resolved.")
            return

        try:
            data = pickle.load(open(iemocap_pkl, 'rb'))
        except Exception as exc:
            logger.warning("Failed to load IEMOCAP metadata pkl %s: %s", iemocap_pkl, exc)
            return

        speaker_dict = data[0]
        sentence_dict = data[2]
        all_ids = data[3] + data[4] + data[5]
        for conv_id in all_ids:
            turns = sentence_dict.get(conv_id, [])
            raw_speakers = speaker_dict.get(conv_id, [])
            speaker_ids = []
            for raw_speaker in raw_speakers:
                raw = str(raw_speaker).strip().upper()
                speaker_ids.append(0 if raw == 'M' else 1)
            self.iemocap_speaker_ids_by_conv[str(conv_id)] = speaker_ids
            self.iemocap_num_turns_by_conv[str(conv_id)] = len(turns)

    def _resolve_qwen_processor_path(self):
        qwen_processor_path = getattr(self.args, 'audio_qwen_processor_path', None)
        if qwen_processor_path in (None, ''):
            qwen_processor_path = getattr(self.args, 'model_name_or_path', None)
        if qwen_processor_path is None or not os.path.exists(qwen_processor_path):
            raise FileNotFoundError(f"Qwen processor path not found: {qwen_processor_path}")
        return qwen_processor_path

    def _init_qwen_processor(self):
        if self.qwen_processor is not None:
            return

        model_path = str(getattr(self.args, 'model_name_or_path', ''))
        try:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as exc:
            raise ValueError(f"Could not read model config for Qwen native audio path: {model_path}") from exc
        if str(getattr(model_config, 'model_type', '')).lower() != 'qwen2_5_omni':
            raise ValueError(f"audio_embedding_source={self.audio_embedding_source} requires a Qwen2.5-Omni-compatible model.")
        thinker_config = getattr(model_config, 'thinker_config', model_config)
        text_config = getattr(thinker_config, 'text_config', None)
        self.qwen_text_hidden_size = int(
            getattr(text_config, 'hidden_size', 0) or 0
        )

        qwen_processor_path = self._resolve_qwen_processor_path()
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_processor_path, trust_remote_code=True)
        self.qwen_tokenizer = getattr(self.qwen_processor, 'tokenizer', self.tokenizer)
        if getattr(self.qwen_tokenizer, 'pad_token', None) is None:
            if getattr(self.tokenizer, 'pad_token', None) is not None:
                self.qwen_tokenizer.pad_token = self.tokenizer.pad_token
            elif getattr(self.qwen_tokenizer, 'eos_token', None) is not None:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        self.qwen_tokenizer.padding_side = getattr(self.tokenizer, 'padding_side', 'left')
        additional_special_tokens = []
        if self.audio_enable and self.audio_embedding_source == 'precomputed_tokens':
            additional_special_tokens.append(AUDIO_UTTERANCE_TOKEN)
        if self.video_enable and self.video_embedding_source == 'precomputed_tokens':
            additional_special_tokens.append(VISUAL_UTTERANCE_TOKEN)
        if str(
            getattr(self.args, "personality_pair_mode", "none") or "none"
        ).strip().lower() != "none":
            additional_special_tokens.extend(
                [
                    PERSONALITY_SPEAKER_0_TOKEN,
                    PERSONALITY_SPEAKER_1_TOKEN,
                ]
            )
        if additional_special_tokens:
            self.qwen_tokenizer.add_special_tokens(
                {'additional_special_tokens': additional_special_tokens}
            )
        self.qwen_sampling_rate = int(getattr(self.qwen_processor.feature_extractor, 'sampling_rate', 16000))
        self.qwen_hop_length = int(getattr(self.qwen_processor.feature_extractor, 'hop_length', 160))

    def _build_qwen_missing_input_features(self):
        waveform = np.zeros(self.qwen_sampling_rate // 10, dtype=np.float32)
        max_length = int(math.ceil(float(waveform.shape[0]) / float(max(1, self.qwen_hop_length))) * self.qwen_hop_length)
        if waveform.shape[0] < max_length:
            waveform = np.pad(waveform, (0, max_length - waveform.shape[0]), mode='constant')
        fe_out = self.qwen_processor.feature_extractor(
            waveform,
            sampling_rate=self.qwen_sampling_rate,
            return_attention_mask=True,
            padding='longest',
            return_tensors='np',
        )
        input_features = fe_out['input_features'][0].astype(np.float32)
        attention_mask = fe_out.get('attention_mask', None)
        if attention_mask is None:
            feature_length = int(input_features.shape[-1])
        else:
            feature_length = int(attention_mask[0].sum())
        feature_length = max(1, min(feature_length, int(input_features.shape[-1])))
        return np.ascontiguousarray(input_features[:, :feature_length]), feature_length

    def _init_qwen_precomputed_text_feature_store(self):
        feature_pkl = self.text_feature_pkl
        if feature_pkl is None or not os.path.exists(str(feature_pkl)):
            raise FileNotFoundError(f"Precomputed text feature pkl not found: {feature_pkl}")
        payload = pickle.load(open(feature_pkl, 'rb'))
        if not isinstance(payload, dict) or payload.get('schema') != 'qwen_text_utterance_features_v1':
            raise ValueError(
                "text_feature_pkl must point to a qwen_text_utterance_features_v1 cache."
            )
        sample_features = payload.get('sample_features')
        if not isinstance(sample_features, dict) or not sample_features:
            raise ValueError("Precomputed text feature cache has no sample_features.")
        feature_dim = int(payload.get('feature_dim', 0))
        clean_features = {}
        for sample_id, entry in sample_features.items():
            features = entry.get('features') if isinstance(entry, dict) else entry
            array = np.asarray(features, dtype=np.float16)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[0] <= 0:
                raise ValueError(f"Invalid text feature shape for {sample_id}: {array.shape}")
            if feature_dim <= 0:
                feature_dim = int(array.shape[1])
            if int(array.shape[1]) != feature_dim:
                raise ValueError(
                    f"Text feature dim mismatch for {sample_id}: "
                    f"{array.shape[1]} != {feature_dim}"
                )
            clean_features[str(sample_id)] = np.ascontiguousarray(array)
        self.qwen_text_features = clean_features
        self.qwen_text_feature_dim = feature_dim

    def _init_qwen_precomputed_audio_store(self):
        dataset = str(getattr(self.args, 'dataset', '')).strip().lower()
        if dataset not in ('iemocap', 'meld', 'msp_improv'):
            raise ValueError("audio_embedding_source=precomputed_pkl supports iemocap/meld/msp_improv native-mel caches.")
        if self.audio_feature_pkl is None or not os.path.exists(self.audio_feature_pkl):
            raise FileNotFoundError(f"Audio feature pkl not found: {self.audio_feature_pkl}")

        self._init_qwen_processor()

        rank = os.getenv("RANK", "-1")
        verbose_audio = os.getenv("INSTRUCTERC_VERBOSE_AUDIO_LOAD", "0") == "1"
        load_start = time.time()
        if rank in ("0", "-1") and verbose_audio:
            print(f"[INFO][rank={rank}] Loading audio feature pkl: {self.audio_feature_pkl}", flush=True)
        payload = pickle.load(open(self.audio_feature_pkl, 'rb'))
        if rank in ("0", "-1") and verbose_audio:
            print(f"[INFO][rank={rank}] Loaded audio feature pkl in {time.time() - load_start:.1f}s", flush=True)

        if not isinstance(payload, dict) or payload.get('schema') != 'qwen_native_mel_v1':
            raise ValueError(
                "precomputed_pkl now requires a qwen_native_mel_v1 cache. "
                "Rebuild it with audio_utils/extract_iemocap_qwen_processor_features.py."
            )

        sample_features = payload.get('sample_features', None)
        if not isinstance(sample_features, dict) or len(sample_features) == 0:
            raise ValueError("Qwen native mel cache must contain non-empty sample_features.")

        sample_lengths = payload.get('sample_feature_lengths', {})
        feature_dim = int(payload.get('feature_dim', 128))
        self.qwen_sampling_rate = int(payload.get('sampling_rate', self.qwen_sampling_rate))
        self.qwen_hop_length = int(payload.get('hop_length', self.qwen_hop_length))

        clean_features = {}
        clean_lengths = {}
        for sample_id, feat in sample_features.items():
            arr = np.asarray(feat, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Cached Qwen mel feature must be 2-D [feature_dim, frames]: {sample_id}")
            if arr.shape[0] != feature_dim:
                raise ValueError(
                    f"Cached Qwen mel feature for {sample_id} has shape {arr.shape}; "
                    f"expected [{feature_dim}, frames]."
                )
            length = int(sample_lengths.get(sample_id, arr.shape[-1]))
            length = max(1, min(length, int(arr.shape[-1])))
            clean_features[str(sample_id)] = np.ascontiguousarray(arr[:, :length])
            clean_lengths[str(sample_id)] = length

        self.qwen_mel_features = clean_features
        self.qwen_mel_lengths = clean_lengths
        self.audio_feature_dim = feature_dim
        self.qwen_missing_input_features, self.qwen_missing_feature_length = self._build_qwen_missing_input_features()

        if rank in ("0", "-1") and verbose_audio:
            print(
                f"[INFO][rank={rank}] Qwen native mel cache samples={len(self.qwen_mel_features)} "
                f"feature_dim={self.audio_feature_dim}",
                flush=True,
            )

    def _init_qwen_runtime_audio_store(self):
        dataset = str(getattr(self.args, 'dataset', '')).strip().lower()
        if dataset != 'iemocap':
            raise ValueError("audio_embedding_source=qwen_runtime_extract currently supports only iemocap dataset.")

        iemocap_root = getattr(self.args, 'audio_iemocap_root', None)
        iemocap_pkl = getattr(self.args, 'audio_iemocap_pkl', None)

        if iemocap_root is None or not os.path.exists(iemocap_root):
            raise FileNotFoundError(f"IEMOCAP root not found: {iemocap_root}")
        if iemocap_pkl is None or not os.path.exists(iemocap_pkl):
            raise FileNotFoundError(f"IEMOCAP meta pkl not found: {iemocap_pkl}")

        self._init_qwen_processor()

        conv_turn_to_uttid = _build_conv_turn_to_uttid_map(iemocap_pkl, iemocap_root)
        self.sampleid_to_wavpath = {}
        for sample_id, utt_id in conv_turn_to_uttid.items():
            if utt_id is None:
                continue
            conv_id = sample_id.rsplit('_', 1)[0]
            session = _dialogue_to_session(conv_id)
            if session is None:
                continue
            wav_path = os.path.join(iemocap_root, session, 'sentences', 'wav', conv_id, f'{utt_id}.wav')
            if os.path.exists(wav_path):
                self.sampleid_to_wavpath[sample_id] = wav_path

        if len(self.sampleid_to_wavpath) == 0:
            raise RuntimeError("No valid wav paths resolved for qwen_runtime_extract.")

    def _init_qwen_precomputed_audio_token_store(self):
        dataset = str(getattr(self.args, 'dataset', '')).strip().lower()
        if dataset not in ('iemocap', 'meld'):
            raise ValueError(
                "audio_embedding_source=precomputed_tokens currently supports "
                "iemocap and meld datasets."
            )
        feature_pkl = self.audio_feature_pkl
        if feature_pkl is None or not os.path.exists(str(feature_pkl)):
            raise FileNotFoundError(
                f"Precomputed audio token pkl not found: {feature_pkl}"
            )

        self._init_qwen_processor()
        payload = pickle.load(open(feature_pkl, 'rb'))
        if (
            not isinstance(payload, dict)
            or payload.get('schema') != 'qwen_audio_temporal_tokens_v1'
        ):
            raise ValueError(
                "audio_embedding_source=precomputed_tokens requires a "
                "qwen_audio_temporal_tokens_v1 cache."
            )
        sample_features = payload.get('sample_features')
        if not isinstance(sample_features, dict) or not sample_features:
            raise ValueError("Precomputed audio token cache has no sample_features.")

        feature_dim = int(payload.get('feature_dim', 0))
        clean_features = {}
        for sample_id, entry in sample_features.items():
            features = entry.get('features') if isinstance(entry, dict) else entry
            array = np.asarray(features, dtype=np.float16)
            if array.ndim != 2 or array.shape[0] <= 0:
                raise ValueError(
                    f"Invalid compressed audio feature shape for {sample_id}: "
                    f"{array.shape}"
                )
            if feature_dim <= 0:
                feature_dim = int(array.shape[1])
            if int(array.shape[1]) != feature_dim:
                raise ValueError(
                    f"Compressed audio feature dim mismatch for {sample_id}: "
                    f"{array.shape[1]} != {feature_dim}"
                )
            clean_features[str(sample_id)] = np.ascontiguousarray(array)

        self.qwen_audio_features = clean_features
        self.qwen_audio_feature_dim = feature_dim
        self.audio_feature_dim = feature_dim
        if (
            self.qwen_text_hidden_size > 0
            and feature_dim != self.qwen_text_hidden_size
        ):
            raise ValueError(
                "Compressed audio features must match the Qwen text hidden "
                f"size: {feature_dim} != {self.qwen_text_hidden_size}"
            )

    def _init_qwen_video_store(self):
        dataset = str(getattr(self.args, 'dataset', '')).strip().lower()
        if dataset != 'iemocap':
            raise ValueError("video_enable currently supports only iemocap dataset.")

        manifest = self.video_manifest_jsonl
        if manifest is None or not os.path.exists(str(manifest)):
            raise FileNotFoundError(f"Video manifest jsonl not found: {manifest}")

        iemocap_root = getattr(self.args, 'audio_iemocap_root', None)
        iemocap_pkl = getattr(self.args, 'audio_iemocap_pkl', None)
        if iemocap_root is None or not os.path.exists(str(iemocap_root)):
            raise FileNotFoundError(f"IEMOCAP root not found for video id alignment: {iemocap_root}")
        if iemocap_pkl is None or not os.path.exists(str(iemocap_pkl)):
            raise FileNotFoundError(f"IEMOCAP metadata pkl not found for video id alignment: {iemocap_pkl}")

        uttid_to_video = {}
        turnkey_to_video = {}
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                utt_id = rec.get('utterance_id') or rec.get('sample_id')
                video_path = rec.get('video_path')
                if utt_id is None or video_path is None:
                    continue
                if os.path.exists(str(video_path)):
                    uttid_to_video[str(utt_id)] = str(video_path)
                    turn_key = rec.get('turn_key')
                    if turn_key is not None:
                        turnkey_to_video[str(turn_key)] = str(video_path)

        conv_turn_to_uttid = _build_conv_turn_to_uttid_map(iemocap_pkl, iemocap_root)
        sampleid_to_video = dict(turnkey_to_video)
        for sample_id, utt_id in conv_turn_to_uttid.items():
            if utt_id is not None and str(utt_id) in uttid_to_video:
                sampleid_to_video[str(sample_id)] = uttid_to_video[str(utt_id)]
        for utt_id, video_path in uttid_to_video.items():
            sampleid_to_video[str(utt_id)] = video_path

        if len(sampleid_to_video) == 0:
            raise RuntimeError(f"No valid video clips resolved from manifest: {manifest}")
        self.sampleid_to_video_path = sampleid_to_video

    def _init_qwen_precomputed_video_store(self):
        feature_pkl = self.video_feature_pkl
        if feature_pkl is None or not os.path.exists(str(feature_pkl)):
            raise FileNotFoundError(f"Precomputed video feature pkl not found: {feature_pkl}")
        payload = pickle.load(open(feature_pkl, 'rb'))
        if not isinstance(payload, dict) or payload.get('schema') != 'qwen_visual_temporal_tokens_v1':
            raise ValueError(
                "video_embedding_source=precomputed_tokens requires a "
                "qwen_visual_temporal_tokens_v1 cache."
            )
        sample_features = payload.get('sample_features')
        if not isinstance(sample_features, dict) or not sample_features:
            raise ValueError("Precomputed video cache has no sample_features.")
        feature_dim = int(payload.get('feature_dim', 0))
        clean_features = {}
        for sample_id, entry in sample_features.items():
            features = entry.get('features') if isinstance(entry, dict) else entry
            array = np.asarray(features, dtype=np.float16)
            if array.ndim != 2 or array.shape[0] <= 0:
                raise ValueError(f"Invalid visual feature shape for {sample_id}: {array.shape}")
            if feature_dim <= 0:
                feature_dim = int(array.shape[1])
            if int(array.shape[1]) != feature_dim:
                raise ValueError(
                    f"Visual feature dim mismatch for {sample_id}: "
                    f"{array.shape[1]} != {feature_dim}"
                )
            clean_features[str(sample_id)] = np.ascontiguousarray(array)
        self.qwen_visual_features = clean_features
        self.qwen_visual_feature_dim = feature_dim
        if len(clean_features) < len(self.sampleid_to_video_path) // 2:
            logger.warning(
                "Visual cache contains only %d keys while video mapping contains %d keys.",
                len(clean_features),
                len(self.sampleid_to_video_path),
            )

    def _extract_sample_ids(self, batch):
        sample_ids = []
        for item in batch:
            if len(item) >= 3 and self.mode == 'dev':
                sample_ids.append(item[2])
            elif len(item) >= 4 and self.mode != 'dev' and self.args.emotion_prediction:
                sample_ids.append(item[3])
            elif len(item) >= 3 and self.mode != 'dev' and not self.args.emotion_prediction:
                sample_ids.append(item[2])
            else:
                sample_ids.append(None)
        return sample_ids

    def _get_cached_qwen_mel(self, sample_id, max_samples=None):
        sid = None if sample_id is None else str(sample_id)
        if sid is not None and self.qwen_mel_features is not None and sid in self.qwen_mel_features:
            feat = self.qwen_mel_features[sid]
            feature_length = int(self.qwen_mel_lengths.get(sid, feat.shape[-1]))
        else:
            feat = self.qwen_missing_input_features
            feature_length = int(self.qwen_missing_feature_length)

        feature_length = max(1, min(feature_length, int(feat.shape[-1])))
        if max_samples is not None and max_samples > 0:
            max_feature_length = int(math.ceil(float(max_samples) / float(max(1, self.qwen_hop_length))))
            max_feature_length = max(1, max_feature_length)
            if feature_length > max_feature_length:
                start = max(0, (feature_length - max_feature_length) // 2)
                end = start + max_feature_length
                feat = feat[:, start:end]
                feature_length = max_feature_length

        return np.ascontiguousarray(feat[:, :feature_length]), feature_length

    @staticmethod
    def _qwen_audio_token_length_from_feature_length(feature_length):
        input_length = (int(feature_length) - 1) // 2 + 1
        return max(0, (input_length - 2) // 2 + 1)

    def _validate_qwen_audio_token_alignment(self, processor_out, records=None, context=""):
        if self.audio_embedding_source == 'precomputed_tokens':
            if 'input_ids' not in processor_out:
                return
            audio_token_id = self.qwen_tokenizer.convert_tokens_to_ids(
                AUDIO_UTTERANCE_TOKEN
            )
            if audio_token_id is None or int(audio_token_id) < 0:
                return
            actual_tokens = int(
                (processor_out['input_ids'] == int(audio_token_id)).sum().item()
            )
            audio_features = processor_out.get('precomputed_audio_features')
            expected_tokens = (
                0 if audio_features is None else int(audio_features.shape[0])
            )
            if actual_tokens == expected_tokens:
                return
            sample_ids = []
            if records is not None:
                sample_ids = [
                    record.get('sample_id', None) for record in records
                ]
            raise ValueError(
                "Compressed audio token/features mismatch before model forward"
                f"{f' ({context})' if context else ''}: "
                f"audio_tokens={actual_tokens}, "
                f"expected_audio_features={expected_tokens}, "
                f"sample_ids={sample_ids}"
            )

        if 'input_ids' not in processor_out or 'feature_attention_mask' not in processor_out:
            return
        audio_token = getattr(self.qwen_processor, 'audio_token', None)
        if audio_token is None:
            audio_token = getattr(self.qwen_tokenizer, 'audio_token', '<|AUDIO|>')
        audio_token_id = self.qwen_tokenizer.convert_tokens_to_ids(audio_token)
        if audio_token_id is None or audio_token_id < 0:
            return

        actual_tokens = int((processor_out['input_ids'] == int(audio_token_id)).sum().item())
        feature_lengths = processor_out['feature_attention_mask'].sum(dim=1).tolist()
        expected_tokens = int(sum(self._qwen_audio_token_length_from_feature_length(length) for length in feature_lengths))
        if actual_tokens == expected_tokens:
            return

        sample_ids = []
        if records is not None:
            sample_ids = [record.get('sample_id', None) for record in records]
        raise ValueError(
            "Qwen audio token/features mismatch before model forward"
            f"{f' ({context})' if context else ''}: "
            f"audio_tokens={actual_tokens}, expected_audio_features={expected_tokens}, "
            f"num_audio_segments={len(feature_lengths)}, sample_ids={sample_ids}, "
            f"feature_lengths={feature_lengths[:20]}"
            f"{'...' if len(feature_lengths) > 20 else ''}"
        )

    def _validate_qwen_visual_token_alignment(self, processor_out, records=None, context=""):
        if self.video_embedding_source != 'precomputed_tokens':
            return
        if 'input_ids' not in processor_out:
            return
        visual_token_id = self.qwen_tokenizer.convert_tokens_to_ids(VISUAL_UTTERANCE_TOKEN)
        if visual_token_id is None or int(visual_token_id) < 0:
            return
        actual_tokens = int(
            (processor_out['input_ids'] == int(visual_token_id)).sum().item()
        )
        visual_features = processor_out.get('precomputed_visual_features')
        expected_tokens = 0 if visual_features is None else int(visual_features.shape[0])
        if actual_tokens == expected_tokens:
            return
        sample_ids = []
        if records is not None:
            sample_ids = [record.get('sample_id', None) for record in records]
        raise ValueError(
            "Visual token/features mismatch before model forward"
            f"{f' ({context})' if context else ''}: "
            f"visual_tokens={actual_tokens}, expected_visual_features={expected_tokens}, "
            f"sample_ids={sample_ids}"
        )

    def _build_cached_qwen_audio_inputs(self, flat_audio_refs):
        features = []
        feature_lengths = []
        audio_token_lengths = []
        for ref in flat_audio_refs:
            feat, feature_length = self._get_cached_qwen_mel(
                ref.get('sample_id', None),
                max_samples=ref.get('audio_sample_limit', None),
            )
            features.append(feat)
            feature_lengths.append(feature_length)
            audio_token_lengths.append(self._qwen_audio_token_length_from_feature_length(feature_length))

        if len(features) == 0:
            return {}, audio_token_lengths

        max_feature_length = max(feature_lengths)
        feature_dim = int(features[0].shape[0])
        input_features = torch.zeros((len(features), feature_dim, max_feature_length), dtype=torch.float32)
        feature_attention_mask = torch.zeros((len(features), max_feature_length), dtype=torch.long)
        for idx, (feat, feature_length) in enumerate(zip(features, feature_lengths)):
            input_features[idx, :, :feature_length] = torch.as_tensor(feat[:, :feature_length], dtype=torch.float32)
            feature_attention_mask[idx, :feature_length] = 1

        return {
            'input_features': input_features,
            'feature_attention_mask': feature_attention_mask,
            'audio_feature_lengths': torch.tensor(feature_lengths, dtype=torch.long),
        }, audio_token_lengths

    def _build_cached_qwen_audio_token_inputs(self, flat_audio_refs):
        features = []
        token_lengths = []
        for ref in flat_audio_refs:
            sample_id = str(ref.get('sample_id', None))
            array = self.qwen_audio_features.get(sample_id)
            if array is None:
                raise KeyError(
                    f"Missing precomputed compressed audio features for {sample_id}"
                )
            features.append(torch.as_tensor(array, dtype=torch.float32))
            token_lengths.append(int(array.shape[0]))
        if not features:
            return {}, token_lengths
        return {
            'precomputed_audio_features': torch.cat(features, dim=0)
        }, token_lengths

    def _parse_iemocap_turn_key(self, turn_key):
        if turn_key is None:
            return None
        turn_key = str(turn_key)
        if '_' not in turn_key:
            return None
        conv_id, turn_str = turn_key.rsplit('_', 1)
        if _dialogue_to_session(conv_id) is None:
            return None
        try:
            turn = int(turn_str)
        except ValueError:
            return None
        if turn < 0:
            return None
        return conv_id, turn

    def _parse_iemocap_utterance_sample_id(self, sample_id):
        if sample_id is None:
            return None
        sample_id = str(sample_id)
        if sample_id.startswith('Speaker_') or sample_id.startswith('Personality_'):
            return None
        return self._parse_iemocap_turn_key(sample_id)

    def _parse_meld_utterance_sample_id(self, sample_id):
        if sample_id is None:
            return None
        match = re.match(r"^(?:(train|valid|dev|test)_)?dia(\d+)_utt(\d+)$", str(sample_id))
        if match is None:
            return None
        split = match.group(1)
        if split == 'dev':
            split = 'valid'
        return split, int(match.group(2)), int(match.group(3))

    def _parse_iemocap_speaker_sample_id(self, sample_id):
        sample_id = str(sample_id)
        if not sample_id.startswith('Speaker_'):
            return None
        return self._parse_iemocap_turn_key(sample_id[len('Speaker_'):])

    def _parse_iemocap_personality_utterance_sample_id(self, sample_id):
        sample_id = str(sample_id)
        prefix = 'Personality_utterance_'
        if not sample_id.startswith(prefix):
            return None
        return self._parse_iemocap_turn_key(sample_id[len(prefix):])

    def _parse_iemocap_personality_speaker_sample_id(self, sample_id, setting):
        sample_id = str(sample_id)
        prefix = f'Personality_{setting}_'
        if not sample_id.startswith(prefix):
            return None
        payload = sample_id[len(prefix):]
        if '_' not in payload:
            return None
        conv_id, speaker_str = payload.rsplit('_', 1)
        if _dialogue_to_session(conv_id) is None:
            return None
        try:
            speaker_idx = int(speaker_str)
        except ValueError:
            return None
        return conv_id, speaker_idx

    def _speaker_turn_sample_ids(self, conv_id, speaker_idx):
        speaker_ids = self.iemocap_speaker_ids_by_conv.get(str(conv_id), [])
        return [
            f'{conv_id}_{turn_idx}'
            for turn_idx, turn_speaker in enumerate(speaker_ids)
            if int(turn_speaker) == int(speaker_idx)
        ]

    def _full_conversation_turn_sample_ids(self, conv_id):
        num_turns = int(self.iemocap_num_turns_by_conv.get(str(conv_id), 0))
        return [f'{conv_id}_{turn_idx}' for turn_idx in range(num_turns)]

    def _selected_qwen_turn_sample_ids(self, sample_id):
        sample_id_text = str(sample_id or "")
        meld_parsed = self._parse_meld_utterance_sample_id(sample_id_text)
        if meld_parsed is not None:
            split, dialogue_id, utterance_id = meld_parsed
            end = utterance_id + 1
            start = 0
            if self.audio_context_window > 0:
                start = max(0, end - self.audio_context_window)
            prefix = f'{split}_' if split else ''
            turn_ids = [f'{prefix}dia{dialogue_id}_utt{idx}' for idx in range(start, end)]
            if (
                getattr(self, 'audio_embedding_source', None) == 'precomputed_tokens'
                and hasattr(self, 'qwen_audio_features')
            ):
                available_turn_ids = [
                    turn_id for turn_id in turn_ids if turn_id in self.qwen_audio_features
                ]
                if available_turn_ids:
                    return available_turn_ids
            return turn_ids

        pair_prefix = "PersonalityPair_full_"
        if sample_id_text.startswith(pair_prefix):
            conv_id = sample_id_text[len(pair_prefix):]
            if _dialogue_to_session(conv_id) is not None:
                return self._full_conversation_turn_sample_ids(conv_id)

        speaker_parsed = self._parse_iemocap_speaker_sample_id(sample_id)
        if speaker_parsed is not None:
            conv_id, turn = speaker_parsed
            return [f'{conv_id}_{turn}']

        personality_utt_parsed = self._parse_iemocap_personality_utterance_sample_id(sample_id)
        if personality_utt_parsed is not None:
            conv_id, turn = personality_utt_parsed
            return [f'{conv_id}_{turn}']

        personality_conv_parsed = self._parse_iemocap_personality_speaker_sample_id(sample_id, 'conversation')
        if personality_conv_parsed is not None:
            conv_id, speaker_idx = personality_conv_parsed
            return self._speaker_turn_sample_ids(conv_id, speaker_idx)

        personality_full_parsed = self._parse_iemocap_personality_speaker_sample_id(sample_id, 'full')
        if personality_full_parsed is not None:
            conv_id, speaker_idx = personality_full_parsed
            if self.personality_media_scope == 'target':
                return self._speaker_turn_sample_ids(conv_id, speaker_idx)
            return self._full_conversation_turn_sample_ids(conv_id)

        parsed = self._parse_iemocap_utterance_sample_id(sample_id)
        if parsed is None:
            return []
        conv_id, turn = parsed
        end = turn + 1
        start = 0
        if self.audio_context_window > 0:
            start = max(0, end - self.audio_context_window)
        return [f'{conv_id}_{idx}' for idx in range(start, end)]

    @staticmethod
    def _record_audio_turn_sample_ids(record):
        meta = record.get('meta', {}) if isinstance(record, dict) else {}
        turn_ids = meta.get('audio_turn_sample_ids')
        if turn_ids is None:
            turn_ids = record.get('audio_turn_sample_ids') if isinstance(record, dict) else None
        if not isinstance(turn_ids, (list, tuple)):
            return None
        clean_ids = [str(item) for item in turn_ids if item is not None and str(item)]
        return clean_ids if clean_ids else None

    def _effective_qwen_turn_sample_ids(self, record):
        parsed = self._parse_iemocap_utterance_sample_id(record['sample_id'])
        if (
            self._utterance_feature_row_enabled(record['sample_id'])
            and parsed is not None
            and int(getattr(self, 'utterance_feature_history_window', -1)) >= 0
        ):
            conv_id, turn = parsed
            start = max(0, turn - int(getattr(self, 'utterance_feature_history_window', -1)))
            return [f'{conv_id}_{idx}' for idx in range(start, turn + 1)]
        turn_ids = (
            self._record_audio_turn_sample_ids(record)
            or self._selected_qwen_turn_sample_ids(record['sample_id'])
        )
        # MELD records store every turn in the textual context. That metadata
        # must not override the requested audio context window.
        if (
            self._parse_meld_utterance_sample_id(record['sample_id']) is not None
            and self.audio_context_window > 0
        ):
            turn_ids = turn_ids[-self.audio_context_window:]
        return turn_ids

    def _qwen_audio_num_samples(self, sample_id):
        if self.audio_embedding_source == 'precomputed_pkl':
            _, feature_length = self._get_cached_qwen_mel(sample_id)
            return int(feature_length) * int(max(1, self.qwen_hop_length))

        wav_path = self.sampleid_to_wavpath.get(str(sample_id), None)
        if wav_path is None:
            return self.qwen_sampling_rate // 10
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
        except Exception:
            return self.qwen_sampling_rate // 10
        if sample_rate != self.qwen_sampling_rate and sample_rate > 0:
            return int(math.ceil(frames * float(self.qwen_sampling_rate) / float(sample_rate)))
        return int(frames)

    @staticmethod
    def _center_crop_audio_array(audio_array, max_samples):
        if max_samples is None or max_samples <= 0 or audio_array.shape[0] <= max_samples:
            return audio_array
        start = max(0, (audio_array.shape[0] - max_samples) // 2)
        return audio_array[start:start + max_samples]

    def _load_qwen_audio_array(self, sample_id, max_samples=None):
        wav_path = self.sampleid_to_wavpath.get(str(sample_id), None)
        if wav_path is None:
            waveform = torch.zeros(self.qwen_sampling_rate // 10, dtype=torch.float32)
        else:
            waveform, sr = _load_wav_mono(wav_path)
            if sr != self.qwen_sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.qwen_sampling_rate)
            waveform = waveform.to(torch.float32)
            if waveform.numel() == 0:
                waveform = torch.zeros(self.qwen_sampling_rate // 10, dtype=torch.float32)
        audio_array = waveform.numpy()
        audio_array = self._center_crop_audio_array(audio_array, max_samples)
        return audio_array

    @staticmethod
    def _append_qwen_text(content_items, text):
        if text:
            content_items.append({'type': 'text', 'text': text})

    def _append_qwen_audio(self, content_items, audio_refs, sample_id, audio_sample_limit=None):
        if self.audio_embedding_source == 'precomputed_tokens':
            content_items.append({'type': 'text', 'text': AUDIO_UTTERANCE_TOKEN})
            audio_refs.append(
                {'sample_id': sample_id, 'audio_sample_limit': None}
            )
            return
        audio_item = {'type': 'audio'}
        wav_path = self.sampleid_to_wavpath.get(str(sample_id), None) if self.sampleid_to_wavpath is not None else None
        if self.audio_embedding_source == 'qwen_runtime_extract' and wav_path is not None:
            audio_item['audio'] = wav_path
        content_items.append(audio_item)
        audio_refs.append({'sample_id': sample_id, 'audio_sample_limit': audio_sample_limit})

    def _append_qwen_video(self, content_items, visual_refs, sample_id):
        video_path = self.sampleid_to_video_path.get(str(sample_id), None)
        if video_path is None:
            return
        if self.video_embedding_source == 'precomputed_tokens':
            content_items.append({'type': 'text', 'text': VISUAL_UTTERANCE_TOKEN})
            visual_refs.append(str(sample_id))
            return
        video_item = {'type': 'video', 'video': video_path}
        if self.video_fps > 0:
            video_item['fps'] = self.video_fps
        if self.video_resized_height > 0:
            video_item['resized_height'] = self.video_resized_height
        if self.video_resized_width > 0:
            video_item['resized_width'] = self.video_resized_width
        content_items.append(video_item)

    def _append_qwen_modalities(
        self,
        content_items,
        audio_refs,
        visual_refs,
        sample_id,
        audio_sample_limit=None,
    ):
        if self.audio_direct_enable and self.audio_input_mode != 'text_only':
            self._append_qwen_audio(content_items, audio_refs, sample_id, audio_sample_limit)
        if self.video_direct_enable:
            self._append_qwen_video(content_items, visual_refs, sample_id)

    @staticmethod
    def _target_speaker_from_iemocap_uttid(sample_id):
        suffix = str(sample_id).rsplit('_', 1)[-1]
        if suffix.startswith('F'):
            return 'female speaker'
        if suffix.startswith('M'):
            return 'male speaker'
        return 'target speaker'

    def _video_only_instruction(self, sample_id=None):
        dataset = str(getattr(self.args, 'dataset', '')).strip().lower()
        labels = CONTRASTIVE_LABELS.get(dataset, CONTRASTIVE_LABELS.get('iemocap'))
        label_text = ", ".join(labels)
        target_speaker = self._target_speaker_from_iemocap_uttid(sample_id) if sample_id is not None else 'target speaker'
        return (
            "You are an expert in sentiment and emotion analysis. "
            f"The target speaker is the {target_speaker}. "
            "If multiple people are visible, focus on the target speaker rather than the listener. "
            "Based only on the provided video clip, select the target speaker's emotional label "
            f"from <{label_text}>. Output only one label word from the candidate set and nothing else:"
        )

    def _build_video_only_content_items(self, sample_id, turn_sample_ids):
        content_items = []
        audio_refs = []
        visual_refs = []
        parsed = self._parse_iemocap_utterance_sample_id(sample_id)
        if parsed is not None:
            conv_id, turn = parsed
            video_sample_id = f'{conv_id}_{turn}'
        elif len(turn_sample_ids) > 0:
            video_sample_id = turn_sample_ids[-1]
        else:
            video_sample_id = sample_id
        self._append_qwen_video(content_items, visual_refs, video_sample_id)
        self._append_qwen_text(content_items, self._video_only_instruction(video_sample_id))
        return content_items, audio_refs, visual_refs

    @staticmethod
    def _find_first_conversation_block(text):
        # Match the actual dialogue delimiter, not the explanatory quoted "'### ###'".
        open_match = re.search(r"###\s*(?=\t?\s*Speaker_\d+:\")", text)
        if open_match is None:
            return None
        content_start = open_match.end()
        close_match = re.search(r"\s###\s", text[content_start:])
        if close_match is None:
            return None
        block_end = content_start + close_match.start()
        return text[:content_start], text[content_start:block_end], text[block_end:]

    @staticmethod
    def _speaker_turn_matches(block_text):
        # Do not require a leading tab: the first turn immediately follows the
        # opening delimiter while later turns are tab-separated. Requiring the
        # tab silently dropped turn 0 and shifted every direct-audio alignment.
        return list(re.finditer(r"Speaker_\d+\*?\s*:\s*\"[^\"]*\"", block_text))

    def _build_generic_media_only_content_items(self, prompt_text, turn_sample_ids, audio_sample_limit=None):
        content_items = []
        audio_refs = []
        visual_refs = []
        marker_matches = list(re.finditer(re.escape(AUDIO_UTTERANCE_MARKER), prompt_text))
        indexed_turns = {}
        indexed_alignment = False
        if len(marker_matches) > 1 and len(turn_sample_ids) > 0:
            parsed_turns = [self._parse_iemocap_turn_key(turn_sid) for turn_sid in turn_sample_ids]
            conv_ids = {parsed[0] for parsed in parsed_turns if parsed is not None}
            if (
                len(conv_ids) == 1
                and all(parsed is not None for parsed in parsed_turns)
                and len(marker_matches)
                == int(self.iemocap_num_turns_by_conv.get(next(iter(conv_ids)), -1))
            ):
                indexed_turns = {parsed[1]: turn_sid for parsed, turn_sid in zip(parsed_turns, turn_sample_ids)}
                indexed_alignment = True

        if indexed_alignment:
            cursor = 0
            for marker_index, marker_match in enumerate(marker_matches):
                self._append_qwen_text(content_items, prompt_text[cursor:marker_match.end()])
                turn_sid = indexed_turns.get(marker_index)
                if turn_sid is not None:
                    self._append_qwen_modalities(
                        content_items,
                        audio_refs,
                        visual_refs,
                        turn_sid,
                        audio_sample_limit,
                    )
                cursor = marker_match.end()
            self._append_qwen_text(content_items, prompt_text[cursor:])
            return content_items, audio_refs, visual_refs

        usable_count = min(len(marker_matches), len(turn_sample_ids))
        cursor = 0

        for marker_match, turn_sid in zip(marker_matches[:usable_count], turn_sample_ids[:usable_count]):
            self._append_qwen_text(content_items, prompt_text[cursor:marker_match.end()])
            self._append_qwen_modalities(
                content_items,
                audio_refs,
                visual_refs,
                turn_sid,
                audio_sample_limit,
            )
            cursor = marker_match.end()

        self._append_qwen_text(content_items, prompt_text[cursor:])

        for turn_sid in turn_sample_ids[usable_count:]:
            self._append_qwen_modalities(
                content_items,
                audio_refs,
                visual_refs,
                turn_sid,
                audio_sample_limit,
            )

        return content_items, audio_refs, visual_refs

    def _build_meld_content_items(self, prompt_text, turn_sample_ids, audio_sample_limit=None):
        """Attach selected MELD audio to its dialogue turn, not the first marker."""
        block_parts = self._find_first_conversation_block(prompt_text)
        if block_parts is None:
            raise ValueError("MELD audio prompt has no dialogue block.")
        prefix, block_text, _ = block_parts
        # In audio-only prompts the masked quotation itself also contains an
        # audio marker. The marker following its closing quote is the actual
        # per-turn insertion point from the MELD data builder.
        marker_matches = list(re.finditer(
            r'"\s*' + re.escape(AUDIO_UTTERANCE_MARKER), block_text
        ))
        if len(marker_matches) < len(turn_sample_ids):
            raise ValueError(
                "MELD dialogue has fewer audio markers than selected turns: "
                f"{len(marker_matches)} < {len(turn_sample_ids)}"
            )
        selected_matches = marker_matches[-len(turn_sample_ids):] if turn_sample_ids else []
        content_items, audio_refs, visual_refs = [], [], []
        cursor = 0
        for match, turn_sid in zip(selected_matches, turn_sample_ids):
            end = len(prefix) + match.end()
            self._append_qwen_text(content_items, prompt_text[cursor:end])
            self._append_qwen_modalities(
                content_items, audio_refs, visual_refs, turn_sid, audio_sample_limit
            )
            cursor = end
        self._append_qwen_text(content_items, prompt_text[cursor:])
        return content_items, audio_refs, visual_refs

    def _build_qwen_content_items(self, prompt_text, sample_id, turn_sample_ids, audio_sample_limit=None):
        content_items = []
        audio_refs = []
        visual_refs = []

        if self._utterance_feature_row_enabled(sample_id) or (
            getattr(self, 'stage1_speaker_audio_only', False) and not str(sample_id or '').startswith('Speaker_')
        ):
            # The full transcript and instructions remain intact; only literal
            # media markers and the old direct-audio stream are replaced.
            self._append_qwen_text(content_items, strip_audio_markers_for_text_only(prompt_text))
            return content_items, audio_refs, visual_refs

        if (
            self.video_enable
            and not self.text_enable
            and not self.audio_enable
            and bool(getattr(self.args, 'emotion_prediction', False))
        ):
            return self._build_video_only_content_items(sample_id, turn_sample_ids)

        if self.audio_input_mode == 'audio_only' or not self.text_enable:
            prompt_text = mask_prompt_content_for_audio_only(prompt_text)
        elif self.audio_input_mode == 'text_only' or not self.audio_enable:
            prompt_text = strip_audio_markers_for_text_only(prompt_text)

        if (
            self.audio_token_placement == 'tail'
            and self.audio_direct_enable
            and self.audio_input_mode != 'text_only'
        ):
            # Keep the text prompt intact, but move the same ordered media
            # stream used by the aligned path behind the final instruction.
            # ERC's historical aligned path repeats target audio after the
            # target statement unless utterance_direct_audio_once is enabled;
            # preserve that behavior so placement is the only changed factor.
            self._append_qwen_text(
                content_items, strip_audio_markers_for_text_only(prompt_text)
            )
            tail_ids = list(turn_sample_ids)
            parsed = self._parse_iemocap_utterance_sample_id(sample_id)
            if parsed is not None and not getattr(
                self, 'utterance_direct_audio_once', False
            ):
                conv_id, turn = parsed
                tail_ids.append(f'{conv_id}_{turn}')
            for turn_sid in tail_ids:
                self._append_qwen_modalities(
                    content_items,
                    audio_refs,
                    visual_refs,
                    turn_sid,
                    audio_sample_limit,
                )
            return content_items, audio_refs, visual_refs

        if (
            self._parse_meld_utterance_sample_id(sample_id) is not None
            and self.audio_direct_enable
            and self.audio_input_mode != 'text_only'
            and AUDIO_UTTERANCE_MARKER in prompt_text
        ):
            return self._build_meld_content_items(
                prompt_text, turn_sample_ids, audio_sample_limit
            )

        if (
            (self.audio_enable or self.video_enable)
            and (not self.audio_enable or self.audio_input_mode != 'text_only' or self.video_enable)
            and len(turn_sample_ids) > 0
            and AUDIO_UTTERANCE_MARKER in prompt_text
        ):
            return self._build_generic_media_only_content_items(
                prompt_text,
                turn_sample_ids,
                audio_sample_limit=audio_sample_limit,
            )

        parsed = self._parse_iemocap_utterance_sample_id(sample_id)
        if parsed is None or len(turn_sample_ids) == 0:
            if (
                len(turn_sample_ids) > 0
                and (
                    self.video_enable
                    or self.audio_input_mode == 'audio_only'
                    or not self.text_enable
                    or self.personality_media_scope != 'legacy'
                )
            ):
                return self._build_generic_media_only_content_items(
                    prompt_text,
                    turn_sample_ids,
                    audio_sample_limit=audio_sample_limit,
                )
            self._append_qwen_text(content_items, prompt_text)
            return content_items, audio_refs, visual_refs

        conv_id, turn = parsed
        current_sample_id = f'{conv_id}_{turn}'
        block_parts = self._find_first_conversation_block(prompt_text)
        if block_parts is None:
            if self.audio_input_mode == 'audio_only' or not self.text_enable:
                return self._build_generic_media_only_content_items(
                    prompt_text,
                    turn_sample_ids,
                    audio_sample_limit=audio_sample_limit,
                )
            self._append_qwen_text(content_items, prompt_text)
            self._append_qwen_modalities(
                content_items,
                audio_refs,
                visual_refs,
                current_sample_id,
                audio_sample_limit,
            )
            return content_items, audio_refs, visual_refs

        prefix, block_text, suffix = block_parts
        turn_matches = self._speaker_turn_matches(block_text)
        if len(turn_matches) == 0:
            self._append_qwen_text(content_items, prompt_text)
            self._append_qwen_modalities(
                content_items,
                audio_refs,
                visual_refs,
                current_sample_id,
                audio_sample_limit,
            )
            return content_items, audio_refs, visual_refs

        usable_count = min(len(turn_matches), len(turn_sample_ids))
        kept_matches = turn_matches[-usable_count:]
        kept_turn_sample_ids = turn_sample_ids[-usable_count:]

        self._append_qwen_text(content_items, prefix)
        first_match_start = kept_matches[0].start()
        self._append_qwen_text(content_items, block_text[:first_match_start])

        cursor = first_match_start
        for match, turn_sid in zip(kept_matches, kept_turn_sample_ids):
            self._append_qwen_text(content_items, block_text[cursor:match.end()])
            self._append_qwen_modalities(
                content_items,
                audio_refs,
                visual_refs,
                turn_sid,
                audio_sample_limit,
            )
            cursor = match.end()
        self._append_qwen_text(content_items, block_text[cursor:])

        target_match = re.search(
            r"Please select the emotional label of <\s*Speaker_\d+\*?\s*:\s*\"[^\"]*\"\s*>",
            suffix,
        )
        if target_match is None:
            self._append_qwen_text(content_items, suffix)
        else:
            self._append_qwen_text(content_items, suffix[:target_match.end()])
            if not getattr(self, 'utterance_direct_audio_once', False):
                self._append_qwen_modalities(
                    content_items,
                    audio_refs,
                    visual_refs,
                    current_sample_id,
                    audio_sample_limit,
                )
            self._append_qwen_text(content_items, suffix[target_match.end():])

        return content_items, audio_refs, visual_refs

    def _qwen_record_audio_source_ids(self, record, turn_sample_ids):
        parsed = self._parse_iemocap_utterance_sample_id(record['sample_id'])
        if parsed is None:
            return list(turn_sample_ids)
        conv_id, turn = parsed
        current_sample_id = f'{conv_id}_{turn}'
        return list(dict.fromkeys(list(turn_sample_ids) + [current_sample_id]))

    def _extract_text_target_sid(self, item):
        if self.mode == 'dev':
            text = item[0]
            target = item[1] if len(item) > 1 else None
            sid = item[2] if len(item) > 2 else None
            meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
            return text, target, sid, meta

        if self.args.emotion_prediction:
            raise ValueError(f"audio_embedding_source={self.audio_embedding_source} does not support emotion_prediction=True.")

        text = item[0]
        target = item[1] if len(item) > 1 else None
        sid = item[2] if len(item) > 2 else None
        meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        return text, target, sid, meta

    def _qwen_target_ids(self, target):
        ids = self.qwen_tokenizer(str(target), add_special_tokens=False)['input_ids']
        if not getattr(self.args, 'open_ended', False):
            eos_id = self.qwen_tokenizer.eos_token_id
            if eos_id is None:
                eos_id = self.qwen_tokenizer.pad_token_id
            if eos_id is not None:
                ids = ids + [eos_id]
        return ids

    def _processor_inputs_from_records(self, records, turn_sample_ids_by_record, audio_sample_limits_by_record=None):
        prompt_texts = []
        flat_audio_refs = []
        flat_visual_refs = []
        audio_counts_by_record = []
        messages_by_record = []
        if audio_sample_limits_by_record is None:
            audio_sample_limits_by_record = [None] * len(records)
        for record, turn_sample_ids, audio_sample_limit in zip(
            records,
            turn_sample_ids_by_record,
            audio_sample_limits_by_record,
        ):
            content_items, audio_refs, visual_refs = self._build_qwen_content_items(
                prompt_text=record['text'],
                sample_id=record['sample_id'],
                turn_sample_ids=turn_sample_ids,
                audio_sample_limit=audio_sample_limit,
            )
            sample_id_text = str(record.get('sample_id') or '')
            if (
                sample_id_text.startswith(('Personality_full_', 'MELDPersonality_'))
                and self.audio_enable
                and self.audio_input_mode != 'text_only'
                and self.personality_media_scope in ('target', 'full')
                and len(audio_refs) == 0
                and not self._utterance_feature_row_enabled(sample_id_text)
                and not getattr(self, 'stage1_speaker_audio_only', False)
            ):
                raise RuntimeError(
                    'Conversation-level personality sample requested direct audio '
                    f'with personality_media_scope={self.personality_media_scope}, '
                    f'but no audio was attached: {sample_id_text}'
                )
            messages = [
                {'role': 'user', 'content': content_items},
            ]
            messages_by_record.append(messages)
            prompt_text = self.qwen_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(prompt_text)
            flat_audio_refs.extend(audio_refs)
            flat_visual_refs.extend(visual_refs)
            audio_counts_by_record.append(len(audio_refs))
        return (
            prompt_texts,
            flat_audio_refs,
            flat_visual_refs,
            audio_counts_by_record,
            messages_by_record,
        )

    def _expand_qwen_audio_tokens(self, prompt_texts, audio_token_lengths):
        audio_token = getattr(self.qwen_processor, 'audio_token', None)
        if audio_token is None:
            audio_token = getattr(self.qwen_tokenizer, 'audio_token', '<|AUDIO|>')
        audio_placeholder = '<|audio_placeholder|>'

        expanded = []
        token_idx = 0
        for text in prompt_texts:
            sample = text
            while audio_token in sample:
                if token_idx >= len(audio_token_lengths):
                    raise ValueError("Qwen audio placeholders exceed cached audio feature count.")
                sample = sample.replace(audio_token, audio_placeholder * int(audio_token_lengths[token_idx]), 1)
                token_idx += 1
            sample = sample.replace(audio_placeholder, audio_token)
            expanded.append(sample)
        if token_idx != len(audio_token_lengths):
            raise ValueError("Cached audio feature count exceeds Qwen audio placeholders.")
        return expanded

    def _expand_qwen_compressed_audio_tokens(
        self,
        prompt_texts,
        audio_token_lengths,
    ):
        placeholder = '<|compressed_audio_feature_placeholder|>'
        expanded = []
        token_idx = 0
        for text in prompt_texts:
            sample = text
            while AUDIO_UTTERANCE_TOKEN in sample:
                if token_idx >= len(audio_token_lengths):
                    raise ValueError(
                        "Compressed audio placeholders exceed cached feature count."
                    )
                sample = sample.replace(
                    AUDIO_UTTERANCE_TOKEN,
                    placeholder * int(audio_token_lengths[token_idx]),
                    1,
                )
                token_idx += 1
            sample = sample.replace(placeholder, AUDIO_UTTERANCE_TOKEN)
            expanded.append(sample)
        if token_idx != len(audio_token_lengths):
            raise ValueError(
                "Cached compressed audio feature count exceeds placeholders."
            )
        return expanded

    def _call_qwen_processor_with_precomputed_mel(self, prompt_texts, flat_audio_refs):
        audio_inputs, audio_token_lengths = self._build_cached_qwen_audio_inputs(flat_audio_refs)
        expanded_prompt_texts = self._expand_qwen_audio_tokens(prompt_texts, audio_token_lengths)
        text_inputs = self.qwen_tokenizer(
            expanded_prompt_texts,
            return_tensors='pt',
            padding=True,
        )
        return {**text_inputs, **audio_inputs}

    def _build_cached_qwen_visual_inputs(self, flat_visual_refs):
        features = []
        token_lengths = []
        for sample_id in flat_visual_refs:
            array = self.qwen_visual_features.get(str(sample_id))
            if array is None:
                raise KeyError(f"Missing precomputed visual features for {sample_id}")
            features.append(torch.as_tensor(array, dtype=torch.float32))
            token_lengths.append(int(array.shape[0]))
        if not features:
            return {}, token_lengths
        return {'precomputed_visual_features': torch.cat(features, dim=0)}, token_lengths

    @staticmethod
    def _mean_pool_context_array(array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            return array
        if array.ndim != 2 or array.shape[0] <= 0:
            raise ValueError(f"Invalid context feature shape: {array.shape}")
        return array.mean(axis=0)

    def _lookup_context_feature(self, store, sample_id, modality):
        if store is None:
            raise RuntimeError(f"{modality} context feature store is not initialized.")
        array = store.get(str(sample_id))
        if array is None:
            raise KeyError(f"Missing {modality} context features for {sample_id}")
        return self._mean_pool_context_array(array)

    def _build_stage1_context_prefix_inputs(self, turn_sample_ids_by_record):
        if not self.context_prefix_enable:
            return {}
        batch_size = len(turn_sample_ids_by_record)
        max_turns = max([len(turn_ids) for turn_ids in turn_sample_ids_by_record] + [1])
        feature_dim = (
            self.qwen_text_feature_dim
            or self.qwen_audio_feature_dim
            or self.qwen_visual_feature_dim
            or int(getattr(self.args, 'stage1_context_feature_dim', 3584))
        )
        if feature_dim <= 0:
            feature_dim = int(getattr(self.args, 'stage1_context_feature_dim', 3584))
        context_turn_mask = torch.zeros((batch_size, max_turns), dtype=torch.bool)
        outputs = {
            'context_turn_mask': context_turn_mask,
        }
        text_tensor = (
            torch.zeros((batch_size, max_turns, feature_dim), dtype=torch.float32)
            if self.text_context_enable
            else None
        )
        audio_tensor = (
            torch.zeros((batch_size, max_turns, feature_dim), dtype=torch.float32)
            if self.audio_context_enable
            else None
        )
        visual_tensor = (
            torch.zeros((batch_size, max_turns, feature_dim), dtype=torch.float32)
            if self.video_context_enable
            else None
        )

        for batch_idx, turn_ids in enumerate(turn_sample_ids_by_record):
            for turn_idx, sample_id in enumerate(turn_ids[:max_turns]):
                context_turn_mask[batch_idx, turn_idx] = True
                if text_tensor is not None:
                    feat = self._lookup_context_feature(
                        self.qwen_text_features,
                        sample_id,
                        'text',
                    )
                    if int(feat.shape[-1]) != feature_dim:
                        raise ValueError(
                            f"Text context dim mismatch for {sample_id}: "
                            f"{feat.shape[-1]} != {feature_dim}"
                        )
                    text_tensor[batch_idx, turn_idx] = torch.from_numpy(feat)
                if audio_tensor is not None:
                    feat = self._lookup_context_feature(
                        self.qwen_audio_features,
                        sample_id,
                        'audio',
                    )
                    if int(feat.shape[-1]) != feature_dim:
                        raise ValueError(
                            f"Audio context dim mismatch for {sample_id}: "
                            f"{feat.shape[-1]} != {feature_dim}"
                        )
                    audio_tensor[batch_idx, turn_idx] = torch.from_numpy(feat)
                if visual_tensor is not None:
                    feat = self._lookup_context_feature(
                        self.qwen_visual_features,
                        sample_id,
                        'visual',
                    )
                    if int(feat.shape[-1]) != feature_dim:
                        raise ValueError(
                            f"Visual context dim mismatch for {sample_id}: "
                            f"{feat.shape[-1]} != {feature_dim}"
                        )
                    visual_tensor[batch_idx, turn_idx] = torch.from_numpy(feat)

        if text_tensor is not None:
            outputs['context_text_features'] = text_tensor
        if audio_tensor is not None:
            outputs['context_audio_features'] = audio_tensor
        if visual_tensor is not None:
            outputs['context_visual_features'] = visual_tensor
        return outputs

    def _expand_qwen_visual_tokens(self, prompt_texts, token_lengths):
        placeholder = '<|visual_feature_placeholder|>'
        expanded = []
        token_idx = 0
        for text in prompt_texts:
            sample = text
            while VISUAL_UTTERANCE_TOKEN in sample:
                if token_idx >= len(token_lengths):
                    raise ValueError("Visual placeholders exceed cached visual feature count.")
                sample = sample.replace(
                    VISUAL_UTTERANCE_TOKEN,
                    placeholder * int(token_lengths[token_idx]),
                    1,
                )
                token_idx += 1
            sample = sample.replace(placeholder, VISUAL_UTTERANCE_TOKEN)
            expanded.append(sample)
        if token_idx != len(token_lengths):
            raise ValueError("Cached visual feature count exceeds visual placeholders.")
        return expanded

    def _call_qwen_processor(
        self,
        prompt_texts,
        flat_audio_refs,
        flat_visual_refs=None,
        messages_by_record=None,
    ):
        flat_visual_refs = flat_visual_refs or []
        if self.video_direct_enable and self.video_embedding_source == 'precomputed_tokens':
            visual_inputs, visual_token_lengths = self._build_cached_qwen_visual_inputs(
                flat_visual_refs
            )
            expanded_prompt_texts = self._expand_qwen_visual_tokens(
                prompt_texts,
                visual_token_lengths,
            )
            if self.audio_direct_enable and self.audio_input_mode != 'text_only':
                if self.audio_embedding_source == 'precomputed_pkl':
                    audio_inputs, audio_token_lengths = self._build_cached_qwen_audio_inputs(
                        flat_audio_refs
                    )
                    expanded_prompt_texts = self._expand_qwen_audio_tokens(
                        expanded_prompt_texts,
                        audio_token_lengths,
                    )
                elif self.audio_embedding_source == 'precomputed_tokens':
                    audio_inputs, audio_token_lengths = (
                        self._build_cached_qwen_audio_token_inputs(flat_audio_refs)
                    )
                    expanded_prompt_texts = (
                        self._expand_qwen_compressed_audio_tokens(
                            expanded_prompt_texts,
                            audio_token_lengths,
                        )
                    )
                else:
                    raise NotImplementedError(
                        "precomputed visual tokens require precomputed_pkl or "
                        "precomputed_tokens audio for joint audio-video training."
                    )
            else:
                audio_inputs = {}
            text_inputs = self.qwen_tokenizer(
                expanded_prompt_texts,
                return_tensors='pt',
                padding=True,
            )
            return {**text_inputs, **audio_inputs, **visual_inputs}

        if self.video_direct_enable:
            if self.audio_direct_enable and self.audio_embedding_source == 'precomputed_pkl' and self.audio_input_mode != 'text_only':
                raise NotImplementedError(
                    "video_enable with precomputed_pkl audio is not implemented yet. "
                    "Use video-only/text-video now, or switch audio_embedding_source to qwen_runtime_extract."
                )
            try:
                from qwen_omni_utils import process_mm_info
            except ImportError as exc:
                raise ImportError("video_enable requires qwen_omni_utils in the training environment.") from exc
            audios, images, videos = process_mm_info(
                messages_by_record,
                use_audio_in_video=self.video_use_audio_in_video,
            )
            kwargs = {
                'text': prompt_texts,
                'return_tensors': 'pt',
                'padding': True,
                'images': images,
                'videos': videos,
                'use_audio_in_video': self.video_use_audio_in_video,
            }
            if audios is not None:
                kwargs['audio'] = audios
                kwargs['sampling_rate'] = self.qwen_sampling_rate
            return self.qwen_processor(**kwargs)

        if not self.audio_direct_enable or self.audio_input_mode == 'text_only':
            return self.qwen_tokenizer(
                prompt_texts,
                return_tensors='pt',
                padding=True,
            )

        if self.audio_embedding_source == 'precomputed_pkl':
            return self._call_qwen_processor_with_precomputed_mel(prompt_texts, flat_audio_refs)

        if self.audio_embedding_source == 'precomputed_tokens':
            audio_inputs, audio_token_lengths = (
                self._build_cached_qwen_audio_token_inputs(flat_audio_refs)
            )
            expanded_prompt_texts = self._expand_qwen_compressed_audio_tokens(
                prompt_texts,
                audio_token_lengths,
            )
            text_inputs = self.qwen_tokenizer(
                expanded_prompt_texts,
                return_tensors='pt',
                padding=True,
            )
            return {**text_inputs, **audio_inputs}

        flat_audio_arrays = [
            self._load_qwen_audio_array(
                ref.get('sample_id', None),
                max_samples=ref.get('audio_sample_limit', None),
            )
            for ref in flat_audio_refs
        ]
        kwargs = {
            'text': prompt_texts,
            'return_tensors': 'pt',
            'padding': True,
        }
        if len(flat_audio_arrays) > 0:
            kwargs['audio'] = flat_audio_arrays
            kwargs['sampling_rate'] = self.qwen_sampling_rate
        return self.qwen_processor(**kwargs)

    def _initial_qwen_audio_sample_limit(self, record, turn_sample_ids):
        source_ids = self._qwen_record_audio_source_ids(record, turn_sample_ids)
        if len(source_ids) == 0:
            return self.qwen_sampling_rate // 10
        return max(self._qwen_audio_num_samples(sid) for sid in source_ids)

    def _reduced_qwen_audio_sample_limit(self, current_limit, prompt_len, target_len, audio_count):
        max_total_len = int(self.args.max_length)
        total_len = int(prompt_len) + int(target_len)
        excess_tokens = max(1, total_len - max_total_len)
        hop_length = int(getattr(self.qwen_processor.feature_extractor, 'hop_length', 160))
        audio_count = max(1, int(audio_count))

        # Qwen2.5-Omni maps roughly four feature frames to one audio token.
        # Use a small safety margin so one retry usually lands under max_length.
        sample_reduction = int(math.ceil(excess_tokens * hop_length * 4 * 1.25 / audio_count))
        new_limit = int(current_limit) - sample_reduction

        budget_ratio = max(0.05, min(0.95, (max_total_len - int(target_len)) / max(float(prompt_len), 1.0)))
        new_limit = min(new_limit, int(current_limit * budget_ratio * 0.98))

        min_limit = max(1, self.qwen_sampling_rate // 2)
        return max(min_limit, new_limit)

    def _pad_qwen_sequences(self, input_ids_list, labels_list=None, type_token_ids_list=None):
        pad_id = self.qwen_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Qwen tokenizer pad_token_id is None.")

        max_len = max(len(ids) for ids in input_ids_list) if input_ids_list else 1
        padding_side = getattr(self.qwen_tokenizer, 'padding_side', 'left')
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = [] if labels_list is not None else None
        batch_type_token_ids = [] if type_token_ids_list is not None else None

        for idx, ids in enumerate(input_ids_list):
            pad_len = max_len - len(ids)
            if padding_side == 'left':
                padded_ids = [pad_id] * pad_len + ids
                attention = [0] * pad_len + [1] * len(ids)
                if labels_list is not None:
                    padded_labels = [-100] * pad_len + labels_list[idx]
                if type_token_ids_list is not None:
                    padded_type_ids = [0] * pad_len + type_token_ids_list[idx]
            else:
                padded_ids = ids + [pad_id] * pad_len
                attention = [1] * len(ids) + [0] * pad_len
                if labels_list is not None:
                    padded_labels = labels_list[idx] + [-100] * pad_len
                if type_token_ids_list is not None:
                    padded_type_ids = type_token_ids_list[idx] + [0] * pad_len

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(attention)
            if labels_list is not None:
                batch_labels.append(padded_labels)
            if type_token_ids_list is not None:
                batch_type_token_ids.append(padded_type_ids)

        result = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
        }
        if batch_labels is not None:
            result['labels'] = torch.tensor(batch_labels, dtype=torch.long)
        if batch_type_token_ids is not None:
            result['type_token_ids'] = torch.tensor(batch_type_token_ids, dtype=torch.long)
        return result

    def _build_qwen_native_batch(self, raw_batch):
        records = []
        include_targets = self.mode != 'dev'
        for item in raw_batch:
            text, target, sid, meta = self._extract_text_target_sid(item)
            records.append({'text': text, 'target': target, 'sample_id': sid, 'meta': meta})

        turn_sample_ids_by_record = [
            self._effective_qwen_turn_sample_ids(record) for record in records
        ]
        audio_sample_limits_by_record = [None] * len(records)
        target_ids_by_record = [
            self._qwen_target_ids(record['target']) if include_targets else []
            for record in records
        ]

        max_attempts = max([len(ids) for ids in turn_sample_ids_by_record] + [1]) + 24
        processor_out = None
        for _ in range(max_attempts):
            (
                prompt_texts,
                flat_audio_refs,
                flat_visual_refs,
                audio_counts_by_record,
                messages_by_record,
            ) = self._processor_inputs_from_records(
                records,
                turn_sample_ids_by_record,
                audio_sample_limits_by_record,
            )
            processor_out = self._call_qwen_processor(
                prompt_texts,
                flat_audio_refs,
                flat_visual_refs=flat_visual_refs,
                messages_by_record=messages_by_record,
            )
            self._validate_qwen_audio_token_alignment(
                processor_out,
                records=records,
                context="prompt",
            )
            self._validate_qwen_visual_token_alignment(
                processor_out,
                records=records,
                context="prompt",
            )
            prompt_lengths = processor_out['attention_mask'].sum(dim=1).tolist()
            too_long = []
            for idx, prompt_len in enumerate(prompt_lengths):
                feature_budget = 0
                if self._utterance_feature_row_enabled(records[idx]['sample_id']):
                    feature_budget = len(turn_sample_ids_by_record[idx]) * (
                        int(self.utterance_feature_text) + int(self.utterance_feature_audio)
                    )
                total_len = int(prompt_len) + len(target_ids_by_record[idx]) + feature_budget
                if total_len > int(self.args.max_length):
                    too_long.append(idx)
            if len(too_long) == 0:
                break

            changed = False
            for idx in too_long:
                if str(records[idx]['sample_id']).startswith('MELDPersonality_') and self.personality_media_scope == 'full':
                    raise ValueError(
                        f"Full-dialogue MELD personality input has {prompt_lengths[idx]} tokens, "
                        f"exceeding max_length={self.args.max_length}; refusing to drop/reorder audio. "
                        "Increase max_length explicitly."
                    )
                if len(turn_sample_ids_by_record[idx]) > 1:
                    turn_sample_ids_by_record[idx] = turn_sample_ids_by_record[idx][1:]
                    changed = True
                    continue

                if (
                    audio_counts_by_record[idx] > 0
                    and self.audio_embedding_source != 'precomputed_tokens'
                ):
                    current_limit = audio_sample_limits_by_record[idx]
                    if current_limit is None:
                        current_limit = self._initial_qwen_audio_sample_limit(
                            records[idx],
                            turn_sample_ids_by_record[idx],
                        )
                    new_limit = self._reduced_qwen_audio_sample_limit(
                        current_limit=current_limit,
                        prompt_len=prompt_lengths[idx],
                        target_len=len(target_ids_by_record[idx]),
                        audio_count=audio_counts_by_record[idx],
                    )
                    if new_limit < current_limit:
                        audio_sample_limits_by_record[idx] = new_limit
                        changed = True
            if not changed:
                detail_parts = []
                for idx in too_long[:3]:
                    detail_parts.append(
                        "sample_id={sid}, prompt_len={prompt_len}, target_len={target_len}, "
                        "audio_count={audio_count}, turn_count={turn_count}, text_chars={text_chars}".format(
                            sid=records[idx].get('sample_id', None),
                            prompt_len=int(prompt_lengths[idx]),
                            target_len=len(target_ids_by_record[idx]),
                            audio_count=int(audio_counts_by_record[idx]),
                            turn_count=len(turn_sample_ids_by_record[idx]),
                            text_chars=len(str(records[idx].get('text', ''))),
                        )
                    )
                details = '; '.join(detail_parts)
                raise ValueError(
                    "Qwen multimodal prompt exceeds max_length even with only "
                    "one utterance-level media item. Increase --max_length or "
                    "shorten the prompt. "
                    f"Details: {details}"
                )
        else:
            raise ValueError(
                "Qwen audio prompt still exceeds max_length after dropping history and truncating audio. "
                "Increase --max_length or shorten the prompt/audio."
            )

        input_ids_list = []
        labels_list = [] if include_targets else None
        type_token_ids_list = [] if include_targets else None
        for idx in range(len(records)):
            row_ids = processor_out['input_ids'][idx]
            row_mask = processor_out['attention_mask'][idx].bool()
            prompt_ids = row_ids[row_mask].tolist()
            if include_targets:
                target_ids = target_ids_by_record[idx]
                full_ids = prompt_ids + target_ids
                input_ids_list.append(full_ids)
                labels_list.append([-100] * len(prompt_ids) + target_ids)
                type_token_ids_list.append([0] * len(prompt_ids) + [1] * len(target_ids))
            else:
                input_ids_list.append(prompt_ids)

        padded = self._pad_qwen_sequences(
            input_ids_list=input_ids_list,
            labels_list=labels_list,
            type_token_ids_list=type_token_ids_list,
        )
        processor_out['input_ids'] = padded['input_ids']
        processor_out['attention_mask'] = padded['attention_mask']
        processor_out['direct_audio_utterance_count'] = torch.tensor(
            audio_counts_by_record, dtype=torch.long
        )
        if include_targets:
            processor_out['labels'] = padded['labels']
            processor_out['type_token_ids'] = padded['type_token_ids']
            _add_contrastive_fields(
                processor_out,
                targets=[record['target'] for record in records],
                metadata=[record.get('meta', {}) for record in records],
                args=self.args,
            )
            _add_personality_regression_fields(
                processor_out,
                targets=[record['target'] for record in records],
                args=self.args,
            )
        _add_personality_pair_fields(
            processor_out,
            metadata=[record.get('meta', {}) for record in records],
            tokenizer=self.qwen_tokenizer,
            args=self.args,
        )
        _add_personality_prefix_fields(
            processor_out,
            metadata=[record.get('meta', {}) for record in records],
            args=self.args,
        )
        _add_mood_dynamics_fields(
            processor_out,
            targets=[record.get('target') for record in records],
            metadata=[record.get('meta', {}) for record in records],
            args=self.args,
        )
        _add_speaker_mood_dynamics_fields(
            processor_out,
            targets=[record.get('target') for record in records],
            metadata=[record.get('meta', {}) for record in records],
            args=self.args,
        )
        processor_out.update(
            self._build_stage1_context_prefix_inputs(turn_sample_ids_by_record)
        )
        processor_out.update(
            self._build_utterance_feature_inputs(records, turn_sample_ids_by_record)
        )

        if 'input_features' in processor_out:
            processor_out['input_features'] = processor_out['input_features'].to(torch.float32)
        if 'feature_attention_mask' in processor_out:
            processor_out['feature_attention_mask'] = processor_out['feature_attention_mask'].to(torch.long)
            processor_out['audio_feature_lengths'] = processor_out['feature_attention_mask'].sum(dim=1).to(torch.long)
        for key in ('pixel_values', 'pixel_values_videos'):
            if key in processor_out:
                processor_out[key] = processor_out[key].to(torch.float32)

        return processor_out

    def __call__(self, batch):
        raw_batch = batch
        if (
            self.context_prefix_enable
            or self.utterance_feature_enable
            or
            self.video_enable
            or (
                self.audio_enable
                and self.audio_embedding_source in (
                    'precomputed_pkl',
                    'precomputed_tokens',
                    'qwen_runtime_extract',
                )
                and self.audio_input_mode != 'text_only'
            )
        ):
            return self._build_qwen_native_batch(raw_batch)

        if self.mode == "dev":
            metadata = [_metadata_from_item(d, bool(self.args.emotion_prediction)) for d in batch]
            batch = [d[0] for d in batch]
            if (
                (self.audio_enable and getattr(self.args, 'audio_input_mode', 'text_audio') == 'audio_only')
                or not _as_bool_flag(getattr(self.args, 'text_enable', True), default=True)
            ):
                batch = [mask_prompt_content_for_audio_only(x) for x in batch]
            elif not self.audio_enable or self.audio_input_mode == 'text_only':
                batch = [strip_audio_markers_for_text_only(x) for x in batch]
            inputs = self.tokenizer(batch, max_length=self.args.max_length, truncation=True, padding=True, return_tensors='pt')
            inputs = _add_personality_prefix_fields(inputs, metadata, self.args)
            inputs = _add_mood_dynamics_fields(
                inputs,
                targets=[None] * len(metadata),
                metadata=metadata,
                args=self.args,
            )
            return _add_speaker_mood_dynamics_fields(
                inputs,
                targets=[None] * len(metadata),
                metadata=metadata,
                args=self.args,
            )
        else:
            inputs = preprocess_data_batch(batch, self.tokenizer, self.args)

        return inputs


def preprocess_data_batch(data, tokenizer, args):
    
    inputs = [d[0] for d in data]
    inputs_pred = None
    metadata = [_metadata_from_item(d, bool(args.emotion_prediction)) for d in data]

    if (
        (_as_bool_flag(getattr(args, 'audio_enable', False), default=False) and getattr(args, 'audio_input_mode', 'text_audio') == 'audio_only')
        or not _as_bool_flag(getattr(args, 'text_enable', True), default=True)
    ):
        inputs = [mask_prompt_content_for_audio_only(x) for x in inputs]
    elif (
        not _as_bool_flag(getattr(args, 'audio_enable', False), default=False)
        or getattr(args, 'audio_input_mode', 'text_audio') == 'text_only'
    ):
        inputs = [strip_audio_markers_for_text_only(x) for x in inputs]

    if args.emotion_prediction:
        inputs_pred = [d[1] for d in data]
        if (
            (_as_bool_flag(getattr(args, 'audio_enable', False), default=False) and getattr(args, 'audio_input_mode', 'text_audio') == 'audio_only')
            or not _as_bool_flag(getattr(args, 'text_enable', True), default=True)
        ):
            inputs_pred = [mask_prompt_content_for_audio_only(x) for x in inputs_pred]
        elif (
            not _as_bool_flag(getattr(args, 'audio_enable', False), default=False)
            or getattr(args, 'audio_input_mode', 'text_audio') == 'text_only'
        ):
            inputs_pred = [strip_audio_markers_for_text_only(x) for x in inputs_pred]
        targets = [d[2] for d in data]
    else:
        targets = [d[1] for d in data]
    target_texts = list(targets)

    if args.model_type == "decoder":
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        if pad_id is None:
            raise ValueError("Tokenizer pad_token_id/eos_token_id are both None.")

        if args.mode == "pretrain":
            inputs = tokenizer(
                inputs,
                max_length=args.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            labels = inputs['input_ids'].clone().contiguous()
            labels[labels[:, :] == tokenizer.pad_token_id] = -100
            type_token_ids = inputs['attention_mask'].long()
            inputs['labels'] = labels
            inputs["type_token_ids"] = type_token_ids
            return inputs
            
        # decoder-only model
        inputs = tokenizer(
            inputs,
            max_length=args.max_length - 1,
            truncation=True
        )

        targets = tokenizer(
            targets,
            add_special_tokens=False,
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        concat_input = [input_ids[i] + target_ids[i] for i in range(len(input_ids))]
        concat_input = [c_[: args.max_length] for c_ in concat_input]
        if not args.open_ended:
            concat_input = [c_ids + [eos_id] for c_ids in concat_input]

        type_token_ids = [[0] * len(input_ids[i]) + [1] * (len(concat_input[i]) - len(input_ids[i])) for i in range(len(input_ids))]
        attention_mask = [[1] * len(concat_input[i]) for i in range(len(input_ids))]
        
        max_batch_length = 0
        for i in range(len(input_ids)):
            max_batch_length = max(max_batch_length, len(type_token_ids[i]))
        


        if  args.emotion_prediction:
            inputs_pred = tokenizer(
                inputs_pred,
                max_length=args.max_length - 1,
                truncation=True
            )
            # 其中，max_length 参数指定了输入文本的最大长度，但是由于 Transformers 模型需要在输入文本的末尾添加一个特殊的结束符 [SEP]，所以实际上输入文本的最大长度应该是 max_length-1。
            input_pred_ids = inputs_pred['input_ids'] #  从tokenizer结果中取出inputs_ids
            concate_pred_input = [input_pred_ids[i] + target_ids[i] for i in range(len(input_pred_ids))] # 逐个样本拼接target_ids
            concate_pred_input = [c_[: args.max_length] for c_ in concate_pred_input] # 按照最大长度进行截断
            if not args.open_ended:
                concate_pred_input = [c_ids + [eos_id] for c_ids in concate_pred_input]

            pred_type_token_ids = [[0] * len(input_pred_ids[i]) + [1] * (len(concate_pred_input[i]) - len(input_pred_ids[i])) for i in range(len(input_pred_ids))]
            pred_attention_mask = [[1] * len(concate_pred_input[i]) for i in range(len(input_pred_ids))]

            # max_batch_length = 0
            for i in range(len(input_pred_ids)):
                max_batch_length = max(max_batch_length, len(pred_type_token_ids[i]))

        type_token_ids = [[0] * (max_batch_length - len(ids)) + ids for ids in type_token_ids]
        attention_mask = [[0] * (max_batch_length - len(ids)) + ids for ids in attention_mask]
        concat_input = [[pad_id] * (max_batch_length - len(ids)) + ids for ids in concat_input]
        type_token_ids = torch.Tensor(type_token_ids).long()
        attention_mask = torch.Tensor(attention_mask).long()
        concat_input = torch.Tensor(concat_input).long()
        labels = concat_input.clone().contiguous()
        labels[type_token_ids[:, :] == 0] = -100

        if args.emotion_prediction:
            pred_type_token_ids = [[0] * (max_batch_length - len(ids)) + ids for ids in pred_type_token_ids]
            pred_attention_mask = [[0] * (max_batch_length - len(ids)) + ids for ids in pred_attention_mask]
            pred_concat_input = [[pad_id] * (max_batch_length -len(ids)) + ids for ids in concate_pred_input]
            pred_type_token_ids = torch.Tensor(pred_type_token_ids).long()
            pred_attention_mask = torch.Tensor(pred_attention_mask).long()
            pred_concat_input = torch.Tensor(pred_concat_input).long()
            pred_labels = pred_concat_input.clone().contiguous()
            pred_labels[pred_type_token_ids[:, :] == 0] = -100

            concat_input = torch.concat([concat_input, pred_concat_input], dim=0)
            attention_mask = torch.concat([attention_mask, pred_attention_mask], dim=0)
            type_token_ids = torch.concat([type_token_ids, pred_type_token_ids], dim=0)
            labels = torch.concat([labels, pred_labels], dim=0)                    


        if "chatglm" in args.model_name_or_path and not "chatglm2" in args.model_name_or_path:
            attention_mask = attention_mask.bool()
        result = {
            "input_ids": concat_input,
            "attention_mask": attention_mask,
            "type_token_ids": type_token_ids,
            "labels": labels,
            "tradoff": args.beta
        }
        result = _add_contrastive_fields(result, target_texts, metadata, args)
        result = _add_personality_regression_fields(result, target_texts, args)
        result = _add_personality_pair_fields(result, metadata, tokenizer, args)
        result = _add_personality_prefix_fields(result, metadata, args)
        result = _add_mood_dynamics_fields(result, target_texts, metadata, args)
        return _add_speaker_mood_dynamics_fields(result, target_texts, metadata, args)
    else:
        ## encoder-decoder model
        inputs = tokenizer(
            inputs,
            max_length=args.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        targets = tokenizer(
            targets,
            max_length=args.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids[:, :] == tokenizer.pad_token_id] = 0
        type_token_ids = torch.ones_like(target_ids)
        type_token_ids[target_ids[:, :] == tokenizer.pad_token_id] = 0
        labels = target_ids.clone().contiguous()
        labels[target_ids[:, :] == tokenizer.pad_token_id] = -100
        result = {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_mask),
            "type_token_ids": torch.LongTensor(type_token_ids)
        }
        result = _add_contrastive_fields(result, target_texts, metadata, args)
        result = _add_personality_regression_fields(result, target_texts, args)
        result = _add_personality_pair_fields(result, metadata, tokenizer, args)
        result = _add_personality_prefix_fields(result, metadata, args)
        result = _add_mood_dynamics_fields(result, target_texts, metadata, args)
        return _add_speaker_mood_dynamics_fields(result, target_texts, metadata, args)

# class SHanglinModel(Pretrained):
#     def __init__(self):
#         self.model = LlamaModel
#         self.lm_head = nn.Linear

@dataclass
class ModelArgs:
    model_type: str = "decoder"
    model_name_or_path: str = "YOUR_MODEL_PATH"
    checkpoint_dir: str = None
    output_dir: str = "YOUR_OUTPUT_DIR_PATH"
    data_dir: str = "DATASET_PATH"
    deepspeed_config = "./deepspeed_config.json"
    do_train: bool = True
    do_eval: bool = False
    num_train_epochs = 10
    warmup_ratio: float = 0.1
    warmup_steps: int = None
    save_steps: int = 500
    save_checkpoint_epochs: str = "all"
    save_best_checkpoint: bool = True
    disable_deepspeed_checkpoint_save: bool = False
    save_lightweight_adapter_epochs: str = "none"
    weight_decay: float = 0.0
    max_seq_length: int = 96
    max_length: int = 32
    num_beams: int = 1
    do_sample: bool = False
    top_k: int = None
    top_p: float = None
    learning_rate: float = 3e-5
    preprocess_inputs: bool = True
    clip_norm: float = 1.0
    open_ended: bool = False
    batch_size: int = 32
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora: bool = True
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj,query_key_value'
    seed: int = 42
    offload_optimizer: bool = False
    deepspeed_config: str = None
    zero_shot: bool = False
    mode: str = "sft"
    gradient_checkpointing: bool = False
    personality_media_scope: str = "legacy"

    def save(self, output_dir):
        def _json_default(obj):
            if isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.detach().cpu().tolist()
            return str(obj)

        os.makedirs(output_dir, exist_ok=True)
        payload = dict(self.__dict__)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            f.write(json.dumps(payload, indent=5, default=_json_default))

    def update(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))
