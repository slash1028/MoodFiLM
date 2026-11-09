from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertTokenizerFast,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    LlamaConfig,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList
)
import pdb
import torch.nn.functional as F
import torch
import torch.nn as nn
import deepspeed
import torch.distributed as dist
from dataclasses import dataclass, asdict
import pandas as pd
import json
import ast
import gc
import hashlib
import logging
import math
import os
import pickle
import random
import re
import resource
import time
import warnings
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
import numpy as np
from data_utils.data_utils import *
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim import AdamW, Adam
from typing import List, Dict
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import argparse
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
# from colossalai.nn.optimizer import HybridAdam
# from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
# from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

# packages for ERC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from personality_pair_utils import (
    build_personality_pair_report,
    compute_personality_pair_loss,
)
from utterance_feature_prefix import UtteranceFeaturePrefix, prepend_masked_prefix
from speaker_mood_recurrence import (
    DialogueOrderedBatchSampler,
    OrderedDatasetView,
    detached_emotion_entries,
    detached_soft_emotion_entries,
    detached_state_entries,
    lookup_previous_emotions,
    lookup_previous_soft_emotions,
    lookup_previous_moods,
    mask_ordered_sample_weights,
    ordered_real_fraction,
    select_recurrent_hidden,
    validate_dialogue_parallel_batch,
)


# evaluation function for ERC
def get_labels_attr(dataset):
    label_list_set = {
        'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld':['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        'EmoryNLP': ['Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'],
        'dialydailog': ['happy', 'neutral', 'angry', 'sad', 'fear', 'surprise','disgust'],
    }
    label_str_set = {
        'iemocap':"'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'",
        'meld':"'neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'",
        'EmoryNLP': "'Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'",
        'dialydailog': "'happy', 'neutral', 'angry', 'sad', 'fear', 'surprise','disgust'",
    }

    emotional_label_dict = {text_label:num_label for num_label, text_label in enumerate(label_list_set[dataset])}
    emotional_label_str = label_str_set[dataset]
    return emotional_label_dict, emotional_label_str


def report_score(dataset, golds, preds, mode='test'):
    if dataset == 'iemocap':
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        digits = 6
    elif dataset == 'meld':
        target_names = ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry']
        digits = 7
    elif dataset == 'EmoryNLP':
        target_names = ['Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared']
        digits = 7


    res = {}
    res['Acc_SA'] = accuracy_score(golds, preds)
    res['F1_SA'] = f1_score(golds, preds, average='weighted')
    res['macro_F1_SA'] = f1_score(golds, preds, average='macro')
    res['mode'] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)

    res_matrix = metrics.classification_report(golds, preds, target_names=target_names, digits=digits)
    res_dict = metrics.classification_report(
        golds,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    # res['matrix'].append(["ACC"])
    # for i in range(len(target_names)):
    #     res['matrix'][-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(golds[golds == i], preds[golds == i])))

    return res, res_matrix, _round_metric_tree(res_dict)


def _round_metric_tree(value):
    if isinstance(value, dict):
        return {str(k): _round_metric_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_metric_tree(v) for v in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _write_eval_outputs(preds_for_eval_path, preds_for_eval, task_report=None):
    with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
        if task_report is not None:
            f.write(json.dumps(task_report, indent=2, ensure_ascii=False))
            f.write('\n\n')
        # Stream records instead of materializing one giant JSON string in rank 0.
        f.write('[\n')
        for idx, record in enumerate(preds_for_eval):
            if idx > 0:
                f.write(',\n')
            # Preserve the historical pretty-printed layout while keeping the
            # low-memory streaming writer used for large distributed evals.
            record_text = json.dumps(record, indent=5, ensure_ascii=False)
            f.write('     ' + record_text.replace('\n', '\n     '))
        f.write('\n]\n')


def _attach_run_modality_config(task_report, args):
    if task_report is None:
        return task_report
    task_report["run_modality_config"] = {
        "data_dir": getattr(args, "data_dir", None),
        "text_enable": bool(getattr(args, "text_enable", True)),
        "audio_enable": bool(getattr(args, "audio_enable", False)),
        "audio_embedding_source": getattr(args, "audio_embedding_source", None),
        "audio_feature_pkl": getattr(args, "audio_feature_pkl", None),
        "audio_input_mode": getattr(args, "audio_input_mode", None),
        "audio_embedding_usage": getattr(args, "audio_embedding_usage", None),
        "audio_adapter_scale_init": getattr(
            args, "audio_adapter_scale_init", None
        ),
        "audio_adapter_output_scale_init": getattr(
            args, "audio_adapter_output_scale_init", None
        ),
        "audio_adapter_mode": getattr(args, "audio_adapter_mode", "residual"),
        "video_enable": bool(getattr(args, "video_enable", False)),
        "video_embedding_source": getattr(args, "video_embedding_source", None),
        "video_embedding_usage": getattr(args, "video_embedding_usage", None),
        "video_feature_pkl": getattr(args, "video_feature_pkl", None),
        "video_manifest_jsonl": getattr(args, "video_manifest_jsonl", None),
        "video_adapter_output_scale_init": getattr(
            args, "video_adapter_output_scale_init", None
        ),
        "stage1_multimodal_fusion": bool(
            getattr(args, "enable_stage1_multimodal_fusion", False)
        ),
        "stage1_fusion_scale_init": getattr(args, "stage1_fusion_scale_init", None),
        "stage1_fusion_input_mode": getattr(args, "stage1_fusion_input_mode", None),
        "stage1_fusion_query_mode": getattr(args, "stage1_fusion_query_mode", None),
        "stage1_fusion_aux_gate_init": getattr(
            args, "stage1_fusion_aux_gate_init", None
        ),
        "stage1_context_prefix": bool(
            getattr(args, "enable_stage1_context_prefix", False)
        ),
        "stage1_context_prefix_mode": getattr(args, "stage1_context_prefix_mode", None),
        "stage1_context_prefix_len": getattr(args, "stage1_context_prefix_len", None),
        "stage1_context_scale_init": getattr(args, "stage1_context_scale_init", None),
        "stage1_context_audio_gate_init": getattr(args, "stage1_context_audio_gate_init", None),
        "stage1_context_visual_gate_init": getattr(args, "stage1_context_visual_gate_init", None),
        "stage1_context_consistency_weight": getattr(args, "stage1_context_consistency_weight", None),
        "text_feature_pkl": getattr(args, "text_feature_pkl", None),
        "personality_media_scope": getattr(args, "personality_media_scope", None),
        "personality_pair_mode": getattr(args, "personality_pair_mode", "none"),
        "personality_pair_loss_weight": getattr(
            args, "personality_pair_loss_weight", None
        ),
        "personality_pair_threshold": getattr(
            args, "personality_pair_threshold", None
        ),
        "personality_pair_margin": getattr(
            args, "personality_pair_margin", None
        ),
    }
    return task_report


def _process_rss_gib():
    try:
        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss_kib <= 0:
            return None
        return float(rss_kib) / (1024.0 * 1024.0)
    except Exception:
        return None


def _log_memory_checkpoint(tag):
    rss_gib = _process_rss_gib()
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if rss_gib is None:
        print(f"[MEM][rank{rank}] {tag}")
    else:
        print(f"[MEM][rank{rank}] {tag}: rss_max_gib={rss_gib:.2f}")


def _add_inference_reasoning_fields(record, args, model, tokenizer, device, prompt_text, pred_label, global_idx):
    if not _is_true_flag(getattr(args, "generate_inference_reasoning", "False")):
        return

    record["pred_reasoning_text"] = ""
    max_samples = int(getattr(args, "inference_reasoning_max_samples", 64) or 0)
    if max_samples > 0 and int(global_idx) >= max_samples:
        record["reasoning_generation_status"] = "skipped_by_limit"
        return

    reasoning_prompt = (
        f"{prompt_text}\n\n"
        f"The predicted emotion label is: {pred_label}.\n"
        "Write 2 concise sentences explaining why this label may fit the target utterance. "
        "Use only the visible dialogue, personality cues, and any visible experience-bank tendencies. "
        "Do not change the predicted label."
    )
    try:
        encoded = tokenizer(
            reasoning_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(getattr(args, "max_length", 2048)),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                num_beams=1,
                do_sample=False,
                max_new_tokens=int(getattr(args, "inference_reasoning_max_new_tokens", 120)),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = outputs[0].detach().cpu()
        input_ids = encoded["input_ids"][0].detach().cpu()
        gen_ids = _strip_prompt_from_generated_ids(input_ids, output_ids)
        reasoning_text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        record["pred_reasoning_text"] = reasoning_text.strip()
        record["reasoning_generation_status"] = "generated"
    except Exception as exc:
        record["reasoning_generation_status"] = f"failed:{type(exc).__name__}"
    
def match_text(text, word_set_):
    if text is None:
        return []
    len_text = len(text)
    s_idx = 0
    match_res = []
    while s_idx < len_text:
        cache = []
        span_length = 1
        while span_length < 12 and s_idx + span_length <= len_text:
            span = text[s_idx: s_idx + span_length]
            if span in word_set_:
                cache.append(span)
            span_length += 1
        if len(cache) > 0:
            match_res.append(cache[-1])
            s_idx += len(cache[-1])
        else:
            s_idx += 1
    return match_res


def edit_distance(s1, s2):
    """
    Calculate the editing distance between two strings
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]

def optimize_output(output, label_set):
    """
    Calculate output and label_ Set the editing distance of each label in the set and return the label corresponding to the minimum editing distance
    """
    min_distance = float('inf')
    optimized_output = None
    for label in label_set:
        distance = edit_distance(output, label)
        if distance < min_distance:
            min_distance = distance
            optimized_output = label
    return optimized_output


def _extract_tagged_section(text, tag):
    raw = str(text)
    match = re.search(
        rf'<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*>',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).strip()


STRUCTURED_REASONING_TAGS = (
    'shift',
    'trigger',
    'style',
    'act',
    'dynamics',
    'pattern',
    'dialogue_type',
    'atmosphere',
)


def _extract_structured_reasoning_states(text):
    states = {}
    compact_states = _extract_tagged_section(text, 'states')
    if compact_states is not None:
        for part in re.split(r'[;\n]+', compact_states):
            if '=' not in part:
                continue
            key, value = part.split('=', 1)
            key = re.sub(r'[^a-zA-Z0-9_]+', '_', key.strip().lower()).strip('_')
            if key in STRUCTURED_REASONING_TAGS:
                states[key] = re.sub(r'\s+', ' ', str(value).strip())
    for tag in STRUCTURED_REASONING_TAGS:
        value = _extract_tagged_section(text, tag)
        if value is not None:
            states[tag] = re.sub(r'\s+', ' ', str(value).strip())
    return states


def _extract_personality_chain(text):
    section = _extract_tagged_section(text, 'personality')
    if section is None:
        return {}
    values = {}
    for trait in ('O', 'C', 'E', 'A', 'N'):
        match = re.search(
            rf'(?:^|[<>\s,;]){trait}\s*[:=]\s*([+-]?\d+(?:\.\d+)?|unknown)',
            section,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        raw_value = match.group(1).strip().lower()
        if raw_value == 'unknown':
            values[trait] = raw_value
        else:
            try:
                values[trait] = round(float(raw_value), 3)
            except ValueError:
                values[trait] = raw_value
    return values


def _canonical_target_text(target_text):
    answer_section = _extract_tagged_section(target_text, 'answer')
    if answer_section is not None:
        return answer_section.strip()
    label_section = _extract_tagged_section(target_text, 'label')
    if label_section is not None:
        return label_section.strip()
    return str(target_text).strip()


def _emotion_fallback_label(label_set):
    labels = [str(x).strip().lower() for x in label_set]
    if 'neutral' in labels:
        return 'neutral'
    return labels[0] if labels else ''


def _match_emotion_label_fragment(fragment, labels):
    text = str(fragment or '').strip().lower()
    if not text:
        return None
    label_union = "|".join([re.escape(lb) for lb in labels])

    first_line = ''
    for line in text.splitlines():
        cleaned = line.strip().strip("\"'`.,:;[](){} ")
        if cleaned:
            first_line = cleaned
            break

    if first_line in labels:
        return first_line
    normalized_first = re.sub(r"^[^a-z]+", "", first_line)
    for label in labels:
        if normalized_first.startswith(label + " ") or normalized_first.startswith(label + ":") or normalized_first == label:
            return label

    short_text = re.sub(r'\s+', ' ', text).strip()
    if len(short_text.split()) <= 12:
        m = re.search(rf"\b({label_union})\b", short_text)
        if m is not None:
            return m.group(1)

    m = re.search(rf"emotion\s+label[^\n]*?\bis\s*[:\-]?\s*({label_union})\b", text)
    if m is not None:
        return m.group(1)
    m = re.search(rf"(?:^|\n)[#:\-\s]*({label_union})\b", text)
    if m is not None:
        return m.group(1)
    return None


def _leading_emotion_label_fragment(text):
    raw = str(text or '')
    lowered = raw.lower()
    cut_positions = []
    for marker in ["\nhuman:", "\nassistant:", "now you are expert of sentiment and emotional analysis", "<reasoning>"]:
        pos = lowered.find(marker)
        if pos != -1:
            cut_positions.append(pos)
    if cut_positions:
        raw = raw[:min(cut_positions)]
    return raw[:160]


def _extract_emotion_label_with_status(answer_text, label_set):
    """Extract emotion labels only from explicit tags or the leading short answer span."""
    labels = [str(x).strip().lower() for x in label_set]
    raw = str(answer_text or '').strip()

    answer_section = _extract_tagged_section(raw, 'answer')
    if answer_section is not None:
        label = _match_emotion_label_fragment(answer_section, labels)
        if label in labels:
            return label, 'answer_tag', False

    label_section = _extract_tagged_section(raw, 'label')
    if label_section is not None:
        label = _match_emotion_label_fragment(label_section, labels)
        if label in labels:
            return label, 'label_tag_fallback', False

    lead_fragment = _leading_emotion_label_fragment(raw)
    label = _match_emotion_label_fragment(lead_fragment, labels)
    if label in labels:
        return label, 'leading_label', False

    return None, 'missing_emotion_label', True


def _extract_emotion_label(answer_text, label_set):
    label, _, is_confused = _extract_emotion_label_with_status(answer_text, label_set)
    return label, is_confused


def _normalize_prediction_for_target(target_text, raw_answer, emotional_label_dict, return_info=False):
    """Convert free-form generation into canonical task output for stable evaluation/storage."""
    task_type = _detect_target_task(target_text)
    answer = str(raw_answer).strip().rstrip()

    if task_type == 'emotion':
        labels = list(emotional_label_dict.keys())
        pred_label, parse_status, is_confused = _extract_emotion_label_with_status(answer, labels)
        if pred_label not in emotional_label_dict:
            pred_label = _emotion_fallback_label(labels)
            parse_status = f'{parse_status}:fallback_{pred_label}'
            is_confused = True
        info = {
            'task_type': task_type,
            'parse_status': parse_status,
            'pred_label': pred_label,
            'is_confused': is_confused,
        }
        return (pred_label, info) if return_info else pred_label

    answer_section = _extract_tagged_section(answer, 'answer')
    if answer_section is not None:
        answer = answer_section.strip().rstrip()

    if task_type == 'speaker':
        spk_idx = _parse_speaker_label(answer)
        if spk_idx is not None:
            output = f"Speaker_{spk_idx}"
            return (output, {'task_type': task_type, 'parse_status': 'speaker_label'}) if return_info else output
        return (answer, {'task_type': task_type, 'parse_status': 'raw_output'}) if return_info else answer

    if task_type == 'personality_discrete':
        parsed = _parse_personality_discrete(answer)
        if parsed is not None:
            ordered = {k: parsed[k] for k in PERSONALITY_TRAITS}
            output = json.dumps(ordered, ensure_ascii=False)
            return (output, {'task_type': task_type, 'parse_status': 'personality_discrete'}) if return_info else output
        return (answer, {'task_type': task_type, 'parse_status': 'raw_output'}) if return_info else answer

    if task_type == 'personality_continuous':
        parsed = _parse_personality_continuous(answer)
        if parsed is not None:
            ordered = [round(float(parsed[k]), 3) for k in PERSONALITY_TRAITS]
            output = json.dumps(ordered, ensure_ascii=False)
            return (output, {'task_type': task_type, 'parse_status': 'personality_continuous'}) if return_info else output
        return (answer, {'task_type': task_type, 'parse_status': 'raw_output'}) if return_info else answer

    pred_label = answer
    if emotional_label_dict and pred_label in emotional_label_dict:
        pred_label = emotional_label_dict[pred_label]
    return (pred_label, {'task_type': task_type, 'parse_status': 'raw_output'}) if return_info else pred_label


PERSONALITY_TRAITS = ['O', 'C', 'E', 'A', 'N']
PERSONALITY_LEVELS = {'low', 'medium', 'high'}
PERSONALITY_CONT_MIN = 1.0
PERSONALITY_CONT_MAX = 7.0


def _is_valid_personality_cont_values(values):
    if values is None or len(values) < 5:
        return False
    for v in values[:5]:
        if v < PERSONALITY_CONT_MIN or v > PERSONALITY_CONT_MAX:
            return False
    return True


def _detect_target_task(target_text):
    target = _canonical_target_text(target_text)
    if target == "PAIR_REGRESSION":
        return "personality_pair"
    if target.startswith('Speaker_'):
        return 'speaker'
    if re.match(r'^\s*TraitRank_[OCEAN]\s*:\s*Speaker_[01]\s*$', target, flags=re.IGNORECASE):
        return 'personality_aux'
    if re.match(
        r'^\s*TraitCompare_[OCEAN]\s*:\s*(?:Speaker_[01]_higher|Similar)\s*$',
        target,
        flags=re.IGNORECASE,
    ):
        return 'personality_aux'
    if re.match(
        r'^\s*(?:TraitScore_)?[OCean]\s*[:=]\s*[+-]?\d+(?:\.\d+)?\s*$',
        target,
        flags=re.IGNORECASE,
    ):
        return 'personality_aux'
    if all(
        re.search(
            rf'(?:^|[<>\s,;]){trait}\s*[:=]\s*[+-]?\d+(?:\.\d+)?',
            target,
            flags=re.IGNORECASE,
        )
        for trait in PERSONALITY_TRAITS
    ):
        return 'personality_continuous'
    try:
        parsed = json.loads(target)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        scores = parsed.get("scores")
        if isinstance(scores, list) and len(scores) == 5:
            if all(isinstance(v, (int, float)) for v in scores):
                return 'personality_continuous'
    if isinstance(parsed, list) and len(parsed) == 5:
        if all(isinstance(v, (int, float)) for v in parsed):
            return 'personality_continuous'
    if isinstance(parsed, dict) and all(k in parsed for k in PERSONALITY_TRAITS):
        values = [str(parsed[k]).strip().lower() for k in PERSONALITY_TRAITS]
        if all(v in PERSONALITY_LEVELS for v in values):
            return 'personality_discrete'
    return 'emotion'


def _decode_target_texts_from_labels(labels, tokenizer):
    target_texts = []
    labels_cpu = labels.detach().cpu()
    for row in labels_cpu:
        target_ids = row[row != -100].tolist()
        if len(target_ids) == 0:
            target_texts.append('')
            continue
        target_texts.append(
            tokenizer.decode(
                target_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
        )
    return target_texts


def _task_weight_by_target_text(target_text, speaker_loss_weight, personality_loss_weight):
    task = _detect_target_task(target_text)
    if task == 'speaker':
        return float(speaker_loss_weight)
    if task in ('personality_continuous', 'personality_discrete', 'personality_aux'):
        return float(personality_loss_weight)
    return 1.0


def compute_task_weighted_loss(
    outputs,
    labels,
    tokenizer,
    speaker_loss_weight,
    personality_loss_weight,
    task_loss_normalization="batch",
    sample_padding_mask=None,
):
    # Recompute token-level CE to support per-sample task weights.
    # In multimodal prefix mode, logits sequence can be one step longer than raw text labels.
    logits = outputs.logits
    if logits.size(1) != labels.size(1):
        if logits.size(1) > labels.size(1):
            prefix_len = logits.size(1) - labels.size(1)
            ignore_prefix = labels.new_full((labels.size(0), prefix_len), -100)
            labels = torch.cat([ignore_prefix, labels], dim=1)
        elif labels.size(1) > logits.size(1):
            labels = labels[:, :logits.size(1)]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='none',
    ).view(shift_labels.size(0), shift_labels.size(1))

    valid_mask = shift_labels.ne(-100)
    valid_token_count = valid_mask.sum(dim=1)
    valid_count = valid_token_count.clamp(min=1).float()
    sample_loss = (token_loss * valid_mask).sum(dim=1) / valid_count

    target_texts = _decode_target_texts_from_labels(labels, tokenizer)
    sample_weights = [
        _task_weight_by_target_text(text, speaker_loss_weight, personality_loss_weight)
        for text in target_texts
    ]
    weights = sample_loss.new_tensor(sample_weights)
    weights = weights * valid_token_count.gt(0).to(weights.dtype)
    weights = mask_ordered_sample_weights(weights, sample_padding_mask)
    if task_loss_normalization == "batch":
        weighted_loss = (sample_loss * weights).sum() / weights.sum().clamp(min=1e-6)
    elif task_loss_normalization == "configured_max":
        reference_weight = max(
            abs(float(speaker_loss_weight)),
            abs(float(personality_loss_weight)),
            1.0,
        )
        present_rows = valid_token_count.gt(0).to(sample_loss.dtype).sum().clamp(min=1.0)
        weighted_loss = (
            (sample_loss * weights).sum() / present_rows / reference_weight
        )
    else:
        raise ValueError(
            "task_loss_normalization must be 'batch' or 'configured_max', "
            f"got {task_loss_normalization!r}."
        )
    return weighted_loss


def _emotion_target_indices_from_labels(labels, tokenizer, label_names):
    label_names = [str(label).strip().lower() for label in label_names]
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    target_texts = _decode_target_texts_from_labels(labels, tokenizer)
    targets = []
    for text in target_texts:
        if _detect_target_task(text) != "emotion":
            targets.append(-1)
            continue
        canonical = _canonical_target_text(text)
        label = _match_emotion_label_fragment(canonical, label_names)
        if label not in label_to_idx:
            label = _match_emotion_label_fragment(text, label_names)
        targets.append(label_to_idx.get(label, -1))
    return labels.new_tensor(targets, dtype=torch.long)


def _align_labels_to_logits(labels, logits):
    if logits.size(1) == labels.size(1):
        return labels
    if logits.size(1) > labels.size(1):
        prefix_len = logits.size(1) - labels.size(1)
        ignore_prefix = labels.new_full((labels.size(0), prefix_len), -100)
        return torch.cat([ignore_prefix, labels], dim=1)
    return labels[:, :logits.size(1)]


def _emotion_label_logits_at_first_target(logits, labels, label_token_ids):
    if logits is None or not label_token_ids:
        return None
    labels = _align_labels_to_logits(labels, logits)
    token_ids = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    valid_token_ids = token_ids.ge(0) & token_ids.lt(logits.size(-1))
    if not bool(valid_token_ids.any()):
        return None
    valid_positions = labels.ne(-100)
    has_target = valid_positions.any(dim=1)
    first_pos = valid_positions.float().argmax(dim=1).long()
    logit_pos = (first_pos - 1).clamp(min=0, max=logits.size(1) - 1)
    row_idx = torch.arange(logits.size(0), device=logits.device)
    gathered = logits[row_idx, logit_pos][:, token_ids[valid_token_ids]]
    if not bool(valid_token_ids.all()):
        padded = logits.new_full((logits.size(0), len(label_token_ids)), float("-inf"))
        padded[:, valid_token_ids] = gathered
        gathered = padded
    gathered = gathered.masked_fill(~has_target.view(-1, 1), float("-inf"))
    return gathered


def _resolve_speaker_mood_loss_weights(args):
    prior_alpha = getattr(args, "speaker_mood_prior_alpha", None)
    alpha = (
        float(prior_alpha)
        if prior_alpha is not None
        else float(getattr(args, "speaker_mood_prior_loss_weight", 0.0))
    )
    legacy_shift = float(getattr(args, "speaker_mood_shift_loss_weight", 0.0))
    legacy_align = float(getattr(args, "speaker_mood_alignment_loss_weight", 0.0))
    dyn_weight = getattr(args, "speaker_mood_dynamics_loss_weight", None)
    eta_value = getattr(args, "speaker_mood_alignment_eta", None)
    if dyn_weight is None:
        if legacy_shift > 0.0:
            beta = legacy_shift
            eta = legacy_align / legacy_shift if legacy_align > 0.0 else 0.0
        elif legacy_align > 0.0:
            beta = legacy_align
            eta = 1.0
        else:
            beta = 0.0
            eta = 0.0 if eta_value is None else float(eta_value)
    else:
        beta = float(dyn_weight)
        if eta_value is not None:
            eta = float(eta_value)
        elif legacy_align > 0.0:
            eta = legacy_align
        else:
            eta = 1.0
    return alpha, beta, eta


def _speaker_mood_dialogue_turn_key(sample_id):
    text = str(sample_id or "")
    if not text:
        return None, None
    match = re.match(r"(.+_dia\d+)_utt(\d+)$", text)
    if match:
        return match.group(1), int(match.group(2))
    match = re.match(r"(.+?)_[FM]?(\d+)$", text)
    if match:
        return match.group(1), int(match.group(2))
    if "_" in text:
        prefix, suffix = text.rsplit("_", 1)
        if suffix.isdigit():
            return prefix, int(suffix)
    return text, None


def _clean_optional_sample_id(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _speaker_mood_recurrence_enabled(args):
    return str(
        getattr(args, "speaker_mood_recurrence_mode", "none")
    ).strip().lower() == "detached"


def _speaker_mood_simplified_enabled(args):
    return str(
        getattr(args, "speaker_mood_architecture", "legacy")
    ).strip().lower() == "simplified"


def _speaker_mood_simplified_active(args):
    return _speaker_mood_simplified_enabled(args) and _is_true_flag(
        getattr(args, "use_speaker_mood_dynamics", "False")
    )


def _speaker_mood_prev_emotion_source(args):
    if _speaker_mood_simplified_enabled(args):
        return "soft"
    source = str(
        getattr(args, "speaker_mood_prev_emotion_source", "legacy")
    ).strip().lower()
    if source != "legacy":
        return source
    if (
        _speaker_mood_recurrence_enabled(args)
        and str(
            getattr(args, "speaker_mood_recurrent_prev_emotion", "none")
        ).strip().lower() != "metadata"
    ):
        return "none"
    return "gold"


def _speaker_mood_predicted_prev_emotion_enabled(args):
    return (
        _speaker_mood_prev_emotion_source(args) == "predicted"
        and not _speaker_mood_simplified_enabled(args)
    )


def _speaker_mood_dialogue_ordered_enabled(args):
    return _is_true_flag(
        getattr(args, "speaker_mood_dialogue_ordered", "False")
    )


def _same_speaker_previous_indices(sample_ids, speaker_ids):
    if not sample_ids or not speaker_ids:
        return None
    items = []
    for idx, (sample_id, speaker_id) in enumerate(zip(sample_ids, speaker_ids)):
        dialogue_id, turn_idx = _speaker_mood_dialogue_turn_key(sample_id)
        if dialogue_id is None or turn_idx is None:
            continue
        speaker = str(speaker_id or "")
        if not speaker:
            continue
        items.append((str(dialogue_id), int(turn_idx), speaker, idx))
    prev_indices = [-1] * len(sample_ids)
    last = {}
    for dialogue_id, turn_idx, speaker, idx in sorted(items, key=lambda item: (item[0], item[1], item[3])):
        key = (dialogue_id, speaker)
        if key in last:
            prev_indices[idx] = int(last[key])
        last[key] = idx
    return prev_indices


def _batch_previous_indices_from_sample_ids(sample_ids, previous_sample_ids):
    if not sample_ids or not previous_sample_ids:
        return None
    sample_to_index = {
        _clean_optional_sample_id(sample_id): idx
        for idx, sample_id in enumerate(sample_ids)
        if _clean_optional_sample_id(sample_id)
    }
    prev_indices = []
    for prev_sample_id in previous_sample_ids:
        prev_key = _clean_optional_sample_id(prev_sample_id)
        prev_indices.append(int(sample_to_index.get(prev_key, -1)) if prev_key else -1)
    return prev_indices


def _speaker_mood_previous_moods_from_state(
    after,
    sample_ids,
    previous_sample_ids,
    fallback_prev_indices=None,
    state_cache=None,
):
    rows = []
    valid = []
    device = after.device
    dtype = after.dtype
    state_cache = state_cache or {}
    fallback_prev_indices = fallback_prev_indices or [-1] * after.size(0)
    batch_indices = _batch_previous_indices_from_sample_ids(sample_ids, previous_sample_ids)

    for row_idx in range(after.size(0)):
        prev_sample_id = ""
        if previous_sample_ids is not None and row_idx < len(previous_sample_ids):
            prev_sample_id = _clean_optional_sample_id(previous_sample_ids[row_idx])

        prev_idx = -1
        if prev_sample_id and batch_indices is not None and row_idx < len(batch_indices):
            prev_idx = int(batch_indices[row_idx])
        if prev_idx < 0 and row_idx < len(fallback_prev_indices):
            prev_idx = int(fallback_prev_indices[row_idx])

        if 0 <= prev_idx < after.size(0):
            rows.append(after[prev_idx])
            valid.append(True)
            continue

        cached = state_cache.get(prev_sample_id) if prev_sample_id else None
        if cached is not None and torch.is_tensor(cached):
            rows.append(cached.to(device=device, dtype=dtype))
            valid.append(True)
            continue

        rows.append(after.new_zeros(after.size(-1)))
        valid.append(False)

    prev_moods = torch.stack(rows, dim=0) if rows else after.new_zeros(after.size())
    prev_valid = torch.tensor(valid, device=device, dtype=torch.bool)
    return prev_moods, prev_valid


def _speaker_mood_sync_state_cache_enabled(args):
    if _speaker_mood_dialogue_ordered_enabled(args):
        return False
    return _is_true_flag(getattr(args, "speaker_mood_sync_state_cache", "True"))


def _update_speaker_mood_state_cache(state_cache, model_inputs, args=None):
    if state_cache is None:
        return
    mood_after = model_inputs.get("speaker_mood_after")
    sample_ids = model_inputs.get("speaker_mood_sample_id")
    if mood_after is None or not sample_ids:
        return
    valid_mask = None
    if args is not None and _speaker_mood_recurrence_enabled(args):
        valid_mask = model_inputs.get("speaker_mood_target_mask_computed")
        if valid_mask is None:
            valid_mask = model_inputs.get("speaker_mood_target_mask")
    local_entries = detached_state_entries(
        mood_after=mood_after,
        sample_ids=sample_ids,
        valid_mask=valid_mask,
        padding_mask=model_inputs.get("speaker_mood_order_padding"),
    )

    if (
        args is not None
        and _speaker_mood_sync_state_cache_enabled(args)
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        payload = [(key, value.tolist()) for key, value in local_entries]
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)
        for part in gathered:
            for key, values in part or []:
                clean_key = _clean_optional_sample_id(key)
                if clean_key:
                    state_cache[clean_key] = torch.tensor(values, dtype=torch.float32)
        return

    for key, value in local_entries:
        state_cache[key] = value


def _update_speaker_mood_emotion_cache(
    emotion_cache,
    model_inputs,
    predicted_indices,
    num_emotions,
):
    if emotion_cache is None or predicted_indices is None:
        return
    valid_mask = model_inputs.get("speaker_mood_target_mask_computed")
    if valid_mask is None:
        valid_mask = model_inputs.get("speaker_mood_target_mask")
    entries = detached_emotion_entries(
        predicted_indices=predicted_indices,
        sample_ids=model_inputs.get("speaker_mood_sample_id"),
        num_emotions=num_emotions,
        valid_mask=valid_mask,
        padding_mask=model_inputs.get("speaker_mood_order_padding"),
    )
    for key, value in entries:
        emotion_cache[key] = value


def _update_speaker_mood_soft_emotion_cache(
    emotion_cache,
    model_inputs,
    final_label_logits,
    num_emotions,
):
    if emotion_cache is None:
        raise RuntimeError("Simplified MoodFiLM requires a local soft-state cache.")
    valid_mask = model_inputs.get("speaker_mood_target_mask_computed")
    if valid_mask is None:
        valid_mask = model_inputs.get("speaker_mood_target_mask")
    speaker_ids = (
        model_inputs.get("speaker_mood_target_speaker_id")
        or model_inputs.get("speaker_mood_target_speaker")
    )
    entries = detached_soft_emotion_entries(
        final_label_logits=final_label_logits,
        sample_ids=model_inputs.get("speaker_mood_sample_id"),
        speaker_ids=speaker_ids,
        num_emotions=num_emotions,
        valid_mask=valid_mask,
        padding_mask=model_inputs.get("speaker_mood_order_padding"),
    )
    for key, value in entries:
        emotion_cache[key] = value


def _generated_emotion_indices(
    tokenizer,
    generated_outputs,
    generation_input_ids,
    label_names,
):
    label_names = [str(label).strip().lower() for label in label_names]
    label_to_index = {label: index for index, label in enumerate(label_names)}
    fallback = _emotion_fallback_label(label_names)
    indices = []
    for output_ids, input_ids in zip(generated_outputs, generation_input_ids):
        gen_ids = _strip_prompt_from_generated_ids(input_ids, output_ids)
        answer = tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        label, _ = _extract_emotion_label(answer, label_names)
        if label not in label_to_index:
            label = fallback
        indices.append(label_to_index.get(label, -1))
    return torch.tensor(indices, dtype=torch.long)


def _compute_speaker_mood_auxiliary_losses(
    model_inputs,
    labels,
    tokenizer,
    label_names,
    args,
    base_label_logits=None,
    speaker_mood_state_cache=None,
):
    if not _is_true_flag(getattr(args, "use_speaker_mood_dynamics", "False")):
        return None

    use_prior_loss = _is_true_flag(getattr(args, "use_speaker_mood_prior_loss", "False"))
    use_dyn_loss = (
        _is_true_flag(getattr(args, "use_speaker_mood_dynamics_consistency_loss", "False"))
        or _is_true_flag(getattr(args, "use_speaker_mood_shift_loss", "False"))
        or _is_true_flag(getattr(args, "use_speaker_mood_alignment_loss", "False"))
    )
    alpha, beta, _ = _resolve_speaker_mood_loss_weights(args)
    if (
        (not use_prior_loss or alpha <= 0.0)
        and (not use_dyn_loss or beta <= 0.0)
    ):
        return None

    transition_prior = model_inputs.get("speaker_mood_transition_prior")
    mood_after = model_inputs.get("speaker_mood_after")
    interpersonal_input = model_inputs.get("speaker_mood_interpersonal_alignment_input")
    if transition_prior is None and (mood_after is None or interpersonal_input is None):
        return None

    device = None
    for tensor in (transition_prior, mood_after, interpersonal_input):
        if tensor is not None:
            device = tensor.device
            break
    if device is None:
        return None

    target_indices = _emotion_target_indices_from_labels(labels, tokenizer, label_names).to(device=device)
    valid = target_indices.ge(0)
    target_mask = model_inputs.get("speaker_mood_target_mask_computed")
    if target_mask is None:
        target_mask = model_inputs.get("speaker_mood_target_mask")
    if target_mask is not None:
        valid = valid & target_mask.to(device=device).bool()
    ordered_padding = model_inputs.get("speaker_mood_order_padding")
    if ordered_padding is not None:
        valid = valid & ~ordered_padding.to(device=device).bool().view(-1)

    losses = {}
    total = None

    if use_prior_loss and alpha > 0.0 and transition_prior is not None:
        prior = transition_prior.to(device=device).float()
        if prior.size(0) == target_indices.size(0):
            prior_valid = valid & target_indices.lt(prior.size(-1))
            if bool(prior_valid.any()):
                per_sample = F.cross_entropy(
                    prior,
                    target_indices.clamp(min=0, max=prior.size(-1) - 1),
                    reduction="none",
                )
                if base_label_logits is not None and base_label_logits.size(0) == prior.size(0):
                    base_scores = base_label_logits.to(device=device).float()
                    if base_scores.size(-1) == prior.size(-1):
                        with torch.no_grad():
                            base_probs = torch.softmax(base_scores, dim=-1)
                            gold_prob = base_probs.gather(
                                dim=-1,
                                index=target_indices.clamp(min=0, max=prior.size(-1) - 1).view(-1, 1),
                            ).squeeze(-1)
                            uncertainty_weight = (1.0 - gold_prob).detach()
                    else:
                        uncertainty_weight = per_sample.new_ones(per_sample.size(0))
                else:
                    uncertainty_weight = per_sample.new_ones(per_sample.size(0))
                prior_loss = (per_sample * uncertainty_weight)[prior_valid].mean()
                losses["prior"] = prior_loss
                losses["prior_uncertainty_weight"] = uncertainty_weight[prior_valid].mean()
                total = alpha * prior_loss if total is None else total + alpha * prior_loss

    if (
        beta > 0.0
        and use_dyn_loss
        and mood_after is not None
        and interpersonal_input is not None
    ):
        after = mood_after.to(device=device).float()
        align_input = interpersonal_input.to(device=device).float()
        if after.size() == align_input.size() and after.size(0) == target_indices.size(0):
            prev_emotion = model_inputs.get("speaker_mood_prev_emotion")
            prev_mask = model_inputs.get("speaker_mood_prev_emotion_mask")
            prev_emotion_valid = valid & torch.zeros_like(valid)
            changed_emotion = valid & torch.zeros_like(valid)
            if prev_emotion is not None and prev_mask is not None:
                prev = prev_emotion.to(device=device).float()
                if prev.size(0) == target_indices.size(0):
                    prev_index = prev.argmax(dim=-1)
                    prev_emotion_valid = (
                        valid
                        & prev_mask.to(device=device).bool()
                        & prev.sum(dim=-1).gt(0.0)
                        & target_indices.lt(prev.size(-1))
                    )
                    changed_emotion = prev_emotion_valid & target_indices.ne(prev_index)

            if _speaker_mood_recurrence_enabled(args):
                before = model_inputs.get("speaker_mood_before")
                recurrent_valid = model_inputs.get("speaker_mood_recurrent_prev_valid")
                if before is None or recurrent_valid is None:
                    prev_mood_valid = torch.zeros_like(valid)
                    delta_m = after.new_zeros(after.size())
                else:
                    before = before.to(device=device).float()
                    prev_mood_valid = recurrent_valid.to(device=device).bool()
                    delta_m = after - before
            else:
                sample_ids = model_inputs.get("speaker_mood_sample_id") or []
                previous_sample_ids = (
                    model_inputs.get("speaker_mood_previous_sample_id")
                    or model_inputs.get("speaker_mood_previous_same_speaker_sample_id")
                    or []
                )
                fallback_prev_indices = None
                if not previous_sample_ids or not any(
                    _clean_optional_sample_id(item) for item in previous_sample_ids
                ):
                    fallback_prev_indices = _same_speaker_previous_indices(
                        sample_ids,
                        model_inputs.get("speaker_mood_target_speaker_id")
                        or model_inputs.get("speaker_mood_target_speaker"),
                    )

                prev_moods, prev_mood_valid = _speaker_mood_previous_moods_from_state(
                    after=after,
                    sample_ids=sample_ids,
                    previous_sample_ids=previous_sample_ids,
                    fallback_prev_indices=fallback_prev_indices,
                    state_cache=speaker_mood_state_cache,
                )
                delta_m = after - prev_moods
            dyn_valid = valid & changed_emotion & prev_mood_valid

            context_mask = model_inputs.get("speaker_mood_context_mask_computed")
            if context_mask is None:
                context_mask = model_inputs.get("speaker_mood_context_mask")
            if context_mask is not None and torch.is_tensor(context_mask):
                dyn_valid = dyn_valid & context_mask.to(device=device).bool().any(dim=-1)

            dyn_count = dyn_valid.to(dtype=after.dtype).sum()
            losses["dyn_count"] = dyn_count.detach()
            losses["dyn_ratio"] = (dyn_count / max(1, int(after.size(0)))).detach()

            if bool(dyn_valid.any()):
                cosine = F.cosine_similarity(align_input, delta_m, dim=-1, eps=1e-6)
                dyn_per_sample = (1.0 - cosine).clamp(min=0.0, max=2.0)
                dyn_loss = dyn_per_sample[dyn_valid].mean()

                rho = getattr(args, "speaker_mood_dynamics_magnitude_weight", None)
                if rho is None:
                    rho = 0.0
                    if (
                        _is_true_flag(getattr(args, "use_speaker_mood_shift_loss", "False"))
                        and not _is_true_flag(getattr(args, "use_speaker_mood_dynamics_consistency_loss", "False"))
                        and not _is_true_flag(getattr(args, "use_speaker_mood_alignment_loss", "False"))
                    ):
                        rho = float(getattr(args, "speaker_mood_shift_change_weight", 1.0))
                rho = float(rho)
                if rho > 0.0:
                    margin = getattr(args, "speaker_mood_dynamics_margin", None)
                    if margin is None:
                        margin = float(getattr(args, "speaker_mood_shift_margin", 0.25))
                    margin = float(margin)
                    shift_norm = torch.linalg.vector_norm(delta_m, dim=-1)
                    magnitude_loss = F.relu(shift_norm.new_tensor(margin) - shift_norm)[dyn_valid].mean()
                    losses["dyn_magnitude"] = magnitude_loss
                    dyn_loss = dyn_loss + rho * magnitude_loss

                losses["dyn"] = dyn_loss
                total = beta * dyn_loss if total is None else total + beta * dyn_loss
            else:
                dyn_loss = after.sum() * 0.0
                losses["dyn"] = dyn_loss
                total = beta * dyn_loss if total is None else total + beta * dyn_loss

    if total is None:
        return None
    losses["total"] = total
    return losses


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, hidden_size, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(proj_dim)),
        )

    def forward(self, hidden):
        return self.net(hidden)


class PersonalityRegressionHead(nn.Module):
    def __init__(self, hidden_size, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 5),
        )

    def forward(self, pooled_hidden):
        raw = self.net(pooled_hidden)
        return 4.0 + 3.0 * torch.tanh(raw)


class PrecomputedVisualAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        bottleneck_dim=256,
        dropout=0.1,
        scale_init=0.1,
        output_scale_init=0.05,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_size))
        self.down = nn.Linear(int(hidden_size), int(bottleneck_dim))
        self.up = nn.Linear(int(bottleneck_dim), int(hidden_size))
        self.dropout = nn.Dropout(float(dropout))
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
        self.output_scale = nn.Parameter(
            torch.tensor(float(output_scale_init), dtype=torch.float32)
        )

    def forward(self, features, dtype=None):
        ref = next(self.parameters())
        features = features.to(device=ref.device, dtype=ref.dtype)
        delta = self.up(self.dropout(F.gelu(self.down(self.norm(features)))))
        output = self.output_scale.to(features.dtype) * (
            features + self.scale.to(features.dtype) * delta
        )
        if dtype is not None:
            output = output.to(dtype=dtype)
        return output


class PrecomputedAudioAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        bottleneck_dim=256,
        dropout=0.1,
        scale_init=0.1,
        output_scale_init=1.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_size))
        self.down = nn.Linear(int(hidden_size), int(bottleneck_dim))
        self.up = nn.Linear(int(bottleneck_dim), int(hidden_size))
        self.dropout = nn.Dropout(float(dropout))
        self.scale = nn.Parameter(
            torch.tensor(float(scale_init), dtype=torch.float32)
        )
        self.output_scale = nn.Parameter(
            torch.tensor(float(output_scale_init), dtype=torch.float32)
        )

    def forward(self, features, dtype=None):
        ref = next(self.parameters())
        features = features.to(device=ref.device, dtype=ref.dtype)
        delta = self.up(self.dropout(F.gelu(self.down(self.norm(features)))))
        output = self.output_scale.to(features.dtype) * (
            features + self.scale.to(features.dtype) * delta
        )
        if dtype is not None:
            output = output.to(dtype=dtype)
        return output


def _prob_to_logit(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class Stage1MultimodalAttentionFusion(nn.Module):
    def __init__(
        self,
        hidden_size,
        fusion_dim=256,
        num_heads=4,
        prefix_len=2,
        dropout=0.1,
        scale_init=0.1,
        query_mode="generic",
        aux_gate_init=0.05,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        fusion_dim = int(fusion_dim)
        num_heads = int(num_heads)
        prefix_len = int(prefix_len)
        query_mode = str(query_mode or "generic").strip().lower()
        if query_mode == "trait_gated":
            prefix_len = len(PERSONALITY_TRAITS)
        if fusion_dim <= 0 or num_heads <= 0 or fusion_dim % num_heads != 0:
            raise ValueError("Stage1 fusion_dim must be positive and divisible by num_heads.")
        if prefix_len <= 0:
            raise ValueError("Stage1 fusion prefix_len must be positive.")
        if query_mode not in ("generic", "trait_gated"):
            raise ValueError("Stage1 fusion query_mode must be generic or trait_gated.")

        self.hidden_size = hidden_size
        self.fusion_dim = fusion_dim
        self.prefix_len = prefix_len
        self.query_mode = query_mode
        self.text_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fusion_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fusion_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fusion_dim),
        )
        self.modality_embeddings = nn.Parameter(torch.empty(3, fusion_dim))
        self.query_tokens = nn.Parameter(torch.empty(prefix_len, fusion_dim))
        aux_gate_init = min(max(float(aux_gate_init), 1e-4), 1.0 - 1e-4)
        gate_logits = torch.full((prefix_len, 3), math.log(aux_gate_init / (1.0 - aux_gate_init)))
        gate_logits[:, 0] = math.log(0.95 / 0.05)
        self.trait_modality_gate_logits = nn.Parameter(gate_logits)
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    def forward(
        self,
        text_summary,
        audio_summary,
        visual_summary,
        modality_available,
        dtype=None,
    ):
        ref = next(self.parameters())
        text_summary = text_summary.to(device=ref.device, dtype=ref.dtype)
        audio_summary = audio_summary.to(device=ref.device, dtype=ref.dtype)
        visual_summary = visual_summary.to(device=ref.device, dtype=ref.dtype)
        modality_available = modality_available.to(device=ref.device, dtype=torch.bool)

        modality_tokens = torch.stack(
            [
                self.text_proj(text_summary),
                self.audio_proj(audio_summary),
                self.visual_proj(visual_summary),
            ],
            dim=1,
        )
        modality_tokens = modality_tokens + self.modality_embeddings.unsqueeze(0)
        queries = self.query_tokens.unsqueeze(0).expand(modality_tokens.size(0), -1, -1)
        if self.query_mode == "trait_gated":
            scale = float(self.fusion_dim) ** -0.5
            scores = torch.matmul(queries, modality_tokens.transpose(1, 2)) * scale
            gates = torch.sigmoid(self.trait_modality_gate_logits).to(
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = scores + torch.log(gates.clamp(min=1e-6)).unsqueeze(0)
            scores = scores.masked_fill(~modality_available.unsqueeze(1), -1e4)
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, modality_tokens)
        else:
            attended, _ = self.attention(
                query=queries,
                key=modality_tokens,
                value=modality_tokens,
                key_padding_mask=~modality_available,
                need_weights=False,
            )
        prefix = self.output(attended)
        prefix = prefix * self.scale.to(prefix.dtype)
        if dtype is not None:
            prefix = prefix.to(dtype=dtype)
        return prefix


class Stage1TriModalContextPrefix(nn.Module):
    def __init__(
        self,
        hidden_size,
        feature_dim=3584,
        fusion_dim=256,
        gru_hidden=128,
        num_heads=4,
        prefix_len=2,
        dropout=0.1,
        scale_init=0.05,
        audio_gate_init=0.10,
        visual_gate_init=0.001,
        mode="ta_cross",
        arch="gru_cross",
        consistency_weight=0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.feature_dim = int(feature_dim)
        self.fusion_dim = int(fusion_dim)
        self.gru_hidden = int(gru_hidden)
        self.prefix_len = int(prefix_len)
        self.mode = str(mode or "ta_cross").strip().lower()
        self.arch = str(arch or "gru_cross").strip().lower()
        self.consistency_weight = float(consistency_weight)
        if self.mode not in ("text_only", "ta_cross", "tav_cross", "visual_side", "tav_prefix_only"):
            raise ValueError("stage1_context_prefix_mode must be text_only, ta_cross, tav_cross, visual_side, or tav_prefix_only")
        if self.arch not in ("gru_cross", "pool", "gru_pool", "cross_only"):
            raise ValueError("stage1_context_arch must be gru_cross, pool, gru_pool, or cross_only")
        if self.fusion_dim <= 0 or self.gru_hidden <= 0 or self.prefix_len <= 0:
            raise ValueError("Stage1 context prefix dimensions must be positive.")
        attention_dim = self.gru_hidden * 2
        if self.arch in ("pool", "cross_only") and self.fusion_dim != attention_dim:
            raise ValueError("stage1_context_fusion_dim must equal 2 * stage1_context_gru_hidden for pool/cross_only.")
        if attention_dim % int(num_heads) != 0:
            raise ValueError("2 * stage1_context_gru_hidden must be divisible by stage1_context_heads.")
        self.attention_dim = attention_dim

        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.fusion_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.fusion_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.fusion_dim),
        )
        self.text_gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.audio_gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.visual_gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.modality_embeddings = nn.Parameter(torch.empty(3, attention_dim))
        self.query_tokens = nn.Parameter(torch.empty(self.prefix_len, attention_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
        self.audio_gate_logit = nn.Parameter(
            torch.tensor(_prob_to_logit(audio_gate_init), dtype=torch.float32)
        )
        self.visual_gate_logit = nn.Parameter(
            torch.tensor(_prob_to_logit(visual_gate_init), dtype=torch.float32)
        )
        self.last_consistency_loss = None
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    @staticmethod
    def _masked_mean(sequence, mask):
        weights = mask.to(sequence.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (sequence * weights).sum(dim=1) / denom

    def _encode(self, features, mask, proj, gru, use_gru=True):
        projected = proj(features)
        projected = projected * mask.to(projected.dtype).unsqueeze(-1)
        if use_gru:
            encoded, _ = gru(projected)
            encoded = encoded * mask.to(encoded.dtype).unsqueeze(-1)
        else:
            encoded = projected
        return encoded

    def _add_modality_embedding(self, encoded, modality_idx, gate=None):
        encoded = encoded + self.modality_embeddings[modality_idx].to(encoded.dtype)
        if gate is not None:
            encoded = encoded * gate.to(encoded.dtype)
        return encoded

    def forward(
        self,
        text_features,
        turn_mask,
        audio_features=None,
        visual_features=None,
        dtype=None,
    ):
        ref = next(self.parameters())
        device = ref.device
        param_dtype = ref.dtype
        turn_mask = turn_mask.to(device=device).bool()
        if not turn_mask.any(dim=1).all():
            turn_mask = turn_mask.clone()
            empty_rows = ~turn_mask.any(dim=1)
            turn_mask[empty_rows, 0] = True
        text_features = text_features.to(device=device, dtype=param_dtype)
        use_gru = self.arch in ("gru_cross", "gru_pool")
        use_cross_attention = self.arch in ("gru_cross", "cross_only")
        text_encoded = self._encode(
            text_features,
            turn_mask,
            self.text_proj,
            self.text_gru,
            use_gru=use_gru,
        )
        text_encoded = self._add_modality_embedding(text_encoded, 0)
        tokens = [text_encoded]
        masks = [turn_mask]
        pooled = [self._masked_mean(text_encoded, turn_mask)]

        use_audio = audio_features is not None and self.mode in (
            "ta_cross",
            "tav_cross",
            "visual_side",
            "tav_prefix_only",
        )
        if use_audio:
            audio_features = audio_features.to(device=device, dtype=param_dtype)
            audio_gate = torch.sigmoid(self.audio_gate_logit)
            audio_encoded = self._encode(
                audio_features,
                turn_mask,
                self.audio_proj,
                self.audio_gru,
                use_gru=use_gru,
            )
            audio_encoded = self._add_modality_embedding(
                audio_encoded,
                1,
                gate=audio_gate,
            )
            tokens.append(audio_encoded)
            masks.append(turn_mask)
            pooled.append(self._masked_mean(audio_encoded, turn_mask))

        use_visual = visual_features is not None and self.mode in (
            "tav_cross",
            "visual_side",
            "tav_prefix_only",
        )
        if use_visual:
            visual_features = visual_features.to(device=device, dtype=param_dtype)
            visual_gate = torch.sigmoid(self.visual_gate_logit)
            visual_encoded = self._encode(
                visual_features,
                turn_mask,
                self.visual_proj,
                self.visual_gru,
                use_gru=use_gru,
            )
            visual_encoded = self._add_modality_embedding(
                visual_encoded,
                2,
                gate=visual_gate,
            )
            tokens.append(visual_encoded)
            masks.append(turn_mask)
            pooled.append(self._masked_mean(visual_encoded, turn_mask))

        if use_cross_attention:
            context_tokens = torch.cat(tokens, dim=1)
            context_mask = torch.cat(masks, dim=1)
            queries = self.query_tokens.unsqueeze(0).expand(context_tokens.size(0), -1, -1)
            attended, _ = self.attention(
                query=queries,
                key=context_tokens,
                value=context_tokens,
                key_padding_mask=~context_mask,
                need_weights=False,
            )
        else:
            pooled_context = torch.stack(pooled, dim=1).mean(dim=1)
            attended = pooled_context.unsqueeze(1).expand(-1, self.prefix_len, -1)
            attended = attended + self.query_tokens.unsqueeze(0).to(attended.dtype)
        prefix = self.output(attended) * self.scale.to(attended.dtype)
        self.last_consistency_loss = None
        if self.consistency_weight > 0 and len(pooled) > 1:
            losses = []
            base = pooled[0]
            for other in pooled[1:]:
                losses.append(1.0 - F.cosine_similarity(base, other, dim=-1).mean())
            if losses:
                self.last_consistency_loss = torch.stack(losses).mean()
        if dtype is not None:
            prefix = prefix.to(dtype=dtype)
        return prefix


class PersonalityInteractionPrefix(nn.Module):
    def __init__(
        self,
        hidden_size,
        prefix_len=4,
        personality_hidden_dim=128,
        dropout=0.1,
        mode="interaction",
        use_gate=True,
        scale_init=0.1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_len = int(prefix_len)
        self.mode = str(mode or "interaction").strip().lower()
        self.use_gate = bool(use_gate)
        personality_hidden_dim = int(personality_hidden_dim)
        self.use_attention_interaction = self.mode == "attention_interaction"

        self.target_encoder = nn.Sequential(
            nn.Linear(5, personality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(personality_hidden_dim, personality_hidden_dim),
            nn.ReLU(),
        )
        self.interlocutor_encoder = nn.Sequential(
            nn.Linear(5, personality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(personality_hidden_dim, personality_hidden_dim),
            nn.ReLU(),
        )

        if self.mode == "target":
            pair_dim = personality_hidden_dim
        elif self.mode in ("target_interlocutor", "shuffled"):
            pair_dim = personality_hidden_dim * 2
        elif self.use_attention_interaction:
            pair_dim = personality_hidden_dim
        else:
            pair_dim = personality_hidden_dim * 4

        if self.use_attention_interaction:
            attention_heads = 2 if personality_hidden_dim % 2 == 0 else 1
            self.attention_norm = nn.LayerNorm(personality_hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=personality_hidden_dim,
                num_heads=attention_heads,
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            self.attention_norm = None
            self.attention = None

        self.interaction = nn.Sequential(
            nn.Linear(pair_dim, personality_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.prefix_proj = nn.Linear(personality_hidden_dim, self.prefix_len * self.hidden_size)
        self.gate_proj = nn.Linear(personality_hidden_dim, self.prefix_len) if self.use_gate else None
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(self, target_personality, interlocutor_personality, mask=None, dtype=None):
        ref_param = next(self.parameters())
        target_personality = target_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        interlocutor_personality = interlocutor_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        h_t = self.target_encoder(target_personality)
        h_i = self.interlocutor_encoder(interlocutor_personality)
        if self.mode == "target":
            pair = h_t
        elif self.mode in ("target_interlocutor", "shuffled"):
            pair = torch.cat([h_t, h_i], dim=-1)
        elif self.use_attention_interaction:
            tokens = torch.stack([h_t, h_i, h_t * h_i, h_t - h_i], dim=1)
            tokens = self.attention_norm(tokens)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            pair = attended.mean(dim=1)
        else:
            pair = torch.cat([h_t, h_i, h_t * h_i, h_t - h_i], dim=-1)

        hidden = self.interaction(pair)
        prefix = self.prefix_proj(hidden).view(-1, self.prefix_len, self.hidden_size)
        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(hidden)).unsqueeze(-1)
            prefix = prefix * gate
        prefix = prefix * self.scale.to(prefix.dtype)
        if mask is not None:
            prefix = prefix * mask.to(prefix.device).bool().view(-1, 1, 1).to(prefix.dtype)
        if dtype is not None:
            prefix = prefix.to(dtype=dtype)
        return prefix


class PersonalityConditionedLatentMoodPrefix(nn.Module):
    def __init__(
        self,
        hidden_size,
        prefix_len=4,
        mood_dim=128,
        mood_hidden_dim=128,
        dropout=0.1,
        mode="interaction",
        scale_init=0.05,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_len = int(prefix_len)
        self.mood_dim = int(mood_dim)
        self.mode = str(mode or "interaction").strip().lower()
        if self.mode not in ("target", "target_interlocutor", "interaction", "attention_interaction", "factorized"):
            self.mode = "interaction"
        self.use_attention_interaction = self.mode == "attention_interaction"
        self.use_factorized_prefix = self.mode == "factorized"
        mood_hidden_dim = int(mood_hidden_dim)

        self.target_encoder = nn.Sequential(
            nn.Linear(5, mood_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mood_hidden_dim, mood_hidden_dim),
            nn.ReLU(),
        )
        self.interlocutor_encoder = nn.Sequential(
            nn.Linear(5, mood_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mood_hidden_dim, mood_hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, mood_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        if self.mode == "target":
            updater_input_dim = mood_hidden_dim * 2
        elif self.mode == "target_interlocutor":
            updater_input_dim = mood_hidden_dim * 3
        elif self.use_attention_interaction:
            updater_input_dim = mood_hidden_dim
        else:
            updater_input_dim = mood_hidden_dim * 5

        if self.use_attention_interaction:
            attention_heads = 2 if mood_hidden_dim % 2 == 0 else 1
            self.attention_norm = nn.LayerNorm(mood_hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=mood_hidden_dim,
                num_heads=attention_heads,
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            self.attention_norm = None
            self.attention = None

        self.mood_candidate = nn.Sequential(
            nn.Linear(updater_input_dim, mood_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mood_hidden_dim, self.mood_dim),
            nn.Tanh(),
        )
        self.mood_gate = nn.Sequential(
            nn.Linear(updater_input_dim, mood_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mood_hidden_dim, self.mood_dim),
            nn.Sigmoid(),
        )
        self.initial_mood = nn.Linear(mood_hidden_dim, self.mood_dim)
        self.prefix_proj = nn.Linear(self.mood_dim, self.prefix_len * self.hidden_size)
        self.prefix_gate = nn.Linear(self.mood_dim, self.prefix_len)
        factor_count = 5
        base_tokens = self.prefix_len // factor_count
        extra_tokens = self.prefix_len % factor_count
        self.factor_token_counts = [
            base_tokens + (1 if idx < extra_tokens else 0)
            for idx in range(factor_count)
        ]
        self.factor_prefix_projs = nn.ModuleList(
            [
                nn.Linear(mood_hidden_dim, count * self.hidden_size)
                for count in self.factor_token_counts
                if count > 0
            ]
        )
        self.factor_gate_projs = nn.ModuleList(
            [
                nn.Linear(mood_hidden_dim, count)
                for count in self.factor_token_counts
                if count > 0
            ]
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(
        self,
        input_embeds,
        attention_mask,
        target_personality,
        interlocutor_personality,
        mask=None,
        dtype=None,
    ):
        ref_param = next(self.parameters())
        input_embeds = input_embeds.to(device=ref_param.device, dtype=ref_param.dtype)
        target_personality = target_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        interlocutor_personality = interlocutor_personality.to(device=ref_param.device, dtype=ref_param.dtype)

        if attention_mask is None:
            attn = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        else:
            attn = attention_mask.to(device=input_embeds.device).bool()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1).to(input_embeds.dtype)
        context = (input_embeds * attn.unsqueeze(-1).to(input_embeds.dtype)).sum(dim=1) / denom
        h_c = self.context_encoder(context)

        h_t = self.target_encoder(target_personality)
        h_i = self.interlocutor_encoder(interlocutor_personality)
        factor_states = [h_c, h_t, h_i, h_t * h_i, h_t - h_i]
        if self.use_factorized_prefix:
            prefix_parts = []
            proj_idx = 0
            for state, count in zip(factor_states, self.factor_token_counts):
                if count <= 0:
                    continue
                part = self.factor_prefix_projs[proj_idx](state).view(
                    -1, count, self.hidden_size
                )
                gate = torch.sigmoid(self.factor_gate_projs[proj_idx](state)).unsqueeze(-1)
                prefix_parts.append(part * gate)
                proj_idx += 1
            prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else input_embeds.new_zeros(
                input_embeds.size(0), 0, self.hidden_size
            )
            prefix = prefix * self.scale.to(prefix.dtype)
            if mask is not None:
                prefix = prefix * mask.to(prefix.device).bool().view(-1, 1, 1).to(prefix.dtype)
            if dtype is not None:
                prefix = prefix.to(dtype=dtype)
            return prefix

        if self.mode == "target":
            updater_input = torch.cat([h_c, h_t], dim=-1)
        elif self.mode == "target_interlocutor":
            updater_input = torch.cat([h_c, h_t, h_i], dim=-1)
        elif self.use_attention_interaction:
            tokens = torch.stack(factor_states, dim=1)
            tokens = self.attention_norm(tokens)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            updater_input = attended.mean(dim=1)
        else:
            updater_input = torch.cat([h_c, h_t, h_i, h_t * h_i, h_t - h_i], dim=-1)

        base_mood = torch.tanh(self.initial_mood(h_t))
        candidate = self.mood_candidate(updater_input)
        gate = self.mood_gate(updater_input)
        mood = gate * candidate + (1.0 - gate) * base_mood

        prefix = self.prefix_proj(mood).view(-1, self.prefix_len, self.hidden_size)
        prefix_gate = torch.sigmoid(self.prefix_gate(mood)).unsqueeze(-1)
        prefix = prefix * prefix_gate * self.scale.to(prefix.dtype)
        if mask is not None:
            prefix = prefix * mask.to(prefix.device).bool().view(-1, 1, 1).to(prefix.dtype)
        if dtype is not None:
            prefix = prefix.to(dtype=dtype)
        return prefix


class PersonalityMoodHybridPrefix(nn.Module):
    def __init__(
        self,
        hidden_size,
        prefix_len=8,
        hybrid_hidden_dim=192,
        mood_dim=128,
        dropout=0.1,
        mode="factorized",
        scale_init=0.05,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prefix_len = int(prefix_len)
        self.mode = str(mode or "factorized").strip().lower()
        if self.mode not in (
            "target_context",
            "interaction",
            "attention_interaction",
            "factorized",
            "factorized_attention",
        ):
            self.mode = "factorized"
        self.use_attention = self.mode in ("attention_interaction", "factorized_attention")
        self.use_factorized = self.mode in ("factorized", "factorized_attention")
        hybrid_hidden_dim = int(hybrid_hidden_dim)
        mood_dim = int(mood_dim)

        self.target_encoder = nn.Sequential(
            nn.Linear(5, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hybrid_hidden_dim, hybrid_hidden_dim),
            nn.ReLU(),
        )
        self.interlocutor_encoder = nn.Sequential(
            nn.Linear(5, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hybrid_hidden_dim, hybrid_hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        if self.use_attention:
            attention_heads = 2 if hybrid_hidden_dim % 2 == 0 else 1
            self.attention_norm = nn.LayerNorm(hybrid_hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=hybrid_hidden_dim,
                num_heads=attention_heads,
                dropout=float(dropout),
                batch_first=True,
            )
        else:
            self.attention_norm = None
            self.attention = None

        fusion_input_dim = hybrid_hidden_dim * (3 if self.mode == "target_context" else 5)
        self.fusion = nn.Sequential(
            nn.Linear(hybrid_hidden_dim if self.mode == "attention_interaction" else fusion_input_dim, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.initial_mood = nn.Linear(hybrid_hidden_dim, mood_dim)
        self.mood_candidate = nn.Sequential(
            nn.Linear(hybrid_hidden_dim if self.mode == "attention_interaction" else fusion_input_dim, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hybrid_hidden_dim, mood_dim),
            nn.Tanh(),
        )
        self.mood_gate = nn.Sequential(
            nn.Linear(hybrid_hidden_dim if self.mode == "attention_interaction" else fusion_input_dim, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hybrid_hidden_dim, mood_dim),
            nn.Sigmoid(),
        )
        self.mood_to_hidden = nn.Sequential(
            nn.Linear(mood_dim, hybrid_hidden_dim),
            nn.ReLU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(hybrid_hidden_dim * 2, hybrid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.prefix_proj = nn.Linear(hybrid_hidden_dim, self.prefix_len * self.hidden_size)
        self.prefix_gate = nn.Linear(hybrid_hidden_dim, self.prefix_len)

        factor_count = 6
        base_tokens = self.prefix_len // factor_count
        extra_tokens = self.prefix_len % factor_count
        self.factor_token_counts = [
            base_tokens + (1 if idx < extra_tokens else 0)
            for idx in range(factor_count)
        ]
        self.factor_prefix_projs = nn.ModuleList(
            [
                nn.Linear(hybrid_hidden_dim, count * self.hidden_size)
                for count in self.factor_token_counts
                if count > 0
            ]
        )
        self.factor_gate_projs = nn.ModuleList(
            [
                nn.Linear(hybrid_hidden_dim, count)
                for count in self.factor_token_counts
                if count > 0
            ]
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))

    def forward(
        self,
        input_embeds,
        attention_mask,
        target_personality,
        interlocutor_personality,
        mask=None,
        dtype=None,
    ):
        ref_param = next(self.parameters())
        input_embeds = input_embeds.to(device=ref_param.device, dtype=ref_param.dtype)
        target_personality = target_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        interlocutor_personality = interlocutor_personality.to(device=ref_param.device, dtype=ref_param.dtype)

        if attention_mask is None:
            attn = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        else:
            attn = attention_mask.to(device=input_embeds.device).bool()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1).to(input_embeds.dtype)
        context = (input_embeds * attn.unsqueeze(-1).to(input_embeds.dtype)).sum(dim=1) / denom

        h_c = self.context_encoder(context)
        h_t = self.target_encoder(target_personality)
        h_i = self.interlocutor_encoder(interlocutor_personality)
        factor_states = [h_c, h_t, h_i, h_t * h_i, h_t - h_i]
        if self.mode == "target_context":
            fusion_input = torch.cat([h_c, h_t, h_i], dim=-1)
            fusion_state = self.fusion(fusion_input)
            mood_input = fusion_input
        elif self.use_attention:
            tokens = torch.stack(factor_states, dim=1)
            attended, _ = self.attention(
                self.attention_norm(tokens),
                self.attention_norm(tokens),
                self.attention_norm(tokens),
                need_weights=False,
            )
            if self.mode == "attention_interaction":
                fusion_state = self.fusion(attended.mean(dim=1))
                mood_input = attended.mean(dim=1)
            else:
                factor_states = [attended[:, idx, :] for idx in range(attended.size(1))]
                fusion_input = torch.cat(factor_states, dim=-1)
                fusion_state = self.fusion(fusion_input)
                mood_input = fusion_input
        else:
            fusion_input = torch.cat(factor_states, dim=-1)
            fusion_state = self.fusion(fusion_input)
            mood_input = fusion_input

        base_mood = torch.tanh(self.initial_mood(h_t))
        candidate = self.mood_candidate(mood_input)
        gate = self.mood_gate(mood_input)
        mood = gate * candidate + (1.0 - gate) * base_mood
        mood_state = self.mood_to_hidden(mood)

        if self.use_factorized:
            prefix_parts = []
            proj_idx = 0
            for state, count in zip(factor_states + [mood_state], self.factor_token_counts):
                if count <= 0:
                    continue
                part = self.factor_prefix_projs[proj_idx](state).view(
                    -1, count, self.hidden_size
                )
                factor_gate = torch.sigmoid(self.factor_gate_projs[proj_idx](state)).unsqueeze(-1)
                prefix_parts.append(part * factor_gate)
                proj_idx += 1
            prefix = torch.cat(prefix_parts, dim=1) if prefix_parts else input_embeds.new_zeros(
                input_embeds.size(0), 0, self.hidden_size
            )
        else:
            joint = self.joint(torch.cat([fusion_state, mood_state], dim=-1))
            prefix = self.prefix_proj(joint).view(-1, self.prefix_len, self.hidden_size)
            prefix = prefix * torch.sigmoid(self.prefix_gate(joint)).unsqueeze(-1)

        prefix = prefix * self.scale.to(prefix.dtype)
        if mask is not None:
            prefix = prefix * mask.to(prefix.device).bool().view(-1, 1, 1).to(prefix.dtype)
        if dtype is not None:
            prefix = prefix.to(dtype=dtype)
        return prefix


class MoodDynamicsTransitionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_emotions,
        mood_dim=128,
        hidden_dim=192,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_emotions = int(num_emotions)
        self.mood_dim = int(mood_dim)
        hidden_dim = int(hidden_dim)
        self.personality_encoder = nn.Sequential(
            nn.Linear(5 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.prev_emotion_encoder = nn.Sequential(
            nn.Linear(self.num_emotions, hidden_dim),
            nn.ReLU(),
        )
        self.mood_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.mood_dim),
            nn.Tanh(),
        )
        self.transition_net = nn.Sequential(
            nn.Linear(self.mood_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_emotions),
        )

    def forward(
        self,
        input_embeds,
        attention_mask,
        target_personality,
        interlocutor_personality,
        prev_emotion=None,
        mask=None,
    ):
        ref_param = next(self.parameters())
        input_embeds = input_embeds.to(device=ref_param.device, dtype=ref_param.dtype)
        target_personality = target_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        interlocutor_personality = interlocutor_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        if prev_emotion is None:
            prev_emotion = target_personality.new_zeros(
                target_personality.size(0),
                self.num_emotions,
            )
        else:
            prev_emotion = prev_emotion.to(device=ref_param.device, dtype=ref_param.dtype)

        if attention_mask is None:
            attn = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        else:
            attn = attention_mask.to(device=input_embeds.device).bool()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1).to(input_embeds.dtype)
        context = (input_embeds * attn.unsqueeze(-1).to(input_embeds.dtype)).sum(dim=1) / denom

        personality_features = torch.cat(
            [
                target_personality,
                interlocutor_personality,
                target_personality * interlocutor_personality,
                target_personality - interlocutor_personality,
            ],
            dim=-1,
        )
        p_state = self.personality_encoder(personality_features)
        c_state = self.context_encoder(context)
        e_state = self.prev_emotion_encoder(prev_emotion)
        mood_state = self.mood_net(torch.cat([p_state, c_state, e_state], dim=-1))
        transition_prior = self.transition_net(mood_state)
        if mask is not None:
            valid = mask.to(transition_prior.device).bool().view(-1, 1).to(transition_prior.dtype)
            transition_prior = transition_prior * valid
            mood_state = mood_state * valid
        return mood_state, transition_prior


class SimplifiedSpeakerMoodDynamicsHead(nn.Module):
    """Leakage-free personality-conditioned mood with shared MoodFiLM heads."""

    def __init__(
        self,
        hidden_size,
        num_emotions,
        speaker_mood_dim=128,
        hidden_dim=192,
        relation_dim=4,
        dropout=0.1,
        use_personality=True,
        use_previous_emotion=True,
        use_interaction=True,
        use_current_evidence=True,
        interaction_attention_mode="learned",
        interaction_use_personality=True,
        interaction_use_relation=True,
        interaction_profile_permutation="none",
        interaction_attention_temperature=1.0,
        interaction_min_interlocutors=1,
        interaction_summary_scale=1.0,
        interaction_recency_bias=0.0,
        interaction_previous_bias=0.0,
        personality_temperature=1.0,
        personality_center=4.0,
        personality_residual_scale=0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_emotions = int(num_emotions)
        self.speaker_mood_dim = int(speaker_mood_dim)
        hidden_dim = int(hidden_dim)
        relation_dim = int(relation_dim)
        dropout = float(dropout)
        self.use_personality = bool(use_personality)
        self.use_previous_emotion = bool(use_previous_emotion)
        self.use_interaction = bool(use_interaction)
        self.use_current_evidence = bool(use_current_evidence)
        self.interaction_attention_mode = str(interaction_attention_mode).strip().lower()
        if self.interaction_attention_mode not in {
            "learned", "uniform", "recency", "previous", "nearest"
        }:
            raise ValueError(
                "interaction_attention_mode must be learned, uniform, recency, previous, or nearest"
            )
        self.interaction_use_personality = bool(interaction_use_personality)
        self.interaction_use_relation = bool(interaction_use_relation)
        self.interaction_profile_permutation = str(
            interaction_profile_permutation
        ).strip().lower()
        if self.interaction_profile_permutation not in {"none", "reverse"}:
            raise ValueError("interaction_profile_permutation must be none or reverse")
        self.interaction_attention_temperature = float(
            interaction_attention_temperature
        )
        if self.interaction_attention_temperature <= 0.0:
            raise ValueError("interaction_attention_temperature must be positive")
        self.interaction_min_interlocutors = int(interaction_min_interlocutors)
        if self.interaction_min_interlocutors < 0:
            raise ValueError("interaction_min_interlocutors must be non-negative")
        self.interaction_summary_scale = float(interaction_summary_scale)
        if self.interaction_summary_scale < 0.0:
            raise ValueError("interaction_summary_scale must be non-negative")
        self.interaction_recency_bias = float(interaction_recency_bias)
        self.interaction_previous_bias = float(interaction_previous_bias)
        self.personality_temperature = float(personality_temperature)
        # The collator converts 1--7 OCEAN scores to [-1, 1] before this head.
        # Accept the historical raw-Likert center (4.0) as well as a normalized
        # center so old launch configurations keep their intended semantics.
        personality_center = float(personality_center)
        if personality_center < -1.0 or personality_center > 1.0:
            personality_center = (personality_center - 4.0) / 3.0
        self.personality_center = max(-1.0, min(1.0, personality_center))
        self.personality_residual_scale = float(personality_residual_scale)

        self.unknown_personality = nn.Parameter(torch.zeros(5))
        self.personality_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.emotion_embedding = nn.Parameter(
            torch.empty(self.num_emotions, hidden_dim)
        )
        nn.init.normal_(self.emotion_embedding, mean=0.0, std=0.02)
        self.initial_emotion_state = nn.Parameter(torch.zeros(hidden_dim))
        self.anchor_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.speaker_mood_dim),
        )
        self.personality_anchor_residual = nn.Linear(
            hidden_dim, self.speaker_mood_dim, bias=False
        )

        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention_query = nn.Linear(self.speaker_mood_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_value = nn.Linear(hidden_dim, hidden_dim)

        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.speaker_mood_dim),
        )
        self.mood_norm = nn.LayerNorm(self.speaker_mood_dim)

        self.film_shared = nn.Sequential(
            nn.LayerNorm(self.speaker_mood_dim),
            nn.Linear(self.speaker_mood_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prior_head = nn.Linear(hidden_dim, self.num_emotions)
        self.gamma_head = nn.Linear(hidden_dim, self.num_emotions)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        self.interpersonal_alignment_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.speaker_mood_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        input_embeds,
        prompt_mask,
        target_personality,
        context_personalities,
        relation_features,
        context_mask,
        target_mask=None,
        target_personality_available=None,
        previous_emotion_probs=None,
        previous_emotion_valid=None,
    ):
        ref_param = next(self.parameters())
        device = ref_param.device
        dtype = ref_param.dtype
        input_embeds = input_embeds.to(device=device, dtype=dtype)
        prompt_mask = prompt_mask.to(device=device).bool()
        target_personality = target_personality.to(device=device, dtype=dtype)
        context_personalities = context_personalities.to(device=device, dtype=dtype)
        relation_features = relation_features.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device).bool()
        if target_mask is None:
            target_mask = torch.ones(
                target_personality.size(0), device=device, dtype=torch.bool
            )
        else:
            target_mask = target_mask.to(device=device).bool()
        if target_personality_available is None:
            target_personality_available = torch.ones(
                target_personality.size(0), device=device, dtype=torch.bool
            )
        else:
            target_personality_available = target_personality_available.to(
                device=device
            ).bool()

        if previous_emotion_probs is None:
            previous_emotion_probs = target_personality.new_zeros(
                target_personality.size(0), self.num_emotions
            )
        else:
            previous_emotion_probs = previous_emotion_probs.to(
                device=device, dtype=dtype
            )
        if previous_emotion_valid is None:
            previous_emotion_valid = torch.zeros(
                target_personality.size(0), device=device, dtype=torch.bool
            )
        else:
            previous_emotion_valid = previous_emotion_valid.to(device=device).bool()

        prompt_count = prompt_mask.sum(dim=1, keepdim=True)
        if bool(prompt_count.eq(0).any()):
            rows = prompt_count.eq(0).view(-1).nonzero(as_tuple=False).view(-1).tolist()
            raise RuntimeError(f"Simplified MoodFiLM found empty prompt rows: {rows[:8]}")
        pooled_prompt = (
            input_embeds * prompt_mask.unsqueeze(-1).to(dtype)
        ).sum(dim=1) / prompt_count.clamp(min=1).to(dtype)
        if self.use_current_evidence:
            current_evidence = self.context_encoder(pooled_prompt)
        else:
            current_evidence = pooled_prompt.new_zeros(
                pooled_prompt.size(0), self.emotion_embedding.size(1)
            )

        availability = relation_features[..., 3:4].clamp(0.0, 1.0)
        unknown = self.unknown_personality.to(dtype).view(1, 1, -1)
        target_personality = torch.where(
            target_personality_available.view(-1, 1),
            target_personality,
            unknown.view(1, -1),
        )
        context_personalities = (
            availability * context_personalities
            + (1.0 - availability) * unknown
        )
        if self.personality_temperature != 1.0:
            center = target_personality.new_tensor(self.personality_center)
            calibrated_target = center + self.personality_temperature * (
                target_personality - center
            )
            calibrated_context = center + self.personality_temperature * (
                context_personalities - center
            )
            target_personality = torch.where(
                target_personality_available.view(-1, 1),
                calibrated_target.clamp(-1.0, 1.0),
                unknown.view(1, -1),
            )
            context_personalities = torch.where(
                availability.bool().expand_as(context_personalities),
                calibrated_context.clamp(-1.0, 1.0),
                unknown.expand_as(context_personalities),
            )
        target_personality_state = self.personality_encoder(target_personality)
        context_personality_state = self.personality_encoder(context_personalities)
        if not self.use_personality:
            target_personality_state = torch.zeros_like(target_personality_state)
            context_personality_state = torch.zeros_like(context_personality_state)

        soft_emotion_state = previous_emotion_probs @ self.emotion_embedding
        initial_state = self.initial_emotion_state.view(1, -1).expand_as(
            soft_emotion_state
        )
        if self.use_previous_emotion:
            previous_emotion_state = torch.where(
                previous_emotion_valid.view(-1, 1),
                soft_emotion_state,
                initial_state,
            )
        else:
            previous_emotion_state = initial_state
        mood_anchor = self.anchor_mlp(
            torch.cat([target_personality_state, previous_emotion_state], dim=-1)
        )
        if self.use_personality and self.personality_residual_scale != 0.0:
            mood_anchor = mood_anchor + self.personality_residual_scale * self.personality_anchor_residual(
                target_personality_state
            )

        if self.use_interaction:
            interaction_context_state = context_personality_state
            if self.interaction_profile_permutation == "reverse":
                interaction_context_state = interaction_context_state.clone()
                for row_idx in range(interaction_context_state.size(0)):
                    valid_idx = context_mask[row_idx].nonzero(as_tuple=False).view(-1)
                    if valid_idx.numel() > 1:
                        interaction_context_state[row_idx, valid_idx] = (
                            interaction_context_state[row_idx, valid_idx.flip(0)]
                        )
            target_expanded = target_personality_state.unsqueeze(1).expand_as(
                context_personality_state
            )
            if not self.interaction_use_personality:
                target_expanded = torch.zeros_like(target_expanded)
                interaction_context_state = torch.zeros_like(interaction_context_state)
            relation_input = relation_features
            if not self.interaction_use_relation:
                relation_input = torch.zeros_like(relation_input)
            relation_state = self.relation_encoder(relation_input)
            interaction_tokens = self.interaction_mlp(
                torch.cat(
                    [
                        target_expanded,
                        interaction_context_state,
                        target_expanded * interaction_context_state,
                        target_expanded - interaction_context_state,
                        relation_state,
                    ],
                    dim=-1,
                )
            )
            query = self.attention_query(mood_anchor)
            keys = self.attention_key(interaction_tokens)
            values = self.attention_value(interaction_tokens)
            no_context = ~context_mask.any(dim=-1)
            if self.interaction_attention_mode == "learned":
                attention_scores = torch.einsum("bd,bnd->bn", query, keys)
                attention_scores = attention_scores / (
                    math.sqrt(keys.size(-1)) * self.interaction_attention_temperature
                )
                if self.interaction_recency_bias != 0.0:
                    attention_scores = attention_scores + self.interaction_recency_bias * torch.log(
                        relation_features[..., 0].float().clamp(min=1e-6)
                    ).to(attention_scores.dtype)
                if self.interaction_previous_bias != 0.0:
                    attention_scores = attention_scores + self.interaction_previous_bias * relation_features[
                        ..., 2
                    ].to(attention_scores.dtype)
                attention_scores = attention_scores.masked_fill(
                    ~context_mask, float("-inf")
                )
                if bool(no_context.any()):
                    attention_scores = attention_scores.clone()
                    attention_scores[no_context] = 0.0
                attention_weights = torch.softmax(
                    attention_scores.float(), dim=-1
                ).to(dtype)
                attention_weights = attention_weights.masked_fill(~context_mask, 0.0)
            else:
                if self.interaction_attention_mode == "uniform":
                    raw_weights = context_mask.to(dtype)
                elif self.interaction_attention_mode == "recency":
                    raw_weights = relation_features[..., 0].clamp(min=0.0) * context_mask
                elif self.interaction_attention_mode == "nearest":
                    # Interlocutors are ordered by minimum turn distance by the
                    # data builder; the first valid token is therefore nearest.
                    nearest_index = context_mask.to(torch.int64).argmax(dim=-1)
                    raw_weights = torch.zeros_like(context_mask, dtype=dtype)
                    raw_weights.scatter_(1, nearest_index.view(-1, 1), 1.0)
                    raw_weights = raw_weights * context_mask
                else:
                    raw_weights = relation_features[..., 2].clamp(min=0.0) * context_mask
                    missing_previous = raw_weights.sum(dim=-1).eq(0) & ~no_context
                    if bool(missing_previous.any()):
                        raw_weights = raw_weights.clone()
                        raw_weights[missing_previous] = (
                            relation_features[missing_previous, :, 0].clamp(min=0.0)
                            * context_mask[missing_previous]
                        )
                denominator = raw_weights.sum(dim=-1, keepdim=True)
                fallback = context_mask.to(dtype)
                raw_weights = torch.where(
                    denominator.gt(0), raw_weights, fallback
                )
                attention_weights = raw_weights / raw_weights.sum(
                    dim=-1, keepdim=True
                ).clamp(min=torch.finfo(dtype).eps)
            interaction_summary = torch.einsum(
                "bn,bnd->bd", attention_weights, values
            )
            interaction_active = context_mask.sum(dim=-1).ge(
                self.interaction_min_interlocutors
            )
            interaction_summary = interaction_summary * interaction_active.view(
                -1, 1
            ).to(dtype)
            interaction_summary = interaction_summary * self.interaction_summary_scale
            if bool(no_context.any()):
                interaction_summary = interaction_summary.masked_fill(
                    no_context.view(-1, 1), 0.0
                )
        else:
            interaction_summary = current_evidence.new_zeros(
                current_evidence.size(0), current_evidence.size(1)
            )
            attention_weights = current_evidence.new_zeros(context_mask.shape)

        mood_update = self.update_mlp(
            torch.cat([current_evidence, interaction_summary], dim=-1)
        )
        mood = self.mood_norm(mood_anchor + mood_update)
        film_state = self.film_shared(mood)
        transition_prior = self.prior_head(film_state)
        film_gamma_raw = self.gamma_head(film_state)
        alignment_input = self.interpersonal_alignment_proj(interaction_summary)

        valid = target_mask.view(-1, 1).to(dtype)
        return {
            "mood_before": mood_anchor * valid,
            "mood_after": mood * valid,
            "transition_prior": transition_prior * valid,
            "film_gamma_raw": film_gamma_raw * valid,
            "attention_weights": attention_weights * target_mask.view(-1, 1).to(dtype),
            "interpersonal_alignment_input": alignment_input * valid,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "previous_emotion_valid": previous_emotion_valid,
        }


class SpeakerSpecificMoodDynamicsHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_emotions,
        speaker_mood_dim=128,
        hidden_dim=192,
        relation_dim=4,
        dropout=0.1,
        logit_modulation="additive",
        film_condition="mood_context",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_emotions = int(num_emotions)
        self.speaker_mood_dim = int(speaker_mood_dim)
        hidden_dim = int(hidden_dim)
        relation_dim = int(relation_dim)
        self.logit_modulation = str(logit_modulation).strip().lower()
        self.film_condition = str(film_condition).strip().lower()
        self.unknown_personality = nn.Parameter(torch.zeros(5))
        self.target_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_personality_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.prev_emotion_encoder = nn.Sequential(
            nn.Linear(self.num_emotions, hidden_dim),
            nn.ReLU(),
        )
        self.prev_mood_proj = nn.Linear(hidden_dim, self.speaker_mood_dim)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.initial_mood = nn.Linear(hidden_dim, self.speaker_mood_dim)
        self.mood_query = nn.Sequential(
            nn.Linear(hidden_dim + self.speaker_mood_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        heads = 2 if hidden_dim % 2 == 0 else 1
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.gru_input = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.mood_gru = nn.GRUCell(hidden_dim, self.speaker_mood_dim)
        self.transition_net = nn.Sequential(
            nn.Linear(self.speaker_mood_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_emotions),
        )
        self.interpersonal_alignment_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.speaker_mood_dim),
            nn.Tanh(),
        )
        self.film_gamma_net = None
        if self.logit_modulation == "film":
            film_condition_dim = self.speaker_mood_dim
            if self.film_condition == "mood_context":
                film_condition_dim += hidden_dim
            self.film_gamma_net = nn.Sequential(
                nn.LayerNorm(film_condition_dim),
                nn.Linear(film_condition_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, self.num_emotions),
            )
            # Zero initialization makes FiLM an exact identity before learning.
            nn.init.zeros_(self.film_gamma_net[-1].weight)
            nn.init.zeros_(self.film_gamma_net[-1].bias)

    def forward(
        self,
        input_embeds,
        attention_mask,
        target_personality,
        context_personalities,
        relation_features,
        context_mask,
        target_mask=None,
        prev_emotion=None,
        previous_mood=None,
        previous_mood_mask=None,
    ):
        ref_param = next(self.parameters())
        input_embeds = input_embeds.to(device=ref_param.device, dtype=ref_param.dtype)
        target_personality = target_personality.to(device=ref_param.device, dtype=ref_param.dtype)
        context_personalities = context_personalities.to(device=ref_param.device, dtype=ref_param.dtype)
        relation_features = relation_features.to(device=ref_param.device, dtype=ref_param.dtype)
        context_mask = context_mask.to(device=ref_param.device).bool()
        if target_mask is None:
            target_mask = torch.ones(target_personality.size(0), dtype=torch.bool, device=ref_param.device)
        else:
            target_mask = target_mask.to(device=ref_param.device).bool()
        if prev_emotion is None:
            prev_emotion = target_personality.new_zeros(
                target_personality.size(0),
                self.num_emotions,
            )
        else:
            prev_emotion = prev_emotion.to(device=ref_param.device, dtype=ref_param.dtype)
        if previous_mood is not None:
            previous_mood = previous_mood.to(
                device=ref_param.device,
                dtype=ref_param.dtype,
            )
            if previous_mood_mask is None:
                previous_mood_mask = torch.zeros(
                    previous_mood.size(0),
                    device=ref_param.device,
                    dtype=torch.bool,
                )
            else:
                previous_mood_mask = previous_mood_mask.to(
                    device=ref_param.device,
                ).bool()

        if attention_mask is None:
            attn = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        else:
            attn = attention_mask.to(device=input_embeds.device).bool()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1).to(input_embeds.dtype)
        pooled_context = (input_embeds * attn.unsqueeze(-1).to(input_embeds.dtype)).sum(dim=1) / denom

        availability = relation_features[..., 3:4].clamp(0.0, 1.0)
        unknown = self.unknown_personality.to(context_personalities.dtype).view(1, 1, -1)
        context_personalities = availability * context_personalities + (1.0 - availability) * unknown

        h_t = self.target_encoder(target_personality)
        h_i = self.context_personality_encoder(context_personalities)
        h_t_expanded = h_t.unsqueeze(1).expand_as(h_i)
        h_rel = self.relation_encoder(relation_features)
        interaction_tokens = self.interaction_mlp(
            torch.cat(
                [
                    h_t_expanded,
                    h_i,
                    h_t_expanded * h_i,
                    h_t_expanded - h_i,
                    h_rel,
                ],
                dim=-1,
            )
        )

        h_context = self.context_encoder(pooled_context)
        h_prev_emotion = self.prev_emotion_encoder(prev_emotion)
        prev_valid = prev_emotion.sum(dim=-1, keepdim=True).gt(0.0).to(h_t.dtype)
        base_mood = self.initial_mood(h_t)
        if previous_mood is None:
            legacy_prev_mood = base_mood + self.prev_mood_proj(h_prev_emotion)
            mood_before = torch.tanh(
                prev_valid * legacy_prev_mood + (1.0 - prev_valid) * base_mood
            )
        else:
            initial_mood = torch.tanh(base_mood)
            mood_before = select_recurrent_hidden(
                initial_mood=initial_mood,
                previous_mood=previous_mood,
                previous_valid=previous_mood_mask,
            )
        query = self.mood_query(torch.cat([h_t, mood_before, h_context], dim=-1)).unsqueeze(1)
        key_padding_mask = ~context_mask
        no_context = key_padding_mask.all(dim=1)
        if bool(no_context.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[no_context, 0] = False
            interaction_tokens = interaction_tokens.clone()
            interaction_tokens[no_context, 0, :] = 0.0
        attended, attention_weights = self.attention(
            self.query_norm(query),
            self.key_norm(interaction_tokens),
            self.key_norm(interaction_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        interpersonal_input = attended.squeeze(1)
        if bool(no_context.any()):
            interpersonal_input = interpersonal_input.masked_fill(no_context.view(-1, 1), 0.0)
            attention_weights = attention_weights.masked_fill(no_context.view(-1, 1, 1), 0.0)
        attention_weights = attention_weights.squeeze(1)
        interpersonal_alignment_input = self.interpersonal_alignment_proj(interpersonal_input)

        gru_input = self.gru_input(torch.cat([h_context, interpersonal_input, h_t, h_prev_emotion], dim=-1))
        mood_after = self.mood_gru(gru_input, mood_before)
        transition_prior = self.transition_net(mood_after)
        film_gamma_raw = None
        if self.film_gamma_net is not None:
            film_condition = mood_after
            if self.film_condition == "mood_context":
                film_condition = torch.cat([mood_after, interpersonal_input], dim=-1)
            film_gamma_raw = self.film_gamma_net(film_condition)
        valid = target_mask.view(-1, 1).to(transition_prior.dtype)
        transition_prior = transition_prior * valid
        if film_gamma_raw is not None:
            film_gamma_raw = film_gamma_raw * valid
        mood_before = mood_before * valid
        mood_after = mood_after * valid
        interpersonal_alignment_input = interpersonal_alignment_input * valid
        attention_weights = attention_weights * target_mask.view(-1, 1).to(attention_weights.dtype)
        return {
            "mood_before": mood_before,
            "mood_after": mood_after,
            "transition_prior": transition_prior,
            "film_gamma_raw": film_gamma_raw,
            "attention_weights": attention_weights,
            "interpersonal_alignment_input": interpersonal_alignment_input,
            "context_mask": context_mask,
            "target_mask": target_mask,
        }


class MoodDynamicsLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        transition_prior,
        label_token_ids,
        scale=0.2,
        initial_length=None,
        film_gamma_raw=None,
        modulation="additive",
        film_norm="center",
        gamma_scale=0.5,
        beta_scale=1.0,
        use_gamma=True,
        use_beta=True,
        beta_gate="none",
        gate_temperature=1.0,
        gate_power=1.0,
        gate_min=0.0,
        eps=1e-6,
        capture_final_label_logits=False,
    ):
        self.transition_prior = transition_prior
        self.label_token_ids = [int(token_id) for token_id in label_token_ids]
        self.scale = float(scale)
        self.initial_length = int(initial_length) if initial_length is not None else None
        self.film_gamma_raw = film_gamma_raw
        self.modulation = str(modulation).strip().lower()
        self.film_norm = str(film_norm).strip().lower()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.use_gamma = bool(use_gamma)
        self.use_beta = bool(use_beta)
        self.beta_gate = str(beta_gate).strip().lower()
        self.gate_temperature = float(gate_temperature)
        self.gate_power = float(gate_power)
        self.gate_min = float(gate_min)
        self.eps = float(eps)
        self.capture_final_label_logits = bool(capture_final_label_logits)
        self.captured_final_label_logits = None

    def __call__(self, input_ids, scores):
        if self.initial_length is not None and int(input_ids.size(1)) > self.initial_length:
            return scores
        if self.transition_prior is None or len(self.label_token_ids) == 0:
            return scores
        prior = self.transition_prior.to(device=scores.device)
        gamma_raw = None
        if self.film_gamma_raw is not None:
            gamma_raw = self.film_gamma_raw.to(device=scores.device)
        if scores.size(0) != prior.size(0):
            if scores.size(0) % max(prior.size(0), 1) != 0:
                return scores
            repeat = scores.size(0) // prior.size(0)
            prior = prior.repeat_interleave(repeat, dim=0)
            if gamma_raw is not None:
                gamma_raw = gamma_raw.repeat_interleave(repeat, dim=0)
        modulation_result = _apply_speaker_mood_logit_modulation(
            scores.unsqueeze(1),
            transition_prior=prior,
            film_gamma_raw=gamma_raw,
            label_token_ids=self.label_token_ids,
            scale=self.scale,
            modulation=self.modulation,
            film_norm=self.film_norm,
            gamma_scale=self.gamma_scale,
            beta_scale=self.beta_scale,
            use_gamma=self.use_gamma,
            use_beta=self.use_beta,
            beta_gate=self.beta_gate,
            gate_temperature=self.gate_temperature,
            gate_power=self.gate_power,
            gate_min=self.gate_min,
            eps=self.eps,
            return_details=self.capture_final_label_logits,
        )
        if self.capture_final_label_logits:
            adjusted, details = modulation_result
            if details is not None:
                captured = details["final_label_logits_fp32"]
                if captured.dim() == 3 and captured.size(1) == 1:
                    captured = captured[:, 0]
                self.captured_final_label_logits = captured.detach()
        else:
            adjusted = modulation_result
        return adjusted.squeeze(1)


def _is_true_flag(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_model_hidden_size(model):
    configs = []
    for candidate in (model, getattr(model, "module", None), getattr(model, "base_model", None)):
        cfg = getattr(candidate, "config", None)
        if cfg is not None:
            configs.append(cfg)
            for nested_name in ("text_config", "thinker_config", "language_config"):
                nested = getattr(cfg, nested_name, None)
                if nested is not None:
                    configs.append(nested)
    for cfg in configs:
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embedding is not None and hasattr(embedding, "weight"):
        return int(embedding.weight.shape[-1])
    raise ValueError("Could not infer model hidden size for contrastive projection head.")


def attach_contrastive_projection_head(model, args):
    if not _is_true_flag(getattr(args, "enable_contrastive", "False")):
        return model
    if hasattr(model, "contrastive_projection"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    proj_dim = int(getattr(args, "contrastive_proj_dim", 128))
    head = ContrastiveProjectionHead(hidden_size, proj_dim)
    try:
        first_param = next(model.parameters())
        head = head.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("contrastive_projection", head)
    if is_main_process:
        print(f"[Contrastive] projection_head hidden={hidden_size}, proj_dim={proj_dim}")
    return model


def attach_personality_regression_head(model, args):
    pair_mode = str(
        getattr(args, "personality_pair_mode", "none") or "none"
    ).strip().lower()
    if (
        not _is_true_flag(
            getattr(args, "enable_personality_regression_head", "False")
        )
        and pair_mode == "none"
    ):
        return model
    if hasattr(model, "personality_regression_head"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    head = PersonalityRegressionHead(
        hidden_size=hidden_size,
        hidden_dim=int(getattr(args, "personality_regression_hidden_dim", 128)),
        dropout=float(getattr(args, "personality_regression_dropout", 0.1)),
    )
    try:
        first_param = next(model.parameters())
        head = head.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("personality_regression_head", head)
    if is_main_process:
        print(
            "[PersonalityRegression] enabled=True "
            f"hidden={hidden_size}, head_hidden={getattr(args, 'personality_regression_hidden_dim', 128)}, "
            f"loss={getattr(args, 'personality_regression_loss_type', 'mse_ccc')}, "
            f"weight={getattr(args, 'personality_regression_loss_weight', 1.0)}"
        )
    return model


def attach_precomputed_visual_adapter(model, args):
    if not (
        _is_true_flag(getattr(args, "video_enable", "False"))
        and str(getattr(args, "video_embedding_source", "")).strip().lower()
        == "precomputed_tokens"
        and str(getattr(args, "video_embedding_usage", "direct")).strip().lower()
        in ("direct", "both")
    ):
        return model
    if hasattr(model, "precomputed_visual_adapter"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    adapter = PrecomputedVisualAdapter(
        hidden_size=hidden_size,
        bottleneck_dim=int(getattr(args, "video_adapter_hidden_dim", 256)),
        dropout=float(getattr(args, "video_adapter_dropout", 0.1)),
        scale_init=float(getattr(args, "video_adapter_scale_init", 0.1)),
        output_scale_init=float(
            getattr(args, "video_adapter_output_scale_init", 0.05)
        ),
    )
    try:
        first_param = next(model.parameters())
        adapter = adapter.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("precomputed_visual_adapter", adapter)
    if is_main_process:
        print(
            "[PrecomputedVisual] adapter enabled "
            f"hidden={hidden_size}, bottleneck={getattr(args, 'video_adapter_hidden_dim', 256)}, "
            f"output_scale={getattr(args, 'video_adapter_output_scale_init', 0.05)}"
        )
    return model


def attach_precomputed_audio_adapter(model, args):
    if not (
        _is_true_flag(getattr(args, "audio_enable", "False"))
        and str(getattr(args, "audio_embedding_source", "")).strip().lower()
        == "precomputed_tokens"
        and str(getattr(args, "audio_embedding_usage", "direct")).strip().lower()
        in ("direct", "both")
    ):
        return model
    adapter_mode = str(
        getattr(args, "audio_adapter_mode", "residual") or "residual"
    ).strip().lower()
    if adapter_mode == "identity":
        if is_main_process:
            print(
                "[PrecomputedAudio] identity insertion enabled; "
                "residual adapter disabled"
            )
        return model
    if hasattr(model, "precomputed_audio_adapter"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    adapter = PrecomputedAudioAdapter(
        hidden_size=hidden_size,
        bottleneck_dim=int(getattr(args, "audio_adapter_hidden_dim", 256)),
        dropout=float(getattr(args, "audio_adapter_dropout", 0.1)),
        scale_init=float(getattr(args, "audio_adapter_scale_init", 0.1)),
        output_scale_init=float(
            getattr(args, "audio_adapter_output_scale_init", 1.0)
        ),
    )
    try:
        first_param = next(model.parameters())
        adapter = adapter.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("precomputed_audio_adapter", adapter)
    if is_main_process:
        print(
            "[PrecomputedAudio] adapter enabled "
            f"hidden={hidden_size}, "
            f"bottleneck={getattr(args, 'audio_adapter_hidden_dim', 256)}, "
            f"residual_scale={getattr(args, 'audio_adapter_scale_init', 0.1)}, "
            f"output_scale={getattr(args, 'audio_adapter_output_scale_init', 1.0)}"
        )
    return model


def attach_stage1_multimodal_fusion(model, args):
    if not _is_true_flag(getattr(args, "enable_stage1_multimodal_fusion", "False")):
        return model
    if hasattr(model, "stage1_multimodal_fusion"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    fusion = Stage1MultimodalAttentionFusion(
        hidden_size=hidden_size,
        fusion_dim=int(getattr(args, "stage1_fusion_dim", 256)),
        num_heads=int(getattr(args, "stage1_fusion_heads", 4)),
        prefix_len=int(getattr(args, "stage1_fusion_prefix_len", 2)),
        dropout=float(getattr(args, "stage1_fusion_dropout", 0.1)),
        scale_init=float(getattr(args, "stage1_fusion_scale_init", 0.1)),
        query_mode=getattr(args, "stage1_fusion_query_mode", "generic"),
        aux_gate_init=float(getattr(args, "stage1_fusion_aux_gate_init", 0.05)),
    )
    try:
        first_param = next(model.parameters())
        fusion = fusion.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("stage1_multimodal_fusion", fusion)
    if is_main_process:
        print(
            "[Stage1MultimodalFusion] enabled "
            f"hidden={hidden_size}, dim={getattr(args, 'stage1_fusion_dim', 256)}, "
            f"heads={getattr(args, 'stage1_fusion_heads', 4)}, "
            f"prefix_len={fusion.prefix_len}, "
            f"query_mode={getattr(args, 'stage1_fusion_query_mode', 'generic')}, "
            f"aux_gate_init={getattr(args, 'stage1_fusion_aux_gate_init', 0.05)}"
        )
    return model


def attach_stage1_context_prefix(model, args):
    if not _is_true_flag(getattr(args, "enable_stage1_context_prefix", "False")):
        return model
    if hasattr(model, "stage1_context_prefix"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    feature_dim = int(getattr(args, "stage1_context_feature_dim", 3584))
    prefix = Stage1TriModalContextPrefix(
        hidden_size=hidden_size,
        feature_dim=feature_dim,
        fusion_dim=int(getattr(args, "stage1_context_fusion_dim", 256)),
        gru_hidden=int(getattr(args, "stage1_context_gru_hidden", 128)),
        num_heads=int(getattr(args, "stage1_context_heads", 4)),
        prefix_len=int(getattr(args, "stage1_context_prefix_len", 2)),
        dropout=float(getattr(args, "stage1_context_dropout", 0.1)),
        scale_init=float(getattr(args, "stage1_context_scale_init", 0.05)),
        audio_gate_init=float(getattr(args, "stage1_context_audio_gate_init", 0.10)),
        visual_gate_init=float(getattr(args, "stage1_context_visual_gate_init", 0.001)),
        mode=str(getattr(args, "stage1_context_prefix_mode", "ta_cross")),
        arch=str(getattr(args, "stage1_context_arch", "gru_cross")),
        consistency_weight=float(getattr(args, "stage1_context_consistency_weight", 0.0)),
    )
    try:
        first_param = next(model.parameters())
        prefix = prefix.to(device=first_param.device)
    except StopIteration:
        pass
    model.add_module("stage1_context_prefix", prefix)
    if is_main_process:
        print(
            "[Stage1ContextPrefix] enabled "
            f"hidden={hidden_size}, feature_dim={feature_dim}, "
            f"mode={getattr(args, 'stage1_context_prefix_mode', 'ta_cross')}, "
            f"arch={getattr(args, 'stage1_context_arch', 'gru_cross')}, "
            f"prefix_len={getattr(args, 'stage1_context_prefix_len', 2)}, "
            f"audio_usage={getattr(args, 'audio_embedding_usage', 'direct')}, "
            f"video_usage={getattr(args, 'video_embedding_usage', 'direct')}, "
            f"ag={getattr(args, 'stage1_context_audio_gate_init', 0.10)}, "
            f"vg={getattr(args, 'stage1_context_visual_gate_init', 0.001)}"
        )
    return model


def attach_utterance_feature_prefix(model, args):
    if not (getattr(args, 'utterance_feature_text', False) or getattr(args, 'utterance_feature_audio', False)):
        return model
    if hasattr(model, 'utterance_feature_prefix'):
        return model
    hidden = _resolve_model_hidden_size(model)
    module = UtteranceFeaturePrefix(
        text_dim=int(getattr(args, 'utterance_feature_text_dim', hidden)),
        audio_dim=int(getattr(args, 'utterance_feature_audio_dim', hidden)),
        hidden_size=hidden,
        use_text=args.utterance_feature_text,
        use_audio=args.utterance_feature_audio,
    )
    try:
        first = next(model.parameters())
        module = module.to(device=first.device, dtype=first.dtype)
    except StopIteration:
        pass
    model.add_module('utterance_feature_prefix', module)
    if is_main_process:
        print(f'[UtteranceFeaturePrefix] text={args.utterance_feature_text} audio={args.utterance_feature_audio} hidden={hidden} order=text,audio', flush=True)
    return model


def _apply_utterance_feature_prefix_to_inputs(model, model_inputs, args):
    if not (getattr(args, 'utterance_feature_text', False) or getattr(args, 'utterance_feature_audio', False)):
        return model_inputs
    result = dict(model_inputs)
    keys = ('utterance_text_features', 'utterance_audio_features', 'utterance_text_mask', 'utterance_audio_mask')
    missing = [key for key in keys if key not in result]
    if missing:
        raise ValueError(f'Utterance feature prefix missing inputs: {missing}')
    text_features, audio_features, text_mask, audio_mask = [result.pop(key) for key in keys]
    result.pop('utterance_turn_ids', None)
    module = getattr(model, 'utterance_feature_prefix', None)
    if module is None and getattr(model, 'module', None) is not None:
        module = getattr(model.module, 'utterance_feature_prefix', None)
    if module is None:
        raise ValueError('Utterance feature projection module is missing')
    input_embeds = result.get('inputs_embeds')
    if input_embeds is None:
        _, embedding = _get_embedding_owner(model)
        input_embeds = embedding(result['input_ids'])
        result['inputs_embeds'] = input_embeds
    prefix, mask = module(text_features, audio_features, text_mask, audio_mask, dtype=input_embeds.dtype)
    if prefix is None:
        return result
    # A common zero dependency guarantees projection gradients exist on ranks
    # containing only speaker auxiliary rows (whose feature mask is all zero).
    result['inputs_embeds'] = result['inputs_embeds'] + prefix.sum() * 0.0
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = getattr(tokenizer, 'eos_token_id', 0)
    return prepend_masked_prefix(result, prefix, mask, int(pad_id or 0))


def attach_personality_prefix_head(model, args):
    if not _is_true_flag(getattr(args, "enable_personality_prefix", "False")):
        return model
    if _is_true_flag(getattr(args, "enable_latent_mood_prefix", "False")):
        raise ValueError("--enable_latent_mood_prefix and --enable_personality_prefix are mutually exclusive.")
    if _is_true_flag(getattr(args, "enable_personality_mood_prefix", "False")):
        raise ValueError("--enable_personality_mood_prefix is mutually exclusive with --enable_personality_prefix.")
    if hasattr(model, "personality_prefix"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    prefix = PersonalityInteractionPrefix(
        hidden_size=hidden_size,
        prefix_len=int(getattr(args, "personality_prefix_len", 4)),
        personality_hidden_dim=int(getattr(args, "personality_prefix_hidden_dim", 128)),
        dropout=float(getattr(args, "personality_prefix_dropout", 0.1)),
        mode=str(getattr(args, "personality_prefix_mode", "interaction")),
        use_gate=_is_true_flag(getattr(args, "personality_prefix_gate", "True")),
        scale_init=float(getattr(args, "personality_prefix_scale_init", 0.1)),
    )
    try:
        first_param = next(model.parameters())
        prefix = prefix.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("personality_prefix", prefix)
    if is_main_process:
        print(
            "[PersonalityPrefix] enabled=True "
            f"mode={getattr(args, 'personality_prefix_mode', 'interaction')}, "
            f"source={getattr(args, 'personality_prefix_source', 'auto')}, "
            f"prefix_len={getattr(args, 'personality_prefix_len', 4)}, hidden={hidden_size}"
        )
    return model


def attach_latent_mood_prefix_head(model, args):
    if not _is_true_flag(getattr(args, "enable_latent_mood_prefix", "False")):
        return model
    if _is_true_flag(getattr(args, "enable_personality_prefix", "False")):
        raise ValueError("--enable_latent_mood_prefix and --enable_personality_prefix are mutually exclusive.")
    if _is_true_flag(getattr(args, "enable_personality_mood_prefix", "False")):
        raise ValueError("--enable_personality_mood_prefix is mutually exclusive with --enable_latent_mood_prefix.")
    if hasattr(model, "latent_mood_prefix"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    prefix = PersonalityConditionedLatentMoodPrefix(
        hidden_size=hidden_size,
        prefix_len=int(getattr(args, "latent_mood_prefix_len", 4)),
        mood_dim=int(getattr(args, "latent_mood_dim", 128)),
        mood_hidden_dim=int(getattr(args, "latent_mood_hidden_dim", 128)),
        dropout=float(getattr(args, "latent_mood_dropout", 0.1)),
        mode=str(getattr(args, "latent_mood_mode", "interaction")),
        scale_init=float(getattr(args, "latent_mood_scale_init", 0.05)),
    )
    try:
        first_param = next(model.parameters())
        prefix = prefix.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("latent_mood_prefix", prefix)
    if is_main_process:
        print(
            "[LatentMoodPrefix] enabled=True "
            f"mode={getattr(args, 'latent_mood_mode', 'interaction')}, "
            f"source={getattr(args, 'latent_mood_source', 'auto')}, "
            f"prefix_len={getattr(args, 'latent_mood_prefix_len', 4)}, "
            f"mood_dim={getattr(args, 'latent_mood_dim', 128)}, hidden={hidden_size}"
        )
    return model


def attach_personality_mood_prefix_head(model, args):
    if not _is_true_flag(getattr(args, "enable_personality_mood_prefix", "False")):
        return model
    if _is_true_flag(getattr(args, "enable_personality_prefix", "False")) or _is_true_flag(
        getattr(args, "enable_latent_mood_prefix", "False")
    ):
        raise ValueError(
            "--enable_personality_mood_prefix is mutually exclusive with "
            "--enable_personality_prefix and --enable_latent_mood_prefix."
        )
    if hasattr(model, "personality_mood_prefix"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    prefix = PersonalityMoodHybridPrefix(
        hidden_size=hidden_size,
        prefix_len=int(getattr(args, "personality_mood_prefix_len", 8)),
        hybrid_hidden_dim=int(getattr(args, "personality_mood_hidden_dim", 192)),
        mood_dim=int(getattr(args, "personality_mood_dim", 128)),
        dropout=float(getattr(args, "personality_mood_dropout", 0.1)),
        mode=str(getattr(args, "personality_mood_mode", "factorized")),
        scale_init=float(getattr(args, "personality_mood_scale_init", 0.05)),
    )
    try:
        first_param = next(model.parameters())
        prefix = prefix.to(device=first_param.device, dtype=first_param.dtype)
    except StopIteration:
        pass
    model.add_module("personality_mood_prefix", prefix)
    if is_main_process:
        print(
            "[PersonalityMoodPrefix] enabled=True "
            f"mode={getattr(args, 'personality_mood_mode', 'factorized')}, "
            f"source={getattr(args, 'personality_mood_source', 'auto')}, "
            f"prefix_len={getattr(args, 'personality_mood_prefix_len', 8)}, "
            f"mood_dim={getattr(args, 'personality_mood_dim', 128)}, hidden={hidden_size}"
        )
    return model


def _get_contrastive_projection(model):
    if hasattr(model, "contrastive_projection"):
        return model.contrastive_projection
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "contrastive_projection"):
        return module.contrastive_projection
    return None


def _get_personality_prefix(model):
    if hasattr(model, "personality_prefix"):
        return model.personality_prefix
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "personality_prefix"):
        return module.personality_prefix
    return None


def _get_personality_regression_head(model):
    if hasattr(model, "personality_regression_head"):
        return model.personality_regression_head
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "personality_regression_head"):
        return module.personality_regression_head
    return None


def _get_latent_mood_prefix(model):
    if hasattr(model, "latent_mood_prefix"):
        return model.latent_mood_prefix
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "latent_mood_prefix"):
        return module.latent_mood_prefix
    return None


def _get_personality_mood_prefix(model):
    if hasattr(model, "personality_mood_prefix"):
        return model.personality_mood_prefix
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "personality_mood_prefix"):
        return module.personality_mood_prefix
    return None


def attach_mood_dynamics_head(model, args, num_emotions):
    if not _is_true_flag(getattr(args, "use_mood_dynamics", "False")):
        return model
    if hasattr(model, "mood_dynamics_head"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    head = MoodDynamicsTransitionHead(
        hidden_size=hidden_size,
        num_emotions=int(num_emotions),
        mood_dim=int(getattr(args, "mood_dynamics_dim", 128)),
        hidden_dim=int(getattr(args, "mood_dynamics_hidden_dim", 192)),
        dropout=float(getattr(args, "mood_dynamics_dropout", 0.1)),
    )
    model.add_module("mood_dynamics_head", head)
    if is_main_process:
        print(
            "[MoodDynamics] enabled=True "
            f"num_emotions={num_emotions}, mood_dim={getattr(args, 'mood_dynamics_dim', 128)}, "
            f"hidden_dim={getattr(args, 'mood_dynamics_hidden_dim', 192)}, "
            f"lambda={getattr(args, 'mood_transition_lambda', 0.2)}"
        )
    return model


def _get_mood_dynamics_head(model):
    if hasattr(model, "mood_dynamics_head"):
        return model.mood_dynamics_head
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "mood_dynamics_head"):
        return module.mood_dynamics_head
    return None


def attach_speaker_mood_dynamics_head(model, args, num_emotions):
    if not _is_true_flag(getattr(args, "use_speaker_mood_dynamics", "False")):
        return model
    if hasattr(model, "speaker_mood_dynamics_head"):
        return model
    hidden_size = _resolve_model_hidden_size(model)
    head_kwargs = {
        "hidden_size": hidden_size,
        "num_emotions": int(num_emotions),
        "speaker_mood_dim": int(getattr(args, "speaker_mood_dim", 128)),
        "hidden_dim": int(getattr(args, "speaker_mood_hidden_dim", 192)),
        "dropout": float(getattr(args, "speaker_mood_dropout", 0.1)),
    }
    if _speaker_mood_simplified_enabled(args):
        simplified_flags = {
            "use_personality": _is_true_flag(
                getattr(args, "speaker_mood_simplified_use_personality", "True")
            ),
            "use_previous_emotion": _is_true_flag(
                getattr(args, "speaker_mood_simplified_use_previous_emotion", "True")
            ),
            "use_interaction": _is_true_flag(
                getattr(args, "speaker_mood_simplified_use_interaction", "True")
            ),
            "use_current_evidence": _is_true_flag(
                getattr(args, "speaker_mood_simplified_use_current_evidence", "True")
            ),
            "interaction_attention_mode": str(
                getattr(args, "speaker_mood_interaction_attention_mode", "learned")
            ),
            "interaction_use_personality": _is_true_flag(
                getattr(args, "speaker_mood_interaction_use_personality", "True")
            ),
            "interaction_use_relation": _is_true_flag(
                getattr(args, "speaker_mood_interaction_use_relation", "True")
            ),
            "interaction_profile_permutation": str(
                getattr(args, "speaker_mood_interaction_profile_permutation", "none")
            ),
            "interaction_attention_temperature": float(
                getattr(args, "speaker_mood_interaction_attention_temperature", 1.0)
            ),
            "interaction_min_interlocutors": int(
                getattr(args, "speaker_mood_interaction_min_interlocutors", 1)
            ),
            "interaction_summary_scale": float(
                getattr(args, "speaker_mood_interaction_summary_scale", 1.0)
            ),
            "interaction_recency_bias": float(
                getattr(args, "speaker_mood_interaction_recency_bias", 0.0)
            ),
            "interaction_previous_bias": float(
                getattr(args, "speaker_mood_interaction_previous_bias", 0.0)
            ),
            "personality_temperature": float(
                getattr(args, "speaker_mood_personality_temperature", 1.0)
            ),
            "personality_center": float(
                getattr(args, "speaker_mood_personality_center", 4.0)
            ),
            "personality_residual_scale": float(
                getattr(args, "speaker_mood_personality_residual_scale", 0.0)
            ),
        }
        head = SimplifiedSpeakerMoodDynamicsHead(**head_kwargs, **simplified_flags)
    else:
        head = SpeakerSpecificMoodDynamicsHead(
            **head_kwargs,
            logit_modulation=getattr(args, "speaker_mood_logit_modulation", "additive"),
            film_condition=getattr(args, "speaker_mood_film_condition", "mood_context"),
        )
    model.add_module("speaker_mood_dynamics_head", head)
    if is_main_process:
        print(
            "[SpeakerMoodDynamics] enabled=True "
            f"architecture={getattr(args, 'speaker_mood_architecture', 'legacy')}, "
            f"num_emotions={num_emotions}, mood_dim={getattr(args, 'speaker_mood_dim', 128)}, "
            f"hidden_dim={getattr(args, 'speaker_mood_hidden_dim', 192)}, "
            f"lambda={getattr(args, 'speaker_mood_lambda', 0.2)}, "
            f"modulation={getattr(args, 'speaker_mood_logit_modulation', 'additive')}, "
            f"film_condition={getattr(args, 'speaker_mood_film_condition', 'mood_context')}, "
            f"film_beta_gate={getattr(args, 'speaker_mood_film_beta_gate', 'none')}, "
            f"recurrence={getattr(args, 'speaker_mood_recurrence_mode', 'none')}, "
            f"prev_emotion={_speaker_mood_prev_emotion_source(args)}, "
            f"dialogue_ordered={getattr(args, 'speaker_mood_dialogue_ordered', False)}"
        )
        if _speaker_mood_simplified_enabled(args):
            trainable = sum(parameter.numel() for parameter in head.parameters())
            print(
                "[SpeakerMoodSimplified] "
                f"personality={head.use_personality}, "
                f"previous_emotion={head.use_previous_emotion}, "
                f"interaction={head.use_interaction}, "
                f"interaction_attention={head.interaction_attention_mode}, "
                f"interaction_personality={head.interaction_use_personality}, "
                f"interaction_relation={head.interaction_use_relation}, "
                f"interaction_profile_permutation={head.interaction_profile_permutation}, "
                f"interaction_temperature={head.interaction_attention_temperature}, "
                f"interaction_min_interlocutors={head.interaction_min_interlocutors}, "
                f"interaction_summary_scale={head.interaction_summary_scale}, "
                f"interaction_recency_bias={head.interaction_recency_bias}, "
                f"interaction_previous_bias={head.interaction_previous_bias}, "
                f"current_evidence={head.use_current_evidence}, "
                f"personality_temperature={head.personality_temperature}, "
                f"personality_residual_scale={head.personality_residual_scale}, "
                f"trainable_head_parameters={trainable}"
            )
    return model


def _get_speaker_mood_dynamics_head(model):
    if hasattr(model, "speaker_mood_dynamics_head"):
        return model.speaker_mood_dynamics_head
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "speaker_mood_dynamics_head"):
        return module.speaker_mood_dynamics_head
    return None


def _emotion_label_token_ids(tokenizer, label_names):
    token_ids = []
    for label in label_names:
        ids = tokenizer(
            str(label),
            add_special_tokens=False,
        ).get("input_ids", [])
        if not ids:
            ids = tokenizer(
                " " + str(label),
                add_special_tokens=False,
            ).get("input_ids", [])
        token_ids.append(int(ids[0]) if ids else -1)
    return token_ids


def _get_precomputed_visual_adapter(model):
    if hasattr(model, "precomputed_visual_adapter"):
        return model.precomputed_visual_adapter
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "precomputed_visual_adapter"):
        return module.precomputed_visual_adapter
    return None


def _get_precomputed_audio_adapter(model):
    if hasattr(model, "precomputed_audio_adapter"):
        return model.precomputed_audio_adapter
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "precomputed_audio_adapter"):
        return module.precomputed_audio_adapter
    return None


def _get_stage1_multimodal_fusion(model):
    if hasattr(model, "stage1_multimodal_fusion"):
        return model.stage1_multimodal_fusion
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "stage1_multimodal_fusion"):
        return module.stage1_multimodal_fusion
    return None


def _get_stage1_context_prefix(model):
    if hasattr(model, "stage1_context_prefix"):
        return model.stage1_context_prefix
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "stage1_context_prefix"):
        return module.stage1_context_prefix
    return None


def _find_qwen_multimodal_owner(model):
    queue = [model]
    visited = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if hasattr(candidate, "get_audio_features") and hasattr(candidate, "config"):
            return candidate
        for attr in ("module", "base_model", "model"):
            child = getattr(candidate, attr, None)
            if child is not None and child is not candidate:
                queue.append(child)
    raise ValueError("Could not locate the Qwen2.5-Omni audio tower for Stage1 fusion.")


def _get_embedding_owner(model):
    for candidate in (model, getattr(model, "module", None)):
        if candidate is not None and hasattr(candidate, "get_input_embeddings"):
            embedding = candidate.get_input_embeddings()
            if embedding is not None:
                return candidate, embedding
    raise ValueError("Could not access model input embeddings for personality soft prefix.")


def _masked_mean_by_position(inputs_embeds, position_mask):
    weights = position_mask.to(inputs_embeds.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    pooled = (inputs_embeds * weights).sum(dim=1) / counts
    available = position_mask.any(dim=1)
    return pooled, available


def _prepend_soft_prefix(model_inputs, prefix_embeds):
    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    batch_size, prefix_len = prefix_embeds.shape[:2]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    prefix_ids = input_ids.new_full((batch_size, prefix_len), int(pad_id or 0))
    model_inputs["input_ids"] = torch.cat([prefix_ids, input_ids], dim=1)
    model_inputs["inputs_embeds"] = torch.cat(
        [prefix_embeds, model_inputs["inputs_embeds"]],
        dim=1,
    )
    if model_inputs.get("attention_mask") is not None:
        prefix_attention = model_inputs["attention_mask"].new_ones((batch_size, prefix_len))
        model_inputs["attention_mask"] = torch.cat(
            [prefix_attention, model_inputs["attention_mask"]],
            dim=1,
        )
    if model_inputs.get("labels") is not None:
        prefix_labels = model_inputs["labels"].new_full((batch_size, prefix_len), -100)
        model_inputs["labels"] = torch.cat([prefix_labels, model_inputs["labels"]], dim=1)
    if model_inputs.get("type_token_ids") is not None:
        prefix_types = model_inputs["type_token_ids"].new_zeros((batch_size, prefix_len))
        model_inputs["type_token_ids"] = torch.cat(
            [prefix_types, model_inputs["type_token_ids"]],
            dim=1,
        )
    return model_inputs


def _compact_model_inputs_by_keep_mask(model, model_inputs, inputs_embeds, keep_mask):
    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)
    attention_dtype = attention_mask.dtype
    keep_mask = keep_mask.to(device=input_ids.device).bool()
    attention_bool = attention_mask.to(device=input_ids.device).bool()

    _, input_embeddings = _get_embedding_owner(model)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    pad_id = int(pad_id or 0)

    kept_indices = []
    for row_idx in range(input_ids.size(0)):
        indices = torch.nonzero(keep_mask[row_idx], as_tuple=False).flatten()
        if indices.numel() == 0:
            fallback = torch.nonzero(attention_bool[row_idx], as_tuple=False).flatten()
            if fallback.numel() == 0:
                fallback = input_ids.new_tensor([0])
            indices = fallback[:1]
        kept_indices.append(indices)

    max_len = max(int(indices.numel()) for indices in kept_indices)
    batch_size = input_ids.size(0)
    new_input_ids = input_ids.new_full((batch_size, max_len), pad_id)
    new_attention_mask = attention_mask.new_zeros((batch_size, max_len), dtype=attention_dtype)
    new_inputs_embeds = input_embeddings(new_input_ids).clone()

    labels = model_inputs.get("labels")
    new_labels = None
    if labels is not None and labels.dim() == 2 and labels.shape == input_ids.shape:
        new_labels = labels.new_full((batch_size, max_len), -100)

    type_token_ids = model_inputs.get("type_token_ids")
    new_type_token_ids = None
    if (
        type_token_ids is not None
        and type_token_ids.dim() == 2
        and type_token_ids.shape == input_ids.shape
    ):
        new_type_token_ids = type_token_ids.new_zeros((batch_size, max_len))

    pad_left = str(getattr(tokenizer, "padding_side", "right")).lower() == "left"
    for row_idx, indices in enumerate(kept_indices):
        length = int(indices.numel())
        start = max_len - length if pad_left else 0
        end = start + length
        new_input_ids[row_idx, start:end] = input_ids[row_idx, indices]
        new_attention_mask[row_idx, start:end] = 1
        new_inputs_embeds[row_idx, start:end] = inputs_embeds[row_idx, indices]
        if new_labels is not None:
            new_labels[row_idx, start:end] = labels[row_idx, indices]
        if new_type_token_ids is not None:
            new_type_token_ids[row_idx, start:end] = type_token_ids[row_idx, indices]

    model_inputs["input_ids"] = new_input_ids
    model_inputs["attention_mask"] = new_attention_mask
    model_inputs["inputs_embeds"] = new_inputs_embeds
    if new_labels is not None:
        model_inputs["labels"] = new_labels
    if new_type_token_ids is not None:
        model_inputs["type_token_ids"] = new_type_token_ids
    if "position_ids" in model_inputs:
        model_inputs.pop("position_ids", None)
    return model_inputs


def _apply_stage1_multimodal_fusion_to_inputs(model, model_inputs, args):
    if not _is_true_flag(getattr(args, "enable_stage1_multimodal_fusion", "False")):
        return _apply_precomputed_visual_to_inputs(model, model_inputs, args)
    audio_source = str(
        getattr(args, "audio_embedding_source", "")
    ).strip().lower()
    input_mode = str(
        getattr(args, "stage1_fusion_input_mode", "direct") or "direct"
    ).strip().lower()
    audio_configured = (
        _is_true_flag(getattr(args, "audio_enable", "False"))
        and str(getattr(args, "audio_input_mode", "text_audio")).strip().lower()
        != "text_only"
    )
    visual_configured = (
        _is_true_flag(getattr(args, "video_enable", "False"))
        and str(getattr(args, "video_embedding_source", "")).strip().lower()
        == "precomputed_tokens"
    )
    if audio_configured and audio_source == "precomputed_tokens":
        audio_enabled = "precomputed_audio_features" in model_inputs
    elif audio_configured:
        audio_enabled = (
            "input_features" in model_inputs
            and "feature_attention_mask" in model_inputs
        )
    else:
        audio_enabled = False
    visual_enabled = visual_configured and "precomputed_visual_features" in model_inputs
    if not (audio_enabled or visual_enabled):
        return _apply_precomputed_visual_to_inputs(model, model_inputs, args)

    required = ["input_ids"]
    if audio_enabled:
        if audio_source == "precomputed_tokens":
            required.append("precomputed_audio_features")
        else:
            required.extend(["input_features", "feature_attention_mask"])
    if visual_enabled:
        required.append("precomputed_visual_features")
    if not all(key in model_inputs for key in required):
        missing = [key for key in required if key not in model_inputs]
        raise ValueError(f"Stage1 multimodal fusion is missing inputs: {missing}")

    fusion = _get_stage1_multimodal_fusion(model)
    visual_adapter = _get_precomputed_visual_adapter(model)
    audio_adapter = _get_precomputed_audio_adapter(model)
    audio_adapter_mode = str(
        getattr(args, "audio_adapter_mode", "residual") or "residual"
    ).strip().lower()
    if fusion is None:
        raise ValueError("Stage1 multimodal fusion module is missing.")
    if visual_enabled and visual_adapter is None:
        raise ValueError("Stage1 visual fusion requires a precomputed visual adapter.")
    if (
        audio_enabled
        and audio_source == "precomputed_tokens"
        and audio_adapter_mode != "identity"
        and audio_adapter is None
    ):
        raise ValueError(
            "Compressed Stage1 audio fusion requires a precomputed audio adapter."
        )

    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    _, input_embeddings = _get_embedding_owner(model)
    inputs_embeds = input_embeddings(input_ids)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    audio_mask = input_ids.new_zeros(input_ids.shape, dtype=torch.bool)
    audio_summary = inputs_embeds.new_zeros((input_ids.size(0), inputs_embeds.size(-1)))
    audio_available = input_ids.new_zeros((input_ids.size(0),), dtype=torch.bool)
    summary_embeds = inputs_embeds

    if audio_enabled and audio_source == "precomputed_tokens":
        audio_features = model_inputs.pop("precomputed_audio_features")
        if audio_adapter_mode == "identity":
            audio_features = audio_features.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            if audio_features.shape[-1] != inputs_embeds.shape[-1]:
                raise ValueError(
                    "Identity audio insertion requires matching feature and hidden "
                    f"dimensions: audio={audio_features.shape[-1]}, "
                    f"hidden={inputs_embeds.shape[-1]}"
                )
        else:
            audio_features = audio_adapter(
                audio_features,
                dtype=inputs_embeds.dtype,
            )
        audio_token_id = tokenizer.convert_tokens_to_ids(
            AUDIO_UTTERANCE_TOKEN
        )
        audio_mask = input_ids.eq(int(audio_token_id))
    elif audio_enabled:
        qwen_owner = _find_qwen_multimodal_owner(model)
        audio_outputs = qwen_owner.get_audio_features(
            model_inputs.pop("input_features"),
            feature_attention_mask=model_inputs.pop(
                "feature_attention_mask", None
            ),
            audio_feature_lengths=model_inputs.pop(
                "audio_feature_lengths", None
            ),
            return_dict=True,
        )
        audio_features = audio_outputs.last_hidden_state.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        audio_token_id = int(qwen_owner.config.audio_token_id)
        audio_mask = input_ids.eq(audio_token_id)
    if audio_enabled:
        if int(audio_mask.sum().item()) != int(audio_features.shape[0]):
            raise ValueError(
                "Stage1 fusion audio placeholder/features mismatch: "
                f"placeholders={int(audio_mask.sum().item())}, features={audio_features.shape[0]}"
            )
        summary_embeds = summary_embeds.masked_scatter(
            audio_mask.unsqueeze(-1).expand_as(summary_embeds),
            audio_features,
        )
        audio_summary, audio_available = _masked_mean_by_position(summary_embeds, audio_mask)
        if input_mode == "direct":
            inputs_embeds = summary_embeds

    visual_mask = input_ids.new_zeros(input_ids.shape, dtype=torch.bool)
    visual_summary = inputs_embeds.new_zeros((input_ids.size(0), inputs_embeds.size(-1)))
    visual_available = input_ids.new_zeros((input_ids.size(0),), dtype=torch.bool)
    if visual_enabled:
        visual_features = model_inputs.pop("precomputed_visual_features")
        visual_token_id = tokenizer.convert_tokens_to_ids(VISUAL_UTTERANCE_TOKEN)
        visual_mask = input_ids.eq(int(visual_token_id))
        if int(visual_mask.sum().item()) != int(visual_features.shape[0]):
            raise ValueError(
                "Stage1 fusion visual placeholder/features mismatch: "
                f"placeholders={int(visual_mask.sum().item())}, features={visual_features.shape[0]}"
            )
        adapted_visual = visual_adapter(visual_features, dtype=inputs_embeds.dtype)
        if input_mode == "direct":
            inputs_embeds = inputs_embeds.masked_scatter(
                visual_mask.unsqueeze(-1).expand_as(inputs_embeds),
                adapted_visual,
            )
            visual_source_embeds = inputs_embeds
        else:
            visual_source_embeds = summary_embeds.masked_scatter(
                visual_mask.unsqueeze(-1).expand_as(summary_embeds),
                adapted_visual,
            )
        visual_summary, visual_available = _masked_mean_by_position(
            visual_source_embeds,
            visual_mask,
        )
        if input_mode != "direct":
            summary_embeds = visual_source_embeds

    prompt_mask = attention_mask.bool() & ~audio_mask & ~visual_mask
    if model_inputs.get("labels") is not None:
        prompt_mask = prompt_mask & model_inputs["labels"].eq(-100)
    text_summary, text_available = _masked_mean_by_position(inputs_embeds, prompt_mask)
    modality_available = torch.stack(
        [text_available, audio_available, visual_available],
        dim=1,
    )
    prefix_embeds = fusion(
        text_summary=text_summary,
        audio_summary=audio_summary,
        visual_summary=visual_summary,
        modality_available=modality_available,
        dtype=inputs_embeds.dtype,
    )
    if input_mode == "prefix_only":
        keep_mask = attention_mask.bool() & ~(audio_mask | visual_mask)
        model_inputs = _compact_model_inputs_by_keep_mask(
            model,
            model_inputs,
            inputs_embeds,
            keep_mask,
        )
    else:
        model_inputs["inputs_embeds"] = inputs_embeds
    return _prepend_soft_prefix(model_inputs, prefix_embeds)


def _apply_stage1_context_prefix_to_inputs(model, model_inputs, args):
    if not _is_true_flag(getattr(args, "enable_stage1_context_prefix", "False")):
        return model_inputs
    prefix_module = _get_stage1_context_prefix(model)
    if prefix_module is None:
        raise ValueError("Stage1 context prefix is enabled but the module is missing.")
    required = ("context_text_features", "context_turn_mask", "input_ids")
    if not all(key in model_inputs for key in required):
        missing = [key for key in required if key not in model_inputs]
        raise ValueError(f"Stage1 context prefix is missing inputs: {missing}")

    model_inputs = dict(model_inputs)
    text_features = model_inputs.pop("context_text_features")
    turn_mask = model_inputs.pop("context_turn_mask")
    audio_features = model_inputs.pop("context_audio_features", None)
    visual_features = model_inputs.pop("context_visual_features", None)
    input_ids = model_inputs["input_ids"]
    if (
        visual_features is not None
        and _is_true_flag(getattr(args, "stage1_context_visual_shuffle", "False"))
        and visual_features.size(0) > 1
    ):
        visual_features = torch.roll(visual_features, shifts=1, dims=0)

    inputs_embeds = model_inputs.get("inputs_embeds")
    if inputs_embeds is None:
        _, input_embeddings = _get_embedding_owner(model)
        inputs_embeds = input_embeddings(input_ids)
        model_inputs["inputs_embeds"] = inputs_embeds

    prefix_embeds = prefix_module(
        text_features=text_features.to(inputs_embeds.device),
        turn_mask=turn_mask.to(inputs_embeds.device),
        audio_features=(
            None if audio_features is None else audio_features.to(inputs_embeds.device)
        ),
        visual_features=(
            None if visual_features is None else visual_features.to(inputs_embeds.device)
        ),
        dtype=inputs_embeds.dtype,
    )
    if prefix_embeds.size(1) <= 0:
        return model_inputs
    return _prepend_soft_prefix(model_inputs, prefix_embeds)


def _consume_stage1_context_aux_loss(model, args):
    if not _is_true_flag(getattr(args, "enable_stage1_context_prefix", "False")):
        return None
    weight = float(getattr(args, "stage1_context_consistency_weight", 0.0))
    if weight <= 0:
        return None
    prefix_module = _get_stage1_context_prefix(model)
    if prefix_module is None:
        return None
    aux_loss = getattr(prefix_module, "last_consistency_loss", None)
    if aux_loss is None:
        return None
    return aux_loss * weight


def _apply_precomputed_visual_to_inputs(model, model_inputs, args):
    model_inputs = _apply_precomputed_audio_to_inputs(
        model,
        model_inputs,
        args,
    )
    if not (
        _is_true_flag(getattr(args, "video_enable", "False"))
        and str(getattr(args, "video_embedding_source", "")).strip().lower()
        == "precomputed_tokens"
    ):
        return model_inputs
    if "precomputed_visual_features" not in model_inputs or "input_ids" not in model_inputs:
        return model_inputs
    adapter = _get_precomputed_visual_adapter(model)
    if adapter is None:
        raise ValueError("Precomputed visual features were provided without a visual adapter.")

    model_inputs = dict(model_inputs)
    visual_features = model_inputs.pop("precomputed_visual_features")
    input_ids = model_inputs["input_ids"]
    visual_token_id = tokenizer.convert_tokens_to_ids(VISUAL_UTTERANCE_TOKEN)
    visual_mask = input_ids.eq(int(visual_token_id))
    expected = int(visual_mask.sum().item())
    actual = int(visual_features.shape[0])
    if expected != actual:
        raise ValueError(
            f"Visual placeholder/features mismatch: placeholders={expected}, features={actual}"
        )

    inputs_embeds = model_inputs.get("inputs_embeds")
    if inputs_embeds is None:
        _, input_embeddings = _get_embedding_owner(model)
        inputs_embeds = input_embeddings(input_ids)
    adapted = adapter(visual_features, dtype=inputs_embeds.dtype)
    visual_mask = visual_mask.unsqueeze(-1).expand_as(inputs_embeds)
    model_inputs["inputs_embeds"] = inputs_embeds.masked_scatter(visual_mask, adapted)
    return model_inputs


def _apply_precomputed_audio_to_inputs(model, model_inputs, args):
    if not (
        _is_true_flag(getattr(args, "audio_enable", "False"))
        and str(getattr(args, "audio_embedding_source", "")).strip().lower()
        == "precomputed_tokens"
    ):
        return model_inputs
    if "input_ids" not in model_inputs:
        return model_inputs
    adapter_mode = str(
        getattr(args, "audio_adapter_mode", "residual") or "residual"
    ).strip().lower()
    adapter = _get_precomputed_audio_adapter(model)
    if adapter_mode != "identity" and adapter is None:
        raise ValueError(
            "Precomputed audio features were provided without an audio adapter."
        )

    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    inputs_embeds = model_inputs.get("inputs_embeds")
    if inputs_embeds is None:
        _, input_embeddings = _get_embedding_owner(model)
        inputs_embeds = input_embeddings(input_ids)
    if "precomputed_audio_features" not in model_inputs:
        if adapter is not None:
            dummy = inputs_embeds.new_zeros((1, inputs_embeds.size(-1)))
            # Same adapter forward exists on ranks whose rows replace direct
            # audio with feature prefixes; its masked zero contribution keeps
            # ZeRO-2 gradient collectives ordered across heterogeneous tasks.
            anchor = adapter(dummy, dtype=inputs_embeds.dtype).sum() * 0.0
            model_inputs["inputs_embeds"] = inputs_embeds + anchor
        return model_inputs
    audio_features = model_inputs.pop("precomputed_audio_features")
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_UTTERANCE_TOKEN)
    audio_mask = input_ids.eq(int(audio_token_id))
    expected = int(audio_mask.sum().item())
    actual = int(audio_features.shape[0])
    if expected != actual:
        raise ValueError(
            "Compressed audio placeholder/features mismatch: "
            f"placeholders={expected}, features={actual}"
        )

    if adapter_mode == "identity":
        adapted = audio_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if adapted.shape[-1] != inputs_embeds.shape[-1]:
            raise ValueError(
                "Identity audio insertion requires matching feature and hidden "
                f"dimensions: audio={adapted.shape[-1]}, "
                f"hidden={inputs_embeds.shape[-1]}"
            )
    else:
        adapted = adapter(audio_features, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + adapted.sum() * 0.0
    expanded_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
    model_inputs["inputs_embeds"] = inputs_embeds.masked_scatter(
        expanded_mask,
        adapted,
    )
    return model_inputs


def _apply_personality_prefix_to_inputs(model, model_inputs, args):
    if not _is_true_flag(getattr(args, "enable_personality_prefix", "False")):
        return model_inputs
    prefix_module = _get_personality_prefix(model)
    if prefix_module is None:
        return model_inputs
    required = ("personality_prefix_target", "personality_prefix_interlocutor", "personality_prefix_mask")
    if not all(key in model_inputs for key in required):
        return model_inputs
    if "input_ids" not in model_inputs:
        return model_inputs

    model_inputs = dict(model_inputs)
    target_personality = model_inputs.pop("personality_prefix_target")
    interlocutor_personality = model_inputs.pop("personality_prefix_interlocutor")
    prefix_mask = model_inputs.pop("personality_prefix_mask")
    input_ids = model_inputs["input_ids"]

    _, input_embeddings = _get_embedding_owner(model)
    inputs_embeds = input_embeddings(input_ids)
    prefix_embeds = prefix_module(
        target_personality.to(inputs_embeds.device),
        interlocutor_personality.to(inputs_embeds.device),
        mask=prefix_mask.to(inputs_embeds.device),
        dtype=inputs_embeds.dtype,
    )
    if prefix_embeds.size(1) <= 0:
        return model_inputs

    batch_size, prefix_len = prefix_embeds.shape[:2]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    prefix_ids = input_ids.new_full((batch_size, prefix_len), int(pad_id or 0))
    model_inputs["input_ids"] = torch.cat([prefix_ids, input_ids], dim=1)
    model_inputs["inputs_embeds"] = torch.cat([prefix_embeds, inputs_embeds], dim=1)

    if "attention_mask" in model_inputs and model_inputs["attention_mask"] is not None:
        prefix_attention = model_inputs["attention_mask"].new_ones((batch_size, prefix_len))
        model_inputs["attention_mask"] = torch.cat([prefix_attention, model_inputs["attention_mask"]], dim=1)
    if "labels" in model_inputs and model_inputs["labels"] is not None:
        prefix_labels = model_inputs["labels"].new_full((batch_size, prefix_len), -100)
        model_inputs["labels"] = torch.cat([prefix_labels, model_inputs["labels"]], dim=1)
    return model_inputs


def _apply_latent_mood_prefix_to_inputs(model, model_inputs, args):
    if not _is_true_flag(getattr(args, "enable_latent_mood_prefix", "False")):
        return model_inputs
    if _is_true_flag(getattr(args, "enable_personality_prefix", "False")):
        raise ValueError("--enable_latent_mood_prefix and --enable_personality_prefix are mutually exclusive.")
    prefix_module = _get_latent_mood_prefix(model)
    if prefix_module is None:
        return model_inputs
    required = ("personality_prefix_target", "personality_prefix_interlocutor", "personality_prefix_mask")
    if not all(key in model_inputs for key in required):
        return model_inputs
    if "input_ids" not in model_inputs:
        return model_inputs

    model_inputs = dict(model_inputs)
    target_personality = model_inputs.pop("personality_prefix_target")
    interlocutor_personality = model_inputs.pop("personality_prefix_interlocutor")
    prefix_mask = model_inputs.pop("personality_prefix_mask")
    input_ids = model_inputs["input_ids"]

    _, input_embeddings = _get_embedding_owner(model)
    inputs_embeds = input_embeddings(input_ids)
    prefix_embeds = prefix_module(
        input_embeds=inputs_embeds,
        attention_mask=model_inputs.get("attention_mask"),
        target_personality=target_personality.to(inputs_embeds.device),
        interlocutor_personality=interlocutor_personality.to(inputs_embeds.device),
        mask=prefix_mask.to(inputs_embeds.device),
        dtype=inputs_embeds.dtype,
    )
    if prefix_embeds.size(1) <= 0:
        return model_inputs

    batch_size, prefix_len = prefix_embeds.shape[:2]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    prefix_ids = input_ids.new_full((batch_size, prefix_len), int(pad_id or 0))
    model_inputs["input_ids"] = torch.cat([prefix_ids, input_ids], dim=1)
    model_inputs["inputs_embeds"] = torch.cat([prefix_embeds, inputs_embeds], dim=1)

    if "attention_mask" in model_inputs and model_inputs["attention_mask"] is not None:
        prefix_attention = model_inputs["attention_mask"].new_ones((batch_size, prefix_len))
        model_inputs["attention_mask"] = torch.cat([prefix_attention, model_inputs["attention_mask"]], dim=1)
    if "labels" in model_inputs and model_inputs["labels"] is not None:
        prefix_labels = model_inputs["labels"].new_full((batch_size, prefix_len), -100)
        model_inputs["labels"] = torch.cat([prefix_labels, model_inputs["labels"]], dim=1)
    return model_inputs


def _apply_personality_mood_prefix_to_inputs(model, model_inputs, args):
    if not _is_true_flag(getattr(args, "enable_personality_mood_prefix", "False")):
        return model_inputs
    if _is_true_flag(getattr(args, "enable_personality_prefix", "False")) or _is_true_flag(
        getattr(args, "enable_latent_mood_prefix", "False")
    ):
        raise ValueError(
            "--enable_personality_mood_prefix is mutually exclusive with "
            "--enable_personality_prefix and --enable_latent_mood_prefix."
        )
    prefix_module = _get_personality_mood_prefix(model)
    if prefix_module is None:
        return model_inputs
    required = ("personality_prefix_target", "personality_prefix_interlocutor", "personality_prefix_mask")
    if not all(key in model_inputs for key in required):
        return model_inputs
    if "input_ids" not in model_inputs:
        return model_inputs

    model_inputs = dict(model_inputs)
    target_personality = model_inputs.pop("personality_prefix_target")
    interlocutor_personality = model_inputs.pop("personality_prefix_interlocutor")
    prefix_mask = model_inputs.pop("personality_prefix_mask")
    input_ids = model_inputs["input_ids"]

    _, input_embeddings = _get_embedding_owner(model)
    inputs_embeds = model_inputs.get("inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = input_embeddings(input_ids)
        model_inputs["inputs_embeds"] = inputs_embeds
    prefix_embeds = prefix_module(
        input_embeds=inputs_embeds,
        attention_mask=model_inputs.get("attention_mask"),
        target_personality=target_personality.to(inputs_embeds.device),
        interlocutor_personality=interlocutor_personality.to(inputs_embeds.device),
        mask=prefix_mask.to(inputs_embeds.device),
        dtype=inputs_embeds.dtype,
    )
    if prefix_embeds.size(1) <= 0:
        return model_inputs
    return _prepend_soft_prefix(model_inputs, prefix_embeds)


def _compute_mood_dynamics_for_inputs(model, model_inputs, args):
    if not _is_true_flag(getattr(args, "use_mood_dynamics", "False")):
        return model_inputs
    head = _get_mood_dynamics_head(model)
    if head is None:
        return model_inputs
    required = ("personality_prefix_target", "personality_prefix_interlocutor", "personality_prefix_mask")
    if not all(key in model_inputs for key in required):
        return model_inputs
    if "input_ids" not in model_inputs:
        return model_inputs

    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    _, input_embeddings = _get_embedding_owner(model)
    inputs_embeds = model_inputs.get("inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = input_embeddings(input_ids)
        model_inputs["inputs_embeds"] = inputs_embeds

    prev_emotion = model_inputs.get("mood_prev_emotion")
    mood_state, transition_prior = head(
        input_embeds=inputs_embeds,
        attention_mask=model_inputs.get("attention_mask"),
        target_personality=model_inputs["personality_prefix_target"].to(inputs_embeds.device),
        interlocutor_personality=model_inputs["personality_prefix_interlocutor"].to(inputs_embeds.device),
        prev_emotion=None if prev_emotion is None else prev_emotion.to(inputs_embeds.device),
        mask=model_inputs["personality_prefix_mask"].to(inputs_embeds.device),
    )
    model_inputs["mood_dynamics_state"] = mood_state
    model_inputs["mood_transition_prior"] = transition_prior
    return model_inputs


def _stable_personality_noise(vector, personality_noise_std, seed, scale="raw"):
    """Return deterministic profile-level OCEAN noise in normalized units."""
    vector_cpu = vector.detach().float().cpu().contiguous()
    profile = ",".join(f"{float(value):.8f}" for value in vector_cpu.tolist())
    digest = hashlib.sha256(f"{int(seed)}|{profile}".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
    scale = str(scale or "raw").strip().lower()
    if scale == "raw":
        # Raw OCEAN is normalized as (value - 4) / 3 in the collator.
        normalized_std = float(personality_noise_std) / 3.0
    elif scale == "normalized":
        normalized_std = float(personality_noise_std)
    else:
        raise ValueError(f"Unsupported personality noise scale: {scale}")
    return torch.randn(vector_cpu.shape, generator=generator) * normalized_std


def _personality_profile_key(vector):
    return tuple(round(float(value), 7) for value in vector.detach().float().cpu())


def _random_personality_profile(vector, seed):
    key = _personality_profile_key(vector)
    digest = hashlib.sha256(f"random|{int(seed)}|{key}".encode("utf-8")).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
    # U(-1, 1) is exactly raw U(1, 7) after the model's OCEAN normalization.
    return torch.rand((5,), generator=generator) * 2.0 - 1.0


def _apply_speaker_mood_personality_noise(
    target_personality,
    context_personalities,
    target_available,
    context_available,
    raw_ocean_std,
    seed,
    noise_scale="raw",
    corruption="gaussian",
):
    raw_ocean_std = float(raw_ocean_std)
    corruption = str(corruption or "gaussian").strip().lower()
    if corruption == "none" or (corruption == "gaussian" and raw_ocean_std <= 0.0):
        return target_personality, context_personalities
    if corruption not in ("gaussian", "random"):
        raise ValueError(f"Unsupported personality corruption: {corruption}")

    target_used = target_personality.detach().clone()
    context_used = context_personalities.detach().clone()
    target_available = target_available.detach().bool().cpu().view(-1)
    context_available = context_available.detach().bool().cpu()

    for row in range(target_used.size(0)):
        if bool(target_available[row]):
            if corruption == "gaussian":
                transformed = target_used[row] + _stable_personality_noise(
                    target_used[row], raw_ocean_std, seed, scale=noise_scale
                ).to(device=target_used.device, dtype=target_used.dtype)
            else:
                transformed = _random_personality_profile(
                    target_used[row], seed
                ).to(device=target_used.device, dtype=target_used.dtype)
            target_used[row] = transformed.clamp(-1.0, 1.0)
        for col in range(context_used.size(1)):
            if not bool(context_available[row, col]):
                continue
            if corruption == "gaussian":
                transformed = context_used[row, col] + _stable_personality_noise(
                    context_used[row, col], raw_ocean_std, seed, scale=noise_scale
                ).to(device=context_used.device, dtype=context_used.dtype)
            else:
                transformed = _random_personality_profile(
                    context_used[row, col], seed
                ).to(device=context_used.device, dtype=context_used.dtype)
            context_used[row, col] = transformed.clamp(-1.0, 1.0)
    return target_used, context_used


def _compute_speaker_mood_dynamics_for_inputs(
    model,
    model_inputs,
    args,
    speaker_mood_state_cache=None,
    speaker_mood_emotion_cache=None,
):
    if not _is_true_flag(getattr(args, "use_speaker_mood_dynamics", "False")):
        return model_inputs
    head = _get_speaker_mood_dynamics_head(model)
    if head is None:
        return model_inputs
    required = (
        "speaker_mood_target_personality",
        "speaker_mood_context_personalities",
        "speaker_mood_relation_features",
        "speaker_mood_context_mask",
        "speaker_mood_target_mask",
    )
    if not all(key in model_inputs for key in required):
        return model_inputs
    if "input_ids" not in model_inputs:
        return model_inputs

    model_inputs = dict(model_inputs)
    input_ids = model_inputs["input_ids"]
    _, input_embeddings = _get_embedding_owner(model)
    inputs_embeds = model_inputs.get("inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = input_embeddings(input_ids)
        model_inputs["inputs_embeds"] = inputs_embeds

    simplified = _speaker_mood_simplified_enabled(args)
    if simplified:
        prompt_mask = model_inputs.get("attention_mask")
        if prompt_mask is None:
            prompt_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.bool,
            )
        else:
            prompt_mask = prompt_mask.to(device=inputs_embeds.device).bool()
        labels = model_inputs.get("labels")
        if labels is not None:
            labels = labels.to(device=inputs_embeds.device)
            if labels.size(1) < inputs_embeds.size(1):
                prefix = labels.new_full(
                    (labels.size(0), inputs_embeds.size(1) - labels.size(1)),
                    -100,
                )
                labels = torch.cat([prefix, labels], dim=1)
            elif labels.size(1) > inputs_embeds.size(1):
                labels = labels[:, : inputs_embeds.size(1)]
            prompt_mask = prompt_mask & labels.eq(-100)

        ordered_padding = model_inputs.get("speaker_mood_order_padding")
        target_personality_available = model_inputs[
            "speaker_mood_target_mask"
        ].to(device=inputs_embeds.device).bool()
        simplified_target_mask = torch.ones_like(
            target_personality_available, dtype=torch.bool
        )
        if ordered_padding is not None:
            simplified_target_mask &= ~ordered_padding.to(
                device=inputs_embeds.device
            ).bool().view(-1)
        previous_probs, soft_previous_valid, soft_previous_expected, _ = (
            lookup_previous_soft_emotions(
                reference=inputs_embeds,
                num_emotions=head.num_emotions,
                sample_ids=model_inputs.get("speaker_mood_sample_id"),
                speaker_ids=(
                    model_inputs.get("speaker_mood_target_speaker_id")
                    or model_inputs.get("speaker_mood_target_speaker")
                ),
                previous_sample_ids=model_inputs.get(
                    "speaker_mood_previous_sample_id"
                ),
                state_cache=speaker_mood_emotion_cache,
                padding_mask=ordered_padding,
            )
        )
        target_personality_clean = model_inputs[
            "speaker_mood_target_personality"
        ]
        context_personalities_clean = model_inputs[
            "speaker_mood_context_personalities"
        ]
        target_personality_used, context_personalities_used = (
            _apply_speaker_mood_personality_noise(
                target_personality=target_personality_clean,
                context_personalities=context_personalities_clean,
                target_available=target_personality_available,
                context_available=(
                    model_inputs["speaker_mood_context_mask"].bool()
                    & model_inputs["speaker_mood_relation_features"][..., 3].gt(0.0)
                ),
                raw_ocean_std=float(
                    getattr(args, "speaker_mood_personality_noise_std", 0.0)
                ),
                seed=int(
                    getattr(args, "speaker_mood_personality_noise_seed", 2026)
                ),
                noise_scale=str(
                    getattr(args, "speaker_mood_personality_noise_scale", "raw")
                ),
                corruption=str(
                    getattr(
                        args,
                        "speaker_mood_personality_corruption",
                        "gaussian",
                    )
                ),
            )
        )
        model_inputs["speaker_mood_target_personality_clean"] = (
            target_personality_clean
        )
        model_inputs["speaker_mood_context_personalities_clean"] = (
            context_personalities_clean
        )
        model_inputs["speaker_mood_target_personality_used"] = (
            target_personality_used
        )
        model_inputs["speaker_mood_context_personalities_used"] = (
            context_personalities_used
        )
        result = head(
            input_embeds=inputs_embeds,
            prompt_mask=prompt_mask,
            target_personality=target_personality_used.to(inputs_embeds.device),
            context_personalities=context_personalities_used.to(
                inputs_embeds.device
            ),
            relation_features=model_inputs[
                "speaker_mood_relation_features"
            ].to(inputs_embeds.device),
            context_mask=model_inputs["speaker_mood_context_mask"].to(
                inputs_embeds.device
            ),
            target_mask=simplified_target_mask,
            target_personality_available=target_personality_available,
            previous_emotion_probs=previous_probs,
            previous_emotion_valid=soft_previous_valid,
        )
        model_inputs["speaker_mood_soft_prev_valid"] = soft_previous_valid
        model_inputs["speaker_mood_soft_prev_expected"] = soft_previous_expected
        model_inputs["speaker_mood_previous_soft_emotion_probs"] = previous_probs
        model_inputs["speaker_mood_prompt_mask"] = prompt_mask
    else:
        prev_emotion_source = _speaker_mood_prev_emotion_source(args)
        prev_emotion = model_inputs.get("speaker_mood_prev_emotion")
        predicted_previous_valid = None
        predicted_previous_expected = None
        if prev_emotion_source == "none":
            prev_emotion = None
        elif prev_emotion_source == "predicted":
            prev_emotion, predicted_previous_valid, predicted_previous_expected = (
                lookup_previous_emotions(
                    reference=inputs_embeds,
                    num_emotions=head.num_emotions,
                    previous_sample_ids=model_inputs.get(
                        "speaker_mood_previous_sample_id"
                    ),
                    emotion_cache=speaker_mood_emotion_cache,
                )
            )
            ordered_padding = model_inputs.get("speaker_mood_order_padding")
            if ordered_padding is not None:
                real_rows = ~ordered_padding.to(
                    device=inputs_embeds.device
                ).bool().view(-1)
                predicted_previous_valid &= real_rows
                predicted_previous_expected &= real_rows

        recurrent_previous_mood = None
        recurrent_previous_valid = None
        recurrent_previous_expected = None
        if _speaker_mood_recurrence_enabled(args):
            recurrent_previous_mood, recurrent_previous_valid, recurrent_previous_expected = (
                lookup_previous_moods(
                    reference=inputs_embeds,
                    mood_dim=head.speaker_mood_dim,
                    previous_sample_ids=model_inputs.get("speaker_mood_previous_sample_id"),
                    state_cache=speaker_mood_state_cache,
                )
            )
            ordered_padding = model_inputs.get("speaker_mood_order_padding")
            if ordered_padding is not None:
                real_rows = ~ordered_padding.to(device=inputs_embeds.device).bool().view(-1)
                recurrent_previous_valid = recurrent_previous_valid & real_rows
                recurrent_previous_expected = recurrent_previous_expected & real_rows

        result = head(
            input_embeds=inputs_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            target_personality=model_inputs["speaker_mood_target_personality"].to(inputs_embeds.device),
            context_personalities=model_inputs["speaker_mood_context_personalities"].to(inputs_embeds.device),
            relation_features=model_inputs["speaker_mood_relation_features"].to(inputs_embeds.device),
            context_mask=model_inputs["speaker_mood_context_mask"].to(inputs_embeds.device),
            target_mask=model_inputs["speaker_mood_target_mask"].to(inputs_embeds.device),
            prev_emotion=None if prev_emotion is None else prev_emotion.to(inputs_embeds.device),
            previous_mood=recurrent_previous_mood,
            previous_mood_mask=recurrent_previous_valid,
        )
        if recurrent_previous_valid is not None:
            model_inputs["speaker_mood_recurrent_prev_valid"] = recurrent_previous_valid
            model_inputs["speaker_mood_recurrent_prev_expected"] = recurrent_previous_expected
        if predicted_previous_valid is not None:
            model_inputs["speaker_mood_predicted_prev_valid"] = predicted_previous_valid
            model_inputs["speaker_mood_predicted_prev_expected"] = (
                predicted_previous_expected
            )
    model_inputs["speaker_mood_before"] = result["mood_before"]
    model_inputs["speaker_mood_after"] = result["mood_after"]
    model_inputs["speaker_mood_transition_prior"] = result["transition_prior"]
    if result["film_gamma_raw"] is not None:
        model_inputs["speaker_mood_film_gamma_raw"] = result["film_gamma_raw"]
    model_inputs["speaker_mood_attention_weights"] = result["attention_weights"]
    model_inputs["speaker_mood_interpersonal_alignment_input"] = result["interpersonal_alignment_input"]
    model_inputs["speaker_mood_context_mask_computed"] = result["context_mask"]
    model_inputs["speaker_mood_target_mask_computed"] = result["target_mask"]
    return model_inputs


def _remove_non_model_conditioning_fields(model_inputs):
    model_inputs = dict(model_inputs)
    for key in (
        "mood_prev_emotion",
        "mood_prev_emotion_mask",
        "mood_dynamics_state",
        "mood_transition_prior",
        "speaker_mood_target_personality",
        "speaker_mood_context_personalities",
        "speaker_mood_relation_features",
        "speaker_mood_context_mask",
        "speaker_mood_target_mask",
        "speaker_mood_prev_emotion",
        "speaker_mood_prev_emotion_mask",
        "speaker_mood_target_speaker",
        "speaker_mood_target_speaker_id",
        "speaker_mood_sample_id",
        "speaker_mood_previous_sample_id",
        "speaker_mood_previous_same_speaker_sample_id",
        "speaker_mood_context_speakers",
        "speaker_mood_before",
        "speaker_mood_after",
        "speaker_mood_transition_prior",
        "speaker_mood_film_gamma_raw",
        "speaker_mood_attention_weights",
        "speaker_mood_interpersonal_alignment_input",
        "speaker_mood_context_mask_computed",
        "speaker_mood_target_mask_computed",
        "speaker_mood_recurrent_prev_valid",
        "speaker_mood_recurrent_prev_expected",
        "speaker_mood_predicted_prev_valid",
        "speaker_mood_predicted_prev_expected",
        "speaker_mood_soft_prev_valid",
        "speaker_mood_soft_prev_expected",
        "speaker_mood_previous_soft_emotion_probs",
        "speaker_mood_target_personality_clean",
        "speaker_mood_context_personalities_clean",
        "speaker_mood_target_personality_used",
        "speaker_mood_context_personalities_used",
        "speaker_mood_prompt_mask",
        "speaker_mood_dataset_index",
        "speaker_mood_order_padding",
        "personality_prefix_target",
        "personality_prefix_interlocutor",
        "personality_prefix_mask",
        "utterance_feature_text_count",
        "utterance_feature_audio_count",
        "direct_audio_utterance_count",
    ):
        model_inputs.pop(key, None)
    return model_inputs


def _apply_conditioning_to_inputs(
    model,
    model_inputs,
    args,
    compute_mood_dynamics=True,
    speaker_mood_state_cache=None,
    speaker_mood_emotion_cache=None,
):
    model_inputs = _apply_stage1_multimodal_fusion_to_inputs(model, model_inputs, args)
    if compute_mood_dynamics:
        model_inputs = _compute_mood_dynamics_for_inputs(model, model_inputs, args)
        model_inputs = _compute_speaker_mood_dynamics_for_inputs(
            model,
            model_inputs,
            args,
            speaker_mood_state_cache=speaker_mood_state_cache,
            speaker_mood_emotion_cache=speaker_mood_emotion_cache,
        )
    model_inputs = _apply_personality_prefix_to_inputs(model, model_inputs, args)
    model_inputs = _apply_latent_mood_prefix_to_inputs(model, model_inputs, args)
    model_inputs = _apply_personality_mood_prefix_to_inputs(model, model_inputs, args)
    model_inputs = _apply_stage1_context_prefix_to_inputs(model, model_inputs, args)
    model_inputs = _apply_utterance_feature_prefix_to_inputs(model, model_inputs, args)
    return model_inputs


def _speaker_mood_modulation_kwargs(args):
    return {
        "modulation": str(
            getattr(args, "speaker_mood_logit_modulation", "additive")
        ).strip().lower(),
        "film_norm": str(getattr(args, "speaker_mood_film_norm", "center")).strip().lower(),
        "gamma_scale": float(getattr(args, "speaker_mood_film_gamma_scale", 0.5)),
        "beta_scale": float(getattr(args, "speaker_mood_film_beta_scale", 1.0)),
        "use_gamma": _is_true_flag(getattr(args, "speaker_mood_film_use_gamma", "True")),
        "use_beta": _is_true_flag(getattr(args, "speaker_mood_film_use_beta", "True")),
        "beta_gate": str(getattr(args, "speaker_mood_film_beta_gate", "none")).strip().lower(),
        "gate_temperature": float(getattr(args, "speaker_mood_film_gate_temperature", 1.0)),
        "gate_power": float(getattr(args, "speaker_mood_film_gate_power", 1.0)),
        "gate_min": float(getattr(args, "speaker_mood_film_gate_min", 0.0)),
        "eps": float(getattr(args, "speaker_mood_film_eps", 1e-6)),
    }


def _apply_speaker_mood_logit_modulation(
    logits,
    transition_prior,
    film_gamma_raw,
    label_token_ids,
    scale,
    modulation="additive",
    film_norm="center",
    gamma_scale=0.5,
    beta_scale=1.0,
    use_gamma=True,
    use_beta=True,
    beta_gate="none",
    gate_temperature=1.0,
    gate_power=1.0,
    gate_min=0.0,
    eps=1e-6,
    return_details=False,
):
    unchanged = (logits, None) if return_details else logits
    if transition_prior is None or not label_token_ids:
        return unchanged

    prior = transition_prior.to(device=logits.device).float()
    if logits.size(0) != prior.size(0):
        return unchanged

    token_ids = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    valid = token_ids.ge(0) & token_ids.lt(logits.size(-1))
    if not bool(valid.any()):
        return unchanged

    valid_token_ids = token_ids[valid]
    base_label_logits = logits.index_select(-1, valid_token_ids).float()
    condition_shape = [prior.size(0)] + [1] * (base_label_logits.dim() - 2) + [int(valid.sum())]
    beta = prior[:, valid].view(*condition_shape)
    beta_ungated = beta
    mode = str(modulation).strip().lower()

    num_labels = int(valid.sum().item())
    if num_labels > 1:
        base_probs = torch.softmax(
            base_label_logits / float(gate_temperature),
            dim=-1,
        )
        base_uncertainty = -(
            base_probs * base_probs.clamp_min(float(eps)).log()
        ).sum(dim=-1, keepdim=True) / math.log(num_labels)
        base_uncertainty = base_uncertainty.clamp(0.0, 1.0).detach()
    else:
        base_uncertainty = torch.zeros_like(base_label_logits[..., :1]).detach()
    beta_gate_value = torch.ones_like(base_uncertainty)
    if mode == "film" and str(beta_gate).strip().lower() == "entropy":
        uncertainty_gate = base_uncertainty.pow(float(gate_power))
        beta_gate_value = float(gate_min) + (1.0 - float(gate_min)) * uncertainty_gate

    if mode == "film":
        centered = base_label_logits - base_label_logits.mean(dim=-1, keepdim=True)
        norm_mode = str(film_norm).strip().lower()
        if norm_mode == "none":
            base_feature = base_label_logits
        elif norm_mode == "zscore":
            rms = centered.square().mean(dim=-1, keepdim=True).add(float(eps)).sqrt()
            base_feature = centered / rms
        else:
            base_feature = centered

        if film_gamma_raw is None:
            gamma_delta = torch.zeros_like(beta)
        else:
            gamma_raw = film_gamma_raw.to(device=logits.device).float()
            if gamma_raw.size(0) != prior.size(0):
                return unchanged
            gamma_delta = float(gamma_scale) * torch.tanh(
                gamma_raw[:, valid].view(*condition_shape)
            )
        if not bool(use_gamma):
            gamma_delta = torch.zeros_like(gamma_delta)
        if not bool(use_beta):
            beta_ungated = torch.zeros_like(beta)
        else:
            beta_ungated = float(beta_scale) * beta
        beta = beta_ungated * beta_gate_value
        effective_delta = float(scale) * (gamma_delta * base_feature + beta)
    else:
        base_feature = torch.zeros_like(base_label_logits)
        gamma_delta = torch.zeros_like(beta)
        beta_ungated = beta
        effective_delta = float(scale) * beta

    final_label_logits = base_label_logits + effective_delta
    adjusted = logits.clone()
    adjusted[..., valid_token_ids] = final_label_logits.to(dtype=adjusted.dtype)
    if not return_details:
        return adjusted

    details = {
        "valid_label_mask": valid,
        "base_label_logits_fp32": base_label_logits,
        "base_feature_fp32": base_feature,
        "film_gamma_delta_fp32": gamma_delta,
        "film_gamma_fp32": 1.0 + gamma_delta,
        "film_beta_ungated_fp32": beta_ungated,
        "film_beta_fp32": beta,
        "base_uncertainty_fp32": base_uncertainty,
        "film_beta_gate_fp32": beta_gate_value,
        "effective_delta_fp32": effective_delta,
        "final_label_logits_fp32": final_label_logits,
    }
    return adjusted, details


def _add_transition_prior_to_logits(logits, transition_prior, label_token_ids, scale):
    if transition_prior is None or not label_token_ids:
        return logits
    prior = transition_prior.to(device=logits.device, dtype=logits.dtype)
    if logits.size(0) != prior.size(0):
        return logits
    token_ids = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    valid = token_ids.ge(0) & token_ids.lt(logits.size(-1))
    if not valid.any():
        return logits
    logits = logits.clone()
    logits[:, :, token_ids[valid]] = logits[:, :, token_ids[valid]] + float(scale) * prior[:, valid].unsqueeze(1)
    return logits


def _apply_mood_dynamics_to_outputs(outputs, model_inputs, args, label_token_ids):
    if not hasattr(outputs, "logits"):
        return outputs
    if _is_true_flag(getattr(args, "use_mood_dynamics", "False")):
        transition_prior = model_inputs.get("mood_transition_prior")
        if transition_prior is not None:
            outputs.logits = _add_transition_prior_to_logits(
                outputs.logits,
                transition_prior,
                label_token_ids,
                float(getattr(args, "mood_transition_lambda", 0.2)),
            )
    if _is_true_flag(getattr(args, "use_speaker_mood_dynamics", "False")):
        speaker_prior = model_inputs.get("speaker_mood_transition_prior")
        if speaker_prior is not None:
            outputs.logits = _apply_speaker_mood_logit_modulation(
                outputs.logits,
                transition_prior=speaker_prior,
                film_gamma_raw=model_inputs.get("speaker_mood_film_gamma_raw"),
                label_token_ids=label_token_ids,
                scale=float(getattr(args, "speaker_mood_lambda", 0.2)),
                **_speaker_mood_modulation_kwargs(args),
            )
    return outputs


def _label_argmax_from_logits(logits, label_token_ids, label_names):
    if logits is None or not label_token_ids:
        return [None] * (0 if logits is None else logits.size(0))
    token_ids = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    valid = token_ids.ge(0) & token_ids.lt(logits.size(-1))
    if not valid.any():
        return [None] * logits.size(0)
    scores = logits[:, token_ids[valid]]
    valid_names = [label_names[idx] for idx, ok in enumerate(valid.detach().cpu().tolist()) if ok]
    indices = scores.argmax(dim=-1).detach().cpu().tolist()
    return [str(valid_names[int(idx)]) for idx in indices]


def _speaker_mood_eval_trace(
    model_inputs,
    label_names,
    label_token_ids=None,
    base_logits=None,
    scale=0.2,
    args=None,
):
    mood_before = model_inputs.get("speaker_mood_before")
    mood_after = model_inputs.get("speaker_mood_after")
    transition_prior = model_inputs.get("speaker_mood_transition_prior")
    film_gamma_raw = model_inputs.get("speaker_mood_film_gamma_raw")
    attention_weights = model_inputs.get("speaker_mood_attention_weights")
    target_personality_clean = model_inputs.get(
        "speaker_mood_target_personality_clean"
    )
    target_personality_used = model_inputs.get(
        "speaker_mood_target_personality_used"
    )
    context_personalities_clean = model_inputs.get(
        "speaker_mood_context_personalities_clean"
    )
    context_personalities_used = model_inputs.get(
        "speaker_mood_context_personalities_used"
    )
    previous_soft_probs = model_inputs.get(
        "speaker_mood_previous_soft_emotion_probs"
    )
    previous_soft_valid = model_inputs.get("speaker_mood_soft_prev_valid")
    context_mask = model_inputs.get("speaker_mood_context_mask_computed")
    if context_mask is None:
        context_mask = model_inputs.get("speaker_mood_context_mask")
    if mood_before is None or mood_after is None or transition_prior is None:
        return None

    settings = _speaker_mood_modulation_kwargs(args)
    mode = settings["modulation"]
    with torch.no_grad():
        before = mood_before.detach().float().cpu()
        after = mood_after.detach().float().cpu()
        shift = torch.linalg.vector_norm(after - before, dim=-1).detach().float().cpu()
        priors = transition_prior.detach().float().cpu()
        transition_probs = torch.softmax(transition_prior.detach().float(), dim=-1).cpu()
        attn = attention_weights.detach().float().cpu() if attention_weights is not None else None
        mask = context_mask.detach().bool().cpu() if torch.is_tensor(context_mask) else None
        if mode == "film" and film_gamma_raw is not None and settings["use_gamma"]:
            gamma_delta = settings["gamma_scale"] * torch.tanh(
                film_gamma_raw.detach().float().cpu()
            )
        else:
            gamma_delta = torch.zeros_like(priors)
        gamma = 1.0 + gamma_delta
        if mode == "film":
            beta = settings["beta_scale"] * priors if settings["use_beta"] else torch.zeros_like(priors)
        else:
            beta = priors
        beta_ungated = beta.clone()
        base_uncertainty = None
        beta_gate_values = None
        base_features = torch.zeros_like(priors)
        effective_deltas = float(scale) * beta

    target_speakers = model_inputs.get("speaker_mood_target_speaker") or ["target"] * priors.size(0)
    context_speakers = model_inputs.get("speaker_mood_context_speakers") or [[] for _ in range(priors.size(0))]
    pred_before = [None] * priors.size(0)
    pred_after = [None] * priors.size(0)
    base_label_logits = None
    final_label_logits = None
    base_label_probs = None
    final_label_probs = None
    valid_label_names = []

    if base_logits is not None and label_token_ids is not None:
        label_logits = base_logits[:, -1, :].detach()
        _, details = _apply_speaker_mood_logit_modulation(
            label_logits.unsqueeze(1),
            transition_prior=transition_prior.to(label_logits.device),
            film_gamma_raw=None if film_gamma_raw is None else film_gamma_raw.to(label_logits.device),
            label_token_ids=label_token_ids,
            scale=float(scale),
            return_details=True,
            **settings,
        )
        if details is not None:
            valid = details["valid_label_mask"].detach().cpu()
            valid_label_names = [
                str(label_names[idx])
                for idx, ok in enumerate(valid.tolist())
                if ok
            ]
            base_label_logits = details["base_label_logits_fp32"].squeeze(1).detach().cpu()
            final_label_logits = details["final_label_logits_fp32"].squeeze(1).detach().cpu()
            base_label_probs = torch.softmax(base_label_logits, dim=-1)
            final_label_probs = torch.softmax(final_label_logits, dim=-1)
            base_indices = base_label_logits.argmax(dim=-1).tolist()
            final_indices = final_label_logits.argmax(dim=-1).tolist()
            pred_before = [valid_label_names[int(idx)] for idx in base_indices]
            pred_after = [valid_label_names[int(idx)] for idx in final_indices]
            base_features[:, valid] = details["base_feature_fp32"].squeeze(1).detach().cpu()
            gamma_delta[:, valid] = details["film_gamma_delta_fp32"].squeeze(1).detach().cpu()
            gamma[:, valid] = details["film_gamma_fp32"].squeeze(1).detach().cpu()
            beta_ungated[:, valid] = (
                details["film_beta_ungated_fp32"].squeeze(1).detach().cpu()
            )
            beta[:, valid] = details["film_beta_fp32"].squeeze(1).detach().cpu()
            base_uncertainty = (
                details["base_uncertainty_fp32"].reshape(priors.size(0), -1).mean(dim=-1).detach().cpu()
            )
            beta_gate_values = (
                details["film_beta_gate_fp32"].reshape(priors.size(0), -1).mean(dim=-1).detach().cpu()
            )
            effective_deltas[:, valid] = details["effective_delta_fp32"].squeeze(1).detach().cpu()

    traces = []
    for row_idx in range(priors.size(0)):
        names = list(context_speakers[row_idx]) if row_idx < len(context_speakers) else []
        weights = []
        if attn is not None:
            for col_idx in range(attn.size(1)):
                if mask is not None and col_idx < mask.size(1) and not bool(mask[row_idx, col_idx]):
                    continue
                name = str(names[col_idx]) if col_idx < len(names) and names[col_idx] else f"context_{col_idx}"
                weights.append(
                    {
                        "speaker": name,
                        "weight": round(float(attn[row_idx, col_idx]), 6),
                    }
                )
        trace = {
            "target_speaker": str(target_speakers[row_idx]) if row_idx < len(target_speakers) else "target",
            "context_speakers": [item["speaker"] for item in weights],
            "target_to_context_attention": weights,
            "speaker_mood_before_head": [round(float(x), 6) for x in before[row_idx, : min(16, before.size(1))].tolist()],
            "speaker_mood_after_head": [round(float(x), 6) for x in after[row_idx, : min(16, after.size(1))].tolist()],
            "speaker_mood_before": [round(float(x), 7) for x in before[row_idx].tolist()],
            "speaker_mood_after": [round(float(x), 7) for x in after[row_idx].tolist()],
            "mood_shift_index": round(float(shift[row_idx]), 6),
            "logit_modulation": {
                "mode": mode,
                "lambda": float(scale),
                "film_norm": settings["film_norm"],
                "gamma_scale": settings["gamma_scale"],
                "beta_scale": settings["beta_scale"],
                "use_gamma": settings["use_gamma"],
                "use_beta": settings["use_beta"],
                "beta_gate": settings["beta_gate"],
                "gate_temperature": settings["gate_temperature"],
                "gate_power": settings["gate_power"],
                "gate_min": settings["gate_min"],
                "base_uncertainty": None if base_uncertainty is None else round(float(base_uncertainty[row_idx]), 6),
                "beta_gate_value": None if beta_gate_values is None else round(float(beta_gate_values[row_idx]), 6),
            },
            "transition_prior": {
                str(label): round(float(priors[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "transition_prob": {
                str(label): round(float(transition_probs[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "film_gamma": {
                str(label): round(float(gamma[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "film_gamma_delta": {
                str(label): round(float(gamma_delta[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "film_beta_ungated": {
                str(label): round(float(beta_ungated[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "film_beta": {
                str(label): round(float(beta[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "film_base_feature": {
                str(label): round(float(base_features[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "effective_logits_delta_fp32": {
                str(label): round(float(effective_deltas[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            # Backward-compatible alias used by existing case-study scripts.
            "final_logits_delta": {
                str(label): round(float(effective_deltas[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "pred_before_modulation": pred_before[row_idx] if row_idx < len(pred_before) else None,
            "pred_after_modulation": pred_after[row_idx] if row_idx < len(pred_after) else None,
            "pred_before_prior": pred_before[row_idx] if row_idx < len(pred_before) else None,
            "pred_after_prior": pred_after[row_idx] if row_idx < len(pred_after) else None,
        }
        if target_personality_clean is not None:
            trace["target_personality_clean"] = [
                round(float(x), 7)
                for x in target_personality_clean[row_idx].detach().float().cpu().tolist()
            ]
        if target_personality_used is not None:
            trace["target_personality_used"] = [
                round(float(x), 7)
                for x in target_personality_used[row_idx].detach().float().cpu().tolist()
            ]
        if previous_soft_probs is not None:
            trace["previous_soft_emotion_prob"] = {
                str(label): round(float(previous_soft_probs[row_idx, idx]), 7)
                for idx, label in enumerate(label_names)
            }
            trace["previous_soft_emotion_valid"] = bool(
                previous_soft_valid[row_idx].detach().cpu().item()
            ) if previous_soft_valid is not None else False
        if context_personalities_clean is not None:
            valid_contexts = []
            for col_idx in range(context_personalities_clean.size(1)):
                if mask is not None and not bool(mask[row_idx, col_idx]):
                    continue
                valid_contexts.append(
                    {
                        "speaker": str(names[col_idx]) if col_idx < len(names) else f"context_{col_idx}",
                        "clean": [
                            round(float(x), 7)
                            for x in context_personalities_clean[row_idx, col_idx]
                            .detach().float().cpu().tolist()
                        ],
                        "used": [
                            round(float(x), 7)
                            for x in context_personalities_used[row_idx, col_idx]
                            .detach().float().cpu().tolist()
                        ] if context_personalities_used is not None else [],
                    }
                )
            trace["context_personalities"] = valid_contexts
        if base_label_logits is not None and final_label_logits is not None:
            trace["base_label_logits"] = {
                str(label): round(float(base_label_logits[row_idx, idx]), 6)
                for idx, label in enumerate(valid_label_names)
            }
            trace["final_label_logits"] = {
                str(label): round(float(final_label_logits[row_idx, idx]), 6)
                for idx, label in enumerate(valid_label_names)
            }
            trace["base_label_prob"] = {
                str(label): round(float(base_label_probs[row_idx, idx]), 6)
                for idx, label in enumerate(valid_label_names)
            }
            trace["final_label_prob"] = {
                str(label): round(float(final_label_probs[row_idx, idx]), 6)
                for idx, label in enumerate(valid_label_names)
            }
            trace["effective_probability_change"] = {
                str(label): round(
                    float(final_label_probs[row_idx, idx] - base_label_probs[row_idx, idx]),
                    6,
                )
                for idx, label in enumerate(valid_label_names)
            }
        traces.append(trace)
    return traces


def _build_mood_logits_processor_list(model_inputs, args, label_token_ids, initial_length):
    processors = []
    if _is_true_flag(getattr(args, "use_mood_dynamics", "False")) and "mood_transition_prior" in model_inputs:
        processors.append(
            MoodDynamicsLogitsProcessor(
                transition_prior=model_inputs["mood_transition_prior"],
                label_token_ids=label_token_ids,
                scale=float(getattr(args, "mood_transition_lambda", 0.2)),
                initial_length=int(initial_length),
            )
        )
    if _is_true_flag(getattr(args, "use_speaker_mood_dynamics", "False")) and "speaker_mood_transition_prior" in model_inputs:
        processors.append(
            MoodDynamicsLogitsProcessor(
                transition_prior=model_inputs["speaker_mood_transition_prior"],
                label_token_ids=label_token_ids,
                scale=float(getattr(args, "speaker_mood_lambda", 0.2)),
                initial_length=int(initial_length),
                film_gamma_raw=model_inputs.get("speaker_mood_film_gamma_raw"),
                capture_final_label_logits=_speaker_mood_simplified_enabled(args),
                **_speaker_mood_modulation_kwargs(args),
            )
        )
    return LogitsProcessorList(processors) if processors else None


def _captured_simplified_generation_label_logits(processors, batch_size):
    if processors is None:
        return None
    for processor in reversed(list(processors)):
        captured = getattr(processor, "captured_final_label_logits", None)
        if captured is None:
            continue
        if captured.dim() != 2 or captured.size(0) % int(batch_size) != 0:
            raise RuntimeError(
                "Invalid captured MoodFiLM generation logits shape: "
                f"{tuple(captured.shape)} for batch_size={batch_size}."
            )
        beam_count = captured.size(0) // int(batch_size)
        return captured.view(int(batch_size), beam_count, captured.size(-1))[:, 0]
    return None


def _maybe_compute_base_generation_logits(model, generation_forward_batch, args):
    if not _is_true_flag(getattr(args, "save_speaker_mood_analysis", "False")):
        return None
    try:
        with torch.no_grad():
            outputs = model(**generation_forward_batch)
        return getattr(outputs, "logits", None)
    except Exception as exc:
        if is_main_process:
            print(f"[WARN] speaker mood analysis base-logit forward failed: {type(exc).__name__}: {exc}")
        return None


def _mood_dynamics_eval_trace(model_inputs, label_names):
    mood_state = model_inputs.get("mood_dynamics_state")
    transition_prior = model_inputs.get("mood_transition_prior")
    if mood_state is None or transition_prior is None:
        return None
    with torch.no_grad():
        probs = torch.softmax(transition_prior.detach().float(), dim=-1).cpu()
        priors = transition_prior.detach().float().cpu()
        moods = mood_state.detach().float().cpu()
    traces = []
    for row_idx in range(priors.size(0)):
        trace = {
            "mood_state_head": [round(float(x), 6) for x in moods[row_idx, : min(16, moods.size(1))].tolist()],
            "transition_prior": {
                str(label): round(float(priors[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
            "final_logits_delta": {
                str(label): round(float(priors[row_idx, idx]) * float(getattr(args, "mood_transition_lambda", 0.2)), 6)
                for idx, label in enumerate(label_names)
            } if "args" in globals() else {},
            "transition_prob": {
                str(label): round(float(probs[row_idx, idx]), 6)
                for idx, label in enumerate(label_names)
            },
        }
        traces.append(trace)
    return traces


def _align_mask_to_hidden(mask, hidden):
    if mask is None:
        return None
    if mask.size(1) == hidden.size(1):
        return mask
    common = min(mask.size(1), hidden.size(1))
    return mask[:, -common:]


def _masked_pool_hidden(hidden, attention_mask=None, type_token_ids=None):
    if attention_mask is None:
        attention_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    else:
        attention_mask = attention_mask.to(hidden.device).bool()
    attention_mask = _align_mask_to_hidden(attention_mask, hidden)
    if attention_mask.size(1) != hidden.size(1):
        common = min(attention_mask.size(1), hidden.size(1))
        hidden = hidden[:, -common:, :]
        attention_mask = attention_mask[:, -common:]

    pool_mask = attention_mask
    if type_token_ids is not None:
        type_mask = _align_mask_to_hidden(type_token_ids.to(hidden.device), hidden)
        if type_mask is not None and type_mask.size(1) == hidden.size(1):
            prompt_mask = type_mask.eq(0) & attention_mask
            has_prompt = prompt_mask.any(dim=1, keepdim=True)
            pool_mask = torch.where(has_prompt, prompt_mask, attention_mask)

    denom = pool_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
    return (hidden * pool_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom


def _batch_trait_ccc_loss(preds, targets):
    pred_mean = preds.mean(dim=0)
    target_mean = targets.mean(dim=0)
    pred_var = preds.var(dim=0, unbiased=False)
    target_var = targets.var(dim=0, unbiased=False)
    cov = ((preds - pred_mean) * (targets - target_mean)).mean(dim=0)
    denom = pred_var + target_var + (pred_mean - target_mean).pow(2)
    ccc = (2.0 * cov) / denom.clamp(min=1e-8)
    return 1.0 - ccc.mean()


def _compute_personality_regression_predictions(model, outputs, batch):
    head = _get_personality_regression_head(model)
    if head is None or not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return None
    hidden = outputs.hidden_states[-1]
    attention_mask = batch.get("attention_mask")
    type_token_ids = batch.get("type_token_ids")
    pooled = _masked_pool_hidden(hidden, attention_mask=attention_mask, type_token_ids=type_token_ids)
    return head(pooled)


def _compute_personality_regression_loss(model, outputs, batch, args):
    if not _is_true_flag(getattr(args, "enable_personality_regression_head", "False")):
        return None
    if "personality_regression_targets" not in batch or "personality_regression_mask" not in batch:
        return None
    preds = _compute_personality_regression_predictions(model, outputs, batch)
    if preds is None:
        return None
    targets = batch["personality_regression_targets"].to(preds.device, dtype=preds.dtype)
    mask = batch["personality_regression_mask"].to(preds.device).bool()
    if preds.size(0) != targets.size(0):
        return None
    if not torch.any(mask):
        return preds.sum() * 0.0

    preds = preds[mask].float()
    targets = targets[mask].float()
    loss_type = str(getattr(args, "personality_regression_loss_type", "mse_ccc") or "mse_ccc").strip().lower()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(preds, targets)
    mse_loss = F.mse_loss(preds, targets)
    if loss_type == "mse_ccc":
        ccc_loss = _batch_trait_ccc_loss(preds, targets)
        return mse_loss + float(getattr(args, "personality_regression_ccc_weight", 0.5)) * ccc_loss
    return mse_loss


def _compute_personality_pair_predictions(model, outputs, batch):
    head = _get_personality_regression_head(model)
    if (
        head is None
        or not hasattr(outputs, "hidden_states")
        or outputs.hidden_states is None
        or "personality_pair_query_positions" not in batch
    ):
        return None
    hidden = outputs.hidden_states[-1]
    raw_positions = batch["personality_pair_query_positions"].to(hidden.device)
    input_ids = batch.get("input_ids")
    input_length = (
        int(input_ids.size(1)) if input_ids is not None else int(hidden.size(1))
    )
    prefix_offset = max(0, int(hidden.size(1)) - input_length)
    valid = raw_positions.ge(0)
    positions = raw_positions + prefix_offset
    valid = valid & positions.lt(hidden.size(1))
    if not torch.all(valid):
        return None
    batch_indices = torch.arange(
        hidden.size(0),
        device=hidden.device,
    ).unsqueeze(1).expand_as(positions)
    query_hidden = hidden[batch_indices, positions]
    return head(query_hidden.reshape(-1, hidden.size(-1))).reshape(
        hidden.size(0),
        2,
        len(PERSONALITY_TRAITS),
    )


def _compute_personality_pair_losses(model, outputs, batch, args):
    mode = str(
        getattr(args, "personality_pair_mode", "none") or "none"
    ).strip().lower()
    if mode == "none":
        return None
    required = {
        "personality_pair_targets",
        "personality_pair_mask",
        "personality_pair_query_positions",
    }
    if not required.issubset(batch):
        return None
    predictions = _compute_personality_pair_predictions(model, outputs, batch)
    if predictions is None:
        return None
    mask = batch["personality_pair_mask"].to(predictions.device).bool()
    if not torch.any(mask):
        zero = predictions.sum() * 0.0
        return {
            "total": zero,
            "regression": zero,
            "mse": zero,
            "ccc": zero,
            "auxiliary": zero,
            "predictions": predictions,
        }
    targets = batch["personality_pair_targets"].to(
        predictions.device,
        dtype=predictions.dtype,
    )
    losses = compute_personality_pair_loss(
        predictions=predictions[mask],
        targets=targets[mask],
        mode=mode,
        auxiliary_weight=float(args.personality_pair_loss_weight),
        threshold=float(args.personality_pair_threshold),
        margin=float(args.personality_pair_margin),
        ccc_weight=float(args.personality_regression_ccc_weight),
    )
    losses["predictions"] = predictions
    return losses


def _gather_first_dim_with_local_grad(tensor, pad_value=0):
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return tensor, 0, tensor.size(0)

    world = dist.get_world_size()
    rank = dist.get_rank()
    device = tensor.device
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
    size_chunks = [torch.zeros_like(local_size) for _ in range(world)]
    dist.all_gather(size_chunks, local_size)
    sizes = torch.cat(size_chunks, dim=0)
    max_size = int(sizes.max().item())
    offset = int(sizes[:rank].sum().item())
    if max_size == 0:
        return tensor[:0], offset, 0

    if tensor.size(0) < max_size:
        pad_shape = (max_size - tensor.size(0),) + tuple(tensor.shape[1:])
        padding = tensor.new_full(pad_shape, pad_value)
        padded = torch.cat([tensor, padding], dim=0)
    else:
        padded = tensor

    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded.detach().contiguous())
    gathered[rank] = padded
    parts = [chunk[: int(size.item())] for chunk, size in zip(gathered, sizes) if int(size.item()) > 0]
    if not parts:
        return tensor[:0], offset, tensor.size(0)
    return torch.cat(parts, dim=0), offset, tensor.size(0)


def _compute_contrastive_loss(model, outputs, batch, args):
    if not _is_true_flag(getattr(args, "enable_contrastive", "False")):
        return None
    projection = _get_contrastive_projection(model)
    if projection is None or not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return None
    if "contrastive_label_ids" not in batch or "contrastive_task_mask" not in batch:
        return None

    hidden = outputs.hidden_states[-1]
    label_ids = batch["contrastive_label_ids"].to(hidden.device)
    task_mask = batch["contrastive_task_mask"].to(hidden.device).bool() & label_ids.ge(0)
    if hidden.size(0) != label_ids.size(0):
        return None

    attention_mask = batch.get("attention_mask")
    attention_mask = attention_mask.to(hidden.device).bool() if attention_mask is not None else torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention_mask = _align_mask_to_hidden(attention_mask, hidden)
    if attention_mask.size(1) != hidden.size(1):
        common = min(attention_mask.size(1), hidden.size(1))
        hidden = hidden[:, -common:, :]
        attention_mask = attention_mask[:, -common:]

    pool_mask = attention_mask
    type_token_ids = batch.get("type_token_ids")
    if type_token_ids is not None:
        type_mask = _align_mask_to_hidden(type_token_ids.to(hidden.device), hidden)
        if type_mask is not None and type_mask.size(1) == hidden.size(1):
            prompt_mask = type_mask.eq(0) & attention_mask
            has_prompt = prompt_mask.any(dim=1, keepdim=True)
            pool_mask = torch.where(has_prompt, prompt_mask, attention_mask)

    denom = pool_mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden.dtype)
    pooled = (hidden * pool_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
    z_all = F.normalize(projection(pooled), p=2, dim=-1)
    zero_loss = z_all.sum() * 0.0

    z = z_all[task_mask]
    labels = label_ids[task_mask]
    local_anchor_count = z.size(0)

    gather_world = _is_true_flag(getattr(args, "contrastive_gather_world", "False"))
    contrastive_mode = str(getattr(args, "contrastive_mode", "emotion")).strip().lower()
    use_personality_weights = (
        contrastive_mode == "personality_aware"
        and "contrastive_personality" in batch
        and "contrastive_personality_mask" in batch
    )
    personality = None
    personality_valid = None
    personality_ref = None
    personality_valid_ref = None
    if gather_world:
        z_ref, local_offset, _ = _gather_first_dim_with_local_grad(z, pad_value=0.0)
        labels_ref, _, _ = _gather_first_dim_with_local_grad(labels, pad_value=-1)
        if use_personality_weights:
            personality = batch["contrastive_personality"].to(z.device, dtype=torch.float32)[task_mask]
            personality_valid = batch["contrastive_personality_mask"].to(z.device).bool()[task_mask]
            personality_ref, _, _ = _gather_first_dim_with_local_grad(personality, pad_value=0.0)
            personality_valid_ref, _, _ = _gather_first_dim_with_local_grad(personality_valid, pad_value=False)
    else:
        if local_anchor_count == 0:
            return zero_loss
        z_ref = z
        labels_ref = labels
        local_offset = 0
        if use_personality_weights:
            personality = batch["contrastive_personality"].to(z.device, dtype=torch.float32)[task_mask]
            personality_valid = batch["contrastive_personality_mask"].to(z.device).bool()[task_mask]
            personality_ref = personality
            personality_valid_ref = personality_valid
    if local_anchor_count == 0 or z_ref.size(0) < 2:
        return zero_loss

    temperature = max(float(getattr(args, "contrastive_temperature", 0.07)), 1e-6)
    logits = torch.matmul(z, z_ref.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(logits, dtype=torch.bool)
    local_positions = torch.arange(local_anchor_count, device=z.device) + int(local_offset)
    valid_positions = local_positions.lt(z_ref.size(0))
    if bool(valid_positions.any().item()):
        logits_mask[torch.arange(local_anchor_count, device=z.device)[valid_positions], local_positions[valid_positions]] = False
    positive_mask = labels.unsqueeze(1).eq(labels_ref.unsqueeze(0)) & logits_mask & labels_ref.unsqueeze(0).ge(0)
    if int(positive_mask.sum().item()) == 0:
        return zero_loss

    exp_logits = torch.exp(logits) * logits_mask.to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

    positive_weights = positive_mask.to(log_prob.dtype)
    if use_personality_weights and personality is not None and personality_ref is not None:
        pair_valid = personality_valid.unsqueeze(1) & personality_valid_ref.unsqueeze(0)
        distances = torch.cdist(personality, personality_ref, p=2)
        alpha = max(float(getattr(args, "personality_contrastive_alpha", 1.0)), 0.0)
        pair_weights = torch.exp(-alpha * distances).to(log_prob.dtype)
        pair_weights = torch.where(pair_valid, pair_weights, torch.ones_like(pair_weights))
        positive_weights = positive_weights * pair_weights

    positive_weight_sum = positive_weights.sum(dim=1)
    valid_anchor = positive_weight_sum.gt(0)
    if int(valid_anchor.sum().item()) == 0:
        return zero_loss
    loss_per_anchor = -(positive_weights * log_prob).sum(dim=1) / positive_weight_sum.clamp(min=1e-12)
    return loss_per_anchor[valid_anchor].mean()


def _parse_speaker_label(text):
    raw = str(text)

    # Prefer explicit short-form answers first (single-line classification outputs).
    non_empty_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    preferred_spans = []
    if non_empty_lines:
        preferred_spans.append(non_empty_lines[-1])
        preferred_spans.append(non_empty_lines[0])
    preferred_spans.append(raw.strip())

    strict_pat = re.compile(r'^\s*Speaker[_\s-]*(\d+)\s*[\.,;:!]?\s*$', flags=re.IGNORECASE)
    for span in preferred_spans:
        m = strict_pat.match(span)
        if m:
            return int(m.group(1))

    # Fallback: if prompt echo exists (e.g., "<Speaker_0, Speaker_1>"), the final
    # mention is usually closer to the model's actual answer than the first mention.
    all_matches = re.findall(r'Speaker[_\s-]*(\d+)', raw, flags=re.IGNORECASE)
    if all_matches:
        return int(all_matches[-1])
    return None


def _parse_personality_continuous(text):
    raw = str(text).strip()
    candidates = [raw]

    for m in re.finditer(r'```(?:json)?\s*(.*?)\s*```', raw, flags=re.IGNORECASE | re.DOTALL):
        body = m.group(1).strip()
        if body:
            candidates.append(body)

    # Include bracketed list snippets to recover from extra chatter before/after.
    for m in re.finditer(r'\[[^\[\]]+\]', raw, flags=re.DOTALL):
        snippet = m.group(0).strip()
        if snippet:
            candidates.append(snippet)

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                scores = parsed.get("scores")
                if isinstance(scores, list) and len(scores) >= 5:
                    vals = [float(scores[i]) for i in range(5)]
                    if _is_valid_personality_cont_values(vals):
                        return dict(zip(PERSONALITY_TRAITS, vals))
            if isinstance(parsed, list) and len(parsed) >= 5:
                vals = [float(parsed[i]) for i in range(5)]
                if not _is_valid_personality_cont_values(vals):
                    continue
                return dict(zip(PERSONALITY_TRAITS, vals))
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(cand)
            if isinstance(parsed, dict):
                scores = parsed.get("scores")
                if isinstance(scores, list) and len(scores) >= 5:
                    vals = [float(scores[i]) for i in range(5)]
                    if _is_valid_personality_cont_values(vals):
                        return dict(zip(PERSONALITY_TRAITS, vals))
            if isinstance(parsed, list) and len(parsed) >= 5:
                vals = [float(parsed[i]) for i in range(5)]
                if not _is_valid_personality_cont_values(vals):
                    continue
                return dict(zip(PERSONALITY_TRAITS, vals))
        except Exception:
            pass

    # Reasoning prompts use: O:x.xxx, C:x.xxx, E:x.xxx, A:x.xxx, N:x.xxx.
    trait_values = {}
    for trait in PERSONALITY_TRAITS:
        match = re.search(
            rf'(?:^|[<>\s,;]){trait}\s*[:=]\s*([+-]?\d+(?:\.\d+)?)',
            raw,
            flags=re.IGNORECASE,
        )
        if match is None:
            trait_values = {}
            break
        trait_values[trait] = float(match.group(1))
    if len(trait_values) == 5:
        vals = [trait_values[trait] for trait in PERSONALITY_TRAITS]
        if _is_valid_personality_cont_values(vals):
            return trait_values

    # Strict mode: accept only structured list outputs or explicit O/C/E/A/N scores.
    return None


def _parse_personality_discrete(text):
    raw = str(text).strip()
    candidates = [raw]

    # Typical Qwen outputs are wrapped as fenced JSON blocks.
    for m in re.finditer(r'```(?:json)?\s*(.*?)\s*```', raw, flags=re.IGNORECASE | re.DOTALL):
        body = m.group(1).strip()
        if body:
            candidates.append(body)

    # Also scan JSON-like objects embedded in longer generations.
    for m in re.finditer(r'\{[^{}]*\}', raw, flags=re.DOTALL):
        snippet = m.group(0).strip()
        if snippet:
            candidates.append(snippet)

    def _from_mapping(parsed_obj):
        if not isinstance(parsed_obj, dict):
            return None
        normalized = {str(k).strip().upper(): str(v).strip().lower() for k, v in parsed_obj.items()}
        if not all(trait in normalized for trait in PERSONALITY_TRAITS):
            return None
        result = {}
        for trait in PERSONALITY_TRAITS:
            value = normalized[trait]
            if value not in PERSONALITY_LEVELS:
                return None
            result[trait] = value
        return result

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            mapped = _from_mapping(parsed)
            if mapped is not None:
                return mapped
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(cand)
            mapped = _from_mapping(parsed)
            if mapped is not None:
                return mapped
        except Exception:
            pass

    lowered = raw.lower()
    trait_alias = {
        'O': 'openness',
        'C': 'conscientiousness',
        'E': 'extraversion',
        'A': 'agreeableness',
        'N': 'neuroticism',
    }
    result = {}
    for trait, alias in trait_alias.items():
        m1 = re.search(rf'([Ll]ow|[Mm]edium|[Hh]igh)\s+{alias}', raw)
        m2 = re.search(rf'["\']?{trait.lower()}["\']?\s*[:=]\s*["\']?(low|medium|high)["\']?', lowered)
        if m1:
            result[trait] = m1.group(1).lower()
        elif m2:
            result[trait] = m2.group(1).lower()
    if len(result) == 5:
        return result
    return None


def _compute_ccc(gold_values, pred_values):
    """Compute concordance correlation coefficient (CCC)."""
    x = np.asarray(gold_values, dtype=np.float64)
    y = np.asarray(pred_values, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    denom = x_var + y_var + (x_mean - y_mean) ** 2
    if denom <= 1e-12:
        return 0.0
    return float((2.0 * cov) / denom)


def build_task_statistics(
    targets,
    answers,
    emotional_label_dict,
    dataset,
    speaker_loss_weight=1.0,
    personality_loss_weight=0.5,
):
    report = {
        'task_counts': {
            'emotion': 0,
            'speaker': 0,
            'personality_continuous': 0,
            'personality_discrete': 0,
            'personality_aux': 0,
            'personality_pair': 0,
        },
        'emotion': None,
        'speaker': None,
        'personality_continuous': None,
        'personality_discrete': None,
        'personality_pairwise': None,
        'selection_metric': None,
    }

    emotion_golds, emotion_preds = [], []
    emotion_confuse_case = []
    speaker_golds, speaker_preds = [], []
    cont_mae_sums = {t: 0.0 for t in PERSONALITY_TRAITS}
    cont_mse_sums = {t: 0.0 for t in PERSONALITY_TRAITS}
    cont_gold_values = {t: [] for t in PERSONALITY_TRAITS}
    cont_pred_values = {t: [] for t in PERSONALITY_TRAITS}
    cont_count = 0
    disc_trait_correct = {t: 0 for t in PERSONALITY_TRAITS}
    disc_trait_total = {t: 0 for t in PERSONALITY_TRAITS}
    disc_exact_correct = 0
    disc_count = 0
    pairwise_trait_correct = {t: 0 for t in PERSONALITY_TRAITS}
    pairwise_trait_total = {t: 0 for t in PERSONALITY_TRAITS}
    pairwise_trait_golds = {t: [] for t in PERSONALITY_TRAITS}
    pairwise_trait_preds = {t: [] for t in PERSONALITY_TRAITS}
    pairwise_parse_failures = 0

    for index, (target, answer) in enumerate(zip(targets, answers)):
        gold_target = _canonical_target_text(target)
        task_type = _detect_target_task(gold_target)
        report['task_counts'][task_type] += 1

        if task_type == 'emotion':
            if gold_target not in emotional_label_dict:
                continue
            emotion_golds.append(emotional_label_dict[gold_target])
            pred_label, _, is_confused = _extract_emotion_label_with_status(answer, list(emotional_label_dict.keys()))
            if pred_label not in emotional_label_dict:
                pred_label = _emotion_fallback_label(list(emotional_label_dict.keys()))
                is_confused = True
            emotion_preds.append(emotional_label_dict[pred_label])
            if is_confused:
                emotion_confuse_case.append(index)
            continue

        if task_type == 'speaker':
            gold_spk = _parse_speaker_label(gold_target)
            pred_spk = _parse_speaker_label(answer)
            if gold_spk is not None and pred_spk is not None:
                speaker_golds.append(gold_spk)
                speaker_preds.append(pred_spk)
            continue

        if task_type == 'personality_continuous':
            gold_map = _parse_personality_continuous(gold_target)
            pred_map = _parse_personality_continuous(answer)
            if gold_map is not None and pred_map is not None:
                cont_count += 1
                for trait in PERSONALITY_TRAITS:
                    diff = float(gold_map[trait]) - float(pred_map[trait])
                    cont_mae_sums[trait] += abs(diff)
                    cont_mse_sums[trait] += diff * diff
                    cont_gold_values[trait].append(float(gold_map[trait]))
                    cont_pred_values[trait].append(float(pred_map[trait]))
            continue

        if task_type == 'personality_discrete':
            gold_map = _parse_personality_discrete(gold_target)
            pred_map = _parse_personality_discrete(answer)
            if gold_map is not None and pred_map is not None:
                disc_count += 1
                trait_match_all = True
                for trait in PERSONALITY_TRAITS:
                    disc_trait_total[trait] += 1
                    is_correct = int(gold_map[trait] == pred_map[trait])
                    disc_trait_correct[trait] += is_correct
                    trait_match_all = trait_match_all and bool(is_correct)
                if trait_match_all:
                    disc_exact_correct += 1
            continue

        if task_type == 'personality_aux':
            binary_gold_match = re.match(
                r'^\s*TraitRank_([OCEAN])\s*:\s*Speaker_([01])\s*$',
                gold_target,
                flags=re.IGNORECASE,
            )
            threeway_gold_match = re.match(
                r'^\s*TraitCompare_([OCEAN])\s*:\s*(Speaker_[01]_higher|Similar)\s*$',
                gold_target,
                flags=re.IGNORECASE,
            )
            if binary_gold_match is not None:
                trait = binary_gold_match.group(1).upper()
                gold_class = f"Speaker_{binary_gold_match.group(2)}"
                pred_match = re.search(
                    r'TraitRank_([OCEAN])\s*:\s*Speaker_([01])',
                    str(answer),
                    flags=re.IGNORECASE,
                )
                pred_trait = pred_match.group(1).upper() if pred_match is not None else None
                pred_class = f"Speaker_{pred_match.group(2)}" if pred_match is not None else None
            elif threeway_gold_match is not None:
                trait = threeway_gold_match.group(1).upper()
                raw_gold_class = threeway_gold_match.group(2)
                gold_class = (
                    "Similar"
                    if raw_gold_class.lower() == "similar"
                    else raw_gold_class.replace("speaker_", "Speaker_").replace("_HIGHER", "_higher")
                )
                pred_match = re.search(
                    r'TraitCompare_([OCEAN])\s*:\s*(Speaker_[01]_higher|Similar)',
                    str(answer),
                    flags=re.IGNORECASE,
                )
                pred_trait = pred_match.group(1).upper() if pred_match is not None else None
                if pred_match is None:
                    pred_class = None
                elif pred_match.group(2).lower() == "similar":
                    pred_class = "Similar"
                else:
                    speaker_index = re.search(r'[01]', pred_match.group(2)).group(0)
                    pred_class = f"Speaker_{speaker_index}_higher"
            else:
                continue
            pairwise_trait_total[trait] += 1
            if pred_class is None or pred_trait != trait:
                pairwise_parse_failures += 1
                pairwise_trait_golds[trait].append(gold_class)
                pairwise_trait_preds[trait].append("__parse_failure__")
                continue
            pairwise_trait_golds[trait].append(gold_class)
            pairwise_trait_preds[trait].append(pred_class)
            if pred_class == gold_class:
                pairwise_trait_correct[trait] += 1
            continue

        if task_type == "personality_pair":
            continue

    selection_value = None
    selection_name = None

    if len(emotion_golds) > 0 and len(emotion_preds) == len(emotion_golds):
        score, res_matrix, res_dict = report_score(dataset=dataset, golds=emotion_golds, preds=emotion_preds)
        score['confuse_case'] = emotion_confuse_case
        score['num_confuse_case'] = len(emotion_confuse_case)
        report['emotion'] = {
            'score': score,
            'classification_report': res_matrix,
            'classification_report_lines': res_matrix.splitlines(),
            'per_class_metrics': res_dict,
        }
        selection_name = 'emotion_F1_SA'
        selection_value = score['F1_SA']

    if len(speaker_golds) > 0 and len(speaker_preds) == len(speaker_golds):
        spk_acc = round(accuracy_score(speaker_golds, speaker_preds) * 100, 3)
        spk_f1 = round(f1_score(speaker_golds, speaker_preds, average='weighted') * 100, 3)
        report['speaker'] = {
            'evaluated_samples': len(speaker_golds),
            'acc': spk_acc,
            'f1_weighted': spk_f1,
        }
        if selection_value is None:
            selection_name = 'speaker_acc'
            selection_value = spk_acc

    if cont_count > 0:
        per_trait_mae = {
            trait: round(cont_mae_sums[trait] / cont_count, 4)
            for trait in PERSONALITY_TRAITS
        }
        per_trait_mse = {
            trait: round(cont_mse_sums[trait] / cont_count, 4)
            for trait in PERSONALITY_TRAITS
        }
        per_trait_ccc = {}
        for trait in PERSONALITY_TRAITS:
            ccc_val = _compute_ccc(cont_gold_values[trait], cont_pred_values[trait])
            per_trait_ccc[trait] = round(ccc_val, 4) if ccc_val is not None else None

        avg_mae = round(sum(per_trait_mae.values()) / len(PERSONALITY_TRAITS), 4)
        avg_mse = round(sum(per_trait_mse.values()) / len(PERSONALITY_TRAITS), 4)

        valid_ccc_values = [v for v in per_trait_ccc.values() if v is not None]
        avg_ccc = round(sum(valid_ccc_values) / len(valid_ccc_values), 4) if len(valid_ccc_values) > 0 else None

        report['personality_continuous'] = {
            'parsed_samples': cont_count,
            'per_trait_mae': per_trait_mae,
            'avg_mae': avg_mae,
            'per_trait_mse': per_trait_mse,
            'avg_mse': avg_mse,
            'per_trait_ccc': per_trait_ccc,
            'avg_ccc': avg_ccc,
        }
        if selection_value is None:
            if avg_ccc is not None:
                selection_name = 'personality_continuous_avg_ccc(%)'
                selection_value = round(100.0 * avg_ccc, 4)
            else:
                selection_name = 'personality_continuous_(100-avg_mae)'
                selection_value = round(100.0 - avg_mae, 4)

    if disc_count > 0:
        per_trait_acc = {}
        for trait in PERSONALITY_TRAITS:
            total = disc_trait_total[trait]
            per_trait_acc[trait] = round((disc_trait_correct[trait] / total) * 100, 3) if total > 0 else None
        exact_acc = round((disc_exact_correct / disc_count) * 100, 3)
        report['personality_discrete'] = {
            'parsed_samples': disc_count,
            'per_trait_acc': per_trait_acc,
            'exact_match_acc': exact_acc,
        }
        if selection_value is None:
            selection_name = 'personality_discrete_exact_match_acc'
            selection_value = exact_acc

    pairwise_total = sum(pairwise_trait_total.values())
    if pairwise_total > 0:
        per_trait_acc = {
            trait: (
                round(100.0 * pairwise_trait_correct[trait] / pairwise_trait_total[trait], 3)
                if pairwise_trait_total[trait] > 0
                else None
            )
            for trait in PERSONALITY_TRAITS
        }
        valid_pairwise_golds = []
        valid_pairwise_preds = []
        per_trait_balanced_acc = {}
        per_trait_macro_f1 = {}
        per_trait_majority_baseline = {}
        per_trait_gold_distribution = {}
        per_trait_pred_distribution = {}
        for trait in PERSONALITY_TRAITS:
            golds = pairwise_trait_golds[trait]
            preds = pairwise_trait_preds[trait]
            valid_pairwise_golds.extend(golds)
            valid_pairwise_preds.extend(preds)
            if golds:
                class_counts = {label: golds.count(label) for label in sorted(set(golds))}
                pred_counts = {label: preds.count(label) for label in sorted(set(preds))}
                per_trait_gold_distribution[trait] = class_counts
                per_trait_pred_distribution[trait] = pred_counts
                per_trait_majority_baseline[trait] = round(
                    100.0 * max(class_counts.values()) / len(golds), 3
                )
                gold_labels = sorted(class_counts)
                class_recalls = [
                    sum(
                        gold == label and pred == label
                        for gold, pred in zip(golds, preds)
                    )
                    / class_counts[label]
                    for label in gold_labels
                ]
                per_trait_balanced_acc[trait] = round(
                    100.0 * sum(class_recalls) / len(class_recalls), 3
                )
                per_trait_macro_f1[trait] = round(
                    100.0 * f1_score(
                        golds,
                        preds,
                        labels=gold_labels,
                        average='macro',
                        zero_division=0,
                    ),
                    3,
                )
            else:
                per_trait_gold_distribution[trait] = {}
                per_trait_pred_distribution[trait] = {}
                per_trait_majority_baseline[trait] = None
                per_trait_balanced_acc[trait] = None
                per_trait_macro_f1[trait] = None

        report['personality_pairwise'] = {
            'evaluated_samples': pairwise_total,
            'parsed_samples': pairwise_total - pairwise_parse_failures,
            'parse_failures': pairwise_parse_failures,
            'per_trait_acc': per_trait_acc,
            'per_trait_balanced_acc': per_trait_balanced_acc,
            'per_trait_macro_f1': per_trait_macro_f1,
            'per_trait_majority_baseline': per_trait_majority_baseline,
            'per_trait_gold_distribution': per_trait_gold_distribution,
            'per_trait_pred_distribution': per_trait_pred_distribution,
            'overall_acc': round(
                100.0 * sum(pairwise_trait_correct.values()) / pairwise_total,
                3,
            ),
            'overall_macro_f1': (
                round(100.0 * f1_score(
                    valid_pairwise_golds,
                    valid_pairwise_preds,
                    labels=sorted(set(valid_pairwise_golds)),
                    average='macro',
                    zero_division=0,
                ), 3)
                if valid_pairwise_golds
                else None
            ),
        }

    if selection_name is not None and selection_value is not None:
        report['selection_metric'] = {
            'name': selection_name,
            'value': selection_value,
            'higher_is_better': True,
        }

    stage1_speaker_count = report['task_counts'].get('speaker', 0)
    stage1_personality_count = (
        report['task_counts'].get('personality_continuous', 0)
        + report['task_counts'].get('personality_discrete', 0)
        + report['task_counts'].get('personality_aux', 0)
        + report['task_counts'].get('personality_pair', 0)
    )
    stage1_total = stage1_speaker_count + stage1_personality_count
    if stage1_total > 0:
        speaker_ratio = round(stage1_speaker_count / stage1_total, 4)
        personality_ratio = round(stage1_personality_count / stage1_total, 4)

        speaker_weighted = stage1_speaker_count * float(speaker_loss_weight)
        personality_weighted = stage1_personality_count * float(personality_loss_weight)
        weighted_total = speaker_weighted + personality_weighted
        if weighted_total > 0:
            speaker_effective_ratio = round(speaker_weighted / weighted_total, 4)
            personality_effective_ratio = round(personality_weighted / weighted_total, 4)
        else:
            speaker_effective_ratio = None
            personality_effective_ratio = None

        report['stage1_task_mix'] = {
            'speaker_samples': stage1_speaker_count,
            'personality_samples': stage1_personality_count,
            'speaker_ratio': speaker_ratio,
            'personality_ratio': personality_ratio,
            'loss_weights': {
                'speaker': float(speaker_loss_weight),
                'personality': float(personality_loss_weight),
            },
            'effective_weighted_ratio': {
                'speaker': speaker_effective_ratio,
                'personality': personality_effective_ratio,
            },
        }

        # For stage1 mixed tasks, avoid selecting checkpoints by speaker only.
        # Use a weighted joint metric when personality and speaker are both present.
        if report['emotion'] is None:
            speaker_acc = report['speaker']['acc'] if report['speaker'] is not None else None
            personality_metric = None
            personality_metric_name = None
            if report['personality_discrete'] is not None:
                personality_metric = report['personality_discrete']['exact_match_acc']
                personality_metric_name = 'personality_discrete_exact_match_acc'
            elif report['personality_continuous'] is not None:
                avg_ccc = report['personality_continuous'].get('avg_ccc')
                if avg_ccc is not None:
                    personality_metric = round(100.0 * avg_ccc, 4)
                    personality_metric_name = 'personality_continuous_avg_ccc(%)'
                else:
                    personality_metric = round(100.0 - report['personality_continuous']['avg_mae'], 4)
                    personality_metric_name = 'personality_continuous_(100-avg_mae)'

            if speaker_acc is not None and personality_metric is not None:
                sw = float(speaker_loss_weight)
                pw = float(personality_loss_weight)
                joint = round((speaker_acc * sw + personality_metric * pw) / max(sw + pw, 1e-8), 4)
                report['selection_metric'] = {
                    'name': f'stage1_joint_weighted(speaker_acc,{personality_metric_name})',
                    'value': joint,
                    'higher_is_better': True,
                }
            elif personality_metric is not None:
                report['selection_metric'] = {
                    'name': personality_metric_name,
                    'value': personality_metric,
                    'higher_is_better': True,
                }
            elif speaker_acc is not None:
                report['selection_metric'] = {
                    'name': 'speaker_acc',
                    'value': speaker_acc,
                    'higher_is_better': True,
                }
    else:
        report['stage1_task_mix'] = None

    return report


def _personality_values_to_dict(values):
    if values is None:
        return None
    try:
        vals = [float(v) for v in values]
    except Exception:
        return None
    if len(vals) != len(PERSONALITY_TRAITS):
        return None
    return {trait: round(max(1.0, min(7.0, vals[i])), 3) for i, trait in enumerate(PERSONALITY_TRAITS)}


def _build_personality_regression_report(targets, predictions):
    if predictions is None:
        return None
    cont_mae_sums = {t: 0.0 for t in PERSONALITY_TRAITS}
    cont_mse_sums = {t: 0.0 for t in PERSONALITY_TRAITS}
    cont_gold_values = {t: [] for t in PERSONALITY_TRAITS}
    cont_pred_values = {t: [] for t in PERSONALITY_TRAITS}
    count = 0

    for target, pred in zip(targets, predictions):
        gold_map = _parse_personality_continuous(target)
        pred_map = pred if isinstance(pred, dict) else _personality_values_to_dict(pred)
        if gold_map is None or pred_map is None:
            continue
        count += 1
        for trait in PERSONALITY_TRAITS:
            gold = float(gold_map[trait])
            pred_value = float(pred_map[trait])
            diff = gold - pred_value
            cont_mae_sums[trait] += abs(diff)
            cont_mse_sums[trait] += diff * diff
            cont_gold_values[trait].append(gold)
            cont_pred_values[trait].append(pred_value)

    if count == 0:
        return None

    per_trait_mae = {trait: round(cont_mae_sums[trait] / count, 4) for trait in PERSONALITY_TRAITS}
    per_trait_mse = {trait: round(cont_mse_sums[trait] / count, 4) for trait in PERSONALITY_TRAITS}
    per_trait_ccc = {}
    for trait in PERSONALITY_TRAITS:
        ccc_val = _compute_ccc(cont_gold_values[trait], cont_pred_values[trait])
        per_trait_ccc[trait] = round(ccc_val, 4) if ccc_val is not None else None
    valid_ccc_values = [v for v in per_trait_ccc.values() if v is not None]
    avg_ccc = round(sum(valid_ccc_values) / len(valid_ccc_values), 4) if valid_ccc_values else None
    return {
        "parsed_samples": count,
        "per_trait_mae": per_trait_mae,
        "avg_mae": round(sum(per_trait_mae.values()) / len(PERSONALITY_TRAITS), 4),
        "per_trait_mse": per_trait_mse,
        "avg_mse": round(sum(per_trait_mse.values()) / len(PERSONALITY_TRAITS), 4),
        "per_trait_ccc": per_trait_ccc,
        "avg_ccc": avg_ccc,
    }


def _attach_personality_regression_report(task_report, targets, regression_predictions, args):
    if not _is_true_flag(getattr(args, "enable_personality_regression_head", "False")):
        return task_report
    regression_report = _build_personality_regression_report(targets, regression_predictions)
    if regression_report is None:
        return task_report
    task_report["personality_regression_continuous"] = regression_report
    export_source = str(getattr(args, "personality_export_source", "auto") or "auto").strip().lower()
    if task_report.get("emotion") is None and export_source in ("auto", "regression"):
        avg_ccc = regression_report.get("avg_ccc")
        if avg_ccc is not None:
            task_report["selection_metric"] = {
                "name": "personality_regression_continuous_avg_ccc(%)",
                "value": round(100.0 * float(avg_ccc), 4),
                "higher_is_better": True,
            }
        else:
            task_report["selection_metric"] = {
                "name": "personality_regression_continuous_(100-avg_mae)",
                "value": round(100.0 - float(regression_report["avg_mae"]), 4),
                "higher_is_better": True,
            }
    return task_report


def _attach_personality_pair_report(task_report, records, args):
    if str(getattr(args, "personality_pair_mode", "none")).strip().lower() == "none":
        return task_report
    pair_targets = []
    pair_predictions = []
    for record in records:
        targets = record.get("personality_pair_targets")
        predictions = record.get("personality_pair_predictions")
        if targets is None or predictions is None:
            continue
        pair_targets.append(targets)
        pair_predictions.append(predictions)
    if not pair_targets:
        return task_report
    report = build_personality_pair_report(
        targets=pair_targets,
        predictions=pair_predictions,
        threshold=float(args.personality_pair_threshold),
    )
    task_report["personality_pair_structured"] = report
    continuous = report.get("continuous") or {}
    avg_ccc = continuous.get("avg_ccc")
    if avg_ccc is not None:
        task_report["selection_metric"] = {
            "name": "personality_pair_structured_avg_ccc(%)",
            "value": round(100.0 * float(avg_ccc), 4),
            "higher_is_better": True,
        }
    return task_report


def _attach_generated_continuous_pair_report(task_report, records, args):
    by_conversation = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        match = re.match(
            r"^Personality_(?:full|conversation)_(.+)_([01])$",
            sample_id,
        )
        if match is None:
            continue
        gold = _parse_personality_continuous(record.get("target"))
        prediction = _parse_personality_continuous(record.get("output"))
        if gold is None or prediction is None:
            continue
        conversation_id = match.group(1)
        speaker_index = int(match.group(2))
        by_conversation.setdefault(conversation_id, {})[speaker_index] = (
            [float(gold[trait]) for trait in PERSONALITY_TRAITS],
            [float(prediction[trait]) for trait in PERSONALITY_TRAITS],
        )

    pair_targets = []
    pair_predictions = []
    for speakers in by_conversation.values():
        if 0 not in speakers or 1 not in speakers:
            continue
        pair_targets.append([speakers[0][0], speakers[1][0]])
        pair_predictions.append([speakers[0][1], speakers[1][1]])
    if not pair_targets:
        return task_report
    task_report["personality_pair_from_continuous"] = build_personality_pair_report(
        targets=pair_targets,
        predictions=pair_predictions,
        threshold=float(getattr(args, "personality_pair_threshold", 0.5)),
    )
    return task_report


device = torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

world_size = int(os.getenv("WORLD_SIZE", '1'))

def is_rank0_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    rank_env = os.getenv("RANK")
    if rank_env is not None:
        return int(rank_env) == 0

    local_rank_env = os.getenv("LOCAL_RANK")
    if local_rank_env is not None:
        return int(local_rank_env) in (-1, 0)

    return True


is_main_process = is_rank0_process()


def load_zero_checkpoint_safe(model, checkpoint_dir):
    import deepspeed.utils.zero_to_fp32 as zero_to_fp32

    ckpt_dir = checkpoint_dir
    trainable_safetensors = os.path.join(ckpt_dir, "trainable_model.safetensors")
    trainable_bin = os.path.join(ckpt_dir, "trainable_model.bin")
    if os.path.isfile(trainable_safetensors) or os.path.isfile(trainable_bin):
        trainable_path = trainable_safetensors if os.path.isfile(trainable_safetensors) else trainable_bin
        if is_main_process:
            print(f"[INFO] Loading lightweight trainable checkpoint: {trainable_path}")
        if trainable_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load_file

            trainable_state_dict = safe_load_file(trainable_path, device="cpu")
        else:
            trainable_state_dict = torch.load(trainable_path, map_location="cpu")
        incompatible = model.load_state_dict(trainable_state_dict, strict=False)
        if is_main_process and incompatible is not None:
            missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
            print(
                "[INFO] Lightweight trainable load result: "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )
        return model

    adapter_safetensors = os.path.join(ckpt_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.isfile(adapter_safetensors) or os.path.isfile(adapter_bin):
        adapter_path = adapter_safetensors if os.path.isfile(adapter_safetensors) else adapter_bin
        if is_main_process:
            print(f"[INFO] Loading PEFT adapter checkpoint without ZeRO fp32 reconstruction: {adapter_path}")
        if adapter_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load_file

            adapter_state_dict = safe_load_file(adapter_path, device="cpu")
        else:
            adapter_state_dict = torch.load(adapter_path, map_location="cpu")
        incompatible = set_peft_model_state_dict(model, adapter_state_dict)
        if is_main_process and incompatible is not None:
            missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
            print(
                "[INFO] PEFT adapter load result: "
                f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
            )
            if unexpected_keys:
                preview = unexpected_keys[:20]
                suffix = " ..." if len(unexpected_keys) > len(preview) else ""
                print(f"[INFO] Unexpected PEFT adapter keys: {preview}{suffix}")
        return model

    latest_file = os.path.join(ckpt_dir, "latest")
    tag = None
    direct_tag_requested = False

    # Support passing a direct DeepSpeed tag directory, e.g.:
    #   .../stage1_true_one/epoch20_global_step1800
    #   .../stage2_label/best_selection_metric
    # In this case DeepSpeed expects checkpoint_dir=parent and tag=basename.
    if os.path.isdir(ckpt_dir) and not os.path.isfile(latest_file):
        maybe_tag = os.path.basename(os.path.normpath(ckpt_dir))
        parent_dir = os.path.dirname(os.path.normpath(ckpt_dir))
        has_model_states = False
        try:
            for entry in os.listdir(ckpt_dir):
                if entry.startswith("mp_rank_") and entry.endswith("_model_states.pt"):
                    has_model_states = True
                    break
        except Exception:
            has_model_states = False

        if has_model_states and os.path.isdir(parent_dir):
            ckpt_dir = parent_dir
            latest_file = os.path.join(ckpt_dir, "latest")
            tag = maybe_tag
            direct_tag_requested = True
            if is_main_process:
                print(f"[INFO] Resolve direct checkpoint tag directory to root='{ckpt_dir}', tag='{tag}'.")

    # Some old runs wrote full paths into "latest"; DeepSpeed expects only a tag.
    if not direct_tag_requested and os.path.isfile(latest_file):
        try:
            latest_raw = open(latest_file, "r", encoding="utf-8").read().strip()
            if latest_raw:
                if os.path.sep in latest_raw or os.path.isabs(latest_raw):
                    candidate_tag = os.path.basename(os.path.normpath(latest_raw))
                    candidate_dir = os.path.join(ckpt_dir, candidate_tag)
                    if os.path.isdir(candidate_dir):
                        tag = candidate_tag
                        if is_main_process:
                            print(f"[INFO] Normalize latest marker to tag='{tag}' when loading checkpoint.")
                else:
                    tag = latest_raw
        except Exception as e:
            if is_main_process:
                print(f"[WARNING] Failed to parse latest marker at {latest_file}: {e}")

    if tag is None:
        try:
            candidate_tags = []
            for entry in os.listdir(ckpt_dir):
                full = os.path.join(ckpt_dir, entry)
                if not os.path.isdir(full):
                    continue
                if entry.startswith("global_step") or (entry.startswith("epoch") and "_global_step" in entry):
                    candidate_tags.append(entry)
            if len(candidate_tags) > 0:
                candidate_tags.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
                tag = candidate_tags[0]
                if is_main_process:
                    print(f"[INFO] Fallback checkpoint tag='{tag}' inferred from checkpoint directory.")
        except Exception as e:
            if is_main_process:
                print(f"[WARNING] Failed to infer checkpoint tag from {ckpt_dir}: {e}")

    if is_main_process:
        print(
            "[INFO] Loading trainable parameters from DeepSpeed checkpoint "
            f"root='{ckpt_dir}', tag='{tag}' (frozen base parameters excluded)."
        )

    # Stage-1 is a LoRA run over a frozen backbone. Reconstructing the complete
    # fp32 ZeRO state would materialize roughly 36 GB in every distributed
    # worker, which can exhaust host RAM before DeepSpeed initialization. The
    # backbone has already been loaded from model_name_or_path, so only restore
    # parameters that were trainable in the Stage-1 checkpoint.
    original_parse_model_states = zero_to_fp32.parse_model_states

    def parse_model_states_mmap_trainable_only(files):
        states = []
        for file in files:
            checkpoint = torch.load(
                file,
                map_location=zero_to_fp32.device,
                mmap=True,
                weights_only=False,
            )
            if zero_to_fp32.BUFFER_NAMES not in checkpoint:
                raise ValueError(f"{file} is not a model state checkpoint")
            buffer_names = checkpoint[zero_to_fp32.BUFFER_NAMES]
            buffers = {
                key: value.float().clone()
                for key, value in checkpoint["module"].items()
                if key in buffer_names
            }
            states.append(
                zero_to_fp32.zero_model_state(
                    buffers=buffers,
                    param_shapes=checkpoint[zero_to_fp32.PARAM_SHAPES],
                    shared_params=[
                        [key, value]
                        for key, value in checkpoint["shared_params"].items()
                    ],
                    ds_version=checkpoint.get(zero_to_fp32.DS_VERSION),
                    frozen_param_shapes=checkpoint.get(
                        zero_to_fp32.FROZEN_PARAM_SHAPES
                    ),
                    # Frozen fragments are already present in the base model and
                    # retaining their mmap tensors is unnecessary here.
                    frozen_param_fragments=None,
                )
            )
            del checkpoint
        return states

    try:
        zero_to_fp32.parse_model_states = parse_model_states_mmap_trainable_only
        trainable_state_dict = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
            ckpt_dir,
            tag=tag,
            exclude_frozen_parameters=True,
        )
    finally:
        zero_to_fp32.parse_model_states = original_parse_model_states
    incompatible = model.load_state_dict(trainable_state_dict, strict=False)
    if is_main_process:
        missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
        print(
            "[INFO] ZeRO trainable-only load result: "
            f"loaded_keys={len(trainable_state_dict)}, "
            f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
        )
        if unexpected_keys:
            preview = unexpected_keys[:20]
            suffix = " ..." if len(unexpected_keys) > len(preview) else ""
            print(f"[INFO] Unexpected ZeRO trainable keys: {preview}{suffix}")
    del trainable_state_dict
    return model


def parse_checkpoint_epoch_spec(spec, num_train_epochs):
    raw = str(spec).strip().lower() if spec is not None else "all"
    if raw in ("", "all"):
        return set(range(1, int(num_train_epochs) + 1))
    if raw in ("none", "off"):
        return set()

    selected = set()
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        if token == "last":
            selected.add(int(num_train_epochs))
            continue
        if '-' in token:
            parts = token.split('-', 1)
            if len(parts) != 2:
                continue
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for v in range(start, end + 1):
                if 1 <= v <= int(num_train_epochs):
                    selected.add(v)
            continue
        try:
            v = int(token)
        except ValueError:
            continue
        if 1 <= v <= int(num_train_epochs):
            selected.add(v)
    return selected


def _unwrap_model_for_lightweight_save(model):
    return getattr(model, "module", model)


def _save_lightweight_adapter_snapshot(model, output_dir, tokenizer=None, config=None, args=None):
    inner_model = _unwrap_model_for_lightweight_save(model)
    trainable_state = {
        name: param.detach().cpu()
        for name, param in inner_model.named_parameters()
        if param.requires_grad
    }
    if not trainable_state:
        return False

    os.makedirs(output_dir, exist_ok=True)
    try:
        from safetensors.torch import save_file as safe_save_file

        safe_save_file(trainable_state, os.path.join(output_dir, "trainable_model.safetensors"))
    except Exception:
        torch.save(trainable_state, os.path.join(output_dir, "trainable_model.bin"))

    if hasattr(inner_model, "save_pretrained"):
        try:
            inner_model.save_pretrained(output_dir)
        except Exception as exc:
            print(f"[WARNING] Failed to save PEFT adapter metadata in lightweight snapshot: {exc}")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if config is not None:
        config.save_pretrained(output_dir)
    if args is not None and hasattr(args, "save"):
        args.save(output_dir)
    return True


def _maybe_disable_deepspeed_checkpoint_save(model, args):
    if not _is_true_flag(getattr(args, "disable_deepspeed_checkpoint_save", "False")):
        return model

    def _skip_save_checkpoint(*_args, **_kwargs):
        if globals().get("is_main_process", False):
            print("[INFO] Skip DeepSpeed save_checkpoint because --disable_deepspeed_checkpoint_save=True.")
        return None

    setattr(model, "save_checkpoint", _skip_save_checkpoint)
    return model

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=['iemocap','meld','EmoryNLP'],
    help="Datasets that need to be evaluated"
)

parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The input data dir. Should contain the source and target files for the task.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Path to the fine-tuned model checkpoint.",
)

parser.add_argument(
    "--gradient_checkpointing",
    action='store_true'
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to save trained model.",
)
parser.add_argument(
    "--run_id",
    type=str,
    default="",
    help="Run identifier shown in progress bars and logs.",
)

parser.add_argument(
    "--mode",
    type=str,
    default="sft"
)

parser.add_argument(
    "--deepspeed_config",
    type=str,
    default="./data_utils/deepspeed_config.json",
    help="Path to save trained model.",
)
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="fp16",
    choices=["fp16", "bf16", "fp32"],
    help="Model/deepspeed precision mode. Use bf16 for Ada/Ampere/Blackwell mixed GPU runs.",
)
parser.add_argument(
    "--attn_implementation",
    type=str,
    default="",
    choices=["", "eager", "sdpa", "flash_attention_2"],
    help="Attention backend passed to Hugging Face model loading.",
)

parser.add_argument(
    "--num_train_epochs",
    default=10,
    type=int,
    help="Number of training epochs.",
)
parser.add_argument(
    "--eval_every_epochs",
    default=1,
    type=int,
    help="Run evaluation every N epochs during training.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1, type=int,
    help="gradient accumulation steps",
)

parser.add_argument(
    "--warmup_ratio",
    default=0.1,
    type=float,
    help="The ratio of warmup.",
)
parser.add_argument(
    '--local_rank', 
    default=-1
)
parser.add_argument(
    "--warmup_steps",
    default=None,
    type=int
)
parser.add_argument(
    "--scheduler_num_training_steps",
    default=None,
    type=int,
    help=(
        "Optimizer-step horizon for the linear learning-rate schedule. "
        "Set this explicitly to keep the LR trajectory independent of --num_train_epochs."
    ),
)

parser.add_argument(
    "--learning_rate",
    default=3e-5,
    type=float
)
parser.add_argument(
    "--max_seq_length",
    default=256, type=int,
    help="Max output seq length",
)
parser.add_argument(
    "--max_length",
    default=2048, type=int,
    help="Max output seq length",
)
parser.add_argument(
    "--max_new_tokens",
    default=128,
    type=int,
    help="Maximum number of new tokens to generate during evaluation.",
)
parser.add_argument(
    "--repetition_penalty",
    default=1.0,
    type=float,
    help="Generation repetition penalty used during evaluation.",
)
parser.add_argument(
    '--weight_decay',
    default=0.0, type=float,
    help='weight decay when updating parameters.'
)

parser.add_argument(
    '--save_steps',
    default=1000, type=int,
)
parser.add_argument(
    "--save_checkpoint_epochs",
    type=str,
    default="all",
    help="Epoch indices to save DeepSpeed checkpoints after eval. Examples: all | none | 3 | 2,4,6 | 3-6 | last",
)
parser.add_argument(
    "--save_best_checkpoint",
    type=str,
    default="True",
    help="Whether to save the DeepSpeed best_selection_metric checkpoint. Set False for low-memory diagnostic runs.",
)
parser.add_argument(
    "--disable_deepspeed_checkpoint_save",
    type=str,
    default="False",
    help="If True, skip all DeepSpeed model.save_checkpoint calls. Useful on nodes where ZeRO checkpoint save kills the job.",
)
parser.add_argument(
    "--save_lightweight_adapter_epochs",
    type=str,
    default="none",
    help="Epoch indices to save lightweight PEFT adapter snapshots. Examples: 3,4,5 | last | none. Does not save DeepSpeed optimizer states.",
)
parser.add_argument(
    "--zero_shot",
    default=True
    # action='store_true',
)

parser.add_argument(
    "--lora",
    default=False
    # type=bool,
)
parser.add_argument(
    "--lora_dim", type=int, default=16,
)
parser.add_argument(
    "--lora_alpha", type=int, default=16,
)
parser.add_argument(
    "--lora_dropout", type=float, default=0.05,
)
parser.add_argument(
    "--lora_module_name", type=str, default='q_proj,k_proj,v_proj,query_key_value',
)
parser.add_argument(
    "--batch_size",
    default=32,
    type=int
)
parser.add_argument(
    "--eval_batch_size",
    default=4,
    type=int
)
parser.add_argument(
    "--top_k",
    default=None,
    type=int
)
parser.add_argument(
    "--num_beams",
    default=1,
    type=int
)
parser.add_argument(
    "--seed",
    default=42,
    type=int
)

parser.add_argument(
    "--top_p",
    type=float,
    default=None
)

parser.add_argument(
    "--clip_norm",
    type=float,
    default=1.0
)

parser.add_argument(
    "--temp",
    type=float,
    default=None,
    help='Temperature for model generation.'
)
parser.add_argument(
    "--do_train",
    default=False,
    # action='store_true',
)
parser.add_argument(
    "--do_eval",
    default=False,
    # action='store_true'
)
parser.add_argument(
    "--few_shot",
    type=bool,
    default=False
)

parser.add_argument(
    '--statistic_mode',
    default='True'
)
parser.add_argument(
    "--offload_optimizer",
    action='store_true'
)

parser.add_argument(
    "--emotion_prediction",
    default=False
)

parser.add_argument(
    "--data_percent",
    default=1.0,
    type=float
)

parser.add_argument(
    "--beta",
    default=1.0,
    help='hyperparameter that determining the emotion prediction weights participating in the final loss'
)

parser.add_argument(
    "--theta",
    default=1.0,
    help='hyperparameter that determining the KL divergency weights participating in the final loss'
)

parser.add_argument(
    "--speaker_loss_weight",
    default=1.0,
    type=float,
    help='Stage-1 speaker loss weight in mixed speaker+personality training.'
)

parser.add_argument(
    "--personality_loss_weight",
    default=0.5,
    type=float,
    help='Stage-1 personality loss weight in mixed speaker+personality training.'
)
parser.add_argument(
    "--task_loss_normalization",
    default="batch",
    choices=("batch", "configured_max"),
    help=(
        "Normalize task-weighted loss by weights present in the current micro-batch "
        "('batch', legacy behavior) or by the largest configured task weight "
        "('configured_max'). Use configured_max when micro-batch size is 1 so "
        "speaker/personality weights remain effective across gradient accumulation."
    ),
)
parser.add_argument(
    "--enable_personality_regression_head",
    default="False",
    help="Enable an auxiliary numeric OCEAN regression head for stage1 continuous personality samples. True/False"
)
parser.add_argument(
    "--personality_regression_hidden_dim",
    default=128,
    type=int,
    help="Hidden dimension for the stage1 personality regression head."
)
parser.add_argument(
    "--personality_regression_dropout",
    default=0.1,
    type=float,
    help="Dropout for the stage1 personality regression head."
)
parser.add_argument(
    "--personality_regression_loss_weight",
    default=1.0,
    type=float,
    help="Weight for the auxiliary numeric personality regression loss."
)
parser.add_argument(
    "--personality_regression_loss_type",
    default="mse_ccc",
    choices=["mse", "smooth_l1", "mse_ccc"],
    help="Loss type for the numeric personality regression head."
)
parser.add_argument(
    "--personality_regression_ccc_weight",
    default=0.5,
    type=float,
    help="CCC loss weight when --personality_regression_loss_type=mse_ccc."
)
parser.add_argument(
    "--personality_export_source",
    default="auto",
    choices=["generation", "regression", "auto"],
    help="Preferred source when exporting stage1 predicted personality for stage2."
)
parser.add_argument(
    "--personality_pair_mode",
    default="none",
    choices=[
        "none",
        "regression",
        "difference",
        "ranknet",
        "margin",
        "similarity_rank",
    ],
    help="Structured paired-personality objective. 'none' preserves the legacy path.",
)
parser.add_argument(
    "--personality_pair_loss_weight",
    default=0.5,
    type=float,
    help="Weight applied to the structured difference/ranking auxiliary loss.",
)
parser.add_argument(
    "--personality_pair_threshold",
    default=0.5,
    type=float,
    help="Difference threshold used for the Similar class.",
)
parser.add_argument(
    "--personality_pair_margin",
    default=0.5,
    type=float,
    help="Margin used by the paired margin-ranking objective.",
)

parser.add_argument(
    "--enable_contrastive",
    default='False',
    help='Enable supervised contrastive auxiliary loss for emotion task samples. True/False'
)
parser.add_argument(
    "--contrastive_weight",
    default=0.05,
    type=float,
    help='Weight for the supervised contrastive auxiliary loss.'
)
parser.add_argument(
    "--contrastive_temperature",
    default=0.07,
    type=float,
    help='Temperature for supervised contrastive similarity logits.'
)
parser.add_argument(
    "--contrastive_proj_dim",
    default=128,
    type=int,
    help='Projection dimension for supervised contrastive representations.'
)
parser.add_argument(
    "--contrastive_mode",
    default='emotion',
    choices=['emotion', 'personality_aware'],
    help='Contrastive mode: emotion or personality_aware.'
)
parser.add_argument(
    "--contrastive_gather_world",
    default='False',
    help='Gather emotion-task contrastive samples across all distributed ranks. True/False'
)
parser.add_argument(
    "--personality_contrastive_alpha",
    default=1.0,
    type=float,
    help='Distance decay coefficient for personality-aware positive-pair weights.'
)
parser.add_argument(
    "--enable_personality_prefix",
    default="False",
    help="Enable personality-conditioned soft prefix. True/False"
)
parser.add_argument(
    "--personality_prefix_source",
    default="auto",
    choices=["gt", "predicted", "auto"],
    help="Personality source for soft prefix: gt, predicted, or auto."
)
parser.add_argument(
    "--personality_prefix_mode",
    default="interaction",
    choices=["target", "target_interlocutor", "interaction", "attention_interaction", "shuffled"],
    help="Soft prefix interaction mode."
)
parser.add_argument(
    "--personality_prefix_len",
    default=4,
    type=int,
    help="Number of personality soft prefix tokens."
)
parser.add_argument(
    "--personality_prefix_hidden_dim",
    default=128,
    type=int,
    help="Hidden size for the personality prefix MLP."
)
parser.add_argument(
    "--personality_prefix_dropout",
    default=0.1,
    type=float,
    help="Dropout for the personality prefix MLP."
)
parser.add_argument(
    "--personality_prefix_gate",
    default="True",
    help="Use a sigmoid gate for personality prefix tokens. True/False"
)
parser.add_argument(
    "--personality_prefix_scale_init",
    default=0.1,
    type=float,
    help="Initial scalar multiplier for personality prefix embeddings."
)
parser.add_argument(
    "--enable_latent_mood_prefix",
    default="False",
    help="Enable personality-conditioned latent mood soft prefix. True/False"
)
parser.add_argument(
    "--latent_mood_source",
    default="auto",
    choices=["gt", "predicted", "auto"],
    help="Personality source for latent mood prefix: gt, predicted, or auto."
)
parser.add_argument(
    "--latent_mood_mode",
    default="interaction",
    choices=["target", "target_interlocutor", "interaction", "attention_interaction", "factorized"],
    help="Latent mood conditioning mode."
)
parser.add_argument(
    "--latent_mood_dim",
    default=128,
    type=int,
    help="Latent mood vector dimension."
)
parser.add_argument(
    "--latent_mood_prefix_len",
    default=4,
    type=int,
    help="Number of latent mood soft prefix tokens."
)
parser.add_argument(
    "--latent_mood_hidden_dim",
    default=128,
    type=int,
    help="Hidden size for latent mood personality/context encoders."
)
parser.add_argument(
    "--latent_mood_dropout",
    default=0.1,
    type=float,
    help="Dropout for latent mood prefix module."
)
parser.add_argument(
    "--latent_mood_scale_init",
    default=0.05,
    type=float,
    help="Initial scalar multiplier for latent mood prefix embeddings."
)
parser.add_argument(
    "--enable_personality_mood_prefix",
    default="False",
    help="Enable hybrid soft prefix using target/context personality plus latent mood. True/False"
)
parser.add_argument(
    "--personality_mood_source",
    default="auto",
    choices=["gt", "predicted", "auto"],
    help="Personality source for hybrid personality-mood prefix: gt, predicted, or auto."
)
parser.add_argument(
    "--personality_mood_mode",
    default="factorized",
    choices=["target_context", "interaction", "attention_interaction", "factorized", "factorized_attention"],
    help="Hybrid prefix mode for target/context personality and latent mood."
)
parser.add_argument(
    "--personality_mood_dim",
    default=128,
    type=int,
    help="Latent mood vector dimension for the hybrid prefix."
)
parser.add_argument(
    "--personality_mood_prefix_len",
    default=8,
    type=int,
    help="Number of hybrid personality-mood prefix tokens."
)
parser.add_argument(
    "--personality_mood_hidden_dim",
    default=192,
    type=int,
    help="Hidden size for hybrid target/context personality and mood encoders."
)
parser.add_argument(
    "--personality_mood_dropout",
    default=0.1,
    type=float,
    help="Dropout for the hybrid personality-mood prefix module."
)
parser.add_argument(
    "--personality_mood_scale_init",
    default=0.05,
    type=float,
    help="Initial scalar multiplier for hybrid personality-mood prefix embeddings."
)
parser.add_argument(
    "--use_mood_dynamics",
    default="False",
    help="Enable personality-conditioned mood dynamics transition prior. True/False"
)
parser.add_argument(
    "--mood_transition_lambda",
    default=0.2,
    type=float,
    help="Scale for adding mood transition prior to emotion label logits."
)
parser.add_argument(
    "--mood_dynamics_dim",
    default=128,
    type=int,
    help="Mood state dimension for the mood dynamics transition module."
)
parser.add_argument(
    "--mood_dynamics_hidden_dim",
    default=192,
    type=int,
    help="Hidden size for the mood dynamics transition module."
)
parser.add_argument(
    "--mood_dynamics_dropout",
    default=0.1,
    type=float,
    help="Dropout for the mood dynamics transition module."
)
parser.add_argument(
    "--use_speaker_mood_dynamics",
    default="False",
    help="Enable speaker-specific personality-conditioned mood dynamics. True/False"
)
parser.add_argument(
    "--speaker_mood_architecture",
    default="legacy",
    choices=["legacy", "simplified"],
    help=(
        "Speaker mood implementation. simplified uses detached soft previous "
        "predictions, residual mood update, and shared MoodFiLM projection."
    ),
)
parser.add_argument(
    "--speaker_mood_simplified_use_personality",
    default="True",
    help="Use target/interlocutor personality in simplified MoodFiLM. True/False",
)
parser.add_argument(
    "--speaker_mood_simplified_use_previous_emotion",
    default="True",
    help="Use detached soft previous-emotion state in simplified MoodFiLM. True/False",
)
parser.add_argument(
    "--speaker_mood_simplified_use_interaction",
    default="True",
    help="Use target-aware interaction summary in simplified MoodFiLM. True/False",
)
parser.add_argument(
    "--speaker_mood_simplified_use_current_evidence",
    default="True",
    help="Use prompt-only current evidence in simplified MoodFiLM. True/False",
)
parser.add_argument(
    "--speaker_mood_interaction_attention_mode",
    default="learned",
    choices=["learned", "uniform", "recency", "previous", "nearest"],
    help="Aggregation rule over interlocutor interaction tokens in simplified MoodFiLM.",
)
parser.add_argument(
    "--speaker_mood_interaction_use_personality",
    default="True",
    help="Use target/interlocutor personality features inside interaction tokens. True/False",
)
parser.add_argument(
    "--speaker_mood_interaction_use_relation",
    default="True",
    help="Use distance/count/previous-speaker relation features inside interaction tokens. True/False",
)
parser.add_argument(
    "--speaker_mood_interaction_profile_permutation",
    default="none",
    choices=["none", "reverse"],
    help="Negative control that reverses valid interlocutor profiles while preserving relations.",
)
parser.add_argument(
    "--speaker_mood_interaction_attention_temperature",
    default=1.0,
    type=float,
    help="Temperature applied to learned interlocutor attention logits.",
)
parser.add_argument(
    "--speaker_mood_interaction_min_interlocutors",
    default=1,
    type=int,
    help="Minimum number of valid interlocutors required to apply the interaction summary.",
)
parser.add_argument(
    "--speaker_mood_interaction_summary_scale",
    default=1.0,
    type=float,
    help="Residual scale applied to the aggregated interaction summary.",
)
parser.add_argument(
    "--speaker_mood_interaction_recency_bias",
    default=0.0,
    type=float,
    help="Coefficient for log inverse-distance bias in learned interaction attention.",
)
parser.add_argument(
    "--speaker_mood_interaction_previous_bias",
    default=0.0,
    type=float,
    help="Additive learned-attention bias for the immediately previous interlocutor.",
)
parser.add_argument(
    "--speaker_mood_personality_temperature",
    default=1.0,
    type=float,
    help="Anchor-only OCEAN contrast temperature around speaker_mood_personality_center.",
)
parser.add_argument(
    "--speaker_mood_personality_center",
    default=4.0,
    type=float,
    help=(
        "OCEAN midpoint used by anchor-only personality calibration. Raw "
        "1--7 values (for example 4.0) are converted to the normalized "
        "[-1, 1] space; normalized values are also accepted."
    ),
)
parser.add_argument(
    "--speaker_mood_personality_residual_scale",
    default=0.0,
    type=float,
    help="Strength of the target-personality residual added within the mood anchor.",
)
parser.add_argument(
    "--speaker_mood_source",
    default="auto",
    choices=["gt", "predicted", "auto"],
    help="Personality source for speaker-specific mood dynamics: gt, predicted, or auto."
)
parser.add_argument(
    "--speaker_mood_dim",
    default=128,
    type=int,
    help="Target speaker mood state dimension for speaker-specific mood dynamics."
)
parser.add_argument(
    "--speaker_mood_hidden_dim",
    default=192,
    type=int,
    help="Hidden size for speaker-specific mood dynamics interaction encoders."
)
parser.add_argument(
    "--speaker_mood_dropout",
    default=0.1,
    type=float,
    help="Dropout for speaker-specific mood dynamics."
)
parser.add_argument(
    "--speaker_mood_lambda",
    default=0.2,
    type=float,
    help="Scale for adding speaker mood transition prior to emotion label logits."
)
parser.add_argument(
    "--speaker_mood_logit_modulation",
    default="additive",
    choices=["additive", "film"],
    help="Emotion-label logit update: backward-compatible additive prior or residual FiLM.",
)
parser.add_argument(
    "--speaker_mood_film_condition",
    default="mood_context",
    choices=["mood", "mood_context"],
    help="Condition gamma on mood alone or on mood plus target-aware context summary.",
)
parser.add_argument(
    "--speaker_mood_film_norm",
    default="center",
    choices=["none", "center", "zscore"],
    help="Normalization applied to base emotion-label logits before FiLM gamma modulation.",
)
parser.add_argument(
    "--speaker_mood_film_gamma_scale",
    default=0.5,
    type=float,
    help="Maximum absolute residual gamma change before the global speaker_mood_lambda scale.",
)
parser.add_argument(
    "--speaker_mood_film_beta_scale",
    default=1.0,
    type=float,
    help="Scale applied to the transition-prior beta term in FiLM mode.",
)
parser.add_argument(
    "--speaker_mood_film_use_gamma",
    default="True",
    help="Enable the multiplicative gamma term in FiLM mode. True/False",
)
parser.add_argument(
    "--speaker_mood_film_use_beta",
    default="True",
    help="Enable the transition-prior beta term in FiLM mode. True/False",
)
parser.add_argument(
    "--speaker_mood_film_beta_gate",
    default="none",
    choices=["none", "entropy"],
    help="Optionally gate only the FiLM beta term using detached normalized entropy of base emotion logits.",
)
parser.add_argument(
    "--speaker_mood_film_gate_temperature",
    default=1.0,
    type=float,
    help="Softmax temperature used to compute the entropy-based FiLM beta gate.",
)
parser.add_argument(
    "--speaker_mood_film_gate_power",
    default=1.0,
    type=float,
    help="Exponent applied to normalized entropy before gating FiLM beta.",
)
parser.add_argument(
    "--speaker_mood_film_gate_min",
    default=0.0,
    type=float,
    help="Minimum FiLM beta gate; final gate is min + (1-min) * uncertainty^power.",
)
parser.add_argument(
    "--speaker_mood_film_eps",
    default=1e-6,
    type=float,
    help="Numerical epsilon for z-score normalization of candidate emotion logits.",
)
parser.add_argument(
    "--use_speaker_mood_prior_loss",
    default="False",
    help="Add CE supervision on speaker mood transition prior. True/False"
)
parser.add_argument(
    "--speaker_mood_prior_loss_weight",
    default=0.0,
    type=float,
    help="Weight for speaker mood transition-prior auxiliary CE loss."
)
parser.add_argument(
    "--speaker_mood_prior_alpha",
    default=None,
    type=float,
    help="Alpha for uncertainty-aware speaker mood transition prior loss. Defaults to speaker_mood_prior_loss_weight."
)
parser.add_argument(
    "--use_speaker_mood_dynamics_consistency_loss",
    default="False",
    help="Enable changed-only speaker mood dynamics consistency loss. True/False"
)
parser.add_argument(
    "--speaker_mood_dynamics_loss_weight",
    default=None,
    type=float,
    help="Beta for unified speaker mood dynamics consistency loss. Defaults to legacy shift/alignment weights."
)
parser.add_argument(
    "--speaker_mood_dynamics_margin",
    default=None,
    type=float,
    help="Magnitude margin for changed same-speaker mood shifts inside L_dyn. Defaults to speaker_mood_shift_margin."
)
parser.add_argument(
    "--speaker_mood_dynamics_magnitude_weight",
    default=None,
    type=float,
    help="Rho for the optional magnitude margin term inside L_dyn. Defaults to 0 for the new dynamics loss."
)
parser.add_argument(
    "--speaker_mood_sync_state_cache",
    default="True",
    help="All-gather speaker mood state cache across distributed ranks. True/False"
)
parser.add_argument(
    "--speaker_mood_recurrence_mode",
    default="none",
    choices=["none", "detached"],
    help="Use the detached previous same-speaker mood as the GRU hidden state.",
)
parser.add_argument(
    "--speaker_mood_dialogue_ordered",
    default="False",
    help="Keep complete dialogues on one rank and process their turns chronologically. True/False",
)
parser.add_argument(
    "--speaker_mood_recurrent_prev_emotion",
    default="none",
    choices=["none", "metadata"],
    help="Legacy recurrent previous-emotion switch; use --speaker_mood_prev_emotion_source for new runs.",
)
parser.add_argument(
    "--speaker_mood_prev_emotion_source",
    default="legacy",
    choices=["legacy", "gold", "predicted", "soft", "none"],
    help=(
        "Previous same-speaker emotion fed to the mood head. legacy preserves old "
        "commands; predicted uses hard cached predictions, while soft is reserved "
        "for the simplified architecture."
    ),
)
parser.add_argument(
    "--speaker_mood_ordered_num_workers",
    default=0,
    type=int,
    help="DataLoader workers for dialogue-ordered mode. Zero is safest for large multimodal collators.",
)
parser.add_argument(
    "--use_speaker_mood_shift_loss",
    default="False",
    help="Add speaker mood shift regularization using previous emotion state. True/False"
)
parser.add_argument(
    "--speaker_mood_shift_loss_weight",
    default=0.0,
    type=float,
    help="Weight for speaker mood shift auxiliary loss."
)
parser.add_argument(
    "--speaker_mood_shift_margin",
    default=0.25,
    type=float,
    help="Minimum normalized mood shift when the current emotion differs from previous emotion."
)
parser.add_argument(
    "--speaker_mood_shift_same_weight",
    default=1.0,
    type=float,
    help="Multiplier for same-emotion mood stability term."
)
parser.add_argument(
    "--speaker_mood_shift_change_weight",
    default=1.0,
    type=float,
    help="Multiplier for changed-emotion margin term."
)
parser.add_argument(
    "--use_speaker_mood_alignment_loss",
    default="False",
    help="Align target-aware interpersonal input with target speaker mood shift. True/False"
)
parser.add_argument(
    "--speaker_mood_alignment_loss_weight",
    default=0.0,
    type=float,
    help="Weight for speaker mood context-interaction alignment loss."
)
parser.add_argument(
    "--speaker_mood_alignment_eta",
    default=None,
    type=float,
    help="Eta multiplier for L_align inside L_dyn. Defaults to a compatibility mapping from legacy alignment weight."
)
parser.add_argument(
    "--speaker_mood_alignment_changed_only",
    default="True",
    help="Apply speaker mood alignment loss only when current emotion differs from previous emotion. True/False"
)
parser.add_argument(
    "--speaker_mood_alignment_min_shift",
    default=0.0,
    type=float,
    help="Minimum normalized mood shift required before applying alignment loss."
)
parser.add_argument(
    "--save_speaker_mood_analysis",
    default="False",
    help="Save speaker mood attention/state/transition traces during evaluation. True/False"
)
parser.add_argument(
    "--speaker_mood_personality_noise_std",
    default=0.0,
    type=float,
    help=(
        "Evaluation-time Gaussian noise standard deviation on the original "
        "1-7 OCEAN scale. The normalized input is perturbed by std/3 and "
        "clipped to [-1, 1]."
    ),
)
parser.add_argument(
    "--speaker_mood_personality_noise_seed",
    default=2026,
    type=int,
    help="Seed for deterministic profile-level personality noise.",
)
parser.add_argument(
    "--speaker_mood_personality_noise_scale",
    default="raw",
    choices=["raw", "normalized"],
    help=(
        "Interpret personality noise std on the original 1-7 OCEAN scale "
        "('raw', divided by 3) or directly on the normalized [-1, 1] model "
        "input ('normalized')."
    ),
)
parser.add_argument(
    "--speaker_mood_personality_corruption",
    default="gaussian",
    choices=["none", "gaussian", "random"],
    help=(
        "Evaluation-time personality corruption. Random samples a stable "
        "per-profile U(1, 7) personality vector."
    ),
)
parser.add_argument(
    "--experience_bank_mode",
    default='none',
    choices=['none', 'student_prompt', 'opcd_teacher'],
    help='Experience bank mode used during data construction/training metadata.'
)
parser.add_argument(
    "--opcd_teacher_include_gold_context",
    default='False',
    help='Whether OPCD teacher metadata includes GT emotion/personality/derived shift. True/False'
)
parser.add_argument(
    "--generate_inference_reasoning",
    default='False',
    help='Generate diagnostic natural-language reasoning text during eval. True/False'
)
parser.add_argument(
    "--inference_reasoning_max_samples",
    default=64,
    type=int,
    help='Maximum eval samples per rank for diagnostic reasoning generation; <=0 means all.'
)
parser.add_argument(
    "--inference_reasoning_max_new_tokens",
    default=120,
    type=int,
    help='Maximum tokens for diagnostic inference reasoning generation.'
)

parser.add_argument(
    "--audio_enable",
    default='False',
    help='Enable text+audio multimodal prefix embedding. True/False'
)
parser.add_argument(
    "--audio_feature_pkl",
    default=None,
    type=str,
    help='Path to precomputed audio feature pkl'
)
parser.add_argument(
    "--audio_embedding_source",
    default="precomputed_pkl",
    type=str,
    help=(
        "Audio source: precomputed_pkl (Qwen native mel cache) | "
        "precomputed_tokens (compact Qwen audio tower tokens) | "
        "qwen_runtime_extract"
    )
)
parser.add_argument(
    "--audio_iemocap_root",
    default=None,
    type=str,
    help='IEMOCAP root directory for qwen_runtime_extract'
)
parser.add_argument(
    "--audio_iemocap_pkl",
    default=None,
    type=str,
    help='IEMOCAP metadata pkl path for qwen_runtime_extract'
)
parser.add_argument(
    "--audio_qwen_processor_path",
    default=None,
    type=str,
    help='Qwen processor/model path for qwen_runtime_extract. Defaults to model_name_or_path.'
)
parser.add_argument(
    "--audio_feature_dim",
    default=None,
    type=int,
    help='Unused for Qwen native audio; kept for argument compatibility.'
)
parser.add_argument(
    "--audio_adapter_hidden_dim",
    default=256,
    type=int,
    help='Bottleneck dimension for the compressed audio residual adapter.'
)
parser.add_argument(
    "--audio_adapter_dropout",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--audio_adapter_scale_init",
    default=0.1,
    type=float,
    help=(
        "Initial residual correction scale for compressed audio tokens. "
        "This does not attenuate the raw audio feature itself."
    ),
)
parser.add_argument(
    "--audio_adapter_output_scale_init",
    default=1.0,
    type=float,
    help=(
        "Initial global output scale for compressed audio tokens. "
        "Values below 1.0 keep text more dominant while preserving audio evidence."
    ),
)
parser.add_argument(
    "--audio_adapter_mode",
    default="residual",
    choices=("residual", "identity"),
    help=(
        "Use the trainable residual adapter or insert precomputed audio tokens "
        "unchanged for an adapter ablation."
    ),
)
parser.add_argument(
    "--audio_context_window",
    default=-1,
    type=int,
    help='Audio context window size; -1 means all history up to current utterance.'
)
parser.add_argument(
    "--audio_gru_hidden_size",
    default=256,
    type=int,
    help='Bi-GRU hidden size for dialogue-level audio context.'
)
parser.add_argument(
    "--audio_gru_layers",
    default=1,
    type=int,
    help='Bi-GRU layers for dialogue-level audio context.'
)
parser.add_argument(
    "--audio_gru_dropout",
    default=0.0,
    type=float,
    help='Bi-GRU dropout for dialogue-level audio context.'
)
parser.add_argument(
    "--audio_input_mode",
    default="text_audio",
    type=str,
    help='Multimodal input mode: text_audio | audio_only | text_only'
)
parser.add_argument(
    "--audio_token_placement",
    default="aligned",
    choices=("aligned", "tail"),
    help=(
        "Place direct audio items after their matching utterances (aligned) or "
        "collect the same ordered audio items after the complete text prompt (tail)."
    ),
)
parser.add_argument(
    "--text_enable",
    default="True",
    type=str,
    help='Whether to keep lexical text content in multimodal prompts.'
)
parser.add_argument(
    "--video_enable",
    default="False",
    type=str,
    help='Whether to enable Qwen2.5-Omni video inputs.'
)
parser.add_argument(
    "--video_embedding_source",
    default="qwen_runtime_extract",
    type=str,
    help='Video source: qwen_runtime_extract | precomputed_tokens.'
)
parser.add_argument(
    "--video_feature_pkl",
    default=None,
    type=str,
    help='Path to qwen_visual_temporal_tokens_v1 cache.'
)
parser.add_argument(
    "--video_adapter_hidden_dim",
    default=256,
    type=int,
    help='Bottleneck dimension for the trainable precomputed visual adapter.'
)
parser.add_argument(
    "--video_adapter_dropout",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--video_adapter_scale_init",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--video_adapter_output_scale_init",
    default=0.05,
    type=float,
)
parser.add_argument(
    "--enable_stage1_multimodal_fusion",
    default="False",
    type=str,
    help=(
        "Enable Stage1-only text/audio/visual attention fusion before the Qwen language model. "
        "Supports native or compressed Qwen audio inputs with precomputed visual tokens."
    ),
)
parser.add_argument(
    "--stage1_fusion_dim",
    default=256,
    type=int,
)
parser.add_argument(
    "--stage1_fusion_heads",
    default=4,
    type=int,
)
parser.add_argument(
    "--stage1_fusion_prefix_len",
    default=2,
    type=int,
)
parser.add_argument(
    "--stage1_fusion_dropout",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--stage1_fusion_scale_init",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--stage1_fusion_input_mode",
    default="direct",
    choices=["direct", "prefix_only"],
    help=(
        "How Stage1 multimodal fusion uses raw modality tokens. "
        "'direct' preserves the previous behavior: modality features replace "
        "placeholder embeddings and a fusion prefix is prepended. "
        "'prefix_only' uses modality features only to build the fusion prefix, "
        "then removes audio/visual placeholder positions from the LM sequence."
    ),
)
parser.add_argument(
    "--stage1_fusion_query_mode",
    default="generic",
    choices=["generic", "trait_gated"],
    help=(
        "Fusion query layout. 'generic' uses learned generic prefix queries; "
        "'trait_gated' uses one query per OCEAN trait and learns trait-specific "
        "text/audio/visual gates."
    ),
)
parser.add_argument(
    "--stage1_fusion_aux_gate_init",
    default=0.05,
    type=float,
    help=(
        "Initial audio/visual gate probability for trait_gated fusion. "
        "Text gate initializes near 0.95 so modalities start as residual evidence."
    ),
)
parser.add_argument(
    "--text_feature_pkl",
    default=None,
    type=str,
    help="Path to qwen_text_utterance_features_v1 cache for Stage1 context prefix.",
)
parser.add_argument('--utterance_feature_text', default='False', type=str,
                    help='Use one pooled frozen-Qwen text feature token per visible utterance.')
parser.add_argument('--utterance_feature_audio', default='False', type=str,
                    help='Use one mean-pooled cached Qwen audio token per visible utterance.')
parser.add_argument('--utterance_feature_text_dim', default=3584, type=int)
parser.add_argument('--utterance_feature_audio_dim', default=3584, type=int)
parser.add_argument('--utterance_feature_history_window', default=-1, type=int,
                    help='ERC history turns represented by feature tokens; target is added automatically. -1 follows existing media selection.')
parser.add_argument('--stage1_speaker_audio_only', default='False', type=str,
                    help='Keep direct audio only for the Stage1 single-utterance speaker task.')
parser.add_argument('--utterance_direct_audio_once', default='False', type=str,
                    help='Avoid duplicating the target utterance direct audio after the ERC label instruction.')
parser.add_argument(
    "--enable_stage1_context_prefix",
    default="False",
    type=str,
    help="Enable Causal-ERC-style text/audio/video context soft prefix.",
)
parser.add_argument(
    "--stage1_context_prefix_mode",
    default="ta_cross",
    choices=["text_only", "ta_cross", "tav_cross", "visual_side", "tav_prefix_only"],
)
parser.add_argument(
    "--stage1_context_arch",
    default="gru_cross",
    choices=["gru_cross", "pool", "gru_pool", "cross_only"],
    help="Context prefix architecture: original GRU+cross attention, pooled MLP, GRU pooled, or cross attention without GRU.",
)
parser.add_argument(
    "--stage1_context_visual_shuffle",
    default="False",
    type=str,
    help="Shuffle visual context features across the batch as a visual sanity check. True/False",
)
parser.add_argument(
    "--audio_embedding_usage",
    default="direct",
    choices=["direct", "context", "both"],
    help="Use compressed audio tokens as direct LM tokens, context prefix evidence, or both.",
)
parser.add_argument(
    "--video_embedding_usage",
    default="direct",
    choices=["none", "direct", "context", "both"],
    help="Use visual tokens as direct LM tokens, context prefix evidence, both, or disabled.",
)
parser.add_argument(
    "--stage1_context_prefix_len",
    default=2,
    type=int,
)
parser.add_argument(
    "--stage1_context_feature_dim",
    default=3584,
    type=int,
)
parser.add_argument(
    "--stage1_context_fusion_dim",
    default=256,
    type=int,
)
parser.add_argument(
    "--stage1_context_gru_hidden",
    default=128,
    type=int,
)
parser.add_argument(
    "--stage1_context_heads",
    default=4,
    type=int,
)
parser.add_argument(
    "--stage1_context_dropout",
    default=0.1,
    type=float,
)
parser.add_argument(
    "--stage1_context_scale_init",
    default=0.05,
    type=float,
)
parser.add_argument(
    "--stage1_context_audio_gate_init",
    default=0.10,
    type=float,
)
parser.add_argument(
    "--stage1_context_visual_gate_init",
    default=0.001,
    type=float,
)
parser.add_argument(
    "--stage1_context_consistency_weight",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--personality_media_scope",
    default="legacy",
    type=str,
    help=(
        "Direct media attached to conversation-level personality samples: "
        "legacy (preserve old behavior), target (target-speaker turns only), "
        "or full (all conversation turns)."
    ),
)
parser.add_argument(
    "--video_manifest_jsonl",
    default=None,
    type=str,
    help='IEMOCAP utterance-level video manifest jsonl.'
)
parser.add_argument(
    "--video_fps",
    default=2.0,
    type=float,
    help='Video sampling fps passed to qwen_omni_utils.'
)
parser.add_argument(
    "--video_resized_height",
    default=336,
    type=int,
    help='Video resized height passed to qwen_omni_utils. Use 0 for auto.'
)
parser.add_argument(
    "--video_resized_width",
    default=0,
    type=int,
    help='Video resized width passed to qwen_omni_utils. Use 0 for auto/aspect-ratio resize.'
)
parser.add_argument(
    "--video_use_audio_in_video",
    default="False",
    type=str,
    help='Whether qwen_omni_utils should read audio tracks from video files.'
)

args = parser.parse_args()
run_prefix = f"[{args.run_id}]" if args.run_id else ""
do_sample = args.top_k is not None or args.top_p is not None or args.num_beams > 1 or args.temp is not None
'''
args.top_k is not None (top_k is a parameter that limits the number of possible tokens to consider at each step)
args.top_p is not None (top_p is a parameter that limits the cumulative probability of the next token)
args.num_beams > 1(num_beams is a parameter that controls the number of beams to use for beam search)
args.temp is not None (temp is a parameter that controls the temperature of softmax distribution used for sampling)

# if any of those conditions are true, then do_sample will be set to True.
# Otherwise, it will be set to False.
# This variable is likely being used later in the code to determine whether or not to use sampling when generating text.
'''

if args.do_train == 'True':
    args.do_train = True
else:
    args.do_train = False

if args.do_eval == 'True': 
    args.do_eval = True
else:
    args.do_eval = False

if args.emotion_prediction == 'True':
    args.emotion_prediction = True
else:
    args.emotion_prediction = False

if args.lora == 'True':
    args.lora = True
else:
    args.lora = False

args.enable_contrastive = _is_true_flag(args.enable_contrastive)
args.contrastive_gather_world = _is_true_flag(args.contrastive_gather_world)
args.contrastive_mode = str(args.contrastive_mode).strip().lower()
if args.contrastive_mode not in ("emotion", "personality_aware"):
    raise ValueError("Invalid --contrastive_mode. Use: emotion | personality_aware")
args.enable_personality_prefix = _is_true_flag(args.enable_personality_prefix)
args.personality_prefix_source = str(args.personality_prefix_source or "auto").strip().lower()
if args.personality_prefix_source not in ("gt", "predicted", "auto"):
    raise ValueError("Invalid --personality_prefix_source. Use: gt | predicted | auto")
args.personality_prefix_mode = str(args.personality_prefix_mode or "interaction").strip().lower()
if args.personality_prefix_mode not in ("target", "target_interlocutor", "interaction", "attention_interaction", "shuffled"):
    raise ValueError(
        "Invalid --personality_prefix_mode. Use: target | target_interlocutor | "
        "interaction | attention_interaction | shuffled"
    )
args.personality_prefix_gate = _is_true_flag(args.personality_prefix_gate)
if args.personality_prefix_len < 0:
    raise ValueError("--personality_prefix_len must be non-negative.")
args.enable_latent_mood_prefix = _is_true_flag(args.enable_latent_mood_prefix)
args.enable_personality_mood_prefix = _is_true_flag(args.enable_personality_mood_prefix)
args.use_mood_dynamics = _is_true_flag(args.use_mood_dynamics)
args.use_speaker_mood_dynamics = _is_true_flag(args.use_speaker_mood_dynamics)
args.use_speaker_mood_prior_loss = _is_true_flag(args.use_speaker_mood_prior_loss)
args.speaker_mood_film_use_gamma = _is_true_flag(args.speaker_mood_film_use_gamma)
args.speaker_mood_film_use_beta = _is_true_flag(args.speaker_mood_film_use_beta)
args.use_speaker_mood_dynamics_consistency_loss = _is_true_flag(args.use_speaker_mood_dynamics_consistency_loss)
args.use_speaker_mood_shift_loss = _is_true_flag(args.use_speaker_mood_shift_loss)
args.use_speaker_mood_alignment_loss = _is_true_flag(args.use_speaker_mood_alignment_loss)
args.speaker_mood_alignment_changed_only = _is_true_flag(args.speaker_mood_alignment_changed_only)
args.speaker_mood_sync_state_cache = _is_true_flag(args.speaker_mood_sync_state_cache)
args.speaker_mood_dialogue_ordered = _is_true_flag(args.speaker_mood_dialogue_ordered)
args.save_speaker_mood_analysis = _is_true_flag(args.save_speaker_mood_analysis)
if args.speaker_mood_personality_noise_std < 0.0:
    raise ValueError("--speaker_mood_personality_noise_std must be non-negative.")
args.speaker_mood_personality_noise_scale = str(
    args.speaker_mood_personality_noise_scale or "raw"
).strip().lower()
args.speaker_mood_personality_corruption = str(
    args.speaker_mood_personality_corruption or "gaussian"
).strip().lower()
if (
    args.speaker_mood_personality_noise_std > 0.0
    or args.speaker_mood_personality_corruption == "random"
) and args.do_train:
    raise ValueError(
        "Personality noise is evaluation-only; set --do_train False."
    )
args.speaker_mood_source = str(args.speaker_mood_source or "auto").strip().lower()
if args.speaker_mood_source not in ("gt", "predicted", "auto"):
    raise ValueError("Invalid --speaker_mood_source. Use: gt | predicted | auto")
args.speaker_mood_architecture = str(
    args.speaker_mood_architecture or "legacy"
).strip().lower()
if args.speaker_mood_architecture not in ("legacy", "simplified"):
    raise ValueError(
        "Invalid --speaker_mood_architecture. Use: legacy | simplified"
    )
args.speaker_mood_recurrence_mode = str(args.speaker_mood_recurrence_mode).strip().lower()
if args.speaker_mood_recurrence_mode not in ("none", "detached"):
    raise ValueError("Invalid --speaker_mood_recurrence_mode. Use: none | detached")
args.speaker_mood_recurrent_prev_emotion = str(
    args.speaker_mood_recurrent_prev_emotion
).strip().lower()
if args.speaker_mood_recurrent_prev_emotion not in ("none", "metadata"):
    raise ValueError(
        "Invalid --speaker_mood_recurrent_prev_emotion. Use: none | metadata"
    )
args.speaker_mood_prev_emotion_source = str(
    args.speaker_mood_prev_emotion_source
).strip().lower()
if args.speaker_mood_prev_emotion_source not in (
    "legacy",
    "gold",
    "predicted",
    "soft",
    "none",
):
    raise ValueError(
        "Invalid --speaker_mood_prev_emotion_source. "
        "Use: legacy | gold | predicted | soft | none"
    )
if args.speaker_mood_ordered_num_workers < 0:
    raise ValueError("--speaker_mood_ordered_num_workers must be non-negative.")
if args.speaker_mood_recurrence_mode != "none" and not args.use_speaker_mood_dynamics:
    raise ValueError(
        "--speaker_mood_recurrence_mode requires --use_speaker_mood_dynamics True."
    )
if args.speaker_mood_architecture == "simplified" and args.use_speaker_mood_dynamics:
    if args.speaker_mood_recurrence_mode != "none":
        raise ValueError(
            "Simplified MoodFiLM does not accept previous mood recurrence; use "
            "--speaker_mood_recurrence_mode none."
        )
    args.speaker_mood_dialogue_ordered = True
    args.speaker_mood_sync_state_cache = False
    args.speaker_mood_prev_emotion_source = "soft"
    args.speaker_mood_logit_modulation = "film"
    args.speaker_mood_film_condition = "mood"
    args.speaker_mood_film_norm = "center"
    args.speaker_mood_film_beta_gate = "none"
elif args.speaker_mood_dialogue_ordered:
    if args.speaker_mood_recurrence_mode != "detached":
        raise ValueError(
            "--speaker_mood_dialogue_ordered True requires "
            "--speaker_mood_recurrence_mode detached."
        )
    args.speaker_mood_sync_state_cache = False
if (
    args.speaker_mood_prev_emotion_source == "soft"
    and args.speaker_mood_architecture != "simplified"
):
    raise ValueError(
        "--speaker_mood_prev_emotion_source soft requires "
        "--speaker_mood_architecture simplified."
    )
if (
    args.speaker_mood_prev_emotion_source == "predicted"
    and not args.speaker_mood_dialogue_ordered
):
    raise ValueError(
        "--speaker_mood_prev_emotion_source predicted requires "
        "--speaker_mood_dialogue_ordered True for causal previous predictions."
    )
args.speaker_mood_logit_modulation = str(args.speaker_mood_logit_modulation).strip().lower()
if args.speaker_mood_logit_modulation not in ("additive", "film"):
    raise ValueError("Invalid --speaker_mood_logit_modulation. Use: additive | film")
args.speaker_mood_film_condition = str(args.speaker_mood_film_condition).strip().lower()
if args.speaker_mood_film_condition not in ("mood", "mood_context"):
    raise ValueError("Invalid --speaker_mood_film_condition. Use: mood | mood_context")
args.speaker_mood_film_norm = str(args.speaker_mood_film_norm).strip().lower()
if args.speaker_mood_film_norm not in ("none", "center", "zscore"):
    raise ValueError("Invalid --speaker_mood_film_norm. Use: none | center | zscore")
args.speaker_mood_film_beta_gate = str(args.speaker_mood_film_beta_gate).strip().lower()
if args.speaker_mood_film_beta_gate not in ("none", "entropy"):
    raise ValueError("Invalid --speaker_mood_film_beta_gate. Use: none | entropy")
if args.speaker_mood_film_gamma_scale < 0:
    raise ValueError("--speaker_mood_film_gamma_scale must be non-negative.")
if args.speaker_mood_film_beta_scale < 0:
    raise ValueError("--speaker_mood_film_beta_scale must be non-negative.")
if args.speaker_mood_film_gate_temperature <= 0:
    raise ValueError("--speaker_mood_film_gate_temperature must be positive.")
if args.speaker_mood_film_gate_power <= 0:
    raise ValueError("--speaker_mood_film_gate_power must be positive.")
if not 0.0 <= args.speaker_mood_film_gate_min <= 1.0:
    raise ValueError("--speaker_mood_film_gate_min must be in [0, 1].")
if args.speaker_mood_film_eps <= 0:
    raise ValueError("--speaker_mood_film_eps must be positive.")
if args.speaker_mood_dim <= 0:
    raise ValueError("--speaker_mood_dim must be positive.")
if args.speaker_mood_hidden_dim <= 0:
    raise ValueError("--speaker_mood_hidden_dim must be positive.")
if args.speaker_mood_lambda < 0:
    raise ValueError("--speaker_mood_lambda must be non-negative.")
if args.speaker_mood_prior_loss_weight < 0:
    raise ValueError("--speaker_mood_prior_loss_weight must be non-negative.")
if args.speaker_mood_prior_alpha is not None and args.speaker_mood_prior_alpha < 0:
    raise ValueError("--speaker_mood_prior_alpha must be non-negative.")
if args.speaker_mood_dynamics_loss_weight is not None and args.speaker_mood_dynamics_loss_weight < 0:
    raise ValueError("--speaker_mood_dynamics_loss_weight must be non-negative.")
if args.speaker_mood_personality_temperature <= 0.0:
    raise ValueError("--speaker_mood_personality_temperature must be positive.")
if args.speaker_mood_personality_residual_scale < 0.0:
    raise ValueError("--speaker_mood_personality_residual_scale must be non-negative.")
if args.speaker_mood_dynamics_margin is not None and args.speaker_mood_dynamics_margin < 0:
    raise ValueError("--speaker_mood_dynamics_margin must be non-negative.")
if args.speaker_mood_dynamics_magnitude_weight is not None and args.speaker_mood_dynamics_magnitude_weight < 0:
    raise ValueError("--speaker_mood_dynamics_magnitude_weight must be non-negative.")
if args.speaker_mood_shift_loss_weight < 0:
    raise ValueError("--speaker_mood_shift_loss_weight must be non-negative.")
if args.speaker_mood_shift_margin < 0:
    raise ValueError("--speaker_mood_shift_margin must be non-negative.")
if args.speaker_mood_shift_same_weight < 0:
    raise ValueError("--speaker_mood_shift_same_weight must be non-negative.")
if args.speaker_mood_shift_change_weight < 0:
    raise ValueError("--speaker_mood_shift_change_weight must be non-negative.")
if args.speaker_mood_alignment_loss_weight < 0:
    raise ValueError("--speaker_mood_alignment_loss_weight must be non-negative.")
if args.speaker_mood_alignment_eta is not None and args.speaker_mood_alignment_eta < 0:
    raise ValueError("--speaker_mood_alignment_eta must be non-negative.")
if args.speaker_mood_alignment_min_shift < 0:
    raise ValueError("--speaker_mood_alignment_min_shift must be non-negative.")
args.personality_mood_source = str(args.personality_mood_source or "auto").strip().lower()
if args.personality_mood_source not in ("gt", "predicted", "auto"):
    raise ValueError("Invalid --personality_mood_source. Use: gt | predicted | auto")
args.personality_mood_mode = str(args.personality_mood_mode or "factorized").strip().lower()
if args.personality_mood_mode not in (
    "target_context",
    "interaction",
    "attention_interaction",
    "factorized",
    "factorized_attention",
):
    raise ValueError(
        "Invalid --personality_mood_mode. Use: target_context | interaction | "
        "attention_interaction | factorized | factorized_attention"
    )
if args.personality_mood_prefix_len < 0:
    raise ValueError("--personality_mood_prefix_len must be non-negative.")
if args.mood_dynamics_dim <= 0:
    raise ValueError("--mood_dynamics_dim must be positive.")
if args.mood_dynamics_hidden_dim <= 0:
    raise ValueError("--mood_dynamics_hidden_dim must be positive.")
enabled_prefix_count = sum(
    int(flag)
    for flag in (
        args.enable_personality_prefix,
        args.enable_latent_mood_prefix,
        args.enable_personality_mood_prefix,
    )
)
if enabled_prefix_count > 1:
    raise ValueError(
        "Use only one of --enable_personality_prefix, "
        "--enable_latent_mood_prefix, or --enable_personality_mood_prefix."
    )
args.experience_bank_mode = str(args.experience_bank_mode or 'none').strip().lower()
if args.experience_bank_mode not in ("none", "student_prompt", "opcd_teacher"):
    raise ValueError("Invalid --experience_bank_mode. Use: none | student_prompt | opcd_teacher")
args.opcd_teacher_include_gold_context = _is_true_flag(args.opcd_teacher_include_gold_context)
args.generate_inference_reasoning = _is_true_flag(args.generate_inference_reasoning)
args.save_best_checkpoint = _is_true_flag(args.save_best_checkpoint)
args.disable_deepspeed_checkpoint_save = _is_true_flag(args.disable_deepspeed_checkpoint_save)
args.personality_pair_mode = str(
    args.personality_pair_mode or "none"
).strip().lower()
if args.personality_pair_loss_weight < 0:
    raise ValueError("--personality_pair_loss_weight must be non-negative.")
if args.personality_pair_threshold < 0:
    raise ValueError("--personality_pair_threshold must be non-negative.")
if args.personality_pair_margin < 0:
    raise ValueError("--personality_pair_margin must be non-negative.")

if args.audio_enable == 'True':
    args.audio_enable = True
else:
    args.audio_enable = False

args.text_enable = _is_true_flag(args.text_enable)
args.video_enable = _is_true_flag(args.video_enable)
args.video_use_audio_in_video = _is_true_flag(args.video_use_audio_in_video)
args.video_embedding_source = str(args.video_embedding_source).strip().lower()
if args.video_embedding_source not in ("qwen_runtime_extract", "precomputed_tokens"):
    raise ValueError(
        "Invalid --video_embedding_source. Use: qwen_runtime_extract | precomputed_tokens"
    )
args.enable_stage1_multimodal_fusion = _is_true_flag(args.enable_stage1_multimodal_fusion)
args.stage1_fusion_input_mode = str(args.stage1_fusion_input_mode).strip().lower()
if args.stage1_fusion_input_mode not in ("direct", "prefix_only"):
    raise ValueError("Invalid --stage1_fusion_input_mode. Use: direct | prefix_only")
args.stage1_fusion_query_mode = str(args.stage1_fusion_query_mode).strip().lower()
if args.stage1_fusion_query_mode not in ("generic", "trait_gated"):
    raise ValueError("Invalid --stage1_fusion_query_mode. Use: generic | trait_gated")
args.stage1_fusion_aux_gate_init = float(args.stage1_fusion_aux_gate_init)
if not (0.0 < args.stage1_fusion_aux_gate_init < 1.0):
    raise ValueError("--stage1_fusion_aux_gate_init must be in (0, 1).")
args.enable_stage1_context_prefix = _is_true_flag(args.enable_stage1_context_prefix)
args.utterance_feature_text = _is_true_flag(args.utterance_feature_text)
args.utterance_feature_audio = _is_true_flag(args.utterance_feature_audio)
args.stage1_speaker_audio_only = _is_true_flag(args.stage1_speaker_audio_only)
args.utterance_direct_audio_once = _is_true_flag(args.utterance_direct_audio_once)
if args.utterance_feature_history_window < -1:
    raise ValueError('--utterance_feature_history_window must be -1 or non-negative')
if args.utterance_feature_text or args.utterance_feature_audio:
    if args.dataset != 'iemocap':
        raise ValueError('Utterance feature prefix is currently validated only for IEMOCAP')
    if args.utterance_feature_text and not args.text_feature_pkl:
        raise ValueError('--utterance_feature_text requires --text_feature_pkl')
    if args.utterance_feature_audio and (
        not args.audio_enable or args.audio_embedding_source != 'precomputed_tokens'
    ):
        raise ValueError('Utterance audio prefix requires --audio_enable True and --audio_embedding_source precomputed_tokens')
    if args.enable_stage1_context_prefix or args.enable_stage1_multimodal_fusion:
        raise ValueError('Utterance feature prefix cannot be combined with old Stage1 fusion/context prefix')
args.stage1_context_visual_shuffle = _is_true_flag(args.stage1_context_visual_shuffle)
args.audio_embedding_usage = str(args.audio_embedding_usage or "direct").strip().lower()
args.video_embedding_usage = str(args.video_embedding_usage or "direct").strip().lower()
if args.audio_embedding_usage not in ("direct", "context", "both"):
    raise ValueError("Invalid --audio_embedding_usage. Use: direct | context | both")
if args.video_embedding_usage not in ("none", "direct", "context", "both"):
    raise ValueError("Invalid --video_embedding_usage. Use: none | direct | context | both")
args.stage1_context_prefix_mode = str(args.stage1_context_prefix_mode or "ta_cross").strip().lower()
if args.stage1_context_prefix_mode not in ("text_only", "ta_cross", "tav_cross", "visual_side", "tav_prefix_only"):
    raise ValueError("Invalid --stage1_context_prefix_mode.")
args.stage1_context_arch = str(args.stage1_context_arch or "gru_cross").strip().lower()
if args.stage1_context_arch not in ("gru_cross", "pool", "gru_pool", "cross_only"):
    raise ValueError("Invalid --stage1_context_arch.")
if args.stage1_context_prefix_len <= 0:
    raise ValueError("--stage1_context_prefix_len must be positive.")
if args.stage1_context_feature_dim <= 0:
    raise ValueError("--stage1_context_feature_dim must be positive.")
if args.stage1_context_fusion_dim <= 0:
    raise ValueError("--stage1_context_fusion_dim must be positive.")
if args.stage1_context_gru_hidden <= 0:
    raise ValueError("--stage1_context_gru_hidden must be positive.")
if args.stage1_context_heads <= 0:
    raise ValueError("--stage1_context_heads must be positive.")
if (2 * int(args.stage1_context_gru_hidden)) % int(args.stage1_context_heads) != 0:
    raise ValueError("2 * --stage1_context_gru_hidden must be divisible by --stage1_context_heads.")
if (
    args.stage1_context_arch in ("pool", "cross_only")
    and int(args.stage1_context_fusion_dim) != 2 * int(args.stage1_context_gru_hidden)
):
    raise ValueError("--stage1_context_fusion_dim must equal 2 * --stage1_context_gru_hidden for pool/cross_only.")
if not (0.0 < float(args.stage1_context_audio_gate_init) < 1.0):
    raise ValueError("--stage1_context_audio_gate_init must be in (0, 1).")
if not (0.0 < float(args.stage1_context_visual_gate_init) < 1.0):
    raise ValueError("--stage1_context_visual_gate_init must be in (0, 1).")
if float(args.stage1_context_consistency_weight) < 0:
    raise ValueError("--stage1_context_consistency_weight must be non-negative.")
if args.enable_stage1_context_prefix:
    if args.emotion_prediction:
        raise ValueError("--enable_stage1_context_prefix is restricted to Stage1.")
    if args.text_feature_pkl is None or not os.path.exists(str(args.text_feature_pkl)):
        raise FileNotFoundError(f"Precomputed text feature pkl not found: {args.text_feature_pkl}")
    if args.audio_embedding_usage in ("context", "both") and not args.audio_enable:
        raise ValueError("audio_embedding_usage=context/both requires --audio_enable True.")
    if args.video_embedding_usage in ("context", "both") and not args.video_enable:
        raise ValueError("video_embedding_usage=context/both requires --video_enable True.")
    if args.stage1_context_prefix_mode in ("tav_cross", "visual_side", "tav_prefix_only") and args.video_embedding_usage not in ("context", "both"):
        raise ValueError(
            f"{args.stage1_context_prefix_mode} requires --video_embedding_usage context or both."
        )
    if args.stage1_context_prefix_mode == "ta_cross" and args.audio_embedding_usage not in ("context", "both"):
        raise ValueError("ta_cross requires --audio_embedding_usage context or both.")
if args.enable_stage1_multimodal_fusion:
    if args.emotion_prediction:
        raise ValueError("--enable_stage1_multimodal_fusion is restricted to Stage1.")
    if not args.text_enable:
        raise ValueError(
            "Stage1 multimodal fusion requires --text_enable True."
        )
    if not (args.audio_enable or args.video_enable):
        raise ValueError(
            "Stage1 multimodal fusion requires at least one of "
            "--audio_enable True or --video_enable True."
        )
    if args.video_enable and args.video_embedding_source != "precomputed_tokens":
        raise ValueError("Stage1 multimodal fusion requires --video_embedding_source precomputed_tokens.")
    if args.stage1_fusion_dim <= 0 or args.stage1_fusion_heads <= 0:
        raise ValueError("Stage1 fusion dimension and head count must be positive.")
    if args.stage1_fusion_dim % args.stage1_fusion_heads != 0:
        raise ValueError("--stage1_fusion_dim must be divisible by --stage1_fusion_heads.")
    if args.stage1_fusion_prefix_len <= 0:
        raise ValueError("--stage1_fusion_prefix_len must be positive.")
args.personality_media_scope = str(args.personality_media_scope).strip().lower()
if args.personality_media_scope not in ("legacy", "target", "full"):
    raise ValueError(
        "Invalid --personality_media_scope. Use: legacy | target | full"
    )

args.audio_input_mode = str(args.audio_input_mode).strip().lower()
if args.audio_input_mode not in ("text_audio", "audio_only", "text_only"):
    raise ValueError("Invalid --audio_input_mode. Use: text_audio | audio_only | text_only")
args.audio_token_placement = str(args.audio_token_placement).strip().lower()
if args.audio_token_placement not in ("aligned", "tail"):
    raise ValueError("Invalid --audio_token_placement. Use: aligned | tail")

args.audio_embedding_source = str(args.audio_embedding_source).strip().lower()
if args.audio_embedding_source not in (
    "precomputed_pkl",
    "precomputed_tokens",
    "qwen_runtime_extract",
):
    raise ValueError(
        "Invalid --audio_embedding_source. Use: precomputed_pkl | "
        "precomputed_tokens | qwen_runtime_extract"
    )

args.mixed_precision = str(args.mixed_precision).strip().lower()
if args.mixed_precision not in ("fp16", "bf16", "fp32"):
    raise ValueError("Invalid --mixed_precision. Use: fp16 | bf16 | fp32")
args.attn_implementation = str(args.attn_implementation or "").strip().lower()

# eval_result_path = args.eval_result_path if args.eval_result_path is not None else args.output_dir
# os.makedirs(eval_result_path, exist_ok=True)
# pdb.set_trace()
model_args = {
    "dataset": args.dataset,
    "model_name_or_path": args.model_name_or_path,
    "checkpoint_dir": args.checkpoint_dir,
    "data_dir": args.data_dir,
    "max_seq_length": args.max_seq_length,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "eval_every_epochs": args.eval_every_epochs,
    "save_steps": args.save_steps,
    "save_checkpoint_epochs": args.save_checkpoint_epochs,
    "save_best_checkpoint": args.save_best_checkpoint,
    "disable_deepspeed_checkpoint_save": args.disable_deepspeed_checkpoint_save,
    "save_lightweight_adapter_epochs": args.save_lightweight_adapter_epochs,
    "output_dir": args.output_dir,
    "max_length": args.max_length,
    "max_new_tokens": args.max_new_tokens,
    "repetition_penalty": args.repetition_penalty,
    "warmup_ratio": args.warmup_ratio,
    "warmup_steps": args.warmup_steps,
    "scheduler_num_training_steps": args.scheduler_num_training_steps,
    "weight_decay": args.weight_decay,
    'data_dir': args.data_dir,
    "lora": args.lora,
    "lora_dim": args.lora_dim,
    "lora_dropout": args.lora_dropout,
    "lora_alpha": args.lora_alpha,
    "lora_module_name": args.lora_module_name,
    "num_beams": args.num_beams,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "do_sample": do_sample,
    "seed": args.seed,
    "do_train": args.do_train,
    "do_eval": args.do_eval,
    "offload_optimizer": args.offload_optimizer,
    "deepspeed_config": args.deepspeed_config,
    "mixed_precision": args.mixed_precision,
    "attn_implementation": args.attn_implementation or None,
    "zero_shot": args.zero_shot,
    "statistic_mode": args.statistic_mode,
    "mode": args.mode,
    "emotion_prediction": args.emotion_prediction,
    "beta": args.beta,
    "theta": args.theta,
    "speaker_loss_weight": args.speaker_loss_weight,
    "personality_loss_weight": args.personality_loss_weight,
    "task_loss_normalization": args.task_loss_normalization,
    "enable_personality_regression_head": args.enable_personality_regression_head,
    "personality_regression_hidden_dim": args.personality_regression_hidden_dim,
    "personality_regression_dropout": args.personality_regression_dropout,
    "personality_regression_loss_weight": args.personality_regression_loss_weight,
    "personality_regression_loss_type": args.personality_regression_loss_type,
    "personality_regression_ccc_weight": args.personality_regression_ccc_weight,
    "personality_export_source": args.personality_export_source,
    "personality_media_scope": args.personality_media_scope,
    "personality_pair_mode": args.personality_pair_mode,
    "personality_pair_loss_weight": args.personality_pair_loss_weight,
    "personality_pair_threshold": args.personality_pair_threshold,
    "personality_pair_margin": args.personality_pair_margin,
    "enable_contrastive": args.enable_contrastive,
    "contrastive_weight": args.contrastive_weight,
    "contrastive_temperature": args.contrastive_temperature,
    "contrastive_proj_dim": args.contrastive_proj_dim,
    "contrastive_mode": args.contrastive_mode,
    "contrastive_gather_world": args.contrastive_gather_world,
    "personality_contrastive_alpha": args.personality_contrastive_alpha,
    "enable_personality_prefix": args.enable_personality_prefix,
    "personality_prefix_source": args.personality_prefix_source,
    "personality_prefix_mode": args.personality_prefix_mode,
    "personality_prefix_len": args.personality_prefix_len,
    "personality_prefix_hidden_dim": args.personality_prefix_hidden_dim,
    "personality_prefix_dropout": args.personality_prefix_dropout,
    "personality_prefix_gate": args.personality_prefix_gate,
    "personality_prefix_scale_init": args.personality_prefix_scale_init,
    "enable_personality_mood_prefix": args.enable_personality_mood_prefix,
    "personality_mood_source": args.personality_mood_source,
    "personality_mood_mode": args.personality_mood_mode,
    "personality_mood_dim": args.personality_mood_dim,
    "personality_mood_prefix_len": args.personality_mood_prefix_len,
    "personality_mood_hidden_dim": args.personality_mood_hidden_dim,
    "personality_mood_dropout": args.personality_mood_dropout,
    "personality_mood_scale_init": args.personality_mood_scale_init,
    "use_mood_dynamics": args.use_mood_dynamics,
    "mood_transition_lambda": args.mood_transition_lambda,
    "mood_dynamics_dim": args.mood_dynamics_dim,
    "mood_dynamics_hidden_dim": args.mood_dynamics_hidden_dim,
    "mood_dynamics_dropout": args.mood_dynamics_dropout,
    "use_speaker_mood_dynamics": args.use_speaker_mood_dynamics,
    "speaker_mood_architecture": args.speaker_mood_architecture,
    "speaker_mood_simplified_use_personality": args.speaker_mood_simplified_use_personality,
    "speaker_mood_simplified_use_previous_emotion": args.speaker_mood_simplified_use_previous_emotion,
    "speaker_mood_simplified_use_interaction": args.speaker_mood_simplified_use_interaction,
    "speaker_mood_simplified_use_current_evidence": args.speaker_mood_simplified_use_current_evidence,
    "speaker_mood_interaction_attention_mode": args.speaker_mood_interaction_attention_mode,
    "speaker_mood_interaction_use_personality": args.speaker_mood_interaction_use_personality,
    "speaker_mood_interaction_use_relation": args.speaker_mood_interaction_use_relation,
    "speaker_mood_interaction_profile_permutation": args.speaker_mood_interaction_profile_permutation,
    "speaker_mood_interaction_attention_temperature": args.speaker_mood_interaction_attention_temperature,
    "speaker_mood_interaction_min_interlocutors": args.speaker_mood_interaction_min_interlocutors,
    "speaker_mood_interaction_summary_scale": args.speaker_mood_interaction_summary_scale,
    "speaker_mood_interaction_recency_bias": args.speaker_mood_interaction_recency_bias,
    "speaker_mood_interaction_previous_bias": args.speaker_mood_interaction_previous_bias,
    "speaker_mood_personality_temperature": args.speaker_mood_personality_temperature,
    "speaker_mood_personality_center": args.speaker_mood_personality_center,
    "speaker_mood_personality_residual_scale": args.speaker_mood_personality_residual_scale,
    "speaker_mood_source": args.speaker_mood_source,
    "speaker_mood_dim": args.speaker_mood_dim,
    "speaker_mood_hidden_dim": args.speaker_mood_hidden_dim,
    "speaker_mood_dropout": args.speaker_mood_dropout,
    "speaker_mood_lambda": args.speaker_mood_lambda,
    "speaker_mood_logit_modulation": args.speaker_mood_logit_modulation,
    "speaker_mood_film_condition": args.speaker_mood_film_condition,
    "speaker_mood_film_norm": args.speaker_mood_film_norm,
    "speaker_mood_film_gamma_scale": args.speaker_mood_film_gamma_scale,
    "speaker_mood_film_beta_scale": args.speaker_mood_film_beta_scale,
    "speaker_mood_film_use_gamma": args.speaker_mood_film_use_gamma,
    "speaker_mood_film_use_beta": args.speaker_mood_film_use_beta,
    "speaker_mood_film_beta_gate": args.speaker_mood_film_beta_gate,
    "speaker_mood_film_gate_temperature": args.speaker_mood_film_gate_temperature,
    "speaker_mood_film_gate_power": args.speaker_mood_film_gate_power,
    "speaker_mood_film_gate_min": args.speaker_mood_film_gate_min,
    "speaker_mood_film_eps": args.speaker_mood_film_eps,
    "use_speaker_mood_prior_loss": args.use_speaker_mood_prior_loss,
    "speaker_mood_prior_loss_weight": args.speaker_mood_prior_loss_weight,
    "speaker_mood_prior_alpha": args.speaker_mood_prior_alpha,
    "use_speaker_mood_dynamics_consistency_loss": args.use_speaker_mood_dynamics_consistency_loss,
    "speaker_mood_dynamics_loss_weight": args.speaker_mood_dynamics_loss_weight,
    "speaker_mood_dynamics_margin": args.speaker_mood_dynamics_margin,
    "speaker_mood_dynamics_magnitude_weight": args.speaker_mood_dynamics_magnitude_weight,
    "speaker_mood_sync_state_cache": args.speaker_mood_sync_state_cache,
    "speaker_mood_recurrence_mode": args.speaker_mood_recurrence_mode,
    "speaker_mood_dialogue_ordered": args.speaker_mood_dialogue_ordered,
    "speaker_mood_recurrent_prev_emotion": args.speaker_mood_recurrent_prev_emotion,
    "speaker_mood_prev_emotion_source": args.speaker_mood_prev_emotion_source,
    "speaker_mood_ordered_num_workers": args.speaker_mood_ordered_num_workers,
    "use_speaker_mood_shift_loss": args.use_speaker_mood_shift_loss,
    "speaker_mood_shift_loss_weight": args.speaker_mood_shift_loss_weight,
    "speaker_mood_shift_margin": args.speaker_mood_shift_margin,
    "speaker_mood_shift_same_weight": args.speaker_mood_shift_same_weight,
    "speaker_mood_shift_change_weight": args.speaker_mood_shift_change_weight,
    "use_speaker_mood_alignment_loss": args.use_speaker_mood_alignment_loss,
    "speaker_mood_alignment_loss_weight": args.speaker_mood_alignment_loss_weight,
    "speaker_mood_alignment_eta": args.speaker_mood_alignment_eta,
    "speaker_mood_alignment_changed_only": args.speaker_mood_alignment_changed_only,
    "speaker_mood_alignment_min_shift": args.speaker_mood_alignment_min_shift,
    "save_speaker_mood_analysis": args.save_speaker_mood_analysis,
    "speaker_mood_personality_noise_std": args.speaker_mood_personality_noise_std,
    "speaker_mood_personality_noise_seed": args.speaker_mood_personality_noise_seed,
    "speaker_mood_personality_noise_scale": args.speaker_mood_personality_noise_scale,
    "speaker_mood_personality_corruption": args.speaker_mood_personality_corruption,
    "experience_bank_mode": args.experience_bank_mode,
    "opcd_teacher_include_gold_context": args.opcd_teacher_include_gold_context,
    "generate_inference_reasoning": args.generate_inference_reasoning,
    "inference_reasoning_max_samples": args.inference_reasoning_max_samples,
    "inference_reasoning_max_new_tokens": args.inference_reasoning_max_new_tokens,
    "audio_enable": args.audio_enable,
    "audio_embedding_source": args.audio_embedding_source,
    "audio_token_placement": args.audio_token_placement,
    "audio_feature_pkl": args.audio_feature_pkl,
    "audio_iemocap_root": args.audio_iemocap_root,
    "audio_iemocap_pkl": args.audio_iemocap_pkl,
    "audio_qwen_processor_path": args.audio_qwen_processor_path,
    "audio_feature_dim": args.audio_feature_dim,
    "audio_adapter_hidden_dim": args.audio_adapter_hidden_dim,
    "audio_adapter_dropout": args.audio_adapter_dropout,
    "audio_adapter_scale_init": args.audio_adapter_scale_init,
    "audio_adapter_output_scale_init": args.audio_adapter_output_scale_init,
    "audio_embedding_usage": args.audio_embedding_usage,
    "audio_context_window": args.audio_context_window,
    "audio_gru_hidden_size": args.audio_gru_hidden_size,
    "audio_gru_layers": args.audio_gru_layers,
    "audio_gru_dropout": args.audio_gru_dropout,
    "audio_input_mode": args.audio_input_mode,
    "text_enable": args.text_enable,
    "video_enable": args.video_enable,
    "video_embedding_source": args.video_embedding_source,
    "video_feature_pkl": args.video_feature_pkl,
    "video_adapter_hidden_dim": args.video_adapter_hidden_dim,
    "video_adapter_dropout": args.video_adapter_dropout,
    "video_adapter_scale_init": args.video_adapter_scale_init,
    "video_adapter_output_scale_init": args.video_adapter_output_scale_init,
    "video_embedding_usage": args.video_embedding_usage,
    "enable_stage1_multimodal_fusion": args.enable_stage1_multimodal_fusion,
    "stage1_fusion_dim": args.stage1_fusion_dim,
    "stage1_fusion_heads": args.stage1_fusion_heads,
    "stage1_fusion_prefix_len": args.stage1_fusion_prefix_len,
    "stage1_fusion_dropout": args.stage1_fusion_dropout,
    "stage1_fusion_scale_init": args.stage1_fusion_scale_init,
    "stage1_fusion_input_mode": args.stage1_fusion_input_mode,
    "stage1_fusion_query_mode": args.stage1_fusion_query_mode,
    "stage1_fusion_aux_gate_init": args.stage1_fusion_aux_gate_init,
    "text_feature_pkl": args.text_feature_pkl,
    'utterance_feature_text': args.utterance_feature_text,
    'utterance_feature_audio': args.utterance_feature_audio,
    'utterance_feature_text_dim': args.utterance_feature_text_dim,
    'utterance_feature_audio_dim': args.utterance_feature_audio_dim,
    'utterance_feature_history_window': args.utterance_feature_history_window,
    'utterance_feature_order': 'text,audio',
    'stage1_speaker_audio_only': args.stage1_speaker_audio_only,
    'utterance_direct_audio_once': args.utterance_direct_audio_once,
    "enable_stage1_context_prefix": args.enable_stage1_context_prefix,
    "stage1_context_prefix_mode": args.stage1_context_prefix_mode,
    "stage1_context_arch": args.stage1_context_arch,
    "stage1_context_visual_shuffle": args.stage1_context_visual_shuffle,
    "stage1_context_prefix_len": args.stage1_context_prefix_len,
    "stage1_context_feature_dim": args.stage1_context_feature_dim,
    "stage1_context_fusion_dim": args.stage1_context_fusion_dim,
    "stage1_context_gru_hidden": args.stage1_context_gru_hidden,
    "stage1_context_heads": args.stage1_context_heads,
    "stage1_context_dropout": args.stage1_context_dropout,
    "stage1_context_scale_init": args.stage1_context_scale_init,
    "stage1_context_audio_gate_init": args.stage1_context_audio_gate_init,
    "stage1_context_visual_gate_init": args.stage1_context_visual_gate_init,
    "stage1_context_consistency_weight": args.stage1_context_consistency_weight,
    "video_manifest_jsonl": args.video_manifest_jsonl,
    "video_fps": args.video_fps,
    "video_resized_height": args.video_resized_height,
    "video_resized_width": args.video_resized_width,
    "video_use_audio_in_video": args.video_use_audio_in_video,
    "gradient_checkpointing": args.gradient_checkpointing,
    "data_percent": args.data_percent
}
args = ModelArgs()
# pdb.set_trace()

args.update(model_args)
# pdb.set_trace()

if args.personality_media_scope not in ("legacy", "target", "full"):
    raise ValueError(
        "Invalid effective personality_media_scope after ModelArgs rebuild: "
        f"{args.personality_media_scope}"
    )
if is_main_process:
    print(
        "[INFO] Effective personality_media_scope="
        f"{args.personality_media_scope}"
    )

print(args)
if is_main_process:
    print(
        f"[Contrastive] enabled={args.enable_contrastive}, mode={args.contrastive_mode}, "
        f"weight={args.contrastive_weight}, temperature={args.contrastive_temperature}, "
        f"proj_dim={args.contrastive_proj_dim}, gather_world={args.contrastive_gather_world}, "
        f"personality_alpha={args.personality_contrastive_alpha}"
    )
    print(
        f"[PersonalityPrefix] enabled={args.enable_personality_prefix}, source={args.personality_prefix_source}, "
        f"mode={args.personality_prefix_mode}, prefix_len={args.personality_prefix_len}, "
        f"hidden_dim={args.personality_prefix_hidden_dim}, gate={args.personality_prefix_gate}, "
        f"scale_init={args.personality_prefix_scale_init}"
    )
    print(
        f"[PersonalityMoodPrefix] enabled={args.enable_personality_mood_prefix}, source={args.personality_mood_source}, "
        f"mode={args.personality_mood_mode}, prefix_len={args.personality_mood_prefix_len}, "
        f"hidden_dim={args.personality_mood_hidden_dim}, mood_dim={args.personality_mood_dim}, "
        f"scale_init={args.personality_mood_scale_init}"
    )
    print(
        f"[MoodDynamics] enabled={args.use_mood_dynamics}, lambda={args.mood_transition_lambda}, "
        f"mood_dim={args.mood_dynamics_dim}, hidden_dim={args.mood_dynamics_hidden_dim}, "
        f"dropout={args.mood_dynamics_dropout}"
    )
    print(
        f"[ExperienceBank] mode={args.experience_bank_mode}, "
        f"opcd_gold_context={args.opcd_teacher_include_gold_context}, "
        f"generate_inference_reasoning={args.generate_inference_reasoning}, "
        f"reasoning_max_samples={args.inference_reasoning_max_samples}"
    )

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lora_dim,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_module_name.split(","),
    bias='none',
)
with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
    deepspeed_config = json.load(f)
deepspeed_config["train_batch_size"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
if deepspeed_config["zero_optimization"]["stage"] == 3:
    deepspeed_config["zero_optimization"]['mics_shard_size'] = world_size

def _model_load_dtype(mixed_precision):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def _apply_deepspeed_precision_config(ds_config, mixed_precision):
    ds_config.setdefault("fp16", {})["enabled"] = mixed_precision == "fp16"
    ds_config.setdefault("bfloat16", {})["enabled"] = mixed_precision == "bf16"


MODEL_TORCH_DTYPE = _model_load_dtype(args.mixed_precision)
_apply_deepspeed_precision_config(deepspeed_config, args.mixed_precision)

def getOptimizerGroup(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    
    return optimizer_grouped_parameters

def _get_pred_input_dict(batch):
    # print(batch["input_ids"].shape,batch["labels"].shape,batch["attention_mask"].shape)
    batch['input_ids'] = batch['input_ids'].unsqueeze(1)
    batch['labels'] = batch['labels'].unsqueeze(1)
    batch['attention_mask'] = batch['attention_mask'].unsqueeze(1)
    batch['type_token_ids'] = batch['type_token_ids'].unsqueeze(1)
    input_ids, labels, attention_mask, type_token_ids = batch["input_ids"][0], \
        batch["labels"][0], batch["attention_mask"][0], batch["type_token_ids"][0]
    
    pred_input_ids, pred_labels, pred_attention_mask, pred_type_token_ids = batch["input_ids"][1], \
        batch["labels"][1], batch["attention_mask"][1], batch["type_token_ids"][1]
     
    
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device) 
    },{
        "input_ids": pred_input_ids.to(device),
        "labels": pred_labels.to(device),
        "attention_mask": pred_attention_mask.to(device) 
    }

def _get_input_dict(batch):
    input_ids, labels, attention_mask, type_token_ids = batch["input_ids"], \
        batch["labels"], batch["attention_mask"], batch["type_token_ids"]

    model_inputs = {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device)
    }

    if "audio_features" in batch and "audio_mask" in batch:
        model_inputs["audio_features"] = batch["audio_features"].to(device)
        model_inputs["audio_mask"] = batch["audio_mask"].to(device)

    if "input_features" in batch:
        model_inputs["input_features"] = batch["input_features"].to(device)
    if "feature_attention_mask" in batch:
        model_inputs["feature_attention_mask"] = batch["feature_attention_mask"].to(device)
    if "audio_feature_lengths" in batch:
        model_inputs["audio_feature_lengths"] = batch["audio_feature_lengths"].to(device)

    excluded_keys = {
        "input_ids",
        "labels",
        "attention_mask",
        "type_token_ids",
        "token_type_ids",
        "contrastive_label_ids",
        "contrastive_task_mask",
        "contrastive_personality",
        "contrastive_personality_mask",
        "personality_regression_targets",
        "personality_regression_mask",
        "personality_pair_targets",
        "personality_pair_mask",
        "personality_pair_query_positions",
    }
    for key, value in batch.items():
        if key in excluded_keys or key in model_inputs:
            continue
        if torch.is_tensor(value):
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value

    return model_inputs


def _move_batch_to_device(batch, target_device):
    if hasattr(batch, "to") and callable(getattr(batch, "to")):
        return batch.to(target_device)
    if isinstance(batch, dict):
        return {
            key: value.to(target_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
    raise TypeError(f"Unsupported batch type for device move: {type(batch)}")


def _get_generation_input_dict(batch):
    excluded_keys = {
        "labels",
        "type_token_ids",
        "token_type_ids",
        "contrastive_label_ids",
        "contrastive_task_mask",
        "contrastive_personality",
        "contrastive_personality_mask",
        "personality_regression_targets",
        "personality_regression_mask",
        "personality_pair_targets",
        "personality_pair_mask",
        "personality_pair_query_positions",
    }
    return {
        key: value
        for key, value in batch.items()
        if key not in excluded_keys
    }


def _infer_audio_feature_dim_from_pkl(audio_feature_pkl):
    if audio_feature_pkl is None or not os.path.exists(audio_feature_pkl):
        return None
    try:
        payload = pickle.load(open(audio_feature_pkl, 'rb'))
    except Exception:
        return None

    if isinstance(payload, dict):
        feature_dim = payload.get('feature_dim', None)
        if isinstance(feature_dim, int) and feature_dim > 0:
            return feature_dim

        sample_features = payload.get('sample_features', None)
        if isinstance(sample_features, dict) and len(sample_features) > 0:
            first = next(iter(sample_features.values()))
            if isinstance(first, (list, tuple)):
                return len(first)
    return None


def _distributed_rank_world():
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))


def _speaker_mood_local_micro_batch_size(args, num_replicas):
    denominator = int(num_replicas) * int(args.gradient_accumulation_steps)
    if denominator <= 0 or int(args.batch_size) % denominator != 0:
        raise ValueError(
            "--batch_size must be divisible by WORLD_SIZE * "
            "--gradient_accumulation_steps for dialogue-ordered mode."
        )
    local_batch_size = int(args.batch_size) // denominator
    if local_batch_size <= 0:
        raise ValueError("Dialogue-ordered local micro-batch size must be positive.")
    return local_batch_size


def _build_dialogue_ordered_loader(
    dataset,
    collator,
    local_batch_size,
    args,
    shuffle_dialogues,
):
    rank, num_replicas = _distributed_rank_world()
    batch_sampler = DialogueOrderedBatchSampler(
        dataset=dataset,
        local_batch_size=int(local_batch_size),
        num_replicas=num_replicas,
        rank=rank,
        shuffle_dialogues=bool(shuffle_dialogues),
        seed=int(args.seed),
    )
    ordered_dataset = OrderedDatasetView(dataset)
    dataloader = DataLoader(
        ordered_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=int(getattr(args, "speaker_mood_ordered_num_workers", 0)),
    )
    return batch_sampler, dataloader


def _build_eval_loader(dev_dataset, dev_collator, eval_batch_size, args=None):
    if args is not None and _speaker_mood_dialogue_ordered_enabled(args):
        eval_sampler, eval_dataloader = _build_dialogue_ordered_loader(
            dataset=dev_dataset,
            collator=dev_collator,
            local_batch_size=int(eval_batch_size),
            args=args,
            shuffle_dialogues=False,
        )
        return eval_sampler, eval_dataloader, None

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        eval_sampler = DistributedSampler(
            dev_dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        eval_sampler = SequentialSampler(dev_dataset)

    eval_dataloader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        sampler=eval_sampler,
        collate_fn=dev_collator,
        num_workers=0,
    )
    local_indices = list(iter(eval_sampler))
    return eval_sampler, eval_dataloader, local_indices


def _eval_batch_dataset_indices(eval_batch, fallback_indices, fallback_cursor):
    batch_size = int(eval_batch["input_ids"].size(0))
    dataset_indices = eval_batch.get("speaker_mood_dataset_index")
    padding_mask = eval_batch.get("speaker_mood_order_padding")
    if dataset_indices is not None:
        indices = dataset_indices.detach().cpu().view(-1).tolist()
        if padding_mask is None:
            padding = [False] * len(indices)
        else:
            padding = padding_mask.detach().cpu().bool().view(-1).tolist()
        return [int(index) for index in indices], padding, fallback_cursor

    if fallback_indices is None:
        raise ValueError("Evaluation batch is missing both dataset indices and fallback indices.")
    indices = fallback_indices[fallback_cursor:fallback_cursor + batch_size]
    return [int(index) for index in indices], [False] * batch_size, fallback_cursor + batch_size


def _strip_prompt_from_generated_ids(input_ids, output_ids):
    """Best-effort removal of the echoed prompt from decoder generations."""
    if input_ids is None or output_ids is None:
        return output_ids

    input_list = input_ids.tolist()
    output_list = output_ids.tolist()
    if len(output_list) <= len(input_list):
        return output_ids

    if output_list[:len(input_list)] == input_list:
        return output_ids[len(input_list):]

    # Multimodal or templated prompts can insert extra prefix tokens before the text prompt.
    # Search for the last exact occurrence of the prompt and drop everything up to its end.
    max_start = len(output_list) - len(input_list)
    for start in range(max_start, -1, -1):
        if output_list[start:start + len(input_list)] == input_list:
            return output_ids[start + len(input_list):]

    return output_ids


def _gather_eval_records(local_records, output_dir=None, gather_tag="eval"):
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        rank = dist.get_rank()
        if output_dir:
            # Avoid gather_object here. With one full Qwen process per rank, its
            # pickle/tensor staging can push rank 0 over the host-memory limit.
            shard_dir = os.path.join(output_dir, ".eval_gather")
            os.makedirs(shard_dir, exist_ok=True)
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(gather_tag))
            shard_path = os.path.join(shard_dir, f"{safe_tag}.rank{rank}.jsonl")
            temp_path = f"{shard_path}.tmp.{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as handle:
                for record in local_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temp_path, shard_path)
            local_records.clear()
            dist.barrier()

            merged = []
            if rank == 0:
                for shard_rank in range(world_size):
                    rank_path = os.path.join(
                        shard_dir,
                        f"{safe_tag}.rank{shard_rank}.jsonl",
                    )
                    with open(rank_path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                merged.append(json.loads(line))

            dist.barrier()
            try:
                os.remove(shard_path)
            except FileNotFoundError:
                pass
            if rank == 0:
                try:
                    os.rmdir(shard_dir)
                except OSError:
                    pass
            return merged

        if rank == 0:
            gathered = [None for _ in range(world_size)]
            dist.gather_object(local_records, object_gather_list=gathered, dst=0)
            return [
                record
                for part in gathered
                if part
                for record in part
            ]
        dist.gather_object(local_records, object_gather_list=None, dst=0)
        return []
    return local_records


def _dedupe_records_by_index(records):
    dedup = {}
    for rec in records:
        idx = int(rec["index"])
        if idx not in dedup:
            dedup[idx] = rec
    return [dedup[i] for i in sorted(dedup.keys())]


def _is_missing_eval_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value == ''
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    except (TypeError, ValueError):
        return False
    return False


def _add_eval_debug_fields(record, sample_id=None, debug_target_utterance=None, utterance_id=None):
    if not _is_missing_eval_value(sample_id):
        record["sample_id"] = sample_id
    if not _is_missing_eval_value(debug_target_utterance):
        record["debug_target_utterance"] = debug_target_utterance
    if not _is_missing_eval_value(utterance_id):
        record["utterance_id"] = utterance_id
    return record


def _add_generation_trace_fields(record, raw_answer, target_text=None, emotional_label_dict=None, parse_info=None):
    raw_answer = str(raw_answer).strip().rstrip()
    record["output_raw"] = raw_answer
    pred_reasoning = _extract_tagged_section(raw_answer, 'reasoning')
    gold_reasoning = _extract_tagged_section(target_text, 'reasoning') if target_text is not None else None
    pred_states = _extract_structured_reasoning_states(raw_answer)
    gold_states = _extract_structured_reasoning_states(target_text) if target_text is not None else {}
    pred_personality = _extract_personality_chain(raw_answer)
    gold_personality = _extract_personality_chain(target_text) if target_text is not None else {}
    record["pred_reasoning"] = pred_reasoning or ""
    record["gold_reasoning"] = gold_reasoning or ""
    record["pred_states"] = pred_states
    record["gold_states"] = gold_states
    record["pred_personality"] = pred_personality
    record["gold_personality"] = gold_personality
    if pred_reasoning:
        record["reasoning"] = pred_reasoning
    record["parse_status"] = parse_info.get("parse_status") if parse_info is not None else None
    record["raw_char_length"] = len(raw_answer)
    record["raw_word_count"] = len(re.findall(r'\S+', raw_answer))

    if emotional_label_dict is not None:
        labels = set(emotional_label_dict.keys())
        if record.get("output") in labels:
            record["pred_label"] = record.get("output")
        if target_text is not None:
            gold_label = _canonical_target_text(target_text).strip().lower()
            if gold_label in labels:
                record["gold_label"] = gold_label
    return record


def _safe_dist_barrier(tag="finalize"):
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.barrier()
    except Exception as e:
        if is_main_process:
            print(f"[WARN] dist.barrier failed at {tag}: {e}")

## prepare model
if "chatglm" in args.model_name_or_path:
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=MODEL_TORCH_DTYPE,
    )
elif "t5" in args.model_name_or_path:
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    try:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    deepspeed_config["bfloat16"]["enabled"] = False
    deepspeed_config["fp16"]["enabled"] = False
    args.model_type = "encoder-decoder"
    deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] = "none"
else:
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model_path_lower = args.model_name_or_path.lower()
    config_model_type = str(getattr(config, "model_type", "")).lower()
    if "bloom" in model_path_lower or "falcon" in model_path_lower:
        ## for bloom, falcon
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=MODEL_TORCH_DTYPE,
        )
    elif config_model_type == "qwen2_5_omni":
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
        qwen_load_kwargs = {}
        if args.attn_implementation:
            qwen_load_kwargs["attn_implementation"] = args.attn_implementation
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=MODEL_TORCH_DTYPE,
            **qwen_load_kwargs,
        )
    else:
        ## for llama, vicuna, belle and newer llama-family tokenizers (e.g., Llama 3.x)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=MODEL_TORCH_DTYPE,
        )

if tokenizer.pad_token is None:
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.pad_token
additional_special_tokens = []
if args.audio_enable and args.audio_embedding_source == "precomputed_tokens":
    additional_special_tokens.append(AUDIO_UTTERANCE_TOKEN)
if args.video_enable and args.video_embedding_source == "precomputed_tokens":
    additional_special_tokens.append(VISUAL_UTTERANCE_TOKEN)
if str(args.personality_pair_mode).strip().lower() != "none":
    additional_special_tokens.extend(
        [
            PERSONALITY_SPEAKER_0_TOKEN,
            PERSONALITY_SPEAKER_1_TOKEN,
        ]
    )
if additional_special_tokens:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

if getattr(tokenizer, "pad_token_id", None) is None:
    raise ValueError("Tokenizer pad_token_id is None after fallback setup.")

# Set model.config.pad_token_id to avoid warnings during generation
if hasattr(model, "config") and hasattr(model.config, "pad_token_id"):
    model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if str(args.personality_pair_mode).strip().lower() != "none":
        input_embeddings = model.get_input_embeddings().weight
        query_initializers = {
            PERSONALITY_SPEAKER_0_TOKEN: "Speaker 0 personality",
            PERSONALITY_SPEAKER_1_TOKEN: "Speaker 1 personality",
        }
        with torch.no_grad():
            for query_token, initializer_text in query_initializers.items():
                query_id = tokenizer.convert_tokens_to_ids(query_token)
                source_ids = tokenizer(
                    initializer_text,
                    add_special_tokens=False,
                )["input_ids"]
                source_ids = [
                    token_id
                    for token_id in source_ids
                    if 0 <= int(token_id) < embedding_size
                ]
                if query_id is not None and query_id >= 0 and source_ids:
                    input_embeddings[int(query_id)].copy_(
                        input_embeddings[source_ids].mean(dim=0)
                    )

if args.model_type == "decoder":
    tokenizer.padding_side = "left"

if args.lora:
    model = get_peft_model(model, lora_config)

model = attach_contrastive_projection_head(model, args)
model = attach_personality_regression_head(model, args)
model = attach_precomputed_audio_adapter(model, args)
model = attach_precomputed_visual_adapter(model, args)
model = attach_stage1_multimodal_fusion(model, args)
model = attach_stage1_context_prefix(model, args)
model = attach_utterance_feature_prefix(model, args)
model = attach_personality_prefix_head(model, args)
model = attach_latent_mood_prefix_head(model, args)
model = attach_personality_mood_prefix_head(model, args)
model = attach_mood_dynamics_head(model, args, len(get_labels_attr(args.dataset)[0]))
model = attach_speaker_mood_dynamics_head(model, args, len(get_labels_attr(args.dataset)[0]))

if args.gradient_checkpointing:
    if "chatglm" in args.model_name_or_path:
        model.supports_gradient_checkpointing = True
        model.transformer.gradient_checkpointing = True
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

if args.checkpoint_dir is not None:
    model = load_zero_checkpoint_safe(model, args.checkpoint_dir)

if args.audio_enable or args.video_enable:
    if str(getattr(config, "model_type", "")).lower() != "qwen2_5_omni":
        raise ValueError("Native audio/video inputs require a Qwen2.5-Omni-compatible model.")

if args.audio_enable:
    if args.audio_embedding_source in (
        'precomputed_pkl',
        'precomputed_tokens',
        'qwen_runtime_extract',
    ):
        if args.audio_embedding_source in (
            'precomputed_pkl',
            'precomputed_tokens',
        ) and (
            args.audio_feature_pkl is None or not os.path.exists(args.audio_feature_pkl)
        ):
            raise FileNotFoundError(
                f"Precomputed audio cache not found: {args.audio_feature_pkl}"
            )
    else:
        raise ValueError(f"Unsupported audio_embedding_source: {args.audio_embedding_source}")

if args.video_enable:
    if args.video_manifest_jsonl is None or not os.path.exists(str(args.video_manifest_jsonl)):
        raise FileNotFoundError(f"Video manifest jsonl not found: {args.video_manifest_jsonl}")
    if args.video_embedding_source == "precomputed_tokens" and (
        args.video_feature_pkl is None or not os.path.exists(str(args.video_feature_pkl))
    ):
        raise FileNotFoundError(
            f"Precomputed video feature pkl not found: {args.video_feature_pkl}"
        )


num_parameters = get_parameter_number(model)
with open(os.path.join(args.output_dir, "model_params.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(num_parameters, indent=5))

## prepare data
train_file = os.path.join(args.data_dir, "train.json")
dev_file = os.path.join(args.data_dir, "test.json")
if not os.path.exists(dev_file):
    print("*****    Desire to evaluate, but test.json file not found *****")
    args.do_eval = False
train_dataset, dev_dataset = None, None
train_collator, dev_collator = None, None
if args.do_train:
    df_train = read_data(train_file, percent=args.data_percent, random_seed=args.seed)
    train_dataset = Seq2SeqDataset(args, df_train, mode='train')
    train_collator = Seq2SeqCollator(args, tokenizer, mode="train")
if args.do_eval:
    dev_datasets = []
    df_dev = read_data(dev_file, percent=1.0, random_seed=args.seed)
    dev_dataset = Seq2SeqDataset(args, df_dev, mode='dev')
    dev_collator = Seq2SeqCollator(args, tokenizer, mode="dev")

stop_word_list = ['sep_token_id', 'eos_token_id', 'pad_token_id']
stop_ids = []
for stop_word in stop_word_list:
    id_ = getattr(tokenizer, stop_word)
    if id_ is not None:
        stop_ids.append(id_)
stop_criteria = KeywordsStoppingCriteria(stop_ids)
## prepare deepspeed model training
ordered_train_batch_sampler = None
ordered_train_dataloader = None
if args.do_train:
    if _speaker_mood_dialogue_ordered_enabled(args):
        ordered_local_batch_size = _speaker_mood_local_micro_batch_size(args, world_size)
        ordered_train_batch_sampler, ordered_train_dataloader = (
            _build_dialogue_ordered_loader(
                dataset=train_dataset,
                collator=train_collator,
                local_batch_size=ordered_local_batch_size,
                args=args,
                shuffle_dialogues=True,
            )
        )
        natural_steps_per_epoch = math.ceil(
            len(ordered_train_batch_sampler) / args.gradient_accumulation_steps
        )
    else:
        natural_steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    natural_t_total = natural_steps_per_epoch * args.num_train_epochs
    t_total = (
        natural_t_total
        if args.scheduler_num_training_steps is None
        else int(args.scheduler_num_training_steps)
    )
    if t_total <= 0:
        raise ValueError("--scheduler_num_training_steps must be greater than zero when provided.")
    warmup_steps = (
        math.ceil(t_total * args.warmup_ratio)
        if args.warmup_steps is None
        else int(args.warmup_steps)
    )
    if warmup_steps < 0 or warmup_steps >= t_total:
        raise ValueError(
            "warmup_steps must satisfy 0 <= warmup_steps < scheduler_num_training_steps; "
            f"got warmup_steps={warmup_steps}, scheduler_num_training_steps={t_total}."
        )
    args.warmup_steps = warmup_steps
    args.scheduler_num_training_steps = t_total

    disable_fused_adam = str(os.getenv("INSTRUCTERC_DISABLE_FUSED_ADAM", "0")).strip().lower() in ("1", "true", "yes", "on")
    use_deepspeed_custom_adam = str(os.getenv("INSTRUCTERC_USE_DEEPSPEED_ADAM", "0")).strip().lower() in ("1", "true", "yes", "on")

    # if args.offload_optimizer or args.zero_shot:
    if args.offload_optimizer:
        deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
        use_deepspeed_custom_adam = True

    optimizer_grouped_parameters = getOptimizerGroup(model=model)

    if use_deepspeed_custom_adam:
        optimizer_class = DeepSpeedCPUAdam if (
            deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu" or disable_fused_adam
        ) else FusedAdam
        optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
        if is_main_process:
            print(
                f"[INFO] Optimizer class: {optimizer_class.__name__} "
                f"(offload_device={deepspeed_config['zero_optimization']['offload_optimizer']['device']}, "
                f"disable_fused_adam={disable_fused_adam}, use_deepspeed_custom_adam={use_deepspeed_custom_adam})"
            )
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
        if is_main_process:
            print(
                f"[INFO] Optimizer class: AdamW "
                f"(use_deepspeed_custom_adam={use_deepspeed_custom_adam}; bypass DeepSpeed op builder)"
            )

    # Build scheduler inside DeepSpeed initialize via callable so the scheduler
    # is created with the correct underlying optimizer type.
    lr_scheduler_factory = lambda opt: get_linear_schedule_with_warmup(
        optimizer=opt,
        num_training_steps=t_total,
        num_warmup_steps=warmup_steps,
    )

    if is_main_process:
        print(
            f"[INFO] LR schedule: peak_lr={args.learning_rate}, warmup_steps={warmup_steps}, "
            f"scheduler_num_training_steps={t_total}, natural_training_steps={natural_t_total}, "
            f"num_train_epochs={args.num_train_epochs}"
        )
        if t_total < natural_t_total:
            print("[INFO] Training steps after the fixed LR schedule horizon run with learning rate clamped at 0.")
        print("[INFO] Start deepspeed.initialize ...")
    ds_init_start = time.time()
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler_factory,
        config=deepspeed_config,
        collate_fn=train_collator
    )
    if ordered_train_dataloader is not None:
        train_dataloader = ordered_train_dataloader
    if is_main_process:
        print(f"[INFO] deepspeed.initialize done in {time.time() - ds_init_start:.1f}s; train batches per epoch: {len(train_dataloader)}")
        if ordered_train_batch_sampler is not None:
            ordered_diag = ordered_train_batch_sampler.diagnostics
            print(
                "[SpeakerMoodOrdered] "
                f"dialogues={ordered_diag['dialogues']}, "
                f"global_steps={ordered_diag['global_steps']}, "
                f"lane_load={ordered_diag['lane_load_min']}..{ordered_diag['lane_load_max']}, "
                f"rank_real={ordered_diag['rank_real_samples']}, "
                f"rank_padding={ordered_diag['rank_padding_samples']}, "
                f"workers={args.speaker_mood_ordered_num_workers}"
            )
    model = _maybe_disable_deepspeed_checkpoint_save(model_engine, args)
    should_save = True
elif args.do_eval:
    if not args.zero_shot:
        model = load_zero_checkpoint_safe(model, args.output_dir)
    dtype = MODEL_TORCH_DTYPE if args.model_type == "decoder" else torch.float32
    model_engine = deepspeed.init_inference(
        model,
        mp_size=world_size,
        replace_with_kernel_inject=True,
        dtype=dtype,
    )
    model = model_engine.module

if __name__ == "__main__":
    eval_score_list = []
    checkpoint_epochs_to_save = parse_checkpoint_epoch_spec(args.save_checkpoint_epochs, args.num_train_epochs)
    lightweight_adapter_epochs_to_save = parse_checkpoint_epoch_spec(
        getattr(args, "save_lightweight_adapter_epochs", ""),
        args.num_train_epochs,
    )
    if is_main_process:
        if len(checkpoint_epochs_to_save) == 0:
            print("[INFO] Epoch checkpoint saving disabled by --save_checkpoint_epochs.")
        else:
            print(f"[INFO] Save epoch checkpoints on epochs: {sorted(checkpoint_epochs_to_save)}")
        print(
            "[INFO] Checkpoint controls: "
            f"save_best_checkpoint={args.save_best_checkpoint}, "
            f"disable_deepspeed_checkpoint_save={args.disable_deepspeed_checkpoint_save}"
        )
        if len(lightweight_adapter_epochs_to_save) > 0:
            print(f"[INFO] Save lightweight adapter snapshots on epochs: {sorted(lightweight_adapter_epochs_to_save)}")
    emotional_label_dict, emotional_label_str = get_labels_attr(dataset=args.dataset)
    emotion_label_names = list(emotional_label_dict.keys())
    mood_dynamics_label_token_ids = _emotion_label_token_ids(tokenizer, emotion_label_names)
    if type(args.theta) == str:
        args.theta = torch.tensor(eval(args.theta))
        args.theta.to(device)
    if type(args.beta) == str:
        args.beta = torch.tensor(eval(args.beta))
        args.beta.to(device)
    # pdb.set_trace()
    if args.do_train:
        global_steps = 0
        best_f1_score = -1e18
        speaker_mood_state_cache = {} if _is_true_flag(getattr(args, "use_speaker_mood_dynamics", "False")) else None
        speaker_mood_emotion_cache = (
            {}
            if (
                _speaker_mood_predicted_prev_emotion_enabled(args)
                or _speaker_mood_simplified_enabled(args)
            )
            else None
        )
        for epoch in range(args.num_train_epochs):
            epoch_started_at = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            if ordered_train_batch_sampler is not None:
                ordered_train_batch_sampler.set_epoch(epoch)
            if (
                speaker_mood_state_cache is not None
                and (
                    _speaker_mood_recurrence_enabled(args)
                    or _speaker_mood_simplified_enabled(args)
                )
            ):
                speaker_mood_state_cache.clear()
            if speaker_mood_emotion_cache is not None:
                speaker_mood_emotion_cache.clear()
            model.train()
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"{run_prefix}Training {epoch + 1}/{args.num_train_epochs}",
                disable=not is_main_process,
                mininterval=0.5,
                dynamic_ncols=True,
                leave=True,
            )
            for step, batch in enumerate(batch_iterator):
                step_started_at = time.time()
                raw_batch = batch
                if args.emotion_prediction:
                    # pdb.set_trace()

                    batch = _get_input_dict(batch)
                    model_inputs = dict(batch)
                    if (
                        args.enable_contrastive
                        or _is_true_flag(
                            getattr(args, "enable_personality_regression_head", "False")
                        )
                        or args.personality_pair_mode != "none"
                    ):
                        model_inputs["output_hidden_states"] = True
                    model_inputs = _apply_conditioning_to_inputs(
                        model,
                        model_inputs,
                        args,
                        speaker_mood_state_cache=speaker_mood_state_cache,
                        speaker_mood_emotion_cache=speaker_mood_emotion_cache,
                    )
                    outputs = model(**_remove_non_model_conditioning_fields(model_inputs))
                    base_label_logits = _emotion_label_logits_at_first_target(
                        outputs.logits,
                        batch["labels"],
                        mood_dynamics_label_token_ids,
                    )
                    outputs = _apply_mood_dynamics_to_outputs(
                        outputs,
                        model_inputs,
                        args,
                        mood_dynamics_label_token_ids,
                    )
                    # print(outputs.loss, outputs.logits.shape,type(outputs))
                    # kl_index = torch.tensor(-2)
                    # res, pre = outputs.logits[:,kl_index,:]
                    # kl_loss = F.kl_div(res.softmax(dim=-1).log(), pre.softmax(dim=-1), reduction='sum')
                    # kl_loss = F.kl_div(F.log_softmax(res, dim=-1), F.softmax(pre, dim=-1), reduction='sum')
                    # print(outputs.loss.shape, kl_loss.shape)
                    
                    # loss = outputs.loss + args.theta*kl_loss
                    if _is_true_flag(getattr(args, "use_mood_dynamics", "False")) or _is_true_flag(
                        getattr(args, "use_speaker_mood_dynamics", "False")
                    ):
                        loss = compute_task_weighted_loss(
                            outputs=outputs,
                            labels=batch["labels"],
                            tokenizer=tokenizer,
                            speaker_loss_weight=args.speaker_loss_weight,
                            personality_loss_weight=args.personality_loss_weight,
                            task_loss_normalization=args.task_loss_normalization,
                            sample_padding_mask=model_inputs.get(
                                "speaker_mood_order_padding"
                            ),
                        )
                    else:
                        loss = outputs.loss
                    erc_loss = loss
                    
                    
                else:
                    batch = _get_input_dict(raw_batch)
                    model_inputs = dict(batch)
                    if (
                        args.enable_contrastive
                        or _is_true_flag(
                            getattr(args, "enable_personality_regression_head", "False")
                        )
                        or args.personality_pair_mode != "none"
                    ):
                        model_inputs["output_hidden_states"] = True
                    model_inputs = _apply_conditioning_to_inputs(
                        model,
                        model_inputs,
                        args,
                        speaker_mood_state_cache=speaker_mood_state_cache,
                        speaker_mood_emotion_cache=speaker_mood_emotion_cache,
                    )
                    outputs = model(**_remove_non_model_conditioning_fields(model_inputs))
                    base_label_logits = _emotion_label_logits_at_first_target(
                        outputs.logits,
                        batch["labels"],
                        mood_dynamics_label_token_ids,
                    )
                    outputs = _apply_mood_dynamics_to_outputs(
                        outputs,
                        model_inputs,
                        args,
                        mood_dynamics_label_token_ids,
                    )
                    loss = compute_task_weighted_loss(
                        outputs=outputs,
                        labels=batch["labels"],
                        tokenizer=tokenizer,
                        speaker_loss_weight=args.speaker_loss_weight,
                        personality_loss_weight=args.personality_loss_weight,
                        task_loss_normalization=args.task_loss_normalization,
                        sample_padding_mask=model_inputs.get(
                            "speaker_mood_order_padding"
                        ),
                    )
                    erc_loss = loss
                final_label_logits = _emotion_label_logits_at_first_target(
                    outputs.logits,
                    batch["labels"],
                    mood_dynamics_label_token_ids,
                )
                predicted_emotion_indices = (
                    final_label_logits.argmax(dim=-1)
                    if final_label_logits is not None
                    else None
                )
                contrastive_loss = _compute_contrastive_loss(
                    model=model,
                    outputs=outputs,
                    batch=raw_batch if not args.emotion_prediction else batch,
                    args=args,
                )
                if contrastive_loss is not None:
                    loss = loss + float(args.contrastive_weight) * contrastive_loss
                regression_loss = _compute_personality_regression_loss(
                    model=model,
                    outputs=outputs,
                    batch=raw_batch if not args.emotion_prediction else batch,
                    args=args,
                )
                if regression_loss is not None:
                    loss = loss + float(args.personality_regression_loss_weight) * regression_loss
                pair_losses = _compute_personality_pair_losses(
                    model=model,
                    outputs=outputs,
                    batch=raw_batch if not args.emotion_prediction else batch,
                    args=args,
                )
                if pair_losses is not None:
                    loss = loss + pair_losses["total"]
                context_aux_loss = _consume_stage1_context_aux_loss(model, args)
                if context_aux_loss is not None:
                    loss = loss + float(args.stage1_context_consistency_weight) * context_aux_loss
                speaker_mood_aux_losses = _compute_speaker_mood_auxiliary_losses(
                    model_inputs=model_inputs,
                    labels=batch["labels"],
                    tokenizer=tokenizer,
                    label_names=emotion_label_names,
                    args=args,
                    base_label_logits=base_label_logits,
                    speaker_mood_state_cache=speaker_mood_state_cache,
                )
                if speaker_mood_aux_losses is not None:
                    loss = loss + speaker_mood_aux_losses["total"]

                ordered_padding = model_inputs.get("speaker_mood_order_padding")
                ordered_real_count = None
                ordered_batch_count = None
                if ordered_padding is not None:
                    ordered_padding = ordered_padding.to(device=loss.device).bool().view(-1)
                    ordered_batch_count = int(ordered_padding.numel())
                    ordered_real_count = int((~ordered_padding).sum().item())
                    batch_fraction = ordered_real_fraction(loss, ordered_padding)
                    loss = loss * batch_fraction
                    erc_loss = erc_loss * batch_fraction

                if (
                    _speaker_mood_dialogue_ordered_enabled(args)
                    and not bool(torch.isfinite(loss.detach()).all())
                ):
                    raise FloatingPointError(
                        f"Non-finite ordered training loss at "
                        f"epoch={epoch + 1}, step={step + 1}, "
                        f"real_rows={ordered_real_count}/{ordered_batch_count}."
                    )
                model.backward(loss)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_norm,
                    error_if_nonfinite=_speaker_mood_dialogue_ordered_enabled(args),
                )
                model.step()
                if is_main_process and step == 0:
                    text_count = int(model_inputs.get('utterance_feature_text_count', torch.tensor(0)).sum().item()) if 'utterance_feature_text_count' in model_inputs else 0
                    audio_count = int(model_inputs.get('utterance_feature_audio_count', torch.tensor(0)).sum().item()) if 'utterance_feature_audio_count' in model_inputs else 0
                    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_UTTERANCE_TOKEN)
                    direct_count = int(batch['input_ids'].eq(int(audio_token_id)).sum().item()) if isinstance(audio_token_id, int) and audio_token_id >= 0 else 0
                    direct_utterances = int(model_inputs.get('direct_audio_utterance_count', torch.tensor(0)).sum().item()) if 'direct_audio_utterance_count' in model_inputs else 0
                    base_count = int(batch['attention_mask'].sum().item())
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                    print(f'[InputStats] epoch={epoch + 1} lm_tokens={base_count} direct_audio_utterances={direct_utterances} direct_audio_tokens={direct_count} feature_text_tokens={text_count} feature_audio_tokens={audio_count} peak_gib={peak_gib:.2f} first_step_s={time.time() - step_started_at:.3f}', flush=True)
                _update_speaker_mood_state_cache(speaker_mood_state_cache, model_inputs, args)
                if _speaker_mood_simplified_active(args):
                    _update_speaker_mood_soft_emotion_cache(
                        speaker_mood_emotion_cache,
                        model_inputs,
                        final_label_logits,
                        len(emotion_label_names),
                    )
                else:
                    _update_speaker_mood_emotion_cache(
                        speaker_mood_emotion_cache,
                        model_inputs,
                        predicted_emotion_indices,
                        len(emotion_label_names),
                    )
                    
                current_loss = loss.item()
                if is_main_process:
                    postfix = {
                        "loss": f"{current_loss:.4f}",
                        "erc": f"{float(erc_loss.detach().item()):.4f}",
                    }
                    if contrastive_loss is not None:
                        postfix["con"] = f"{float(contrastive_loss.detach().item()):.4f}"
                    if regression_loss is not None:
                        postfix["preg"] = f"{float(regression_loss.detach().item()):.4f}"
                    if pair_losses is not None:
                        postfix["ppair"] = (
                            f"{float(pair_losses['total'].detach().item()):.4f}"
                        )
                        postfix["paux"] = (
                            f"{float(pair_losses['auxiliary'].detach().item()):.4f}"
                        )
                    if context_aux_loss is not None:
                        postfix["ctx"] = f"{float(context_aux_loss.detach().item()):.4f}"
                    if speaker_mood_aux_losses is not None:
                        if "prior" in speaker_mood_aux_losses:
                            postfix["mpr"] = (
                                f"{float(speaker_mood_aux_losses['prior'].detach().item()):.4f}"
                            )
                        if "prior_uncertainty_weight" in speaker_mood_aux_losses:
                            postfix["mpw"] = (
                                f"{float(speaker_mood_aux_losses['prior_uncertainty_weight'].detach().item()):.4f}"
                            )
                        if "shift" in speaker_mood_aux_losses:
                            postfix["msh"] = (
                                f"{float(speaker_mood_aux_losses['shift'].detach().item()):.4f}"
                            )
                        if "align" in speaker_mood_aux_losses:
                            postfix["mal"] = (
                                f"{float(speaker_mood_aux_losses['align'].detach().item()):.4f}"
                            )
                        if "dyn" in speaker_mood_aux_losses:
                            postfix["mdyn"] = (
                                f"{float(speaker_mood_aux_losses['dyn'].detach().item()):.4f}"
                            )
                        if "dyn_count" in speaker_mood_aux_losses:
                            postfix["mdync"] = (
                                f"{float(speaker_mood_aux_losses['dyn_count'].detach().item()):.0f}"
                            )
                        if "dyn_ratio" in speaker_mood_aux_losses:
                            postfix["mdynr"] = (
                                f"{float(speaker_mood_aux_losses['dyn_ratio'].detach().item()):.2f}"
                            )
                        if "total" in speaker_mood_aux_losses:
                            postfix["maux"] = (
                                f"{float(speaker_mood_aux_losses['total'].detach().item()):.4f}"
                            )
                    if _speaker_mood_simplified_enabled(args):
                        soft_valid = model_inputs.get("speaker_mood_soft_prev_valid")
                        soft_expected = model_inputs.get(
                            "speaker_mood_soft_prev_expected"
                        )
                        if soft_valid is not None and soft_expected is not None:
                            hit_count = int(soft_valid.detach().sum().item())
                            expected_count = int(soft_expected.detach().sum().item())
                            postfix["msoft"] = f"{hit_count}/{expected_count}"
                    elif _speaker_mood_predicted_prev_emotion_enabled(args):
                        emotion_valid = model_inputs.get(
                            "speaker_mood_predicted_prev_valid"
                        )
                        emotion_expected = model_inputs.get(
                            "speaker_mood_predicted_prev_expected"
                        )
                        if emotion_valid is not None and emotion_expected is not None:
                            hit_count = int(emotion_valid.detach().sum().item())
                            expected_count = int(
                                emotion_expected.detach().sum().item()
                            )
                            postfix["meh"] = f"{hit_count}/{expected_count}"
                    if _speaker_mood_recurrence_enabled(args):
                        recurrent_valid = model_inputs.get(
                            "speaker_mood_recurrent_prev_valid"
                        )
                        recurrent_expected = model_inputs.get(
                            "speaker_mood_recurrent_prev_expected"
                        )
                        if recurrent_valid is not None and recurrent_expected is not None:
                            hit_count = int(recurrent_valid.detach().sum().item())
                            expected_count = int(recurrent_expected.detach().sum().item())
                            postfix["mrh"] = f"{hit_count}/{expected_count}"
                    if ordered_real_count is not None:
                        postfix["orb"] = (
                            f"{ordered_real_count}/{ordered_batch_count}"
                        )
                    batch_iterator.set_postfix(postfix, refresh=False)

                if step % args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    should_save = True
                if global_steps % args.save_steps == 0 and should_save:
                    model.save_checkpoint(args.output_dir)
                    should_save = False

                # model starts to evaluation
            
            if is_main_process:
                elapsed = max(time.time() - epoch_started_at, 1e-6)
                peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                print(f'[EpochThroughput] epoch={epoch + 1} batches={len(train_dataloader)} batches_per_s={len(train_dataloader) / elapsed:.4f} peak_gib={peak_gib:.2f}', flush=True)
            should_eval_this_epoch = args.do_eval and ((epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.num_train_epochs)
            if not should_eval_this_epoch:
                continue

            model.eval()
            targets = list(df_dev["output"])
            eval_sample_ids = list(df_dev["sample_id"]) if "sample_id" in df_dev.columns else [None] * len(targets)
            eval_debug_target_utterances = (
                list(df_dev["debug_target_utterance"])
                if "debug_target_utterance" in df_dev.columns
                else [None] * len(targets)
            )
            eval_utterance_ids = (
                list(df_dev["utterance_id"])
                if "utterance_id" in df_dev.columns
                else [None] * len(targets)
            )
            eval_sampler, eval_dataloader, local_eval_indices = _build_eval_loader(
                dev_dataset=dev_dataset,
                dev_collator=dev_collator,
                eval_batch_size=args.eval_batch_size,
                args=args,
            )
            eval_speaker_mood_state_cache = (
                {} if _speaker_mood_recurrence_enabled(args) else None
            )
            eval_speaker_mood_emotion_cache = (
                {}
                if (
                    _speaker_mood_predicted_prev_emotion_enabled(args)
                    or _speaker_mood_simplified_enabled(args)
                )
                else None
            )
            if is_main_process:
                print(f"[INFO] Eval samples: {len(dev_dataset)}, local eval batches: {len(eval_dataloader)}, world_size: {world_size}, eval_batch_size: {args.eval_batch_size}")

            preds_for_eval_path = os.path.join(args.output_dir, f"preds_for_eval_{epoch}.text")
            if is_main_process:
                print("\n*****    Evaluating  *****\n")
            
            # Suppress pad_token_id warnings during generation
            import logging
            transformers_logger = logging.getLogger("transformers")
            old_level = transformers_logger.level
            transformers_logger.setLevel(logging.ERROR)

            local_records = []
            local_cursor = 0
            eval_iterator = tqdm(
                eval_dataloader,
                desc=f"{run_prefix}Evaluating",
                disable=not is_main_process,
                mininterval=0.5,
                dynamic_ncols=True,
                leave=True,
            )
            for eval_step, eval_batch in enumerate(eval_iterator):
                eval_batch = _move_batch_to_device(eval_batch, device)
                input_ids_cpu = eval_batch["input_ids"].detach().cpu()
                if "attention_mask" in eval_batch:
                    input_lens = eval_batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
                else:
                    input_lens = [eval_batch["input_ids"].size(1)] * eval_batch["input_ids"].size(0)
                regression_personality = None
                pair_personality = None
                pair_targets = None
                if (
                    _is_true_flag(
                        getattr(args, "enable_personality_regression_head", "False")
                    )
                    or args.personality_pair_mode != "none"
                ):
                    regression_batch = _get_generation_input_dict(eval_batch)
                    regression_batch = dict(regression_batch)
                    regression_batch["output_hidden_states"] = True
                    regression_batch = _apply_conditioning_to_inputs(
                        model,
                        regression_batch,
                        args,
                        speaker_mood_state_cache=eval_speaker_mood_state_cache,
                        speaker_mood_emotion_cache=eval_speaker_mood_emotion_cache,
                    )
                    with torch.no_grad():
                        regression_outputs = model(**_remove_non_model_conditioning_fields(regression_batch))
                        regression_pred_tensor = _compute_personality_regression_predictions(
                            model=model,
                            outputs=regression_outputs,
                            batch=regression_batch,
                        )
                    if regression_pred_tensor is not None:
                        regression_personality = regression_pred_tensor.detach().float().cpu().tolist()
                    if args.personality_pair_mode != "none":
                        pair_pred_tensor = _compute_personality_pair_predictions(
                            model=model,
                            outputs=regression_outputs,
                            batch=eval_batch,
                        )
                        if pair_pred_tensor is not None:
                            pair_personality = (
                                pair_pred_tensor.detach().float().cpu().tolist()
                            )
                            pair_targets = (
                                eval_batch["personality_pair_targets"]
                                .detach()
                                .float()
                                .cpu()
                                .tolist()
                            )
                generation_batch = _get_generation_input_dict(eval_batch)
                generation_batch = _apply_conditioning_to_inputs(
                    model,
                    generation_batch,
                    args,
                    speaker_mood_state_cache=eval_speaker_mood_state_cache,
                    speaker_mood_emotion_cache=eval_speaker_mood_emotion_cache,
                )
                mood_eval_traces = _mood_dynamics_eval_trace(generation_batch, emotion_label_names)
                generation_forward_batch = _remove_non_model_conditioning_fields(generation_batch)
                generation_input_ids_cpu = generation_forward_batch["input_ids"].detach().cpu()
                base_generation_logits = _maybe_compute_base_generation_logits(
                    model,
                    generation_forward_batch,
                    args,
                )
                speaker_mood_eval_traces = _speaker_mood_eval_trace(
                    generation_batch,
                    emotion_label_names,
                    label_token_ids=mood_dynamics_label_token_ids,
                    base_logits=base_generation_logits,
                    scale=float(getattr(args, "speaker_mood_lambda", 0.2)),
                    args=args,
                ) if _is_true_flag(getattr(args, "save_speaker_mood_analysis", "False")) else None
                mood_logits_processor = _build_mood_logits_processor_list(
                    generation_batch,
                    args,
                    mood_dynamics_label_token_ids,
                    initial_length=int(generation_forward_batch["input_ids"].size(1)),
                )
                pair_only_batch = (
                    args.personality_pair_mode != "none"
                    and "personality_pair_mask" in eval_batch
                    and bool(eval_batch["personality_pair_mask"].all().item())
                )
                with torch.no_grad():
                    if pair_only_batch:
                        outputs = generation_forward_batch["input_ids"].detach().clone()
                    elif "chatglm" in args.model_name_or_path:
                        outputs = model.generate(
                            **generation_forward_batch,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            logits_processor=mood_logits_processor,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    else:
                        outputs = model.generate(
                            **generation_forward_batch,
                            num_beams=args.num_beams,
                            do_sample=False,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            repetition_penalty=args.repetition_penalty,
                            num_return_sequences=1,
                            logits_processor=mood_logits_processor,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            # stopping_criteria=StoppingCriteriaList([stop_criteria]
                        )
                _update_speaker_mood_state_cache(
                    eval_speaker_mood_state_cache,
                    generation_batch,
                    args,
                )
                simplified_generation_logits = None
                if _speaker_mood_simplified_active(args):
                    simplified_generation_logits = (
                        _captured_simplified_generation_label_logits(
                            mood_logits_processor,
                            generation_forward_batch["input_ids"].size(0),
                        )
                    )
                    if simplified_generation_logits is None and not pair_only_batch:
                        raise RuntimeError(
                            "Simplified MoodFiLM failed to capture final first-step "
                            "generation logits."
                        )
                    if simplified_generation_logits is not None:
                        _update_speaker_mood_soft_emotion_cache(
                            eval_speaker_mood_emotion_cache,
                            generation_batch,
                            simplified_generation_logits,
                            len(emotion_label_names),
                        )
                outputs = outputs.detach().cpu()
                outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
                predicted_emotion_indices = _generated_emotion_indices(
                    tokenizer=tokenizer,
                    generated_outputs=outputs,
                    generation_input_ids=generation_input_ids_cpu,
                    label_names=emotion_label_names,
                )
                if not _speaker_mood_simplified_active(args):
                    _update_speaker_mood_emotion_cache(
                        eval_speaker_mood_emotion_cache,
                        generation_batch,
                        predicted_emotion_indices,
                        len(emotion_label_names),
                    )

                batch_size_now = outputs.size(0)
                batch_indices, batch_padding, local_cursor = _eval_batch_dataset_indices(
                    eval_batch,
                    local_eval_indices,
                    local_cursor,
                )

                for i in range(batch_size_now):
                    if batch_padding[i]:
                        continue
                    global_idx = int(batch_indices[i])
                    this_input = tokenizer.decode(
                        input_ids_cpu[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    if getattr(args, 'audio_input_mode', 'text_audio') == 'audio_only':
                        this_input = mask_prompt_content_for_audio_only(this_input)
                    if args.model_type == "decoder":
                        output_ids = outputs[i]
                        input_ids = generation_input_ids_cpu[i]
                        gen_ids = _strip_prompt_from_generated_ids(input_ids, output_ids)
                        answer = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    else:
                        answer = tokenizer.decode(outputs[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    raw_answer = answer.strip().rstrip()
                    answer, parse_info = _normalize_prediction_for_target(
                        target_text=targets[global_idx],
                        raw_answer=raw_answer,
                        emotional_label_dict=emotional_label_dict,
                        return_info=True,
                    )

                    record = {
                        "index": global_idx,
                        "input": this_input,
                        "output": answer.strip().rstrip(),
                        "target": targets[global_idx],
                    }
                    _add_generation_trace_fields(
                        record,
                        raw_answer,
                        target_text=targets[global_idx],
                        emotional_label_dict=emotional_label_dict,
                        parse_info=parse_info,
                    )
                    _add_eval_debug_fields(
                        record,
                        sample_id=eval_sample_ids[global_idx],
                        debug_target_utterance=eval_debug_target_utterances[global_idx],
                        utterance_id=eval_utterance_ids[global_idx],
                    )
                    if regression_personality is not None and i < len(regression_personality):
                        record["personality_regression"] = _personality_values_to_dict(regression_personality[i])
                    if mood_eval_traces is not None and i < len(mood_eval_traces):
                        record["mood_dynamics"] = mood_eval_traces[i]
                    if speaker_mood_eval_traces is not None and i < len(speaker_mood_eval_traces):
                        record["speaker_mood_dynamics"] = speaker_mood_eval_traces[i]
                    if pair_personality is not None and i < len(pair_personality):
                        pair_mask = bool(
                            eval_batch["personality_pair_mask"][i].item()
                        )
                        if pair_mask:
                            record["personality_pair_predictions"] = pair_personality[i]
                            record["personality_pair_targets"] = pair_targets[i]
                    _add_inference_reasoning_fields(
                        record,
                        args,
                        model,
                        tokenizer,
                        device,
                        this_input,
                        answer.strip().rstrip(),
                        global_idx,
                    )
                    local_records.append(record)
            
            # Restore logging level
            transformers_logger.setLevel(old_level)

            # Release the final generation batch before rank 0 starts merging
            # evaluation records. This is important on multi-process 7B runs.
            eval_batch = None
            generation_batch = None
            outputs = None
            input_ids_cpu = None
            generation_input_ids_cpu = None
            del eval_iterator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_memory_checkpoint(f"epoch_{epoch}_after_local_eval_before_gather")

            all_records = _gather_eval_records(
                local_records,
                output_dir=args.output_dir,
                gather_tag=f"epoch_{epoch}",
            )
            if is_main_process:
                _log_memory_checkpoint(f"epoch_{epoch}_after_gather_records={len(all_records)}")
            preds_for_eval = _dedupe_records_by_index(all_records)
            all_answers = [item["output"] for item in preds_for_eval]
            eval_targets = [item["target"] for item in preds_for_eval]
            regression_predictions = [item.get("personality_regression") for item in preds_for_eval]
            if args.statistic_mode == "True":
                task_report = build_task_statistics(
                    targets=eval_targets,
                    answers=all_answers,
                    emotional_label_dict=emotional_label_dict,
                    dataset=args.dataset,
                    speaker_loss_weight=args.speaker_loss_weight,
                    personality_loss_weight=args.personality_loss_weight,
                )
                task_report = _attach_personality_regression_report(
                    task_report,
                    targets=eval_targets,
                    regression_predictions=regression_predictions,
                    args=args,
                )
                task_report = _attach_personality_pair_report(
                    task_report,
                    records=preds_for_eval,
                    args=args,
                )
                task_report = _attach_generated_continuous_pair_report(
                    task_report,
                    records=preds_for_eval,
                    args=args,
                )
                task_report = _attach_run_modality_config(task_report, args)

                if is_main_process:
                    _log_memory_checkpoint(f"epoch_{epoch}_before_write_outputs")
                    _write_eval_outputs(preds_for_eval_path, preds_for_eval, task_report=task_report)
                    _log_memory_checkpoint(f"epoch_{epoch}_after_write_outputs")

                selection_metric = task_report.get('selection_metric') if is_main_process else None
                should_save_best = False
                next_best_f1_score = best_f1_score
                if selection_metric is not None and best_f1_score < selection_metric['value']:
                    should_save_best = True
                    next_best_f1_score = selection_metric['value']
                if dist.is_available() and dist.is_initialized() and world_size > 1:
                    best_payload = torch.tensor(
                        [float(should_save_best), float(next_best_f1_score)],
                        device=device,
                        dtype=torch.float64,
                    )
                    if is_main_process:
                        _log_memory_checkpoint(f"epoch_{epoch}_before_best_broadcast")
                    dist.broadcast(best_payload, src=0)
                    should_save_best = bool(best_payload[0].item())
                    next_best_f1_score = float(best_payload[1].item())
                    if is_main_process:
                        _log_memory_checkpoint(f"epoch_{epoch}_after_best_broadcast")
                if should_save_best:
                    best_f1_score = next_best_f1_score
                    if (
                        bool(getattr(args, "save_best_checkpoint", False))
                        and not bool(getattr(args, "disable_deepspeed_checkpoint_save", False))
                    ):
                        model.save_checkpoint(args.output_dir, tag="best_selection_metric")
                        if is_main_process:
                            tokenizer.save_pretrained(args.output_dir)
                            config.save_pretrained(args.output_dir)
                            args.save(args.output_dir)
                            if args.lora:
                                _save_lightweight_adapter_snapshot(
                                    model,
                                    args.output_dir,
                                    tokenizer=None,
                                    config=None,
                                    args=None,
                                )
                            with open(os.path.join(args.output_dir, "deepspeed_config.json"), 'w', encoding='utf-8') as f:
                                f.write(json.dumps(deepspeed_config, indent=5))
                    elif is_main_process:
                        print(
                            "[INFO] New best selection metric reached "
                            f"({best_f1_score:.6f}); skip DeepSpeed best checkpoint "
                            f"(save_best_checkpoint={args.save_best_checkpoint}, "
                            f"disable_deepspeed_checkpoint_save={args.disable_deepspeed_checkpoint_save})."
                        )
            if args.statistic_mode != 'True':
                if is_main_process:
                    _write_eval_outputs(preds_for_eval_path, preds_for_eval, task_report=None)
                    tokenizer.save_pretrained(args.output_dir)
                    config.save_pretrained(args.output_dir)
                    args.save(args.output_dir)
                    # Save LoRA adapter weights if using PEFT
                    if args.lora and hasattr(model, 'save_pretrained'):
                        model.save_pretrained(args.output_dir)
                if is_main_process:
                    with open(os.path.join(args.output_dir, "deepspeed_config.json"), 'w', encoding='utf-8') as f:
                        f.write(json.dumps(deepspeed_config, indent=5))

            current_epoch = epoch + 1
            if current_epoch in lightweight_adapter_epochs_to_save and is_main_process:
                adapter_dir = os.path.join(args.output_dir, f"adapter_epoch{current_epoch}")
                saved = _save_lightweight_adapter_snapshot(
                    model,
                    adapter_dir,
                    tokenizer=tokenizer,
                    config=config,
                    args=args,
                )
                if saved:
                    print(f"[INFO] Saved lightweight adapter snapshot: {adapter_dir}")
                else:
                    print("[WARNING] Requested lightweight adapter snapshot, but model does not expose save_pretrained.")
            if current_epoch in checkpoint_epochs_to_save:
                epoch_ckpt_tag = f"epoch{current_epoch}_global_step{global_steps}"
                model.save_checkpoint(args.output_dir, tag=epoch_ckpt_tag)
            elif is_main_process:
                print(f"[INFO] Skip checkpoint save at epoch {current_epoch} by --save_checkpoint_epochs setting.")

            local_records.clear()
            all_records.clear()
            preds_for_eval.clear()
            del local_records, all_records, preds_for_eval
            del all_answers, eval_targets, regression_predictions
            if args.statistic_mode == "True":
                del task_report
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_main_process:
                _log_memory_checkpoint(f"epoch_{current_epoch}_before_post_eval_barrier")
            _safe_dist_barrier(tag=f"post_eval_cleanup_epoch_{current_epoch}")
            
    if not args.do_train and args.do_eval:
        # model starts to evaluation
        model.eval()
        targets = list(df_dev["output"])
        eval_sample_ids = list(df_dev["sample_id"]) if "sample_id" in df_dev.columns else [None] * len(targets)
        eval_debug_target_utterances = (
            list(df_dev["debug_target_utterance"])
            if "debug_target_utterance" in df_dev.columns
            else [None] * len(targets)
        )
        eval_utterance_ids = (
            list(df_dev["utterance_id"])
            if "utterance_id" in df_dev.columns
            else [None] * len(targets)
        )
        eval_sampler, eval_dataloader, local_eval_indices = _build_eval_loader(
            dev_dataset=dev_dataset,
            dev_collator=dev_collator,
            eval_batch_size=args.eval_batch_size,
            args=args,
        )
        eval_speaker_mood_state_cache = (
            {} if _speaker_mood_recurrence_enabled(args) else None
        )
        eval_speaker_mood_emotion_cache = (
            {}
            if (
                _speaker_mood_predicted_prev_emotion_enabled(args)
                or _speaker_mood_simplified_enabled(args)
            )
            else None
        )
        if is_main_process:
            print(f"[INFO] Eval samples: {len(dev_dataset)}, local eval batches: {len(eval_dataloader)}, world_size: {world_size}, eval_batch_size: {args.eval_batch_size}")

        preds_for_eval_path = os.path.join(args.output_dir, f"preds_for_eval.text")
        if is_main_process:
            print("\n*****    Evaluating  *****\n")
        
        # Suppress pad_token_id warnings during generation
        import logging
        transformers_logger = logging.getLogger("transformers")
        old_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        local_records = []
        local_cursor = 0
        eval_iterator = tqdm(
            eval_dataloader,
            desc=f"{run_prefix}Evaluating",
            disable=not is_main_process,
            mininterval=0.5,
            dynamic_ncols=True,
            leave=True,
        )
        for eval_step, eval_batch in enumerate(eval_iterator):
            eval_batch = _move_batch_to_device(eval_batch, device)
            input_ids_cpu = eval_batch["input_ids"].detach().cpu()
            if "attention_mask" in eval_batch:
                input_lens = eval_batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
            else:
                input_lens = [eval_batch["input_ids"].size(1)] * eval_batch["input_ids"].size(0)
            regression_personality = None
            pair_personality = None
            pair_targets = None
            if (
                _is_true_flag(
                    getattr(args, "enable_personality_regression_head", "False")
                )
                or args.personality_pair_mode != "none"
            ):
                regression_batch = _get_generation_input_dict(eval_batch)
                regression_batch = dict(regression_batch)
                regression_batch["output_hidden_states"] = True
                regression_batch = _apply_conditioning_to_inputs(
                    model,
                    regression_batch,
                    args,
                    speaker_mood_state_cache=eval_speaker_mood_state_cache,
                    speaker_mood_emotion_cache=eval_speaker_mood_emotion_cache,
                )
                with torch.no_grad():
                    regression_outputs = model(**_remove_non_model_conditioning_fields(regression_batch))
                    regression_pred_tensor = _compute_personality_regression_predictions(
                        model=model,
                        outputs=regression_outputs,
                        batch=regression_batch,
                    )
                if regression_pred_tensor is not None:
                    regression_personality = regression_pred_tensor.detach().float().cpu().tolist()
                if args.personality_pair_mode != "none":
                    pair_pred_tensor = _compute_personality_pair_predictions(
                        model=model,
                        outputs=regression_outputs,
                        batch=eval_batch,
                    )
                    if pair_pred_tensor is not None:
                        pair_personality = (
                            pair_pred_tensor.detach().float().cpu().tolist()
                        )
                        pair_targets = (
                            eval_batch["personality_pair_targets"]
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                        )
            generation_batch = _get_generation_input_dict(eval_batch)
            generation_batch = _apply_conditioning_to_inputs(
                model,
                generation_batch,
                args,
                speaker_mood_state_cache=eval_speaker_mood_state_cache,
                speaker_mood_emotion_cache=eval_speaker_mood_emotion_cache,
            )
            mood_eval_traces = _mood_dynamics_eval_trace(generation_batch, emotion_label_names)
            generation_forward_batch = _remove_non_model_conditioning_fields(generation_batch)
            generation_input_ids_cpu = generation_forward_batch["input_ids"].detach().cpu()
            base_generation_logits = _maybe_compute_base_generation_logits(
                model,
                generation_forward_batch,
                args,
            )
            speaker_mood_eval_traces = _speaker_mood_eval_trace(
                generation_batch,
                emotion_label_names,
                label_token_ids=mood_dynamics_label_token_ids,
                base_logits=base_generation_logits,
                scale=float(getattr(args, "speaker_mood_lambda", 0.2)),
                args=args,
            ) if _is_true_flag(getattr(args, "save_speaker_mood_analysis", "False")) else None
            mood_logits_processor = _build_mood_logits_processor_list(
                generation_batch,
                args,
                mood_dynamics_label_token_ids,
                initial_length=int(generation_forward_batch["input_ids"].size(1)),
            )
            pair_only_batch = (
                args.personality_pair_mode != "none"
                and "personality_pair_mask" in eval_batch
                and bool(eval_batch["personality_pair_mask"].all().item())
            )
            with torch.no_grad():
                if pair_only_batch:
                    outputs = generation_forward_batch["input_ids"].detach().clone()
                elif "chatglm" in args.model_name_or_path:
                    outputs = model.generate(
                        **generation_forward_batch,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        logits_processor=mood_logits_processor,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                else:
                    outputs = model.generate(
                        **generation_forward_batch,
                        num_beams=args.num_beams,
                        do_sample=False,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                        num_return_sequences=1,
                        logits_processor=mood_logits_processor,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        # stopping_criteria=StoppingCriteriaList([stop_criteria])
                    )
            _update_speaker_mood_state_cache(
                eval_speaker_mood_state_cache,
                generation_batch,
                args,
            )
            simplified_generation_logits = None
            if _speaker_mood_simplified_active(args):
                simplified_generation_logits = (
                    _captured_simplified_generation_label_logits(
                        mood_logits_processor,
                        generation_forward_batch["input_ids"].size(0),
                    )
                )
                if simplified_generation_logits is None and not pair_only_batch:
                    raise RuntimeError(
                        "Simplified MoodFiLM failed to capture final first-step "
                        "generation logits."
                    )
                if simplified_generation_logits is not None:
                    _update_speaker_mood_soft_emotion_cache(
                        eval_speaker_mood_emotion_cache,
                        generation_batch,
                        simplified_generation_logits,
                        len(emotion_label_names),
                    )
            outputs = outputs.detach().cpu()
            outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
            predicted_emotion_indices = _generated_emotion_indices(
                tokenizer=tokenizer,
                generated_outputs=outputs,
                generation_input_ids=generation_input_ids_cpu,
                label_names=emotion_label_names,
            )
            if not _speaker_mood_simplified_active(args):
                _update_speaker_mood_emotion_cache(
                    eval_speaker_mood_emotion_cache,
                    generation_batch,
                    predicted_emotion_indices,
                    len(emotion_label_names),
                )

            batch_size_now = outputs.size(0)
            batch_indices, batch_padding, local_cursor = _eval_batch_dataset_indices(
                eval_batch,
                local_eval_indices,
                local_cursor,
            )

            for i in range(batch_size_now):
                if batch_padding[i]:
                    continue
                global_idx = int(batch_indices[i])
                this_input = tokenizer.decode(
                    input_ids_cpu[i],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                if getattr(args, 'audio_input_mode', 'text_audio') == 'audio_only':
                    this_input = mask_prompt_content_for_audio_only(this_input)
                if args.model_type == "decoder":
                    output_ids = outputs[i]
                    input_ids = generation_input_ids_cpu[i]
                    gen_ids = _strip_prompt_from_generated_ids(input_ids, output_ids)
                    answer = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                else:
                    answer = tokenizer.decode(outputs[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                raw_answer = answer.strip().rstrip()
                answer, parse_info = _normalize_prediction_for_target(
                    target_text=targets[global_idx],
                    raw_answer=raw_answer,
                    emotional_label_dict=emotional_label_dict,
                    return_info=True,
                )

                record = {
                    "index": global_idx,
                    "input": this_input,
                    "output": answer.strip().rstrip(),
                    "target": targets[global_idx],
                }
                _add_generation_trace_fields(
                    record,
                    raw_answer,
                    target_text=targets[global_idx],
                    emotional_label_dict=emotional_label_dict,
                    parse_info=parse_info,
                )
                _add_eval_debug_fields(
                    record,
                    sample_id=eval_sample_ids[global_idx],
                    debug_target_utterance=eval_debug_target_utterances[global_idx],
                    utterance_id=eval_utterance_ids[global_idx],
                )
                if regression_personality is not None and i < len(regression_personality):
                    record["personality_regression"] = _personality_values_to_dict(regression_personality[i])
                if mood_eval_traces is not None and i < len(mood_eval_traces):
                    record["mood_dynamics"] = mood_eval_traces[i]
                if speaker_mood_eval_traces is not None and i < len(speaker_mood_eval_traces):
                    record["speaker_mood_dynamics"] = speaker_mood_eval_traces[i]
                if pair_personality is not None and i < len(pair_personality):
                    pair_mask = bool(
                        eval_batch["personality_pair_mask"][i].item()
                    )
                    if pair_mask:
                        record["personality_pair_predictions"] = pair_personality[i]
                        record["personality_pair_targets"] = pair_targets[i]
                _add_inference_reasoning_fields(
                    record,
                    args,
                    model,
                    tokenizer,
                    device,
                    this_input,
                    answer.strip().rstrip(),
                    global_idx,
                )
                local_records.append(record)
        
        # Restore logging level
        transformers_logger.setLevel(old_level)

        eval_batch = None
        generation_batch = None
        outputs = None
        input_ids_cpu = None
        generation_input_ids_cpu = None
        del eval_iterator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_records = _gather_eval_records(
            local_records,
            output_dir=args.output_dir,
            gather_tag="eval_only",
        )
        preds_for_eval = _dedupe_records_by_index(all_records)
        all_answers = [item["output"] for item in preds_for_eval]
        eval_targets = [item["target"] for item in preds_for_eval]
        regression_predictions = [item.get("personality_regression") for item in preds_for_eval]

        task_report = build_task_statistics(
            targets=eval_targets,
            answers=all_answers,
            emotional_label_dict=emotional_label_dict,
            dataset=args.dataset,
            speaker_loss_weight=args.speaker_loss_weight,
            personality_loss_weight=args.personality_loss_weight,
        )
        task_report = _attach_personality_regression_report(
            task_report,
            targets=eval_targets,
            regression_predictions=regression_predictions,
            args=args,
        )
        task_report = _attach_personality_pair_report(
            task_report,
            records=preds_for_eval,
            args=args,
        )
        task_report = _attach_generated_continuous_pair_report(
            task_report,
            records=preds_for_eval,
            args=args,
        )
        task_report = _attach_run_modality_config(task_report, args)
        if is_main_process and task_report.get('emotion') is not None:
            eval_score_list.append(task_report['emotion']['score'])

        if is_main_process:
            _write_eval_outputs(preds_for_eval_path, preds_for_eval, task_report=task_report)

    if is_main_process:
        for i in eval_score_list:
            print(i)

    _safe_dist_barrier(tag="program_end")
