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
    StoppingCriteria,
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
import logging
import math
import os
import pickle
import random
import re
import time
import warnings
from contextlib import nullcontext
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
import numpy as np
from data_utils.data_utils import *
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim import AdamW, Adam
from typing import List, Dict
from peft import LoraConfig, get_peft_model
import argparse
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
# from colossalai.nn.optimizer import HybridAdam
# from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
# from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

# packages for ERC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score


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
        f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))


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

    return (answer, {'task_type': task_type, 'parse_status': 'raw_output'}) if return_info else answer


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
    if target.startswith('Speaker_'):
        return 'speaker'
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
    if task in ('personality_continuous', 'personality_discrete'):
        return float(personality_loss_weight)
    return 1.0


def compute_task_weighted_loss(outputs, labels, tokenizer, speaker_loss_weight, personality_loss_weight):
    # Recompute token-level CE to support per-sample task weights.
    # In multimodal prefix mode, logits sequence can be one step longer than raw text labels.
    logits = outputs.logits
    if logits.size(1) != labels.size(1):
        if logits.size(1) == labels.size(1) + 1:
            ignore_prefix = labels.new_full((labels.size(0), 1), -100)
            labels = torch.cat([ignore_prefix, labels], dim=1)
        elif labels.size(1) > logits.size(1):
            labels = labels[:, :logits.size(1)]
        else:
            logits = logits[:, :labels.size(1), :]

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
    valid_count = valid_mask.sum(dim=1).clamp(min=1).float()
    sample_loss = (token_loss * valid_mask).sum(dim=1) / valid_count

    target_texts = _decode_target_texts_from_labels(labels, tokenizer)
    sample_weights = [
        _task_weight_by_target_text(text, speaker_loss_weight, personality_loss_weight)
        for text in target_texts
    ]
    weights = sample_loss.new_tensor(sample_weights)
    weighted_loss = (sample_loss * weights).sum() / weights.sum().clamp(min=1e-6)
    return weighted_loss


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


def _get_contrastive_projection(model):
    if hasattr(model, "contrastive_projection"):
        return model.contrastive_projection
    module = getattr(model, "module", None)
    if module is not None and hasattr(module, "contrastive_projection"):
        return module.contrastive_projection
    return None


def _align_mask_to_hidden(mask, hidden):
    if mask is None:
        return None
    if mask.size(1) == hidden.size(1):
        return mask
    common = min(mask.size(1), hidden.size(1))
    return mask[:, -common:]


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
            if isinstance(parsed, list) and len(parsed) >= 5:
                vals = [float(parsed[i]) for i in range(5)]
                if not _is_valid_personality_cont_values(vals):
                    continue
                return dict(zip(PERSONALITY_TRAITS, vals))
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(cand)
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
        },
        'emotion': None,
        'speaker': None,
        'personality_continuous': None,
        'personality_discrete': None,
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
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

    ckpt_dir = checkpoint_dir
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

    if tag is None:
        return load_state_dict_from_zero_checkpoint(model, ckpt_dir)
    if is_main_process:
        print(f"[INFO] Loading DeepSpeed checkpoint root='{ckpt_dir}', tag='{tag}'.")
    return load_state_dict_from_zero_checkpoint(model, ckpt_dir, tag=tag)


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
    help="Generation repetition penalty used during evaluation and OPSD student generation.",
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
    help='Audio source: precomputed_pkl (Qwen native mel cache) | qwen_runtime_extract'
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
    "--text_enable",
    default='True',
    help='Compatibility with main_new.py modality flags. OPSD currently keeps text behavior controlled by audio_input_mode.'
)
parser.add_argument(
    "--video_enable",
    default='False',
    help='Compatibility with main_new.py modality flags. OPSD currently ignores video input.'
)
parser.add_argument(
    "--video_manifest_jsonl",
    default=None,
    type=str,
    help='Compatibility argument; ignored by OPSD.'
)
parser.add_argument(
    "--video_fps",
    default=2.0,
    type=float,
    help='Compatibility argument; ignored by OPSD.'
)
parser.add_argument(
    "--video_resized_height",
    default=336,
    type=int,
    help='Compatibility argument; ignored by OPSD.'
)
parser.add_argument(
    "--video_resized_width",
    default=0,
    type=int,
    help='Compatibility argument; ignored by OPSD.'
)
parser.add_argument(
    "--video_use_audio_in_video",
    default='False',
    help='Compatibility argument; ignored by OPSD.'
)
parser.add_argument(
    "--opsd_max_completion_tokens",
    default=32,
    type=int,
    help="Maximum on-policy completion tokens generated by the student during OPSD training.",
)
parser.add_argument(
    "--opsd_top_k_loss",
    default=256,
    type=int,
    help="Restrict OPSD JSD to teacher top-k tokens. Use 0 for the full vocabulary.",
)
parser.add_argument(
    "--opsd_jsd_token_clip",
    default=0.05,
    type=float,
    help="Clip per-token OPSD divergence before reduction. Use 0 to disable clipping.",
)
parser.add_argument(
    "--opsd_beta",
    default=0.5,
    type=float,
    help="Generalized JSD interpolation coefficient.",
)
parser.add_argument(
    "--opsd_temperature",
    default=1.0,
    type=float,
    help="Temperature used for OPSD generation and distribution matching.",
)
parser.add_argument(
    "--opsd_sft_loss_weight",
    default=0.0,
    type=float,
    help="Optional supervised CE loss weight mixed with OPSD loss.",
)
parser.add_argument(
    "--opsd_answer_token_weight",
    default=1.0,
    type=float,
    help="Multiplier for OPSD loss on tokens inside <answer>...</answer>. Use >1 to emphasize final labels while still distilling reasoning tokens.",
)
parser.add_argument(
    "--opsd_fixed_teacher",
    default='False',
    help="If True and LoRA is enabled, disable adapters for teacher-side forward passes.",
)
parser.add_argument(
    "--opsd_do_sample",
    default='False',
    help="Whether OPSD on-policy generation should sample. False uses greedy generation for stability.",
)
parser.add_argument(
    "--opsd_sft_target_mode",
    default="full_target",
    choices=["full_target", "canonical_answer"],
    help="Target used for the supervised CE anchor in OPSD training.",
)
parser.add_argument(
    "--opsd_teacher_target_mode",
    default="canonical_answer",
    choices=["canonical_answer", "full_target"],
    help="Privileged target shown to the OPSD teacher prompt.",
)
parser.add_argument(
    "--opsd_stop_after_answer",
    default='False',
    help="If True, stop OPSD on-policy generation once all samples have generated an </answer> tag.",
)
parser.add_argument(
    "--opsd_forced_completion_prefix_mode",
    default="none",
    choices=["none", "reasoning_xml"],
    help="Optionally force the on-policy completion to start from a short prefix. reasoning_xml uses '<reasoning>\\n'.",
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
args.experience_bank_mode = str(args.experience_bank_mode or 'none').strip().lower()
if args.experience_bank_mode not in ("none", "student_prompt", "opcd_teacher"):
    raise ValueError("Invalid --experience_bank_mode. Use: none | student_prompt | opcd_teacher")
args.opcd_teacher_include_gold_context = _is_true_flag(args.opcd_teacher_include_gold_context)
args.generate_inference_reasoning = _is_true_flag(args.generate_inference_reasoning)

if args.audio_enable == 'True':
    args.audio_enable = True
else:
    args.audio_enable = False

args.audio_input_mode = str(args.audio_input_mode).strip().lower()
if args.audio_input_mode not in ("text_audio", "audio_only", "text_only"):
    raise ValueError("Invalid --audio_input_mode. Use: text_audio | audio_only | text_only")

if args.opsd_fixed_teacher == 'True':
    args.opsd_fixed_teacher = True
else:
    args.opsd_fixed_teacher = False

if args.opsd_do_sample == 'True':
    args.opsd_do_sample = True
else:
    args.opsd_do_sample = False

if args.opsd_stop_after_answer == 'True':
    args.opsd_stop_after_answer = True
else:
    args.opsd_stop_after_answer = False

args.opsd_sft_target_mode = str(args.opsd_sft_target_mode).strip().lower()
if args.opsd_sft_target_mode not in ("full_target", "canonical_answer"):
    raise ValueError("Invalid --opsd_sft_target_mode. Use: full_target | canonical_answer")

args.opsd_teacher_target_mode = str(args.opsd_teacher_target_mode).strip().lower()
if args.opsd_teacher_target_mode not in ("canonical_answer", "full_target"):
    raise ValueError("Invalid --opsd_teacher_target_mode. Use: canonical_answer | full_target")
args.opsd_forced_completion_prefix_mode = str(args.opsd_forced_completion_prefix_mode or "none").strip().lower()
if args.opsd_forced_completion_prefix_mode not in ("none", "reasoning_xml"):
    raise ValueError("Invalid --opsd_forced_completion_prefix_mode. Use: none | reasoning_xml")

args.audio_embedding_source = str(args.audio_embedding_source).strip().lower()
if args.audio_embedding_source not in ("precomputed_pkl", "qwen_runtime_extract"):
    raise ValueError("Invalid --audio_embedding_source. Use: precomputed_pkl | qwen_runtime_extract")

args.mixed_precision = str(args.mixed_precision).strip().lower()
if args.mixed_precision not in ("fp16", "bf16", "fp32"):
    raise ValueError("Invalid --mixed_precision. Use: fp16 | bf16 | fp32")

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
    "zero_shot": args.zero_shot,
    "statistic_mode": args.statistic_mode,
    "mode": args.mode,
    "emotion_prediction": args.emotion_prediction,
    "beta": args.beta,
    "theta": args.theta,
    "speaker_loss_weight": args.speaker_loss_weight,
    "personality_loss_weight": args.personality_loss_weight,
    "enable_contrastive": args.enable_contrastive,
    "contrastive_weight": args.contrastive_weight,
    "contrastive_temperature": args.contrastive_temperature,
    "contrastive_proj_dim": args.contrastive_proj_dim,
    "contrastive_mode": args.contrastive_mode,
    "contrastive_gather_world": args.contrastive_gather_world,
    "personality_contrastive_alpha": args.personality_contrastive_alpha,
    "experience_bank_mode": args.experience_bank_mode,
    "opcd_teacher_include_gold_context": args.opcd_teacher_include_gold_context,
    "generate_inference_reasoning": args.generate_inference_reasoning,
    "inference_reasoning_max_samples": args.inference_reasoning_max_samples,
    "inference_reasoning_max_new_tokens": args.inference_reasoning_max_new_tokens,
    "audio_enable": args.audio_enable,
    "audio_embedding_source": args.audio_embedding_source,
    "audio_feature_pkl": args.audio_feature_pkl,
    "audio_iemocap_root": args.audio_iemocap_root,
    "audio_iemocap_pkl": args.audio_iemocap_pkl,
    "audio_qwen_processor_path": args.audio_qwen_processor_path,
    "audio_feature_dim": args.audio_feature_dim,
    "audio_context_window": args.audio_context_window,
    "audio_gru_hidden_size": args.audio_gru_hidden_size,
    "audio_gru_layers": args.audio_gru_layers,
    "audio_gru_dropout": args.audio_gru_dropout,
    "audio_input_mode": args.audio_input_mode,
    "opsd_max_completion_tokens": args.opsd_max_completion_tokens,
    "opsd_top_k_loss": args.opsd_top_k_loss,
    "opsd_jsd_token_clip": args.opsd_jsd_token_clip,
    "opsd_beta": args.opsd_beta,
    "opsd_temperature": args.opsd_temperature,
    "opsd_sft_loss_weight": args.opsd_sft_loss_weight,
    "opsd_answer_token_weight": args.opsd_answer_token_weight,
    "opsd_fixed_teacher": args.opsd_fixed_teacher,
    "opsd_do_sample": args.opsd_do_sample,
    "opsd_sft_target_mode": args.opsd_sft_target_mode,
    "opsd_teacher_target_mode": args.opsd_teacher_target_mode,
    "opsd_stop_after_answer": args.opsd_stop_after_answer,
    "opsd_forced_completion_prefix_mode": args.opsd_forced_completion_prefix_mode,
    "gradient_checkpointing": args.gradient_checkpointing,
    "data_percent": args.data_percent
}
args = ModelArgs()
# pdb.set_trace()

args.update(model_args)
# pdb.set_trace()

print(args)
if is_main_process:
    print(
        f"[Contrastive] enabled={args.enable_contrastive}, mode={args.contrastive_mode}, "
        f"weight={args.contrastive_weight}, temperature={args.contrastive_temperature}, "
        f"proj_dim={args.contrastive_proj_dim}, gather_world={args.contrastive_gather_world}, "
        f"personality_alpha={args.personality_contrastive_alpha}"
    )
    print(
        f"[ExperienceBank] mode={args.experience_bank_mode}, "
        f"opcd_gold_context={args.opcd_teacher_include_gold_context}, "
        f"generate_inference_reasoning={args.generate_inference_reasoning}, "
        f"reasoning_max_samples={args.inference_reasoning_max_samples}"
    )
    print(
        f"[OPSD] enabled: max_completion_tokens={args.opsd_max_completion_tokens}, "
        f"top_k_loss={args.opsd_top_k_loss}, beta={args.opsd_beta}, "
        f"temperature={args.opsd_temperature}, sft_loss_weight={args.opsd_sft_loss_weight}, "
        f"answer_token_weight={args.opsd_answer_token_weight}, "
        f"fixed_teacher={args.opsd_fixed_teacher}, do_sample={args.opsd_do_sample}, "
        f"sft_target_mode={args.opsd_sft_target_mode}, "
        f"teacher_target_mode={args.opsd_teacher_target_mode}, "
        f"stop_after_answer={args.opsd_stop_after_answer}, "
        f"forced_completion_prefix_mode={args.opsd_forced_completion_prefix_mode}"
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
    }
    return {
        key: value
        for key, value in batch.items()
        if key not in excluded_keys
    }


def _target_for_opsd_mode(target, mode):
    mode = str(mode or "full_target").strip().lower()
    if mode == "canonical_answer":
        return _canonical_target_text(target)
    if mode == "full_target":
        return str(target).strip()
    raise ValueError(f"Invalid OPSD target mode: {mode}")


class OPSDSeq2SeqCollator(object):
    """Construct prompt-only student/teacher batches from existing InstructERC records."""

    def __init__(self, args, tokenizer):
        self.args = args
        self.sft_target_mode = str(getattr(args, "opsd_sft_target_mode", "full_target")).strip().lower()
        self.teacher_target_mode = str(getattr(args, "opsd_teacher_target_mode", "canonical_answer")).strip().lower()
        self.student_collator = Seq2SeqCollator(args, tokenizer, mode="dev")
        self.teacher_collator = Seq2SeqCollator(args, tokenizer, mode="dev")
        self.sft_collator = Seq2SeqCollator(args, tokenizer, mode="train")

    @staticmethod
    def _unpack_item(item):
        text = item[0]
        target = item[1] if len(item) > 1 else ""
        sid = item[2] if len(item) > 2 else None
        meta = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        return text, target, sid, meta

    def _build_teacher_prompt(self, text, target, meta=None):
        teacher_target = _target_for_opsd_mode(target, self.teacher_target_mode)
        if self.teacher_target_mode == "canonical_answer":
            privileged_block = f"<answer>\n{teacher_target}\n</answer>"
        else:
            privileged_block = teacher_target
        teacher_reasoning = ""
        experience_bank_context = ""
        opcd_gold_context = None
        if isinstance(meta, dict):
            teacher_reasoning = str(meta.get("teacher_reasoning") or "").strip()
            experience_bank_context = str(meta.get("experience_bank_context") or "").strip()
            opcd_gold_context = meta.get("opcd_gold_context")
        teacher_reasoning_block = ""
        if teacher_reasoning:
            teacher_reasoning_block = (
                "\nThe API-generated natural-language explanation is available only as teacher-side context:\n"
                f"<teacher_reasoning>\n{teacher_reasoning}\n</teacher_reasoning>\n"
                "Do not require the student to copy this explanation."
            )
        experience_bank_block = ""
        if str(getattr(self.args, "experience_bank_mode", "none")) == "opcd_teacher" and experience_bank_context:
            experience_bank_block = (
                "\nTeacher-only transferable experience bank:\n"
                f"<experience_bank>\n{experience_bank_context}\n</experience_bank>\n"
                "Use these probabilistic tendencies as context, not as deterministic rules.\n"
            )
        opcd_gold_context_block = ""
        if (
            str(getattr(self.args, "experience_bank_mode", "none")) == "opcd_teacher"
            and _is_true_flag(getattr(self.args, "opcd_teacher_include_gold_context", "False"))
            and opcd_gold_context is not None
        ):
            try:
                gold_context_text = json.dumps(opcd_gold_context, ensure_ascii=False, indent=2)
            except Exception:
                gold_context_text = str(opcd_gold_context)
            opcd_gold_context_block = (
                "\nTeacher-only gold-side context for distillation:\n"
                f"<gold_context>\n{gold_context_text}\n</gold_context>\n"
                "This context is not visible to the student prompt or inference prompt.\n"
            )
        return (
            f"{text}\n\n"
            "[Privileged OPSD training information]\n"
            f"{experience_bank_block}"
            f"{opcd_gold_context_block}"
            "The gold reference response for this training example is:\n"
            f"{privileged_block}\n"
            f"{teacher_reasoning_block}\n"
            "Use this reference only for teacher-side supervision of the continuation."
        )

    def _build_sft_item(self, item):
        text, target, sid, meta = self._unpack_item(item)
        return [text, _target_for_opsd_mode(target, self.sft_target_mode), sid, meta]

    def __call__(self, batch):
        if self.args.emotion_prediction:
            raise ValueError("OPSD does not support emotion_prediction=True in this first version.")

        student_items = []
        teacher_items = []
        for item in batch:
            text, target, sid, meta = self._unpack_item(item)
            student_items.append([text, target, sid, meta])
            teacher_items.append([self._build_teacher_prompt(text, target, meta=meta), target, sid, meta])

        result = {
            "student": self.student_collator(student_items),
            "teacher": self.teacher_collator(teacher_items),
        }
        if (
            float(getattr(self.args, "opsd_sft_loss_weight", 0.0)) > 0.0
            or _is_true_flag(getattr(self.args, "enable_contrastive", "False"))
        ):
            sft_items = [self._build_sft_item(item) for item in batch]
            result["sft"] = self.sft_collator(sft_items)
        return result


def _move_nested_to_device(obj, target_device):
    if torch.is_tensor(obj):
        return obj.to(target_device)
    if isinstance(obj, dict):
        return {
            key: _move_nested_to_device(value, target_device)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_move_nested_to_device(value, target_device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_nested_to_device(value, target_device) for value in obj)
    return obj


def _model_forward_inputs(batch):
    excluded = {
        "labels",
        "type_token_ids",
        "token_type_ids",
        "contrastive_label_ids",
        "contrastive_task_mask",
        "contrastive_personality",
        "contrastive_personality_mask",
    }
    return {
        key: value
        for key, value in batch.items()
        if key not in excluded
    }


def _iter_model_candidates(model):
    seen = set()
    queue = [model, getattr(model, "module", None)]
    while queue:
        module = queue.pop(0)
        if module is None:
            continue
        ident = id(module)
        if ident in seen:
            continue
        seen.add(ident)
        yield module
        queue.extend([
            getattr(module, "base_model", None),
            getattr(getattr(module, "base_model", None), "model", None),
            getattr(module, "model", None),
        ])


def _find_model_with_attrs(model, attrs):
    for module in _iter_model_candidates(model):
        if all(hasattr(module, attr) for attr in attrs):
            return module
    return None


def _reset_qwen_rope_deltas(model):
    for module in _iter_model_candidates(model):
        if hasattr(module, "rope_deltas"):
            try:
                module.rope_deltas = None
            except Exception:
                pass


def _is_qwen_native_audio_batch(batch):
    return "input_features" in batch and "feature_attention_mask" in batch and "audio_feature_lengths" in batch


def _prepare_qwen_full_forward_batch(model, batch):
    if not _is_qwen_native_audio_batch(batch):
        return batch

    qwen_model = _find_model_with_attrs(
        model,
        ("get_input_embeddings", "get_audio_features", "get_placeholder_mask"),
    )
    if qwen_model is None:
        return batch

    input_ids = batch["input_ids"]
    with torch.no_grad():
        inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        audio_features = qwen_model.get_audio_features(
            batch["input_features"],
            feature_attention_mask=batch["feature_attention_mask"],
            audio_feature_lengths=batch["audio_feature_lengths"],
            return_dict=True,
        ).last_hidden_state
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        _, _, audio_mask = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
        expected = int(audio_mask.sum().item())
        actual = int(audio_features.numel())
        if expected != actual:
            raise ValueError(
                "Qwen OPSD audio/token mismatch after appending generated completion: "
                f"audio_token_elements={expected}, audio_feature_elements={actual}, "
                f"input_shape={tuple(input_ids.shape)}, feature_shape={tuple(batch['input_features'].shape)}"
            )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features).detach()

    prepared = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": batch["attention_mask"],
        "feature_attention_mask": batch["feature_attention_mask"],
        "audio_feature_lengths": batch["audio_feature_lengths"],
    }
    if "labels" in batch:
        prepared["labels"] = batch["labels"]
    return prepared


def _extract_completion_ids(prompt_batch, generated_outputs, tokenizer, max_new_tokens, stop_sequences=None):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    prompt_padded_len = int(prompt_batch["input_ids"].size(1))
    completion_ids = []
    for idx in range(generated_outputs.size(0)):
        completion = generated_outputs[idx, prompt_padded_len:]
        if pad_id is not None:
            completion = completion[completion.ne(pad_id)]
        completion = completion[:int(max_new_tokens)]
        completion = _truncate_completion_at_stop(completion, stop_sequences)
        if completion.numel() == 0:
            completion = torch.tensor([eos_id], dtype=torch.long, device=generated_outputs.device)
        completion_ids.append(completion.detach())
    return completion_ids


def _opsd_completion_prefix_text(args):
    mode = str(getattr(args, "opsd_forced_completion_prefix_mode", "none") or "none").strip().lower()
    if mode == "reasoning_xml":
        return "<reasoning>\n"
    return ""


def _build_forced_completion_prefix_ids(batch_size, tokenizer, device, args):
    prefix_text = _opsd_completion_prefix_text(args)
    if not prefix_text:
        return None
    encoded = tokenizer(prefix_text, add_special_tokens=False)
    ids = encoded.get("input_ids", [])
    if not ids:
        return None
    prefix = torch.tensor(ids, dtype=torch.long, device=device)
    return [prefix.clone() for _ in range(int(batch_size))]


def _concat_prefix_and_completion(prefix_ids, completion_ids, tokenizer, max_completion_tokens=None, stop_sequences=None):
    if not prefix_ids:
        return completion_ids
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    merged = []
    max_completion_tokens = int(max_completion_tokens) if max_completion_tokens is not None else None
    for prefix, completion in zip(prefix_ids, completion_ids):
        if completion is None or completion.numel() == 0:
            completion = torch.tensor([eos_id], dtype=torch.long, device=prefix.device)
        combined = torch.cat([prefix.to(completion.device).to(torch.long), completion.to(torch.long)], dim=0)
        if max_completion_tokens is not None:
            combined = combined[:max_completion_tokens]
        combined = _truncate_completion_at_stop(combined, stop_sequences)
        if combined.numel() == 0:
            combined = torch.tensor([eos_id], dtype=torch.long, device=prefix.device)
        merged.append(combined.detach())
    return merged


def _build_full_batch_from_prompt(prompt_batch, completion_ids, tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    padding_side = getattr(tokenizer, "padding_side", "left")

    input_rows = []
    label_rows = []
    attention_rows = []
    for idx, completion in enumerate(completion_ids):
        prompt_ids = prompt_batch["input_ids"][idx]
        prompt_mask = prompt_batch["attention_mask"][idx].bool()
        prompt_tokens = prompt_ids[prompt_mask].to(torch.long)
        completion = completion.to(prompt_tokens.device).to(torch.long)
        full_ids = torch.cat([prompt_tokens, completion], dim=0)
        labels = torch.cat([torch.full_like(prompt_tokens, -100), completion], dim=0)
        input_rows.append(full_ids)
        label_rows.append(labels)
        attention_rows.append(torch.ones_like(full_ids, dtype=torch.long))

    max_len = max(int(ids.numel()) for ids in input_rows)
    padded_inputs = []
    padded_labels = []
    padded_attention = []
    for ids, labels, mask in zip(input_rows, label_rows, attention_rows):
        pad_len = max_len - int(ids.numel())
        if padding_side == "left":
            padded_inputs.append(torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long, device=ids.device), ids], dim=0))
            padded_labels.append(torch.cat([torch.full((pad_len,), -100, dtype=torch.long, device=ids.device), labels], dim=0))
            padded_attention.append(torch.cat([torch.zeros((pad_len,), dtype=torch.long, device=ids.device), mask], dim=0))
        else:
            padded_inputs.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long, device=ids.device)], dim=0))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long, device=ids.device)], dim=0))
            padded_attention.append(torch.cat([mask, torch.zeros((pad_len,), dtype=torch.long, device=ids.device)], dim=0))

    full_batch = {
        "input_ids": torch.stack(padded_inputs, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "attention_mask": torch.stack(padded_attention, dim=0),
    }
    for key in ("input_features", "feature_attention_mask", "audio_feature_lengths", "audio_features", "audio_mask"):
        if key in prompt_batch:
            full_batch[key] = prompt_batch[key]
    return full_batch


def _select_completion_logits(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    if int(mask.sum().item()) == 0:
        return None
    return shift_logits[mask]


def _opsd_generalized_jsd(student_logits, teacher_logits, beta=0.5, temperature=1.0, top_k=256, token_clip=0.05, token_weights=None):
    temperature = max(float(temperature), 1e-6)
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if int(top_k) > 0 and int(top_k) < teacher_logits.size(-1):
        _, top_indices = torch.topk(teacher_logits, k=int(top_k), dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=top_indices)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_indices)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    beta_t = torch.tensor(float(beta), dtype=student_log_probs.dtype, device=student_log_probs.device)
    if float(beta) <= 0.0:
        per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif float(beta) >= 1.0:
        per_vocab = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log1p(-beta_t),
                teacher_log_probs + torch.log(beta_t),
            ]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        per_vocab = beta_t * kl_teacher + (1.0 - beta_t) * kl_student

    per_token = per_vocab.sum(dim=-1)
    if float(token_clip) > 0.0:
        per_token = per_token.clamp(max=float(token_clip))
    if token_weights is not None:
        token_weights = token_weights.to(device=per_token.device, dtype=per_token.dtype)
        token_weights = token_weights[:per_token.size(0)].clamp_min(0.0)
        denom = token_weights.sum().clamp_min(1e-6)
        return (per_token * token_weights).sum() / denom
    return per_token.mean()


def _completion_answer_token_weights(completion_ids, tokenizer, answer_token_weight=1.0):
    weight = max(float(answer_token_weight), 1.0)
    answer_re = re.compile(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)
    weights = []
    for completion in completion_ids:
        ids = completion.detach().cpu().tolist()
        token_weights = [1.0] * len(ids)
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        match = answer_re.search(text)
        if match is None:
            # Canonical answer completions have no tags; treat short completions as answer-only.
            if len(ids) <= 8:
                token_weights = [weight] * len(ids)
            weights.extend(token_weights)
            continue
        body_start, body_end = match.start(1), match.end(1)
        prefix_lengths = [0]
        for i in range(1, len(ids) + 1):
            prefix = tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            prefix_lengths.append(len(prefix))
        for i in range(len(ids)):
            tok_start, tok_end = prefix_lengths[i], prefix_lengths[i + 1]
            tok_mid = (tok_start + tok_end) / 2.0
            if body_start <= tok_mid <= body_end:
                token_weights[i] = weight
        weights.extend(token_weights)
    if not weights:
        return None
    return torch.tensor(weights, dtype=torch.float32)


def _adapter_disabled_context(model):
    module = getattr(model, "module", model)
    if hasattr(module, "disable_adapter"):
        return module.disable_adapter()
    base_model = getattr(module, "base_model", None)
    if hasattr(base_model, "disable_adapter"):
        return base_model.disable_adapter()
    return nullcontext()


class _AllSequencesEndWithAnyStop(StoppingCriteria):
    def __init__(self, stop_sequences, start_index=0):
        self.stop_sequences = [
            torch.tensor(seq, dtype=torch.long)
            for seq in stop_sequences
            if seq is not None and len(seq) > 0
        ]
        self.start_index = max(0, int(start_index))

    def __call__(self, input_ids, scores, **kwargs):
        if len(self.stop_sequences) == 0 or input_ids is None or input_ids.numel() == 0:
            return False
        for row in input_ids:
            generated = row[self.start_index:]
            row_done = False
            for seq in self.stop_sequences:
                seq = seq.to(generated.device)
                if generated.numel() < seq.numel():
                    continue
                if torch.equal(generated[-seq.numel():], seq):
                    row_done = True
                    break
            if not row_done:
                return False
        return True


def _answer_stop_sequences(tokenizer):
    stop_sequences = []
    for text in ("</answer>", "</answer>\n", "\n</answer>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            stop_sequences.append(ids)
    return stop_sequences


def _answer_stopping_criteria(tokenizer, start_index=0):
    stop_sequences = _answer_stop_sequences(tokenizer)
    if not stop_sequences:
        return None
    return StoppingCriteriaList([_AllSequencesEndWithAnyStop(stop_sequences, start_index=start_index)])


def _truncate_completion_at_stop(completion, stop_sequences):
    if completion is None or completion.numel() == 0 or not stop_sequences:
        return completion
    earliest_end = None
    for seq in stop_sequences:
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=completion.device)
        if completion.numel() < seq_tensor.numel():
            continue
        for start in range(0, int(completion.numel() - seq_tensor.numel()) + 1):
            if torch.equal(completion[start:start + seq_tensor.numel()], seq_tensor):
                end = start + int(seq_tensor.numel())
                if earliest_end is None or end < earliest_end:
                    earliest_end = end
                break
    if earliest_end is not None:
        return completion[:earliest_end]
    return completion


def compute_opsd_training_loss(model, batch, tokenizer, args):
    student_prompt = batch["student"]
    teacher_prompt = batch["teacher"]
    forced_prefix_ids = _build_forced_completion_prefix_ids(
        batch_size=int(student_prompt["input_ids"].size(0)),
        tokenizer=tokenizer,
        device=student_prompt["input_ids"].device,
        args=args,
    )
    generation_prompt = student_prompt
    generation_max_new_tokens = int(args.opsd_max_completion_tokens)
    if forced_prefix_ids:
        generation_prompt = _build_full_batch_from_prompt(student_prompt, forced_prefix_ids, tokenizer)
        max_prefix_len = max(int(prefix.numel()) for prefix in forced_prefix_ids)
        generation_max_new_tokens = max(1, int(args.opsd_max_completion_tokens) - max_prefix_len)

    generation_kwargs = {
        "num_beams": 1,
        "do_sample": bool(args.opsd_do_sample),
        "max_new_tokens": int(generation_max_new_tokens),
        "repetition_penalty": float(args.repetition_penalty),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": False,
    }
    answer_stop_sequences = None
    if bool(getattr(args, "opsd_stop_after_answer", False)):
        answer_stop_sequences = _answer_stop_sequences(tokenizer)
        stopping_criteria = _answer_stopping_criteria(
            tokenizer,
            start_index=int(generation_prompt["input_ids"].size(1)),
        )
        if stopping_criteria is not None:
            generation_kwargs["stopping_criteria"] = stopping_criteria
    if bool(args.opsd_do_sample):
        generation_kwargs["temperature"] = max(float(args.opsd_temperature), 1e-6)
    if bool(args.opsd_do_sample) and args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if bool(args.opsd_do_sample) and args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            generated_outputs = model.generate(
                **_model_forward_inputs(generation_prompt),
                **generation_kwargs,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "OPSD student generation failed. This happens before the teacher forward; "
            "for Qwen2.5-Omni audio it usually indicates a generate/cache/audio-position issue."
        ) from exc
    _reset_qwen_rope_deltas(model)
    if was_training:
        model.train()

    completion_ids = _extract_completion_ids(
        prompt_batch=generation_prompt,
        generated_outputs=generated_outputs,
        tokenizer=tokenizer,
        max_new_tokens=generation_max_new_tokens,
        stop_sequences=answer_stop_sequences,
    )
    if forced_prefix_ids:
        completion_ids = _concat_prefix_and_completion(
            prefix_ids=forced_prefix_ids,
            completion_ids=completion_ids,
            tokenizer=tokenizer,
            max_completion_tokens=args.opsd_max_completion_tokens,
            stop_sequences=answer_stop_sequences,
        )

    student_full = _prepare_qwen_full_forward_batch(
        model,
        _build_full_batch_from_prompt(student_prompt, completion_ids, tokenizer),
    )
    teacher_full = _prepare_qwen_full_forward_batch(
        model,
        _build_full_batch_from_prompt(teacher_prompt, completion_ids, tokenizer),
    )

    student_inputs = _model_forward_inputs(student_full)
    student_inputs["use_cache"] = False
    try:
        student_outputs = model(**student_inputs)
    except RuntimeError as exc:
        raise RuntimeError(
            "OPSD student full forward failed after appending generated tokens. "
            "The Qwen audio features have already been merged into inputs_embeds at this point."
        ) from exc
    teacher_context = _adapter_disabled_context(model) if bool(args.opsd_fixed_teacher) else nullcontext()
    with torch.no_grad(), teacher_context:
        teacher_inputs = _model_forward_inputs(teacher_full)
        teacher_inputs["use_cache"] = False
        try:
            teacher_outputs = model(**teacher_inputs)
        except RuntimeError as exc:
            raise RuntimeError(
                "OPSD teacher full forward failed after appending the student's generated tokens."
            ) from exc

    student_logits = _select_completion_logits(student_outputs.logits, student_full["labels"])
    teacher_logits = _select_completion_logits(teacher_outputs.logits, teacher_full["labels"])
    if student_logits is None or teacher_logits is None:
        raise ValueError("OPSD generated no trainable completion tokens. Increase --opsd_max_completion_tokens.")

    num_tokens = min(student_logits.size(0), teacher_logits.size(0))
    student_logits = student_logits[:num_tokens]
    teacher_logits = teacher_logits[:num_tokens]
    opsd_token_weights = None
    if float(getattr(args, "opsd_answer_token_weight", 1.0)) != 1.0:
        opsd_token_weights = _completion_answer_token_weights(
            completion_ids=completion_ids,
            tokenizer=tokenizer,
            answer_token_weight=args.opsd_answer_token_weight,
        )
        if opsd_token_weights is not None:
            opsd_token_weights = opsd_token_weights[:num_tokens]

    opsd_loss = _opsd_generalized_jsd(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        beta=args.opsd_beta,
        temperature=args.opsd_temperature,
        top_k=args.opsd_top_k_loss,
        token_clip=args.opsd_jsd_token_clip,
        token_weights=opsd_token_weights,
    )

    loss = opsd_loss
    sft_loss_value = None
    contrastive_loss_value = None
    need_sft_forward = (
        "sft" in batch
        and (
            float(args.opsd_sft_loss_weight) > 0.0
            or _is_true_flag(getattr(args, "enable_contrastive", "False"))
        )
    )
    if need_sft_forward:
        sft_inputs = _get_input_dict(batch["sft"])
        sft_model_inputs = dict(sft_inputs)
        if args.enable_contrastive:
            sft_model_inputs["output_hidden_states"] = True
        sft_outputs = model(**sft_model_inputs)
        if float(args.opsd_sft_loss_weight) > 0.0:
            sft_loss = compute_task_weighted_loss(
                outputs=sft_outputs,
                labels=sft_inputs["labels"],
                tokenizer=tokenizer,
                speaker_loss_weight=args.speaker_loss_weight,
                personality_loss_weight=args.personality_loss_weight,
            )
            loss = loss + float(args.opsd_sft_loss_weight) * sft_loss
            sft_loss_value = float(sft_loss.detach().item())
        contrastive_loss = _compute_contrastive_loss(
            model=model,
            outputs=sft_outputs,
            batch=batch["sft"],
            args=args,
        )
        if contrastive_loss is not None:
            loss = loss + float(args.contrastive_weight) * contrastive_loss
            contrastive_loss_value = float(contrastive_loss.detach().item())

    stats = {
        "opsd_loss": float(opsd_loss.detach().item()),
        "sft_loss": sft_loss_value,
        "contrastive_loss": contrastive_loss_value,
        "gen_tokens": int(sum(comp.numel() for comp in completion_ids)),
    }
    return loss, stats


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


def _build_eval_loader(dev_dataset, dev_collator, eval_batch_size):
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


def _gather_eval_records(local_records):
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        rank = dist.get_rank()
        if rank == 0:
            gathered = [None for _ in range(world_size)]
            dist.gather_object(local_records, object_gather_list=gathered, dst=0)
            merged = []
            for part in gathered:
                if part:
                    merged.extend(part)
            return merged
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
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            output_hidden_states=True,
            torch_dtype=MODEL_TORCH_DTYPE,
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

if getattr(tokenizer, "pad_token_id", None) is None:
    raise ValueError("Tokenizer pad_token_id is None after fallback setup.")

# Set model.config.pad_token_id to avoid warnings during generation
if hasattr(model, "config") and hasattr(model.config, "pad_token_id"):
    model.config.pad_token_id = tokenizer.pad_token_id

if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

if args.model_type == "decoder":
    tokenizer.padding_side = "left"

if args.lora:
    model = get_peft_model(model, lora_config)

model = attach_contrastive_projection_head(model, args)

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

if args.audio_enable:
    if args.audio_embedding_source in ('precomputed_pkl', 'qwen_runtime_extract'):
        if str(getattr(config, "model_type", "")).lower() != "qwen2_5_omni":
            raise ValueError(f"audio_embedding_source={args.audio_embedding_source} requires a Qwen2.5-Omni-compatible model.")
        if args.audio_embedding_source == 'precomputed_pkl' and (
            args.audio_feature_pkl is None or not os.path.exists(args.audio_feature_pkl)
        ):
            raise FileNotFoundError(f"Qwen native mel cache not found: {args.audio_feature_pkl}")
        # Both sources use Qwen native audio inputs:
        # input_features, feature_attention_mask, and audio_feature_lengths.
    else:
        raise ValueError(f"Unsupported audio_embedding_source: {args.audio_embedding_source}")


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
    train_collator = OPSDSeq2SeqCollator(args, tokenizer)
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
if args.do_train:
    natural_t_total = math.ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
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
    if is_main_process:
        print(f"[INFO] deepspeed.initialize done in {time.time() - ds_init_start:.1f}s; train batches per epoch: {len(train_dataloader)}")
    model = model_engine    
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
    if is_main_process:
        if len(checkpoint_epochs_to_save) == 0:
            print("[INFO] Epoch checkpoint saving disabled by --save_checkpoint_epochs.")
        else:
            print(f"[INFO] Save epoch checkpoints on epochs: {sorted(checkpoint_epochs_to_save)}")
    emotional_label_dict, emotional_label_str = get_labels_attr(dataset=args.dataset)
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
        for epoch in range(args.num_train_epochs):
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
                batch = _move_nested_to_device(batch, device)
                loss, opsd_stats = compute_opsd_training_loss(model, batch, tokenizer, args)

                model.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                model.step()
                    
                current_loss = loss.item()
                if is_main_process:
                    postfix = {
                        "loss": f"{current_loss:.4f}",
                        "opsd": f"{opsd_stats['opsd_loss']:.4f}",
                        "gen": int(opsd_stats["gen_tokens"]),
                    }
                    if opsd_stats.get("sft_loss") is not None:
                        postfix["sft"] = f"{opsd_stats['sft_loss']:.4f}"
                    if opsd_stats.get("contrastive_loss") is not None:
                        postfix["con"] = f"{opsd_stats['contrastive_loss']:.4f}"
                    batch_iterator.set_postfix(postfix, refresh=False)

                if step % args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    should_save = True
                if global_steps % args.save_steps == 0 and should_save:
                    model.save_checkpoint(args.output_dir)
                    should_save = False

                # model starts to evaluation
            
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
                generation_batch = _get_generation_input_dict(eval_batch)
                with torch.no_grad():
                    if "chatglm" in args.model_name_or_path:
                        outputs = model.generate(
                            **generation_batch,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    else:
                        outputs = model.generate(
                            **generation_batch,
                            num_beams=args.num_beams,
                            do_sample=False,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            repetition_penalty=args.repetition_penalty,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            # stopping_criteria=StoppingCriteriaList([stop_criteria]
                        )
                outputs = outputs.detach().cpu()
                outputs[outputs[:, :] < 0] = tokenizer.pad_token_id

                batch_size_now = outputs.size(0)
                batch_indices = local_eval_indices[local_cursor:local_cursor + batch_size_now]
                local_cursor += batch_size_now

                for i in range(batch_size_now):
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
                        input_ids = input_ids_cpu[i]
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

            all_records = _gather_eval_records(local_records)
            preds_for_eval = _dedupe_records_by_index(all_records)
            all_answers = [item["output"] for item in preds_for_eval]
            eval_targets = [item["target"] for item in preds_for_eval]
            if args.statistic_mode == "True":
                task_report = build_task_statistics(
                    targets=eval_targets,
                    answers=all_answers,
                    emotional_label_dict=emotional_label_dict,
                    dataset=args.dataset,
                    speaker_loss_weight=args.speaker_loss_weight,
                    personality_loss_weight=args.personality_loss_weight,
                )

                if is_main_process:
                    _write_eval_outputs(preds_for_eval_path, preds_for_eval, task_report=task_report)

                selection_metric = task_report.get('selection_metric') if is_main_process else None
                should_save_best = False
                next_best_f1_score = best_f1_score
                if selection_metric is not None and best_f1_score < selection_metric['value']:
                    should_save_best = True
                    next_best_f1_score = selection_metric['value']
                if dist.is_available() and dist.is_initialized() and world_size > 1:
                    best_payload = [should_save_best, next_best_f1_score]
                    dist.broadcast_object_list(best_payload, src=0)
                    should_save_best, next_best_f1_score = best_payload
                if should_save_best:
                    best_f1_score = next_best_f1_score
                    model.save_checkpoint(args.output_dir, tag="best_selection_metric")
                    if is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                        config.save_pretrained(args.output_dir)
                        args.save(args.output_dir)
                        # Save LoRA adapter weights if using PEFT
                        if args.lora and hasattr(model, 'save_pretrained'):
                            model.save_pretrained(args.output_dir)
                    if is_main_process:
                        with open(os.path.join(args.output_dir, "deepspeed_config.json"), 'w', encoding='utf-8') as f:
                            f.write(json.dumps(deepspeed_config, indent=5))
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
            if current_epoch in checkpoint_epochs_to_save:
                epoch_ckpt_tag = f"epoch{current_epoch}_global_step{global_steps}"
                model.save_checkpoint(args.output_dir, tag=epoch_ckpt_tag)
            elif is_main_process:
                print(f"[INFO] Skip checkpoint save at epoch {current_epoch} by --save_checkpoint_epochs setting.")
            
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
            generation_batch = _get_generation_input_dict(eval_batch)
            with torch.no_grad():
                if "chatglm" in args.model_name_or_path:
                    outputs = model.generate(
                        **generation_batch,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                else:
                    outputs = model.generate(
                        **generation_batch,
                        num_beams=args.num_beams,
                        do_sample=False,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        # stopping_criteria=StoppingCriteriaList([stop_criteria])
                    )
            outputs = outputs.detach().cpu()
            outputs[outputs[:, :] < 0] = tokenizer.pad_token_id

            batch_size_now = outputs.size(0)
            batch_indices = local_eval_indices[local_cursor:local_cursor + batch_size_now]
            local_cursor += batch_size_now

            for i in range(batch_size_now):
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
                    input_ids = input_ids_cpu[i]
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

        all_records = _gather_eval_records(local_records)
        preds_for_eval = _dedupe_records_by_index(all_records)
        all_answers = [item["output"] for item in preds_for_eval]
        eval_targets = [item["target"] for item in preds_for_eval]

        task_report = build_task_statistics(
            targets=eval_targets,
            answers=all_answers,
            emotional_label_dict=emotional_label_dict,
            dataset=args.dataset,
            speaker_loss_weight=args.speaker_loss_weight,
            personality_loss_weight=args.personality_loss_weight,
        )
        if is_main_process and task_report.get('emotion') is not None:
            eval_score_list.append(task_report['emotion']['score'])

        if is_main_process:
            _write_eval_outputs(preds_for_eval_path, preds_for_eval, task_report=task_report)

    if is_main_process:
        for i in eval_score_list:
            print(i)

    _safe_dist_barrier(tag="program_end")
