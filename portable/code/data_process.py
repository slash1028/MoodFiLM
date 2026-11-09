import argparse
import pickle
import json
import os 
import csv
import re
from datetime import datetime
from collections import defaultdict


TRAIT_ORDER = ['O', 'C', 'E', 'A', 'N']
DEFAULT_EXTERNAL_REASONING_DIR = '/n/work6/yiwang/MoodFilm/portable/processed_reasoning_data_full/reasoning_samples_llm'
DEFAULT_IEMOCAP_ROOT = '/n/work6/yiwang/MoodFilm/portable/datasets/IEMOCAP_full_release'
DEFAULT_EMOTION_REASONING_HIDDEN_CUE_TERMS = (
    'emotion_trigger',
    'trajectory',
    'dominant',
    'speaking style',
    'gold label',
    'gold emotion',
    'stated gold',
    'emotion cue set',
    'va_change',
    'delta_change',
)


def _is_true_flag(value):
    return str(value).strip().lower() in ('true', '1', 'yes', 'y')


def _clean_demo_text(text):
    """Remove explicit speaker tags to avoid speaker-identity leakage in demonstrations."""
    clean = re.sub(r'Speaker_\d+\s*:\s*', '', str(text))
    return clean.strip()


def _token_set(text):
    return set(re.findall(r"\w+", str(text).lower()))


def _lexical_cosine(query_tokens, doc_tokens):
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / ((len(query_tokens) * len(doc_tokens)) ** 0.5)


def _coerce_personality_traits(personality_value):
    if personality_value is None:
        return None

    if isinstance(personality_value, str):
        raw = personality_value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        if parsed == personality_value:
            return None
        return _coerce_personality_traits(parsed)

    if not isinstance(personality_value, dict):
        return None

    if not set(TRAIT_ORDER).issubset(set(personality_value.keys())):
        return None

    discrete_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
    values = []
    has_discrete = False
    for trait in TRAIT_ORDER:
        raw = personality_value.get(trait)
        if isinstance(raw, (int, float)):
            values.append(float(raw))
            continue
        token = str(raw).strip().lower()
        if token in discrete_map:
            has_discrete = True
            values.append(discrete_map[token])
            continue
        return None

    if has_discrete:
        return values

    if all(0.0 <= v <= 1.0 for v in values):
        return values

    return [max(0.0, min(1.0, (v - 1.0) / 6.0)) for v in values]


def _personality_similarity(query_personality, candidate_personality):
    query_vec = _coerce_personality_traits(query_personality)
    candidate_vec = _coerce_personality_traits(candidate_personality)
    if query_vec is None or candidate_vec is None:
        return 0.0
    diff = sum(abs(q - c) for q, c in zip(query_vec, candidate_vec)) / len(TRAIT_ORDER)
    return max(0.0, 1.0 - diff)


def _format_personality_for_demo(personality_value):
    if personality_value is None:
        return None
    if isinstance(personality_value, str):
        text = personality_value.strip()
        return text if text else None
    if not isinstance(personality_value, dict):
        return None
    if not set(TRAIT_ORDER).issubset(set(personality_value.keys())):
        return None

    first_value = str(personality_value.get('O', '')).strip().lower()
    if first_value in ['low', 'medium', 'high']:
        return _format_discrete_personality_from_levels(personality_value)

    return (
        f"<O:{personality_value['O']}, C:{personality_value['C']}, "
        f"E:{personality_value['E']}, A:{personality_value['A']}, N:{personality_value['N']}>"
    )


def _truncate_text_chars(text, max_chars=5200):
    raw = str(text or '')
    if max_chars is None or int(max_chars) <= 0 or len(raw) <= int(max_chars):
        return raw
    max_chars = int(max_chars)
    marker = '\n...[truncated]...\n'
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return raw[:head].rstrip() + marker + raw[-tail:].lstrip()


def build_balanced_domain_base(examples, label_list):
    """
    Build a label-balanced domain base from training examples.
    Each example should be: {"key": str, "text": str, "label": str}
    """
    by_label = defaultdict(list)
    for ex in examples:
        if ex['label'] in label_list:
            by_label[ex['label']].append(ex)

    available_labels = [lb for lb in label_list if len(by_label[lb]) > 0]
    if not available_labels:
        return []

    min_count = min(len(by_label[lb]) for lb in available_labels)
    balanced = []
    for lb in label_list:
        if lb not in by_label:
            continue
        # deterministic order to keep preprocessing reproducible
        sorted_bucket = sorted(by_label[lb], key=lambda x: x['key'])
        balanced.extend(sorted_bucket[:min_count])
    return balanced


def build_demo_retriever(domain_examples, backend='auto', model_name='sentence-transformers/all-MiniLM-L6-v2',
                         top_k=16, alpha=0.75, exclude_same_dialogue=True, personality_weight=0.2):
    """
    Build retrieval function for domain demonstrations.
    Preference: SBERT semantic retrieval; fallback: TF-IDF cosine retrieval.
    Returns callable: retrieve(query_text, query_key=None, query_personality=None, return_meta=False) -> example dict or None
    """
    if not domain_examples:
        return lambda query_text, query_key=None, return_meta=False: (None, {}) if return_meta else None

    texts = [ex['text'] for ex in domain_examples]
    keys = [ex['key'] for ex in domain_examples]

    base_tokens = [_token_set(t) for t in texts]
    key_to_dialog = {k: k.rsplit('_', 1)[0] if '_' in k else k for k in keys}

    use_sbert = backend in ('auto', 'sbert')
    sbert_ready = False
    sbert_model = None
    st_util = None
    base_emb = None

    if use_sbert:
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers import util as st_util_mod

            sbert_model = SentenceTransformer(model_name)
            st_util = st_util_mod
            base_emb = sbert_model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sbert_ready = True
        except Exception:
            # In auto mode fallback to lexical; in strict sbert mode we still fallback for robustness.
            sbert_ready = False

    top_k = max(int(top_k), 1)
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    personality_weight = max(0.0, float(personality_weight))

    def _eligible(idx, query_key):
        if query_key is None:
            return True
        if keys[idx] == query_key:
            return False
        if exclude_same_dialogue:
            q_dialog = key_to_dialog.get(query_key)
            d_dialog = key_to_dialog.get(keys[idx])
            if q_dialog is not None and d_dialog is not None and q_dialog == d_dialog:
                return False
        return True

    def retrieve(query_text, query_key=None, query_personality=None, return_meta=False):
        q_tokens = _token_set(query_text)

        def _score_candidate(idx, sem_score, lex_score, query_personality):
            personality_score = _personality_similarity(query_personality, domain_examples[idx].get('personality'))
            hybrid_score = alpha * sem_score + (1.0 - alpha) * lex_score + personality_weight * personality_score
            return hybrid_score, personality_score

        # Stage 1: semantic candidates if available.
        candidate_idx = []
        semantic_score = {}
        if sbert_ready:
            query_emb = sbert_model.encode(
                [query_text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sims = st_util.cos_sim(query_emb, base_emb)[0]
            ranked = sims.argsort(descending=True).tolist()
            for idx in ranked:
                if not _eligible(idx, query_key):
                    continue
                candidate_idx.append(idx)
                semantic_score[idx] = float(sims[idx].item())
                if len(candidate_idx) >= top_k:
                    break

        # Stage 2: lexical retrieval fallback or rerank supplement.
        if not candidate_idx:
            all_ranked = []
            for idx, d_tokens in enumerate(base_tokens):
                if not _eligible(idx, query_key):
                    continue
                lex = _lexical_cosine(q_tokens, d_tokens)
                hybrid, personality_score = _score_candidate(idx, 0.0, lex, query_personality)
                all_ranked.append((hybrid, idx, 0.0, lex, personality_score))
            all_ranked.sort(key=lambda x: x[0], reverse=True)
            if not all_ranked:
                if return_meta:
                    return None, {
                        'retriever_backend': 'lexical',
                        'semantic_score': None,
                        'lexical_score': None,
                        'personality_score': None,
                        'hybrid_score': None,
                    }
                return None
            best_score, best_idx_lex, best_sem, best_lex, best_personality = all_ranked[0]
            if return_meta:
                return domain_examples[best_idx_lex], {
                    'retriever_backend': 'lexical',
                    'semantic_score': float(best_sem),
                    'lexical_score': float(best_lex),
                    'personality_score': float(best_personality),
                    'hybrid_score': float(best_score),
                }
            return domain_examples[best_idx_lex]

        best_idx = None
        best_sem = None
        best_lex = None
        best_personality = None
        best_hybrid = -1.0
        for idx in candidate_idx:
            lex = _lexical_cosine(q_tokens, base_tokens[idx])
            sem = semantic_score.get(idx, 0.0)
            hybrid, personality_score = _score_candidate(idx, sem, lex, query_personality)
            if hybrid > best_hybrid:
                best_hybrid = hybrid
                best_idx = idx
                best_sem = sem
                best_lex = lex
                best_personality = personality_score

        if best_idx is None:
            if return_meta:
                return None, {
                    'retriever_backend': 'sbert_hybrid',
                    'semantic_score': None,
                    'lexical_score': None,
                    'personality_score': None,
                    'hybrid_score': None,
                }
            return None
        if return_meta:
            return domain_examples[best_idx], {
                'retriever_backend': 'sbert_hybrid',
                'semantic_score': float(best_sem),
                'lexical_score': float(best_lex),
                'personality_score': float(best_personality),
                'hybrid_score': float(best_hybrid),
            }
        return domain_examples[best_idx]

    return retrieve


def format_retrieved_demonstration(example, personality_condition_mode='False', thresholds=None):
    """Format one retrieved demonstration into textual prompt part."""
    if example is None:
        return ''

    personality_text = None
    personality_value = example.get('personality')
    if personality_condition_mode in ('True_discrete', 'True_continuous'):
        personality_text = resolve_personality_condition_text(
            personality_value,
            personality_condition_mode,
            thresholds=thresholds,
        )
    else:
        personality_text = _format_personality_for_demo(personality_value)

    if personality_text:
        return (
            'A similar utterance and its emotion label are given for reference, along with the speaker personality: '
            f'<Utterance: "{example["text"]}"; Emotion label: {example["label"]}; Personality traits: {personality_text}>.'
        )
    return (
        'A similar utterance and its emotion label are given for reference: '
        f'<Utterance: "{example["text"]}"; Emotion label: {example["label"]}>.'
    )


def _resolve_run_id(run_id=None):
    if run_id:
        return str(run_id).strip()
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def _with_run_suffix(base_path, run_id):
    if not run_id:
        return base_path
    return os.path.join(base_path, f'run_{run_id}')


def _build_run_root(dataset, run_id, run_root=None):
    if run_root is not None and str(run_root).strip() != '':
        return str(run_root).strip()
    run_folder = str(run_id)
    if not run_folder.startswith('run_'):
        run_folder = f'run_{run_folder}'
    return f'/n/work6/yiwang/MoodFilm/portable/code/experiments/ablation/{dataset}/runs/{run_folder}'


def _write_run_manifest(output_path, manifest):
    os.makedirs(output_path, exist_ok=True)
    manifest_path = os.path.join(output_path, 'run_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f_manifest:
        json.dump(manifest, f_manifest, ensure_ascii=False, indent=2)
    return manifest_path


def _resolve_personality_task_setting(personality_task, personality_mode='raw'):
    """Resolve personality task switch into (enabled, normalized_mode)."""
    task_flag = str(personality_task)
    if task_flag in ('False', 'None', ''):
        return False, 'raw'
    if task_flag in ('True', 'True_continuous'):
        return True, 'raw'
    if task_flag == 'True_discrete':
        return True, 'ternary'
    if task_flag == 'True_binary':
        return True, 'binary'

    # Backward compatibility: if personality_task is True-like, trust explicit personality_mode.
    mode = personality_mode if personality_mode in ('raw', 'binary', 'ternary') else 'raw'
    return task_flag.lower() == 'true', mode



def normalize_iemocap_dialogue_id(dialogue_id):
    """Normalize ids like script1_1 -> script01_1 for robust matching."""
    return re.sub(r'(script)(\d)_', r'\g<1>0\2_', dialogue_id.strip())


def _dialogue_to_iemocap_session(dialogue_id):
    match = re.match(r'^Ses(\d+)', str(dialogue_id))
    if match is None:
        return None
    return f'Session{int(match.group(1))}'


def _normalize_iemocap_alignment_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    return text


def _parse_iemocap_transcript_lines(transcript_path):
    utterances = []
    if not os.path.exists(transcript_path):
        return utterances
    line_pattern = re.compile(r'^(\S+)\s+\[[^\]]+\]:\s*(.*)$')
    with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = line_pattern.match(line.strip())
            if match is None:
                continue
            utterances.append((match.group(1), match.group(2)))
    return utterances


def build_iemocap_official_utterance_id_map(sentence_dict, conv_ids, iemocap_root=DEFAULT_IEMOCAP_ROOT):
    """
    Map processed pkl turn keys to original IEMOCAP utterance ids.

    The processed iemocap.pkl can omit or merge some transcript lines, so simply
    counting speaker turns produces ids like F026 for a turn whose official id is
    F032. We align each processed sentence to the original transcript in order to
    preserve stable official ids while keeping sample_id as the pkl turn key.
    """
    if not iemocap_root or not os.path.isdir(iemocap_root):
        return {}

    mapping = {}
    for conv_id in conv_ids:
        session = _dialogue_to_iemocap_session(conv_id)
        if session is None:
            continue
        transcript_path = os.path.join(
            iemocap_root,
            session,
            'dialog',
            'transcriptions',
            f'{conv_id}.txt',
        )
        transcript_utts = _parse_iemocap_transcript_lines(transcript_path)
        if not transcript_utts:
            continue

        search_start = 0
        sentences = sentence_dict.get(conv_id, [])
        for turn_idx, sentence in enumerate(sentences):
            norm_sentence = _normalize_iemocap_alignment_text(sentence)
            if not norm_sentence:
                continue

            found_idx = None
            for transcript_idx in range(search_start, len(transcript_utts)):
                if _normalize_iemocap_alignment_text(transcript_utts[transcript_idx][1]) == norm_sentence:
                    found_idx = transcript_idx
                    break
            if found_idx is None:
                continue

            mapping[(conv_id, turn_idx)] = transcript_utts[found_idx][0]
            search_start = found_idx + 1

    return mapping


def load_ocean_personality_map(csv_path):
    """
    Build mapping: (dialogue_id, gender[m/f]) -> OCEAN vector.
    PA-IEMOCAP columns: EX, AG, CO, NE, OP.
    """
    personality_map = {}
    if not os.path.exists(csv_path):
        return personality_map

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogue_id = normalize_iemocap_dialogue_id(row['dialogue_id'])
            gender = row['gender'].strip().lower()
            # Re-order to O, C, E, A, N for downstream consistency.
            personality_map[(dialogue_id, gender)] = {
                'O': float(row['OP']),
                'C': float(row['CO']),
                'E': float(row['EX']),
                'A': float(row['AG']),
                'N': float(row['NE'])
            }
    return personality_map


def compute_personality_thresholds(personality_vectors, trait_keys=['O', 'C', 'E', 'A', 'N']):
    """
    Compute percentile-based thresholds from train split for discretization.
    Returns: dict of trait -> (low_threshold, high_threshold)
    """
    thresholds = {}
    for trait in trait_keys:
        scores = [v[trait] for v in personality_vectors if v is not None]
        if scores:
            low_th = sorted(scores)[len(scores) // 3]      # 33rd percentile
            high_th = sorted(scores)[2 * len(scores) // 3]  # 67th percentile
            thresholds[trait] = (low_th, high_th)
    return thresholds


def discretize_personality(personality_vector, thresholds=None):
    """
    Convert OCEAN vector to discrete labels using thresholds.
    Returns: "low openness, medium conscientiousness, high extraversion, ..."
    """
    trait_names = {'O': 'openness', 'C': 'conscientiousness', 'E': 'extraversion', 
                   'A': 'agreeableness', 'N': 'neuroticism'}
    trait_order = ['O', 'C', 'E', 'A', 'N']
    
    if thresholds is None:
        # Default thresholds if not provided
        thresholds = {t: (3.0, 5.0) for t in trait_order}
    
    labels = []
    for trait in trait_order:
        score = personality_vector.get(trait, 4.0)
        low_th, high_th = thresholds.get(trait, (3.0, 5.0))
        
        if score < low_th:
            label = 'low'
        elif score > high_th:
            label = 'high'
        else:
            label = 'medium'
        
        labels.append(f"{label} {trait_names[trait]}")
    
    return ', '.join(labels)


def personality_to_target(personality_vector, mode='raw'):
    """
    Convert OCEAN vector to a string target.
    mode:
      - raw:    [O, C, E, A, N] continuous 1-7 scores
      - binary: each dimension -> low/high
      - ternary: each dimension -> low/medium/high
    """
    if mode == 'raw':
        return json.dumps([
            round(float(personality_vector['O']), 3),
            round(float(personality_vector['C']), 3),
            round(float(personality_vector['E']), 3),
            round(float(personality_vector['A']), 3),
            round(float(personality_vector['N']), 3)
        ], ensure_ascii=False)

    def discretize(score):
        if mode == 'binary':
            return 'high' if score >= 4.0 else 'low'
        if mode == 'ternary':
            if score < 3.0:
                return 'low'
            if score <= 5.0:
                return 'medium'
            return 'high'
        return score

    return json.dumps({
        'O': discretize(personality_vector['O']),
        'C': discretize(personality_vector['C']),
        'E': discretize(personality_vector['E']),
        'A': discretize(personality_vector['A']),
        'N': discretize(personality_vector['N'])
    }, ensure_ascii=False)


def load_pseudo_personality_map(json_path):
    if json_path is None or str(json_path).strip() == '':
        return {}
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        return {}
    except Exception:
        return {}


def _format_discrete_personality_from_levels(levels):
    trait_names = {
        'O': 'openness',
        'C': 'conscientiousness',
        'E': 'extraversion',
        'A': 'agreeableness',
        'N': 'neuroticism',
    }
    parts = []
    for trait in ['O', 'C', 'E', 'A', 'N']:
        v = str(levels.get(trait, 'medium')).strip().lower()
        if v not in ['low', 'medium', 'high']:
            v = 'medium'
        parts.append(f"{v} {trait_names[trait]}")
    return ', '.join(parts)


def resolve_personality_condition_text(personality_value, condition_mode, thresholds=None):
    if personality_value is None:
        return None

    if condition_mode == 'True_discrete':
        if isinstance(personality_value, dict):
            keys = set(personality_value.keys())
            if {'O', 'C', 'E', 'A', 'N'}.issubset(keys):
                first_val = str(personality_value.get('O', '')).strip().lower()
                if first_val in ['low', 'medium', 'high']:
                    return _format_discrete_personality_from_levels(personality_value)
                return discretize_personality(personality_value, thresholds)
        if isinstance(personality_value, str) and len(personality_value.strip()) > 0:
            return personality_value.strip()
        return None

    if isinstance(personality_value, dict):
        keys = set(personality_value.keys())
        if {'O', 'C', 'E', 'A', 'N'}.issubset(keys):
            first_val = str(personality_value.get('O', '')).strip().lower()
            if first_val in ['low', 'medium', 'high']:
                return None
            return (
                f"<O:{personality_value['O']}, C:{personality_value['C']}, "
                f"E:{personality_value['E']}, A:{personality_value['A']}, N:{personality_value['N']}>"
            )
    return None


def build_personality_condition_prompt(personality_value, condition_mode, thresholds=None):
    """Build richer personality condition text for emotion prompt."""
    cond_text = resolve_personality_condition_text(personality_value, condition_mode, thresholds=thresholds)
    if cond_text is None:
        return None

    if condition_mode == 'True_discrete':
        return (
            'Target speaker personality profile (OCEAN tendencies) is: '
            f'{cond_text}. '
            'Use these tendencies to disambiguate subtle emotions in tone and wording.'
        )

    if isinstance(personality_value, dict):
        tendency = discretize_personality(personality_value, thresholds=thresholds)
        return (
            'Target speaker personality profile (OCEAN, 1.0-7.0 where higher means stronger trait) is '
            f'{cond_text}. '
            f'Interpreted tendencies: {tendency}. '
            'Use this profile to disambiguate subtle emotions in tone and wording.'
        )

    return (
        'Target speaker personality profile is '
        f'{cond_text}. '
        'Use this profile to disambiguate subtle emotions in tone and wording.'
    )


def load_experience_bank_items(jsonl_path, confidence='high', max_items=0):
    if jsonl_path is None or str(jsonl_path).strip() == '':
        return []
    if not os.path.exists(jsonl_path):
        return []

    allowed_confidence = str(confidence or '').strip().lower()
    max_items = int(max_items or 0)
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get('use_or_not', True) is False:
                continue
            if allowed_confidence and allowed_confidence != 'all':
                if str(item.get('confidence', '')).strip().lower() != allowed_confidence:
                    continue
            experience = str(item.get('experience_item') or '').strip()
            if not experience:
                continue
            items.append(item)
            if max_items > 0 and len(items) >= max_items:
                break
    return items


def format_experience_bank_context(items, include_evidence=False):
    if not items:
        return ''
    lines = [
        'Personality-interaction experience bank:',
        'These are probabilistic training-split tendencies, not rules and not current-sample labels.',
    ]
    for idx, item in enumerate(items, start=1):
        experience = str(item.get('experience_item') or '').strip()
        if not experience:
            continue
        lines.append(f'{idx}. {experience}')
        if include_evidence:
            evidence = str(item.get('evidence') or '').strip()
            scope = str(item.get('scope') or '').strip()
            risk = str(item.get('risk') or '').strip()
            if evidence:
                lines.append(f'   Evidence: {evidence}')
            if scope:
                lines.append(f'   Scope: {scope}')
            if risk:
                lines.append(f'   Risk: {risk}')
    return '\n'.join(lines)


def _normalize_personality_setting(setting, arg_name):
    value = str(setting or 'conversation').strip().lower()
    aliases = {
        'u': 'utterance',
        'utt': 'utterance',
        'utterance_level': 'utterance',
        'c': 'conversation',
        'conv': 'conversation',
        'conversation_level': 'conversation',
        'f': 'full',
        'full_dialogue': 'full',
        'dialogue': 'full',
    }
    value = aliases.get(value, value)
    if value not in ('utterance', 'conversation', 'full'):
        raise ValueError(f"Invalid {arg_name}: {setting}. Use utterance / conversation / full")
    return value


def _extract_tagged_answer(text):
    match = re.search(
        r'<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>',
        str(text or ''),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).strip()


def _extract_tagged_reasoning(text):
    match = re.search(
        r'<\s*reasoning\s*>\s*(.*?)\s*<\s*/\s*reasoning\s*>',
        str(text or ''),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).strip()


def _target_has_reasoning_and_answer(text):
    raw = str(text or '')
    return bool(re.search(r'<\s*reasoning\s*>', raw, flags=re.IGNORECASE)) and _extract_tagged_answer(raw) is not None


STRUCTURED_EMOTION_STATE_TAGS = (
    ('shift', 'emotion_shift'),
    ('trigger', 'emotion_trigger'),
    ('style', 'speaking_style'),
    ('act', 'interaction_type'),
    ('dynamics', 'speaker_emotion_dynamics'),
    ('pattern', 'speaker_interaction_pattern'),
    ('dialogue_type', 'dialogue_type'),
    ('atmosphere', 'dialogue_atmosphere'),
)

STRUCTURED_EMOTION_CUE_ALIASES = {
    'shift': 'emotion_shift',
    'emotion_shift': 'emotion_shift',
    'trigger': 'emotion_trigger',
    'emotion_trigger': 'emotion_trigger',
    'style': 'speaking_style',
    'speaking_style': 'speaking_style',
    'act': 'interaction_type',
    'interaction': 'interaction_type',
    'interaction_type': 'interaction_type',
    'dynamics': 'speaker_emotion_dynamics',
    'speaker_dynamics': 'speaker_emotion_dynamics',
    'speaker_emotion_dynamics': 'speaker_emotion_dynamics',
    'pattern': 'speaker_interaction_pattern',
    'speaker_pattern': 'speaker_interaction_pattern',
    'interaction_pattern': 'speaker_interaction_pattern',
    'speaker_interaction_pattern': 'speaker_interaction_pattern',
    'dialogue_type': 'dialogue_type',
    'dtype': 'dialogue_type',
    'atmosphere': 'dialogue_atmosphere',
    'dialogue_atmosphere': 'dialogue_atmosphere',
    'atm': 'dialogue_atmosphere',
}


def _target_has_answer(text):
    return _extract_tagged_answer(text) is not None


def _target_has_valid_emotion_reasoning_target(text, structured_reasoning=False):
    if _is_true_flag(structured_reasoning):
        return _target_has_answer(text)
    return _target_has_reasoning_and_answer(text)


def _normalize_structured_cue_value(value):
    value = str(value or '').strip().strip("\"'` ")
    value = re.sub(r'\s+', '_', value)
    value = value.strip('.,;:')
    return value.lower() if value else 'unknown'


def _parse_structured_emotion_cues(text):
    raw = str(text or '')
    match = re.search(
        r'(?:Structured\s+cues|Structured\s+labels)\s*:\s*(.*?)(?:\n|$)',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return {}

    cue_line = match.group(1).strip()
    cues = {}
    for part in cue_line.split(';'):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = re.sub(r'[^a-zA-Z0-9_]+', '_', key.strip().lower()).strip('_')
        canonical_key = STRUCTURED_EMOTION_CUE_ALIASES.get(key)
        if canonical_key is None:
            continue
        cues[canonical_key] = _normalize_structured_cue_value(value)
    return cues


def _format_ocean_chain_value(personality_vector):
    if not isinstance(personality_vector, dict):
        return 'O=unknown; C=unknown; E=unknown; A=unknown; N=unknown'
    parts = []
    for trait in ['O', 'C', 'E', 'A', 'N']:
        try:
            value = float(personality_vector[trait])
            parts.append(f'{trait}={value:.3f}')
        except Exception:
            parts.append(f'{trait}=unknown')
    return '; '.join(parts)


def _format_compact_structured_states(structured_cues):
    cues = structured_cues if isinstance(structured_cues, dict) else {}
    return ';'.join([
        f'{tag}={_normalize_structured_cue_value(cues.get(cue_key, "unknown"))}'
        for tag, cue_key in STRUCTURED_EMOTION_STATE_TAGS
    ])


def _format_structured_emotion_target(answer, structured_cues, personality_vector=None, predict_personality_in_chain=False):
    answer = _normalize_structured_cue_value(answer)
    cues = structured_cues if isinstance(structured_cues, dict) else {}
    states = _format_compact_structured_states(cues)
    if _is_true_flag(predict_personality_in_chain):
        personality = _format_ocean_chain_value(personality_vector).replace('; ', ';')
        return (
            f'<personality>{personality}</personality>'
            f'<label>{answer}</label>'
            f'<states>{states}</states>'
            f'<answer>{answer}</answer>'
        )

    return f'<label>{answer}</label><states>{states}</states><answer>{answer}</answer>'


def _normalize_reasoning_words(text):
    return re.sub(r'\s+', ' ', str(text or '').strip())


def _split_reasoning_sentences(text):
    normalized = _normalize_reasoning_words(text)
    if not normalized:
        return []
    return [
        part.strip()
        for part in re.split(r'(?<=[.!?])\s+', normalized)
        if part.strip()
    ]


def _word_count(text):
    return len(re.findall(r'\S+', str(text or '').strip()))


def _limit_reasoning_words(sentences, max_words):
    if max_words is None or int(max_words) <= 0:
        return _normalize_reasoning_words(' '.join(sentences))

    max_words = int(max_words)
    selected = []
    total = 0
    for sent in sentences:
        sent_words = re.findall(r'\S+', sent)
        if not sent_words:
            continue
        if selected and total + len(sent_words) > max_words:
            break
        if not selected and len(sent_words) > max_words:
            selected = [' '.join(sent_words[:max_words]).rstrip(',;:')]
            total = max_words
            break
        selected.append(sent)
        total += len(sent_words)

    text = _normalize_reasoning_words(' '.join(selected))
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def _clean_emotion_reasoning_text(reasoning, clean_hidden_cues=True, min_words=0, max_words=0):
    sentences = _split_reasoning_sentences(reasoning)
    if clean_hidden_cues:
        filtered = []
        for sent in sentences:
            lowered = sent.lower()
            if any(term in lowered for term in DEFAULT_EMOTION_REASONING_HIDDEN_CUE_TERMS):
                continue
            filtered.append(sent)
        if filtered:
            sentences = filtered

    text = _limit_reasoning_words(sentences, max_words)
    if min_words and _word_count(text) < int(min_words):
        # The source reasoning is not expanded synthetically. This keeps targets faithful
        # while still preventing long generations from pushing the answer tag too far back.
        return text
    return text


def _format_emotion_reasoning_target(target, label_list, output_format='answer_last',
                                     min_words=0, max_words=0, clean_hidden_cues=True,
                                     structured_reasoning=False, structured_cues=None,
                                     personality_vector=None, predict_personality_in_chain=False):
    raw = str(target or '').strip()
    answer = _extract_tagged_answer(raw)
    reasoning = _extract_tagged_reasoning(raw)
    if answer is None:
        return raw

    answer = answer.strip().lower()
    if answer not in label_list:
        return raw

    if _is_true_flag(structured_reasoning):
        return _format_structured_emotion_target(
            answer,
            structured_cues,
            personality_vector=personality_vector,
            predict_personality_in_chain=predict_personality_in_chain,
        )

    if reasoning is None:
        return raw

    should_rewrite = (
        str(output_format or '').strip().lower() == 'label_anchor_answer_last'
        or int(max_words or 0) > 0
        or _is_true_flag(clean_hidden_cues)
    )
    if not should_rewrite:
        return raw

    reasoning = _clean_emotion_reasoning_text(
        reasoning,
        clean_hidden_cues=_is_true_flag(clean_hidden_cues),
        min_words=int(min_words or 0),
        max_words=int(max_words or 0),
    )
    if str(output_format or '').strip().lower() == 'label_anchor_answer_last':
        return '\n'.join([
            '<label>',
            answer,
            '</label>',
            '<reasoning>',
            reasoning,
            '</reasoning>',
            '<answer>',
            answer,
            '</answer>',
        ])
    return '\n'.join([
        '<reasoning>',
        reasoning,
        '</reasoning>',
        '<answer>',
        answer,
        '</answer>',
    ])


def _iter_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {exc}') from exc


def _dialogue_key_variants(dialogue_id):
    raw = str(dialogue_id).strip()
    variants = [raw]
    norm = normalize_iemocap_dialogue_id(raw)
    if norm != raw:
        variants.append(norm)
    return variants


def _normalize_reasoning_match_text(text):
    return re.sub(r'\s+', ' ', str(text or '').strip().lower())


def load_external_reasoning_targets(external_reasoning_dir, need_emotion=False, need_personality=False):
    """Load LLM-written reasoning targets keyed to original IEMOCAP ids."""
    root = str(external_reasoning_dir or '').strip()
    emotion_targets = {}
    emotion_utterance_id_targets = {}
    emotion_text_targets = defaultdict(list)
    personality_targets = {}

    if need_emotion:
        emotion_path = os.path.join(root, 'sft', 'emotion_reasoning.jsonl')
        if not os.path.exists(emotion_path):
            raise FileNotFoundError(f'External emotion reasoning file not found: {emotion_path}')
        for line_no, row in _iter_jsonl(emotion_path):
            meta = row.get('metadata', {}) if isinstance(row.get('metadata'), dict) else {}
            dialogue_id = meta.get('dialogue_id')
            utterance_index = meta.get('utterance_index')
            utterance_id = meta.get('utterance_id')
            if dialogue_id is None or utterance_index is None:
                sample_id = str(row.get('sample_id', ''))
                match = re.match(r'^EmotionReasoning_(.+)_(\d+)$', sample_id)
                if match:
                    dialogue_id = match.group(1)
                    utterance_index = int(match.group(2))
            target = str(row.get('target', '')).strip()
            if dialogue_id is None or utterance_index is None or not _target_has_reasoning_and_answer(target):
                raise ValueError(f'Invalid emotion reasoning row at {emotion_path}:{line_no}')
            debug_text = row.get('debug_target_utterance')
            if not isinstance(debug_text, str) or len(debug_text.strip()) == 0:
                input_text = str(row.get('input', ''))
                match = re.search(
                    r'Target utterance:\s*(?:Speaker_\d+\s*:\s*)?"(.*?)"',
                    input_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if match:
                    debug_text = match.group(1)
            entry = {
                'target': target,
                'text': debug_text if isinstance(debug_text, str) else '',
                'utterance_index': int(utterance_index),
                'utterance_id': str(utterance_id).strip() if utterance_id is not None else '',
                'structured_cues': _parse_structured_emotion_cues(row.get('input', '')),
                'sample_id': row.get('sample_id'),
            }
            if entry['utterance_id']:
                emotion_utterance_id_targets[entry['utterance_id']] = entry
            for key_dialogue_id in _dialogue_key_variants(dialogue_id):
                emotion_targets[(key_dialogue_id, int(utterance_index))] = entry
                norm_text = _normalize_reasoning_match_text(entry['text'])
                if norm_text:
                    emotion_text_targets[(key_dialogue_id, norm_text)].append(entry)

        for entries in emotion_text_targets.values():
            entries.sort(key=lambda item: int(item.get('utterance_index', 10 ** 9)))

    if need_personality:
        personality_path = os.path.join(root, 'sft', 'personality_reasoning.jsonl')
        if not os.path.exists(personality_path):
            raise FileNotFoundError(f'External personality reasoning file not found: {personality_path}')
        for line_no, row in _iter_jsonl(personality_path):
            meta = row.get('metadata', {}) if isinstance(row.get('metadata'), dict) else {}
            dialogue_id = meta.get('dialogue_id')
            speaker = meta.get('speaker')
            if dialogue_id is None or speaker is None:
                sample_id = str(row.get('sample_id', ''))
                match = re.match(r'^PersonalityReasoning_(.+)_(Speaker_\d+)$', sample_id)
                if match:
                    dialogue_id = match.group(1)
                    speaker = match.group(2)
            target = str(row.get('target', '')).strip()
            if dialogue_id is None or speaker is None or not _target_has_reasoning_and_answer(target):
                raise ValueError(f'Invalid personality reasoning row at {personality_path}:{line_no}')
            for key_dialogue_id in _dialogue_key_variants(dialogue_id):
                personality_targets[(key_dialogue_id, str(speaker))] = target

    return emotion_targets, personality_targets, emotion_text_targets, emotion_utterance_id_targets



def process_dataset(dataset, window=110, speaker_task='True', demons='False', predictions='True',
                    personality_meta='False', personality_task='False', personality_mode='raw',
                    personality_condition='False',
                    emotion_context_mode='context',
                    personality_csv_path='/n/work6/yiwang/MoodFilm/portable/datasets/PA-IEMOCAP/ocean.csv',
                    iemocap_root=DEFAULT_IEMOCAP_ROOT,
                    personality_pseudo_json_path=None,
                    personality_use_pseudo_first='False',
                    external_reasoning_dir=DEFAULT_EXTERNAL_REASONING_DIR,
                    demo_backend='auto', demo_model_name='sentence-transformers/all-MiniLM-L6-v2',
                    demo_top_k=16, demo_alpha=0.75, demo_exclude_same_dialogue='True',
                    demo_personality_weight=0.2,
                    demo_trace='False', run_id=None, run_root=None,
                    personality_oversample_factor=1,
                    personality_train_setting='conversation',
                    personality_inference_setting='conversation',
                    personality_reasoning='False',
                    emotion_reasoning='False',
                    emotion_reasoning_format='answer_last',
                    emotion_reasoning_min_words=0,
                    emotion_reasoning_max_words=0,
                    emotion_reasoning_clean_hidden_cues='False',
                    structured_reasoning='False',
                    predict_personality_in_chain='False',
                    personality_input_mode='auto',
                    generate_reasoning_from_states='False',
                    emotion_prompt_output_format='label_only',
                    experience_bank_mode='none',
                    experience_bank_path=None,
                    experience_bank_confidence='high',
                    experience_bank_max_items=0,
                    opcd_teacher_include_gold_context='False'):
    '''
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    '''
    label_set = {
        'iemocap':['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld':   ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        'EmoryNLP':['Joyful', 'Mad', 'Peaceful','Neutral', 'Sad', 'Powerful', 'Scared']
    }
    label_text_set = {
        'iemocap':'happy, sad, neutral, angry, excited, frustrated',
        'meld'   :'neutral, surprise, fear, sad, joyful, disgust, angry',
        'EmoryNLP':'Joyful, Mad, Peaceful, Neutral, Sad, Powerful, Scared'
    }
    run_id = _resolve_run_id(run_id)
    personality_oversample_factor = max(1, int(personality_oversample_factor))
    personality_reasoning_enabled = _is_true_flag(personality_reasoning)
    emotion_reasoning_enabled = _is_true_flag(emotion_reasoning)
    structured_reasoning_enabled = _is_true_flag(structured_reasoning)
    predict_personality_in_chain = _is_true_flag(predict_personality_in_chain)
    generate_reasoning_from_states = _is_true_flag(generate_reasoning_from_states)
    emotion_prompt_output_format = str(emotion_prompt_output_format or 'label_only').strip().lower()
    emotion_prompt_output_format = {
        'label': 'label_only',
        'answer_only': 'label_only',
        'short_reasoning': 'short_reasoning_answer',
        'reasoning_answer': 'short_reasoning_answer',
    }.get(emotion_prompt_output_format, emotion_prompt_output_format)
    if emotion_prompt_output_format not in ('label_only', 'short_reasoning_answer'):
        raise ValueError("Invalid emotion_prompt_output_format. Use label_only / short_reasoning_answer")
    experience_bank_mode = str(experience_bank_mode or 'none').strip().lower()
    if experience_bank_mode not in ('none', 'student_prompt', 'opcd_teacher'):
        raise ValueError("Invalid experience_bank_mode. Use none / student_prompt / opcd_teacher")
    opcd_teacher_include_gold_context = _is_true_flag(opcd_teacher_include_gold_context)
    experience_bank_items = load_experience_bank_items(
        experience_bank_path,
        confidence=experience_bank_confidence,
        max_items=experience_bank_max_items,
    )
    student_experience_bank_context = (
        format_experience_bank_context(experience_bank_items, include_evidence=False)
        if experience_bank_mode == 'student_prompt'
        else ''
    )
    teacher_experience_bank_context = (
        format_experience_bank_context(experience_bank_items, include_evidence=False)
        if experience_bank_mode == 'opcd_teacher'
        else ''
    )
    personality_input_mode = str(personality_input_mode or 'auto').strip().lower()
    if personality_input_mode not in ('auto', 'gt', 'stage1_predicted', 'none'):
        raise ValueError("Invalid personality_input_mode. Use auto / gt / stage1_predicted / none")
    emotion_reasoning_format = str(emotion_reasoning_format or 'answer_last').strip().lower()
    emotion_reasoning_format_aliases = {
        'default': 'answer_last',
        'reasoning_answer': 'answer_last',
        'label_anchor': 'label_anchor_answer_last',
        'label_first': 'label_anchor_answer_last',
        'structured': 'structured_chain',
    }
    emotion_reasoning_format = emotion_reasoning_format_aliases.get(
        emotion_reasoning_format,
        emotion_reasoning_format,
    )
    if structured_reasoning_enabled:
        emotion_reasoning_format = 'structured_chain'
    if emotion_reasoning_format not in ('answer_last', 'label_anchor_answer_last', 'structured_chain'):
        raise ValueError("Invalid emotion_reasoning_format. Use answer_last / label_anchor_answer_last / structured_chain")
    emotion_reasoning_min_words = max(0, int(emotion_reasoning_min_words or 0))
    emotion_reasoning_max_words = max(0, int(emotion_reasoning_max_words or 0))
    short_prompt_min_words = emotion_reasoning_min_words if emotion_reasoning_min_words > 0 else 50
    short_prompt_max_words = emotion_reasoning_max_words if emotion_reasoning_max_words > 0 else 100
    if short_prompt_max_words < short_prompt_min_words:
        short_prompt_max_words = short_prompt_min_words
    emotion_reasoning_clean_hidden_cues = _is_true_flag(emotion_reasoning_clean_hidden_cues)
    personality_train_setting = _normalize_personality_setting(
        personality_train_setting,
        '--personality_train_setting',
    )
    personality_inference_setting = _normalize_personality_setting(
        personality_inference_setting,
        '--personality_inference_setting',
    )
    personality_task_enabled, resolved_personality_mode = _resolve_personality_task_setting(
        personality_task,
        personality_mode,
    )
    if personality_reasoning_enabled and personality_task_enabled and resolved_personality_mode != 'raw':
        raise ValueError("--personality_reasoning currently supports only continuous personality targets.")
    if (
        personality_reasoning_enabled
        and personality_task_enabled
        and dataset == 'iemocap'
        and (personality_train_setting == 'utterance' or personality_inference_setting == 'utterance')
    ):
        raise ValueError("--personality_reasoning with external LLM reasoning supports conversation/full speaker-level samples, not utterance-level samples.")
    emotion_context_mode = str(emotion_context_mode or 'context').strip().lower()
    if emotion_context_mode not in ('context', 'single'):
        raise ValueError("Invalid emotion_context_mode. Use context / single")
    need_external_emotion_reasoning = dataset == 'iemocap' and emotion_reasoning_enabled
    need_external_personality_reasoning = dataset == 'iemocap' and personality_reasoning_enabled and personality_task_enabled
    external_emotion_targets, external_personality_targets, external_emotion_text_targets, external_emotion_utterance_id_targets = load_external_reasoning_targets(
        external_reasoning_dir,
        need_emotion=need_external_emotion_reasoning,
        need_personality=need_external_personality_reasoning,
    )
    speaker_label_text_set = {
        'iemocap':'Speaker_0, Speaker_1',
        'meld':   'Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8',
        'EmoryNLP':   'Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8',
    }



    emotional_dict = {text_label:num_label for num_label, text_label in enumerate(label_set[dataset])}
    speaker_label_dict = {}
    content_target_dict = {}
    speaker_target_dict = {}
    content_task_dict = {}
    speaker_task_dict = {}
    personality_task_dict = {}
    sentence_dict = {}
    personality_target_dict = {}
    personality_reasoning_task_dict = {}
    content_reasoning_task_dict = {}
    content_teacher_reasoning_dict = {}
    debug_target_utterance_dict = {}
    utterance_id_by_turn = {}
    content_utterance_id_dict = {}
    personality_meta_dict = {}
    experience_bank_context_dict = {}
    opcd_gold_context_dict = {}
    gt_personality_meta_dict = {}
    interlocutor_personality_meta_dict = {}
    derived_shift_meta_dict = {}
    speaker_utterances_by_conv = defaultdict(lambda: defaultdict(list))
    speaker_label_by_conv_raw = {}
    personality_ids_by_conv_setting = defaultdict(lambda: defaultdict(list))
    data = pickle.load(open(f'/n/work6/yiwang/MoodFilm/portable/original_data/{dataset}/{dataset}.pkl','rb'))
    missing_external_emotion_targets = set()
    missing_external_personality_targets = set()
    used_external_emotion_entry_ids = set()

    def _external_emotion_target(conv_id, conv_turn):
        current_text = sentence_dict[conv_id][conv_turn] if conv_id in sentence_dict else ''
        current_norm_text = _normalize_reasoning_match_text(current_text)
        current_utterance_id = utterance_id_by_turn.get((conv_id, int(conv_turn)), '')

        def _entry_matches_current(entry):
            entry_text = _normalize_reasoning_match_text(entry.get('text') if isinstance(entry, dict) else '')
            return not entry_text or not current_norm_text or entry_text == current_norm_text

        def _claim_entry(entry):
            entry_id = id(entry)
            if entry_id in used_external_emotion_entry_ids:
                return None
            used_external_emotion_entry_ids.add(entry_id)
            return entry

        if current_utterance_id and current_utterance_id in external_emotion_utterance_id_targets:
            entry = external_emotion_utterance_id_targets[current_utterance_id]
            if _entry_matches_current(entry):
                target = _claim_entry(entry)
                if target is not None:
                    return target

        for key_dialogue_id in _dialogue_key_variants(conv_id):
            key = (key_dialogue_id, int(conv_turn))
            if key in external_emotion_targets:
                entry = external_emotion_targets[key]
                if _entry_matches_current(entry):
                    target = _claim_entry(entry)
                    if target is not None:
                        return target

        if current_norm_text:
            for key_dialogue_id in _dialogue_key_variants(conv_id):
                text_key = (key_dialogue_id, current_norm_text)
                for entry in external_emotion_text_targets.get(text_key, []):
                    target = _claim_entry(entry)
                    if target is not None:
                        return target

        missing_id = current_utterance_id if current_utterance_id else f'{conv_id}_{conv_turn}'
        missing_external_emotion_targets.add(missing_id)
        return None

    def _external_personality_target(conv_id, speaker_idx):
        speaker = f'Speaker_{speaker_idx}'
        for key_dialogue_id in _dialogue_key_variants(conv_id):
            key = (key_dialogue_id, speaker)
            if key in external_personality_targets:
                return external_personality_targets[key]
        missing_external_personality_targets.add(f'{conv_id}|{speaker}')
        return None

    pseudo_personality_map = load_pseudo_personality_map(personality_pseudo_json_path)
    personality_use_pseudo_first = _is_true_flag(personality_use_pseudo_first)

    use_personality = (personality_meta == 'True' or personality_task_enabled) and dataset == 'iemocap'
    use_personality_condition = (
        personality_condition != 'False'
        and personality_input_mode != 'none'
        and (dataset == 'iemocap' or len(pseudo_personality_map) > 0)
    )
    personality_condition_mode = personality_condition  # 'False', 'True_continuous', 'True_discrete'
    need_personality_map = (
        use_personality
        or use_personality_condition
        or predict_personality_in_chain
        or experience_bank_mode == 'opcd_teacher'
    )
    ocean_personality_map = load_ocean_personality_map(personality_csv_path) if need_personality_map else {}

    def _lookup_gt_personality(conv_id, conv_turn):
        raw_speaker = None
        if dataset == 'iemocap':
            raw_speaker = str(data[0][conv_id][conv_turn]).strip().lower()
            if raw_speaker in ['m', 'f']:
                norm_conv_id = normalize_iemocap_dialogue_id(conv_id)
                return ocean_personality_map.get((norm_conv_id, raw_speaker), None), raw_speaker
        return None, raw_speaker

    def _lookup_pseudo_personality_for_speaker(conv_id, speaker_idx):
        if len(pseudo_personality_map) == 0:
            return None
        keys = [f'{conv_id}|Speaker_{speaker_idx}']
        if dataset == 'iemocap':
            norm_conv_id = normalize_iemocap_dialogue_id(conv_id)
            if norm_conv_id != conv_id:
                keys.append(f'{norm_conv_id}|Speaker_{speaker_idx}')
        for pseudo_key in keys:
            if pseudo_key in pseudo_personality_map:
                return pseudo_personality_map[pseudo_key]
        return None

    def _lookup_pseudo_personality(conv_id, conv_turn):
        if len(pseudo_personality_map) == 0:
            return None
        speaker_idx = speaker_label_dict[conv_id][conv_turn]
        return _lookup_pseudo_personality_for_speaker(conv_id, speaker_idx)

    def _resolve_sample_personality(conv_id, conv_turn):
        raw_speaker = None
        if personality_input_mode == 'none':
            if dataset == 'iemocap':
                raw_speaker = str(data[0][conv_id][conv_turn]).strip().lower()
            return None, raw_speaker

        if personality_input_mode == 'stage1_predicted' or (personality_input_mode == 'auto' and personality_use_pseudo_first):
            if dataset == 'iemocap':
                raw_speaker = str(data[0][conv_id][conv_turn]).strip().lower()
            pseudo_value = _lookup_pseudo_personality(conv_id, conv_turn)
            if pseudo_value is not None:
                return pseudo_value, raw_speaker
            return None, raw_speaker

        if personality_input_mode in ('gt', 'auto') and dataset == 'iemocap':
            return _lookup_gt_personality(conv_id, conv_turn)

        pseudo_value = _lookup_pseudo_personality(conv_id, conv_turn)
        if pseudo_value is not None:
            return pseudo_value, raw_speaker

        return None, raw_speaker

    def _lookup_interlocutor_personality(conv_id, conv_turn):
        if dataset != 'iemocap':
            return None
        raw_speaker = str(data[0][conv_id][conv_turn]).strip().lower()
        if raw_speaker not in ('m', 'f'):
            return None
        other_speaker = 'f' if raw_speaker == 'm' else 'm'
        other_speaker_idx = 1 - int(speaker_label_dict[conv_id][conv_turn])
        if personality_use_pseudo_first and len(pseudo_personality_map) > 0:
            pseudo_value = _lookup_pseudo_personality_for_speaker(conv_id, other_speaker_idx)
            if pseudo_value is not None:
                return pseudo_value
        norm_conv_id = normalize_iemocap_dialogue_id(conv_id)
        return ocean_personality_map.get((norm_conv_id, other_speaker), None)

    def _derive_emotion_shift(conv_id, conv_turn):
        current_label = label_set[dataset][data[1][conv_id][conv_turn]]
        current_speaker = speaker_label_dict[conv_id][conv_turn]
        previous_label = None
        for prev_turn in range(conv_turn - 1, -1, -1):
            if speaker_label_dict[conv_id][prev_turn] == current_speaker:
                previous_label = label_set[dataset][data[1][conv_id][prev_turn]]
                break

        if previous_label is None:
            shift = 'unknown'
        elif previous_label == current_label:
            shift = 'same'
        else:
            valence = {
                'angry': -2,
                'frustrated': -2,
                'sad': -1,
                'neutral': 0,
                'happy': 1,
                'excited': 2,
                'joyful': 1,
                'surprise': 1,
                'fear': -1,
                'disgust': -2,
            }
            prev_v = valence.get(str(previous_label).lower(), 0)
            cur_v = valence.get(str(current_label).lower(), 0)
            if prev_v == 0 or cur_v == 0 or (prev_v > 0) != (cur_v > 0):
                shift = 'change'
            elif abs(cur_v) > abs(prev_v):
                shift = 'intensify'
            elif abs(cur_v) < abs(prev_v):
                shift = 'soften'
            else:
                shift = 'change'

        return {
            'previous_emotion': previous_label,
            'current_emotion': current_label,
            'shift': shift,
        }

    def _build_opcd_gold_context(conv_id, conv_turn, gt_personality, interlocutor_personality, derived_shift):
        return {
            'gold_emotion': label_set[dataset][data[1][conv_id][conv_turn]],
            'target_speaker_gt_personality': gt_personality,
            'interlocutor_gt_personality': interlocutor_personality,
            'derived_shift': derived_shift,
        }

    def _format_ocean_score_lines(personality_value):
        if not isinstance(personality_value, dict):
            return None
        if not set(TRAIT_ORDER).issubset(set(personality_value.keys())):
            return None
        return ', '.join([f'{trait}:{personality_value[trait]}' for trait in TRAIT_ORDER])

    # Compute global thresholds for discretization (from all data)
    personality_thresholds = None
    if use_personality_condition and personality_condition_mode == 'True_discrete' and dataset == 'iemocap':
        all_ids = data[3] + data[4] + data[5]  # train + test + valid
        all_personality_vectors = []
        for conv_id in all_ids:
            for conv_turn in range(len(data[2][conv_id])):
                raw_speaker = str(data[0][conv_id][conv_turn]).strip().lower()
                if raw_speaker in ['m', 'f']:
                    norm_conv_id = normalize_iemocap_dialogue_id(conv_id)
                    pers_vec = ocean_personality_map.get((norm_conv_id, raw_speaker), None)
                    if pers_vec is not None:
                        all_personality_vectors.append(pers_vec)
        if all_personality_vectors:
            personality_thresholds = compute_personality_thresholds(all_personality_vectors)

    # 不同的数据集有不同的speaker_label的处理方式
    #Different datasets have different ways of handling speaker_label
    if dataset == 'iemocap':
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        official_utterance_id_by_turn = build_iemocap_official_utterance_id_map(
            sentence_dict,
            all_conv_id,
            iemocap_root=iemocap_root,
        )
        for conv_id in all_conv_id:
            temp_speaker_list = []
            speaker_turn_counts = defaultdict(int)
            for speaker_label in data[0][conv_id]:
                if speaker_label == 'M':
                    temp_speaker_list.append(0)
                else:
                    temp_speaker_list.append(1)
            for turn_idx, raw_speaker in enumerate(data[0][conv_id]):
                raw = str(raw_speaker).strip().upper()
                if raw in ('M', 'F'):
                    fallback_utterance_id = (
                        f'{normalize_iemocap_dialogue_id(conv_id)}_{raw}{speaker_turn_counts[raw]:03d}'
                    )
                    utterance_id_by_turn[(conv_id, turn_idx)] = official_utterance_id_by_turn.get(
                        (conv_id, turn_idx),
                        fallback_utterance_id,
                    )
                    speaker_turn_counts[raw] += 1
            speaker_label_dict[conv_id] = temp_speaker_list

    elif dataset == 'meld':
        all_conv_id = data[4] + data[5] + data[6]
        sentence_dict = data[3]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list
    
    elif dataset == 'EmoryNLP':
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list

    if dataset == 'iemocap':
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]
    elif dataset == 'meld':
        train_ids, test_ids, valid_ids = data[4], data[5], data[6]
    elif dataset == 'EmoryNLP':
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]

    conv_split_map = {}
    for _cid in train_ids:
        conv_split_map[_cid] = 'train'
    for _cid in test_ids:
        conv_split_map[_cid] = 'test'
    for _cid in valid_ids:
        conv_split_map[_cid] = 'valid'

    trace_records = []
    generated_paths = []

    def build_conversation_window_block(conv_id, start_turn, end_turn):
        lines = []
        for turn_idx in range(start_turn, end_turn + 1):
            turn_speaker = speaker_label_dict[conv_id][turn_idx]
            turn_text = sentence_dict[conv_id][turn_idx]
            lines.append(f'Speaker_{turn_speaker}: {json.dumps(str(turn_text), ensure_ascii=False)}')
        return '\n'.join(lines)

    def build_emotion_reasoning_prompt(
        conv_id,
        conv_turn,
        conversation_window,
        personality_vector=None,
        retrieved_demo=None,
    ):
        target_speaker = f'Speaker_{speaker_label_dict[conv_id][conv_turn]}'
        target_text = sentence_dict[conv_id][conv_turn]
        parts = [
            'Now you are an expert in sentiment and emotion analysis.',
            '',
            'The following conversation involves several speakers:',
            '###',
            conversation_window,
            '###',
            '',
        ]

        consider_items = ['conversational context', 'wording and tone']
        score_text = _format_ocean_score_lines(personality_vector)
        if score_text is not None:
            parts.extend([
                'Target speaker personality profile:',
                score_text,
                '',
                'Interpretation:',
                discretize_personality(personality_vector, thresholds=personality_thresholds),
                '',
            ])
            consider_items.append('personality traits')

        if retrieved_demo is not None:
            parts.extend([
                'A similar utterance and its emotion label are given for reference:',
                f'<Utterance: {json.dumps(str(retrieved_demo["text"]), ensure_ascii=False)}; Emotion: {retrieved_demo["label"]}>',
                '',
            ])
            consider_items.append('similarity to reference example')

        if student_experience_bank_context:
            parts.extend([
                student_experience_bank_context,
                '',
                'Use the experience bank only as probabilistic background knowledge; do not treat it as a rule.',
                '',
            ])
            consider_items.append('personality-interaction experience bank')

        if emotion_reasoning_max_words > 0 and emotion_reasoning_min_words > 0:
            reasoning_length_text = f'around {emotion_reasoning_min_words}-{emotion_reasoning_max_words} words'
        elif emotion_reasoning_max_words > 0:
            reasoning_length_text = f'no more than {emotion_reasoning_max_words} words'
        else:
            reasoning_length_text = 'around 120-220 words'

        if structured_reasoning_enabled:
            if predict_personality_in_chain:
                task_instruction = 'Your task is to infer the target speaker personality, concise structured affective states, and then the final target utterance emotion.'
            else:
                task_instruction = 'Your task is to infer concise structured affective reasoning states and then the final target utterance emotion.'
            format_lines = [
                'Output only the XML tags below. Do not write paragraph reasoning, JSON, bullet points, or extra text.',
                'Use unknown only when a value cannot be inferred.',
                f'Allowed final emotion labels are: <{label_text_set[dataset]}>',
                '',
            ]
            if predict_personality_in_chain:
                format_lines.extend([
                    'Personality must be one compact OCEAN line with continuous 1.0-7.0 scores:',
                    'O=score; C=score; E=score; A=score; N=score',
                    '',
                ])
            format_lines.extend([
                'State meanings:',
                '- <shift>: emotion change or stability, such as same, shift, intensify, soften, unknown',
                '- <trigger>: main emotion source, such as self, other, context, mixed, unknown',
                '- <style>: speaking style, such as neutral, aggressive, calm, playful, anxious, sad, unknown',
                '- <act>: interaction act, such as statement, question, instruction, agreement, laughter, unknown',
                '- <dynamics>: speaker emotion dynamics, such as low, moderate, high, stable, volatile, unknown',
                '- <pattern>: speaker interaction pattern, such as reactive, dominant, supportive, avoidant, unknown',
                '- <dialogue_type>: dialogue type, such as conflict, casual, positive, task, unknown',
                '- <atmosphere>: dialogue atmosphere, such as positive, negative, neutral, mixed, unknown',
                '',
                'IMPORTANT:',
                '- Put only ONE label word inside <label>',
                '- Put only ONE label word inside <answer>',
                '- The label inside <answer> must match the label inside <label>',
                '- Do not mention that a gold label was provided',
                '',
                'Use the following format:',
                'Keep it as one compact XML line.',
                '',
            ])
            if predict_personality_in_chain:
                format_lines.extend([
                    '<personality>O={score};C={score};E={score};A={score};N={score}</personality><label>{emotion_label}</label><states>shift={state};trigger={state};style={state};act={state};dynamics={state};pattern={state};dialogue_type={state};atmosphere={state}</states><answer>{emotion_label}</answer>',
                ])
            else:
                format_lines.extend([
                    '<label>{emotion_label}</label><states>shift={state};trigger={state};style={state};act={state};dynamics={state};pattern={state};dialogue_type={state};atmosphere={state}</states><answer>{emotion_label}</answer>',
                ])
        elif emotion_reasoning_format == 'label_anchor_answer_last':
            task_instruction = 'Your task is to infer the target utterance emotion, output an initial label, then write a concise reasoning explanation, and finally repeat the same label as the final answer.'
            format_lines = [
                'The reasoning should be detailed and useful for training a student model, '
                f'{reasoning_length_text}.',
                'First output exactly one initial label inside <label>, then the reasoning, then exactly one final label inside <answer>.',
                f'Allowed labels are: <{label_text_set[dataset]}>',
                '',
                'IMPORTANT:',
                '- Do not mention that a gold label was provided',
                '- Do not output JSON',
                '- Put only ONE label word inside <label>',
                '- Put only ONE label word inside <answer>',
                '- The label inside <answer> must match the label inside <label>',
                '',
                'Use the following format:',
                '',
                '<label>',
                '{emotion_label}',
                '</label>',
                '<reasoning>',
                '{detailed emotion reasoning}',
                '</reasoning>',
                '<answer>',
                '{emotion_label}',
                '</answer>',
            ]
        else:
            task_instruction = 'Your task is to infer the target utterance emotion and write a high-quality reasoning explanation before the final answer.'
            format_lines = [
                'The reasoning should be detailed and useful for training a student model, '
                f'{reasoning_length_text}.',
                'Then output exactly one final label from:',
                f'<{label_text_set[dataset]}>',
                '',
                'IMPORTANT:',
                '- Do not mention that a gold label was provided',
                '- Do not output JSON',
                '- Put only ONE label word inside <answer>',
                '',
                'Use the following format:',
                '',
                '<reasoning>',
                '{detailed emotion reasoning}',
                '</reasoning>',
                '<answer>',
                '{emotion_label}',
                '</answer>',
            ]

        parts.extend([
            'Target utterance:',
            f'<{target_speaker}: {json.dumps(str(target_text), ensure_ascii=False)}>',
            '',
            task_instruction,
        ])
        if not structured_reasoning_enabled:
            parts.extend([
                'Do not directly state the final emotion label at the beginning of the reasoning text itself.',
                'Reason like a human listener: first consider the local wording and delivery of the target utterance, then the surrounding dialogue context, then how the target speaker reacts to others, and then how the dialogue atmosphere and speaker personality may shape the emotional interpretation.',
                'If audio is available, use prosodic evidence such as vocal intensity, pitch, pacing, hesitation, or calmness when it helps the interpretation.',
                'If the speaker appears to shift, intensify, or calm down compared with earlier turns, explain what changed and why.',
                '',
                'Use these available non-gold cues:',
            ])
        else:
            parts.extend([
                'Infer the states from the available non-gold cues. Do not assume access to gold-side structured cues.',
                '',
                'Use these available non-gold cues:',
            ])
        parts.extend([f'- {item}' for item in consider_items])
        parts.extend([''])
        parts.extend(format_lines)
        return '\n'.join(parts)

    demo_retrieve = None
    if demons == 'True':
        # Build balanced domain base from TRAIN split only.
        raw_domain_examples = []
        for conv_id in train_ids:
            for conv_turn in range(len(sentence_dict[conv_id])):
                sample_key = f'{conv_id}_{conv_turn}'
                clean_text = _clean_demo_text(sentence_dict[conv_id][conv_turn])
                if not clean_text:
                    continue
                personality_vector, _ = _resolve_sample_personality(conv_id, conv_turn)
                raw_domain_examples.append({
                    'key': sample_key,
                    'text': clean_text,
                    'label': label_set[dataset][data[1][conv_id][conv_turn]],
                    'personality': personality_vector,
                })
        balanced_domain_base = build_balanced_domain_base(raw_domain_examples, label_set[dataset])
        demo_retrieve = build_demo_retriever(
            balanced_domain_base,
            backend=demo_backend,
            model_name=demo_model_name,
            top_k=demo_top_k,
            alpha=demo_alpha,
            exclude_same_dialogue=(demo_exclude_same_dialogue == 'True'),
            personality_weight=demo_personality_weight,
        )

    # 对conversation的utterance进行处理，其中index_w用于处理窗口大小设置下面的起始index
    # Process the utterances in the conversation, where 'index_w' is used to handle the starting index under the window size setting.
    for conv_id in all_conv_id:
        for conv_turn in range(len(sentence_dict[conv_id])):
            temp_content_str = 'Now you are an expert in sentiment and emotion analysis. '
            demo_suffix = ''
            retrieved_demo = None
            retrieve_meta = {}
            if demons == 'True':
                sample_key = f'{conv_id}_{conv_turn}'
                query_text = _clean_demo_text(sentence_dict[conv_id][conv_turn])
                query_personality, _ = _resolve_sample_personality(conv_id, conv_turn)
                if demo_retrieve is not None:
                    retrieved_demo, retrieve_meta = demo_retrieve(
                        query_text,
                        sample_key,
                        query_personality=query_personality,
                        return_meta=True,
                    )
                else:
                    retrieved_demo, retrieve_meta = None, {}
                demo_suffix = format_retrieved_demonstration(
                    retrieved_demo,
                    personality_condition_mode=personality_condition_mode,
                    thresholds=personality_thresholds,
                )
                if demo_trace == 'True':
                    trace_item = {
                        'sample_key': sample_key,
                        'split': conv_split_map.get(conv_id, 'unknown'),
                        'query_text': query_text,
                        'query_label': label_set[dataset][data[1][conv_id][conv_turn]],
                        'retrieved_key': retrieved_demo['key'] if retrieved_demo is not None else None,
                        'retrieved_text': retrieved_demo['text'] if retrieved_demo is not None else None,
                        'retrieved_label': retrieved_demo['label'] if retrieved_demo is not None else None,
                        'query_personality': query_personality,
                        'retrieved_personality': retrieved_demo.get('personality') if retrieved_demo is not None else None,
                    }
                    trace_item.update(retrieve_meta)
                    trace_records.append(trace_item)
            content_target = label_set[dataset][data[1][conv_id][conv_turn]]
            sample_key = f'{conv_id}_{conv_turn}'
            target_personality_vector, _ = _lookup_gt_personality(conv_id, conv_turn)
            interlocutor_personality_vector = _lookup_interlocutor_personality(conv_id, conv_turn)
            derived_shift = _derive_emotion_shift(conv_id, conv_turn)
            if experience_bank_mode == 'opcd_teacher' and teacher_experience_bank_context:
                experience_bank_context_dict[sample_key] = teacher_experience_bank_context
            if opcd_teacher_include_gold_context:
                opcd_gold_context_dict[sample_key] = _build_opcd_gold_context(
                    conv_id,
                    conv_turn,
                    target_personality_vector,
                    interlocutor_personality_vector,
                    derived_shift,
                )
            if target_personality_vector is not None:
                gt_personality_meta_dict[sample_key] = target_personality_vector
            if interlocutor_personality_vector is not None:
                interlocutor_personality_meta_dict[sample_key] = interlocutor_personality_vector
            if derived_shift is not None:
                derived_shift_meta_dict[sample_key] = derived_shift
            if dataset == 'iemocap':
                default_utterance_id = utterance_id_by_turn.get((conv_id, conv_turn))
                if default_utterance_id:
                    content_utterance_id_dict[sample_key] = default_utterance_id
            if need_external_emotion_reasoning:
                external_entry = _external_emotion_target(conv_id, conv_turn)
                if external_entry is not None:
                    if isinstance(external_entry, dict) and external_entry.get('utterance_id'):
                        content_utterance_id_dict[sample_key] = external_entry.get('utterance_id')
                    raw_external_target = external_entry.get('target') if isinstance(external_entry, dict) else external_entry
                    teacher_reasoning = _extract_tagged_reasoning(raw_external_target)
                    if teacher_reasoning:
                        content_teacher_reasoning_dict[sample_key] = _clean_emotion_reasoning_text(
                            teacher_reasoning,
                            clean_hidden_cues=emotion_reasoning_clean_hidden_cues,
                            min_words=0,
                            max_words=0,
                        )
                    content_target = _format_emotion_reasoning_target(
                        raw_external_target,
                        label_set[dataset],
                        output_format=emotion_reasoning_format,
                        min_words=emotion_reasoning_min_words,
                        max_words=emotion_reasoning_max_words,
                        clean_hidden_cues=emotion_reasoning_clean_hidden_cues,
                        structured_reasoning=structured_reasoning_enabled,
                        structured_cues=external_entry.get('structured_cues') if isinstance(external_entry, dict) else None,
                        personality_vector=target_personality_vector,
                        predict_personality_in_chain=predict_personality_in_chain,
                    )
            content_target_dict[sample_key] = content_target
            target_speaker_label = speaker_label_dict[conv_id][conv_turn]
            target_text = sentence_dict[conv_id][conv_turn]
            target_utterance = f'Speaker_{target_speaker_label}:"{target_text}"'
            personality_vector = None
            raw_speaker = None
            if need_personality_map:
                personality_vector, raw_speaker = _resolve_sample_personality(conv_id, conv_turn)

            cond_prompt = None
            if use_personality_condition and personality_vector is not None:
                cond_prompt = build_personality_condition_prompt(
                    personality_vector,
                    personality_condition_mode,
                    thresholds=personality_thresholds,
                )

            if emotion_context_mode == 'single':
                index_w = conv_turn
                temp_content_str += 'The target utterance is marked between ### markers. ### '
                temp_content_str += f'\t {target_utterance}'
                temp_content_str += ' ### '
                if cond_prompt is not None:
                    temp_content_str += f'{cond_prompt} '
                if student_experience_bank_context:
                    temp_content_str += f'{student_experience_bank_context} '
                if emotion_prompt_output_format == 'short_reasoning_answer':
                    temp_content_str += (
                        f'Write a focused {short_prompt_min_words}-{short_prompt_max_words} word emotion explanation for the target utterance. '
                        'Ground it in the target wording, the immediate dialogue context, speaker personality cues, and any visible experience-bank tendencies. '
                        'Mention only cues that are visible in the prompt; do not invent facts, do not quote long passages, and do not repeat the same point. '
                        f'Then select one emotional label from <{label_text_set[dataset]}>. '
                        'Output exactly this format with one concise reasoning paragraph and one final label: '
                        '<reasoning>{short explanation}</reasoning><answer>{label}</answer>'
                    )
                else:
                    temp_content_str += (
                        f'Please select the emotional label of the target utterance from <{label_text_set[dataset]}>. '
                        'Output only one label word from the candidate set and nothing else:'
                    )
            else:
                temp_content_str += 'The following conversation involves several speakers. Conversation: ### '
                index_w = max(conv_turn-window, 0)
                for speaker_label, sub_sent in zip(speaker_label_dict[conv_id][index_w:conv_turn+1], sentence_dict[conv_id][index_w:conv_turn+1]):
                    temp_content_str += (f'\t Speaker_{speaker_label}:"{sub_sent}"')
                    # temp_id_task_str += (f'\t Speaker_?:"{sub_sent}"')
                temp_content_str += ' ### '
                if cond_prompt is not None:
                    temp_content_str += f'{cond_prompt} '
                if student_experience_bank_context:
                    temp_content_str += f'{student_experience_bank_context} '
                if emotion_prompt_output_format == 'short_reasoning_answer':
                    temp_content_str += (
                        f'Write a focused {short_prompt_min_words}-{short_prompt_max_words} word emotion explanation for <{target_utterance}>. '
                        'Ground it in the target wording, the immediate dialogue context, speaker personality cues, and any visible experience-bank tendencies. '
                        'Mention only cues that are visible in the prompt; do not invent facts, do not quote long passages, and do not repeat the same point. '
                        f'Then select one emotional label from <{label_text_set[dataset]}>. '
                        'Output exactly this format with one concise reasoning paragraph and one final label: '
                        '<reasoning>{short explanation}</reasoning><answer>{label}</answer>'
                    )
                else:
                    temp_content_str += (
                        f'Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}>. '
                        'Output only one label word from the candidate set and nothing else:'
                    )
            if demo_suffix:
                temp_content_str += (' ' + demo_suffix)
            # demon
            # temp_content_str += f'Here is a demonstration. {demon}'
            if predictions == 'True' and conv_turn > 0:
                temp_predict_str = '\t'.join(temp_content_str.split('\t')[:-1])
                temp_predict_str += '###'
                temp_predict_str += f'Based on the above historical utterances, next utterance is spoken by <{speaker_label_dict[conv_id][conv_turn]}>, please predict the emotion states of <{speaker_label_dict[conv_id][conv_turn]}> from <{label_text_set[dataset]}>:'
                temp_content_str += ('***'+temp_predict_str)
            elif predictions == 'True' and conv_turn == 0:
                temp_content_str += ('***'+temp_content_str)
            content_task_dict[f'{conv_id}_{conv_turn}'] = temp_content_str
            if emotion_reasoning_enabled and (
                not need_external_emotion_reasoning
                or _target_has_valid_emotion_reasoning_target(
                    content_target,
                    structured_reasoning=structured_reasoning_enabled,
                )
            ):
                content_reasoning_task_dict[f'{conv_id}_{conv_turn}'] = build_emotion_reasoning_prompt(
                    conv_id=conv_id,
                    conv_turn=conv_turn,
                    conversation_window=build_conversation_window_block(conv_id, index_w, conv_turn),
                    personality_vector=personality_vector if use_personality_condition else None,
                    retrieved_demo=retrieved_demo,
                )

            if use_personality and personality_vector is not None:
                sample_key = f'{conv_id}_{conv_turn}'
                personality_meta_dict[sample_key] = {
                    'speaker_id': f'Speaker_{speaker_label_dict[conv_id][conv_turn]}',
                    'personality': personality_vector
                }
                if personality_task_enabled and dataset == 'iemocap' and raw_speaker in ['m', 'f']:
                    speaker_idx = speaker_label_dict[conv_id][conv_turn]
                    speaker_utterances_by_conv[conv_id][raw_speaker].append(sentence_dict[conv_id][conv_turn])
                    speaker_label_by_conv_raw[(conv_id, raw_speaker)] = speaker_idx

    # Build personality task samples for all supported settings; split-specific
    # train/test/valid selection happens below.
    if personality_task_enabled and use_personality and dataset == 'iemocap':
        continuous_instruction = (
            'Predict the OCEAN personality vector [O, C, E, A, N] of this speaker. '
            'Use the Big Five meanings: O=Openness, C=Conscientiousness, E=Extraversion, '
            'A=Agreeableness, N=Neuroticism. '
            'Use a 1.0-7.0 Likert scale where 1.0 is very low, 4.0 is medium, and 7.0 is very high. '
            'Each output value must be a number within [1.0, 7.0]. '
            'Output only a valid JSON array with exactly 5 numbers in this exact order'
        )
        discrete_instruction = (
            'Trait order is [O, C, E, A, N]. '
            'Do not continue or rewrite the input. '
            'Output only one valid JSON object with keys O,C,E,A,N and values from {low, medium, high}, and nothing else:'
        )

        def build_full_dialogue_block(conv_id, target_speaker_idx, line_separator=' '):
            dialogue_lines = []
            for turn_idx, turn_text in enumerate(sentence_dict[conv_id]):
                turn_speaker = speaker_label_dict[conv_id][turn_idx]
                marker = '*' if turn_speaker == target_speaker_idx else ''
                dialogue_lines.append(
                    f'Speaker_{turn_speaker}{marker}: {json.dumps(str(turn_text), ensure_ascii=False)}'
                )
            return line_separator.join(dialogue_lines)

        def build_personality_prompt(setting, speaker_idx, utt_list=None, utterance_text=None, dialogue_block=None):
            speaker_tag = f'Speaker_{speaker_idx}'
            if setting == 'utterance':
                context_text = (
                    f'Given this utterance from {speaker_tag}: '
                    f'{json.dumps(str(utterance_text), ensure_ascii=False)} '
                )
            elif setting == 'conversation':
                utterance_str = ' '.join([json.dumps(str(u), ensure_ascii=False) for u in (utt_list or [])])
                context_text = (
                    f'Given all utterances from {speaker_tag} in this conversation: {utterance_str} '
                )
            elif setting == 'full':
                context_text = (
                    f'Given the full conversation where target speaker turns are marked as <{speaker_tag}*>: '
                    f'<{dialogue_block}> '
                )
            else:
                raise ValueError(f"Unsupported personality setting: {setting}")

            if resolved_personality_mode == 'raw':
                return 'Now you are expert of personality analysis. ' + context_text + continuous_instruction
            return 'Now you are expert of personality analysis. ' + context_text + discrete_instruction

        def build_personality_reasoning_prompt(setting, speaker_idx, utt_list=None, dialogue_block=None):
            speaker_tag = f'Speaker_{speaker_idx}'
            if setting == 'conversation':
                conversation = '\n'.join([
                    f'{speaker_tag}: {json.dumps(str(u), ensure_ascii=False)}'
                    for u in (utt_list or [])
                ])
            elif setting == 'full':
                conversation = str(dialogue_block or '')
            else:
                return None

            return '\n'.join([
                'Now you are an expert in personality psychology and conversation analysis.',
                '',
                'The following conversation involves several speakers:',
                '###',
                conversation,
                '###',
                '',
                f'The target speaker is {speaker_tag}.',
                '',
                "Your task is to infer the target speaker's Big Five personality profile and write a high-quality reasoning explanation before the final OCEAN answer.",
                '',
                'Guidelines:',
                '- Do not simply restate scores at the beginning',
                '- Reason from stable behavior across the whole dialogue, not temporary emotion',
                '- Explain how the speaker responds to others: supporting, dominating, avoiding, or reacting',
                '- Explain how emotional expression changes over time and how the dialogue atmosphere shapes behavior',
                '- Connect local utterance evidence to broader personality tendencies without over-relying on one extreme sentence',
                '- If audio is available, use prosodic evidence such as pitch, energy, speaking rate, hesitation, or vocal intensity when it helps',
                '',
                'The reasoning should be detailed and useful for training a student model, around 150-260 words.',
                'Then output the final OCEAN scores in the range 1.0-7.0.',
                '',
                'IMPORTANT:',
                '- Do not mention that a gold personality profile was provided',
                '- Do not output JSON',
                '- Put the final scores only inside <answer>',
                '',
                'Use the following format:',
                '',
                '<reasoning>',
                '{detailed personality reasoning}',
                '</reasoning>',
                '<answer>',
                'O:x.xxx, C:x.xxx, E:x.xxx, A:x.xxx, N:x.xxx',
                '</answer>',
            ])

        def add_personality_sample(setting, sample_key, conv_id, speaker_idx, personality_vector, **prompt_kwargs):
            debug_target_utterance = prompt_kwargs.pop('debug_target_utterance', None)
            personality_task_key = f'Personality_{setting}_{sample_key}'
            personality_task_dict[personality_task_key] = build_personality_prompt(
                setting,
                speaker_idx,
                **prompt_kwargs,
            )
            personality_target = personality_to_target(
                personality_vector,
                mode=resolved_personality_mode,
            )
            if need_external_personality_reasoning and setting in ('conversation', 'full'):
                external_target = _external_personality_target(conv_id, speaker_idx)
                if external_target is not None:
                    personality_target = external_target
            personality_target_dict[personality_task_key] = personality_target
            if debug_target_utterance is not None:
                debug_target_utterance_dict[personality_task_key] = debug_target_utterance
            reasoning_prompt = build_personality_reasoning_prompt(
                setting,
                speaker_idx,
                utt_list=prompt_kwargs.get('utt_list'),
                dialogue_block=build_full_dialogue_block(conv_id, speaker_idx, line_separator='\n')
                if setting == 'full'
                else prompt_kwargs.get('dialogue_block'),
            )
            if (
                personality_reasoning_enabled
                and reasoning_prompt is not None
                and (not need_external_personality_reasoning or _target_has_reasoning_and_answer(personality_target))
            ):
                personality_reasoning_task_dict[personality_task_key] = reasoning_prompt
            personality_ids_by_conv_setting[setting][conv_id].append(personality_task_key)

        for conv_id in all_conv_id:
            norm_conv_id = normalize_iemocap_dialogue_id(conv_id)
            for conv_turn, utterance_text in enumerate(sentence_dict[conv_id]):
                raw_speaker = str(data[0][conv_id][conv_turn]).strip().lower()
                if raw_speaker not in ['m', 'f']:
                    continue
                personality_vector = ocean_personality_map.get((norm_conv_id, raw_speaker), None)
                if personality_vector is None:
                    continue
                speaker_idx = speaker_label_dict[conv_id][conv_turn]
                add_personality_sample(
                    'utterance',
                    f'{conv_id}_{conv_turn}',
                    conv_id,
                    speaker_idx,
                    personality_vector,
                    utterance_text=utterance_text,
                    debug_target_utterance=utterance_text,
                )

        for conv_id, speaker_map in speaker_utterances_by_conv.items():
            norm_conv_id = normalize_iemocap_dialogue_id(conv_id)
            for raw_speaker, utt_list in speaker_map.items():
                if len(utt_list) == 0:
                    continue
                personality_vector = ocean_personality_map.get((norm_conv_id, raw_speaker), None)
                if personality_vector is None:
                    continue
                speaker_idx = speaker_label_by_conv_raw.get((conv_id, raw_speaker), None)
                if speaker_idx is None:
                    continue

                add_personality_sample(
                    'conversation',
                    f'{conv_id}_{speaker_idx}',
                    conv_id,
                    speaker_idx,
                    personality_vector,
                    utt_list=utt_list,
                    debug_target_utterance=list(utt_list),
                )
                add_personality_sample(
                    'full',
                    f'{conv_id}_{speaker_idx}',
                    conv_id,
                    speaker_idx,
                    personality_vector,
                    dialogue_block=build_full_dialogue_block(conv_id, speaker_idx),
                    debug_target_utterance=list(utt_list),
                )

    path_suffix = ''
    if personality_condition_mode == 'True_discrete':
        path_suffix = '_personality_discrete'
    elif personality_condition_mode == 'True_continuous':
        path_suffix = '_personality_continuous'
    if personality_use_pseudo_first and len(pseudo_personality_map) > 0 and path_suffix:
        path_suffix += '_predicted'

    run_root = _build_run_root(dataset, run_id, run_root=run_root)
    os.makedirs(run_root, exist_ok=True)

    for conv_id in all_conv_id:
        for conv_turn in range(len(data[2][conv_id])):
            temp_speaker_task_str = (
                'Now you are expert of sentiment and emotional analysis. '
                f'Please select the Speaker label of the utterance <Speaker: {sentence_dict[conv_id][conv_turn]}> '
                f'from <{speaker_label_text_set[dataset]}>. '
                'Output only one speaker label and nothing else:'
            )
            speaker_target_dict[f'Speaker_{conv_id}_{conv_turn}'] = f'Speaker_{speaker_label_dict[conv_id][conv_turn]}'
            speaker_task_dict[f'Speaker_{conv_id}_{conv_turn}'] = temp_speaker_task_str

    def debug_utterance_for_turn_key(turn_key):
        if turn_key is None or '_' not in str(turn_key):
            return None
        conv_id, turn_str = str(turn_key).rsplit('_', 1)
        if conv_id not in sentence_dict:
            return None
        try:
            turn_idx = int(turn_str)
        except ValueError:
            return None
        if turn_idx < 0 or turn_idx >= len(sentence_dict[conv_id]):
            return None
        return sentence_dict[conv_id][turn_idx]

    def build_content_record(sample_id, strip_prediction=False, reasoning=False):
        input_text = content_reasoning_task_dict.get(sample_id, content_task_dict[sample_id]) if reasoning else content_task_dict[sample_id]
        if strip_prediction and '***' in input_text:
            input_text = input_text.split('***')[0]
        record = {
            'input': f'{input_text}',
            'target': f'{content_target_dict[sample_id]}',
            'sample_id': sample_id,
        }
        debug_target = debug_utterance_for_turn_key(sample_id)
        if debug_target is not None:
            record['debug_target_utterance'] = debug_target
        utterance_id = content_utterance_id_dict.get(sample_id)
        if utterance_id is None and dataset == 'iemocap' and '_' in str(sample_id):
            conv_id, turn_str = str(sample_id).rsplit('_', 1)
            try:
                utterance_id = utterance_id_by_turn.get((conv_id, int(turn_str)))
            except ValueError:
                utterance_id = None
        if utterance_id:
            record['utterance_id'] = utterance_id
        if sample_id in content_teacher_reasoning_dict:
            record['teacher_reasoning'] = content_teacher_reasoning_dict[sample_id]
        if sample_id in experience_bank_context_dict:
            record['experience_bank_context'] = experience_bank_context_dict[sample_id]
        if sample_id in opcd_gold_context_dict:
            record['opcd_gold_context'] = opcd_gold_context_dict[sample_id]
        if sample_id in gt_personality_meta_dict:
            record['gt_personality'] = gt_personality_meta_dict[sample_id]
        if sample_id in interlocutor_personality_meta_dict:
            record['interlocutor_personality'] = interlocutor_personality_meta_dict[sample_id]
        if sample_id in derived_shift_meta_dict:
            record['derived_shift'] = derived_shift_meta_dict[sample_id]
        if personality_meta == 'True' and sample_id in personality_meta_dict:
            record['speaker_id'] = personality_meta_dict[sample_id]['speaker_id']
            record['personality'] = personality_meta_dict[sample_id]['personality']
        return record

    def build_speaker_record(sample_id):
        record = {
            'input': f'{speaker_task_dict[sample_id]}',
            'target': f'{speaker_target_dict[sample_id]}',
            'sample_id': sample_id,
        }
        debug_target = debug_utterance_for_turn_key(str(sample_id)[len('Speaker_'):])
        if debug_target is not None:
            record['debug_target_utterance'] = debug_target
        return record

    def build_personality_record(sample_id, reasoning=False):
        input_text = personality_reasoning_task_dict.get(sample_id, personality_task_dict[sample_id]) if reasoning else personality_task_dict[sample_id]
        record = {
            'input': f'{input_text}',
            'target': f'{personality_target_dict[sample_id]}',
            'sample_id': sample_id,
        }
        if sample_id in debug_target_utterance_dict:
            record['debug_target_utterance'] = debug_target_utterance_dict[sample_id]
        return record

    new_train_id, new_test_id, new_valid_id = [], [], []
    # new_train_target, new_test_target, new_valid_target = [], [], []
    for train_id in train_ids:
        for conv_turn in range(len(sentence_dict[train_id])):
            new_train_id.append(f'{train_id}_{conv_turn}')
            if speaker_task == 'True' or speaker_task == 'True_mixed':
                new_train_id.append(f'Speaker_{train_id}_{conv_turn}')
        if personality_task_enabled and use_personality:
            pers_ids = personality_ids_by_conv_setting[personality_train_setting].get(train_id, [])
            for _ in range(personality_oversample_factor):
                new_train_id.extend(pers_ids)
            
        
    for test_id in test_ids:
        for conv_turn in range(len(sentence_dict[test_id])):
            new_test_id.append(f'{test_id}_{conv_turn}')
            if speaker_task == 'True' or speaker_task == 'True_mixed':
                new_test_id.append(f'Speaker_{test_id}_{conv_turn}')
        if personality_task_enabled and use_personality:
            new_test_id.extend(personality_ids_by_conv_setting[personality_inference_setting].get(test_id, []))
    
    for valid_id in valid_ids:
        for conv_turn in range(len(sentence_dict[valid_id])):
            new_valid_id.append(f'{valid_id}_{conv_turn}')
            if speaker_task == 'True' or speaker_task == 'True_mixed':
                new_valid_id.append(f'Speaker_{valid_id}_{conv_turn}')
        if personality_task_enabled and use_personality:
            new_valid_id.extend(personality_ids_by_conv_setting[personality_inference_setting].get(valid_id, []))


            # new_train_id =  speaker_task_dict.keys() + content_dict.keys()
    # dataset_list = ['train', 'test', 'valid']
    if predictions == 'False':
        if speaker_task == 'True_mixed':
            data_path = os.path.join(run_root, f'stage_mixed_speaker_window{path_suffix}')
            generated_paths.append(data_path)
            os.makedirs(data_path, exist_ok=True)
            with open(f'{data_path}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in content_task_dict:
                        f_train.write(json.dumps(build_content_record(train_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
                    else:
                        if train_id in speaker_task_dict:
                            f_train.write(json.dumps(build_speaker_record(train_id), ensure_ascii=False)+ '\n')
                        elif train_id in personality_task_dict:
                            f_train.write(json.dumps(build_personality_record(train_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in content_task_dict:
                        f_test.write(json.dumps(build_content_record(test_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
                    # else:
                        # f_test.write(json.dumps({'input':f'{speaker_task_dict[test_id]}','target':f'{speaker_target_dict[test_id]}'}, ensure_ascii=False)+ '\n')

            with open(f'{data_path}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in content_task_dict:
                        f_valid.write(json.dumps(build_content_record(valid_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
                    else:
                        if valid_id in speaker_task_dict:
                            f_valid.write(json.dumps(build_speaker_record(valid_id), ensure_ascii=False)+ '\n')
                        elif valid_id in personality_task_dict:
                            f_valid.write(json.dumps(build_personality_record(valid_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

        elif speaker_task == 'True':
            data_path_speaker = os.path.join(run_root, f'stage1_speaker{path_suffix}')
            generated_paths.append(data_path_speaker)
            os.makedirs(data_path_speaker, exist_ok=True)
            with open(f'{data_path_speaker}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in speaker_task_dict:
                        f_train.write(json.dumps(build_speaker_record(train_id), ensure_ascii=False)+ '\n')
                    elif train_id in personality_task_dict:
                        f_train.write(json.dumps(build_personality_record(train_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path_speaker}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in speaker_task_dict:
                        f_test.write(json.dumps(build_speaker_record(test_id), ensure_ascii=False)+ '\n')
                    elif test_id in personality_task_dict:
                        f_test.write(json.dumps(build_personality_record(test_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')
            
            with open(f'{data_path_speaker}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in speaker_task_dict:
                        f_valid.write(json.dumps(build_speaker_record(valid_id), ensure_ascii=False)+ '\n')
                    elif valid_id in personality_task_dict:
                        f_valid.write(json.dumps(build_personality_record(valid_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

            data_path_window = os.path.join(run_root, f'stage2_window_demon{path_suffix}' if demons == 'True' else f'stage2_window{path_suffix}')
            generated_paths.append(data_path_window)
            os.makedirs(data_path_window, exist_ok=True)
            with open(f'{data_path_window}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in content_task_dict:
                        f_train.write(json.dumps(build_content_record(train_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path_window}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in content_task_dict:
                        f_test.write(json.dumps(build_content_record(test_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')


            with open(f'{data_path_window}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in content_task_dict:
                        f_valid.write(json.dumps(build_content_record(valid_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')

            data_path = data_path_speaker + "," +data_path_window         
                        

        elif speaker_task == 'None' and personality_task_enabled and use_personality:
            data_path = os.path.join(run_root, f'stage1_personality{path_suffix}')
            generated_paths.append(data_path)
            os.makedirs(data_path, exist_ok=True)

            with open(f'{data_path}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in personality_task_dict:
                        f_train.write(json.dumps(build_personality_record(train_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in personality_task_dict:
                        f_test.write(json.dumps(build_personality_record(test_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in personality_task_dict:
                        f_valid.write(json.dumps(build_personality_record(valid_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

        elif speaker_task == 'None' and demons == 'False':
            data_path = os.path.join(run_root, f'stage_window{path_suffix}')
            generated_paths.append(data_path)
            os.makedirs(data_path, exist_ok=True)

            with open(f'{data_path}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in content_task_dict:
                        f_train.write(json.dumps(build_content_record(train_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in content_task_dict:
                        f_test.write(json.dumps(build_content_record(test_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in content_task_dict:
                        f_valid.write(json.dumps(build_content_record(valid_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
        
        elif speaker_task == 'None' and demons == 'True':
            data_path = os.path.join(run_root, f'stage_window_demon{path_suffix}')
            generated_paths.append(data_path)
            os.makedirs(data_path, exist_ok=True)
            
            with open(f'{data_path}/train.json', 'w') as f_train:
                for train_id in new_train_id:
                    if train_id in content_task_dict:
                        f_train.write(json.dumps(build_content_record(train_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/test.json', 'w') as f_test:
                for test_id in new_test_id:
                    if test_id in content_task_dict:
                        f_test.write(json.dumps(build_content_record(test_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')

            with open(f'{data_path}/valid.json', 'w') as f_valid:
                for valid_id in new_valid_id:
                    if valid_id in content_task_dict:
                        f_valid.write(json.dumps(build_content_record(valid_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
    elif predictions == 'True':
        if speaker_task == 'None' and demons == 'False':
            data_path = os.path.join(run_root, f'predict_window{path_suffix}')
        elif speaker_task == 'None' and demons == 'True':
            data_path = os.path.join(run_root, f'predict_window_demon{path_suffix}')
        generated_paths.append(data_path)
        os.makedirs(data_path, exist_ok=True)
            
        with open(f'{data_path}/train.json', 'w') as f_train:
            for train_id in new_train_id:
                if train_id in content_task_dict:
                    f_train.write(json.dumps(build_content_record(train_id, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
                elif train_id in personality_task_dict:
                    f_train.write(json.dumps(build_personality_record(train_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

        with open(f'{data_path}/test.json', 'w') as f_test:
            for test_id in new_test_id:
                if test_id in content_task_dict:
                    f_test.write(json.dumps(build_content_record(test_id, strip_prediction=True, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
                elif test_id in personality_task_dict:
                    f_test.write(json.dumps(build_personality_record(test_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

        with open(f'{data_path}/valid.json', 'w') as f_valid:
            for valid_id in new_valid_id:
                if valid_id in content_task_dict:
                    f_valid.write(json.dumps(build_content_record(valid_id, strip_prediction=True, reasoning=emotion_reasoning_enabled), ensure_ascii=False)+ '\n')
                elif valid_id in personality_task_dict:
                    f_valid.write(json.dumps(build_personality_record(valid_id, reasoning=personality_reasoning_enabled), ensure_ascii=False)+ '\n')

    if demons == 'True' and demo_trace == 'True' and len(trace_records) > 0:
        if speaker_task == 'True':
            trace_path = f'{data_path_window}/demo_retrieval_trace.jsonl'
        else:
            trace_path = f'{data_path}/demo_retrieval_trace.jsonl'
        with open(trace_path, 'w', encoding='utf-8') as f_trace:
            for item in trace_records:
                f_trace.write(json.dumps(item, ensure_ascii=False) + '\n')

    module_usage = {
        'speaker_task': speaker_task,
        'personality_meta': personality_meta,
        'personality_task': personality_task,
        'personality_task_enabled': 'True' if personality_task_enabled else 'False',
        'personality_task_mode': resolved_personality_mode,
        'personality_condition': personality_condition,
        'personality_pseudo_json_path': personality_pseudo_json_path,
        'personality_use_pseudo_first': 'True' if personality_use_pseudo_first else 'False',
        'personality_train_setting': personality_train_setting,
        'personality_inference_setting': personality_inference_setting,
        'personality_reasoning': 'True' if personality_reasoning_enabled else 'False',
        'emotion_reasoning': 'True' if emotion_reasoning_enabled else 'False',
        'emotion_context_mode': emotion_context_mode,
        'domain_base': demons,
        'emotion_prediction': predictions,
        'demo_backend': demo_backend,
        'demo_model_name': demo_model_name,
        'demo_trace': demo_trace,
        'personality_oversample_factor': personality_oversample_factor,
        'personality_reasoning': 'True' if personality_reasoning_enabled else 'False',
        'emotion_reasoning': 'True' if emotion_reasoning_enabled else 'False',
        'emotion_reasoning_format': emotion_reasoning_format,
        'emotion_reasoning_min_words': emotion_reasoning_min_words,
        'emotion_reasoning_max_words': emotion_reasoning_max_words,
        'emotion_reasoning_clean_hidden_cues': 'True' if emotion_reasoning_clean_hidden_cues else 'False',
        'structured_reasoning': 'True' if structured_reasoning_enabled else 'False',
        'predict_personality_in_chain': 'True' if predict_personality_in_chain else 'False',
        'personality_input_mode': personality_input_mode,
        'generate_reasoning_from_states': 'True' if generate_reasoning_from_states else 'False',
        'emotion_prompt_output_format': emotion_prompt_output_format,
        'experience_bank_mode': experience_bank_mode,
        'experience_bank_path': experience_bank_path,
        'experience_bank_confidence': experience_bank_confidence,
        'experience_bank_max_items': int(experience_bank_max_items or 0),
        'experience_bank_loaded_items': len(experience_bank_items),
        'opcd_teacher_include_gold_context': 'True' if opcd_teacher_include_gold_context else 'False',
    }
    active_modules = []
    if speaker_task in ('True', 'True_mixed'):
        active_modules.append('speaker_task')
    if personality_meta == 'True':
        active_modules.append('personality_meta')
    if personality_task_enabled:
        active_modules.append('personality_task')
    if personality_condition != 'False':
        active_modules.append('personality_condition')
    if demons == 'True':
        active_modules.append('domain_base')
    if predictions == 'True':
        active_modules.append('emotion_prediction')
    if demo_trace == 'True':
        active_modules.append('demo_trace')
    if personality_reasoning_enabled:
        active_modules.append('personality_reasoning')
    if emotion_reasoning_enabled:
        active_modules.append(f'emotion_reasoning:{emotion_reasoning_format}')
    if structured_reasoning_enabled:
        active_modules.append('structured_reasoning')
    if predict_personality_in_chain:
        active_modules.append('personality_states_chain')
    if emotion_prompt_output_format != 'label_only':
        active_modules.append(f'emotion_prompt_output:{emotion_prompt_output_format}')
    if experience_bank_mode != 'none':
        active_modules.append(f'experience_bank:{experience_bank_mode}')
    if opcd_teacher_include_gold_context:
        active_modules.append('opcd_gold_context')

    manifest = {
        'run_id': run_id,
        'run_root': run_root,
        'dataset': dataset,
        'historical_window': window,
        'speaker_task': speaker_task,
        'domain_base': demons,
        'emotion_prediction': predictions,
        'personality_meta': personality_meta,
        'personality_task': personality_task,
        'personality_mode': resolved_personality_mode,
        'personality_condition': personality_condition,
        'personality_train_setting': personality_train_setting,
        'personality_inference_setting': personality_inference_setting,
        'personality_reasoning': 'True' if personality_reasoning_enabled else 'False',
        'emotion_reasoning': 'True' if emotion_reasoning_enabled else 'False',
        'emotion_reasoning_format': emotion_reasoning_format,
        'emotion_reasoning_min_words': emotion_reasoning_min_words,
        'emotion_reasoning_max_words': emotion_reasoning_max_words,
        'emotion_reasoning_clean_hidden_cues': 'True' if emotion_reasoning_clean_hidden_cues else 'False',
        'structured_reasoning': 'True' if structured_reasoning_enabled else 'False',
        'predict_personality_in_chain': 'True' if predict_personality_in_chain else 'False',
        'personality_input_mode': personality_input_mode,
        'generate_reasoning_from_states': 'True' if generate_reasoning_from_states else 'False',
        'emotion_prompt_output_format': emotion_prompt_output_format,
        'experience_bank_mode': experience_bank_mode,
        'experience_bank_path': experience_bank_path,
        'experience_bank_confidence': experience_bank_confidence,
        'experience_bank_max_items': int(experience_bank_max_items or 0),
        'experience_bank_loaded_items': len(experience_bank_items),
        'opcd_teacher_include_gold_context': 'True' if opcd_teacher_include_gold_context else 'False',
        'emotion_context_mode': emotion_context_mode,
        'external_reasoning_dir': external_reasoning_dir,
        'external_reasoning_counts': {
            'emotion': len(external_emotion_targets),
            'personality': len(external_personality_targets),
        },
        'external_reasoning_missing': {
            'emotion': sorted(missing_external_emotion_targets),
            'personality': sorted(missing_external_personality_targets),
        },
        'personality_csv_path': personality_csv_path,
        'personality_pseudo_json_path': personality_pseudo_json_path,
        'demo_backend': demo_backend,
        'demo_model_name': demo_model_name,
        'demo_top_k': demo_top_k,
        'demo_alpha': demo_alpha,
        'demo_exclude_same_dialogue': demo_exclude_same_dialogue,
        'demo_trace': demo_trace,
        'personality_oversample_factor': personality_oversample_factor,
        'module_usage': module_usage,
        'active_modules': active_modules,
        'output_paths': generated_paths,
        'trace_path': trace_path if demons == 'True' and demo_trace == 'True' and len(trace_records) > 0 else None,
    }
    for output_path in dict.fromkeys([run_root] + generated_paths):
        _write_run_manifest(output_path, manifest)

    return data_path

# def window_process_data():
#     pass

# def perform_speaker_task():
#     pass

# def perform_domain_base_task():
    # pass

def perform_emotion_prediction_task():
    pass


parser = argparse.ArgumentParser(description='Data processing script')
parser.add_argument('--dataset', type=str, default='iemocap', help='Dataset name or path')
parser.add_argument('--historical_window', type=int, default=20, help='Historical window size')
parser.add_argument('--speaker_task', type=str, default='add speaker_task to main task', help='Speaker task type')
parser.add_argument('--domain_base', type=str, default='select demonstration from domain_base to add to input', help='domain_base mode')
parser.add_argument('--emotion_prediction', type=str, default='add emotion_prediction to main task', help='Emotion prediction task type')
parser.add_argument('--personality_meta', type=str, default='False', help='Whether to attach personality metadata into utterance samples')
parser.add_argument('--personality_task', type=str, default='False', help='Whether/how to add personality prediction task samples. Options: False, True_continuous, True_discrete (True is accepted for compatibility).')
parser.add_argument('--personality_mode', type=str, default='raw', choices=['raw', 'binary', 'ternary'], help='Target format for personality task; used directly when personality_task=True compatibility mode is used')
parser.add_argument('--personality_condition', type=str, default='False', help='Whether/how to condition emotion input on ground-truth speaker OCEAN. Options: False, True_continuous (continuous OCEAN), True_discrete (discrete low/medium/high)')
parser.add_argument('--emotion_context_mode', type=str, default='context', choices=['context', 'single'], help='Emotion prompt mode: context keeps dialogue context and repeats the target utterance; single keeps only the target utterance once')
parser.add_argument('--personality_csv_path', type=str, default='/n/work6/yiwang/MoodFilm/portable/datasets/PA-IEMOCAP/ocean.csv', help='Path to PA-IEMOCAP ocean.csv')
parser.add_argument('--iemocap_root', type=str, default=DEFAULT_IEMOCAP_ROOT, help='IEMOCAP_full_release root used to align official utterance ids')
parser.add_argument('--personality_pseudo_json_path', type=str, default=None, help='Optional pseudo personality json map for personality_condition on datasets without ground-truth OCEAN')
parser.add_argument('--personality_use_pseudo_first', type=str, default='False', choices=['True', 'False'], help='Prefer personality_pseudo_json_path over IEMOCAP ground-truth OCEAN for personality condition/meta')
parser.add_argument('--external_reasoning_dir', type=str, default=DEFAULT_EXTERNAL_REASONING_DIR, help='Directory containing LLM reasoning_samples_llm outputs for IEMOCAP')
parser.add_argument('--demo_backend', type=str, default='auto', choices=['auto', 'sbert', 'lexical'], help='Demonstration retrieval backend')
parser.add_argument('--demo_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='SBERT model name for retrieval')
parser.add_argument('--demo_top_k', type=int, default=16, help='Top-k semantic candidates before lexical rerank')
parser.add_argument('--demo_alpha', type=float, default=0.75, help='Hybrid score weight: alpha*semantic + (1-alpha)*lexical')
parser.add_argument('--demo_personality_weight', type=float, default=0.2, help='Extra weight for personality similarity during demo retrieval')
parser.add_argument('--demo_exclude_same_dialogue', type=str, default='True', help='Whether to exclude same dialogue candidates in retrieval')
parser.add_argument('--demo_trace', type=str, default='False', help='Whether to export per-sample demonstration retrieval trace jsonl')
parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier used to namespace output folders')
parser.add_argument('--run_root', type=str, default=None, help='Optional output root directory (overrides default ablation run root)')
parser.add_argument('--personality_oversample_factor', type=int, default=1, help='Repeat personality training samples this many times to rebalance stage1')
parser.add_argument('--personality_train_setting', type=str, default='conversation', choices=['utterance', 'conversation', 'full'], help='Personality prompt granularity for train split')
parser.add_argument('--personality_inference_setting', type=str, default='conversation', choices=['utterance', 'conversation', 'full'], help='Personality prompt granularity for test/valid splits')
parser.add_argument('--personality_reasoning', type=str, default='False', choices=['True', 'False'], help='Use long reasoning prompt/target for personality prediction')
parser.add_argument('--emotion_reasoning', type=str, default='False', choices=['True', 'False'], help='Use long reasoning prompt/target for emotion prediction')
parser.add_argument('--emotion_reasoning_format', type=str, default='answer_last', choices=['answer_last', 'label_anchor_answer_last', 'label_anchor', 'label_first', 'structured_chain', 'structured'], help='Emotion reasoning target format')
parser.add_argument('--emotion_reasoning_min_words', type=int, default=0, help='Preferred minimum words for cleaned emotion reasoning targets; targets are not synthetically expanded')
parser.add_argument('--emotion_reasoning_max_words', type=int, default=0, help='Maximum words for cleaned emotion reasoning targets; 0 keeps the full reasoning')
parser.add_argument('--emotion_reasoning_clean_hidden_cues', type=str, default='False', choices=['True', 'False'], help='Remove gold-side cue sentences from external emotion reasoning targets')
parser.add_argument('--structured_reasoning', type=str, default='False', choices=['True', 'False'], help='Use concise structured affective reasoning states instead of paragraph reasoning for emotion targets')
parser.add_argument('--predict_personality_in_chain', type=str, default='False', choices=['True', 'False'], help='In structured emotion targets, include a generated <personality> OCEAN chain before states and answer')
parser.add_argument('--personality_input_mode', type=str, default='auto', choices=['auto', 'gt', 'stage1_predicted', 'none'], help='Personality source for Stage2 prompt/meta: auto, gt, stage1_predicted, or none')
parser.add_argument('--generate_reasoning_from_states', type=str, default='False', choices=['True', 'False'], help='Reserved inference/demo flag: generate natural-language reasoning from predicted states without adding it to training targets')
parser.add_argument('--emotion_prompt_output_format', type=str, default='label_only', choices=['label_only', 'short_reasoning_answer'], help='Stage2 emotion prompt output style without changing gold reasoning targets.')
parser.add_argument('--experience_bank_mode', type=str, default='none', choices=['none', 'student_prompt', 'opcd_teacher'], help='Inject experience bank into student prompt, or keep it teacher-only for OPCD.')
parser.add_argument('--experience_bank_path', type=str, default=None, help='JSONL path containing candidate experience items.')
parser.add_argument('--experience_bank_confidence', type=str, default='high', help='Experience item confidence filter, e.g. high or all.')
parser.add_argument('--experience_bank_max_items', type=int, default=0, help='Maximum experience items to load; 0 keeps all matching items.')
parser.add_argument('--opcd_teacher_include_gold_context', type=str, default='False', choices=['True', 'False'], help='Attach GT emotion/personality/derived shift as teacher-only OPCD metadata.')
args = parser.parse_args()




# Process data
processed_data_path = process_dataset(
    dataset=args.dataset,
    window=args.historical_window,
    speaker_task=args.speaker_task,
    demons=args.domain_base,
    predictions=args.emotion_prediction,
    personality_meta=args.personality_meta,
    personality_task=args.personality_task,
    personality_mode=args.personality_mode,
    personality_condition=args.personality_condition,
    emotion_context_mode=args.emotion_context_mode,
    personality_csv_path=args.personality_csv_path,
    iemocap_root=args.iemocap_root,
    personality_pseudo_json_path=args.personality_pseudo_json_path,
    personality_use_pseudo_first=args.personality_use_pseudo_first,
    external_reasoning_dir=args.external_reasoning_dir,
    demo_backend=args.demo_backend,
    demo_model_name=args.demo_model_name,
    demo_top_k=args.demo_top_k,
    demo_alpha=args.demo_alpha,
    demo_personality_weight=args.demo_personality_weight,
    demo_exclude_same_dialogue=args.demo_exclude_same_dialogue,
    demo_trace=args.demo_trace,
    run_id=args.run_id,
    run_root=args.run_root,
    personality_oversample_factor=args.personality_oversample_factor,
    personality_train_setting=args.personality_train_setting,
    personality_inference_setting=args.personality_inference_setting,
    personality_reasoning=args.personality_reasoning,
    emotion_reasoning=args.emotion_reasoning,
    emotion_reasoning_format=args.emotion_reasoning_format,
    emotion_reasoning_min_words=args.emotion_reasoning_min_words,
    emotion_reasoning_max_words=args.emotion_reasoning_max_words,
    emotion_reasoning_clean_hidden_cues=args.emotion_reasoning_clean_hidden_cues,
    structured_reasoning=args.structured_reasoning,
    predict_personality_in_chain=args.predict_personality_in_chain,
    personality_input_mode=args.personality_input_mode,
    generate_reasoning_from_states=args.generate_reasoning_from_states,
    emotion_prompt_output_format=args.emotion_prompt_output_format,
    experience_bank_mode=args.experience_bank_mode,
    experience_bank_path=args.experience_bank_path,
    experience_bank_confidence=args.experience_bank_confidence,
    experience_bank_max_items=args.experience_bank_max_items,
    opcd_teacher_include_gold_context=args.opcd_teacher_include_gold_context,
)

print(processed_data_path)
