import random
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler


ORDERED_DATASET_INDEX = "_speaker_mood_dataset_index"
ORDERED_PADDING = "_speaker_mood_order_padding"


def clean_sample_id(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def dialogue_turn_key(sample_id):
    sample_id = clean_sample_id(sample_id)
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


def _dataset_sample_metadata(dataset, index):
    item = dataset[index]
    if isinstance(item, (list, tuple)) and item and isinstance(item[-1], dict):
        return item[-1]
    return {}


def _dataset_sample_id(dataset, index):
    metadata = _dataset_sample_metadata(dataset, index)
    sample_id = metadata.get("sample_id") or metadata.get("utterance_id")
    return clean_sample_id(sample_id)


class OrderedDatasetView(Dataset):
    """Adds sampler bookkeeping without mutating the underlying examples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ordered_index):
        is_padding = False
        if isinstance(ordered_index, (tuple, list)):
            if len(ordered_index) != 2:
                raise ValueError(f"Invalid ordered dataset index: {ordered_index!r}")
            dataset_index, is_padding = ordered_index
        else:
            dataset_index = ordered_index

        dataset_index = int(dataset_index)
        item = self.dataset[dataset_index]
        if not isinstance(item, (list, tuple)) or not item or not isinstance(item[-1], dict):
            raise TypeError("OrderedDatasetView requires examples ending in a metadata dict.")

        copied = list(item)
        metadata = dict(copied[-1])
        metadata[ORDERED_DATASET_INDEX] = dataset_index
        metadata[ORDERED_PADDING] = bool(is_padding)
        copied[-1] = metadata
        return copied


class DialogueOrderedBatchSampler(Sampler):
    """
    Assign complete dialogues to global lanes.

    One lane contributes one chronologically ordered turn per step. A rank owns
    local_batch_size lanes, so no batch can contain two turns from one dialogue.
    Masked padding equalizes the number of distributed steps.
    """

    def __init__(
        self,
        dataset,
        local_batch_size,
        num_replicas=1,
        rank=0,
        shuffle_dialogues=True,
        seed=0,
    ):
        if int(local_batch_size) <= 0:
            raise ValueError("local_batch_size must be positive.")
        if int(num_replicas) <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas).")

        self.dataset = dataset
        self.local_batch_size = int(local_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle_dialogues = bool(shuffle_dialogues)
        self.seed = int(seed)
        self.epoch = 0
        self._cached_epoch = None
        self._rank_batches = None
        self._diagnostics = None

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self._cached_epoch = None

    def _dialogues(self):
        grouped = defaultdict(list)
        for index in range(len(self.dataset)):
            sample_id = _dataset_sample_id(self.dataset, index)
            dialogue_id, turn_index = dialogue_turn_key(sample_id)
            if dialogue_id is None or turn_index is None:
                dialogue_id = f"__singleton_{index}"
                turn_index = 0
            grouped[str(dialogue_id)].append((int(turn_index), int(index)))

        dialogues = []
        for dialogue_id, rows in grouped.items():
            rows.sort(key=lambda item: (item[0], item[1]))
            dialogues.append((dialogue_id, [index for _, index in rows]))
        return dialogues

    def _build(self):
        if self._cached_epoch == self.epoch and self._rank_batches is not None:
            return

        dialogues = self._dialogues()
        if not dialogues:
            self._rank_batches = []
            self._diagnostics = {
                "dialogues": 0,
                "global_steps": 0,
                "rank_real_samples": 0,
                "rank_padding_samples": 0,
            }
            self._cached_epoch = self.epoch
            return

        rng = random.Random(self.seed + self.epoch)
        if self.shuffle_dialogues:
            rng.shuffle(dialogues)
        # Stable sorting keeps the seeded order for equal-length dialogues.
        dialogues.sort(key=lambda item: len(item[1]), reverse=True)

        lane_count = self.num_replicas * self.local_batch_size
        lane_dialogues = [[] for _ in range(lane_count)]
        lane_loads = [0 for _ in range(lane_count)]
        dialogue_to_lane = {}
        for dialogue_id, indices in dialogues:
            lane_index = min(range(lane_count), key=lambda idx: (lane_loads[idx], idx))
            lane_dialogues[lane_index].append((dialogue_id, indices))
            lane_loads[lane_index] += len(indices)
            dialogue_to_lane[dialogue_id] = lane_index

        lane_sequences = []
        for lane_index, assigned in enumerate(lane_dialogues):
            if self.shuffle_dialogues and len(assigned) > 1:
                random.Random(self.seed + self.epoch * 1009 + lane_index).shuffle(assigned)
            sequence = []
            for _, indices in assigned:
                sequence.extend(indices)
            lane_sequences.append(sequence)

        global_steps = max(len(sequence) for sequence in lane_sequences)
        global_fallback = dialogues[0][1][0]
        lane_fallbacks = [
            sequence[0] if sequence else global_fallback
            for sequence in lane_sequences
        ]

        first_lane = self.rank * self.local_batch_size
        local_lanes = range(first_lane, first_lane + self.local_batch_size)
        rank_batches = []
        rank_real_samples = 0
        rank_padding_samples = 0
        for step in range(global_steps):
            batch = []
            for lane_index in local_lanes:
                sequence = lane_sequences[lane_index]
                if step < len(sequence):
                    batch.append((sequence[step], False))
                    rank_real_samples += 1
                else:
                    batch.append((lane_fallbacks[lane_index], True))
                    rank_padding_samples += 1
            rank_batches.append(batch)

        self._rank_batches = rank_batches
        self._diagnostics = {
            "dialogues": len(dialogues),
            "global_steps": global_steps,
            "rank_real_samples": rank_real_samples,
            "rank_padding_samples": rank_padding_samples,
            "lane_load_min": min(lane_loads),
            "lane_load_max": max(lane_loads),
            "dialogue_to_rank": {
                dialogue_id: lane_index // self.local_batch_size
                for dialogue_id, lane_index in dialogue_to_lane.items()
            },
        }
        self._cached_epoch = self.epoch

    @property
    def diagnostics(self):
        self._build()
        return dict(self._diagnostics)

    def __iter__(self):
        self._build()
        for batch in self._rank_batches:
            yield list(batch)

    def __len__(self):
        self._build()
        return len(self._rank_batches)


def lookup_previous_moods(reference, mood_dim, previous_sample_ids, state_cache):
    batch_size = int(reference.size(0))
    mood_dim = int(mood_dim)
    previous_sample_ids = list(previous_sample_ids or [])
    state_cache = state_cache if state_cache is not None else {}

    previous = reference.new_zeros((batch_size, mood_dim))
    valid = torch.zeros(batch_size, device=reference.device, dtype=torch.bool)
    expected = torch.zeros(batch_size, device=reference.device, dtype=torch.bool)

    for row_index in range(batch_size):
        previous_id = (
            clean_sample_id(previous_sample_ids[row_index])
            if row_index < len(previous_sample_ids)
            else ""
        )
        if not previous_id:
            continue
        expected[row_index] = True
        cached = state_cache.get(previous_id)
        if not torch.is_tensor(cached) or cached.numel() != mood_dim:
            continue
        previous[row_index] = cached.detach().reshape(-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        valid[row_index] = True
    return previous, valid, expected


def lookup_previous_emotions(
    reference,
    num_emotions,
    previous_sample_ids,
    emotion_cache,
):
    """Look up detached one-hot predictions for the previous same-speaker turn."""
    batch_size = int(reference.size(0))
    num_emotions = int(num_emotions)
    previous_sample_ids = list(previous_sample_ids or [])
    emotion_cache = emotion_cache if emotion_cache is not None else {}

    previous = reference.new_zeros((batch_size, num_emotions))
    valid = torch.zeros(batch_size, device=reference.device, dtype=torch.bool)
    expected = torch.zeros(batch_size, device=reference.device, dtype=torch.bool)

    for row_index in range(batch_size):
        previous_id = (
            clean_sample_id(previous_sample_ids[row_index])
            if row_index < len(previous_sample_ids)
            else ""
        )
        if not previous_id:
            continue
        expected[row_index] = True
        cached = emotion_cache.get(previous_id)
        if not torch.is_tensor(cached) or cached.numel() != num_emotions:
            continue
        previous[row_index] = cached.detach().reshape(-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        valid[row_index] = True
    return previous, valid, expected


def select_recurrent_hidden(initial_mood, previous_mood, previous_valid):
    if initial_mood.shape != previous_mood.shape:
        raise ValueError(
            f"Initial and previous mood shapes differ: "
            f"{tuple(initial_mood.shape)} vs {tuple(previous_mood.shape)}"
        )
    previous_valid = previous_valid.to(device=initial_mood.device).bool().view(-1, 1)
    return torch.where(previous_valid, previous_mood.to(initial_mood.dtype), initial_mood)


def mask_ordered_sample_weights(weights, padding_mask=None):
    if padding_mask is None:
        return weights
    padding_mask = padding_mask.to(device=weights.device).bool().view(-1)
    if padding_mask.numel() != weights.numel():
        raise ValueError(
            f"Padding mask size {padding_mask.numel()} does not match "
            f"sample weights size {weights.numel()}."
        )
    return weights * (~padding_mask).to(weights.dtype)


def ordered_real_fraction(reference, padding_mask=None):
    if padding_mask is None:
        return reference.new_tensor(1.0)
    padding_mask = padding_mask.to(device=reference.device).bool().view(-1)
    if padding_mask.numel() == 0:
        return reference.new_tensor(1.0)
    return (~padding_mask).to(reference.dtype).mean()


def detached_state_entries(
    mood_after,
    sample_ids,
    valid_mask=None,
    padding_mask=None,
):
    if mood_after is None or not torch.is_tensor(mood_after):
        return []
    sample_ids = list(sample_ids or [])
    row_count = int(mood_after.size(0))
    valid = torch.ones(row_count, dtype=torch.bool, device=mood_after.device)
    if valid_mask is not None:
        valid &= valid_mask.to(device=mood_after.device).bool().view(-1)
    if padding_mask is not None:
        valid &= ~padding_mask.to(device=mood_after.device).bool().view(-1)

    detached = mood_after.detach().float().cpu()
    valid_cpu = valid.detach().cpu().tolist()
    entries = []
    for row_index in range(min(row_count, len(sample_ids))):
        sample_id = clean_sample_id(sample_ids[row_index])
        if sample_id and valid_cpu[row_index]:
            entries.append((sample_id, detached[row_index].clone()))
    return entries


def detached_emotion_entries(
    predicted_indices,
    sample_ids,
    num_emotions,
    valid_mask=None,
    padding_mask=None,
):
    """Build CPU one-hot cache entries from current-turn predicted classes."""
    if predicted_indices is None or not torch.is_tensor(predicted_indices):
        return []
    predicted_indices = predicted_indices.detach().long().view(-1)
    sample_ids = list(sample_ids or [])
    row_count = int(predicted_indices.numel())
    valid = predicted_indices.ge(0) & predicted_indices.lt(int(num_emotions))
    if valid_mask is not None:
        valid &= valid_mask.to(device=predicted_indices.device).bool().view(-1)
    if padding_mask is not None:
        valid &= ~padding_mask.to(device=predicted_indices.device).bool().view(-1)

    indices_cpu = predicted_indices.cpu().tolist()
    valid_cpu = valid.cpu().tolist()
    entries = []
    for row_index in range(min(row_count, len(sample_ids))):
        sample_id = clean_sample_id(sample_ids[row_index])
        if not sample_id or not valid_cpu[row_index]:
            continue
        one_hot = torch.zeros(int(num_emotions), dtype=torch.float32)
        one_hot[int(indices_cpu[row_index])] = 1.0
        entries.append((sample_id, one_hot))
    return entries


def dialogue_speaker_key(sample_id, speaker_id):
    dialogue_id, _ = dialogue_turn_key(sample_id)
    speaker_id = str(speaker_id or "").strip()
    if not dialogue_id or not speaker_id:
        return None
    return str(dialogue_id), speaker_id


def validate_dialogue_parallel_batch(
    sample_ids,
    speaker_ids,
    padding_mask=None,
):
    """Require every real batch row to come from a different dialogue."""
    sample_ids = list(sample_ids or [])
    speaker_ids = list(speaker_ids or [])
    row_count = max(len(sample_ids), len(speaker_ids))
    padding = [False] * row_count
    if padding_mask is not None:
        padding = padding_mask.detach().bool().cpu().view(-1).tolist()

    seen_dialogues = set()
    keys = []
    for row_index in range(row_count):
        if row_index < len(padding) and padding[row_index]:
            keys.append(None)
            continue
        sample_id = sample_ids[row_index] if row_index < len(sample_ids) else ""
        speaker_id = speaker_ids[row_index] if row_index < len(speaker_ids) else ""
        key = dialogue_speaker_key(sample_id, speaker_id)
        if key is None:
            raise ValueError(
                f"Missing dialogue/speaker identity for real batch row {row_index}: "
                f"sample_id={sample_id!r}, speaker_id={speaker_id!r}."
            )
        dialogue_id, _ = key
        if dialogue_id in seen_dialogues:
            raise RuntimeError(
                "Temporal dependency violation: a dialogue appears more than once "
                f"in one parallel batch ({dialogue_id!r})."
            )
        seen_dialogues.add(dialogue_id)
        keys.append(key)
    return keys


def lookup_previous_soft_emotions(
    reference,
    num_emotions,
    sample_ids,
    speaker_ids,
    previous_sample_ids,
    state_cache,
    padding_mask=None,
):
    """Read detached soft states keyed by (dialogue_id, speaker_id)."""
    batch_size = int(reference.size(0))
    num_emotions = int(num_emotions)
    previous_sample_ids = list(previous_sample_ids or [])
    state_cache = state_cache if state_cache is not None else {}
    keys = validate_dialogue_parallel_batch(
        sample_ids=sample_ids,
        speaker_ids=speaker_ids,
        padding_mask=padding_mask,
    )

    previous = reference.new_zeros((batch_size, num_emotions))
    valid = torch.zeros(batch_size, device=reference.device, dtype=torch.bool)
    expected = torch.zeros(batch_size, device=reference.device, dtype=torch.bool)
    for row_index in range(batch_size):
        key = keys[row_index] if row_index < len(keys) else None
        if key is None:
            continue
        previous_id = (
            clean_sample_id(previous_sample_ids[row_index])
            if row_index < len(previous_sample_ids)
            else ""
        )
        expected[row_index] = bool(previous_id)
        cached_entry = state_cache.get(key)
        if cached_entry is None:
            continue
        if not expected[row_index]:
            raise RuntimeError(
                f"Unexpected existing previous state for first speaker turn: {key}."
            )
        if isinstance(cached_entry, dict):
            cached_sample_id = clean_sample_id(cached_entry.get("sample_id"))
            cached = cached_entry.get("probabilities")
        else:
            cached_sample_id = ""
            cached = cached_entry
        if cached_sample_id and cached_sample_id != previous_id:
            raise RuntimeError(
                "Previous-state cache is not chronological for "
                f"{key}: expected sample {previous_id!r}, cached "
                f"{cached_sample_id!r}."
            )
        if not torch.is_tensor(cached) or cached.numel() != num_emotions:
            raise RuntimeError(
                f"Invalid soft previous-emotion cache entry for {key}: "
                f"expected {num_emotions} values."
            )
        previous[row_index] = cached.detach().reshape(-1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        valid[row_index] = True

    missing = expected & ~valid
    if bool(missing.any()):
        rows = missing.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        detail = [
            {
                "row": int(row),
                "sample_id": str(sample_ids[row]),
                "speaker_id": str(speaker_ids[row]),
                "previous_sample_id": str(previous_sample_ids[row]),
            }
            for row in rows[:8]
        ]
        raise RuntimeError(
            "Missing required simplified MoodFiLM previous state. Dialogue "
            f"ordering or cache update is invalid: {detail}"
        )
    return previous, valid, expected, keys


def detached_soft_emotion_entries(
    final_label_logits,
    sample_ids,
    speaker_ids,
    num_emotions,
    valid_mask=None,
    padding_mask=None,
):
    """Convert current final MoodFiLM logits into detached CPU probabilities."""
    if final_label_logits is None or not torch.is_tensor(final_label_logits):
        return []
    if final_label_logits.dim() != 2 or final_label_logits.size(-1) != int(num_emotions):
        raise ValueError(
            "final_label_logits must have shape [batch, num_emotions], got "
            f"{tuple(final_label_logits.shape)}."
        )

    row_count = int(final_label_logits.size(0))
    keys = validate_dialogue_parallel_batch(
        sample_ids=sample_ids,
        speaker_ids=speaker_ids,
        padding_mask=padding_mask,
    )
    valid = torch.isfinite(final_label_logits).all(dim=-1)
    if valid_mask is not None:
        valid &= valid_mask.to(device=final_label_logits.device).bool().view(-1)
    if padding_mask is not None:
        valid &= ~padding_mask.to(device=final_label_logits.device).bool().view(-1)

    probabilities = torch.softmax(
        final_label_logits.detach().float(),
        dim=-1,
    ).cpu()
    valid_cpu = valid.detach().cpu().tolist()
    entries = []
    for row_index in range(row_count):
        key = keys[row_index] if row_index < len(keys) else None
        if key is not None and valid_cpu[row_index]:
            sample_id = clean_sample_id(sample_ids[row_index])
            entries.append(
                (
                    key,
                    {
                        "sample_id": sample_id,
                        "probabilities": probabilities[row_index].clone(),
                    },
                )
            )
    return entries
