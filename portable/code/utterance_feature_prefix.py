"""Simple, ordered utterance-level feature tokens for Qwen ERC experiments.

The cache tensors are already per-utterance encoder outputs.  This module only
projects them into the language-model hidden size; it does not encode audio or
perform dialogue modelling.
"""

import torch
from torch import nn


class UtteranceFeaturePrefix(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_size, use_text, use_audio):
        super().__init__()
        self.use_text = bool(use_text)
        self.use_audio = bool(use_audio)
        if not (self.use_text or self.use_audio):
            raise ValueError("At least one utterance feature modality is required")
        self.text_projection = nn.Linear(text_dim, hidden_size) if self.use_text else None
        self.audio_projection = nn.Linear(audio_dim, hidden_size) if self.use_audio else None
        for projection in (self.text_projection, self.audio_projection):
            if projection is not None and projection.in_features == projection.out_features:
                nn.init.eye_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(self, text_features, audio_features, text_mask, audio_mask, dtype=None):
        if text_features.shape[:2] != audio_features.shape[:2]:
            raise ValueError("Text/audio feature turn dimensions differ")
        if text_mask.shape != audio_mask.shape or text_mask.shape != text_features.shape[:2]:
            raise ValueError("Utterance feature masks do not match the turn dimensions")
        parameter = next(self.parameters())
        device = parameter.device
        projected, masks = [], []
        if self.use_text:
            projected.append(self.text_projection(text_features.to(device=device, dtype=self.text_projection.weight.dtype)))
            masks.append(text_mask.to(device=device).bool())
        if self.use_audio:
            projected.append(self.audio_projection(audio_features.to(device=device, dtype=self.audio_projection.weight.dtype)))
            masks.append(audio_mask.to(device=device).bool())
        # Stack modality inside turn before flattening: t1,a1,t2,a2,...
        flat_features = torch.stack(projected, dim=2).flatten(1, 2)
        flat_mask = torch.stack(masks, dim=2).flatten(1, 2)
        lengths = flat_mask.sum(dim=1)
        length = int(lengths.max().item()) if lengths.numel() else 0
        if length <= 0:
            # Keep both projections in the autograd graph even when this rank
            # receives only Stage-1 speaker rows. The masked slot is batch
            # padding, not a valid feature token.
            anchor = flat_features.sum(dim=(1, 2), keepdim=True)
            prefix = anchor.expand(-1, 1, flat_features.size(-1)) * 0.0
            mask = torch.zeros(prefix.size(0), 1, device=device, dtype=torch.long)
            return prefix.to(dtype or parameter.dtype), mask
        # Compact masked (missing-modality) slots without a Python/GPU sync per
        # utterance. scatter_add is safe because each valid item has a unique
        # cumulative destination index within its row.
        destination = (flat_mask.long().cumsum(dim=1) - 1).clamp_min(0)
        destination = destination.unsqueeze(-1).expand(-1, -1, flat_features.size(-1))
        compact = flat_features.new_zeros(flat_features.size(0), length, flat_features.size(-1))
        compact.scatter_add_(1, destination, flat_features * flat_mask.unsqueeze(-1).to(flat_features.dtype))
        compact_mask = torch.arange(length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return compact.to(dtype or parameter.dtype), compact_mask.long()


def prepend_masked_prefix(model_inputs, prefix_embeds, prefix_mask, pad_token_id):
    """Prepend per-row variable tokens, masking all batch padding from loss/attention."""
    if prefix_embeds is None:
        return model_inputs
    result = dict(model_inputs)
    input_ids = result["input_ids"]
    batch_size, length = prefix_mask.shape
    if input_ids.size(0) != batch_size or prefix_embeds.shape[:2] != (batch_size, length):
        raise ValueError("Prefix/input batch dimensions differ")
    result["input_ids"] = torch.cat((input_ids.new_full((batch_size, length), int(pad_token_id)), input_ids), dim=1)
    result["inputs_embeds"] = torch.cat((prefix_embeds, result["inputs_embeds"]), dim=1)
    attention = result.get("attention_mask")
    if attention is None:
        attention = input_ids.new_ones(input_ids.shape)
    result["attention_mask"] = torch.cat((prefix_mask.to(attention.dtype), attention), dim=1)
    if result.get("labels") is not None:
        labels = result["labels"]
        result["labels"] = torch.cat((labels.new_full((batch_size, length), -100), labels), dim=1)
    if result.get("type_token_ids") is not None:
        kinds = result["type_token_ids"]
        result["type_token_ids"] = torch.cat((kinds.new_zeros((batch_size, length)), kinds), dim=1)
    if result.get("position_ids") is not None:
        result.pop("position_ids")
    return result
