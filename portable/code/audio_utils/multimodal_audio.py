import torch
import torch.nn as nn
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AudioContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self._warned_cudnn_fallback = False
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Old CUDA/PyTorch stacks can crash on very new GPUs when cuDNN RNN flattens weights.
        self.disable_cudnn_rnn = os.getenv("INSTRUCTERC_DISABLE_CUDNN_RNN", "0") == "1"
        if torch.cuda.is_available() and not self.disable_cudnn_rnn:
            capability_major, _ = torch.cuda.get_device_capability()
            if capability_major >= 9:
                self.disable_cudnn_rnn = True

        if self.disable_cudnn_rnn:
            # Avoid cudnn flatten path during module .to(device) and forward.
            self.gru.flatten_parameters = lambda: None

    def forward(self, audio_features, audio_mask):
        # audio_features: [B, T, D], audio_mask: [B, T]
        # Keep GRU weights contiguous for cudnn to avoid per-call compaction overhead.
        if not self.disable_cudnn_rnn:
            self.gru.flatten_parameters()
        lengths = audio_mask.sum(dim=1).clamp(min=1).to(torch.long).cpu()
        packed = pack_padded_sequence(audio_features, lengths, batch_first=True, enforce_sorted=False)
        if self.disable_cudnn_rnn:
            if not self._warned_cudnn_fallback:
                print("[WARN] Audio GRU is running with cuDNN disabled for compatibility.")
                self._warned_cudnn_fallback = True
            with torch.backends.cudnn.flags(enabled=False):
                packed_out, _ = self.gru(packed)
        else:
            packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=audio_features.size(1))
        return out


class AudioProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)


class MultimodalAudioPrefixLM(nn.Module):
    """Wrap a causal LM and prepend one audio prefix token in embedding space."""

    def __init__(
        self,
        base_model,
        audio_feature_dim,
        llm_hidden_size,
        audio_gru_hidden_size=256,
        audio_gru_layers=1,
        audio_gru_dropout=0.0,
        enabled=True,
        input_mode="text_audio",
    ):
        super().__init__()
        self.base_model = base_model
        self.enabled = enabled
        valid_modes = {"text_audio", "audio_only", "text_only"}
        if input_mode not in valid_modes:
            raise ValueError(f"Invalid input_mode: {input_mode}. Use one of {sorted(valid_modes)}")
        self.input_mode = input_mode
        self.audio_context = AudioContextEncoder(
            input_dim=audio_feature_dim,
            hidden_size=audio_gru_hidden_size,
            num_layers=audio_gru_layers,
            dropout=audio_gru_dropout,
        )
        self.audio_projector = AudioProjector(audio_gru_hidden_size * 2, llm_hidden_size)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @property
    def config(self):
        return self.base_model.config

    def _build_multimodal_inputs(self, input_ids, attention_mask, labels, audio_features, audio_mask):
        text_embeds = self.base_model.get_input_embeddings()(input_ids)
        # For audio_only, lexical dialogue content is masked upstream in data collation.
        # Keep instruction/label-candidate scaffold embeddings to preserve task format.

        # Keep audio branch dtype aligned with GRU weights (fp16/fp32) to avoid runtime mismatch.
        audio_dtype = next(self.audio_context.parameters()).dtype
        audio_features = audio_features.to(device=text_embeds.device, dtype=audio_dtype)
        audio_mask = audio_mask.to(text_embeds.device)
        audio_context_seq = self.audio_context(audio_features, audio_mask)

        # Gather the last valid step for each sample as current-utterance context-aware audio embedding.
        lengths = audio_mask.sum(dim=1).clamp(min=1).to(torch.long)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, audio_context_seq.size(-1))
        last_audio = torch.gather(audio_context_seq, 1, gather_idx).squeeze(1)

        audio_token = self.audio_projector(last_audio).to(text_embeds.dtype).unsqueeze(1)
        inputs_embeds = torch.cat([audio_token, text_embeds], dim=1)

        prefix_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        new_labels = None
        if labels is not None:
            ignore_prefix = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device)
            new_labels = torch.cat([ignore_prefix, labels], dim=1)

        return inputs_embeds, new_attention_mask, new_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        audio_features=None,
        audio_mask=None,
        **kwargs,
    ):
        if (not self.enabled) or self.input_mode == "text_only" or (audio_features is None) or (audio_mask is None):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        inputs_embeds, new_attention_mask, new_labels = self._build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_features=audio_features,
            audio_mask=audio_mask,
        )

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        audio_features=None,
        audio_mask=None,
        **kwargs,
    ):
        if (not self.enabled) or self.input_mode == "text_only" or (audio_features is None) or (audio_mask is None):
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        inputs_embeds, new_attention_mask, _ = self._build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            audio_features=audio_features,
            audio_mask=audio_mask,
        )

        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            **kwargs,
        )
