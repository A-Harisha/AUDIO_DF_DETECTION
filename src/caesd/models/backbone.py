"""Self-supervised speech backbone: XLS-R or WavLM. Frozen by default; LoRA is optional.

Input : waveform  [B, T]  (mono, 16 kHz, float32, zero-padded) and lengths [B] (samples)
Output: pooled    [B, L, D]  (time-mean of every hidden layer; L = n_layers, D = feature_dim)

Every layer is returned because the layers differ strongly: on ASVspoof 5 dev a linear probe gets
2.2% EER on the last layer and 1.0% on layer 20 (docs/experiment_log.md, 2026-10-07). The head
learns the layer weights. The time-mean commutes with that weighted sum, so pooled layers can be
cached once for a frozen backbone.
"""
from __future__ import annotations

import torch
from torch import nn

# Multilingual front end helps cross-lingual transfer; WavLM is the config swap.
DEFAULT_XLSR = "facebook/wav2vec2-xls-r-300m"
DEFAULT_WAVLM = "microsoft/wavlm-large"

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class Backbone(nn.Module):
    def __init__(self, model: nn.Module, feature_dim: int, pooled: bool = True) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.n_layers = model.config.num_hidden_layers + 1  # + the CNN output
        # pooled=False returns the FRAMES instead of their time-mean, for a back-end that models
        # time. It is a separate mode rather than the default because the time-mean is what makes
        # the feature cache possible: un-pooled, the 34.8 GB eval cache would be about 870 GB, so a
        # frame back-end always runs with the backbone in the loop (experiment.features=false).
        self.pooled = pooled

    @classmethod
    def from_pretrained(
        cls, name: str = DEFAULT_XLSR, freeze: bool = True, lora_rank: int = 0,
        pooled: bool = True,
    ) -> Backbone:
        from transformers import AutoModel

        # layerdrop would randomly skip layers in train mode and corrupt the per-layer outputs.
        model = AutoModel.from_pretrained(name, layerdrop=0.0)
        feature_dim = model.config.hidden_size
        if freeze:
            for param in model.parameters():
                param.requires_grad_(False)
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, target_modules=_LORA_TARGETS)
            model = get_peft_model(model, cfg)
        return cls(model, feature_dim, pooled=pooled)

    @classmethod
    def tiny(cls, feature_dim: int = 32) -> Backbone:
        """Small randomly-initialised wav2vec2 for offline forward-pass tests (no download)."""
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        cfg = Wav2Vec2Config(
            hidden_size=feature_dim,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=feature_dim * 2,
            conv_dim=(feature_dim, feature_dim),
            conv_stride=(5, 2),
            conv_kernel=(10, 3),
            num_feat_extract_layers=2,
            do_stable_layer_norm=True,
            feat_extract_norm="layer",
            layerdrop=0.0,
        )
        return cls(Wav2Vec2Model(cfg), feature_dim)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor):
        """pooled: [B, L, D]. Un-pooled: ([B, L, T, D], frame mask [B, T])."""
        mask = torch.arange(wave.shape[1], device=wave.device)[None] < lengths[:, None]
        # Per-utterance zero-mean, unit-variance input over the valid samples (the SSL front ends
        # were pretrained with this normalisation).
        n = lengths[:, None].to(wave.dtype)
        mean = (wave * mask).sum(1, keepdim=True) / n
        var = (((wave - mean) * mask) ** 2).sum(1, keepdim=True) / n
        wave = (wave - mean) * mask / torch.sqrt(var + 1e-7)
        hidden = self.model(wave, attention_mask=mask, output_hidden_states=True).hidden_states
        frames = self.model._get_feat_extract_output_lengths(lengths)
        frame_mask = torch.arange(hidden[0].shape[1], device=wave.device)[None] < frames[:, None]
        if not self.pooled:
            # [B, L, T, D] with the frame validity mask [B, T]: the back-end does its own pooling
            # and MUST use the mask, or it would attend to the padding.
            return torch.stack(hidden, dim=1), frame_mask
        frame_mask = frame_mask[:, :, None].to(hidden[0].dtype)
        return torch.stack([(h * frame_mask).sum(1) / frames[:, None] for h in hidden], dim=1)
