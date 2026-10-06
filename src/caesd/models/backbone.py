"""Self-supervised speech backbone: XLS-R or WavLM, frozen, adapted with LoRA.

Input : waveform  [B, T]  (mono, 16 kHz, float32)
Output: features  [B, L, D]  (L frames, D = feature_dim)
"""
from __future__ import annotations

import torch
from torch import nn

# Multilingual front end helps cross-lingual transfer; WavLM is the config swap.
DEFAULT_XLSR = "facebook/wav2vec2-xls-r-300m"
DEFAULT_WAVLM = "microsoft/wavlm-large"

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class Backbone(nn.Module):
    def __init__(self, model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim

    @classmethod
    def from_pretrained(
        cls, name: str = DEFAULT_XLSR, freeze: bool = True, lora_rank: int = 16
    ) -> Backbone:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(name)
        feature_dim = model.config.hidden_size
        if freeze:
            for param in model.parameters():
                param.requires_grad_(False)
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, target_modules=_LORA_TARGETS)
            model = get_peft_model(model, cfg)
        return cls(model, feature_dim)

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
        )
        return cls(Wav2Vec2Model(cfg), feature_dim)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.model(wave).last_hidden_state
