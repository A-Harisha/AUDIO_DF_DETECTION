"""The detector: an SSL backbone and a small head. One model, no ensemble.

``Head`` maps the pooled layer stack to one score. It is all that trains when the backbone is
frozen, so it can train on cached features (``CaesdDetector(None, head)``).

Optional ablation (``n_channels > 0``): block A predicts the codec from the pooled features and
FiLM modulates the features with the condition vector. Calibration (block C) is NOT part of the
detector: it is a post-hoc stage fit on held-out scores (``block_c_hypernet``), so it cannot
change discrimination.

pooled layers [B, L, D]  ->  {score [B] (higher => bona fide), channel_logits [B, n_channels] | None}

With ``n_out > 1`` the same head is a plain classifier (``score`` is [B, n_out]); the calibration
study uses it as the stand-alone codec classifier.
"""
from __future__ import annotations

import torch
from torch import nn

from .backbone import Backbone
from .block_a_condition import ConditionEncoder
from .film import FiLM


class Head(nn.Module):
    """``band`` restricts the layer weighting to ``range(*band)``: the head may then use only that
    slice of the backbone. Every unbanded head converges to nearly the same near-uniform weighting
    (top layers 24, 10, 9, 8-11 on all four WavLM runs), which is why their eval scores correlate
    at 0.99 and fusing them gains 0.1 points (docs/experiment_log.md, 2026-10-13). A band forces a
    different feature path, so banded heads are the diverse members of the ensemble."""

    def __init__(self, n_layers: int, feature_dim: int, hidden: int = 256, dropout: float = 0.2,
                 n_channels: int = 0, condition_dim: int = 32, n_out: int = 1,
                 band: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)  # layers differ in scale
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        keep = torch.zeros(n_layers, dtype=torch.bool)
        keep[slice(*band) if band is not None else slice(None)] = True
        if not keep.any():
            raise ValueError(f"band {band} selects no layer of {n_layers}")
        self.register_buffer("band_mask", torch.where(keep, 0.0, -torch.inf))
        self.condition_encoder = ConditionEncoder(feature_dim, condition_dim, n_channels) if n_channels else None
        self.film = FiLM(feature_dim, condition_dim) if n_channels else None
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, layers: torch.Tensor) -> dict[str, torch.Tensor | None]:
        weights = torch.softmax(self.layer_logits + self.band_mask, dim=0)
        x = (weights[None, :, None] * self.norm(layers.float())).sum(dim=1)
        channel_logits = None
        if self.condition_encoder is not None:
            cond = self.condition_encoder(x)
            channel_logits = cond["channel_logits"]
            x = self.film(x, cond["condition"])
        out = self.mlp(x)
        return {"score": out.squeeze(-1) if out.shape[-1] == 1 else out, "channel_logits": channel_logits}


class MHFA(nn.Module):
    """Multi-head factorized attentive pooling over the backbone FRAMES.

    Why it exists. ``Head`` consumes the time-MEAN of every layer, so a 4 s clip's 199 frames
    become one point and nothing in the model sees time. Every member built on it differs only in
    WHICH layers it averages, which is why six members correlate at 0.85-0.99 on eval and fusing
    them gains nothing (docs/experiment_log.md, 2026-10-23). This back-end learns WHERE in the clip
    to look instead, so it is the first member with a different view of the same features.

    Shape: ([B, L, T, D], mask [B, T]) -> score [B].

    Factorized means two SEPARATE learned layer weightings: the layers that say where to look need
    not be the layers that carry the evidence. ``n_heads`` time-attentions are pooled independently
    and concatenated, so different heads can settle on different parts of the clip.
    """

    def __init__(self, n_layers: int, feature_dim: int, compression: int = 128, n_heads: int = 8,
                 hidden: int = 256, dropout: float = 0.2, n_out: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)  # layers differ in scale
        self.key_logits = nn.Parameter(torch.zeros(n_layers))
        self.value_logits = nn.Parameter(torch.zeros(n_layers))
        self.to_key = nn.Linear(feature_dim, compression)
        self.to_value = nn.Linear(feature_dim, compression)
        self.attend = nn.Linear(compression, n_heads)
        self.n_heads = n_heads
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_heads * compression, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def _stack(self, layers: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Layer-weighted frames [B, T, D] from [B, L, T, D]."""
        return (torch.softmax(logits, dim=0)[None, :, None, None] * layers).sum(dim=1)

    def forward(self, layers: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor | None]:
        layers = self.norm(layers.float())
        key = self.to_key(self._stack(layers, self.key_logits))        # [B, T, C]
        value = self.to_value(self._stack(layers, self.value_logits))  # [B, T, C]
        # Masked softmax over TIME. Without the mask the attention would spend weight on the zero
        # padding, and a clip's score would depend on the longest clip in its batch.
        logits = self.attend(key).masked_fill(~mask[:, :, None], -torch.inf)   # [B, T, H]
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bth,btc->bhc", weights, value).flatten(1)       # [B, H * C]
        out = self.mlp(pooled)
        return {"score": out.squeeze(-1) if out.shape[-1] == 1 else out, "channel_logits": None}


class CaesdDetector(nn.Module):
    def __init__(self, backbone: Backbone | None, head: Head) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> dict:
        """``x`` is a waveform batch [B, T], or cached pooled layers [B, L, D] with no backbone.

        An un-pooled backbone returns (frames, mask) and its head takes both, so the tuple is
        unpacked rather than passed whole."""
        out = x if self.backbone is None else self.backbone(x, lengths)
        return self.head(*out) if isinstance(out, tuple) else self.head(out)
