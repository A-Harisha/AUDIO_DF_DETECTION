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

    def forward(self, layers: torch.Tensor, mask: torch.Tensor, wave: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        # wave and lengths are the frame back-end contract; MHFA reads only the frames.
        del wave, lengths
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


class LFCCHybrid(nn.Module):
    """SSL frames plus hand-crafted cepstra, aligned in time, then attentive pooling.

    Why it exists. ``MHFA`` changed only the POOLING and that was enough to drop the eval
    correlation with ``lora_enc`` from 0.994 to 0.917, worth 0.10 EER points (log, 2026-10-24). It
    still reads nothing but the SSL features. This back-end adds a second REPRESENTATION -- 20
    linear-frequency cepstral coefficients with their first and second deltas, computed from the
    waveform -- so part of its evidence cannot correlate with anything the backbone encodes. It is
    T45's third family: wav2vec2 embeddings combined with 60-dimensional LFCCs that a Conv1d aligns
    in time and feature dimension (arXiv:2409.01695).

    Shape: ([B, L, T, D], mask [B, T], wave [B, S]) -> score [B].

    Alignment is frame-to-frame, not a stretch. With ``center=False`` and a 320-sample hop the
    cepstral frame count equals the backbone's exactly on a 4 s crop (199), and cepstral frame i
    always reads wave samples [320i, 320i + 512) whatever else is in the batch. Interpolating onto
    the backbone grid instead made a clip's score depend on the batch's LONGEST clip, because the
    stretch factor came from the padded length: that was measured at 1e-4 and is now gone. A count
    that differs by more than one frame raises rather than being quietly resampled.

    What remains is a boundary effect of about 1e-5: the last valid frame's 512-sample window
    reaches past the clip into the zero padding. The SSL front end's convolutions do the same thing
    at the same boundary, so this is a property of the pipeline, not of this back-end.

    The invalid frames are zeroed BEFORE the Conv1d, so its kernel cannot drag padding inwards.
    """

    def __init__(self, n_layers: int, feature_dim: int, n_lfcc: int = 20, compression: int = 128,
                 n_heads: int = 8, hidden: int = 256, dropout: float = 0.2, n_out: int = 1,
                 sample_rate: int = 16000) -> None:
        super().__init__()
        from torchaudio.transforms import LFCC

        # hop 320 at 16 kHz is 20 ms, the backbone's own frame rate, so the interpolation below is
        # a small correction rather than a resample.
        # center=False is load-bearing: it makes the cepstral frame count match the backbone's on
        # a 4 s crop and makes each frame read a fixed window of the waveform.
        self.lfcc = LFCC(sample_rate=sample_rate, n_filter=n_lfcc, n_lfcc=n_lfcc,
                         speckwargs={"n_fft": 512, "hop_length": 320, "win_length": 400,
                                     "center": False})
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.cep_norm = nn.LayerNorm(3 * n_lfcc, elementwise_affine=False)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.ssl_proj = nn.Linear(feature_dim, compression)
        self.cep_proj = nn.Conv1d(3 * n_lfcc, compression, kernel_size=5, padding=2)
        self.to_key = nn.Linear(2 * compression, compression)
        self.to_value = nn.Linear(2 * compression, compression)
        self.attend = nn.Linear(compression, n_heads)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_heads * compression, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def cepstra(self, wave: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """60-dim cepstra on the backbone's time grid, [B, 3 * n_lfcc, T], padding zeroed.

        Computed in float32: a spectrogram and its log are not safe in float16.
        """
        from torchaudio.functional import compute_deltas

        frames = mask.shape[1]
        with torch.autocast("cuda", enabled=False):
            # Pad by n_fft - hop so the LAST frame is always computable. Without it, a clip whose
            # length equals the batch maximum has no room for that window and gets a zero frame,
            # while the same clip in a longer batch does not -- a score that moves with the batch.
            base = self.lfcc(torch.nn.functional.pad(wave.float(), (0, 512 - 320)))
            if abs(base.shape[-1] - frames) > 1:
                raise ValueError(
                    f"cepstra gave {base.shape[-1]} frames for the backbone's {frames}: more than "
                    "a rounding difference means the hop no longer matches the backbone's stride, "
                    "and padding to it would misalign the two representations")
            base = (base[:, :, :frames] if base.shape[-1] >= frames
                    else torch.nn.functional.pad(base, (0, frames - base.shape[-1])))
            # Zero the padding BEFORE the deltas. compute_deltas couples each frame to its
            # neighbours, and the frames past a clip's end are cepstra of silence in a long batch
            # but absent in a short one -- which made a clip's score move with the batch's longest
            # member, by 4.3 in cepstral units. Masking first makes both cases identical. The cost
            # is an artificial step at the boundary, which is the same step in every batch.
            base = base * mask[:, None, :]

            def deltas(x: torch.Tensor) -> torch.Tensor:
                """compute_deltas pads its edges by REPLICATION, so the last valid frame of the
                longest clip in a batch sees a copy of itself where a shorter clip sees the masked
                zeros of its neighbours. Two zero frames -- the delta half-window -- make both
                cases the same."""
                pad = torch.nn.functional.pad(x, (0, 2))
                return compute_deltas(pad)[:, :, : x.shape[-1]] * mask[:, None, :]

            delta = deltas(base)
            stack = torch.cat([base, delta, deltas(delta)], dim=1)
            stack = self.cep_norm(stack.transpose(1, 2)).transpose(1, 2)
        return (stack * mask[:, None, :]).to(self.cep_proj.weight.dtype)

    def forward(self, layers: torch.Tensor, mask: torch.Tensor, wave: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        del lengths
        if wave is None:
            raise ValueError("LFCCHybrid reads a second representation from the waveform; "
                             "it cannot run on cached features")
        ssl = (torch.softmax(self.layer_logits, dim=0)[None, :, None, None]
               * self.norm(layers.float())).sum(dim=1)
        joint = torch.cat([self.ssl_proj(ssl),
                           self.cep_proj(self.cepstra(wave, mask)).transpose(1, 2)], dim=-1)
        logits = self.attend(self.to_key(joint)).masked_fill(~mask[:, :, None], -torch.inf)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bth,btc->bhc", weights, self.to_value(joint)).flatten(1)
        out = self.mlp(pooled)
        return {"score": out.squeeze(-1) if out.shape[-1] == 1 else out, "channel_logits": None}


class GraphAttentionLayer(nn.Module):
    """Graph attention over a node set [B, N, D].

    Ported from the AASIST reference implementation, models/AASIST.py, MIT licence,
    Copyright (c) 2021-present NAVER Corp. (https://github.com/clovaai/aasist). The arithmetic is
    unchanged; the temperature is a plain argument instead of a kwargs lookup.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(out_dim, 1)))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(dropout)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_drop(x)
        pair = x.unsqueeze(2) * x.unsqueeze(1)
        att = torch.matmul(torch.tanh(self.att_proj(pair)), self.att_weight) / self.temp
        att = torch.softmax(att, dim=-2).squeeze(-1)
        x = self.proj_with_att(torch.matmul(att, x)) + self.proj_without_att(x)
        return self.act(self.bn(x.reshape(-1, x.shape[-1])).view(x.shape))


class LayerTimeGraph(nn.Module):
    """Graph attention over the backbone frames on TWO node sets: time and layers.

    This is NOT AASIST and no result from it may be reported under that name, the same rule the raw
    front end carries. AASIST runs HETEROGENEOUS graph attention, with cross-attention and a master
    node, over a spectral and a temporal sub-graph of a 2-D CNN map. Here only AASIST's
    ``GraphAttentionLayer`` is reused. SSL frames have no frequency axis, so the spectral sub-graph
    is replaced by a LAYER sub-graph -- one node per backbone layer, pooled over its valid frames,
    which is a real second axis of this representation.

    Shape: ([B, L, T, D], mask [B, T]) -> score [B].

    Temporal nodes are MASKED-ATTENTION pooled rather than segment pooled, for a reason: clips vary
    in length, and fixed segments over a padded axis would put whole nodes inside the padding for a
    short clip. ``n_time`` attention slots over the valid frames have no such boundary.
    """

    def __init__(self, n_layers: int, feature_dim: int, compression: int = 128, n_time: int = 32,
                 hidden: int = 256, dropout: float = 0.2, n_out: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.proj = nn.Linear(feature_dim, compression)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.slots = nn.Linear(compression, n_time)
        self.time_graph = GraphAttentionLayer(compression, compression, dropout=dropout)
        self.layer_graph = GraphAttentionLayer(compression, compression, dropout=dropout)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * compression, hidden),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, layers: torch.Tensor, mask: torch.Tensor, wave: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        del wave, lengths
        x = self.proj(self.norm(layers.float()))
        n_valid = mask.sum(1)[:, None, None].clamp_min(1)
        layer_nodes = (x * mask[:, None, :, None]).sum(dim=2) / n_valid
        frames = (torch.softmax(self.layer_logits, dim=0)[None, :, None, None] * x).sum(dim=1)
        slot_logits = self.slots(frames).masked_fill(~mask[:, :, None], -torch.inf)
        time_nodes = torch.einsum("bts,btc->bsc", torch.softmax(slot_logits, dim=1), frames)
        pooled = torch.cat([self.time_graph(time_nodes).mean(dim=1),
                            self.layer_graph(layer_nodes).mean(dim=1)], dim=-1)
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
        if not isinstance(out, tuple):
            return self.head(out)
        frames, mask = out
        # The waveform goes through too: a back-end may read a second representation from it
        # (LFCCHybrid does). Heads that do not want it drop it.
        return self.head(frames, mask, wave=x, lengths=lengths)
