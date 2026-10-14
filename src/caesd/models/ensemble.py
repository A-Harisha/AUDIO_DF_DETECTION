"""The ensemble as one differentiable waveform -> score model.

    waveform [B, T], lengths [B]  ->  {score [B], members [B, M]}

Why it exists and not just a score-fusion of parquet files: an honest white-box robustness number
needs the gradient of the exact system under test, so the whole path from the waveform through
every backbone and head to the combined score must be one graph. The score-level study
(``scripts/ensemble.py``) and this model must agree on clean data, which ``scripts/attack.py``
checks before it attacks anything.

Backbones are SHARED between members with the same tag, so an ensemble of four WavLM heads costs
one backbone pass, not four. That is what keeps a 30-step PGD over two backbones affordable.

Members differ in score scale, so each is standardised by statistics fit on the dev ``calib`` split
(written by ``scripts/ensemble.py``) before the fixed combiner weights are applied. Nothing is
learned here: this module only replays a combiner that was already fit and selected on dev.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .backbone import Backbone
from .detector import Head


def load_member(run_dir: Path, backbone: Backbone, device: str) -> tuple[Head, dict]:
    """The trained head of a run, on ``backbone``. The run's config decides the head's shape, so a
    checkpoint can never be loaded into a head that differs from the one that produced it."""
    from ..train.loop import load_trainable_state  # imported here: models <- train <- models

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    exp = cfg["experiment"]
    if exp["film"]:
        raise NotImplementedError(f"{run_dir.name} uses block A/FiLM; the ensemble path does not")
    band = None if exp.get("band") is None else tuple(exp["band"])
    head = Head(backbone.n_layers, backbone.feature_dim, dropout=exp["dropout"], band=band)
    # The checkpoint was saved from the whole CaesdDetector, so its keys carry the "head." prefix.
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    load_trainable_state(head, {k.removeprefix("head."): v for k, v in state.items()
                                if k.startswith("head.")})
    # A trained-from-scratch front end is in the SAME checkpoint under "backbone.", because it is
    # trainable and ``trainable_state`` saves every parameter that requires grad. A frozen SSL
    # backbone contributes no such keys, so this is how the two cases are told apart.
    body = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    if body:
        load_trainable_state(backbone, body)
    return head.to(device).eval(), cfg


class ScoreEnsemble(nn.Module):
    """``members`` is one (backbone tag, head) pair per member, in the order the statistics and the
    weights are given in. ``weights`` must sum to 1: the combined score stays on the scale of a
    single standardised member, so one threshold serves every ensemble size."""

    def __init__(self, backbones: dict[str, Backbone], tags: list[str], heads: list[Head],
                 mean: torch.Tensor, std: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        if not (len(tags) == len(heads) == mean.numel() == std.numel() == weights.numel()):
            raise ValueError("tags, heads, mean, std and weights must describe the same members")
        if abs(float(weights.sum()) - 1.0) > 1e-6:
            raise ValueError(f"combiner weights sum to {float(weights.sum())}, not 1")
        self.backbones = nn.ModuleDict(backbones)
        self.heads = nn.ModuleList(heads)
        self.tags = tags
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.register_buffer("weights", weights.float())

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = {tag: self.backbones[tag](wave, lengths) for tag in dict.fromkeys(self.tags)}
        members = torch.stack([head(pooled[tag])["score"] for tag, head in zip(self.tags, self.heads, strict=True)],
                              dim=-1)
        standardised = (members - self.mean) / self.std
        return {"score": (standardised * self.weights).sum(-1), "members": standardised}

    def score_fn(self):
        """The ``ScoreFn`` the attacks in ``caesd.attacks`` take."""
        return lambda wave, lengths: self(wave, lengths)["score"]

    @classmethod
    def from_runs(cls, runs_dir: Path, runs: list[str], mean: torch.Tensor, std: torch.Tensor,
                  weights: torch.Tensor, device: str = "cuda") -> ScoreEnsemble:
        """Build from the run directories ``scripts/ensemble.py`` combined, loading each distinct
        backbone once and freezing everything: the attack perturbs the input, never the model."""
        backbones: dict[str, nn.Module] = {}
        tags, heads = [], []
        for run in runs:
            run_dir = runs_dir / run
            cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            tag, name = cfg["backbone"]["tag"], cfg["backbone"]["name"]
            if cfg["experiment"]["lora_rank"] or cfg["experiment"]["unfreeze"]:
                raise NotImplementedError(f"{run}: the ensemble path assumes a frozen SSL backbone")
            if name == "raw":
                # A from-scratch front end carries its OWN weights, so two raw runs are two
                # different front ends and must not share a cache entry the way one pretrained
                # checkpoint can. This member is the point of the WavLM-free transfer test: it
                # shares no weights with an SSL surrogate, so a gradient from one may not reach it.
                from .raw_frontend import RawFrontEnd

                key = f"{tag}:{run}"
                backbones[key] = RawFrontEnd(
                    channels=cfg["backbone"]["channels"], n_stages=cfg["backbone"]["n_stages"],
                    n_filters=cfg["backbone"]["n_filters"], kernel_size=cfg["backbone"]["kernel_size"],
                ).to(device).eval()
            else:
                key = tag
                if key not in backbones:
                    backbones[key] = Backbone.from_pretrained(name, freeze=True).to(device).eval()
            head, _ = load_member(run_dir, backbones[key], device)
            tags.append(key)
            heads.append(head)
        model = cls(backbones, tags, heads, mean, std, weights).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)  # the gradient the attack needs is w.r.t. the waveform
        return model
