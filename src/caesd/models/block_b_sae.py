"""Block B -- sparse autoencoder on SSL features.

Recovers separable directions in the feature space. You name the directions that carry the
fake evidence, so the explanation comes from inside the model rather than from a fragile
post-hoc attribution map.

features [B, D]  ->  {code [B, K] (sparse), recon [B, D]}
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, feature_dim: int, n_latents: int = 256, l1_coeff: float = 1e-3) -> None:
        super().__init__()
        self.encoder = nn.Linear(feature_dim, n_latents)
        self.decoder = nn.Linear(n_latents, feature_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(feature_dim))
        self.l1_coeff = l1_coeff

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        code = torch.relu(self.encoder(features - self.pre_bias))
        recon = self.decoder(code) + self.pre_bias
        return {"code": code, "recon": recon}

    def loss(self, features: torch.Tensor, out: dict[str, torch.Tensor]) -> torch.Tensor:
        recon_err = torch.mean((out["recon"] - features) ** 2)
        sparsity = self.l1_coeff * out["code"].abs().mean()
        return recon_err + sparsity
