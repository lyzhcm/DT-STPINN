"""Temperature Decoder Head.

Predicts temperature from fused latent features via a simple MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z_f: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        T = self.mlp(z_f)
        if mask is not None:
            T = T * mask.unsqueeze(-1).to(T.dtype)
        return T
