"""Temporal Encoder: Transformer-based thermal history learner.

Processes sequences of spatially-encoded node features along the time
axis, capturing long-range dependencies in the thermal history of each
spatial location independently.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 4, heads: int = 8,
                 ff_dim: int = 1024, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process temporal sequence for each node independently.

        Args:
            x: [B, L, N, D] spatial features across time steps.
            mask: [B, L, N] boolean mask indicating active nodes per time step.

        Returns:
            [B, L, N, D] temporally-encoded features.
        """
        B, L, N, D = x.shape

        x = x.permute(0, 2, 1, 3).reshape(B * N, L, D)

        x = self.pos_encoding(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (~mask).permute(0, 2, 1).reshape(B * N, L)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        x = x.reshape(B, N, L, D).permute(0, 2, 1, 3)
        return x
