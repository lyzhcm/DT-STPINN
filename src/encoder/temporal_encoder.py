"""Temporal Encoder: Transformer-based thermal history learner.

Processes sequences of spatially-encoded node features along the time
axis, capturing long-range dependencies in the thermal history of each
spatial location independently. Uses chunked processing for large graphs.
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
    def __init__(self, d_model: int, num_layers: int = 4, heads: int = 4,
                 ff_dim: int = 1024, dropout: float = 0.1, max_len: int = 128,
                 chunk_size: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
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
        """Process temporal sequence for each node independently (chunked).

        Args:
            x: [B, L, N, D] spatial features across time steps.
            mask: [B, L, N] boolean mask indicating active nodes per time step.

        Returns:
            [B, L, N, D] temporally-encoded features.
        """
        B, L, N, D = x.shape
        CS = self.chunk_size
        outputs = []

        for start in range(0, N, CS):
            end = min(start + CS, N)
            chunk_len = end - start

            x_chunk = x[:, :, start:end, :]
            x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(B * chunk_len, L, D)

            x_chunk = self.pos_encoding(x_chunk)

            key_padding_mask = None
            if mask is not None:
                mask_chunk = mask[:, :, start:end]
                key_padding_mask = (~mask_chunk).permute(0, 2, 1).reshape(B * chunk_len, L)

            x_chunk = self.transformer(x_chunk, src_key_padding_mask=key_padding_mask)

            x_chunk = x_chunk.reshape(B, chunk_len, L, D).permute(0, 2, 1, 3)
            outputs.append(x_chunk)

        return torch.cat(outputs, dim=2)
