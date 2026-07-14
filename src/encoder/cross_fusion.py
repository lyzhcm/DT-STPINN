"""Cross Fusion Module: Cross-attention between spatial and temporal features.

Uses spatial features (last time step) as queries to attend over
the full temporal history, allowing the model to learn which
historical moments matter most for each spatial location.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossFusion(nn.Module):
    def __init__(self, d_model: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Cross-attend spatial queries over temporal keys/values.

        Args:
            z_s: [B, N, D] spatial features (from last time step).
            z_t: [B, L, N, D] temporal features (full history).
            mask: [B, L, N] boolean mask for inactive nodes in time.

        Returns:
            [B, N, D] fused features.
        """
        B, L, N, D = z_t.shape

        Q = self.query_proj(z_s)
        K = self.key_proj(z_t.permute(0, 2, 1, 3).reshape(B * N, L, D))
        V = self.value_proj(z_t.permute(0, 2, 1, 3).reshape(B * N, L, D))

        Q_attn = Q.reshape(B * N, 1, D)

        key_mask = None
        if mask is not None:
            key_mask = (~mask).permute(0, 2, 1).reshape(B * N, L)

        attn_out, _ = self.attention(Q_attn, K, V, key_padding_mask=key_mask)
        attn_out = attn_out.reshape(B, N, D)

        z_f = self.norm1(z_s + attn_out)
        z_f = self.norm2(z_f + self.ffn(z_f))

        return z_f
