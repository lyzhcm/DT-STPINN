"""Cross Fusion Module: Cross-attention between spatial and temporal features.

Uses spatial features (last time step) as queries to attend over
the full temporal history. Supports chunked processing for large graphs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossFusion(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1,
                 chunk_size: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

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
        """Cross-attend spatial queries over temporal keys/values (chunked).

        Args:
            z_s: [B, N, D] spatial features (from last time step).
            z_t: [B, L, N, D] temporal features (full history).
            mask: [B, L, N] boolean mask for inactive nodes in time.

        Returns:
            [B, N, D] fused features.
        """
        B, L, N, D = z_t.shape
        CS = self.chunk_size

        Q = self.query_proj(z_s)

        outputs = []
        for start in range(0, N, CS):
            end = min(start + CS, N)
            chunk_len = end - start

            Q_c = Q[:, start:end, :]
            K_c = self.key_proj(z_t[:, :, start:end, :].permute(0, 2, 1, 3).reshape(B * chunk_len, L, D))
            V_c = self.value_proj(z_t[:, :, start:end, :].permute(0, 2, 1, 3).reshape(B * chunk_len, L, D))

            Q_attn = Q_c.reshape(B * chunk_len, 1, D)

            key_mask = None
            if mask is not None:
                mask_c = mask[:, :, start:end]
                key_mask = (~mask_c).permute(0, 2, 1).reshape(B * chunk_len, L)

            attn_out, _ = self.attention(Q_attn, K_c, V_c, key_padding_mask=key_mask)
            attn_out = attn_out.reshape(B, chunk_len, D)

            z_f_c = self.norm1(Q_c + attn_out)
            z_f_c = self.norm2(z_f_c + self.ffn(z_f_c))
            outputs.append(z_f_c)

        z_f = torch.cat(outputs, dim=1)

        return z_f
