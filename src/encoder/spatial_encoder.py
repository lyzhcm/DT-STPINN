"""Spatial Encoder: GATv2-based graph neural network.

Encodes spatial physical interactions (heat conduction) across the
computational mesh using attention mechanisms with edge features.
Supports gradient checkpointing for memory efficiency on large graphs.
Multi-head GAT outputs are averaged (`concat=False`) so downstream modules keep
the configured hidden dimension instead of hidden_dim * heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv


class SpatialEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 heads: int = 4, edge_dim: int = 5, dropout: float = 0.1,
                 use_checkpoint: bool = True, edge_embed_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_checkpoint = use_checkpoint

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, edge_embed_dim),
            nn.GELU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_c = hidden_dim
            out_c = hidden_dim
            self.convs.append(
                GATv2Conv(in_c, out_c, heads=heads, edge_dim=edge_embed_dim,
                          dropout=dropout, concat=False)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        mask = getattr(data, "mask", None)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.activation(x)

        edge_emb = self.edge_proj(edge_attr)

        for i in range(self.num_layers):
            residual = x
            if self.use_checkpoint and self.training:
                x = checkpoint(
                    self._conv_forward,
                    self.convs[i], x, edge_index, edge_emb,
                    use_reentrant=False,
                )
            else:
                x = self.convs[i](x, edge_index, edge_attr=edge_emb)
            x = self.activation(x)
            x = self.dropout(x)
            if residual.shape[-1] == x.shape[-1]:
                x = x + residual
            x = self.norms[i](x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        return x

    @staticmethod
    def _conv_forward(conv, x, edge_index, edge_attr):
        return conv(x, edge_index, edge_attr=edge_attr)
