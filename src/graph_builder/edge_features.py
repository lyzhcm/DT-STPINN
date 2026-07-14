"""Edge feature computation for DT-STPINN.

Builds per-edge features encoding geometric relationships and physical
connectivity between nodes.
"""
from __future__ import annotations

import torch


class EdgeFeatureBuilder:
    def __init__(self, material_props):
        self.k_ref = material_props.thermal_conductivity

    def build(self, edge_index: torch.Tensor, coords: torch.Tensor,
              node_k: torch.Tensor | None = None) -> torch.Tensor:
        """Build edge feature matrix.

        Args:
            edge_index: [2, M] source→target indices.
            coords: [N, 3] node coordinates.
            node_k: [N] per-node thermal conductivity (optional).

        Returns:
            [M, 5] edge feature tensor.
        """
        device = edge_index.device
        dtype = coords.dtype

        row, col = edge_index[0], edge_index[1]
        src = coords[row]
        dst = coords[col]

        delta = dst - src
        dist = delta.norm(dim=1, keepdim=True)
        dx = delta[:, 0:1]
        dy = delta[:, 1:2]
        dz = delta[:, 2:3]

        if node_k is not None:
            k_row = node_k[row].view(-1, 1).to(dtype=dtype, device=device)
            k_col = node_k[col].view(-1, 1).to(dtype=dtype, device=device)
            k_avg = (k_row + k_col) / 2.0
        else:
            k_avg = torch.full((dist.shape[0], 1), self.k_ref,
                               device=device, dtype=dtype)

        features = torch.cat([dist, dx, dy, dz, k_avg], dim=1)

        return features

    @property
    def feature_dim(self) -> int:
        return 5

    @property
    def feature_names(self) -> list[str]:
        return ["distance", "dx", "dy", "dz", "k_avg"]
