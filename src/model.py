"""DT-STPINN: Dynamic Twin Spatio-Temporal Physics-informed Neural Network.

Full model for Paper 1: temperature field prediction in DED thin-walled parts.

Architecture:
    Dynamic Graph → Spatial Encoder (GATv2) → Temporal Encoder (Transformer)
    → Cross Fusion → Temperature Head

Supports physics-informed losses via PDE/Boundary/Initial condition constraints.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder.spatial_encoder import SpatialEncoder
from .encoder.temporal_encoder import TemporalEncoder
from .encoder.cross_fusion import CrossFusion
from .decoder.temperature_head import TemperatureHead


class DTSTPINN(nn.Module):
    def __init__(self, config, material_props):
        super().__init__()

        self.hidden_dim = config.model.hidden_dim
        self.node_feature_dim = config.model.node_feature_dim
        self.edge_feature_dim = config.model.edge_feature_dim

        self.spatial_encoder = SpatialEncoder(
            in_dim=self.node_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=config.model.spatial.num_layers,
            heads=config.model.spatial.heads,
            edge_dim=self.edge_feature_dim,
            dropout=config.model.spatial.dropout,
        )

        self.temporal_encoder = TemporalEncoder(
            d_model=self.spatial_encoder.out_dim,
            num_layers=config.model.temporal.num_layers,
            heads=config.model.temporal.heads,
            ff_dim=config.model.temporal.ff_dim,
            dropout=config.model.temporal.dropout,
            max_len=config.model.temporal.max_seq_len,
        )

        self.cross_fusion = CrossFusion(
            d_model=self.spatial_encoder.out_dim,
            heads=config.model.fusion.heads,
        )

        self.temperature_head = TemperatureHead(
            hidden_dim=self.spatial_encoder.out_dim,
        )

        self._out_dim = self.spatial_encoder.out_dim

    def forward(self, graph_sequence: list, dt: float = 1.0) -> dict:
        """Forward pass.

        Args:
            graph_sequence: list of L PyG Data objects.
            dt: time step size.

        Returns:
            dict with keys: T_pred, spatial_features, temporal_features
        """
        L = len(graph_sequence)

        spatial_features = []
        masks = []

        for data in graph_sequence:
            z_s = self.spatial_encoder(data)
            spatial_features.append(z_s)
            mask = getattr(data, "mask", torch.ones(z_s.shape[0], dtype=torch.bool,
                                                     device=z_s.device))
            masks.append(mask)

        z_s = torch.stack(spatial_features, dim=0).unsqueeze(0)
        mask_stack = torch.stack(masks, dim=0).unsqueeze(0)

        z_t = self.temporal_encoder(z_s, mask_stack)

        z_s_last = z_s[:, -1]
        mask_last = mask_stack[:, -1]

        z_f = self.cross_fusion(z_s_last, z_t, mask_stack)

        T_pred = self.temperature_head(z_f, mask_last)

        return {
            "T_pred": T_pred,
            "spatial_features": z_s,
            "temporal_features": z_t,
            "fused_features": z_f,
        }

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
