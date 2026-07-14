"""Node feature computation for DT-STPINN.

Builds per-node input features combining geometry, thermal state,
process conditions, and material properties.
"""
from __future__ import annotations

import torch


class NodeFeatureBuilder:
    def __init__(self, material_props):
        self.rho = material_props.density
        self.Cp = material_props.specific_heat
        self.k = material_props.thermal_conductivity
        self.alpha = material_props.thermal_expansion

    def build(self, coords: torch.Tensor, temperature: torch.Tensor,
              live: torch.Tensor, layer_ids: torch.Tensor,
              laser_pos: torch.Tensor, scan_dir: float,
              time: float, dt: float = 1.0) -> torch.Tensor:
        """Build node feature matrix.

        Args:
            coords: [N, 3] spatial coordinates (unnormalized).
            temperature: [N] temperature values.
            live: [N] active material mask (1=deposited).
            layer_ids: [N] per-node layer index.
            laser_pos: [3] current laser position.
            scan_dir: float, scan direction angle in radians.
            time: current time.
            dt: time step size.

        Returns:
            [N, 12] feature tensor.
        """
        N = coords.shape[0]
        device = coords.device
        dtype = coords.dtype

        T = temperature.view(N, 1).to(dtype)

        delta = coords - laser_pos.view(1, 3).to(device=device, dtype=dtype)
        d_laser = delta.norm(dim=1, keepdim=True)

        layer = layer_ids.view(N, 1).to(dtype=dtype, device=device)

        rho = torch.full((N, 1), self.rho, device=device, dtype=dtype)
        cp = torch.full((N, 1), self.Cp, device=device, dtype=dtype)
        kv = torch.full((N, 1), self.k, device=device, dtype=dtype)
        alpha = torch.full((N, 1), self.alpha, device=device, dtype=dtype)

        sin_sd = torch.full((N, 1), float(torch.sin(torch.tensor(scan_dir))),
                            device=device, dtype=dtype)
        cos_sd = torch.full((N, 1), float(torch.cos(torch.tensor(scan_dir))),
                            device=device, dtype=dtype)

        features = torch.cat([
            coords.to(dtype),
            T,
            d_laser,
            layer,
            rho, cp, kv, alpha,
            sin_sd, cos_sd,
        ], dim=1)

        return features

    @property
    def feature_dim(self) -> int:
        return 12

    @property
    def feature_names(self) -> list[str]:
        return [
            "x", "y", "z",
            "T",
            "d_laser",
            "layer_id",
            "rho", "Cp", "k", "alpha",
            "sin_theta", "cos_theta",
        ]
