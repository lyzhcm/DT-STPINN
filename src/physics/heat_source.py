"""Laser heat source models for DED process simulation.

Implements Goldak double ellipsoid and simplified Gaussian heat sources
for computing the volumetric laser heat input term Q_laser in the
heat conduction PDE.
"""
from __future__ import annotations

import torch


def goldak_heat_source(coords: torch.Tensor, laser_pos: torch.Tensor,
                       scan_dir: float = 0.0, power: float = 900.0,
                       efficiency: float = 0.7,
                       a_f: float = 2e-3, a_r: float = 4e-3,
                       b: float = 1e-3, c: float = 1e-3) -> torch.Tensor:
    """Goldak double ellipsoid heat source model.

    Args:
        coords: [N, 3] node coordinates.
        laser_pos: [3] current laser center position.
        scan_dir: scan direction angle in radians.
        power: laser power in Watts.
        efficiency: absorption efficiency.
        a_f: front ellipsoid semi-axis (m).
        a_r: rear ellipsoid semi-axis (m).
        b: width semi-axis (m).
        c: depth semi-axis (m).

    Returns:
        [N] volumetric heat generation at each node.
    """
    laser_pos = laser_pos.to(device=coords.device, dtype=coords.dtype)
    pi = coords.new_tensor(torch.pi)
    sqrt3 = coords.new_tensor(3.0).sqrt()
    sqrt_pi = pi.sqrt()

    cos_t = float(torch.cos(coords.new_tensor(scan_dir)))
    sin_t = float(torch.sin(coords.new_tensor(scan_dir)))

    dx = coords[:, 0] - laser_pos[0]
    dy = coords[:, 1] - laser_pos[1]
    dz = coords[:, 2] - laser_pos[2]

    x_local = dx * cos_t + dy * sin_t
    y_local = -dx * sin_t + dy * cos_t

    Q = power * efficiency
    f_f = 2.0 * a_f / (a_f + a_r)
    f_r = 2.0 * a_r / (a_f + a_r)

    coeff_f = 6 * sqrt3 * f_f * Q / (a_f * b * c * pi * sqrt_pi)
    coeff_r = 6 * sqrt3 * f_r * Q / (a_r * b * c * pi * sqrt_pi)

    exp_arg_f = -3 * (x_local ** 2) / (a_f ** 2) - 3 * (y_local ** 2) / (b ** 2) - 3 * (dz ** 2) / (c ** 2)
    exp_arg_r = -3 * (x_local ** 2) / (a_r ** 2) - 3 * (y_local ** 2) / (b ** 2) - 3 * (dz ** 2) / (c ** 2)

    q = torch.where(x_local >= 0,
                    coeff_f * torch.exp(exp_arg_f),
                    coeff_r * torch.exp(exp_arg_r))

    return q


def gaussian_heat_source(coords: torch.Tensor, laser_pos: torch.Tensor,
                         power: float = 900.0, efficiency: float = 0.7,
                         radius: float = 2e-3, depth: float = 1e-3) -> torch.Tensor:
    """Simplified Gaussian heat source.

    Args:
        coords: [N, 3] node coordinates.
        laser_pos: [3] current laser center position.
        power: laser power in Watts.
        efficiency: absorption efficiency.
        radius: Gaussian radius (m).
        depth: penetration depth (m).

    Returns:
        [N] volumetric heat generation at each node.
    """
    laser_pos = laser_pos.to(device=coords.device, dtype=coords.dtype)
    pi = coords.new_tensor(torch.pi)
    Q = power * efficiency
    dr = coords[:, :2] - laser_pos[:2]
    dz = coords[:, 2] - laser_pos[2]

    r_sq = (dr ** 2).sum(dim=1)
    z_sq = dz ** 2

    coeff = 2 * Q / (pi * radius ** 2 * depth * pi.sqrt())

    q = coeff * torch.exp(-2 * r_sq / radius ** 2) * torch.exp(-z_sq / depth ** 2)
    return q
