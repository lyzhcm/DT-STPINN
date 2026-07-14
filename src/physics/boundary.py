"""Boundary condition loss functions.

Enforces thermal boundary conditions including:
- Dirichlet (fixed temperature at substrate base)
- Convection (Newton's law of cooling on side surfaces)
- Radiation (Stefan-Boltzmann cooling on top surface)
"""
from __future__ import annotations

import torch

from .differentiation import spatial_gradient


class BoundaryConditionLoss:
    def __init__(self, k: float, h_conv: float = 10.0,
                 emissivity: float = 0.35, T_ambient: float = 293.15):
        self.k = k
        self.h_conv = h_conv
        self.emissivity = emissivity
        self.T_ambient = T_ambient
        self.sigma_sb = 5.67e-8

    def compute(self, T: torch.Tensor, coords: torch.Tensor,
                edge_index: torch.Tensor, boundary: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute boundary condition residual.

        Uses `boundary` labels to identify surface nodes:
        - negative values: Dirichlet-type boundary (e.g., substrate)
        - positive/zero: Robin-type boundary (convection + radiation)

        Args:
            T: [N] temperature.
            coords: [N, 3] spatial coordinates.
            edge_index: [2, M] graph connectivity.
            boundary: [N] boundary label field.
            mask: [N] boolean mask for active nodes.

        Returns:
            scalar BC residual loss.
        """
        grad_T = spatial_gradient(T.squeeze(-1), coords, edge_index)

        bdry_nodes = boundary.abs() > 1e-6
        if not bdry_nodes.any():
            return torch.tensor(0.0, device=T.device, dtype=T.dtype)

        is_dirichlet = (boundary < -1.0) & bdry_nodes
        is_robin = (boundary >= -1.0) & bdry_nodes

        losses = []

        if is_dirichlet.any():
            losses.append((T[is_dirichlet].squeeze(-1) - self.T_ambient).pow(2).mean())

        if is_robin.any():
            T_surf = T[is_robin].squeeze(-1)
            grad_norm = grad_T[is_robin].norm(dim=1)

            conv_residual = self.k * grad_norm - self.h_conv * (T_surf - self.T_ambient)
            rad_residual = (self.k * grad_norm
                            - self.emissivity * self.sigma_sb
                            * (T_surf ** 4 - self.T_ambient ** 4))

            losses.append(conv_residual.pow(2).mean())
            losses.append(rad_residual.pow(2).mean())

        if not losses:
            return torch.tensor(0.0, device=T.device, dtype=T.dtype)

        return torch.stack(losses).mean()


class InitialConditionLoss:
    """Initial condition enforcement.

    T(t=0) = T_initial (e.g., substrate at ambient temperature)
    """

    def __init__(self, T_initial: float = 293.15):
        self.T_initial = T_initial

    def compute(self, T: torch.Tensor, is_initial: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute initial condition residual.

        Args:
            T: [N] predicted initial temperature.
            is_initial: [N] boolean, True for nodes that should be at T_initial.
            mask: [N] boolean mask for active nodes.

        Returns:
            scalar MSE of initial condition deviation.
        """
        if not is_initial.any():
            return torch.tensor(0.0, device=T.device, dtype=T.dtype)

        T_init = T[is_initial].squeeze(-1)
        loss = (T_init - self.T_initial).pow(2).mean()
        return loss
