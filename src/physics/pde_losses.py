"""Physics-informed PDE loss functions.

Computes residuals of the transient heat conduction equation and
Fourier's law for enforcing physical consistency during training.
"""
from __future__ import annotations

import torch

from .differentiation import spatial_gradient, spatial_laplacian


class HeatConductionLoss:
    """Transient heat conduction PDE residual loss.

    PDE: ρ * Cp * ∂T/∂t - ∇ · (k ∇T) - Q_laser = 0
    """

    def __init__(self, rho: float, Cp: float, k: float):
        self.rho = rho
        self.Cp = Cp
        self.k = k

    def compute(self, T: torch.Tensor, T_prev: torch.Tensor,
                coords: torch.Tensor, edge_index: torch.Tensor,
                dt: float, Q_laser: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute PDE residual.

        Args:
            T: [N] current temperature prediction.
            T_prev: [N] previous time step temperature.
            coords: [N, 3] spatial coordinates.
            edge_index: [2, M] graph connectivity.
            dt: time step size.
            Q_laser: [N] laser heat source term.
            mask: [N] boolean mask for active nodes.

        Returns:
            scalar MSE of PDE residual.
        """
        dT_dt = (T - T_prev).squeeze(-1) / max(dt, 1e-8)

        laplacian_T = spatial_laplacian(T.squeeze(-1), coords, edge_index)

        residual = self.rho * self.Cp * dT_dt - self.k * laplacian_T - Q_laser

        if mask is not None:
            residual = residual[mask]
            if residual.numel() == 0:
                return torch.tensor(0.0, device=T.device, dtype=T.dtype)

        return (residual ** 2).mean()


class FourierFluxLoss:
    """Fourier's law constraint.

    q = -k ∇T

    Enforces consistency between predicted heat flux and temperature gradient.
    """

    def __init__(self, k: float):
        self.k = k

    def compute(self, T: torch.Tensor, q_pred: torch.Tensor,
                coords: torch.Tensor, edge_index: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Fourier law residual.

        Args:
            T: [N] temperature.
            q_pred: [N, 3] predicted heat flux.
            coords: [N, 3] spatial coordinates.
            edge_index: [2, M] graph connectivity.
            mask: [N] boolean mask.

        Returns:
            scalar MSE residual.
        """
        grad_T = spatial_gradient(T.squeeze(-1), coords, edge_index)
        q_expected = -self.k * grad_T

        residual = (q_pred - q_expected).pow(2).sum(dim=1)

        if mask is not None:
            residual = residual[mask]
            if residual.numel() == 0:
                return torch.tensor(0.0, device=T.device, dtype=T.dtype)

        return residual.mean()
