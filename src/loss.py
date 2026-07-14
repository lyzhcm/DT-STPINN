"""DT-STPINN loss function assembly.

Combines data-driven losses with physics-informed PDE constraints:
L = λ_T * L_MSE + λ_PDE * L_PDE + λ_BC * L_BC + λ_IC * L_IC + λ_smooth * L_Smooth
"""
from __future__ import annotations

import torch

from .physics.pde_losses import HeatConductionLoss
from .physics.boundary import BoundaryConditionLoss, InitialConditionLoss
from .physics.heat_source import gaussian_heat_source
from .physics.differentiation import graph_smoothness_loss


class DTSTPINNLoss:
    def __init__(self, config, material_props):
        self.lambda_T = config.loss.lambda_T
        self.lambda_PDE = config.loss.lambda_PDE
        self.lambda_BC = config.loss.lambda_BC
        self.lambda_IC = config.loss.lambda_IC
        self.lambda_smooth = config.loss.lambda_smooth

        self.physics_config = config.physics
        self.material = material_props

        self.heat_conduction = HeatConductionLoss(
            rho=material_props.density,
            Cp=material_props.specific_heat,
            k=material_props.thermal_conductivity,
        )
        self.boundary_loss = BoundaryConditionLoss(
            k=material_props.thermal_conductivity,
            h_conv=material_props.convection_coeff,
            emissivity=material_props.emissivity,
            T_ambient=material_props.ambient_temp,
        )
        self.initial_condition = InitialConditionLoss(
            T_initial=material_props.ambient_temp,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                prev_temp: torch.Tensor, coords: torch.Tensor,
                edge_index: torch.Tensor, boundary: torch.Tensor,
                dt: float, laser_pos: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                is_initial: torch.Tensor | None = None,
                power: float = 900.0,
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = {}

        if pred.dim() == 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        if target.dim() == 3 and target.shape[0] == 1:
            target = target.squeeze(0)
        if prev_temp.dim() == 3 and prev_temp.shape[0] == 1:
            prev_temp = prev_temp.squeeze(0)

        mask_bool = mask.bool() if mask is not None else None
        if mask_bool is not None and mask_bool.dim() == 2 and mask_bool.shape[0] == 1:
            mask_bool = mask_bool.squeeze(0)

        T_pred = pred.squeeze(-1)
        T_target = target.squeeze(-1)

        loss_T = ((T_pred - T_target) ** 2)
        if mask_bool is not None:
            loss_T = loss_T[mask_bool]
        losses["T"] = self.lambda_T * loss_T.mean()

        if self.physics_config.heat_conduction and prev_temp is not None:
            if laser_pos is not None:
                Q_laser = gaussian_heat_source(coords, laser_pos, power=power)
            else:
                Q_laser = torch.zeros(coords.shape[0], device=pred.device, dtype=pred.dtype)
            losses["PDE"] = self.lambda_PDE * self.heat_conduction.compute(
                pred.squeeze(-1), prev_temp.squeeze(-1),
                coords, edge_index, dt, Q_laser, mask_bool
            )

        if self.physics_config.boundary_convection:
            losses["BC"] = self.lambda_BC * self.boundary_loss.compute(
                pred, coords, edge_index, boundary, mask_bool
            )

        if self.physics_config.initial_condition and is_initial is not None:
            losses["IC"] = self.lambda_IC * self.initial_condition.compute(
                pred, is_initial, mask_bool
            )

        if self.lambda_smooth > 0:
            losses["Smooth"] = self.lambda_smooth * graph_smoothness_loss(
                pred, edge_index
            )

        total = sum(losses.values())
        return total, losses
