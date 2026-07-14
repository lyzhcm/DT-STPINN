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

        pred = self._as_node_field(pred, "pred")
        target = self._as_node_field(target, "target")
        prev_temp = self._as_node_field(prev_temp, "prev_temp") if prev_temp is not None else None
        mask_bool = self._as_node_mask(mask)

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
                pred, edge_index, mask_bool
            )

        total = sum(losses.values())
        return total, losses

    @staticmethod
    def _as_node_field(x: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize scalar node fields to [N, C].

        The model may return [1, N, C] for a single graph batch, while graph
        labels are usually stored as [N, C]. Keeping one convention inside the
        loss prevents mask/indexing mismatches.
        """
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.dim() != 2:
            raise ValueError(
                f"{name} must have shape [N], [N, C], or [1, N, C]; "
                f"got {tuple(x.shape)}"
            )
        return x

    @staticmethod
    def _as_node_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.bool()
        if mask.dim() == 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.dim() != 1:
            raise ValueError(
                f"mask must have shape [N] or [1, N]; got {tuple(mask.shape)}"
            )
        return mask
