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

        # Hotspot re-weighting: penalises missed melt-pool predictions.
        self.lambda_hot = getattr(config.loss, "lambda_hot", 0.0)
        self.hot_threshold = getattr(config.loss, "hot_threshold", 500.0)
        self.hot_weight = getattr(config.loss, "hot_weight", 5.0)
        self.hot_weight_power = getattr(config.loss, "hot_weight_power", 2.0)
        # Reference temperature for weight ramp: liquidus gives the steepest
        # gradient right where melting physics matters most.
        self._hot_ref_temp = max(
            getattr(material_props, "liquidus_temp", 1654.85),
            self.hot_threshold + 1.0,
        )

        self.physics_config = config.physics
        self.material = material_props

        self.coordinate_scale_to_m = getattr(
            config.physics, "coordinate_scale_to_m", 1.0
        )
        self.time_scale_to_s = getattr(config.physics, "time_scale_to_s", 1.0)
        normalize_pde = getattr(
            config.physics, "normalize_pde_residual", False
        )
        temperature_scale = getattr(
            config.physics, "pde_temperature_scale", 1000.0
        )
        time_scale = getattr(config.physics, "pde_time_scale", 1.0)
        if self.coordinate_scale_to_m <= 0 or self.time_scale_to_s <= 0:
            raise ValueError("Physics coordinate/time scale factors must be positive.")
        if temperature_scale <= 0 or time_scale <= 0:
            raise ValueError("PDE characteristic scales must be positive.")

        self.pde_residual_scale = 1.0
        if normalize_pde:
            self.pde_residual_scale = (
                material_props.density
                * material_props.specific_heat
                * temperature_scale
                / time_scale
            )

        self.heat_conduction = HeatConductionLoss(
            rho=material_props.density,
            Cp=material_props.specific_heat,
            k=material_props.thermal_conductivity,
            residual_scale=self.pde_residual_scale,
        )
        self.boundary_loss = BoundaryConditionLoss(
            k=material_props.thermal_conductivity,
            h_conv=material_props.convection_coeff,
            emissivity=material_props.emissivity,
            T_ambient=material_props.ambient_temp,
            label_mode=getattr(
                config.physics, "boundary_label_mode", "legacy_signed"
            ),
            enable_convection=config.physics.boundary_convection,
            enable_radiation=config.physics.boundary_radiation,
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
        coords_physics = coords * self.coordinate_scale_to_m
        dt_physics = dt * self.time_scale_to_s

        sq_error = (T_pred - T_target) ** 2
        if mask_bool is not None:
            sq_error_active = sq_error[mask_bool]
        else:
            sq_error_active = sq_error
        losses["T"] = self.lambda_T * sq_error_active.mean()

        # Hotspot-weighted loss — continuous ramp so every degree above
        # ``hot_threshold`` receives extra penalty.  The regular T loss
        # keeps the bulk pinned while this term drives melt-pool recall.
        if self.lambda_hot > 0:
            hot_ratio = (
                (T_target - self.hot_threshold)
                / (self._hot_ref_temp - self.hot_threshold)
            ).clamp(0.0, 1.0)
            weights = 1.0 + self.hot_weight * hot_ratio.pow(self.hot_weight_power)
            loss_hot_per_node = weights * sq_error
            if mask_bool is not None:
                loss_hot_per_node = loss_hot_per_node[mask_bool]
            losses["hotspot"] = self.lambda_hot * loss_hot_per_node.mean()

        if self.physics_config.heat_conduction and prev_temp is not None:
            if laser_pos is not None:
                laser_pos_physics = laser_pos * self.coordinate_scale_to_m
                Q_laser = gaussian_heat_source(
                    coords_physics, laser_pos_physics, power=power
                )
            else:
                Q_laser = torch.zeros(coords.shape[0], device=pred.device, dtype=pred.dtype)
            losses["PDE"] = self.lambda_PDE * self.heat_conduction.compute(
                pred.squeeze(-1), prev_temp.squeeze(-1),
                coords_physics, edge_index, dt_physics, Q_laser, mask_bool
            )

        if self.physics_config.boundary_convection:
            losses["BC"] = self.lambda_BC * self.boundary_loss.compute(
                pred, coords_physics, edge_index, boundary, mask_bool
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
