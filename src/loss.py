"""DT-STPINN loss function assembly.

Combines data-driven losses with physics-informed PDE constraints:
L = λ_T * L_MSE + λ_PDE * L_PDE + λ_BC * L_BC + λ_IC * L_IC + λ_smooth * L_Smooth
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

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
        self.lambda_hot_cls = getattr(config.loss, "lambda_hot_cls", 0.0)
        self.hot_cls_threshold = getattr(config.loss, "hot_cls_threshold", 0.0)
        if self.hot_cls_threshold <= 0:
            self.hot_cls_threshold = getattr(material_props, "solidus_temp", 1604.85)
        self.hot_cls_pos_weight = getattr(config.loss, "hot_cls_pos_weight", 0.0)
        self.hot_cls_max_pos_weight = getattr(config.loss, "hot_cls_max_pos_weight", 2000.0)
        self.lambda_hot_delta = getattr(config.loss, "lambda_hot_delta", 0.0)
        self.hot_delta_threshold = getattr(config.loss, "hot_delta_threshold", 0.0)
        if self.hot_delta_threshold <= 0:
            self.hot_delta_threshold = getattr(material_props, "solidus_temp", 1604.85)
        self.hot_delta_scale = getattr(config.loss, "hot_delta_scale", 1000.0)
        if self.hot_delta_scale <= 0:
            raise ValueError("hot_delta_scale must be positive.")
        self.lambda_hot_specialist = getattr(config.loss, "lambda_hot_specialist", 0.0)
        self.hot_specialist_threshold = getattr(config.loss, "hot_specialist_threshold", 0.0)
        if self.hot_specialist_threshold <= 0:
            self.hot_specialist_threshold = getattr(material_props, "solidus_temp", 1604.85)
        self.hot_specialist_scale = getattr(config.loss, "hot_specialist_scale", 1000.0)
        if self.hot_specialist_scale <= 0:
            raise ValueError("hot_specialist_scale must be positive.")
        self.hot_specialist_peak_weight = getattr(
            config.loss, "hot_specialist_peak_weight", 0.0
        )
        self.hot_specialist_weight_power = max(
            getattr(config.loss, "hot_specialist_weight_power", 1.0), 1.0e-6
        )
        self.hot_specialist_candidate_min_temp = getattr(
            config.loss, "hot_specialist_candidate_min_temp", 0.0
        )
        self.hot_specialist_process_gate_threshold = getattr(
            config.loss, "hot_specialist_process_gate_threshold", 0.0
        )
        self.hot_specialist_neighbor_gate_threshold = getattr(
            config.loss, "hot_specialist_neighbor_gate_threshold", 0.0
        )
        self.lambda_hot_process_specialist = getattr(
            config.loss, "lambda_hot_process_specialist", 0.0
        )
        self.hot_process_specialist_threshold = getattr(
            config.loss, "hot_process_specialist_threshold", 0.0
        )
        if self.hot_process_specialist_threshold <= 0:
            self.hot_process_specialist_threshold = getattr(
                material_props, "solidus_temp", 1604.85
            )
        self.hot_process_specialist_scale = getattr(
            config.loss, "hot_process_specialist_scale", 1000.0
        )
        if self.hot_process_specialist_scale <= 0:
            raise ValueError("hot_process_specialist_scale must be positive.")
        self.hot_process_specialist_peak_weight = getattr(
            config.loss, "hot_process_specialist_peak_weight", 0.0
        )
        self.hot_process_specialist_weight_power = max(
            getattr(config.loss, "hot_process_specialist_weight_power", 1.0), 1.0e-6
        )
        self.hot_process_specialist_gate_source = str(getattr(
            config.loss, "hot_process_specialist_gate_source", "arrival_or_process"
        )).lower()
        if self.hot_process_specialist_gate_source not in {
            "process", "arrival", "arrival_or_process", "arrival_process_product",
            "residual_prior", "residual_gate",
        }:
            raise ValueError(
                "hot_process_specialist_gate_source must be 'process', 'arrival', "
                "'arrival_or_process', 'arrival_process_product', "
                "'residual_prior', or 'residual_gate'."
            )
        self.hot_process_specialist_gate_threshold = min(
            max(getattr(config.loss, "hot_process_specialist_gate_threshold", 0.0), 0.0),
            0.99,
        )
        self.hot_process_specialist_gate_power = max(
            getattr(config.loss, "hot_process_specialist_gate_power", 1.0), 1.0e-6
        )
        self.hot_process_specialist_prev_max_temp = getattr(
            config.loss, "hot_process_specialist_prev_max_temp", 0.0
        )
        self.lambda_hot_final = getattr(config.loss, "lambda_hot_final", 0.0)
        self.hot_final_threshold = getattr(config.loss, "hot_final_threshold", 0.0)
        if self.hot_final_threshold <= 0:
            self.hot_final_threshold = getattr(material_props, "solidus_temp", 1604.85)
        self.hot_final_scale = getattr(config.loss, "hot_final_scale", 1000.0)
        if self.hot_final_scale <= 0:
            raise ValueError("hot_final_scale must be positive.")
        self.hot_final_peak_weight = getattr(config.loss, "hot_final_peak_weight", 0.0)
        self.hot_final_weight_power = max(
            getattr(config.loss, "hot_final_weight_power", 1.0), 1.0e-6
        )
        self.lambda_cold_false_hot = getattr(config.loss, "lambda_cold_false_hot", 0.0)
        self.cold_false_hot_target_threshold = getattr(
            config.loss, "cold_false_hot_target_threshold", 100.0
        )
        self.cold_false_hot_pred_threshold = getattr(
            config.loss, "cold_false_hot_pred_threshold", 500.0
        )
        self.cold_false_hot_scale = getattr(config.loss, "cold_false_hot_scale", 1000.0)
        if self.cold_false_hot_scale <= 0:
            raise ValueError("cold_false_hot_scale must be positive.")
        self.cold_false_hot_gate_source = str(getattr(
            config.loss, "cold_false_hot_gate_source", "specialist"
        )).lower()
        if self.cold_false_hot_gate_source not in {
            "none", "specialist", "process", "arrival", "neighbor",
            "arrival_process_product", "arrival_or_process", "path_post_endpoint",
        }:
            raise ValueError(
                "cold_false_hot_gate_source must be 'none', 'specialist', "
                "'process', 'arrival', 'neighbor', 'arrival_process_product', "
                "'arrival_or_process', or 'path_post_endpoint'."
            )
        self.cold_false_hot_gate_threshold = min(
            max(getattr(config.loss, "cold_false_hot_gate_threshold", 0.1), 0.0),
            0.99,
        )
        self.cold_false_hot_gate_power = max(
            getattr(config.loss, "cold_false_hot_gate_power", 1.0), 1.0e-6
        )
        self.lambda_specialist_false_hot = getattr(
            config.loss, "lambda_specialist_false_hot", 0.0
        )
        self.specialist_false_hot_target_threshold = getattr(
            config.loss, "specialist_false_hot_target_threshold", 1200.0
        )
        self.specialist_false_hot_pred_threshold = getattr(
            config.loss, "specialist_false_hot_pred_threshold", 1400.0
        )
        self.specialist_false_hot_scale = getattr(
            config.loss, "specialist_false_hot_scale", 1000.0
        )
        if self.specialist_false_hot_scale <= 0:
            raise ValueError("specialist_false_hot_scale must be positive.")
        self.specialist_false_hot_gate_source = str(getattr(
            config.loss, "specialist_false_hot_gate_source", "specialist"
        )).lower()
        if self.specialist_false_hot_gate_source not in {
            "none", "specialist", "process", "arrival", "neighbor",
            "arrival_process_product", "arrival_or_process",
        }:
            raise ValueError(
                "specialist_false_hot_gate_source must be 'none', 'specialist', "
                "'process', 'arrival', 'neighbor', 'arrival_process_product', "
                "or 'arrival_or_process'."
            )
        self.specialist_false_hot_gate_threshold = min(
            max(getattr(config.loss, "specialist_false_hot_gate_threshold", 0.1), 0.0),
            0.99,
        )
        self.specialist_false_hot_gate_power = max(
            getattr(config.loss, "specialist_false_hot_gate_power", 1.0), 1.0e-6
        )
        self.lambda_laser_residual = getattr(config.loss, "lambda_laser_residual", 0.0)
        self.laser_residual_threshold = getattr(
            config.loss, "laser_residual_threshold", 0.0
        )
        if self.laser_residual_threshold <= 0:
            self.laser_residual_threshold = getattr(material_props, "solidus_temp", 1604.85)
        self.laser_residual_scale = getattr(config.loss, "laser_residual_scale", 1000.0)
        if self.laser_residual_scale <= 0:
            raise ValueError("laser_residual_scale must be positive.")
        self.laser_residual_peak_weight = getattr(
            config.loss, "laser_residual_peak_weight", 0.0
        )
        self.laser_residual_weight_power = max(
            getattr(config.loss, "laser_residual_weight_power", 1.0), 1.0e-6
        )
        self.laser_residual_gate_source = str(getattr(
            config.loss, "laser_residual_gate_source", "residual_prior"
        )).lower()
        if self.laser_residual_gate_source not in {
            "none", "residual_prior", "residual_gate", "arrival", "process",
            "arrival_or_process", "arrival_process_product",
        }:
            raise ValueError(
                "laser_residual_gate_source must be 'none', 'residual_prior', "
                "'residual_gate', 'arrival', 'process', 'arrival_or_process', "
                "or 'arrival_process_product'."
            )
        self.laser_residual_gate_threshold = min(
            max(getattr(config.loss, "laser_residual_gate_threshold", 0.0), 0.0),
            0.99,
        )
        self.laser_residual_gate_power = max(
            getattr(config.loss, "laser_residual_gate_power", 1.0), 1.0e-6
        )
        self.laser_residual_prev_max_temp = getattr(
            config.loss, "laser_residual_prev_max_temp", 0.0
        )
        self.laser_residual_min_target_gap = getattr(
            config.loss, "laser_residual_min_target_gap", 0.0
        )
        self.lambda_laser_residual_delta = getattr(
            config.loss, "lambda_laser_residual_delta", 0.0
        )
        self.laser_residual_delta_scale = getattr(
            config.loss, "laser_residual_delta_scale", 1000.0
        )
        if self.laser_residual_delta_scale <= 0:
            raise ValueError("laser_residual_delta_scale must be positive.")
        self.laser_residual_delta_gate_threshold = min(
            max(getattr(config.loss, "laser_residual_delta_gate_threshold", 0.0), 0.0),
            0.99,
        )
        self.laser_residual_delta_gate_power = max(
            getattr(config.loss, "laser_residual_delta_gate_power", 1.0), 1.0e-6
        )
        self.laser_residual_delta_target_divide_gate = bool(getattr(
            config.loss, "laser_residual_delta_target_divide_gate", True
        ))
        self.laser_residual_delta_target_max = max(
            getattr(config.loss, "laser_residual_delta_target_max", 3600.0), 0.0
        )
        self.laser_residual_delta_min_support = max(
            getattr(config.loss, "laser_residual_delta_min_support", 0.05), 1.0e-6
        )
        self.laser_residual_delta_prev_max_temp = getattr(
            config.loss, "laser_residual_delta_prev_max_temp", 0.0
        )
        self.laser_residual_delta_min_target_gap = getattr(
            config.loss, "laser_residual_delta_min_target_gap", 0.0
        )
        self.lambda_laser_residual_gate = getattr(
            config.loss, "lambda_laser_residual_gate", 0.0
        )
        self.laser_residual_gate_pos_threshold = getattr(
            config.loss, "laser_residual_gate_pos_threshold", 0.0
        )
        if self.laser_residual_gate_pos_threshold <= 0:
            self.laser_residual_gate_pos_threshold = self.laser_residual_threshold
        self.laser_residual_gate_neg_threshold = getattr(
            config.loss, "laser_residual_gate_neg_threshold", 1200.0
        )
        self.laser_residual_gate_support_threshold = min(
            max(getattr(config.loss, "laser_residual_gate_support_threshold", 0.0), 0.0),
            0.99,
        )
        self.laser_residual_gate_prev_max_temp = getattr(
            config.loss, "laser_residual_gate_prev_max_temp", 0.0
        )
        self.laser_residual_gate_min_target_gap = getattr(
            config.loss, "laser_residual_gate_min_target_gap", 0.0
        )
        self.laser_residual_gate_pos_weight = max(
            getattr(config.loss, "laser_residual_gate_pos_weight", 0.0), 0.0
        )
        self.laser_residual_gate_max_pos_weight = max(
            getattr(config.loss, "laser_residual_gate_max_pos_weight", 2000.0), 1.0
        )
        self.lambda_cold_to_hot_cls = getattr(
            config.loss, "lambda_cold_to_hot_cls", 0.0
        )
        self.cold_to_hot_threshold = getattr(
            config.loss, "cold_to_hot_threshold", 0.0
        )
        if self.cold_to_hot_threshold <= 0:
            self.cold_to_hot_threshold = self.laser_residual_threshold
        self.cold_to_hot_neg_threshold = getattr(
            config.loss, "cold_to_hot_neg_threshold", 1200.0
        )
        self.cold_to_hot_prev_max_temp = getattr(
            config.loss, "cold_to_hot_prev_max_temp", 120.0
        )
        self.cold_to_hot_min_target_gap = getattr(
            config.loss, "cold_to_hot_min_target_gap", 1200.0
        )
        self.cold_to_hot_support_gate_source = str(getattr(
            config.loss, "cold_to_hot_support_gate_source", "residual_prior"
        )).lower()
        if self.cold_to_hot_support_gate_source not in {
                "none", "residual_prior", "residual_gate", "arrival",
                "process", "arrival_or_process", "arrival_process_product"}:
            raise ValueError(
                "cold_to_hot_support_gate_source must be one of: none, "
                "residual_prior, residual_gate, arrival, process, "
                "arrival_or_process, arrival_process_product."
            )
        self.cold_to_hot_support_threshold = min(
            max(getattr(config.loss, "cold_to_hot_support_threshold", 0.0), 0.0),
            0.99,
        )
        self.cold_to_hot_pos_weight = max(
            getattr(config.loss, "cold_to_hot_pos_weight", 0.0), 0.0
        )
        self.cold_to_hot_max_pos_weight = max(
            getattr(config.loss, "cold_to_hot_max_pos_weight", 2500.0), 1.0
        )
        self.lambda_final_T = getattr(config.loss, "lambda_final_T", 0.0)
        self.hot_peak_ref_temp = max(
            getattr(config.loss, "hot_peak_ref_temp", 3000.0),
            self.hot_specialist_threshold + 1.0,
            self.hot_final_threshold + 1.0,
            self.laser_residual_threshold + 1.0,
        )
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
                hotspot_logit: torch.Tensor | None = None,
                hotspot_delta: torch.Tensor | None = None,
                hotspot_specialist_temp: torch.Tensor | None = None,
                hotspot_specialist_gate: torch.Tensor | None = None,
                process_gate: torch.Tensor | None = None,
                laser_arrival_gate: torch.Tensor | None = None,
                laser_path_post_arrival_gate: torch.Tensor | None = None,
                laser_endpoint_gate: torch.Tensor | None = None,
                neighbor_hot_gate: torch.Tensor | None = None,
                final_pred: torch.Tensor | None = None,
                laser_residual_prior_gate: torch.Tensor | None = None,
                laser_residual_learned_gate: torch.Tensor | None = None,
                laser_residual_gate: torch.Tensor | None = None,
                laser_residual_delta: torch.Tensor | None = None,
                laser_residual_boost: torch.Tensor | None = None,
                cold_to_hot_logit: torch.Tensor | None = None,
                cold_to_hot_gate: torch.Tensor | None = None,
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

        if self.lambda_hot_cls > 0 and hotspot_logit is not None:
            logit = self._as_node_field(hotspot_logit, "hotspot_logit").squeeze(-1)
            label = (T_target >= self.hot_cls_threshold).to(dtype=logit.dtype)
            if mask_bool is not None:
                logit = logit[mask_bool]
                label = label[mask_bool]
            if label.numel() > 0:
                pos = label.sum()
                neg = label.numel() - pos
                if self.hot_cls_pos_weight > 0:
                    pos_weight_value = self.hot_cls_pos_weight
                elif pos.item() > 0:
                    pos_weight_value = (neg / pos.clamp_min(1.0)).item()
                else:
                    pos_weight_value = 1.0
                pos_weight_value = min(pos_weight_value, self.hot_cls_max_pos_weight)
                pos_weight = torch.as_tensor(
                    pos_weight_value, device=logit.device, dtype=logit.dtype
                )
                losses["HotCls"] = self.lambda_hot_cls * F.binary_cross_entropy_with_logits(
                    logit, label, pos_weight=pos_weight
                )

        if self.lambda_hot_delta > 0 and hotspot_delta is not None and prev_temp is not None:
            delta_pred = self._as_node_field(hotspot_delta, "hotspot_delta").squeeze(-1)
            delta_target = (T_target - prev_temp.squeeze(-1)).clamp_min(0.0)
            hot_delta_mask = T_target >= self.hot_delta_threshold
            if mask_bool is not None:
                hot_delta_mask &= mask_bool
            if hot_delta_mask.any():
                pred_scaled = delta_pred[hot_delta_mask] / self.hot_delta_scale
                target_scaled = delta_target[hot_delta_mask] / self.hot_delta_scale
                losses["HotDelta"] = self.lambda_hot_delta * F.mse_loss(
                    pred_scaled, target_scaled
                )

        if self.lambda_hot_specialist > 0 and hotspot_specialist_temp is not None:
            spec_pred = self._as_node_field(
                hotspot_specialist_temp, "hotspot_specialist_temp"
            ).squeeze(-1)
            spec_mask = T_target >= self.hot_specialist_threshold
            candidate_min_temp = self.hot_specialist_candidate_min_temp
            if candidate_min_temp > 0:
                candidate_mask = T_target >= candidate_min_temp
                if (
                    process_gate is not None
                    and self.hot_specialist_process_gate_threshold > 0
                ):
                    process_score = self._as_node_field(
                        process_gate, "process_gate"
                    ).squeeze(-1)
                    spec_mask |= (
                        candidate_mask
                        & (process_score >= self.hot_specialist_process_gate_threshold)
                    )
                if (
                    neighbor_hot_gate is not None
                    and self.hot_specialist_neighbor_gate_threshold > 0
                ):
                    neighbor_score = self._as_node_field(
                        neighbor_hot_gate, "neighbor_hot_gate"
                    ).squeeze(-1)
                    spec_mask |= (
                        candidate_mask
                        & (neighbor_score >= self.hot_specialist_neighbor_gate_threshold)
                    )
            if mask_bool is not None:
                spec_mask &= mask_bool
            if spec_mask.any():
                pred_scaled = spec_pred[spec_mask] / self.hot_specialist_scale
                target_scaled = T_target[spec_mask] / self.hot_specialist_scale
                spec_loss = (pred_scaled - target_scaled).pow(2)
                spec_loss = self._apply_peak_weights(
                    spec_loss,
                    T_target[spec_mask],
                    threshold=self.hot_specialist_threshold,
                    peak_weight=self.hot_specialist_peak_weight,
                    weight_power=self.hot_specialist_weight_power,
                )
                losses["HotSpec"] = self.lambda_hot_specialist * spec_loss.mean()

        if (
                self.lambda_hot_process_specialist > 0
                and hotspot_specialist_temp is not None):
            spec_pred = self._as_node_field(
                hotspot_specialist_temp, "hotspot_specialist_temp"
            ).squeeze(-1)

            process_score = None
            if process_gate is not None:
                process_score = self._as_node_field(
                    process_gate, "process_gate"
                ).squeeze(-1).clamp(0.0, 1.0)

            arrival_score = None
            if laser_arrival_gate is not None:
                arrival_score = self._as_node_field(
                    laser_arrival_gate, "laser_arrival_gate"
                ).squeeze(-1).clamp(0.0, 1.0)

            residual_prior_score = None
            if laser_residual_prior_gate is not None:
                residual_prior_score = self._as_node_field(
                    laser_residual_prior_gate, "laser_residual_prior_gate"
                ).squeeze(-1).clamp(0.0, 1.0)

            residual_gate_score = None
            if laser_residual_gate is not None:
                residual_gate_score = self._as_node_field(
                    laser_residual_gate, "laser_residual_gate"
                ).squeeze(-1).clamp(0.0, 1.0)

            support_score = None
            if self.hot_process_specialist_gate_source == "process":
                support_score = process_score
            elif self.hot_process_specialist_gate_source == "arrival":
                support_score = arrival_score
            elif self.hot_process_specialist_gate_source == "residual_prior":
                support_score = residual_prior_score
            elif self.hot_process_specialist_gate_source == "residual_gate":
                support_score = residual_gate_score
            elif self.hot_process_specialist_gate_source == "arrival_process_product":
                if process_score is not None and arrival_score is not None:
                    support_score = process_score * arrival_score
            else:
                support_gates = [
                    gate for gate in (arrival_score, process_score) if gate is not None
                ]
                if support_gates:
                    support_score = support_gates[0]
                    for gate in support_gates[1:]:
                        support_score = torch.maximum(support_score, gate)

            if support_score is not None:
                threshold = self.hot_process_specialist_gate_threshold
                if threshold > 0:
                    support_weight = (
                        (support_score - threshold) / max(1.0 - threshold, 1.0e-6)
                    ).clamp(0.0, 1.0)
                else:
                    support_weight = support_score.clamp(0.0, 1.0)
                support_weight = support_weight.pow(
                    self.hot_process_specialist_gate_power
                )

                process_hot_mask = (
                    (T_target >= self.hot_process_specialist_threshold)
                    & (support_weight > 0)
                )
                if (
                        self.hot_process_specialist_prev_max_temp > 0
                        and prev_temp is not None):
                    prev_field = prev_temp.squeeze(-1)
                    process_hot_mask &= (
                        prev_field <= self.hot_process_specialist_prev_max_temp
                    )
                if mask_bool is not None:
                    process_hot_mask &= mask_bool
                if process_hot_mask.any():
                    pred_scaled = (
                        spec_pred[process_hot_mask]
                        / self.hot_process_specialist_scale
                    )
                    target_scaled = (
                        T_target[process_hot_mask]
                        / self.hot_process_specialist_scale
                    )
                    process_spec_loss = (pred_scaled - target_scaled).pow(2)
                    process_spec_loss = self._apply_peak_weights(
                        process_spec_loss,
                        T_target[process_hot_mask],
                        threshold=self.hot_process_specialist_threshold,
                        peak_weight=self.hot_process_specialist_peak_weight,
                        weight_power=self.hot_process_specialist_weight_power,
                    )
                    process_spec_loss = (
                        process_spec_loss * support_weight[process_hot_mask]
                    )
                    losses["HotSpecProcess"] = (
                        self.lambda_hot_process_specialist * process_spec_loss.mean()
                    )

        if self.lambda_hot_final > 0:
            final_pred = pred if final_pred is None else self._as_node_field(
                final_pred, "final_pred"
            )
            T_final_pred = final_pred.squeeze(-1)
            final_mask = T_target >= self.hot_final_threshold
            if mask_bool is not None:
                final_mask &= mask_bool
            if final_mask.any():
                pred_scaled = T_final_pred[final_mask] / self.hot_final_scale
                target_scaled = T_target[final_mask] / self.hot_final_scale
                final_loss = (pred_scaled - target_scaled).pow(2)
                final_loss = self._apply_peak_weights(
                    final_loss,
                    T_target[final_mask],
                    threshold=self.hot_final_threshold,
                    peak_weight=self.hot_final_peak_weight,
                    weight_power=self.hot_final_weight_power,
                )
                losses["HotFinal"] = self.lambda_hot_final * final_loss.mean()

        if self.lambda_cold_false_hot > 0:
            final_field = pred if final_pred is None else self._as_node_field(
                final_pred, "final_pred"
            )
            T_final_pred = final_field.squeeze(-1)
            gate_score = self._select_cold_false_hot_gate(
                hotspot_specialist_gate=hotspot_specialist_gate,
                process_gate=process_gate,
                laser_arrival_gate=laser_arrival_gate,
                laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                laser_endpoint_gate=laser_endpoint_gate,
                neighbor_hot_gate=neighbor_hot_gate,
                like=T_final_pred,
            )
            threshold = self.cold_false_hot_gate_threshold
            if threshold > 0:
                gate_weight = (
                    (gate_score - threshold) / max(1.0 - threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                gate_weight = gate_score.clamp(0.0, 1.0)
            gate_weight = gate_weight.pow(self.cold_false_hot_gate_power)

            false_hot_mask = (
                (T_target <= self.cold_false_hot_target_threshold)
                & (T_final_pred > self.cold_false_hot_pred_threshold)
                & (gate_weight > 0)
            )
            if mask_bool is not None:
                false_hot_mask &= mask_bool
            if false_hot_mask.any():
                excess = F.relu(
                    T_final_pred[false_hot_mask] - self.cold_false_hot_pred_threshold
                ) / self.cold_false_hot_scale
                cold_false_hot_loss = excess.pow(2) * gate_weight[false_hot_mask]
                losses["ColdFalseHot"] = (
                    self.lambda_cold_false_hot * cold_false_hot_loss.mean()
                )

        if (
                self.lambda_specialist_false_hot > 0
                and hotspot_specialist_temp is not None):
            specialist_temp = self._as_node_field(
                hotspot_specialist_temp, "hotspot_specialist_temp"
            ).squeeze(-1)
            gate_score = self._select_cold_false_hot_gate(
                hotspot_specialist_gate=hotspot_specialist_gate,
                process_gate=process_gate,
                laser_arrival_gate=laser_arrival_gate,
                laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                laser_endpoint_gate=laser_endpoint_gate,
                neighbor_hot_gate=neighbor_hot_gate,
                like=specialist_temp,
                source=self.specialist_false_hot_gate_source,
            )
            threshold = self.specialist_false_hot_gate_threshold
            if threshold > 0:
                gate_weight = (
                    (gate_score - threshold) / max(1.0 - threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                gate_weight = gate_score.clamp(0.0, 1.0)
            gate_weight = gate_weight.pow(self.specialist_false_hot_gate_power)

            false_hot_mask = (
                (T_target <= self.specialist_false_hot_target_threshold)
                & (specialist_temp > self.specialist_false_hot_pred_threshold)
                & (gate_weight > 0)
            )
            if mask_bool is not None:
                false_hot_mask &= mask_bool
            if false_hot_mask.any():
                excess = F.relu(
                    specialist_temp[false_hot_mask]
                    - self.specialist_false_hot_pred_threshold
                ) / self.specialist_false_hot_scale
                specialist_false_hot_loss = excess.pow(2) * gate_weight[false_hot_mask]
                losses["SpecialistFalseHot"] = (
                    self.lambda_specialist_false_hot
                    * specialist_false_hot_loss.mean()
                )

        if self.lambda_laser_residual > 0 and laser_residual_boost is not None:
            boost_pred = self._as_node_field(
                laser_residual_boost, "laser_residual_boost"
            ).squeeze(-1)
            final_field = pred if final_pred is None else self._as_node_field(
                final_pred, "final_pred"
            )
            T_final_pred = final_field.squeeze(-1)
            support_score = self._select_laser_residual_gate(
                laser_residual_prior_gate=laser_residual_prior_gate,
                laser_residual_gate=laser_residual_gate,
                process_gate=process_gate,
                laser_arrival_gate=laser_arrival_gate,
                like=boost_pred,
            )
            threshold = self.laser_residual_gate_threshold
            if threshold > 0:
                support_weight = (
                    (support_score - threshold) / max(1.0 - threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                support_weight = support_score.clamp(0.0, 1.0)
            support_weight = support_weight.pow(self.laser_residual_gate_power)

            pre_residual_pred = (T_final_pred - boost_pred).detach()
            target_boost = (T_target - pre_residual_pred).clamp_min(0.0)
            residual_mask = (
                (T_target >= self.laser_residual_threshold)
                & (support_weight > 0)
            )
            if self.laser_residual_prev_max_temp > 0 and prev_temp is not None:
                residual_mask &= prev_temp.squeeze(-1) <= self.laser_residual_prev_max_temp
            if self.laser_residual_min_target_gap > 0:
                residual_mask &= target_boost >= self.laser_residual_min_target_gap
            if mask_bool is not None:
                residual_mask &= mask_bool
            if residual_mask.any():
                pred_scaled = boost_pred[residual_mask] / self.laser_residual_scale
                target_scaled = target_boost[residual_mask] / self.laser_residual_scale
                residual_loss = (pred_scaled - target_scaled).pow(2)
                residual_loss = self._apply_peak_weights(
                    residual_loss,
                    T_target[residual_mask],
                    threshold=self.laser_residual_threshold,
                    peak_weight=self.laser_residual_peak_weight,
                    weight_power=self.laser_residual_weight_power,
                )
                residual_loss = residual_loss * support_weight[residual_mask]
                losses["LaserResidual"] = (
                    self.lambda_laser_residual * residual_loss.mean()
                )

        if self.lambda_laser_residual_delta > 0 and laser_residual_delta is not None:
            delta_pred = self._as_node_field(
                laser_residual_delta, "laser_residual_delta"
            ).squeeze(-1)
            boost_pred = None
            if laser_residual_boost is not None:
                boost_pred = self._as_node_field(
                    laser_residual_boost, "laser_residual_boost"
                ).squeeze(-1)
            final_field = pred if final_pred is None else self._as_node_field(
                final_pred, "final_pred"
            )
            T_final_pred = final_field.squeeze(-1)
            support_score = self._select_laser_residual_gate(
                laser_residual_prior_gate=laser_residual_prior_gate,
                laser_residual_gate=laser_residual_gate,
                process_gate=process_gate,
                laser_arrival_gate=laser_arrival_gate,
                like=delta_pred,
            )
            threshold = self.laser_residual_delta_gate_threshold
            if threshold > 0:
                support_weight = (
                    (support_score - threshold) / max(1.0 - threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                support_weight = support_score.clamp(0.0, 1.0)
            support_weight = support_weight.pow(self.laser_residual_delta_gate_power)

            if boost_pred is not None:
                pre_residual_pred = (T_final_pred - boost_pred).detach()
            else:
                pre_residual_pred = T_final_pred.detach()
            target_delta = (T_target - pre_residual_pred).clamp_min(0.0)
            if self.laser_residual_delta_target_divide_gate:
                target_delta = target_delta / support_weight.clamp_min(
                    self.laser_residual_delta_min_support
                )
            if self.laser_residual_delta_target_max > 0:
                target_delta = target_delta.clamp(
                    max=self.laser_residual_delta_target_max
                )

            delta_mask = (
                (T_target >= self.laser_residual_threshold)
                & (support_weight > 0)
            )
            if self.laser_residual_delta_prev_max_temp > 0 and prev_temp is not None:
                delta_mask &= prev_temp.squeeze(-1) <= self.laser_residual_delta_prev_max_temp
            if self.laser_residual_delta_min_target_gap > 0:
                raw_gap = (T_target - pre_residual_pred).clamp_min(0.0)
                delta_mask &= raw_gap >= self.laser_residual_delta_min_target_gap
            if mask_bool is not None:
                delta_mask &= mask_bool
            if delta_mask.any():
                delta_loss = (
                    delta_pred[delta_mask] / self.laser_residual_delta_scale
                    - target_delta[delta_mask] / self.laser_residual_delta_scale
                ).pow(2)
                delta_loss = self._apply_peak_weights(
                    delta_loss,
                    T_target[delta_mask],
                    threshold=self.laser_residual_threshold,
                    peak_weight=self.laser_residual_peak_weight,
                    weight_power=self.laser_residual_weight_power,
                )
                delta_loss = delta_loss * support_weight[delta_mask]
                losses["LaserResidualDelta"] = (
                    self.lambda_laser_residual_delta * delta_loss.mean()
                )

        if (
                self.lambda_laser_residual_gate > 0
                and laser_residual_learned_gate is not None):
            learned_gate = self._as_node_field(
                laser_residual_learned_gate, "laser_residual_learned_gate"
            ).squeeze(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
            support_score = self._select_laser_residual_gate(
                laser_residual_prior_gate=laser_residual_prior_gate,
                laser_residual_gate=None,
                process_gate=process_gate,
                laser_arrival_gate=laser_arrival_gate,
                like=learned_gate,
            )
            support_mask = support_score > self.laser_residual_gate_support_threshold
            pos_mask = (
                (T_target >= self.laser_residual_gate_pos_threshold)
                & support_mask
            )
            if self.laser_residual_gate_prev_max_temp > 0 and prev_temp is not None:
                prev_field = prev_temp.squeeze(-1)
                pos_mask &= prev_field <= self.laser_residual_gate_prev_max_temp
            if self.laser_residual_gate_min_target_gap > 0 and prev_temp is not None:
                prev_field = prev_temp.squeeze(-1)
                target_gap_from_input = (T_target - prev_field).clamp_min(0.0)
                pos_mask &= (
                    target_gap_from_input >= self.laser_residual_gate_min_target_gap
                )
            neg_mask = (
                (T_target <= self.laser_residual_gate_neg_threshold)
                & support_mask
            )
            if mask_bool is not None:
                pos_mask &= mask_bool
                neg_mask &= mask_bool
            gate_mask = pos_mask | neg_mask
            if gate_mask.any():
                labels = pos_mask[gate_mask].to(dtype=learned_gate.dtype)
                gate_weight = torch.ones_like(labels)
                num_pos = labels.sum()
                num_neg = labels.numel() - num_pos
                if num_pos > 0:
                    if self.laser_residual_gate_pos_weight > 0:
                        pos_weight = self.laser_residual_gate_pos_weight
                    elif num_neg > 0:
                        pos_weight = (num_neg / num_pos).clamp(
                            max=self.laser_residual_gate_max_pos_weight
                        ).item()
                    else:
                        pos_weight = 1.0
                    gate_weight = torch.where(
                        labels > 0.5,
                        torch.full_like(labels, float(pos_weight)),
                        gate_weight,
                    )
                gate_prob = learned_gate[gate_mask].float().clamp(
                    1.0e-6, 1.0 - 1.0e-6
                )
                labels = labels.float()
                gate_weight = gate_weight.float()
                gate_loss = -(
                    labels * torch.log(gate_prob)
                    + (1.0 - labels) * torch.log(1.0 - gate_prob)
                )
                gate_loss = (gate_loss * gate_weight).mean()
                losses["LaserResidualGate"] = (
                    self.lambda_laser_residual_gate * gate_loss
                )

        if self.lambda_cold_to_hot_cls > 0 and cold_to_hot_logit is not None:
            logit = self._as_node_field(
                cold_to_hot_logit, "cold_to_hot_logit"
            ).squeeze(-1)
            support_score = self._select_laser_residual_gate(
                laser_residual_prior_gate=laser_residual_prior_gate,
                laser_residual_gate=laser_residual_gate,
                process_gate=process_gate,
                laser_arrival_gate=laser_arrival_gate,
                like=logit,
                source=self.cold_to_hot_support_gate_source,
            )
            support_mask = support_score > self.cold_to_hot_support_threshold
            pos_mask = (T_target >= self.cold_to_hot_threshold) & support_mask
            if self.cold_to_hot_prev_max_temp > 0 and prev_temp is not None:
                prev_field = prev_temp.squeeze(-1)
                pos_mask &= prev_field <= self.cold_to_hot_prev_max_temp
            if self.cold_to_hot_min_target_gap > 0 and prev_temp is not None:
                prev_field = prev_temp.squeeze(-1)
                target_gap_from_input = (T_target - prev_field).clamp_min(0.0)
                pos_mask &= target_gap_from_input >= self.cold_to_hot_min_target_gap

            neg_mask = (T_target <= self.cold_to_hot_neg_threshold) & support_mask
            if mask_bool is not None:
                pos_mask &= mask_bool
                neg_mask &= mask_bool
            cls_mask = pos_mask | neg_mask
            if cls_mask.any():
                labels = pos_mask[cls_mask].to(dtype=logit.dtype)
                cls_logit = logit[cls_mask].float()
                labels = labels.float()
                num_pos = labels.sum()
                num_neg = labels.numel() - num_pos
                if self.cold_to_hot_pos_weight > 0:
                    pos_weight_value = self.cold_to_hot_pos_weight
                elif num_pos.item() > 0 and num_neg.item() > 0:
                    pos_weight_value = (num_neg / num_pos).clamp(
                        max=self.cold_to_hot_max_pos_weight
                    ).item()
                else:
                    pos_weight_value = 1.0
                pos_weight = torch.as_tensor(
                    min(float(pos_weight_value), self.cold_to_hot_max_pos_weight),
                    device=cls_logit.device,
                    dtype=cls_logit.dtype,
                )
                losses["ColdToHotCls"] = (
                    self.lambda_cold_to_hot_cls
                    * F.binary_cross_entropy_with_logits(
                        cls_logit, labels, pos_weight=pos_weight
                    )
                )

        if self.lambda_final_T > 0 and final_pred is not None:
            final_field = self._as_node_field(final_pred, "final_pred")
            final_sq_error = (final_field.squeeze(-1) - T_target) ** 2
            if mask_bool is not None:
                final_sq_error = final_sq_error[mask_bool]
            losses["FinalT"] = self.lambda_final_T * final_sq_error.mean()

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

    def _select_cold_false_hot_gate(
            self,
            *,
            hotspot_specialist_gate: torch.Tensor | None,
            process_gate: torch.Tensor | None,
            laser_arrival_gate: torch.Tensor | None,
            laser_path_post_arrival_gate: torch.Tensor | None,
            laser_endpoint_gate: torch.Tensor | None,
            neighbor_hot_gate: torch.Tensor | None,
            like: torch.Tensor,
            source: str | None = None,
            ) -> torch.Tensor:
        def norm_gate(gate: torch.Tensor | None, name: str) -> torch.Tensor | None:
            if gate is None:
                return None
            return self._as_node_field(gate, name).squeeze(-1).clamp(0.0, 1.0)

        source = self.cold_false_hot_gate_source if source is None else source
        if source == "none":
            return torch.ones_like(like)

        specialist_score = norm_gate(hotspot_specialist_gate, "hotspot_specialist_gate")
        process_score = norm_gate(process_gate, "process_gate")
        arrival_score = norm_gate(laser_arrival_gate, "laser_arrival_gate")
        path_post_score = norm_gate(
            laser_path_post_arrival_gate, "laser_path_post_arrival_gate"
        )
        endpoint_score = norm_gate(laser_endpoint_gate, "laser_endpoint_gate")
        neighbor_score = norm_gate(neighbor_hot_gate, "neighbor_hot_gate")

        if source == "specialist":
            score = specialist_score
        elif source == "process":
            score = process_score
        elif source == "arrival":
            score = arrival_score
        elif source == "neighbor":
            score = neighbor_score
        elif source == "arrival_process_product":
            score = None
            if process_score is not None and arrival_score is not None:
                score = process_score * arrival_score
        elif source == "path_post_endpoint":
            score = None
            if path_post_score is not None and endpoint_score is not None:
                score = path_post_score * endpoint_score
        else:
            gates = [gate for gate in (arrival_score, process_score) if gate is not None]
            score = None
            if gates:
                score = gates[0]
                for gate in gates[1:]:
                    score = torch.maximum(score, gate)

        if score is None:
            return torch.zeros_like(like)
        return score

    def _select_laser_residual_gate(
            self,
            *,
            laser_residual_prior_gate: torch.Tensor | None,
            laser_residual_gate: torch.Tensor | None,
            process_gate: torch.Tensor | None,
            laser_arrival_gate: torch.Tensor | None,
            like: torch.Tensor,
            source: str | None = None,
            ) -> torch.Tensor:
        def norm_gate(gate: torch.Tensor | None, name: str) -> torch.Tensor | None:
            if gate is None:
                return None
            return self._as_node_field(gate, name).squeeze(-1).clamp(0.0, 1.0)

        source = self.laser_residual_gate_source if source is None else source
        if source == "none":
            return torch.ones_like(like)

        prior_score = norm_gate(laser_residual_prior_gate, "laser_residual_prior_gate")
        residual_score = norm_gate(laser_residual_gate, "laser_residual_gate")
        process_score = norm_gate(process_gate, "process_gate")
        arrival_score = norm_gate(laser_arrival_gate, "laser_arrival_gate")

        if source == "residual_prior":
            score = prior_score
        elif source == "residual_gate":
            score = residual_score
        elif source == "process":
            score = process_score
        elif source == "arrival":
            score = arrival_score
        elif source == "arrival_process_product":
            score = None
            if process_score is not None and arrival_score is not None:
                score = process_score * arrival_score
        else:
            gates = [gate for gate in (arrival_score, process_score) if gate is not None]
            score = None
            if gates:
                score = gates[0]
                for gate in gates[1:]:
                    score = torch.maximum(score, gate)

        if score is None:
            return torch.zeros_like(like)
        return score

    def _apply_peak_weights(self, loss: torch.Tensor, target: torch.Tensor, *,
                            threshold: float, peak_weight: float,
                            weight_power: float) -> torch.Tensor:
        if peak_weight <= 0:
            return loss
        ratio = (
            (target - threshold)
            / max(self.hot_peak_ref_temp - threshold, 1.0e-6)
        ).clamp(0.0, 1.0)
        weights = 1.0 + peak_weight * ratio.pow(weight_power)
        return loss * weights

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
