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
        self.predict_temperature_delta = getattr(
            config.model, "predict_temperature_delta", False
        )
        self.enable_hotspot_head = getattr(config.model, "enable_hotspot_head", False)
        self.enable_direct_laser_head = getattr(
            config.model, "enable_direct_laser_head", False
        )
        self.direct_laser_features_to_head = bool(getattr(
            config.model, "direct_laser_features_to_head", True
        ))
        self.enable_hotspot_delta_head = getattr(
            config.model, "enable_hotspot_delta_head", False
        )
        self.enable_hotspot_specialist_head = getattr(
            config.model, "enable_hotspot_specialist_head", False
        )
        self.hotspot_specialist_min_temp = float(getattr(
            config.model, "hotspot_specialist_min_temp",
            getattr(material_props, "ambient_temp", 20.0),
        ))
        self.hotspot_specialist_max_temp = float(getattr(
            config.model, "hotspot_specialist_max_temp", 3500.0,
        ))
        self.hotspot_specialist_gate_threshold = float(getattr(
            config.model, "hotspot_specialist_gate_threshold", 0.5,
        ))
        self.hotspot_specialist_gate_threshold = min(
            max(self.hotspot_specialist_gate_threshold, 0.0), 0.99
        )
        self.hotspot_specialist_process_gate_threshold = float(getattr(
            config.model, "hotspot_specialist_process_gate_threshold", 0.0,
        ))
        self.hotspot_specialist_process_gate_threshold = min(
            max(self.hotspot_specialist_process_gate_threshold, 0.0), 0.99
        )
        self.hotspot_specialist_process_gate_power = max(
            float(getattr(config.model, "hotspot_specialist_process_gate_power", 1.0)),
            1.0e-6,
        )
        self.hotspot_specialist_process_gate_max = min(
            max(float(getattr(config.model, "hotspot_specialist_process_gate_max", 1.0)), 0.0),
            1.0,
        )
        self.hotspot_specialist_gate_power = float(getattr(
            config.model, "hotspot_specialist_gate_power", 1.0,
        ))
        self.hotspot_specialist_gate_power = max(
            self.hotspot_specialist_gate_power, 1.0e-6
        )
        self.hotspot_specialist_use_cls_gate = bool(getattr(
            config.model, "hotspot_specialist_use_cls_gate", True,
        ))
        self.hotspot_specialist_hard_process_gate = bool(getattr(
            config.model, "hotspot_specialist_hard_process_gate", False,
        ))
        self.hotspot_specialist_neighbor_gate_threshold = float(getattr(
            config.model, "hotspot_specialist_neighbor_gate_threshold", 0.0,
        ))
        self.hotspot_specialist_neighbor_gate_threshold = min(
            max(self.hotspot_specialist_neighbor_gate_threshold, 0.0), 0.99
        )
        self.hotspot_specialist_neighbor_gate_power = max(
            float(getattr(config.model, "hotspot_specialist_neighbor_gate_power", 1.0)),
            1.0e-6,
        )
        self.hotspot_specialist_neighbor_gate_max = min(
            max(float(getattr(config.model, "hotspot_specialist_neighbor_gate_max", 1.0)), 0.0),
            1.0,
        )
        self.hotspot_specialist_gate_merge = str(getattr(
            config.model, "hotspot_specialist_gate_merge", "max",
        )).lower()
        self.hotspot_specialist_neighbor_support_base = min(
            max(float(getattr(
                config.model,
                "hotspot_specialist_neighbor_support_base",
                0.5,
            )), 0.0),
            1.0,
        )
        self.hotspot_specialist_neighbor_support_weight = min(
            max(float(getattr(
                config.model,
                "hotspot_specialist_neighbor_support_weight",
                0.5,
            )), 0.0),
            1.0,
        )
        self.hotspot_specialist_blend_warmup_epochs = max(
            0, int(getattr(config.model, "hotspot_specialist_blend_warmup_epochs", 0))
        )
        self.hotspot_specialist_blend_ramp_epochs = max(
            0, int(getattr(config.model, "hotspot_specialist_blend_ramp_epochs", 0))
        )
        self.hotspot_specialist_detach_base = bool(getattr(
            config.model, "hotspot_specialist_detach_base", False,
        ))
        self.hotspot_specialist_hidden_dim = max(
            8, int(getattr(config.model, "hotspot_specialist_hidden_dim", 64))
        )
        self.hotspot_specialist_num_layers = max(
            1, int(getattr(config.model, "hotspot_specialist_num_layers", 1))
        )
        self.hotspot_specialist_output_mode = str(getattr(
            config.model, "hotspot_specialist_output_mode", "absolute"
        )).lower()
        if self.hotspot_specialist_output_mode not in {"absolute", "residual"}:
            raise ValueError(
                "hotspot_specialist_output_mode must be 'absolute' or 'residual'."
            )
        self.hotspot_specialist_residual_max_delta = max(
            1.0,
            float(getattr(
                config.model, "hotspot_specialist_residual_max_delta", 3500.0
            )),
        )
        self.enable_hotspot_laser_prior = bool(getattr(
            config.model, "enable_hotspot_laser_prior", False,
        ))
        self.hotspot_laser_prior_mode = str(getattr(
            config.model, "hotspot_laser_prior_mode", "specialist_floor",
        )).lower()
        if self.hotspot_laser_prior_mode not in {
                "specialist_floor", "final_floor", "blend_floor"}:
            raise ValueError(
                "hotspot_laser_prior_mode must be 'specialist_floor', "
                "'final_floor', or 'blend_floor'."
            )
        self.hotspot_laser_prior_max_delta = max(
            0.0,
            float(getattr(config.model, "hotspot_laser_prior_max_delta", 3300.0)),
        )
        self.hotspot_laser_prior_gate_power = max(
            float(getattr(config.model, "hotspot_laser_prior_gate_power", 1.0)),
            1.0e-6,
        )
        self.hotspot_laser_prior_require_neighbor = bool(getattr(
            config.model, "hotspot_laser_prior_require_neighbor", True,
        ))
        self.hotspot_laser_prior_raw_process_threshold = min(
            max(float(getattr(
                config.model,
                "hotspot_laser_prior_raw_process_threshold",
                0.0,
            )), 0.0),
            1.0,
        )
        self.hotspot_laser_prior_gate_source = str(getattr(
            config.model,
            "hotspot_laser_prior_gate_source",
            "process",
        )).lower()
        if self.hotspot_laser_prior_gate_source not in {
                "process", "raw_process", "arrival", "arrival_or_process",
                "arrival_process_product", "raw_process_or_neighbor_supported",
                "target_heat_or_neighbor_supported",
                "target_sweep",
                "target_sweep_neighbor_product",
                "target_sweep_or_neighbor_supported", "residual_prior"}:
            raise ValueError(
                "hotspot_laser_prior_gate_source must be 'process', "
                "'raw_process', 'arrival', 'arrival_or_process', "
                "'arrival_process_product', or "
                "'raw_process_or_neighbor_supported', or "
                "'target_heat_or_neighbor_supported', or "
                "'target_sweep', 'target_sweep_neighbor_product', or "
                "'target_sweep_or_neighbor_supported', or 'residual_prior'."
            )
        self.hotspot_laser_prior_gate_threshold = min(
            max(float(getattr(
                config.model,
                "hotspot_laser_prior_gate_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.hotspot_laser_prior_neighbor_threshold = min(
            max(float(getattr(
                config.model,
                "hotspot_laser_prior_neighbor_threshold",
                0.0,
            )), 0.0),
            1.0,
        )
        self.enable_laser_residual_head = bool(getattr(
            config.model, "enable_laser_residual_head", False,
        ))
        self.laser_residual_max_delta = max(
            0.0,
            float(getattr(config.model, "laser_residual_max_delta", 2200.0)),
        )
        self.laser_residual_gate_source = str(getattr(
            config.model, "laser_residual_gate_source", "arrival",
        )).lower()
        if self.laser_residual_gate_source not in {
                "learned", "arrival", "process", "arrival_or_process",
                "arrival_process_product", "arrival_sweep_product",
                "arrival_track_support_product", "arrival_track_support_blend",
                "target_heat", "target_body", "target_sweep",
                "target_sweep_or_arrival", "arrival_neighbor_product",
                "arrival_with_neighbor_support",
                "path_active_with_neighbor_support",
                "path_active_neighbor_product", "endpoint",
                "endpoint_or_arrival", "endpoint_or_sweep",
                "endpoint_sweep_or_arrival", "arrival_endpoint_product",
                "arrival_support_blend",
                "pre_arrival",
                "pre_arrival_or_arrival", "pre_arrival_with_endpoint_support",
                "pre_arrival_with_body_support",
                "pre_arrival_with_body_or_endpoint_support",
                "path_body", "path_body_or_endpoint",
                "pre_arrival_with_path_body_support",
                "pre_arrival_active_path_body_support",
                "pre_arrival_with_path_body_or_endpoint_support",
                "pre_arrival_active_body_or_endpoint_support",
                "path_phase_active_body_or_track_support",
                "pre_arrival_with_sweep_support",
                "pre_arrival_with_track_heat_or_body_high_pre_support",
                "pre_arrival_with_high_pre_track_heat_or_body_support",
                "pre_or_post_arrival_with_strong_track_support",
                "high_phase_with_track_heat_or_body_support",
                "high_phase_with_midline_body_or_track_support",
                "high_phase_with_endpoint_or_midline_body_support",
                "high_phase_with_endpoint_midline_or_active_body_support",
                "high_phase_with_active_path_body_support",
                "high_phase_with_endpoint_or_active_path_body_support",
                "high_phase_with_neighbor_or_endpoint_or_active_path_body_support",
                "high_phase_with_post_neighbor_or_endpoint_or_active_body_support",
                "high_phase_with_endpoint_active_body_or_post_sweep_neighbor_support",
                "high_phase_with_endpoint_active_body_pre_sweep_or_post_sweep_neighbor_support",
                "high_phase_with_endpoint_active_body_soft_track_or_post_sweep_neighbor_support",
                "high_phase_with_active_body_pre_sweep_neighbor_or_post_sweep_neighbor_support",
                "high_phase_with_pre_sweep_neighbor_endpoint_or_post_sweep_neighbor_support",
                "high_phase_with_pre_wake_endpoint_or_post_wake_support",
                "high_phase_with_late_endpoint_or_pre_wake_or_post_wake_support",
                "xml_post_scan_memory",
                "xml_recent_scan_memory",
                "xml_future_arrival_track_support",
                "xml_future_arrival_track_product",
                "xml_prearrival_heat_track_product",
                "xml_prearrival_strong_support_product",
                "xml_phase_strong_support_product",
                "xml_prearrival_plus_strict_post_rescue_product",
                "pre_arrival_track_support_blend",
                "pre_arrival_with_track_heat_support",
                "pre_arrival_endpoint_sweep",
                "pre_arrival_endpoint_sweep_or_arrival"}:
            raise ValueError(
                "laser_residual_gate_source must be one of: learned, arrival, "
                "process, arrival_or_process, arrival_process_product, "
                "arrival_sweep_product, arrival_track_support_product, "
                "arrival_track_support_blend, target_heat, target_body, "
                "target_sweep, "
                "target_sweep_or_arrival, "
                "arrival_neighbor_product, arrival_with_neighbor_support, "
                "path_active_with_neighbor_support, "
                "path_active_neighbor_product, "
                "endpoint, endpoint_or_arrival, endpoint_or_sweep, "
                "endpoint_sweep_or_arrival, arrival_endpoint_product, "
                "arrival_support_blend, pre_arrival, "
                "pre_arrival_or_arrival, pre_arrival_with_endpoint_support, "
                "pre_arrival_with_body_support, "
                "pre_arrival_with_body_or_endpoint_support, "
                "path_body, path_body_or_endpoint, "
                "pre_arrival_with_path_body_support, "
                "pre_arrival_active_path_body_support, "
                "pre_arrival_with_path_body_or_endpoint_support, "
                "pre_arrival_with_sweep_support, "
                "pre_arrival_with_track_heat_or_body_high_pre_support, "
                "pre_arrival_with_high_pre_track_heat_or_body_support, "
                "pre_or_post_arrival_with_strong_track_support, "
                "high_phase_with_track_heat_or_body_support, "
                "high_phase_with_midline_body_or_track_support, "
                "high_phase_with_endpoint_or_midline_body_support, "
                "high_phase_with_endpoint_midline_or_active_body_support, "
                "high_phase_with_active_path_body_support, "
                "high_phase_with_endpoint_or_active_path_body_support, "
                "high_phase_with_neighbor_or_endpoint_or_active_path_body_support, "
                "high_phase_with_post_neighbor_or_endpoint_or_active_body_support, "
                "high_phase_with_endpoint_active_body_or_post_sweep_neighbor_support, "
                "high_phase_with_endpoint_active_body_pre_sweep_or_post_sweep_neighbor_support, "
                "high_phase_with_endpoint_active_body_soft_track_or_post_sweep_neighbor_support, "
                "high_phase_with_active_body_pre_sweep_neighbor_or_post_sweep_neighbor_support, "
                "high_phase_with_pre_sweep_neighbor_endpoint_or_post_sweep_neighbor_support, "
                "high_phase_with_pre_wake_endpoint_or_post_wake_support, "
                "high_phase_with_late_endpoint_or_pre_wake_or_post_wake_support, "
                "xml_post_scan_memory, "
                "xml_recent_scan_memory, "
                "xml_future_arrival_track_support, "
                "xml_future_arrival_track_product, "
                "xml_prearrival_heat_track_product, "
                "xml_prearrival_strong_support_product, "
                "xml_phase_strong_support_product, "
                "xml_prearrival_plus_strict_post_rescue_product, "
                "pre_arrival_track_support_blend, "
                "pre_arrival_with_track_heat_support, "
                "pre_arrival_endpoint_sweep, "
                "pre_arrival_endpoint_sweep_or_arrival."
            )
        self.laser_residual_gate_mix = str(getattr(
            config.model, "laser_residual_gate_mix", "learned",
        )).lower()
        if self.laser_residual_gate_mix not in {
                "learned", "cold_to_hot", "max_learned_cold_to_hot",
                "learned_times_cold_to_hot", "calibrated_cold_to_hot",
                "prior_only"}:
            raise ValueError(
                "laser_residual_gate_mix must be one of: learned, cold_to_hot, "
                "max_learned_cold_to_hot, learned_times_cold_to_hot, "
                "calibrated_cold_to_hot, prior_only."
            )
        self.laser_residual_prior_gate_threshold = min(
            max(float(getattr(
                config.model, "laser_residual_prior_gate_threshold", 0.0,
            )), 0.0),
            0.99,
        )
        self.laser_residual_prior_gate_power = max(
            float(getattr(config.model, "laser_residual_prior_gate_power", 1.0)),
            1.0e-6,
        )
        self.laser_residual_prior_gate_hard_threshold = bool(getattr(
            config.model,
            "laser_residual_prior_gate_hard_threshold",
            False,
        ))
        self.laser_residual_prior_support_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_prior_support_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        active_support_threshold = float(getattr(
            config.model,
            "laser_residual_active_support_threshold",
            0.0,
        ))
        if active_support_threshold <= 0.0:
            active_support_threshold = self.laser_residual_prior_support_threshold
        self.laser_residual_active_support_threshold = min(
            max(active_support_threshold, 0.0),
            0.99,
        )
        body_prearrival_threshold = float(getattr(
            config.model,
            "laser_residual_body_prearrival_threshold",
            0.10,
        ))
        self.laser_residual_body_prearrival_threshold = min(
            max(body_prearrival_threshold, 0.0),
            0.99,
        )
        post_phase_threshold = float(getattr(
            config.model,
            "laser_residual_post_phase_threshold",
            0.0,
        ))
        if post_phase_threshold <= 0.0:
            post_phase_threshold = self.laser_residual_body_prearrival_threshold
        self.laser_residual_post_phase_threshold = min(
            max(post_phase_threshold, 0.0),
            0.99,
        )
        post_support_threshold = float(getattr(
            config.model,
            "laser_residual_post_support_threshold",
            0.0,
        ))
        if post_support_threshold <= 0.0:
            post_support_threshold = self.laser_residual_prior_support_threshold
        self.laser_residual_post_support_threshold = min(
            max(post_support_threshold, 0.0),
            0.99,
        )
        pre_sweep_threshold = float(getattr(
            config.model,
            "laser_residual_pre_sweep_support_threshold",
            self.laser_residual_prior_support_threshold,
        ))
        if pre_sweep_threshold <= 0.0:
            pre_sweep_threshold = self.laser_residual_prior_support_threshold
        self.laser_residual_pre_sweep_support_threshold = min(
            max(pre_sweep_threshold, 0.0),
            0.99,
        )
        self.laser_residual_sweep_body_support_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_sweep_body_support_threshold",
                0.25,
            )), 0.0),
            0.99,
        )
        self.laser_residual_sweep_arrival_support_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_sweep_arrival_support_threshold",
                0.70,
            )), 0.0),
            0.99,
        )
        self.laser_residual_sweep_cold_to_hot_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_sweep_cold_to_hot_threshold",
                0.80,
            )), 0.0),
            0.99,
        )
        self.laser_residual_sweep_program_track_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_sweep_program_track_threshold",
                0.60,
            )), 0.0),
            0.99,
        )
        pre_endpoint_threshold = float(getattr(
            config.model,
            "laser_residual_pre_endpoint_support_threshold",
            self.laser_residual_prior_support_threshold,
        ))
        if pre_endpoint_threshold <= 0.0:
            pre_endpoint_threshold = self.laser_residual_prior_support_threshold
        self.laser_residual_pre_endpoint_support_threshold = min(
            max(pre_endpoint_threshold, 0.0),
            0.99,
        )
        self.laser_residual_endpoint_program_track_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_endpoint_program_track_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.laser_residual_endpoint_high_program_track_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_endpoint_high_program_track_threshold",
                max(self.laser_residual_endpoint_program_track_threshold, 0.75),
            )), 0.0),
            0.99,
        )
        self.laser_residual_endpoint_path_elapsed_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_endpoint_path_elapsed_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.laser_residual_time_until_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_time_until_threshold",
                0.0,
            )), 0.0),
            1.0,
        )
        wake_threshold = float(getattr(
            config.model,
            "laser_residual_wake_support_threshold",
            0.08,
        ))
        self.laser_residual_wake_support_threshold = min(
            max(wake_threshold, 0.0),
            0.99,
        )
        wake_body_threshold = float(getattr(
            config.model,
            "laser_residual_wake_body_support_threshold",
            self.laser_residual_prior_support_threshold,
        ))
        if wake_body_threshold <= 0.0:
            wake_body_threshold = self.laser_residual_prior_support_threshold
        self.laser_residual_wake_body_support_threshold = min(
            max(wake_body_threshold, 0.0),
            0.99,
        )
        wake_body_track_threshold = float(getattr(
            config.model,
            "laser_residual_wake_body_program_track_threshold",
            self.laser_residual_endpoint_program_track_threshold,
        ))
        if wake_body_track_threshold <= 0.0:
            wake_body_track_threshold = self.laser_residual_endpoint_program_track_threshold
        self.laser_residual_wake_body_program_track_threshold = min(
            max(wake_body_track_threshold, 0.0),
            0.99,
        )
        self.laser_residual_pre_track_support_floor = min(
            max(float(getattr(
                config.model,
                "laser_residual_pre_track_support_floor",
                0.35,
            )), 0.0),
            1.0,
        )
        self.laser_residual_path_body_endpoint_max = min(
            max(float(getattr(
                config.model,
                "laser_residual_path_body_endpoint_max",
                1.0,
            )), 0.0),
            1.0,
        )
        self.laser_residual_post_suppress_strength = min(
            max(float(getattr(
                config.model, "laser_residual_post_suppress_strength", 0.0,
            )), 0.0),
            1.0,
        )
        self.laser_residual_control_gate_floor = min(
            max(float(getattr(
                config.model, "laser_residual_control_gate_floor", 0.0,
            )), 0.0),
            1.0,
        )
        self.laser_residual_cold_to_hot_gate_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_cold_to_hot_gate_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.laser_residual_cold_to_hot_gate_power = max(
            float(getattr(
                config.model,
                "laser_residual_cold_to_hot_gate_power",
                1.0,
            )),
            1.0e-6,
        )
        self.laser_residual_cold_to_hot_gate_floor = min(
            max(float(getattr(
                config.model,
                "laser_residual_cold_to_hot_gate_floor",
                0.0,
            )), 0.0),
            1.0,
        )
        self.laser_residual_cold_start_threshold = max(
            0.0,
            float(getattr(config.model, "laser_residual_cold_start_threshold", 0.0)),
        )
        self.laser_residual_cold_start_softness = max(
            0.0,
            float(getattr(config.model, "laser_residual_cold_start_softness", 0.0)),
        )
        self.laser_residual_prior_bypass_cold_start = bool(getattr(
            config.model,
            "laser_residual_prior_bypass_cold_start",
            False,
        ))
        self.laser_residual_prior_bypass_cold_start_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_prior_bypass_cold_start_threshold",
                0.5,
            )), 0.0),
            0.99,
        )
        self.laser_residual_min_delta = max(
            0.0,
            float(getattr(config.model, "laser_residual_min_delta", 0.0)),
        )
        self.laser_residual_use_tiered_min_delta = bool(getattr(
            config.model,
            "laser_residual_use_tiered_min_delta",
            False,
        ))
        self.laser_residual_weak_min_delta = max(
            0.0,
            float(getattr(config.model, "laser_residual_weak_min_delta", 0.0)),
        )
        strong_support_threshold = float(getattr(
            config.model,
            "laser_residual_strong_support_threshold",
            self.laser_residual_prior_support_threshold,
        ))
        if strong_support_threshold <= 0.0:
            strong_support_threshold = self.laser_residual_prior_support_threshold
        self.laser_residual_strong_support_threshold = min(
            max(strong_support_threshold, 0.0),
            0.99,
        )
        self.laser_residual_strong_support_sources = str(getattr(
            config.model,
            "laser_residual_strong_support_sources",
            "target_heat,sweep,endpoint,neighbor",
        )).lower()
        self.laser_residual_wake_body_cold_to_hot_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_wake_body_cold_to_hot_threshold",
                0.85,
            )), 0.0),
            0.99,
        )
        self.laser_residual_wake_body_cold_program_track_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_wake_body_cold_program_track_threshold",
                0.45,
            )), 0.0),
            0.99,
        )
        endpoint_cold_threshold = float(getattr(
            config.model,
            "laser_residual_endpoint_cold_to_hot_threshold",
            self.laser_residual_cold_to_hot_gate_threshold,
        ))
        self.laser_residual_endpoint_cold_to_hot_threshold = min(
            max(endpoint_cold_threshold, 0.0),
            0.99,
        )
        post_wake_body_threshold = float(getattr(
            config.model,
            "laser_residual_post_wake_body_support_threshold",
            0.0,
        ))
        if post_wake_body_threshold <= 0.0:
            post_wake_body_threshold = self.laser_residual_wake_body_support_threshold
        self.laser_residual_post_wake_body_support_threshold = min(
            max(post_wake_body_threshold, 0.0),
            0.99,
        )
        post_wake_threshold = float(getattr(
            config.model,
            "laser_residual_post_wake_support_threshold",
            0.0,
        ))
        if post_wake_threshold <= 0.0:
            post_wake_threshold = self.laser_residual_wake_support_threshold
        self.laser_residual_post_wake_support_threshold = min(
            max(post_wake_threshold, 0.0),
            0.99,
        )
        post_program_track_threshold = float(getattr(
            config.model,
            "laser_residual_post_wake_body_program_track_threshold",
            0.0,
        ))
        if post_program_track_threshold <= 0.0:
            post_program_track_threshold = self.laser_residual_wake_body_cold_program_track_threshold
        self.laser_residual_post_wake_body_program_track_threshold = min(
            max(post_program_track_threshold, 0.0),
            0.99,
        )
        post_cold_threshold = float(getattr(
            config.model,
            "laser_residual_post_wake_body_cold_to_hot_threshold",
            0.0,
        ))
        if post_cold_threshold <= 0.0:
            post_cold_threshold = self.laser_residual_wake_body_cold_to_hot_threshold
        self.laser_residual_post_wake_body_cold_to_hot_threshold = min(
            max(post_cold_threshold, 0.0),
            0.99,
        )
        self.laser_residual_post_rescue_cap_margin = max(
            0.0,
            float(getattr(
                config.model,
                "laser_residual_post_rescue_cap_margin",
                0.0,
            )),
        )
        self.laser_residual_post_rescue_cap_threshold = min(
            max(float(getattr(
                config.model,
                "laser_residual_post_rescue_cap_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.laser_residual_hidden_dim = max(
            8, int(getattr(config.model, "laser_residual_hidden_dim", 128))
        )
        self.laser_residual_gate_bias = float(getattr(
            config.model, "laser_residual_gate_bias", -4.0,
        ))
        self.laser_residual_delta_init = max(
            0.0,
            float(getattr(config.model, "laser_residual_delta_init", 400.0)),
        )
        self.enable_cold_to_hot_head = bool(getattr(
            config.model, "enable_cold_to_hot_head", False,
        ))
        self.cold_to_hot_hidden_dim = max(
            8, int(getattr(config.model, "cold_to_hot_hidden_dim", 64))
        )
        self.cold_to_hot_gate_bias = float(getattr(
            config.model, "cold_to_hot_gate_bias", -2.0,
        ))
        self.post_endpoint_false_hot_guard_strength = max(
            0.0,
            float(getattr(
                config.model,
                "post_endpoint_false_hot_guard_strength",
                0.0,
            )),
        )
        self.post_endpoint_false_hot_guard_threshold = min(
            max(float(getattr(
                config.model,
                "post_endpoint_false_hot_guard_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.post_endpoint_false_hot_guard_power = max(
            float(getattr(
                config.model,
                "post_endpoint_false_hot_guard_power",
                1.0,
            )),
            1.0e-6,
        )
        self.post_endpoint_false_hot_guard_max = min(
            max(float(getattr(
                config.model,
                "post_endpoint_false_hot_guard_max",
                1.0,
            )), 0.0),
            1.0,
        )
        self.post_endpoint_false_hot_guard_require_no_prearrival = bool(getattr(
            config.model,
            "post_endpoint_false_hot_guard_require_no_prearrival",
            True,
        ))
        self.post_low_sweep_false_hot_guard_strength = max(
            0.0,
            float(getattr(
                config.model,
                "post_low_sweep_false_hot_guard_strength",
                0.0,
            )),
        )
        self.post_low_sweep_false_hot_guard_threshold = min(
            max(float(getattr(
                config.model,
                "post_low_sweep_false_hot_guard_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.post_low_sweep_false_hot_guard_power = max(
            float(getattr(
                config.model,
                "post_low_sweep_false_hot_guard_power",
                1.0,
            )),
            1.0e-6,
        )
        self.post_low_sweep_false_hot_guard_max = min(
            max(float(getattr(
                config.model,
                "post_low_sweep_false_hot_guard_max",
                1.0,
            )), 0.0),
            1.0,
        )
        self.post_mid_phase_false_hot_guard_strength = max(
            0.0,
            float(getattr(
                config.model,
                "post_mid_phase_false_hot_guard_strength",
                0.0,
            )),
        )
        self.post_mid_phase_false_hot_post_min = min(
            max(float(getattr(
                config.model,
                "post_mid_phase_false_hot_post_min",
                0.0,
            )), 0.0),
            0.99,
        )
        self.post_mid_phase_false_hot_post_max = min(
            max(float(getattr(
                config.model,
                "post_mid_phase_false_hot_post_max",
                1.0,
            )), self.post_mid_phase_false_hot_post_min),
            1.0,
        )
        self.post_mid_phase_false_hot_guard_threshold = min(
            max(float(getattr(
                config.model,
                "post_mid_phase_false_hot_guard_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.post_mid_phase_false_hot_guard_power = max(
            float(getattr(
                config.model,
                "post_mid_phase_false_hot_guard_power",
                1.0,
            )),
            1.0e-6,
        )
        self.post_mid_phase_false_hot_guard_max = min(
            max(float(getattr(
                config.model,
                "post_mid_phase_false_hot_guard_max",
                1.0,
            )), 0.0),
            1.0,
        )
        self.post_mid_phase_false_hot_endpoint_max = min(
            max(float(getattr(
                config.model,
                "post_mid_phase_false_hot_endpoint_max",
                0.05,
            )), 0.0),
            1.0,
        )
        self.post_endpoint_false_hot_temperature_ceiling = float(getattr(
            config.model,
            "post_endpoint_false_hot_temperature_ceiling",
            0.0,
        ))
        self.post_endpoint_false_hot_hard_ceiling = bool(getattr(
            config.model,
            "post_endpoint_false_hot_hard_ceiling",
            False,
        ))
        self.post_endpoint_false_hot_hard_ceiling_threshold = min(
            max(float(getattr(
                config.model,
                "post_endpoint_false_hot_hard_ceiling_threshold",
                0.0,
            )), 0.0),
            0.99,
        )
        self.hotspot_specialist_blend_weight = 1.0
        self.solidus_temp = float(getattr(material_props, "solidus_temp", 1604.85))
        self.include_neighbor_temp_gate = bool(getattr(
            config.data, "laser_feature_include_neighbor_temp", False
        ))
        self.use_exposure_for_process_gate = bool(getattr(
            config.data, "laser_feature_exposure_gate", True
        ))
        self.direct_laser_feature_dim = (
            max(0, self.node_feature_dim - 12)
            if self.enable_direct_laser_head else 0
        )
        self.direct_laser_summary_pairs = []
        self.process_heat_indices = []
        self.laser_target_heat_idx = None
        self.laser_body_heat_idx = None
        self.laser_sweep_heat_idx = None
        self.laser_arrival_gate_idx = None
        self.laser_arrival_gate_indices = []
        self.laser_path_active_gate_idx = None
        self.laser_path_pre_arrival_gate_idx = None
        self.laser_path_post_arrival_gate_idx = None
        self.laser_path_body_heat_idx = None
        self.laser_path_body_gate_idx = None
        self.laser_endpoint_gate_idx = None
        self.laser_endpoint_gate_indices = []
        self.laser_path_wake_gate_idx = None
        self.laser_path_pre_wake_gate_idx = None
        self.laser_path_post_wake_gate_idx = None
        self.laser_path_program_track_idx = None
        self.laser_path_elapsed_idx = None
        self.laser_path_scanned_idx = None
        self.laser_path_time_since_idx = None
        self.laser_path_time_until_idx = None
        self.laser_path_cooling_tail_idx = None
        self.neighbor_temp_start_idx = None
        self.neighbor_hot_stats_start_idx = None
        self.neighbor_warm_stats_start_idx = None
        feature_offset = 0
        if self.direct_laser_feature_dim > 0 and getattr(
            config.data, "use_target_laser_features", False
        ):
            if self.direct_laser_feature_dim >= 6:
                self.direct_laser_summary_pairs.append((3, 5))
                self.process_heat_indices.append(5)
                self.laser_target_heat_idx = 5
            feature_offset = 9

            if getattr(config.data, "laser_feature_include_body_source", False):
                if self.direct_laser_feature_dim >= feature_offset + 5:
                    body_heat_idx = feature_offset
                    body_gate_idx = feature_offset + 1
                    body_cross_norm_idx = feature_offset + 3
                    self.laser_body_heat_idx = body_heat_idx
                    self.direct_laser_summary_pairs.append(
                        (body_cross_norm_idx, body_heat_idx)
                    )
                    self.process_heat_indices.append(body_heat_idx)
                    self.process_heat_indices.append(body_gate_idx)
                feature_offset += 5

            if getattr(config.data, "laser_feature_include_sweep", False):
                if self.direct_laser_feature_dim >= feature_offset + 3:
                    self.direct_laser_summary_pairs.append(
                        (feature_offset, feature_offset + 2)
                    )
                    self.process_heat_indices.append(feature_offset + 2)
                    self.laser_sweep_heat_idx = feature_offset + 2
                feature_offset += 4

            if getattr(config.data, "laser_feature_include_exposure", False):
                exposure_feature_dim = 8
                body_exposure_start = None
                if getattr(config.data, "laser_feature_include_body_source", False):
                    body_exposure_start = feature_offset + exposure_feature_dim
                    exposure_feature_dim += 3
                split_start = feature_offset + exposure_feature_dim
                if getattr(config.data, "laser_feature_include_exposure_split", False):
                    exposure_feature_dim += 3
                arrival_start = feature_offset + exposure_feature_dim
                if getattr(config.data, "laser_feature_include_arrival_time", False):
                    exposure_feature_dim += 8
                if self.direct_laser_feature_dim >= feature_offset + 7:
                    self.direct_laser_summary_pairs.append(
                        (feature_offset + 3, feature_offset + 5)
                    )
                    if self.use_exposure_for_process_gate:
                        exposure_gate_mode = str(getattr(
                            config.data,
                            "laser_feature_exposure_gate_mode",
                            "integral",
                        )).lower()
                        if exposure_gate_mode in {"heat", "best_heat", "both"}:
                            self.process_heat_indices.append(feature_offset + 5)
                        if exposure_gate_mode in {"integral", "decayed", "both"}:
                            self.process_heat_indices.append(feature_offset + 6)
                        if body_exposure_start is not None:
                            if exposure_gate_mode in {
                                "body",
                                "body_heat",
                                "body_integral",
                                "integral",
                                "decayed",
                                "both",
                            }:
                                self.process_heat_indices.append(body_exposure_start)
                                self.process_heat_indices.append(body_exposure_start + 1)
                            self.direct_laser_summary_pairs.append(
                                (feature_offset + 3, body_exposure_start)
                            )
                        if getattr(config.data, "laser_feature_include_exposure_split", False):
                            if exposure_gate_mode in {"integral", "decayed", "both", "split"}:
                                self.process_heat_indices.append(split_start)
                                self.process_heat_indices.append(split_start + 1)
                        if getattr(config.data, "laser_feature_include_arrival_time", False):
                            self.direct_laser_summary_pairs.append(
                                (arrival_start + 4, arrival_start + 7)
                            )
                            self.laser_arrival_gate_idx = arrival_start + 7
                            self.laser_arrival_gate_indices.append(arrival_start + 7)
                            if exposure_gate_mode in {"arrival", "arrival_time", "both"}:
                                self.process_heat_indices.append(arrival_start + 7)
                feature_offset += exposure_feature_dim

            if getattr(config.data, "laser_feature_include_path_arrival", False):
                path_arrival_start = feature_offset
                if self.direct_laser_feature_dim >= path_arrival_start + 8:
                    self.direct_laser_summary_pairs.append(
                        (path_arrival_start + 4, path_arrival_start + 7)
                    )
                    self.laser_arrival_gate_indices.append(path_arrival_start + 7)
                    self.laser_arrival_gate_idx = path_arrival_start + 7
                    exposure_gate_mode = str(getattr(
                        config.data,
                        "laser_feature_exposure_gate_mode",
                        "integral",
                    )).lower()
                    if exposure_gate_mode in {
                        "arrival", "arrival_time", "path_arrival", "both"
                    }:
                        self.process_heat_indices.append(path_arrival_start + 7)
                feature_offset += 8

            if getattr(config.data, "laser_feature_include_path_phase", False):
                path_phase_start = feature_offset
                if (
                        self.direct_laser_feature_dim >= path_phase_start + 10
                        and getattr(config.data, "laser_feature_path_phase_use_as_gate", False)):
                    active_gate_idx = path_phase_start + 7
                    pre_gate_idx = path_phase_start + 8
                    post_gate_idx = path_phase_start + 9
                    self.laser_path_active_gate_idx = active_gate_idx
                    self.laser_path_pre_arrival_gate_idx = pre_gate_idx
                    self.laser_path_post_arrival_gate_idx = post_gate_idx
                    self.direct_laser_summary_pairs.append(
                        (path_phase_start + 5, active_gate_idx)
                    )
                    self.process_heat_indices.extend([
                        active_gate_idx,
                        pre_gate_idx,
                        post_gate_idx,
                    ])
                    self.laser_arrival_gate_indices.extend([
                        pre_gate_idx,
                        post_gate_idx,
                    ])
                    self.laser_arrival_gate_idx = pre_gate_idx
                feature_offset += 10

            if getattr(config.data, "laser_feature_include_path_coordinates", False):
                path_coordinate_start = feature_offset
                if self.direct_laser_feature_dim >= path_coordinate_start + 12:
                    program_track_idx = path_coordinate_start + 2
                    cross_idx = path_coordinate_start + 5
                    start_gate_idx = path_coordinate_start + 8
                    end_gate_idx = path_coordinate_start + 9
                    arrival_dt_idx = path_coordinate_start + 10
                    path_elapsed_idx = path_coordinate_start + 11
                    self.laser_path_program_track_idx = program_track_idx
                    self.laser_path_elapsed_idx = path_elapsed_idx
                    self.direct_laser_summary_pairs.extend([
                        (cross_idx, start_gate_idx),
                        (arrival_dt_idx, end_gate_idx),
                    ])
                    if getattr(config.data, "laser_feature_include_path_timing", False):
                        scanned_idx = path_coordinate_start + 12
                        time_since_idx = path_coordinate_start + 13
                        time_until_idx = path_coordinate_start + 14
                        cooling_tail_idx = path_coordinate_start + 15
                        if self.direct_laser_feature_dim >= path_coordinate_start + 16:
                            self.laser_path_scanned_idx = scanned_idx
                            self.laser_path_time_since_idx = time_since_idx
                            self.laser_path_time_until_idx = time_until_idx
                            self.laser_path_cooling_tail_idx = cooling_tail_idx
                            self.direct_laser_summary_pairs.append(
                                (time_since_idx, cooling_tail_idx)
                            )
                feature_offset += 16 if getattr(
                    config.data, "laser_feature_include_path_timing", False
                ) else 12

            if getattr(config.data, "laser_feature_include_path_body_support", False):
                path_body_start = feature_offset
                if self.direct_laser_feature_dim >= path_body_start + 5:
                    body_heat_idx = path_body_start
                    body_gate_idx = path_body_start + 1
                    body_cross_norm_idx = path_body_start + 3
                    self.laser_path_body_heat_idx = body_heat_idx
                    self.laser_path_body_gate_idx = body_gate_idx
                    self.direct_laser_summary_pairs.append(
                        (body_cross_norm_idx, body_heat_idx)
                    )
                    if getattr(
                            config.data,
                            "laser_feature_path_body_use_as_gate",
                            False,
                    ):
                        self.process_heat_indices.extend([
                            body_heat_idx,
                            body_gate_idx,
                        ])
                feature_offset += 5

            if getattr(config.data, "laser_feature_include_path_endpoint", False):
                path_endpoint_start = feature_offset
                if self.direct_laser_feature_dim >= path_endpoint_start + 6:
                    endpoint_gate_idx = path_endpoint_start + 3
                    start_gate_idx = path_endpoint_start + 4
                    end_gate_idx = path_endpoint_start + 5
                    self.direct_laser_summary_pairs.append(
                        (path_endpoint_start, endpoint_gate_idx)
                    )
                    self.laser_endpoint_gate_idx = endpoint_gate_idx
                    self.laser_endpoint_gate_indices.extend([
                        endpoint_gate_idx,
                        start_gate_idx,
                        end_gate_idx,
                    ])
                    if getattr(
                            config.data,
                            "laser_feature_path_endpoint_use_as_gate",
                            False,
                    ):
                        self.process_heat_indices.extend([
                            endpoint_gate_idx,
                            start_gate_idx,
                            end_gate_idx,
                        ])
                feature_offset += 6

            if getattr(config.data, "laser_feature_include_path_wake", False):
                path_wake_start = feature_offset
                if self.direct_laser_feature_dim >= path_wake_start + 6:
                    wake_gate_idx = path_wake_start
                    pre_wake_gate_idx = path_wake_start + 1
                    post_wake_gate_idx = path_wake_start + 2
                    cross_gate_idx = path_wake_start + 4
                    temporal_gate_idx = path_wake_start + 5
                    self.laser_path_wake_gate_idx = wake_gate_idx
                    self.laser_path_pre_wake_gate_idx = pre_wake_gate_idx
                    self.laser_path_post_wake_gate_idx = post_wake_gate_idx
                    self.direct_laser_summary_pairs.extend([
                        (cross_gate_idx, wake_gate_idx),
                        (temporal_gate_idx, post_wake_gate_idx),
                    ])
                    self.process_heat_indices.extend([
                        wake_gate_idx,
                        pre_wake_gate_idx,
                        post_wake_gate_idx,
                    ])
                feature_offset += 6

            if getattr(config.data, "laser_feature_include_neighbor_temp", False):
                if self.direct_laser_feature_dim >= feature_offset + 3:
                    self.neighbor_temp_start_idx = feature_offset
                feature_offset += 3

            if getattr(config.data, "laser_feature_include_neighbor_hot_stats", False):
                if self.direct_laser_feature_dim >= feature_offset + 2:
                    self.neighbor_hot_stats_start_idx = feature_offset
                feature_offset += 2

            if getattr(config.data, "laser_feature_include_neighbor_warm_stats", False):
                if self.direct_laser_feature_dim >= feature_offset + 2:
                    self.neighbor_warm_stats_start_idx = feature_offset
                feature_offset += 2

        self.direct_laser_summary_dim = 2 * len(self.direct_laser_summary_pairs)

        self.spatial_encoder = SpatialEncoder(
            in_dim=self.node_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=config.model.spatial.num_layers,
            heads=config.model.spatial.heads,
            edge_dim=self.edge_feature_dim,
            dropout=config.model.spatial.dropout,
            use_checkpoint=getattr(config.model.spatial, "use_checkpoint", False),
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

        direct_head_dim = (
            self.direct_laser_feature_dim + self.direct_laser_summary_dim
            if self.direct_laser_features_to_head else 0
        )
        head_in_dim = self.spatial_encoder.out_dim + direct_head_dim

        self.temperature_head = TemperatureHead(hidden_dim=head_in_dim)

        self.hotspot_head = None
        if self.enable_hotspot_head:
            self.hotspot_head = nn.Sequential(
                nn.Linear(head_in_dim, 64),
                nn.GELU(),
                nn.Dropout(config.model.spatial.dropout),
                nn.Linear(64, 1),
            )

        self.cold_to_hot_head = None
        if self.enable_cold_to_hot_head:
            self.cold_to_hot_head = nn.Sequential(
                nn.Linear(head_in_dim, self.cold_to_hot_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.model.spatial.dropout),
                nn.Linear(self.cold_to_hot_hidden_dim, 1),
            )
            nn.init.constant_(self.cold_to_hot_head[-1].bias, self.cold_to_hot_gate_bias)

        self.hotspot_delta_head = None
        if self.enable_hotspot_delta_head:
            self.hotspot_delta_head = nn.Sequential(
                nn.Linear(head_in_dim, 64),
                nn.GELU(),
                nn.Dropout(config.model.spatial.dropout),
                nn.Linear(64, 1),
                nn.Softplus(),
            )

        self.hotspot_specialist_head = None
        if self.enable_hotspot_specialist_head:
            specialist_layers = []
            in_dim = head_in_dim
            for _ in range(self.hotspot_specialist_num_layers):
                specialist_layers.extend([
                    nn.Linear(in_dim, self.hotspot_specialist_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.model.spatial.dropout),
                ])
                in_dim = self.hotspot_specialist_hidden_dim
            specialist_layers.append(nn.Linear(in_dim, 1))
            self.hotspot_specialist_head = nn.Sequential(*specialist_layers)
            # Default remains near ambient; peak-focused ablations can start
            # the specialist near solidus so rare melt-pool gradients bite early.
            init_temp = float(getattr(config.model, "hotspot_specialist_init_temp", 0.0))
            if init_temp <= 0.0:
                nn.init.constant_(self.hotspot_specialist_head[-1].bias, -4.0)
            else:
                if self.hotspot_specialist_output_mode == "residual":
                    init_ratio = (
                        (init_temp - self.hotspot_specialist_min_temp)
                        / self.hotspot_specialist_residual_max_delta
                    )
                else:
                    init_temp_span = (
                        self.hotspot_specialist_max_temp
                        - self.hotspot_specialist_min_temp
                    )
                    init_ratio = (
                        (init_temp - self.hotspot_specialist_min_temp)
                        / max(init_temp_span, 1.0e-6)
                    )
                init_ratio = min(max(init_ratio, 1.0e-6), 1.0 - 1.0e-6)
                init_bias = torch.logit(torch.as_tensor(init_ratio)).item()
                nn.init.constant_(self.hotspot_specialist_head[-1].bias, init_bias)

        self.laser_residual_head = None
        if self.enable_laser_residual_head and self.laser_residual_max_delta > 0:
            self.laser_residual_head = nn.Sequential(
                nn.Linear(head_in_dim, self.laser_residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.model.spatial.dropout),
                nn.Linear(self.laser_residual_hidden_dim, 2),
            )
            nn.init.constant_(self.laser_residual_head[-1].bias[0], self.laser_residual_gate_bias)
            init_ratio = min(
                max(self.laser_residual_delta_init / self.laser_residual_max_delta, 1.0e-6),
                1.0 - 1.0e-6,
            )
            nn.init.constant_(
                self.laser_residual_head[-1].bias[1],
                torch.logit(torch.as_tensor(init_ratio)).item(),
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
            spatial_features.append(z_s.detach() if not self.training else z_s)
            mask = getattr(data, "mask", torch.ones(z_s.shape[0], dtype=torch.bool,
                                                     device=z_s.device))
            masks.append(mask)

        z_s = torch.stack([f for f in spatial_features], dim=0).unsqueeze(0)
        mask_stack = torch.stack(masks, dim=0).unsqueeze(0)

        z_t = self.temporal_encoder(z_s, mask_stack)

        z_s_last = z_s[:, -1]
        mask_last = mask_stack[:, -1]

        z_f = self.cross_fusion(z_s_last, z_t, mask_stack)

        z_head = z_f
        process_gate = None
        laser_target_heat_gate = None
        laser_body_heat_gate = None
        laser_sweep_heat_gate = None
        neighbor_hot_gate = None
        laser_arrival_gate = None
        laser_path_active_gate = None
        laser_path_pre_arrival_gate = None
        laser_path_post_arrival_gate = None
        laser_path_body_heat_gate = None
        laser_endpoint_gate = None
        laser_path_wake_gate = None
        laser_path_pre_wake_gate = None
        laser_path_post_wake_gate = None
        laser_path_program_track_gate = None
        laser_path_elapsed_gate = None
        laser_path_scanned_gate = None
        laser_path_time_until_gate = None
        laser_path_cooling_tail_gate = None
        if self.direct_laser_feature_dim > 0:
            direct_seq = torch.stack([
                data.x[:, -self.direct_laser_feature_dim:].to(
                    device=z_f.device, dtype=z_f.dtype
                )
                for data in graph_sequence
            ], dim=0)
            x_last = direct_seq[-1].unsqueeze(0).expand(z_f.shape[0], -1, -1)

            summary_features = []
            for dxy_idx, heat_idx in self.direct_laser_summary_pairs:
                if heat_idx < self.direct_laser_feature_dim:
                    min_dxy = direct_seq[:, :, dxy_idx].min(dim=0).values
                    max_heat = direct_seq[:, :, heat_idx].max(dim=0).values
                    summary_features.extend([min_dxy, max_heat])

            if summary_features:
                x_summary = torch.stack(summary_features, dim=-1)
                x_summary = x_summary.unsqueeze(0).expand(z_f.shape[0], -1, -1)
                x_last = torch.cat([x_last, x_summary], dim=-1)

            heat_features = []
            for heat_idx in self.process_heat_indices:
                if heat_idx < self.direct_laser_feature_dim:
                    heat_features.append(direct_seq[:, :, heat_idx].max(dim=0).values)
            if heat_features:
                process_heat = torch.stack(heat_features, dim=-1).max(dim=-1).values
                process_gate = process_heat.unsqueeze(0).unsqueeze(-1)
                process_gate = process_gate.expand(z_f.shape[0], -1, -1).clamp(0.0, 1.0)

            if (
                    self.laser_target_heat_idx is not None
                    and self.laser_target_heat_idx < self.direct_laser_feature_dim):
                target_heat = direct_seq[:, :, self.laser_target_heat_idx].max(dim=0).values
                laser_target_heat_gate = target_heat.unsqueeze(0).unsqueeze(-1)
                laser_target_heat_gate = laser_target_heat_gate.expand(z_f.shape[0], -1, -1).clamp(0.0, 1.0)

            if (
                    self.laser_body_heat_idx is not None
                    and self.laser_body_heat_idx < self.direct_laser_feature_dim):
                body_heat = direct_seq[:, :, self.laser_body_heat_idx].max(dim=0).values
                laser_body_heat_gate = body_heat.unsqueeze(0).unsqueeze(-1)
                laser_body_heat_gate = laser_body_heat_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_sweep_heat_idx is not None
                    and self.laser_sweep_heat_idx < self.direct_laser_feature_dim):
                sweep_heat = direct_seq[:, :, self.laser_sweep_heat_idx].max(dim=0).values
                laser_sweep_heat_gate = sweep_heat.unsqueeze(0).unsqueeze(-1)
                laser_sweep_heat_gate = laser_sweep_heat_gate.expand(z_f.shape[0], -1, -1).clamp(0.0, 1.0)

            arrival_gate_indices = [
                idx for idx in self.laser_arrival_gate_indices
                if idx < self.direct_laser_feature_dim
            ]
            if arrival_gate_indices:
                arrival_heat = torch.stack([
                    direct_seq[:, :, idx].max(dim=0).values
                    for idx in arrival_gate_indices
                ], dim=-1).max(dim=-1).values
                laser_arrival_gate = arrival_heat.unsqueeze(0).unsqueeze(-1)
                laser_arrival_gate = laser_arrival_gate.expand(z_f.shape[0], -1, -1).clamp(0.0, 1.0)

            if (
                    self.laser_path_active_gate_idx is not None
                    and self.laser_path_active_gate_idx < self.direct_laser_feature_dim):
                active_heat = direct_seq[:, :, self.laser_path_active_gate_idx].max(dim=0).values
                laser_path_active_gate = active_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_active_gate = laser_path_active_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_pre_arrival_gate_idx is not None
                    and self.laser_path_pre_arrival_gate_idx < self.direct_laser_feature_dim):
                pre_arrival_heat = direct_seq[:, :, self.laser_path_pre_arrival_gate_idx].max(dim=0).values
                laser_path_pre_arrival_gate = pre_arrival_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_pre_arrival_gate = laser_path_pre_arrival_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_post_arrival_gate_idx is not None
                    and self.laser_path_post_arrival_gate_idx < self.direct_laser_feature_dim):
                post_arrival_heat = direct_seq[:, :, self.laser_path_post_arrival_gate_idx].max(dim=0).values
                laser_path_post_arrival_gate = post_arrival_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_post_arrival_gate = laser_path_post_arrival_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_body_heat_idx is not None
                    and self.laser_path_body_heat_idx < self.direct_laser_feature_dim):
                path_body_heat = direct_seq[:, :, self.laser_path_body_heat_idx].max(dim=0).values
                laser_path_body_heat_gate = path_body_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_body_heat_gate = laser_path_body_heat_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_wake_gate_idx is not None
                    and self.laser_path_wake_gate_idx < self.direct_laser_feature_dim):
                wake_heat = direct_seq[:, :, self.laser_path_wake_gate_idx].max(dim=0).values
                laser_path_wake_gate = wake_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_wake_gate = laser_path_wake_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_pre_wake_gate_idx is not None
                    and self.laser_path_pre_wake_gate_idx < self.direct_laser_feature_dim):
                pre_wake_heat = direct_seq[:, :, self.laser_path_pre_wake_gate_idx].max(dim=0).values
                laser_path_pre_wake_gate = pre_wake_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_pre_wake_gate = laser_path_pre_wake_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_post_wake_gate_idx is not None
                    and self.laser_path_post_wake_gate_idx < self.direct_laser_feature_dim):
                post_wake_heat = direct_seq[:, :, self.laser_path_post_wake_gate_idx].max(dim=0).values
                laser_path_post_wake_gate = post_wake_heat.unsqueeze(0).unsqueeze(-1)
                laser_path_post_wake_gate = laser_path_post_wake_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            endpoint_gate_indices = [
                idx for idx in self.laser_endpoint_gate_indices
                if idx < self.direct_laser_feature_dim
            ]
            if endpoint_gate_indices:
                endpoint_heat = torch.stack([
                    direct_seq[:, :, idx].max(dim=0).values
                    for idx in endpoint_gate_indices
                ], dim=-1).max(dim=-1).values
                laser_endpoint_gate = endpoint_heat.unsqueeze(0).unsqueeze(-1)
                laser_endpoint_gate = laser_endpoint_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_program_track_idx is not None
                    and self.laser_path_program_track_idx < self.direct_laser_feature_dim):
                program_track_signed = direct_seq[-1, :, self.laser_path_program_track_idx]
                program_track_norm = (program_track_signed + 1.0) * 0.5
                laser_path_program_track_gate = program_track_norm.unsqueeze(0).unsqueeze(-1)
                laser_path_program_track_gate = laser_path_program_track_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_elapsed_idx is not None
                    and self.laser_path_elapsed_idx < self.direct_laser_feature_dim):
                path_elapsed = direct_seq[-1, :, self.laser_path_elapsed_idx]
                laser_path_elapsed_gate = path_elapsed.unsqueeze(0).unsqueeze(-1)
                laser_path_elapsed_gate = laser_path_elapsed_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_scanned_idx is not None
                    and self.laser_path_scanned_idx < self.direct_laser_feature_dim):
                scanned = direct_seq[-1, :, self.laser_path_scanned_idx]
                laser_path_scanned_gate = scanned.unsqueeze(0).unsqueeze(-1)
                laser_path_scanned_gate = laser_path_scanned_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_time_until_idx is not None
                    and self.laser_path_time_until_idx < self.direct_laser_feature_dim):
                time_until = direct_seq[-1, :, self.laser_path_time_until_idx]
                laser_path_time_until_gate = time_until.unsqueeze(0).unsqueeze(-1)
                laser_path_time_until_gate = laser_path_time_until_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if (
                    self.laser_path_cooling_tail_idx is not None
                    and self.laser_path_cooling_tail_idx < self.direct_laser_feature_dim):
                cooling_tail = direct_seq[-1, :, self.laser_path_cooling_tail_idx]
                laser_path_cooling_tail_gate = cooling_tail.unsqueeze(0).unsqueeze(-1)
                laser_path_cooling_tail_gate = laser_path_cooling_tail_gate.expand(
                    z_f.shape[0], -1, -1
                ).clamp(0.0, 1.0)

            if self.include_neighbor_temp_gate and self.neighbor_temp_start_idx is not None:
                neigh_max_norm = direct_seq[:, :, self.neighbor_temp_start_idx].max(dim=0).values
                solidus_norm = max(self.solidus_temp / 1000.0, 1.001)
                neighbor_hot_gate = (
                    (neigh_max_norm - 1.0) / (solidus_norm - 1.0)
                ).clamp(0.0, 1.0)
                neighbor_hot_gate = neighbor_hot_gate.unsqueeze(0).unsqueeze(-1)
                neighbor_hot_gate = neighbor_hot_gate.expand(z_f.shape[0], -1, -1)

            if self.neighbor_hot_stats_start_idx is not None:
                hot_fraction = direct_seq[:, :, self.neighbor_hot_stats_start_idx].max(dim=0).values
                hot_excess = direct_seq[:, :, self.neighbor_hot_stats_start_idx + 1].max(dim=0).values
                stats_gate = torch.maximum(hot_fraction, hot_excess.clamp(0.0, 1.0))
                stats_gate = stats_gate.unsqueeze(0).unsqueeze(-1)
                stats_gate = stats_gate.expand(z_f.shape[0], -1, -1)
                if neighbor_hot_gate is None:
                    neighbor_hot_gate = stats_gate
                else:
                    neighbor_hot_gate = torch.maximum(neighbor_hot_gate, stats_gate)

            if self.neighbor_warm_stats_start_idx is not None:
                warm_fraction = direct_seq[:, :, self.neighbor_warm_stats_start_idx].max(dim=0).values
                warm_excess = direct_seq[:, :, self.neighbor_warm_stats_start_idx + 1].max(dim=0).values
                warm_gate = torch.maximum(warm_fraction, warm_excess.clamp(0.0, 1.0))
                warm_gate = warm_gate.unsqueeze(0).unsqueeze(-1)
                warm_gate = warm_gate.expand(z_f.shape[0], -1, -1)
                if neighbor_hot_gate is None:
                    neighbor_hot_gate = warm_gate
                else:
                    neighbor_hot_gate = torch.maximum(neighbor_hot_gate, warm_gate)

            if self.direct_laser_features_to_head:
                z_head = torch.cat([z_f, x_last], dim=-1)

        T_head = self.temperature_head(z_head, mask_last)
        hotspot_logit = None
        if self.hotspot_head is not None:
            hotspot_logit = self.hotspot_head(z_head)
            if mask_last is not None:
                hotspot_logit = hotspot_logit * mask_last.unsqueeze(-1).to(hotspot_logit.dtype)

        cold_to_hot_logit = None
        cold_to_hot_gate = None
        post_endpoint_false_hot_guard = None
        if self.cold_to_hot_head is not None:
            cold_to_hot_logit = self.cold_to_hot_head(z_head)
            cold_to_hot_gate = torch.sigmoid(cold_to_hot_logit)
            if mask_last is not None:
                mask_value = mask_last.unsqueeze(-1).to(cold_to_hot_logit.dtype)
                cold_to_hot_logit = cold_to_hot_logit * mask_value
                cold_to_hot_gate = cold_to_hot_gate * mask_value.to(cold_to_hot_gate.dtype)

        post_endpoint_false_hot_guard = self._post_endpoint_false_hot_guard(
            laser_path_post_arrival_gate=laser_path_post_arrival_gate,
            laser_endpoint_gate=laser_endpoint_gate,
            laser_sweep_heat_gate=laser_sweep_heat_gate,
            laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
            dtype=T_head.dtype,
        )
        if post_endpoint_false_hot_guard is not None:
            if mask_last is not None:
                post_endpoint_false_hot_guard = (
                    post_endpoint_false_hot_guard
                    * mask_last.unsqueeze(-1).to(post_endpoint_false_hot_guard.dtype)
                )
            if cold_to_hot_gate is not None:
                cold_to_hot_gate = cold_to_hot_gate * (1.0 - post_endpoint_false_hot_guard)

        hotspot_delta = None
        if self.hotspot_delta_head is not None:
            hotspot_delta = self.hotspot_delta_head(z_head)
            if hotspot_logit is not None:
                cls_gate = torch.sigmoid(hotspot_logit)
                gate = cls_gate if process_gate is None else torch.maximum(cls_gate, process_gate)
            else:
                gate = process_gate
            if gate is not None:
                hotspot_delta = hotspot_delta * gate.to(dtype=hotspot_delta.dtype)
            if mask_last is not None:
                hotspot_delta = hotspot_delta * mask_last.unsqueeze(-1).to(hotspot_delta.dtype)
            T_head = T_head + hotspot_delta

        if self.predict_temperature_delta:
            T_last = graph_sequence[-1].y.to(device=T_head.device, dtype=T_head.dtype)
            if T_last.ndim == 2:
                T_last = T_last.unsqueeze(0)
            T_pred = T_last + T_head
        else:
            T_pred = T_head

        T_base = T_pred
        hotspot_specialist_temp = None
        hotspot_specialist_gate = None
        hotspot_laser_prior_temp = None
        hotspot_laser_prior_gate = None
        laser_residual_prior_gate = None
        laser_residual_learned_gate = None
        laser_residual_control_gate = None
        laser_residual_cold_start_gate = None
        laser_residual_gate = None
        laser_residual_delta = None
        laser_residual_boost = None
        laser_residual_post_rescue_cap_gate = None
        laser_residual_prior_gate = self._select_laser_residual_prior_gate(
            process_gate=process_gate,
            laser_arrival_gate=laser_arrival_gate,
            laser_path_active_gate=laser_path_active_gate,
            laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
            laser_path_post_arrival_gate=laser_path_post_arrival_gate,
            laser_endpoint_gate=laser_endpoint_gate,
            laser_target_heat_gate=laser_target_heat_gate,
            laser_body_heat_gate=laser_body_heat_gate,
            laser_path_body_heat_gate=laser_path_body_heat_gate,
            laser_path_wake_gate=laser_path_wake_gate,
            laser_path_pre_wake_gate=laser_path_pre_wake_gate,
            laser_path_post_wake_gate=laser_path_post_wake_gate,
            laser_path_program_track_gate=laser_path_program_track_gate,
            laser_path_elapsed_gate=laser_path_elapsed_gate,
            laser_path_scanned_gate=laser_path_scanned_gate,
            laser_path_time_until_gate=laser_path_time_until_gate,
            laser_path_cooling_tail_gate=laser_path_cooling_tail_gate,
            laser_sweep_heat_gate=laser_sweep_heat_gate,
            neighbor_hot_gate=neighbor_hot_gate,
            cold_to_hot_gate=cold_to_hot_gate,
            dtype=T_pred.dtype,
        )
        if self.hotspot_specialist_head is not None:
            raw_temp = self.hotspot_specialist_head(z_head)
            if self.hotspot_specialist_output_mode == "residual":
                base_for_specialist = (
                    T_base.detach() if self.hotspot_specialist_detach_base else T_base
                )
                specialist_delta = (
                    self.hotspot_specialist_residual_max_delta
                    * torch.sigmoid(raw_temp)
                )
                hotspot_specialist_temp = (base_for_specialist + specialist_delta).clamp(
                    min=self.hotspot_specialist_min_temp,
                    max=self.hotspot_specialist_max_temp,
                )
            else:
                temp_span = self.hotspot_specialist_max_temp - self.hotspot_specialist_min_temp
                hotspot_specialist_temp = (
                    self.hotspot_specialist_min_temp + temp_span * torch.sigmoid(raw_temp)
                )

            cls_gate_for_specialist = None
            process_gate_for_specialist = None
            raw_process_gate_for_specialist = None
            neighbor_gate_for_specialist = None
            raw_neighbor_gate_for_specialist = None
            gates = []
            if hotspot_logit is not None and self.hotspot_specialist_use_cls_gate:
                cls_prob = torch.sigmoid(hotspot_logit)
                denom = max(1.0 - self.hotspot_specialist_gate_threshold, 1.0e-6)
                cls_gate_for_specialist = (
                    (cls_prob - self.hotspot_specialist_gate_threshold) / denom
                ).clamp(0.0, 1.0)
            if process_gate is not None:
                raw_process_gate = process_gate.to(dtype=T_pred.dtype)
                raw_process_gate_for_specialist = raw_process_gate
                process_gate_for_specialist = raw_process_gate
                if self.hotspot_specialist_process_gate_threshold > 0:
                    if self.hotspot_specialist_hard_process_gate:
                        process_gate_for_specialist = (
                            raw_process_gate >= self.hotspot_specialist_process_gate_threshold
                        ).to(dtype=T_pred.dtype)
                    else:
                        denom = max(
                            1.0 - self.hotspot_specialist_process_gate_threshold,
                            1.0e-6,
                        )
                        process_gate_for_specialist = (
                            (raw_process_gate - self.hotspot_specialist_process_gate_threshold)
                            / denom
                        ).clamp(0.0, 1.0)
                if abs(self.hotspot_specialist_process_gate_power - 1.0) > 1.0e-6:
                    process_gate_for_specialist = self._apply_gate_power(
                        process_gate_for_specialist,
                        self.hotspot_specialist_process_gate_power,
                    )
                if self.hotspot_specialist_process_gate_max < 1.0:
                    process_gate_for_specialist = process_gate_for_specialist.clamp(
                        max=self.hotspot_specialist_process_gate_max
                    )

            if neighbor_hot_gate is not None:
                raw_neighbor_gate_for_specialist = neighbor_hot_gate.to(dtype=T_pred.dtype)
                neighbor_gate_for_specialist = raw_neighbor_gate_for_specialist
                if self.hotspot_specialist_neighbor_gate_threshold > 0:
                    denom = max(
                        1.0 - self.hotspot_specialist_neighbor_gate_threshold,
                        1.0e-6,
                    )
                    neighbor_gate_for_specialist = (
                        (neighbor_gate_for_specialist - self.hotspot_specialist_neighbor_gate_threshold)
                        / denom
                    ).clamp(0.0, 1.0)
                if abs(self.hotspot_specialist_neighbor_gate_power - 1.0) > 1.0e-6:
                    neighbor_gate_for_specialist = self._apply_gate_power(
                        neighbor_gate_for_specialist,
                        self.hotspot_specialist_neighbor_gate_power,
                    )
                if self.hotspot_specialist_neighbor_gate_max < 1.0:
                    neighbor_gate_for_specialist = neighbor_gate_for_specialist.clamp(
                        max=self.hotspot_specialist_neighbor_gate_max
                    )

            if self.enable_hotspot_laser_prior and self.hotspot_laser_prior_max_delta > 0:
                if self.hotspot_laser_prior_gate_source == "residual_prior":
                    hotspot_laser_prior_gate = (
                        laser_residual_prior_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                        if laser_residual_prior_gate is not None else None
                    )
                elif self.hotspot_laser_prior_gate_source == "arrival":
                    hotspot_laser_prior_gate = (
                        laser_arrival_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                        if laser_arrival_gate is not None else None
                    )
                elif self.hotspot_laser_prior_gate_source == "arrival_or_process":
                    gates_for_prior = []
                    if laser_arrival_gate is not None:
                        gates_for_prior.append(laser_arrival_gate.to(dtype=T_pred.dtype))
                    if process_gate_for_specialist is not None:
                        gates_for_prior.append(process_gate_for_specialist)
                    if gates_for_prior:
                        hotspot_laser_prior_gate = gates_for_prior[0].clamp(0.0, 1.0)
                        for gate in gates_for_prior[1:]:
                            hotspot_laser_prior_gate = torch.maximum(
                                hotspot_laser_prior_gate,
                                gate.clamp(0.0, 1.0),
                            )
                elif self.hotspot_laser_prior_gate_source == "arrival_process_product":
                    if laser_arrival_gate is not None and process_gate_for_specialist is not None:
                        hotspot_laser_prior_gate = (
                            laser_arrival_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                            * process_gate_for_specialist.clamp(0.0, 1.0)
                        )
                elif self.hotspot_laser_prior_gate_source == "raw_process":
                    if raw_process_gate_for_specialist is not None:
                        hotspot_laser_prior_gate = raw_process_gate_for_specialist.clamp(0.0, 1.0)
                elif self.hotspot_laser_prior_gate_source == "raw_process_or_neighbor_supported":
                    if raw_process_gate_for_specialist is not None:
                        raw_process = raw_process_gate_for_specialist.clamp(0.0, 1.0)
                        high_process = raw_process >= self.hotspot_laser_prior_raw_process_threshold
                        if raw_neighbor_gate_for_specialist is not None:
                            neighbor_supported = (
                                raw_neighbor_gate_for_specialist.clamp(0.0, 1.0)
                                >= self.hotspot_laser_prior_neighbor_threshold
                            )
                            high_process = high_process | neighbor_supported
                        hotspot_laser_prior_gate = raw_process * high_process.to(
                            dtype=raw_process.dtype
                        )
                elif self.hotspot_laser_prior_gate_source in {
                        "target_heat_or_neighbor_supported",
                        "target_sweep",
                        "target_sweep_neighbor_product",
                        "target_sweep_or_neighbor_supported",
                }:
                    candidate_gates = []
                    direct_gates = []
                    if laser_target_heat_gate is not None:
                        direct_gates.append(
                            laser_target_heat_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                        )
                    if (
                            self.hotspot_laser_prior_gate_source in {
                                "target_sweep",
                                "target_sweep_neighbor_product",
                                "target_sweep_or_neighbor_supported",
                            }
                            and laser_sweep_heat_gate is not None):
                        direct_gates.append(
                            laser_sweep_heat_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                        )
                    if direct_gates:
                        direct_gate = direct_gates[0]
                        for gate in direct_gates[1:]:
                            direct_gate = torch.maximum(direct_gate, gate)
                        confident_direct = (
                            direct_gate >= self.hotspot_laser_prior_raw_process_threshold
                        )
                        direct_gate = direct_gate * confident_direct.to(
                            dtype=direct_gate.dtype
                        )
                        if (
                                self.hotspot_laser_prior_gate_source
                                == "target_sweep_neighbor_product"):
                            if raw_neighbor_gate_for_specialist is not None:
                                neighbor_score = raw_neighbor_gate_for_specialist.clamp(
                                    0.0, 1.0
                                )
                                if self.hotspot_laser_prior_neighbor_threshold > 0:
                                    neighbor_score = (
                                        (
                                            neighbor_score
                                            - self.hotspot_laser_prior_neighbor_threshold
                                        )
                                        / max(
                                            1.0 - self.hotspot_laser_prior_neighbor_threshold,
                                            1.0e-6,
                                        )
                                    ).clamp(0.0, 1.0)
                                candidate_gates.append(direct_gate * neighbor_score)
                        else:
                            candidate_gates.append(direct_gate)
                    if (
                            self.hotspot_laser_prior_gate_source
                            not in {"target_sweep", "target_sweep_neighbor_product"}
                            and
                            raw_process_gate_for_specialist is not None
                            and raw_neighbor_gate_for_specialist is not None):
                        raw_process = raw_process_gate_for_specialist.clamp(0.0, 1.0)
                        neighbor_supported = (
                            raw_neighbor_gate_for_specialist.clamp(0.0, 1.0)
                            >= self.hotspot_laser_prior_neighbor_threshold
                        )
                        candidate_gates.append(
                            raw_process * neighbor_supported.to(dtype=raw_process.dtype)
                        )
                    if candidate_gates:
                        hotspot_laser_prior_gate = candidate_gates[0]
                        for gate in candidate_gates[1:]:
                            hotspot_laser_prior_gate = torch.maximum(
                                hotspot_laser_prior_gate,
                                gate,
                            )
                elif process_gate_for_specialist is not None:
                    hotspot_laser_prior_gate = process_gate_for_specialist.clamp(0.0, 1.0)

                if hotspot_laser_prior_gate is not None:
                    if self.hotspot_laser_prior_require_neighbor:
                        if neighbor_gate_for_specialist is not None:
                            hotspot_laser_prior_gate = (
                                hotspot_laser_prior_gate
                                * neighbor_gate_for_specialist.clamp(0.0, 1.0)
                            )
                        else:
                            hotspot_laser_prior_gate = torch.zeros_like(
                                hotspot_laser_prior_gate
                            )
                    if (
                            raw_process_gate_for_specialist is not None
                            and self.hotspot_laser_prior_raw_process_threshold > 0.0
                            and self.hotspot_laser_prior_gate_source not in {
                                "raw_process_or_neighbor_supported",
                                "target_heat_or_neighbor_supported",
                                "target_sweep",
                                "target_sweep_or_neighbor_supported",
                            }):
                        hotspot_laser_prior_gate = hotspot_laser_prior_gate * (
                            raw_process_gate_for_specialist
                            >= self.hotspot_laser_prior_raw_process_threshold
                        ).to(dtype=hotspot_laser_prior_gate.dtype)
                    if self.hotspot_laser_prior_gate_threshold > 0.0:
                        denom = max(
                            1.0 - self.hotspot_laser_prior_gate_threshold,
                            1.0e-6,
                        )
                        hotspot_laser_prior_gate = (
                            (hotspot_laser_prior_gate - self.hotspot_laser_prior_gate_threshold)
                            / denom
                        ).clamp(0.0, 1.0)
                    if abs(self.hotspot_laser_prior_gate_power - 1.0) > 1.0e-6:
                        hotspot_laser_prior_gate = self._apply_gate_power(
                            hotspot_laser_prior_gate,
                            self.hotspot_laser_prior_gate_power,
                        )
                    hotspot_laser_prior_temp = (
                        self.hotspot_specialist_min_temp
                        + self.hotspot_laser_prior_max_delta * hotspot_laser_prior_gate
                    ).clamp(
                        min=self.hotspot_specialist_min_temp,
                        max=self.hotspot_specialist_max_temp,
                    )
                    if self.hotspot_laser_prior_mode in {"specialist_floor", "blend_floor"}:
                        hotspot_specialist_temp = torch.maximum(
                            hotspot_specialist_temp,
                            hotspot_laser_prior_temp.to(dtype=hotspot_specialist_temp.dtype),
                        )

            if (self.hotspot_specialist_gate_merge in {
                    "laser_residual_prior_strong_support_product",
                    "residual_prior_strong_support_product",
                    "calibrated_scan_prior_strong_support_product",
                }
                    and laser_residual_prior_gate is not None):
                prior_gate = laser_residual_prior_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                strong_support = self._select_laser_residual_strong_support(
                    laser_residual_prior_gate=laser_residual_prior_gate,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=T_pred.dtype,
                )
                if strong_support is not None:
                    gates.append(prior_gate * strong_support)
                else:
                    gates.append(prior_gate)
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "laser_residual_prior",
                    "residual_prior",
                    "calibrated_scan_prior",
                }
                    and laser_residual_prior_gate is not None):
                gates.append(
                    laser_residual_prior_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                )
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "laser_residual_prior_cold_to_hot_product",
                    "residual_prior_cold_to_hot_product",
                    "laser_residual_prior_times_cold_to_hot",
                    "residual_prior_times_cold_to_hot",
                    "calibrated_scan_prior_cold_to_hot_product",
                }
                    and laser_residual_prior_gate is not None
                    and cold_to_hot_gate is not None):
                prior_gate = laser_residual_prior_gate.to(
                    dtype=T_pred.dtype
                ).clamp(0.0, 1.0)
                cold_gate = self._calibrated_cold_to_hot_gate(
                    cold_to_hot_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                )
                gates.append(prior_gate * cold_gate)
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist * prior_gate)
            elif (self.hotspot_specialist_gate_merge in {
                    "laser_residual_prior_or_cold_to_hot",
                    "residual_prior_or_cold_to_hot",
                    "calibrated_scan_prior_or_cold_to_hot",
                }
                    and laser_residual_prior_gate is not None
                    and cold_to_hot_gate is not None):
                prior_gate = laser_residual_prior_gate.to(
                    dtype=T_pred.dtype
                ).clamp(0.0, 1.0)
                cold_gate = self._calibrated_cold_to_hot_gate(
                    cold_to_hot_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                )
                gates.append(torch.maximum(prior_gate, cold_gate))
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "arrival",
                    "arrival_only",
                    "path_arrival",
                }
                    and laser_arrival_gate is not None):
                gates.append(
                    laser_arrival_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                )
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "arrival_or_neighbor",
                    "arrival_neighbor_max",
                }
                    and laser_arrival_gate is not None):
                gates.append(
                    laser_arrival_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                )
                if neighbor_gate_for_specialist is not None:
                    gates.append(neighbor_gate_for_specialist)
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "arrival_neighbor_product",
                    "product_arrival_neighbor",
                }
                    and laser_arrival_gate is not None
                    and neighbor_gate_for_specialist is not None):
                arrival_gate_for_specialist = laser_arrival_gate.to(
                    dtype=T_pred.dtype
                ).clamp(0.0, 1.0)
                gates.append(
                    arrival_gate_for_specialist
                    * neighbor_gate_for_specialist.clamp(0.0, 1.0)
                )
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "arrival_neighbor_support",
                    "arrival_with_neighbor_support",
                }
                    and laser_arrival_gate is not None):
                arrival_gate_for_specialist = laser_arrival_gate.to(
                    dtype=T_pred.dtype
                ).clamp(0.0, 1.0)
                if neighbor_gate_for_specialist is not None:
                    support = (
                        self.hotspot_specialist_neighbor_support_base
                        + self.hotspot_specialist_neighbor_support_weight
                        * neighbor_gate_for_specialist.clamp(0.0, 1.0)
                    ).clamp(0.0, 1.0)
                    gates.append(arrival_gate_for_specialist * support)
                else:
                    gates.append(arrival_gate_for_specialist)
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "arrival_process_product",
                    "process_arrival_product",
                }
                    and laser_arrival_gate is not None
                    and process_gate_for_specialist is not None):
                arrival_gate_for_specialist = laser_arrival_gate.to(
                    dtype=T_pred.dtype
                ).clamp(0.0, 1.0)
                gates.append(
                    arrival_gate_for_specialist
                    * process_gate_for_specialist.clamp(0.0, 1.0)
                )
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge == "multiply_cls"
                    and cls_gate_for_specialist is not None
                    and process_gate_for_specialist is not None):
                gates.append(cls_gate_for_specialist * process_gate_for_specialist)
                if neighbor_gate_for_specialist is not None:
                    gates.append(neighbor_gate_for_specialist)
            elif (self.hotspot_specialist_gate_merge in {
                    "multiply_process_neighbor",
                    "process_neighbor_product",
                    "product_process_neighbor",
                }
                    and process_gate_for_specialist is not None
                    and neighbor_gate_for_specialist is not None):
                gates.append(process_gate_for_specialist * neighbor_gate_for_specialist)
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
            else:
                if cls_gate_for_specialist is not None:
                    gates.append(cls_gate_for_specialist)
                if process_gate_for_specialist is not None:
                    gates.append(process_gate_for_specialist)
                if neighbor_gate_for_specialist is not None:
                    gates.append(neighbor_gate_for_specialist)

            if gates:
                hotspot_specialist_gate = gates[0]
                for gate in gates[1:]:
                    hotspot_specialist_gate = torch.maximum(hotspot_specialist_gate, gate)
                if abs(self.hotspot_specialist_gate_power - 1.0) > 1.0e-6:
                    gate_for_power = hotspot_specialist_gate.clamp(0.0, 1.0)
                    if self.hotspot_specialist_gate_power < 1.0:
                        gate_for_power = gate_for_power.clamp_min(1.0e-6)
                    hotspot_specialist_gate = gate_for_power.pow(
                        self.hotspot_specialist_gate_power
                    )
            else:
                hotspot_specialist_gate = torch.zeros_like(hotspot_specialist_temp)

            if mask_last is not None:
                mask_value = mask_last.unsqueeze(-1).to(T_pred.dtype)
                hotspot_specialist_gate = hotspot_specialist_gate * mask_value

            blend_weight = float(getattr(self, "hotspot_specialist_blend_weight", 1.0))
            effective_specialist_gate = hotspot_specialist_gate * blend_weight
            if post_endpoint_false_hot_guard is not None:
                effective_specialist_gate = effective_specialist_gate * (
                    1.0 - post_endpoint_false_hot_guard.to(dtype=effective_specialist_gate.dtype)
                )
            base_for_blend = T_pred.detach() if self.hotspot_specialist_detach_base else T_pred
            T_pred = (
                base_for_blend * (1.0 - effective_specialist_gate)
                + hotspot_specialist_temp * effective_specialist_gate
            )
            if (
                    hotspot_laser_prior_temp is not None
                    and self.hotspot_laser_prior_mode in {"final_floor", "blend_floor"}):
                T_pred = torch.maximum(
                    T_pred,
                    hotspot_laser_prior_temp.to(dtype=T_pred.dtype),
                )
            hotspot_specialist_raw_gate = hotspot_specialist_gate
            hotspot_specialist_gate = effective_specialist_gate
        else:
            hotspot_specialist_raw_gate = None

        if self.laser_residual_head is not None:
            residual_raw = self.laser_residual_head(z_head)
            laser_residual_learned_gate = torch.sigmoid(residual_raw[..., :1])
            laser_residual_delta = (
                self.laser_residual_max_delta * torch.sigmoid(residual_raw[..., 1:2])
            )
            if (
                    self.laser_residual_min_delta > 0.0
                    and not self.laser_residual_use_tiered_min_delta):
                min_delta = torch.full_like(
                    laser_residual_delta,
                    min(self.laser_residual_min_delta, self.laser_residual_max_delta),
                )
                laser_residual_delta = torch.maximum(laser_residual_delta, min_delta)
            if laser_residual_prior_gate is None:
                if self.laser_residual_gate_source == "learned":
                    laser_residual_prior_gate = torch.ones_like(laser_residual_learned_gate)
                else:
                    laser_residual_prior_gate = torch.zeros_like(laser_residual_learned_gate)
            if (
                    self.laser_residual_min_delta > 0.0
                    and self.laser_residual_use_tiered_min_delta):
                laser_residual_delta = self._apply_tiered_laser_residual_min_delta(
                    laser_residual_delta=laser_residual_delta,
                    laser_residual_prior_gate=laser_residual_prior_gate,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=T_pred.dtype,
                )
            laser_residual_control_gate = self._mix_laser_residual_control_gate(
                learned_gate=laser_residual_learned_gate,
                cold_to_hot_gate=cold_to_hot_gate,
            )
            if self.laser_residual_control_gate_floor > 0.0:
                floor = torch.full_like(
                    laser_residual_control_gate,
                    self.laser_residual_control_gate_floor,
                )
                laser_residual_control_gate = torch.maximum(
                    laser_residual_control_gate,
                    floor,
                )
            laser_residual_gate = (
                laser_residual_control_gate.to(dtype=T_pred.dtype)
                * laser_residual_prior_gate.to(dtype=T_pred.dtype)
            ).clamp(0.0, 1.0)
            if (
                    self.laser_residual_post_suppress_strength > 0.0
                    and laser_path_post_arrival_gate is not None):
                suppress = (
                    self.laser_residual_post_suppress_strength
                    * laser_path_post_arrival_gate.to(dtype=T_pred.dtype)
                ).clamp(0.0, 1.0)
                laser_residual_gate = laser_residual_gate * (1.0 - suppress)
            if self.laser_residual_cold_start_threshold > 0.0:
                input_temp_max = self._input_window_temperature_max(
                    graph_sequence=graph_sequence,
                    device=T_pred.device,
                    dtype=T_pred.dtype,
                )
                threshold = self.laser_residual_cold_start_threshold
                softness = self.laser_residual_cold_start_softness
                if softness > 0.0:
                    laser_residual_cold_start_gate = (
                        (threshold + softness - input_temp_max) / softness
                    ).clamp(0.0, 1.0)
                else:
                    laser_residual_cold_start_gate = (
                        input_temp_max <= threshold
                    ).to(dtype=T_pred.dtype)
                laser_residual_cold_start_gate = (
                    laser_residual_cold_start_gate
                    .view(1, -1, 1)
                    .expand(T_pred.shape[0], -1, -1)
                )
                if self.laser_residual_prior_bypass_cold_start:
                    prior_bypass_gate = (
                        laser_residual_prior_gate.to(dtype=T_pred.dtype)
                        >= self.laser_residual_prior_bypass_cold_start_threshold
                    ).to(dtype=T_pred.dtype)
                    laser_residual_cold_start_gate = torch.maximum(
                        laser_residual_cold_start_gate,
                        prior_bypass_gate,
                    )
                laser_residual_gate = laser_residual_gate * laser_residual_cold_start_gate
            laser_residual_boost = laser_residual_gate * laser_residual_delta.to(
                dtype=T_pred.dtype
            )
            if mask_last is not None:
                mask_value = mask_last.unsqueeze(-1).to(T_pred.dtype)
                laser_residual_learned_gate = laser_residual_learned_gate * mask_value
                laser_residual_control_gate = laser_residual_control_gate * mask_value
                laser_residual_prior_gate = laser_residual_prior_gate * mask_value
                if laser_residual_cold_start_gate is not None:
                    laser_residual_cold_start_gate = (
                        laser_residual_cold_start_gate * mask_value
                    )
                laser_residual_gate = laser_residual_gate * mask_value
                laser_residual_delta = laser_residual_delta * mask_value
                laser_residual_boost = laser_residual_boost * mask_value
            T_pred = T_pred + laser_residual_boost
            if (
                    self.laser_residual_post_rescue_cap_margin > 0.0
                    and hotspot_specialist_temp is not None
                    and laser_path_post_arrival_gate is not None):
                post_support = self._select_laser_residual_strong_support(
                    laser_residual_prior_gate=None,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=T_pred.dtype,
                    source_tokens_override="postarrival_strict_wake_body_track_cold_all",
                )
                if post_support is not None:
                    post_phase = laser_path_post_arrival_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                    laser_residual_post_rescue_cap_gate = (
                        post_phase
                        * (post_phase >= self.laser_residual_post_phase_threshold).to(dtype=T_pred.dtype)
                        * post_support.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                    )
                    if laser_path_pre_arrival_gate is not None:
                        pre_phase = laser_path_pre_arrival_gate.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
                        no_pre_rescue = (
                            pre_phase < self.laser_residual_body_prearrival_threshold
                        ).to(dtype=T_pred.dtype)
                        laser_residual_post_rescue_cap_gate = (
                            laser_residual_post_rescue_cap_gate * no_pre_rescue
                        )
                    if self.laser_residual_post_rescue_cap_threshold > 0.0:
                        cap_mask = (
                            laser_residual_post_rescue_cap_gate
                            >= self.laser_residual_post_rescue_cap_threshold
                        ).to(dtype=T_pred.dtype)
                    else:
                        cap_mask = laser_residual_post_rescue_cap_gate.clamp(0.0, 1.0)
                    cap_temp = (
                        hotspot_specialist_temp.to(dtype=T_pred.dtype)
                        + self.laser_residual_post_rescue_cap_margin
                    )
                    capped = torch.minimum(T_pred, cap_temp)
                    T_pred = T_pred * (1.0 - cap_mask) + capped * cap_mask

        if (
                post_endpoint_false_hot_guard is not None
                and self.post_endpoint_false_hot_temperature_ceiling > 0.0):
            ceiling = torch.full_like(
                T_pred,
                self.post_endpoint_false_hot_temperature_ceiling,
            )
            capped = torch.minimum(T_pred, ceiling)
            guard = post_endpoint_false_hot_guard.to(dtype=T_pred.dtype).clamp(0.0, 1.0)
            if self.post_endpoint_false_hot_hard_ceiling:
                hard_mask = (
                    guard >= self.post_endpoint_false_hot_hard_ceiling_threshold
                ).to(dtype=T_pred.dtype)
                T_pred = T_pred * (1.0 - hard_mask) + capped * hard_mask
            else:
                T_pred = T_pred * (1.0 - guard) + capped * guard

        return {
            "T_pred": T_pred,
            "T_base": T_base,
            "T_delta": T_head if self.predict_temperature_delta else None,
            "hotspot_delta": hotspot_delta,
            "hotspot_specialist_temp": hotspot_specialist_temp,
            "hotspot_specialist_gate": hotspot_specialist_gate,
            "hotspot_specialist_raw_gate": hotspot_specialist_raw_gate,
            "hotspot_laser_prior_temp": hotspot_laser_prior_temp,
            "hotspot_laser_prior_gate": hotspot_laser_prior_gate,
            "laser_residual_prior_gate": laser_residual_prior_gate,
            "laser_residual_learned_gate": laser_residual_learned_gate,
            "laser_residual_control_gate": laser_residual_control_gate,
            "laser_residual_cold_start_gate": laser_residual_cold_start_gate,
            "laser_residual_gate": laser_residual_gate,
            "laser_residual_delta": laser_residual_delta,
            "laser_residual_boost": laser_residual_boost,
            "laser_residual_post_rescue_cap_gate": laser_residual_post_rescue_cap_gate,
            "cold_to_hot_logit": cold_to_hot_logit,
            "cold_to_hot_gate": cold_to_hot_gate,
            "post_endpoint_false_hot_guard": post_endpoint_false_hot_guard,
            "process_gate": process_gate,
            "laser_target_heat_gate": laser_target_heat_gate,
            "laser_body_heat_gate": laser_body_heat_gate,
            "laser_sweep_heat_gate": laser_sweep_heat_gate,
            "laser_arrival_gate": laser_arrival_gate,
            "laser_path_active_gate": laser_path_active_gate,
            "laser_path_pre_arrival_gate": laser_path_pre_arrival_gate,
            "laser_path_post_arrival_gate": laser_path_post_arrival_gate,
            "laser_path_body_heat_gate": laser_path_body_heat_gate,
            "laser_path_wake_gate": laser_path_wake_gate,
            "laser_path_pre_wake_gate": laser_path_pre_wake_gate,
            "laser_path_post_wake_gate": laser_path_post_wake_gate,
            "laser_path_program_track_gate": laser_path_program_track_gate,
            "laser_path_elapsed_gate": laser_path_elapsed_gate,
            "laser_path_scanned_gate": laser_path_scanned_gate,
            "laser_path_time_until_gate": laser_path_time_until_gate,
            "laser_path_cooling_tail_gate": laser_path_cooling_tail_gate,
            "laser_endpoint_gate": laser_endpoint_gate,
            "neighbor_hot_gate": neighbor_hot_gate,
            "hotspot_logit": hotspot_logit,
            "spatial_features": z_s,
            "temporal_features": z_t,
            "fused_features": z_f,
        }

    def _post_endpoint_false_hot_guard(
            self,
            *,
            laser_path_post_arrival_gate: torch.Tensor | None,
            laser_endpoint_gate: torch.Tensor | None,
            laser_sweep_heat_gate: torch.Tensor | None,
            laser_path_pre_arrival_gate: torch.Tensor | None,
            dtype: torch.dtype) -> torch.Tensor | None:
        if (
                self.post_endpoint_false_hot_guard_strength <= 0.0
                and self.post_low_sweep_false_hot_guard_strength <= 0.0
                and self.post_mid_phase_false_hot_guard_strength <= 0.0):
            return None
        if laser_path_post_arrival_gate is None:
            return None

        post_gate = laser_path_post_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
        prearrival_block = None
        if (
                self.post_endpoint_false_hot_guard_require_no_prearrival
                and laser_path_pre_arrival_gate is not None):
            prearrival_block = (
                1.0 - laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
            )
        guard = None
        if (
                self.post_endpoint_false_hot_guard_strength > 0.0
                and laser_endpoint_gate is not None):
            endpoint_guard = (
                post_gate
                * laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
            )
            if prearrival_block is not None:
                endpoint_guard = endpoint_guard * prearrival_block
            endpoint_threshold = self.post_endpoint_false_hot_guard_threshold
            if endpoint_threshold > 0.0:
                endpoint_guard = (
                    (endpoint_guard - endpoint_threshold)
                    / max(1.0 - endpoint_threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                endpoint_guard = endpoint_guard.clamp(0.0, 1.0)
            endpoint_guard = self._apply_gate_power(
                endpoint_guard,
                self.post_endpoint_false_hot_guard_power,
            )
            endpoint_guard = (
                endpoint_guard * self.post_endpoint_false_hot_guard_strength
            ).clamp(0.0, self.post_endpoint_false_hot_guard_max)
            guard = endpoint_guard

        if (
                self.post_low_sweep_false_hot_guard_strength > 0.0
                and laser_sweep_heat_gate is not None):
            low_sweep_guard = post_gate * (
                1.0 - laser_sweep_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
            )
            if prearrival_block is not None:
                low_sweep_guard = low_sweep_guard * prearrival_block
            threshold = self.post_low_sweep_false_hot_guard_threshold
            if threshold > 0.0:
                low_sweep_guard = (
                    (low_sweep_guard - threshold)
                    / max(1.0 - threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                low_sweep_guard = low_sweep_guard.clamp(0.0, 1.0)
            low_sweep_guard = self._apply_gate_power(
                low_sweep_guard,
                self.post_low_sweep_false_hot_guard_power,
            )
            low_sweep_guard = (
                low_sweep_guard * self.post_low_sweep_false_hot_guard_strength
            ).clamp(0.0, self.post_low_sweep_false_hot_guard_max)
            guard = low_sweep_guard if guard is None else torch.maximum(
                guard,
                low_sweep_guard,
            )

        if self.post_mid_phase_false_hot_guard_strength > 0.0:
            in_mid_post = (
                (post_gate >= self.post_mid_phase_false_hot_post_min)
                & (post_gate <= self.post_mid_phase_false_hot_post_max)
            ).to(dtype=dtype)
            mid_phase_guard = post_gate * in_mid_post
            if prearrival_block is not None:
                mid_phase_guard = mid_phase_guard * prearrival_block
            if laser_endpoint_gate is not None:
                endpoint_block = (
                    laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                    <= self.post_mid_phase_false_hot_endpoint_max
                ).to(dtype=dtype)
                mid_phase_guard = mid_phase_guard * endpoint_block
            threshold = self.post_mid_phase_false_hot_guard_threshold
            if threshold > 0.0:
                mid_phase_guard = (
                    (mid_phase_guard - threshold)
                    / max(1.0 - threshold, 1.0e-6)
                ).clamp(0.0, 1.0)
            else:
                mid_phase_guard = mid_phase_guard.clamp(0.0, 1.0)
            mid_phase_guard = self._apply_gate_power(
                mid_phase_guard,
                self.post_mid_phase_false_hot_guard_power,
            )
            mid_phase_guard = (
                mid_phase_guard * self.post_mid_phase_false_hot_guard_strength
            ).clamp(0.0, self.post_mid_phase_false_hot_guard_max)
            guard = mid_phase_guard if guard is None else torch.maximum(
                guard,
                mid_phase_guard,
            )

        if guard is None:
            return None
        return guard.clamp(0.0, 1.0)

    def _select_laser_residual_prior_gate(
            self,
            process_gate: torch.Tensor | None,
            laser_arrival_gate: torch.Tensor | None,
            laser_path_active_gate: torch.Tensor | None,
            laser_path_pre_arrival_gate: torch.Tensor | None,
            laser_path_post_arrival_gate: torch.Tensor | None,
            laser_endpoint_gate: torch.Tensor | None,
            laser_target_heat_gate: torch.Tensor | None,
            laser_body_heat_gate: torch.Tensor | None,
            laser_path_body_heat_gate: torch.Tensor | None,
            laser_path_wake_gate: torch.Tensor | None,
            laser_path_pre_wake_gate: torch.Tensor | None,
            laser_path_post_wake_gate: torch.Tensor | None,
            laser_path_program_track_gate: torch.Tensor | None,
            laser_path_elapsed_gate: torch.Tensor | None,
            laser_path_scanned_gate: torch.Tensor | None,
            laser_path_time_until_gate: torch.Tensor | None,
            laser_path_cooling_tail_gate: torch.Tensor | None,
            laser_sweep_heat_gate: torch.Tensor | None,
            neighbor_hot_gate: torch.Tensor | None,
            cold_to_hot_gate: torch.Tensor | None,
            dtype: torch.dtype) -> torch.Tensor | None:
        source = self.laser_residual_gate_source
        if source == "learned":
            return None

        def norm(gate: torch.Tensor | None) -> torch.Tensor | None:
            if gate is None:
                return None
            return gate.to(dtype=dtype).clamp(0.0, 1.0)

        arrival = norm(laser_arrival_gate)
        path_active = norm(laser_path_active_gate)
        path_pre_arrival = norm(laser_path_pre_arrival_gate)
        path_post_arrival = norm(laser_path_post_arrival_gate)
        endpoint = norm(laser_endpoint_gate)
        process = norm(process_gate)
        target_heat = norm(laser_target_heat_gate)
        body = norm(laser_body_heat_gate)
        path_body = norm(laser_path_body_heat_gate)
        path_wake = norm(laser_path_wake_gate)
        path_pre_wake = norm(laser_path_pre_wake_gate)
        path_post_wake = norm(laser_path_post_wake_gate)
        path_program_track = norm(laser_path_program_track_gate)
        path_elapsed = norm(laser_path_elapsed_gate)
        path_scanned = norm(laser_path_scanned_gate)
        path_time_until = norm(laser_path_time_until_gate)
        path_cooling_tail = norm(laser_path_cooling_tail_gate)
        sweep = norm(laser_sweep_heat_gate)
        neighbor = norm(neighbor_hot_gate)

        selected = None
        if source == "arrival":
            selected = arrival
        elif source == "process":
            selected = process
        elif source == "arrival_or_process":
            gates = [gate for gate in (arrival, process) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "arrival_process_product":
            if arrival is not None and process is not None:
                selected = arrival * process
        elif source == "target_heat":
            selected = target_heat
        elif source == "target_body":
            selected = body
        elif source == "path_body":
            selected = path_body
        elif source == "path_body_or_endpoint":
            gates = [gate for gate in (path_body, endpoint) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "target_sweep":
            selected = sweep
        elif source == "target_sweep_or_arrival":
            gates = [gate for gate in (sweep, arrival) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "arrival_sweep_product":
            if arrival is not None and sweep is not None:
                selected = arrival * sweep
        elif source == "arrival_track_support_product":
            if arrival is not None:
                support_gates = [gate for gate in (sweep, endpoint) if gate is not None]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    selected = arrival * support
        elif source == "arrival_track_support_blend":
            if arrival is not None:
                support_gates = [gate for gate in (sweep, endpoint) if gate is not None]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    selected = arrival * (0.45 + 0.55 * support).clamp(0.0, 1.0)
                else:
                    selected = arrival * 0.45
        elif source == "endpoint":
            selected = endpoint
        elif source == "endpoint_or_arrival":
            gates = [gate for gate in (endpoint, arrival) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "endpoint_or_sweep":
            gates = [gate for gate in (endpoint, sweep) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "endpoint_sweep_or_arrival":
            gates = [gate for gate in (endpoint, sweep, arrival) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "arrival_endpoint_product":
            if arrival is not None and endpoint is not None:
                selected = arrival * endpoint
        elif source == "arrival_support_blend":
            if arrival is not None:
                support_gates = [
                    gate for gate in (sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    selected = arrival * (0.65 + 0.35 * support).clamp(0.0, 1.0)
                else:
                    selected = arrival * 0.65
        elif source == "pre_arrival":
            selected = path_pre_arrival
        elif source == "pre_arrival_or_arrival":
            gates = [gate for gate in (path_pre_arrival, arrival) if gate is not None]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "pre_arrival_with_endpoint_support":
            if path_pre_arrival is not None:
                if endpoint is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    support = (
                        endpoint >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
        elif source == "pre_arrival_with_body_support":
            if path_pre_arrival is not None:
                if body is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    support = (
                        body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
        elif source == "pre_arrival_with_body_or_endpoint_support":
            if path_pre_arrival is not None:
                support_gates = [gate for gate in (body, endpoint) if gate is not None]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
                else:
                    selected = torch.zeros_like(path_pre_arrival)
        elif source == "pre_arrival_with_path_body_support":
            if path_pre_arrival is not None:
                if path_body is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    support = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
        elif source == "pre_arrival_active_path_body_support":
            if path_pre_arrival is not None:
                if path_active is None or path_body is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    body_support = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_support = (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * path_active * body_support * active_support
        elif source == "pre_arrival_active_body_or_endpoint_support":
            if path_pre_arrival is not None:
                high_pre = (
                    path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)
                support = None
                if endpoint is not None:
                    support = (
                        endpoint >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                if path_body is not None and path_active is not None:
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )
                if support is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    selected = path_pre_arrival * high_pre * support
        elif source == "path_phase_active_body_or_track_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                support_gates = [
                    gate for gate in (path_body, sweep, target_heat, endpoint)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                if path_active is not None:
                    active_support = (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    support = active_support if support is None else support * active_support
                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "pre_arrival_with_path_body_or_endpoint_support":
            if path_pre_arrival is not None:
                support_gates = [
                    gate for gate in (path_body, endpoint) if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
                else:
                    selected = torch.zeros_like(path_pre_arrival)
        elif source == "pre_arrival_with_sweep_support":
            if path_pre_arrival is not None:
                if sweep is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    support = (
                        sweep >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
        elif source == "pre_arrival_track_support_blend":
            if path_pre_arrival is not None:
                support_gates = [
                    gate for gate in (sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    selected = path_pre_arrival * (0.45 + 0.55 * support).clamp(
                        0.0, 1.0
                    )
                else:
                    selected = path_pre_arrival * 0.45
        elif source == "pre_arrival_with_track_heat_support":
            if path_pre_arrival is not None:
                support_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = path_pre_arrival * support
                else:
                    selected = torch.zeros_like(path_pre_arrival)
        elif source == "pre_arrival_with_track_heat_or_body_high_pre_support":
            if path_pre_arrival is not None:
                support = None
                support_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                if path_body is not None:
                    body_support = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    high_pre = (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    body_support = body_support * high_pre
                    support = body_support if support is None else torch.maximum(
                        support, body_support
                    )
                if support is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    selected = path_pre_arrival * support
        elif source == "pre_arrival_with_high_pre_track_heat_or_body_support":
            if path_pre_arrival is not None:
                high_pre = (
                    path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)
                support = None
                support_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor, path_body)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    support = support * high_pre
                if support is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    selected = path_pre_arrival * support
        elif source == "pre_or_post_arrival_with_strong_track_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                support_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = phase * support
                else:
                    selected = torch.zeros_like(phase)
        elif source == "high_phase_with_track_heat_or_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)
                support_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor, path_body)
                    if gate is not None
                ]
                if support_gates:
                    support = support_gates[0]
                    for gate in support_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    selected = phase * high_phase * support
                else:
                    selected = torch.zeros_like(phase)
        elif source == "high_phase_with_midline_body_or_track_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                track_gates = [
                    gate for gate in (target_heat, sweep, neighbor)
                    if gate is not None
                ]
                if track_gates:
                    support = track_gates[0]
                    for gate in track_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if path_body is not None:
                    if endpoint is None:
                        midline = torch.ones_like(path_body, dtype=dtype)
                    else:
                        midline = (
                            endpoint <= self.laser_residual_path_body_endpoint_max
                        ).to(dtype=dtype)
                    body_support = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype) * midline
                    support = body_support if support is None else torch.maximum(
                        support, body_support
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_endpoint_or_midline_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                track_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if track_gates:
                    support = track_gates[0]
                    for gate in track_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if path_body is not None:
                    if endpoint is None:
                        midline = torch.ones_like(path_body, dtype=dtype)
                    else:
                        midline = (
                            endpoint <= self.laser_residual_path_body_endpoint_max
                        ).to(dtype=dtype)
                    body_support = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype) * midline
                    support = body_support if support is None else torch.maximum(
                        support, body_support
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_endpoint_midline_or_active_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                track_gates = [
                    gate for gate in (target_heat, sweep, endpoint, neighbor)
                    if gate is not None
                ]
                if track_gates:
                    support = track_gates[0]
                    for gate in track_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if path_body is not None:
                    if endpoint is None:
                        midline = torch.ones_like(path_body, dtype=dtype)
                    else:
                        midline = (
                            endpoint <= self.laser_residual_path_body_endpoint_max
                        ).to(dtype=dtype)
                    body_support = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype) * midline

                    if path_pre_arrival is not None and path_active is not None:
                        active_body = (
                            path_body >= self.laser_residual_prior_support_threshold
                        ).to(dtype=dtype)
                        active_body = active_body * (
                            path_active >= self.laser_residual_active_support_threshold
                        ).to(dtype=dtype)
                        active_body = active_body * (
                            path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                        ).to(dtype=dtype)
                        body_support = torch.maximum(body_support, active_body)

                    support = body_support if support is None else torch.maximum(
                        support, body_support
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_active_path_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    selected = phase * high_phase * active_body
                else:
                    selected = torch.zeros_like(phase)
        elif source == "high_phase_with_endpoint_or_active_path_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if endpoint is not None:
                    support = (
                        endpoint >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_neighbor_or_endpoint_or_active_path_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                track_gates = [
                    gate for gate in (endpoint, neighbor)
                    if gate is not None
                ]
                if track_gates:
                    support = track_gates[0]
                    for gate in track_gates[1:]:
                        support = torch.maximum(support, gate)
                    support = (
                        support >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_post_neighbor_or_endpoint_or_active_body_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    post_gates = [
                        gate for gate in (endpoint, neighbor)
                        if gate is not None
                    ]
                    if post_gates:
                        post_support = post_gates[0]
                        for gate in post_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_prior_support_threshold
                        ).to(dtype=dtype) * post_phase
                        support = post_support

                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_endpoint_active_body_or_post_sweep_neighbor_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if endpoint is not None:
                    support = (
                        endpoint >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if post_support_gates:
                        post_support = post_support_gates[0]
                        for gate in post_support_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype) * post_phase
                        support = post_support if support is None else torch.maximum(
                            support, post_support
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_endpoint_active_body_pre_sweep_or_post_sweep_neighbor_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if endpoint is not None:
                    support = (
                        endpoint >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None
                        and sweep is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        sweep >= self.laser_residual_pre_sweep_support_threshold
                    ).to(dtype=dtype)
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if post_support_gates:
                        post_support = post_support_gates[0]
                        for gate in post_support_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype) * post_phase
                        support = post_support if support is None else torch.maximum(
                            support, post_support
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_endpoint_active_body_soft_track_or_post_sweep_neighbor_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if endpoint is not None:
                    support = (
                        endpoint >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)

                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)

                    track_gates = [
                        gate for gate in (sweep, endpoint, neighbor)
                        if gate is not None
                    ]
                    if track_gates:
                        track_conf = track_gates[0]
                        for gate in track_gates[1:]:
                            track_conf = torch.maximum(track_conf, gate)
                    else:
                        track_conf = torch.zeros_like(path_body)
                    track_floor = self.laser_residual_pre_track_support_floor
                    track_support = (
                        track_floor + (1.0 - track_floor) * track_conf
                    ).clamp(0.0, 1.0)
                    active_body = active_body * track_support
                    support = active_body if support is None else torch.maximum(
                        support, active_body
                    )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if post_support_gates:
                        post_support = post_support_gates[0]
                        for gate in post_support_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype) * post_phase
                        support = post_support if support is None else torch.maximum(
                            support, post_support
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_active_body_pre_sweep_neighbor_or_post_sweep_neighbor_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if (
                        path_body is not None
                        and path_active is not None
                        and path_pre_arrival is not None):
                    active_body = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)
                    active_body = active_body * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    pre_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if pre_support_gates:
                        pre_support = pre_support_gates[0]
                        for gate in pre_support_gates[1:]:
                            pre_support = torch.maximum(pre_support, gate)
                        pre_support = (
                            pre_support >= self.laser_residual_pre_sweep_support_threshold
                        ).to(dtype=dtype)
                        active_body = active_body * pre_support
                        support = active_body if support is None else torch.maximum(
                            support, active_body
                        )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if post_support_gates:
                        post_support = post_support_gates[0]
                        for gate in post_support_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype) * post_phase
                        support = post_support if support is None else torch.maximum(
                            support, post_support
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_pre_sweep_neighbor_endpoint_or_post_sweep_neighbor_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if path_pre_arrival is not None:
                    pre_phase = (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    pre_support = None
                    pre_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if pre_support_gates:
                        pre_support = pre_support_gates[0]
                        for gate in pre_support_gates[1:]:
                            pre_support = torch.maximum(pre_support, gate)
                        pre_support = (
                            pre_support >= self.laser_residual_pre_sweep_support_threshold
                        ).to(dtype=dtype)
                    if endpoint is not None:
                        endpoint_support = (
                            endpoint >= self.laser_residual_pre_endpoint_support_threshold
                        ).to(dtype=dtype)
                        pre_support = endpoint_support if pre_support is None else torch.maximum(
                            pre_support,
                            endpoint_support,
                        )
                    if pre_support is not None:
                        pre_support = pre_support * pre_phase
                        support = pre_support if support is None else torch.maximum(
                            support,
                            pre_support,
                        )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if post_support_gates:
                        post_support = post_support_gates[0]
                        for gate in post_support_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype) * post_phase
                        support = post_support if support is None else torch.maximum(
                            support,
                            post_support,
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_pre_wake_endpoint_or_post_wake_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                if path_pre_arrival is not None:
                    pre_phase = (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    pre_support = None
                    pre_support_gates = [
                        gate for gate in (path_pre_wake, path_wake)
                        if gate is not None
                    ]
                    if pre_support_gates:
                        pre_support = pre_support_gates[0]
                        for gate in pre_support_gates[1:]:
                            pre_support = torch.maximum(pre_support, gate)
                        pre_support = (
                            pre_support >= self.laser_residual_wake_support_threshold
                        ).to(dtype=dtype)
                    if endpoint is not None:
                        endpoint_support = (
                            endpoint >= self.laser_residual_pre_endpoint_support_threshold
                        ).to(dtype=dtype)
                        pre_support = endpoint_support if pre_support is None else torch.maximum(
                            pre_support,
                            endpoint_support,
                        )
                    if pre_support is not None:
                        pre_support = pre_support * pre_phase
                        support = pre_support if support is None else torch.maximum(
                            support,
                            pre_support,
                        )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support = None
                    post_wake_gates = [
                        gate for gate in (path_post_wake, path_wake)
                        if gate is not None
                    ]
                    if post_wake_gates:
                        post_support = post_wake_gates[0]
                        for gate in post_wake_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_wake_support_threshold
                        ).to(dtype=dtype)
                    other_post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if other_post_support_gates:
                        other_post_support = other_post_support_gates[0]
                        for gate in other_post_support_gates[1:]:
                            other_post_support = torch.maximum(other_post_support, gate)
                        other_post_support = (
                            other_post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype)
                        post_support = other_post_support if post_support is None else torch.maximum(
                            post_support,
                            other_post_support,
                        )
                    if post_support is not None:
                        post_support = post_support * post_phase
                        support = post_support if support is None else torch.maximum(
                            support,
                            post_support,
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "high_phase_with_late_endpoint_or_pre_wake_or_post_wake_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                late_endpoint_window = None
                if (
                        path_program_track is not None
                        and self.laser_residual_endpoint_program_track_threshold > 0.0):
                    late_endpoint_window = (
                        path_program_track
                        >= self.laser_residual_endpoint_program_track_threshold
                    ).to(dtype=dtype)
                if (
                        path_elapsed is not None
                        and self.laser_residual_endpoint_path_elapsed_threshold > 0.0):
                    elapsed_window = (
                        path_elapsed
                        >= self.laser_residual_endpoint_path_elapsed_threshold
                    ).to(dtype=dtype)
                    late_endpoint_window = (
                        elapsed_window
                        if late_endpoint_window is None
                        else torch.maximum(late_endpoint_window, elapsed_window)
                    )

                support = None
                if path_pre_arrival is not None:
                    pre_phase = (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    pre_support = None
                    pre_support_gates = [
                        gate for gate in (path_pre_wake, path_wake)
                        if gate is not None
                    ]
                    if pre_support_gates:
                        pre_support = pre_support_gates[0]
                        for gate in pre_support_gates[1:]:
                            pre_support = torch.maximum(pre_support, gate)
                        pre_support = (
                            pre_support >= self.laser_residual_wake_support_threshold
                        ).to(dtype=dtype)
                    if endpoint is not None:
                        endpoint_support = (
                            endpoint >= self.laser_residual_pre_endpoint_support_threshold
                        ).to(dtype=dtype)
                        if late_endpoint_window is not None:
                            endpoint_support = endpoint_support * late_endpoint_window
                        pre_support = endpoint_support if pre_support is None else torch.maximum(
                            pre_support,
                            endpoint_support,
                        )
                    if pre_support is not None:
                        pre_support = pre_support * pre_phase
                        support = pre_support if support is None else torch.maximum(
                            support,
                            pre_support,
                        )

                if path_post_arrival is not None:
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    post_support = None
                    post_wake_gates = [
                        gate for gate in (path_post_wake, path_wake)
                        if gate is not None
                    ]
                    if post_wake_gates:
                        post_support = post_wake_gates[0]
                        for gate in post_wake_gates[1:]:
                            post_support = torch.maximum(post_support, gate)
                        post_support = (
                            post_support >= self.laser_residual_wake_support_threshold
                        ).to(dtype=dtype)
                    other_post_support_gates = [
                        gate for gate in (sweep, neighbor)
                        if gate is not None
                    ]
                    if other_post_support_gates:
                        other_post_support = other_post_support_gates[0]
                        for gate in other_post_support_gates[1:]:
                            other_post_support = torch.maximum(other_post_support, gate)
                        other_post_support = (
                            other_post_support >= self.laser_residual_post_support_threshold
                        ).to(dtype=dtype)
                        post_support = other_post_support if post_support is None else torch.maximum(
                            post_support,
                            other_post_support,
                        )
                    if post_support is not None:
                        post_support = post_support * post_phase
                        support = post_support if support is None else torch.maximum(
                            support,
                            post_support,
                        )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "xml_post_scan_memory":
            if path_elapsed is not None and path_body is not None:
                if self.laser_residual_endpoint_path_elapsed_threshold > 0.0:
                    elapsed_support = (
                        path_elapsed >= self.laser_residual_endpoint_path_elapsed_threshold
                    ).to(dtype=dtype)
                else:
                    elapsed_support = path_elapsed

                if self.laser_residual_prior_support_threshold > 0.0:
                    body_support = path_body * (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                else:
                    body_support = path_body

                track_support = None
                if path_program_track is not None:
                    if self.laser_residual_endpoint_program_track_threshold > 0.0:
                        track_support = path_program_track * (
                            path_program_track
                            >= self.laser_residual_endpoint_program_track_threshold
                        ).to(dtype=dtype)
                    else:
                        track_support = path_program_track
                if process is not None:
                    if self.laser_residual_active_support_threshold > 0.0:
                        process_support = process * (
                            process >= self.laser_residual_active_support_threshold
                        ).to(dtype=dtype)
                    else:
                        process_support = process
                    track_support = (
                        process_support
                        if track_support is None
                        else torch.maximum(track_support, process_support)
                    )

                if track_support is None:
                    selected = torch.zeros_like(path_elapsed)
                else:
                    selected = (
                        2.0 * elapsed_support * body_support * track_support
                    ).clamp(0.0, 1.0)
        elif source == "xml_future_arrival_track_support":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_active)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)

                high_phase = (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                support = None
                immediate_gates = [
                    (target_heat, self.laser_residual_prior_support_threshold),
                    (sweep, self.laser_residual_pre_sweep_support_threshold),
                    (endpoint, self.laser_residual_pre_endpoint_support_threshold),
                ]
                for gate, threshold in immediate_gates:
                    if gate is None:
                        continue
                    gate_support = (gate >= threshold).to(dtype=dtype)
                    support = gate_support if support is None else torch.maximum(
                        support,
                        gate_support,
                    )

                if path_body is not None and path_active is not None:
                    active_body_base = (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                    active_body_base = active_body_base * (
                        path_active >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)

                    active_body_support = None
                    if neighbor is not None:
                        neighbor_supported = active_body_base * (
                            neighbor >= self.laser_residual_pre_sweep_support_threshold
                        ).to(dtype=dtype)
                        active_body_support = neighbor_supported

                    if (
                            path_program_track is not None
                            and self.laser_residual_endpoint_program_track_threshold > 0.0):
                        track_supported = active_body_base * (
                            path_program_track
                            >= self.laser_residual_endpoint_program_track_threshold
                        ).to(dtype=dtype)
                        active_body_support = (
                            track_supported
                            if active_body_support is None
                            else torch.maximum(active_body_support, track_supported)
                        )
                    elif active_body_support is None:
                        active_body_support = active_body_base

                    support = (
                        active_body_support
                        if support is None
                        else torch.maximum(support, active_body_support)
                    )

                if support is None:
                    selected = torch.zeros_like(phase)
                else:
                    selected = phase * high_phase * support
        elif source == "xml_future_arrival_track_product":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_active)
                if gate is not None
            ]
            track_gates = [
                gate for gate in (target_heat, sweep, endpoint, path_program_track)
                if gate is not None
            ]
            if phase_gates and path_body is not None and track_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                phase = phase * (
                    phase >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                body = path_body * (
                    path_body >= self.laser_residual_prior_support_threshold
                ).to(dtype=dtype)

                track = track_gates[0]
                for gate in track_gates[1:]:
                    track = torch.maximum(track, gate)
                track = track * (
                    track >= self.laser_residual_endpoint_program_track_threshold
                ).to(dtype=dtype)

                selected = (4.0 * phase * body * track).clamp(0.0, 1.0)
        elif source == "xml_prearrival_heat_track_product":
            track_gates = [
                (gate, threshold)
                for gate, threshold in (
                    (target_heat, self.laser_residual_prior_support_threshold),
                    (sweep, self.laser_residual_pre_sweep_support_threshold),
                    (endpoint, self.laser_residual_pre_endpoint_support_threshold),
                )
                if gate is not None
            ]
            if path_pre_arrival is not None and path_body is not None and track_gates:
                phase = path_pre_arrival * (
                    path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                ).to(dtype=dtype)

                body = path_body * (
                    path_body >= self.laser_residual_prior_support_threshold
                ).to(dtype=dtype)

                track, threshold = track_gates[0]
                track = track * (track >= threshold).to(dtype=dtype)
                for gate, threshold in track_gates[1:]:
                    gated = gate * (gate >= threshold).to(dtype=dtype)
                    track = torch.maximum(track, gated)

                selected = (4.0 * phase * body * track).clamp(0.0, 1.0)
        elif source == "xml_prearrival_strong_support_product":
            if path_pre_arrival is not None:
                strong_support = self._select_laser_residual_strong_support(
                    laser_residual_prior_gate=None,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=dtype,
                )
                if strong_support is None:
                    selected = torch.zeros_like(path_pre_arrival)
                else:
                    phase = path_pre_arrival * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    selected = phase * strong_support.to(dtype=dtype).clamp(0.0, 1.0)
        elif source == "xml_phase_strong_support_product":
            phase_gates = [
                gate for gate in (path_pre_arrival, path_post_arrival)
                if gate is not None
            ]
            if phase_gates:
                phase = phase_gates[0]
                for gate in phase_gates[1:]:
                    phase = torch.maximum(phase, gate)
                strong_support = self._select_laser_residual_strong_support(
                    laser_residual_prior_gate=None,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=dtype,
                )
                if strong_support is None:
                    selected = torch.zeros_like(phase)
                else:
                    pre_phase = (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype) if path_pre_arrival is not None else None
                    post_phase = (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype) if path_post_arrival is not None else None
                    phase_support = pre_phase
                    if post_phase is not None:
                        phase_support = post_phase if phase_support is None else torch.maximum(
                            phase_support,
                            post_phase,
                        )
                    if phase_support is None:
                        selected = torch.zeros_like(phase)
                    else:
                        selected = phase * phase_support * strong_support.to(dtype=dtype).clamp(0.0, 1.0)
        elif source == "xml_prearrival_plus_strict_post_rescue_product":
            selected_parts = []
            strict_post_token = "postarrival_strict_wake_body_track_cold_all"
            post_tokens = {strict_post_token, "postarrival_wake_body_track_cold_all"}
            pre_sources = ",".join(
                token.strip()
                for token in self.laser_residual_strong_support_sources.replace("+", ",").split(",")
                if token.strip() and token.strip() not in post_tokens
            )
            if path_pre_arrival is not None and pre_sources:
                strong_support = self._select_laser_residual_strong_support(
                    laser_residual_prior_gate=None,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=dtype,
                    source_tokens_override=pre_sources,
                )
                if strong_support is not None:
                    phase = path_pre_arrival * (
                        path_pre_arrival >= self.laser_residual_body_prearrival_threshold
                    ).to(dtype=dtype)
                    selected_parts.append(
                        phase * strong_support.to(dtype=dtype).clamp(0.0, 1.0)
                    )
            if path_post_arrival is not None:
                post_support = self._select_laser_residual_strong_support(
                    laser_residual_prior_gate=None,
                    laser_path_active_gate=laser_path_active_gate,
                    laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
                    laser_path_post_arrival_gate=laser_path_post_arrival_gate,
                    laser_endpoint_gate=laser_endpoint_gate,
                    laser_target_heat_gate=laser_target_heat_gate,
                    laser_body_heat_gate=laser_body_heat_gate,
                    laser_path_body_heat_gate=laser_path_body_heat_gate,
                    laser_path_wake_gate=laser_path_wake_gate,
                    laser_path_program_track_gate=laser_path_program_track_gate,
                    laser_path_time_until_gate=laser_path_time_until_gate,
                    laser_sweep_heat_gate=laser_sweep_heat_gate,
                    neighbor_hot_gate=neighbor_hot_gate,
                    cold_to_hot_gate=cold_to_hot_gate,
                    dtype=dtype,
                    source_tokens_override=strict_post_token,
                )
                if post_support is not None:
                    phase = path_post_arrival * (
                        path_post_arrival >= self.laser_residual_post_phase_threshold
                    ).to(dtype=dtype)
                    selected_parts.append(
                        phase * post_support.to(dtype=dtype).clamp(0.0, 1.0)
                    )
            if selected_parts:
                selected = selected_parts[0]
                for gate in selected_parts[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "xml_recent_scan_memory":
            if (
                    path_elapsed is not None
                    and path_body is not None
                    and path_scanned is not None
                    and path_cooling_tail is not None):
                if self.laser_residual_endpoint_path_elapsed_threshold > 0.0:
                    elapsed_support = (
                        path_elapsed >= self.laser_residual_endpoint_path_elapsed_threshold
                    ).to(dtype=dtype)
                else:
                    elapsed_support = path_elapsed

                if self.laser_residual_prior_support_threshold > 0.0:
                    body_support = path_body * (
                        path_body >= self.laser_residual_prior_support_threshold
                    ).to(dtype=dtype)
                else:
                    body_support = path_body

                recent_support = path_scanned * path_cooling_tail
                if self.laser_residual_active_support_threshold > 0.0:
                    recent_support = recent_support * (
                        path_cooling_tail >= self.laser_residual_active_support_threshold
                    ).to(dtype=dtype)

                selected = (
                    2.0 * elapsed_support * body_support * recent_support
                ).clamp(0.0, 1.0)
        elif source == "pre_arrival_endpoint_sweep":
            gates = [
                gate for gate in (path_pre_arrival, path_active, endpoint, sweep)
                if gate is not None
            ]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "pre_arrival_endpoint_sweep_or_arrival":
            gates = [
                gate for gate in (path_pre_arrival, path_active, endpoint, sweep, arrival)
                if gate is not None
            ]
            if gates:
                selected = gates[0]
                for gate in gates[1:]:
                    selected = torch.maximum(selected, gate)
        elif source == "arrival_neighbor_product":
            if arrival is not None and neighbor is not None:
                selected = arrival * neighbor
        elif source == "arrival_with_neighbor_support":
            if arrival is not None:
                if neighbor is None:
                    selected = arrival
                else:
                    support = (0.35 + 0.45 * neighbor).clamp(0.0, 1.0)
                    selected = arrival * support
        elif source == "path_active_with_neighbor_support":
            if path_active is not None:
                if neighbor is None:
                    selected = path_active
                else:
                    support = (0.35 + 0.45 * neighbor).clamp(0.0, 1.0)
                    selected = path_active * support
        elif source == "path_active_neighbor_product":
            if path_active is not None and neighbor is not None:
                selected = path_active * neighbor

        if selected is None:
            return None
        if self.laser_residual_prior_gate_threshold > 0.0:
            if self.laser_residual_prior_gate_hard_threshold:
                selected = (
                    selected >= self.laser_residual_prior_gate_threshold
                ).to(dtype=dtype)
            else:
                denom = max(1.0 - self.laser_residual_prior_gate_threshold, 1.0e-6)
                selected = (
                    (selected - self.laser_residual_prior_gate_threshold) / denom
                ).clamp(0.0, 1.0)
        if abs(self.laser_residual_prior_gate_power - 1.0) > 1.0e-6:
            selected = self._apply_gate_power(
                selected,
                self.laser_residual_prior_gate_power,
            )
        return selected.clamp(0.0, 1.0)

    def _apply_tiered_laser_residual_min_delta(
            self,
            *,
            laser_residual_delta: torch.Tensor,
            laser_residual_prior_gate: torch.Tensor | None,
            laser_path_active_gate: torch.Tensor | None,
            laser_path_pre_arrival_gate: torch.Tensor | None,
            laser_path_post_arrival_gate: torch.Tensor | None,
            laser_endpoint_gate: torch.Tensor | None,
            laser_target_heat_gate: torch.Tensor | None,
            laser_body_heat_gate: torch.Tensor | None,
            laser_path_body_heat_gate: torch.Tensor | None,
            laser_path_wake_gate: torch.Tensor | None,
            laser_path_program_track_gate: torch.Tensor | None,
            laser_path_time_until_gate: torch.Tensor | None,
            laser_sweep_heat_gate: torch.Tensor | None,
            neighbor_hot_gate: torch.Tensor | None,
            cold_to_hot_gate: torch.Tensor | None,
            dtype: torch.dtype) -> torch.Tensor:
        strong_floor = min(self.laser_residual_min_delta, self.laser_residual_max_delta)
        weak_floor = min(self.laser_residual_weak_min_delta, self.laser_residual_max_delta)
        if strong_floor <= 0.0 and weak_floor <= 0.0:
            return laser_residual_delta

        strong_support = self._select_laser_residual_strong_support(
            laser_residual_prior_gate=laser_residual_prior_gate,
            laser_path_active_gate=laser_path_active_gate,
            laser_path_pre_arrival_gate=laser_path_pre_arrival_gate,
            laser_path_post_arrival_gate=laser_path_post_arrival_gate,
            laser_endpoint_gate=laser_endpoint_gate,
            laser_target_heat_gate=laser_target_heat_gate,
            laser_body_heat_gate=laser_body_heat_gate,
            laser_path_body_heat_gate=laser_path_body_heat_gate,
            laser_path_wake_gate=laser_path_wake_gate,
            laser_path_program_track_gate=laser_path_program_track_gate,
            laser_path_time_until_gate=laser_path_time_until_gate,
            laser_sweep_heat_gate=laser_sweep_heat_gate,
            neighbor_hot_gate=neighbor_hot_gate,
            cold_to_hot_gate=cold_to_hot_gate,
            dtype=dtype,
        )
        if strong_support is None:
            tiered_floor = torch.full_like(laser_residual_delta, weak_floor)
        else:
            strong_mask = (
                strong_support >= self.laser_residual_strong_support_threshold
            ).to(dtype=laser_residual_delta.dtype)
            weak = torch.full_like(laser_residual_delta, weak_floor)
            strong = torch.full_like(laser_residual_delta, strong_floor)
            tiered_floor = weak * (1.0 - strong_mask) + strong * strong_mask
        return torch.maximum(laser_residual_delta, tiered_floor)

    def _select_laser_residual_strong_support(
            self,
            *,
            laser_residual_prior_gate: torch.Tensor | None,
            laser_path_active_gate: torch.Tensor | None,
            laser_path_pre_arrival_gate: torch.Tensor | None,
            laser_path_post_arrival_gate: torch.Tensor | None,
            laser_endpoint_gate: torch.Tensor | None,
            laser_target_heat_gate: torch.Tensor | None,
            laser_body_heat_gate: torch.Tensor | None,
            laser_path_body_heat_gate: torch.Tensor | None,
            laser_path_wake_gate: torch.Tensor | None,
            laser_path_program_track_gate: torch.Tensor | None,
            laser_path_time_until_gate: torch.Tensor | None,
            laser_sweep_heat_gate: torch.Tensor | None,
            neighbor_hot_gate: torch.Tensor | None,
            cold_to_hot_gate: torch.Tensor | None,
            dtype: torch.dtype,
            source_tokens_override: str | None = None) -> torch.Tensor | None:
        source_text = (
            self.laser_residual_strong_support_sources
            if source_tokens_override is None else source_tokens_override
        )
        source_tokens = {
            token.strip()
            for token in source_text.replace("+", ",").split(",")
            if token.strip()
        }
        aliases = {
            "track_heat": {"target_heat", "sweep", "endpoint", "neighbor"},
            "track": {"sweep", "endpoint"},
            "all": {
                "target_heat", "body", "path_body", "sweep", "endpoint",
                "neighbor", "path_active", "residual_prior", "wake",
                "program_track", "future_track_all", "prearrival_heat_track_all",
                "prearrival_endpoint_track_all", "prearrival_endpoint_track_cold_all",
                "prearrival_endpoint_program_track_all",
                "prearrival_endpoint_time_until_all",
                "prearrival_endpoint_time_until_track_all",
                "prearrival_endpoint_time_until_high_track_all",
                "prearrival_endpoint_time_until_track_cold_all",
                "prearrival_wake_neighbor_track_all",
                "prearrival_wake_body_track_all",
                "prearrival_wake_body_track_cold_all",
                "prearrival_sweep_body_arrival_cold_track_all",
                "postarrival_wake_body_track_cold_all",
                "postarrival_strict_wake_body_track_cold_all",
            },
            "prior": {"residual_prior"},
            "xml_prior": {"residual_prior"},
        }
        expanded_tokens = set()
        for token in source_tokens:
            expanded_tokens.update(aliases.get(token, {token}))

        candidates = {
            "residual_prior": laser_residual_prior_gate,
            "target_heat": laser_target_heat_gate,
            "body": laser_body_heat_gate,
            "path_body": laser_path_body_heat_gate,
            "wake": laser_path_wake_gate,
            "program_track": laser_path_program_track_gate,
            "sweep": laser_sweep_heat_gate,
            "endpoint": laser_endpoint_gate,
            "neighbor": neighbor_hot_gate,
            "path_active": laser_path_active_gate,
        }
        gates = [
            gate.to(dtype=dtype).clamp(0.0, 1.0)
            for name, gate in candidates.items()
            if name in expanded_tokens and gate is not None
        ]
        if "future_track_all" in expanded_tokens:
            strict_support = None
            if laser_residual_prior_gate is not None and laser_path_body_heat_gate is not None:
                prior = laser_residual_prior_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                track_parts = []
                for gate, threshold in (
                        (laser_endpoint_gate, self.laser_residual_pre_endpoint_support_threshold),
                        (laser_sweep_heat_gate, self.laser_residual_pre_sweep_support_threshold),
                        (laser_target_heat_gate, self.laser_residual_prior_support_threshold),
                        (laser_path_program_track_gate, self.laser_residual_endpoint_program_track_threshold),
                ):
                    if gate is None:
                        continue
                    track_parts.append(
                        (gate.to(dtype=dtype).clamp(0.0, 1.0) >= threshold).to(dtype=dtype)
                    )
                if track_parts:
                    track = track_parts[0]
                    for gate in track_parts[1:]:
                        track = torch.maximum(track, gate)
                    strict_support = (
                        (prior >= self.laser_residual_prior_bypass_cold_start_threshold).to(dtype=dtype)
                        * (body >= self.laser_residual_prior_support_threshold).to(dtype=dtype)
                        * track
                    )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_heat_track_all" in expanded_tokens:
            strict_support = None
            if laser_path_pre_arrival_gate is not None and laser_path_body_heat_gate is not None:
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                track_parts = []
                for gate, threshold in (
                        (laser_endpoint_gate, self.laser_residual_pre_endpoint_support_threshold),
                        (laser_sweep_heat_gate, self.laser_residual_pre_sweep_support_threshold),
                        (laser_target_heat_gate, self.laser_residual_prior_support_threshold),
                ):
                    if gate is None:
                        continue
                    track_parts.append(
                        (gate.to(dtype=dtype).clamp(0.0, 1.0) >= threshold).to(dtype=dtype)
                    )
                if track_parts:
                    track = track_parts[0]
                    for gate in track_parts[1:]:
                        track = torch.maximum(track, gate)
                    strict_support = (
                        (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                        * (body >= self.laser_residual_prior_support_threshold).to(dtype=dtype)
                        * track
                    )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_sweep_body_arrival_cold_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_path_active_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_sweep_heat_gate is not None
                    and laser_path_program_track_gate is not None
                    and cold_to_hot_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                active = laser_path_active_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                sweep = laser_sweep_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                cold = cold_to_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (active >= self.laser_residual_sweep_arrival_support_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_sweep_body_support_threshold).to(dtype=dtype)
                    * (sweep >= self.laser_residual_pre_sweep_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_sweep_program_track_threshold).to(dtype=dtype)
                    * (cold >= self.laser_residual_sweep_cold_to_hot_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_program_track_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_prior_support_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_program_track_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_track_cold_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_program_track_gate is not None
                    and cold_to_hot_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                cold = cold_to_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_program_track_threshold).to(dtype=dtype)
                    * (cold >= self.laser_residual_endpoint_cold_to_hot_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_program_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_program_track_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_program_track_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_time_until_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_time_until_gate is not None
                    and self.laser_residual_time_until_threshold > 0.0):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                time_until = laser_path_time_until_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (time_until <= self.laser_residual_time_until_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_time_until_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_time_until_gate is not None
                    and laser_path_program_track_gate is not None
                    and self.laser_residual_time_until_threshold > 0.0):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                time_until = laser_path_time_until_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (time_until <= self.laser_residual_time_until_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_program_track_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_time_until_high_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_time_until_gate is not None
                    and laser_path_program_track_gate is not None
                    and self.laser_residual_time_until_threshold > 0.0):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                time_until = laser_path_time_until_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (time_until <= self.laser_residual_time_until_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_high_program_track_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_endpoint_time_until_track_cold_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_endpoint_gate is not None
                    and laser_path_time_until_gate is not None
                    and laser_path_program_track_gate is not None
                    and cold_to_hot_gate is not None
                    and self.laser_residual_time_until_threshold > 0.0):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                endpoint = laser_endpoint_gate.to(dtype=dtype).clamp(0.0, 1.0)
                time_until = laser_path_time_until_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                cold = cold_to_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (endpoint >= self.laser_residual_pre_endpoint_support_threshold).to(dtype=dtype)
                    * (time_until <= self.laser_residual_time_until_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_program_track_threshold).to(dtype=dtype)
                    * (cold >= self.laser_residual_endpoint_cold_to_hot_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_wake_neighbor_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_path_wake_gate is not None
                    and neighbor_hot_gate is not None
                    and laser_path_program_track_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                wake = laser_path_wake_gate.to(dtype=dtype).clamp(0.0, 1.0)
                neighbor = neighbor_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_prior_support_threshold).to(dtype=dtype)
                    * (wake >= self.laser_residual_wake_support_threshold).to(dtype=dtype)
                    * (neighbor >= self.laser_residual_pre_sweep_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_endpoint_program_track_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_wake_body_track_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_path_wake_gate is not None
                    and laser_path_program_track_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                wake = laser_path_wake_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_wake_body_support_threshold).to(dtype=dtype)
                    * (wake >= self.laser_residual_wake_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_wake_body_program_track_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "prearrival_wake_body_track_cold_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_pre_arrival_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_path_wake_gate is not None
                    and laser_path_program_track_gate is not None
                    and cold_to_hot_gate is not None):
                pre = laser_path_pre_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                wake = laser_path_wake_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                cold = cold_to_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (pre >= self.laser_residual_body_prearrival_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_wake_body_support_threshold).to(dtype=dtype)
                    * (wake >= self.laser_residual_wake_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_wake_body_cold_program_track_threshold).to(dtype=dtype)
                    * (cold >= self.laser_residual_wake_body_cold_to_hot_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "postarrival_wake_body_track_cold_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_post_arrival_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_path_wake_gate is not None
                    and laser_path_program_track_gate is not None
                    and cold_to_hot_gate is not None):
                post = laser_path_post_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                wake = laser_path_wake_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                cold = cold_to_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (post >= self.laser_residual_post_phase_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_wake_body_support_threshold).to(dtype=dtype)
                    * (wake >= self.laser_residual_wake_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_wake_body_cold_program_track_threshold).to(dtype=dtype)
                    * (cold >= self.laser_residual_wake_body_cold_to_hot_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if "postarrival_strict_wake_body_track_cold_all" in expanded_tokens:
            strict_support = None
            if (
                    laser_path_post_arrival_gate is not None
                    and laser_path_body_heat_gate is not None
                    and laser_path_wake_gate is not None
                    and laser_path_program_track_gate is not None
                    and cold_to_hot_gate is not None):
                post = laser_path_post_arrival_gate.to(dtype=dtype).clamp(0.0, 1.0)
                body = laser_path_body_heat_gate.to(dtype=dtype).clamp(0.0, 1.0)
                wake = laser_path_wake_gate.to(dtype=dtype).clamp(0.0, 1.0)
                program_track = laser_path_program_track_gate.to(dtype=dtype).clamp(0.0, 1.0)
                cold = cold_to_hot_gate.to(dtype=dtype).clamp(0.0, 1.0)
                strict_support = (
                    (post >= self.laser_residual_post_phase_threshold).to(dtype=dtype)
                    * (body >= self.laser_residual_post_wake_body_support_threshold).to(dtype=dtype)
                    * (wake >= self.laser_residual_post_wake_support_threshold).to(dtype=dtype)
                    * (program_track >= self.laser_residual_post_wake_body_program_track_threshold).to(dtype=dtype)
                    * (cold >= self.laser_residual_post_wake_body_cold_to_hot_threshold).to(dtype=dtype)
                )
            if strict_support is not None:
                gates.append(strict_support)
        if not gates:
            return None
        support = gates[0]
        for gate in gates[1:]:
            support = torch.maximum(support, gate)
        return support.clamp(0.0, 1.0)

    @staticmethod
    def _input_window_temperature_max(graph_sequence, device, dtype):
        frames = []
        for data in graph_sequence:
            temp = data.y
            if temp.ndim > 1:
                temp = temp[..., 0]
            frames.append(temp.to(device=device, dtype=dtype).reshape(-1))
        return torch.stack(frames, dim=0).max(dim=0).values

    def _mix_laser_residual_control_gate(
            self,
            *,
            learned_gate: torch.Tensor,
            cold_to_hot_gate: torch.Tensor | None,
            ) -> torch.Tensor:
        cold_gate = cold_to_hot_gate
        if self.laser_residual_gate_mix == "prior_only":
            return torch.ones_like(learned_gate)
        if cold_gate is None:
            return learned_gate
        cold_gate = cold_gate.to(device=learned_gate.device, dtype=learned_gate.dtype)
        if self.laser_residual_gate_mix == "cold_to_hot":
            return cold_gate
        if self.laser_residual_gate_mix == "max_learned_cold_to_hot":
            return torch.maximum(learned_gate, cold_gate)
        if self.laser_residual_gate_mix == "learned_times_cold_to_hot":
            return learned_gate * cold_gate
        if self.laser_residual_gate_mix == "calibrated_cold_to_hot":
            cold_gate = self._calibrated_cold_to_hot_gate(cold_gate)
            if self.laser_residual_cold_to_hot_gate_floor > 0.0:
                floor = torch.full_like(
                    cold_gate,
                    self.laser_residual_cold_to_hot_gate_floor,
                )
                cold_gate = torch.maximum(cold_gate, floor)
            return cold_gate
        return learned_gate

    def _calibrated_cold_to_hot_gate(self, gate: torch.Tensor) -> torch.Tensor:
        threshold = self.laser_residual_cold_to_hot_gate_threshold
        if threshold > 0.0:
            gate = ((gate - threshold) / max(1.0 - threshold, 1.0e-6)).clamp(
                0.0, 1.0
            )
        else:
            gate = gate.clamp(0.0, 1.0)
        return self._apply_gate_power(
            gate,
            self.laser_residual_cold_to_hot_gate_power,
        ).clamp(0.0, 1.0)

    @staticmethod
    def _apply_gate_power(gate: torch.Tensor, power: float) -> torch.Tensor:
        gate_for_power = gate.clamp(0.0, 1.0)
        if power < 1.0:
            positive = gate_for_power > 0.0
            powered = gate_for_power.clamp_min(1.0e-6).pow(power)
            return torch.where(positive, powered, torch.zeros_like(powered))
        return gate_for_power.pow(power)

    def set_training_epoch(self, epoch: int) -> None:
        """Update epoch-dependent training controls."""
        warmup = self.hotspot_specialist_blend_warmup_epochs
        ramp = self.hotspot_specialist_blend_ramp_epochs
        if warmup <= 0 and ramp <= 0:
            self.hotspot_specialist_blend_weight = 1.0
            return
        if epoch <= warmup:
            self.hotspot_specialist_blend_weight = 0.0
        elif ramp <= 0:
            self.hotspot_specialist_blend_weight = 1.0
        else:
            self.hotspot_specialist_blend_weight = min(
                1.0, max(0.0, (epoch - warmup) / float(ramp))
            )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
