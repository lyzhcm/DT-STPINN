"""Configuration system for DT-STPINN.

Uses Python dataclasses with YAML marshalling for type-safe configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MaterialProps:
    name: str = "Ti-6Al-4V"
    density: float = 4430.0
    specific_heat: float = 526.3
    thermal_conductivity: float = 6.7
    thermal_expansion: float = 8.6e-6
    solidus_temp: float = 1878.0
    liquidus_temp: float = 1928.0
    latent_heat: float = 2.86e5
    emissivity: float = 0.35
    convection_coeff: float = 10.0
    ambient_temp: float = 293.15


@dataclass
class SpatialConfig:
    gnn_type: str = "GATv2Conv"
    num_layers: int = 2
    heads: int = 8
    dropout: float = 0.1
    use_checkpoint: bool = False


@dataclass
class TemporalConfig:
    num_layers: int = 4
    heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 64


@dataclass
class FusionConfig:
    type: str = "cross_attention"
    heads: int = 8


@dataclass
class DecoderConfig:
    temperature: bool = True
    stress: bool = False
    displacement: bool = False
    heat_flux: bool = False


@dataclass
class ModelConfig:
    node_feature_dim: int = 12
    edge_feature_dim: int = 5
    hidden_dim: int = 256
    predict_temperature_delta: bool = False
    enable_hotspot_head: bool = False
    enable_direct_laser_head: bool = False
    direct_laser_features_to_head: bool = True
    enable_hotspot_delta_head: bool = False
    enable_hotspot_specialist_head: bool = False
    hotspot_specialist_min_temp: float = 20.0
    hotspot_specialist_max_temp: float = 3500.0
    hotspot_specialist_gate_threshold: float = 0.5
    hotspot_specialist_process_gate_threshold: float = 0.0
    hotspot_specialist_process_gate_power: float = 1.0
    hotspot_specialist_process_gate_max: float = 1.0
    hotspot_specialist_gate_power: float = 1.0
    hotspot_specialist_use_cls_gate: bool = True
    hotspot_specialist_hard_process_gate: bool = False
    hotspot_specialist_neighbor_gate_threshold: float = 0.0
    hotspot_specialist_neighbor_gate_power: float = 1.0
    hotspot_specialist_neighbor_gate_max: float = 1.0
    hotspot_specialist_gate_merge: str = "max"
    hotspot_specialist_neighbor_support_base: float = 0.5
    hotspot_specialist_neighbor_support_weight: float = 0.5
    hotspot_specialist_init_temp: float = 0.0
    hotspot_specialist_blend_warmup_epochs: int = 0
    hotspot_specialist_blend_ramp_epochs: int = 0
    hotspot_specialist_detach_base: bool = False
    hotspot_specialist_hidden_dim: int = 64
    hotspot_specialist_num_layers: int = 1
    hotspot_specialist_output_mode: str = "absolute"
    hotspot_specialist_residual_max_delta: float = 3500.0
    enable_hotspot_laser_prior: bool = False
    hotspot_laser_prior_mode: str = "specialist_floor"
    hotspot_laser_prior_max_delta: float = 3300.0
    hotspot_laser_prior_gate_power: float = 1.0
    hotspot_laser_prior_require_neighbor: bool = True
    hotspot_laser_prior_raw_process_threshold: float = 0.0
    hotspot_laser_prior_gate_source: str = "process"
    hotspot_laser_prior_gate_threshold: float = 0.0
    hotspot_laser_prior_neighbor_threshold: float = 0.0
    enable_laser_residual_head: bool = False
    laser_residual_max_delta: float = 2200.0
    laser_residual_gate_source: str = "arrival"
    laser_residual_gate_mix: str = "learned"
    laser_residual_prior_gate_threshold: float = 0.0
    laser_residual_prior_gate_power: float = 1.0
    laser_residual_prior_gate_hard_threshold: bool = False
    laser_residual_prior_support_threshold: float = 0.0
    laser_residual_active_support_threshold: float = 0.0
    laser_residual_body_prearrival_threshold: float = 0.10
    laser_residual_post_phase_threshold: float = 0.0
    laser_residual_post_support_threshold: float = 0.0
    laser_residual_pre_sweep_support_threshold: float = 0.0
    laser_residual_sweep_body_support_threshold: float = 0.25
    laser_residual_sweep_arrival_support_threshold: float = 0.70
    laser_residual_sweep_cold_to_hot_threshold: float = 0.80
    laser_residual_sweep_program_track_threshold: float = 0.60
    laser_residual_pre_endpoint_support_threshold: float = 0.0
    laser_residual_endpoint_program_track_threshold: float = 0.0
    laser_residual_endpoint_high_program_track_threshold: float = 0.75
    laser_residual_endpoint_path_elapsed_threshold: float = 0.0
    laser_residual_time_until_threshold: float = 0.0
    laser_residual_wake_support_threshold: float = 0.08
    laser_residual_wake_body_support_threshold: float = 0.0
    laser_residual_wake_body_program_track_threshold: float = 0.0
    laser_residual_wake_body_cold_to_hot_threshold: float = 0.85
    laser_residual_wake_body_cold_program_track_threshold: float = 0.45
    laser_residual_endpoint_cold_to_hot_threshold: float = 0.0
    laser_residual_post_wake_body_support_threshold: float = 0.0
    laser_residual_post_wake_support_threshold: float = 0.0
    laser_residual_post_wake_body_program_track_threshold: float = 0.0
    laser_residual_post_wake_body_cold_to_hot_threshold: float = 0.0
    laser_residual_post_rescue_cap_margin: float = 0.0
    laser_residual_post_rescue_cap_threshold: float = 0.0
    laser_residual_pre_track_support_floor: float = 0.35
    laser_residual_path_body_endpoint_max: float = 1.0
    laser_residual_post_suppress_strength: float = 0.0
    laser_residual_control_gate_floor: float = 0.0
    laser_residual_cold_to_hot_gate_threshold: float = 0.0
    laser_residual_cold_to_hot_gate_power: float = 1.0
    laser_residual_cold_to_hot_gate_floor: float = 0.0
    laser_residual_cold_start_threshold: float = 0.0
    laser_residual_cold_start_softness: float = 0.0
    laser_residual_prior_bypass_cold_start: bool = False
    laser_residual_prior_bypass_cold_start_threshold: float = 0.5
    laser_residual_min_delta: float = 0.0
    laser_residual_use_tiered_min_delta: bool = False
    laser_residual_weak_min_delta: float = 0.0
    laser_residual_strong_support_threshold: float = 0.0
    laser_residual_strong_support_sources: str = "target_heat,sweep,endpoint,neighbor"
    laser_residual_hidden_dim: int = 128
    laser_residual_gate_bias: float = -4.0
    laser_residual_delta_init: float = 400.0
    enable_cold_to_hot_head: bool = False
    cold_to_hot_hidden_dim: int = 64
    cold_to_hot_gate_bias: float = -2.0
    post_endpoint_false_hot_guard_strength: float = 0.0
    post_endpoint_false_hot_guard_threshold: float = 0.0
    post_endpoint_false_hot_guard_power: float = 1.0
    post_endpoint_false_hot_guard_max: float = 1.0
    post_endpoint_false_hot_guard_require_no_prearrival: bool = True
    post_low_sweep_false_hot_guard_strength: float = 0.0
    post_low_sweep_false_hot_guard_threshold: float = 0.0
    post_low_sweep_false_hot_guard_power: float = 1.0
    post_low_sweep_false_hot_guard_max: float = 1.0
    post_mid_phase_false_hot_guard_strength: float = 0.0
    post_mid_phase_false_hot_post_min: float = 0.0
    post_mid_phase_false_hot_post_max: float = 1.0
    post_mid_phase_false_hot_guard_threshold: float = 0.0
    post_mid_phase_false_hot_guard_power: float = 1.0
    post_mid_phase_false_hot_guard_max: float = 1.0
    post_mid_phase_false_hot_endpoint_max: float = 0.05
    post_endpoint_false_hot_temperature_ceiling: float = 0.0
    post_endpoint_false_hot_hard_ceiling: bool = False
    post_endpoint_false_hot_hard_ceiling_threshold: float = 0.0
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class PhysicsConfig:
    heat_conduction: bool = True
    fourier_flux: bool = True
    boundary_convection: bool = True
    boundary_radiation: bool = True
    initial_condition: bool = True
    coordinate_scale_to_m: float = 1.0e-3
    time_scale_to_s: float = 1.0e-3
    normalize_pde_residual: bool = True
    pde_temperature_scale: float = 1000.0
    pde_time_scale: float = 1.0
    boundary_label_mode: str = "positive_dirichlet"


@dataclass
class LossConfig:
    lambda_T: float = 1.0
    lambda_PDE: float = 0.1
    lambda_BC: float = 0.1
    lambda_IC: float = 0.5
    lambda_smooth: float = 0.01
    # Hotspot re-weighting — penalises under-prediction in the melt pool.
    lambda_hot: float = 0.0          # 0.0 → hotspot term disabled.
    hot_threshold: float = 500.0     # °C, lower bound for weighting ramp.
    hot_weight: float = 5.0          # peak additive weight at liquidus.
    hot_weight_power: float = 2.0    # exponent on the temperature ratio.
    lambda_hot_cls: float = 0.0
    hot_cls_threshold: float = 0.0
    hot_cls_pos_weight: float = 0.0
    hot_cls_max_pos_weight: float = 2000.0
    lambda_hot_delta: float = 0.0
    hot_delta_threshold: float = 0.0
    hot_delta_scale: float = 1000.0
    lambda_hot_specialist: float = 0.0
    hot_specialist_threshold: float = 0.0
    hot_specialist_scale: float = 1000.0
    hot_specialist_peak_weight: float = 0.0
    hot_specialist_weight_power: float = 1.0
    hot_specialist_candidate_min_temp: float = 0.0
    hot_specialist_process_gate_threshold: float = 0.0
    hot_specialist_neighbor_gate_threshold: float = 0.0
    lambda_hot_process_specialist: float = 0.0
    hot_process_specialist_threshold: float = 0.0
    hot_process_specialist_scale: float = 1000.0
    hot_process_specialist_peak_weight: float = 0.0
    hot_process_specialist_weight_power: float = 1.0
    hot_process_specialist_gate_source: str = "arrival_or_process"
    hot_process_specialist_gate_threshold: float = 0.0
    hot_process_specialist_gate_power: float = 1.0
    hot_process_specialist_prev_max_temp: float = 0.0
    lambda_hot_final: float = 0.0
    hot_final_threshold: float = 0.0
    hot_final_scale: float = 1000.0
    hot_final_peak_weight: float = 0.0
    hot_final_weight_power: float = 1.0
    lambda_cold_false_hot: float = 0.0
    cold_false_hot_target_threshold: float = 100.0
    cold_false_hot_pred_threshold: float = 500.0
    cold_false_hot_scale: float = 1000.0
    cold_false_hot_gate_source: str = "specialist"
    cold_false_hot_gate_threshold: float = 0.1
    cold_false_hot_gate_power: float = 1.0
    lambda_specialist_false_hot: float = 0.0
    specialist_false_hot_target_threshold: float = 1200.0
    specialist_false_hot_pred_threshold: float = 1400.0
    specialist_false_hot_scale: float = 1000.0
    specialist_false_hot_gate_source: str = "specialist"
    specialist_false_hot_gate_threshold: float = 0.1
    specialist_false_hot_gate_power: float = 1.0
    lambda_laser_residual: float = 0.0
    laser_residual_threshold: float = 0.0
    laser_residual_scale: float = 1000.0
    laser_residual_peak_weight: float = 0.0
    laser_residual_weight_power: float = 1.0
    laser_residual_gate_source: str = "residual_prior"
    laser_residual_gate_threshold: float = 0.0
    laser_residual_gate_power: float = 1.0
    laser_residual_prev_max_temp: float = 0.0
    laser_residual_min_target_gap: float = 0.0
    lambda_laser_residual_delta: float = 0.0
    laser_residual_delta_scale: float = 1000.0
    laser_residual_delta_gate_threshold: float = 0.0
    laser_residual_delta_gate_power: float = 1.0
    laser_residual_delta_target_divide_gate: bool = True
    laser_residual_delta_target_max: float = 3600.0
    laser_residual_delta_min_support: float = 0.05
    laser_residual_delta_prev_max_temp: float = 0.0
    laser_residual_delta_min_target_gap: float = 0.0
    lambda_laser_residual_gate: float = 0.0
    laser_residual_gate_pos_threshold: float = 0.0
    laser_residual_gate_neg_threshold: float = 1200.0
    laser_residual_gate_support_threshold: float = 0.0
    laser_residual_gate_prev_max_temp: float = 0.0
    laser_residual_gate_min_target_gap: float = 0.0
    laser_residual_gate_pos_weight: float = 0.0
    laser_residual_gate_max_pos_weight: float = 2000.0
    lambda_cold_to_hot_cls: float = 0.0
    cold_to_hot_threshold: float = 0.0
    cold_to_hot_neg_threshold: float = 1200.0
    cold_to_hot_prev_max_temp: float = 120.0
    cold_to_hot_min_target_gap: float = 1200.0
    cold_to_hot_support_gate_source: str = "residual_prior"
    cold_to_hot_support_threshold: float = 0.0
    cold_to_hot_pos_weight: float = 0.0
    cold_to_hot_max_pos_weight: float = 2500.0
    lambda_final_T: float = 0.0
    hot_peak_ref_temp: float = 3000.0
    use_base_for_field_losses: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 1
    accumulate_grad_batches: int = 8
    epochs: int = 500
    lr: float = 1e-3
    lr_scheduler: str = "cosine"
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 50
    use_amp: bool = True
    amp_dtype: str = "auto"


@dataclass
class DataConfig:
    vtu_dir: str = "data/raw"
    time_sampling: str = "log"
    window_size: int = 16
    predict_steps: int = 1
    k_neighbors: int = 16
    use_mesh_edges: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    # Stratified window sampling (only affects training; val/test are untouched).
    stratified_sampling: bool = False
    stratified_normal: float = 0.5     # ratio: max T < 500 °C
    stratified_hot: float = 0.3        # ratio: max T ∈ [500, solidus) °C
    stratified_melting: float = 0.2    # ratio: max T ≥ solidus °C

    # Optional process lookahead features. These use known target-step laser
    # trajectory/geometry, not target temperature labels.
    # Set to E0/E1/E2/E3 to expand the documented minimal feature ablations.
    # Keep "manual" for legacy configs that set the individual flags below.
    laser_feature_group: str = "manual"
    use_target_laser_features: bool = False
    laser_feature_radius_mm: float = 0.4
    laser_feature_along_radius_mm: float = 0.0
    laser_feature_depth_mm: float = 0.1
    laser_feature_time_scale_to_s: float = 1.0e-3
    laser_feature_include_laser_coordinates: bool = False
    laser_feature_include_scan_geometry: bool = False
    laser_feature_include_sweep: bool = False
    laser_feature_include_exposure: bool = False
    laser_feature_exposure_source: str = "samples"
    laser_feature_exposure_gate: bool = True
    laser_feature_exposure_gate_mode: str = "integral"
    laser_feature_exposure_past_steps: int = 4
    laser_feature_exposure_future_steps: int = 0
    laser_feature_exposure_time_decay_s: float = 0.05
    laser_feature_exposure_use_segments: bool = False
    laser_feature_include_exposure_split: bool = False
    laser_feature_include_arrival_time: bool = False
    laser_feature_arrival_time_decay_s: float = 0.08
    laser_feature_include_neighbor_temp: bool = False
    laser_feature_include_neighbor_hot_stats: bool = False
    laser_feature_neighbor_hot_threshold: float = 1604.85
    laser_feature_include_neighbor_warm_stats: bool = False
    laser_feature_neighbor_warm_threshold: float = 1000.0
    laser_feature_include_body_source: bool = False
    laser_body_radius_mm: float = 0.1
    laser_body_height_mm: float = 0.1
    laser_body_radius_front_mm: float = 0.4
    laser_body_radius_back_mm: float = 0.4
    laser_body_coeff_front: float = 1.0
    laser_body_coeff_back: float = 1.0
    laser_feature_include_path_arrival: bool = False
    laser_feature_path_arrival_time_decay_s: float = 0.14
    laser_feature_path_arrival_gate_mode: str = "symmetric"
    laser_feature_path_arrival_neighbor_tracks: bool = False
    laser_feature_include_path_phase: bool = False
    laser_feature_path_phase_use_as_gate: bool = False
    laser_feature_include_path_coordinates: bool = False
    laser_feature_include_path_timing: bool = False
    laser_feature_include_path_body_support: bool = False
    laser_feature_path_body_use_as_gate: bool = False
    laser_feature_include_path_endpoint: bool = False
    laser_feature_endpoint_radius_mm: float = 0.75
    laser_feature_endpoint_time_decay_s: float = 0.075
    laser_feature_path_endpoint_use_as_gate: bool = False
    laser_feature_include_path_wake: bool = False
    laser_feature_wake_cross_radius_mm: float = 0.22
    laser_feature_wake_tail_decay_mm: float = 1.25
    laser_feature_wake_lead_decay_mm: float = 0.45
    laser_feature_wake_time_decay_s: float = 0.08
    laser_path_mode: str = "estimated"
    laser_path_time_scale_to_s: float = 1.0e-3
    laser_path_time_offset_s: float = 0.0
    laser_start_point_mm: list[float] = field(default_factory=lambda: [10.3165, -10.1562, 0.1])
    laser_scan_direction: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    laser_scan_length_mm: float = 20.3124
    laser_hatch_direction: list[float] = field(default_factory=lambda: [-1.0, 0.0, 0.0])
    laser_hatch_count: int = 104
    laser_hatch_spacing_mm: float = 0.2004
    laser_layer_count: int = 10
    laser_layer_thickness_mm: float = 0.1
    laser_velocity_mm_s: float = 600.0
    laser_each_path_time_s: float = 0.001
    laser_each_layer_time_s: float = 0.0
    laser_alternate_layer_scan_direction: bool = False
    laser_reverse_hatch_order_parity: int = -1

    def apply_laser_feature_group(self) -> None:
        """Expand E0/E1/E2/E3 into explicit feature flags.

        The grouped settings are intentionally narrow and deterministic so the
        next ablation family differs only by prescribed laser-path features.
        Legacy/manual configs keep their individual flag values untouched.
        """
        group = str(self.laser_feature_group or "manual").strip().lower()
        if group in {"", "manual", "custom"}:
            return
        if group not in {"e0", "e1", "e2", "e3"}:
            raise ValueError(
                "data.laser_feature_group must be one of: manual, E0, E1, E2, E3"
            )

        self.use_target_laser_features = group != "e0"
        self.laser_feature_include_laser_coordinates = group in {"e1", "e2", "e3"}
        self.laser_feature_include_scan_geometry = group in {"e2", "e3"}
        self.laser_feature_include_path_arrival = group in {"e2", "e3"}
        self.laser_feature_include_path_phase = group == "e3"

        self.laser_feature_include_sweep = False
        self.laser_feature_include_exposure = False
        self.laser_feature_include_exposure_split = False
        self.laser_feature_include_arrival_time = False
        self.laser_feature_include_neighbor_temp = False
        self.laser_feature_include_neighbor_hot_stats = False
        self.laser_feature_include_neighbor_warm_stats = False
        self.laser_feature_include_body_source = False
        self.laser_feature_include_path_coordinates = False
        self.laser_feature_include_path_timing = False
        self.laser_feature_include_path_body_support = False
        self.laser_feature_include_path_endpoint = False
        self.laser_feature_include_path_wake = False


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    experiment_name: str = "paper1_temperature"
    save_every: int = 50
    eval_every: int = 10
    save_hot_checkpoint: bool = True
    hot_checkpoint_metric: str = "TempF1AboveSolidus"
    hot_checkpoint_mode: str = "max"
    use_wandb: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    material: MaterialProps = field(default_factory=MaterialProps)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def _populate(section: str, dataclass_type):
            if section not in raw:
                return dataclass_type()
            d = {}
            raw_section = raw[section]
            for fld in dataclass_type.__dataclass_fields__:
                if fld in raw_section:
                    val = raw_section[fld]
                    field_type = dataclass_type.__dataclass_fields__[fld].type
                    if hasattr(field_type, "__dataclass_fields__"):
                        d[fld] = _populate(fld, field_type)
                    else:
                        d[fld] = val
            return dataclass_type(**d)

        model = ModelConfig(
            node_feature_dim=raw.get("model", {}).get("node_feature_dim", 12),
            edge_feature_dim=raw.get("model", {}).get("edge_feature_dim", 5),
            hidden_dim=raw.get("model", {}).get("hidden_dim", 256),
            predict_temperature_delta=raw.get("model", {}).get("predict_temperature_delta", False),
            enable_hotspot_head=raw.get("model", {}).get("enable_hotspot_head", False),
            enable_direct_laser_head=raw.get("model", {}).get("enable_direct_laser_head", False),
            direct_laser_features_to_head=raw.get("model", {}).get("direct_laser_features_to_head", True),
            enable_hotspot_delta_head=raw.get("model", {}).get("enable_hotspot_delta_head", False),
            enable_hotspot_specialist_head=raw.get("model", {}).get("enable_hotspot_specialist_head", False),
            hotspot_specialist_min_temp=raw.get("model", {}).get("hotspot_specialist_min_temp", 20.0),
            hotspot_specialist_max_temp=raw.get("model", {}).get("hotspot_specialist_max_temp", 3500.0),
            hotspot_specialist_gate_threshold=raw.get("model", {}).get("hotspot_specialist_gate_threshold", 0.5),
            hotspot_specialist_process_gate_threshold=raw.get("model", {}).get("hotspot_specialist_process_gate_threshold", 0.0),
            hotspot_specialist_process_gate_power=raw.get("model", {}).get("hotspot_specialist_process_gate_power", 1.0),
            hotspot_specialist_process_gate_max=raw.get("model", {}).get("hotspot_specialist_process_gate_max", 1.0),
            hotspot_specialist_gate_power=raw.get("model", {}).get("hotspot_specialist_gate_power", 1.0),
            hotspot_specialist_use_cls_gate=raw.get("model", {}).get("hotspot_specialist_use_cls_gate", True),
            hotspot_specialist_hard_process_gate=raw.get("model", {}).get("hotspot_specialist_hard_process_gate", False),
            hotspot_specialist_neighbor_gate_threshold=raw.get("model", {}).get("hotspot_specialist_neighbor_gate_threshold", 0.0),
            hotspot_specialist_neighbor_gate_power=raw.get("model", {}).get("hotspot_specialist_neighbor_gate_power", 1.0),
            hotspot_specialist_neighbor_gate_max=raw.get("model", {}).get("hotspot_specialist_neighbor_gate_max", 1.0),
            hotspot_specialist_gate_merge=raw.get("model", {}).get("hotspot_specialist_gate_merge", "max"),
            hotspot_specialist_neighbor_support_base=raw.get("model", {}).get("hotspot_specialist_neighbor_support_base", 0.5),
            hotspot_specialist_neighbor_support_weight=raw.get("model", {}).get("hotspot_specialist_neighbor_support_weight", 0.5),
            hotspot_specialist_init_temp=raw.get("model", {}).get("hotspot_specialist_init_temp", 0.0),
            hotspot_specialist_blend_warmup_epochs=raw.get("model", {}).get("hotspot_specialist_blend_warmup_epochs", 0),
            hotspot_specialist_blend_ramp_epochs=raw.get("model", {}).get("hotspot_specialist_blend_ramp_epochs", 0),
            hotspot_specialist_detach_base=raw.get("model", {}).get("hotspot_specialist_detach_base", False),
            hotspot_specialist_hidden_dim=raw.get("model", {}).get("hotspot_specialist_hidden_dim", 64),
            hotspot_specialist_num_layers=raw.get("model", {}).get("hotspot_specialist_num_layers", 1),
            hotspot_specialist_output_mode=raw.get("model", {}).get("hotspot_specialist_output_mode", "absolute"),
            hotspot_specialist_residual_max_delta=raw.get("model", {}).get("hotspot_specialist_residual_max_delta", 3500.0),
            enable_hotspot_laser_prior=raw.get("model", {}).get("enable_hotspot_laser_prior", False),
            hotspot_laser_prior_mode=raw.get("model", {}).get("hotspot_laser_prior_mode", "specialist_floor"),
            hotspot_laser_prior_max_delta=raw.get("model", {}).get("hotspot_laser_prior_max_delta", 3300.0),
            hotspot_laser_prior_gate_power=raw.get("model", {}).get("hotspot_laser_prior_gate_power", 1.0),
            hotspot_laser_prior_require_neighbor=raw.get("model", {}).get("hotspot_laser_prior_require_neighbor", True),
            hotspot_laser_prior_raw_process_threshold=raw.get("model", {}).get("hotspot_laser_prior_raw_process_threshold", 0.0),
            hotspot_laser_prior_gate_source=raw.get("model", {}).get("hotspot_laser_prior_gate_source", "process"),
            hotspot_laser_prior_gate_threshold=raw.get("model", {}).get("hotspot_laser_prior_gate_threshold", 0.0),
            hotspot_laser_prior_neighbor_threshold=raw.get("model", {}).get("hotspot_laser_prior_neighbor_threshold", 0.0),
            enable_laser_residual_head=raw.get("model", {}).get("enable_laser_residual_head", False),
            laser_residual_max_delta=raw.get("model", {}).get("laser_residual_max_delta", 2200.0),
            laser_residual_gate_source=raw.get("model", {}).get("laser_residual_gate_source", "arrival"),
            laser_residual_gate_mix=raw.get("model", {}).get("laser_residual_gate_mix", "learned"),
            laser_residual_prior_gate_threshold=raw.get("model", {}).get("laser_residual_prior_gate_threshold", 0.0),
            laser_residual_prior_gate_power=raw.get("model", {}).get("laser_residual_prior_gate_power", 1.0),
            laser_residual_prior_gate_hard_threshold=raw.get("model", {}).get("laser_residual_prior_gate_hard_threshold", False),
            laser_residual_prior_support_threshold=raw.get("model", {}).get("laser_residual_prior_support_threshold", 0.0),
            laser_residual_active_support_threshold=raw.get("model", {}).get("laser_residual_active_support_threshold", 0.0),
            laser_residual_body_prearrival_threshold=raw.get("model", {}).get("laser_residual_body_prearrival_threshold", 0.10),
            laser_residual_post_phase_threshold=raw.get("model", {}).get("laser_residual_post_phase_threshold", 0.0),
            laser_residual_post_support_threshold=raw.get("model", {}).get("laser_residual_post_support_threshold", 0.0),
            laser_residual_pre_sweep_support_threshold=raw.get("model", {}).get("laser_residual_pre_sweep_support_threshold", 0.0),
            laser_residual_sweep_body_support_threshold=raw.get("model", {}).get("laser_residual_sweep_body_support_threshold", 0.25),
            laser_residual_sweep_arrival_support_threshold=raw.get("model", {}).get("laser_residual_sweep_arrival_support_threshold", 0.70),
            laser_residual_sweep_cold_to_hot_threshold=raw.get("model", {}).get("laser_residual_sweep_cold_to_hot_threshold", 0.80),
            laser_residual_sweep_program_track_threshold=raw.get("model", {}).get("laser_residual_sweep_program_track_threshold", 0.60),
            laser_residual_pre_endpoint_support_threshold=raw.get("model", {}).get("laser_residual_pre_endpoint_support_threshold", 0.0),
            laser_residual_endpoint_program_track_threshold=raw.get("model", {}).get("laser_residual_endpoint_program_track_threshold", 0.0),
            laser_residual_endpoint_high_program_track_threshold=raw.get("model", {}).get("laser_residual_endpoint_high_program_track_threshold", 0.75),
            laser_residual_endpoint_path_elapsed_threshold=raw.get("model", {}).get("laser_residual_endpoint_path_elapsed_threshold", 0.0),
            laser_residual_time_until_threshold=raw.get("model", {}).get("laser_residual_time_until_threshold", 0.0),
            laser_residual_wake_support_threshold=raw.get("model", {}).get("laser_residual_wake_support_threshold", 0.08),
            laser_residual_wake_body_support_threshold=raw.get("model", {}).get("laser_residual_wake_body_support_threshold", 0.0),
            laser_residual_wake_body_program_track_threshold=raw.get("model", {}).get("laser_residual_wake_body_program_track_threshold", 0.0),
            laser_residual_wake_body_cold_to_hot_threshold=raw.get("model", {}).get("laser_residual_wake_body_cold_to_hot_threshold", 0.85),
            laser_residual_wake_body_cold_program_track_threshold=raw.get("model", {}).get("laser_residual_wake_body_cold_program_track_threshold", 0.45),
            laser_residual_endpoint_cold_to_hot_threshold=raw.get("model", {}).get("laser_residual_endpoint_cold_to_hot_threshold", 0.0),
            laser_residual_post_wake_body_support_threshold=raw.get("model", {}).get("laser_residual_post_wake_body_support_threshold", 0.0),
            laser_residual_post_wake_support_threshold=raw.get("model", {}).get("laser_residual_post_wake_support_threshold", 0.0),
            laser_residual_post_wake_body_program_track_threshold=raw.get("model", {}).get("laser_residual_post_wake_body_program_track_threshold", 0.0),
            laser_residual_post_wake_body_cold_to_hot_threshold=raw.get("model", {}).get("laser_residual_post_wake_body_cold_to_hot_threshold", 0.0),
            laser_residual_post_rescue_cap_margin=raw.get("model", {}).get("laser_residual_post_rescue_cap_margin", 0.0),
            laser_residual_post_rescue_cap_threshold=raw.get("model", {}).get("laser_residual_post_rescue_cap_threshold", 0.0),
            laser_residual_pre_track_support_floor=raw.get("model", {}).get("laser_residual_pre_track_support_floor", 0.35),
            laser_residual_path_body_endpoint_max=raw.get("model", {}).get("laser_residual_path_body_endpoint_max", 1.0),
            laser_residual_post_suppress_strength=raw.get("model", {}).get("laser_residual_post_suppress_strength", 0.0),
            laser_residual_control_gate_floor=raw.get("model", {}).get("laser_residual_control_gate_floor", 0.0),
            laser_residual_cold_to_hot_gate_threshold=raw.get("model", {}).get("laser_residual_cold_to_hot_gate_threshold", 0.0),
            laser_residual_cold_to_hot_gate_power=raw.get("model", {}).get("laser_residual_cold_to_hot_gate_power", 1.0),
            laser_residual_cold_to_hot_gate_floor=raw.get("model", {}).get("laser_residual_cold_to_hot_gate_floor", 0.0),
            laser_residual_cold_start_threshold=raw.get("model", {}).get("laser_residual_cold_start_threshold", 0.0),
            laser_residual_cold_start_softness=raw.get("model", {}).get("laser_residual_cold_start_softness", 0.0),
            laser_residual_prior_bypass_cold_start=raw.get("model", {}).get("laser_residual_prior_bypass_cold_start", False),
            laser_residual_prior_bypass_cold_start_threshold=raw.get("model", {}).get("laser_residual_prior_bypass_cold_start_threshold", 0.5),
            laser_residual_min_delta=raw.get("model", {}).get("laser_residual_min_delta", 0.0),
            laser_residual_use_tiered_min_delta=raw.get("model", {}).get("laser_residual_use_tiered_min_delta", False),
            laser_residual_weak_min_delta=raw.get("model", {}).get("laser_residual_weak_min_delta", 0.0),
            laser_residual_strong_support_threshold=raw.get("model", {}).get("laser_residual_strong_support_threshold", 0.0),
            laser_residual_strong_support_sources=raw.get("model", {}).get("laser_residual_strong_support_sources", "target_heat,sweep,endpoint,neighbor"),
            laser_residual_hidden_dim=raw.get("model", {}).get("laser_residual_hidden_dim", 128),
            laser_residual_gate_bias=raw.get("model", {}).get("laser_residual_gate_bias", -4.0),
            laser_residual_delta_init=raw.get("model", {}).get("laser_residual_delta_init", 400.0),
            enable_cold_to_hot_head=raw.get("model", {}).get("enable_cold_to_hot_head", False),
            cold_to_hot_hidden_dim=raw.get("model", {}).get("cold_to_hot_hidden_dim", 64),
            cold_to_hot_gate_bias=raw.get("model", {}).get("cold_to_hot_gate_bias", -2.0),
            post_endpoint_false_hot_guard_strength=raw.get("model", {}).get("post_endpoint_false_hot_guard_strength", 0.0),
            post_endpoint_false_hot_guard_threshold=raw.get("model", {}).get("post_endpoint_false_hot_guard_threshold", 0.0),
            post_endpoint_false_hot_guard_power=raw.get("model", {}).get("post_endpoint_false_hot_guard_power", 1.0),
            post_endpoint_false_hot_guard_max=raw.get("model", {}).get("post_endpoint_false_hot_guard_max", 1.0),
            post_endpoint_false_hot_guard_require_no_prearrival=raw.get("model", {}).get("post_endpoint_false_hot_guard_require_no_prearrival", True),
            post_low_sweep_false_hot_guard_strength=raw.get("model", {}).get("post_low_sweep_false_hot_guard_strength", 0.0),
            post_low_sweep_false_hot_guard_threshold=raw.get("model", {}).get("post_low_sweep_false_hot_guard_threshold", 0.0),
            post_low_sweep_false_hot_guard_power=raw.get("model", {}).get("post_low_sweep_false_hot_guard_power", 1.0),
            post_low_sweep_false_hot_guard_max=raw.get("model", {}).get("post_low_sweep_false_hot_guard_max", 1.0),
            post_mid_phase_false_hot_guard_strength=raw.get("model", {}).get("post_mid_phase_false_hot_guard_strength", 0.0),
            post_mid_phase_false_hot_post_min=raw.get("model", {}).get("post_mid_phase_false_hot_post_min", 0.0),
            post_mid_phase_false_hot_post_max=raw.get("model", {}).get("post_mid_phase_false_hot_post_max", 1.0),
            post_mid_phase_false_hot_guard_threshold=raw.get("model", {}).get("post_mid_phase_false_hot_guard_threshold", 0.0),
            post_mid_phase_false_hot_guard_power=raw.get("model", {}).get("post_mid_phase_false_hot_guard_power", 1.0),
            post_mid_phase_false_hot_guard_max=raw.get("model", {}).get("post_mid_phase_false_hot_guard_max", 1.0),
            post_mid_phase_false_hot_endpoint_max=raw.get("model", {}).get("post_mid_phase_false_hot_endpoint_max", 0.05),
            post_endpoint_false_hot_temperature_ceiling=raw.get("model", {}).get("post_endpoint_false_hot_temperature_ceiling", 0.0),
            post_endpoint_false_hot_hard_ceiling=raw.get("model", {}).get("post_endpoint_false_hot_hard_ceiling", False),
            post_endpoint_false_hot_hard_ceiling_threshold=raw.get("model", {}).get("post_endpoint_false_hot_hard_ceiling_threshold", 0.0),
            spatial=_populate_sub(raw.get("model", {}).get("spatial", {}), SpatialConfig),
            temporal=_populate_sub(raw.get("model", {}).get("temporal", {}), TemporalConfig),
            fusion=_populate_sub(raw.get("model", {}).get("fusion", {}), FusionConfig),
            decoder=_populate_sub(raw.get("model", {}).get("decoder", {}), DecoderConfig),
        )

        material = _populate_sub(raw.get("material", {}), MaterialProps)
        physics = _populate_sub(raw.get("physics", {}), PhysicsConfig)
        loss = _populate_sub(raw.get("loss", {}), LossConfig)
        training = _populate_sub(raw.get("training", {}), TrainingConfig)
        data = _populate_sub(raw.get("data", {}), DataConfig)
        data.apply_laser_feature_group()
        logging = _populate_sub(raw.get("logging", {}), LoggingConfig)

        return cls(
            model=model,
            material=material,
            physics=physics,
            loss=loss,
            training=training,
            data=data,
            logging=logging,
        )


def _populate_sub(raw_section: dict, dataclass_type):
    if raw_section is None:
        return dataclass_type()
    kwargs = {}
    for fld in dataclass_type.__dataclass_fields__:
        if fld in raw_section:
            kwargs[fld] = raw_section[fld]
    return dataclass_type(**kwargs)
