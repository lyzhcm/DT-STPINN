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


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    experiment_name: str = "paper1_temperature"
    save_every: int = 50
    eval_every: int = 10
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
