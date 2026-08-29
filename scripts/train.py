"""Training script for DT-STPINN Paper 1: Temperature prediction."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, MaterialProps
from src.data.vtu_loader import VTULoader
from src.data.dataset import DEDTemporalDataset, StratifiedWindowSampler, collate_temporal_batch
from src.data.preprocessing import split_indices
from src.graph_builder.dynamic_graph import DynamicGraph
from src.model import DTSTPINN
from src.trainer import Trainer


def graph_cache_path(cache_dir: str | Path, vtu_dir: str | Path,
                     loader: VTULoader, config: Config) -> Path:
    cache_root = Path(cache_dir)
    h = hashlib.sha256()
    h.update(str(Path(vtu_dir).resolve()).encode("utf-8"))
    h.update(f"k={config.data.k_neighbors};mesh={config.data.use_mesh_edges};".encode("utf-8"))
    h.update(f"kmat={config.material.thermal_conductivity};".encode("utf-8"))

    for fp in loader.files:
        st = fp.stat()
        h.update(f"{fp.name}:{st.st_size}:{st.st_mtime_ns}\n".encode("utf-8"))

    return cache_root / f"dynamic_graph_{h.hexdigest()[:16]}.pt"


def load_graph_cache(path: Path, material_props):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return DynamicGraph.from_cache_dict(state["graph"], material_props)


def save_graph_cache(path: Path, graph: DynamicGraph, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": metadata,
        "graph": graph.to_cache_dict(),
    }, path)


def to_jsonable(value):
    """Convert common metric value types to JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return to_jsonable(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_stratified_subset_indices(dataset: DEDTemporalDataset, limit: int,
                                     config: Config, seed: int) -> list[int]:
    """Pick a deterministic class-balanced subset across the full train span."""
    buckets: dict[str, list[int]] = {"normal": [], "hot": [], "melting": []}
    hot_threshold = getattr(config.loss, "hot_threshold", 500.0)
    solidus = config.material.solidus_temp
    for idx in range(len(dataset)):
        max_temp = dataset.target_max_temperature(idx)
        if max_temp >= solidus:
            buckets["melting"].append(idx)
        elif max_temp >= hot_threshold:
            buckets["hot"].append(idx)
        else:
            buckets["normal"].append(idx)

    ratios = {
        "normal": config.data.stratified_normal,
        "hot": config.data.stratified_hot,
        "melting": config.data.stratified_melting,
    }
    active = {cat: ratios[cat] for cat, values in buckets.items() if values}
    active_total = sum(active.values())
    if active_total <= 0:
        return list(range(min(limit, len(dataset))))
    active = {cat: ratio / active_total for cat, ratio in active.items()}

    rng = random.Random(seed)
    selected: list[int] = []
    for cat, ratio in active.items():
        values = buckets[cat]
        count = min(len(values), max(1, int(round(limit * ratio))))
        selected.extend(rng.sample(values, count))

    if len(selected) < limit:
        remaining = [idx for idx in range(len(dataset)) if idx not in set(selected)]
        if remaining:
            selected.extend(rng.sample(remaining, min(limit - len(selected), len(remaining))))
    elif len(selected) > limit:
        selected = selected[:limit]

    selected.sort()
    counts = {cat: sum(1 for idx in selected if idx in set(values))
              for cat, values in buckets.items()}
    print(
        "Limited train subset (stratified across full split): "
        f"normal={counts['normal']}, hot={counts['hot']}, "
        f"melting={counts['melting']}"
    )
    return selected


def main():
    parser = argparse.ArgumentParser(description="Train DT-STPINN for temperature prediction")
    parser.add_argument("--config", type=str, default="configs/paper1.yaml")
    parser.add_argument("--vtu_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache_dir", type=str, default="data/processed")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved training checkpoint.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override logging.experiment_name to keep runs separate.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs from the YAML config.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples for quick experiments.")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation samples for quick experiments.")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Limit test samples for quick experiments.")
    parser.add_argument("--benchmark_batches", type=int, default=0,
                        help="Benchmark N training batches, then exit.")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip final test-set evaluation.")
    parser.add_argument(
        "--test_report",
        type=str,
        default=None,
        help=(
            "Path for final test metrics JSON. Defaults to "
            "logs/<experiment_name>/test_metrics.json."
        ),
    )
    parser.add_argument(
        "--graph_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to keep preprocessed graph tensors. auto uses CUDA when available.",
    )
    args = parser.parse_args()

    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be greater than zero.")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.benchmark_batches < 0:
        parser.error("--benchmark_batches cannot be negative.")
    if args.resume is not None and not Path(args.resume).is_file():
        parser.error(f"--resume checkpoint not found: {args.resume}")

    set_seed(args.seed)

    config = Config.from_yaml(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name
    vtu_dir = args.vtu_dir or config.data.vtu_dir

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"VTU directory: {vtu_dir}")
    print(f"Window size: {config.data.window_size}")
    print(f"Hidden dim: {config.model.hidden_dim}")

    loader = VTULoader(vtu_dir)
    if loader.num_steps == 0:
        raise FileNotFoundError(f"No Data-*.vtu files found in {vtu_dir}")

    cache_path = graph_cache_path(args.cache_dir, vtu_dir, loader, config)
    graph = None

    if not args.no_cache and cache_path.exists() and not args.rebuild_cache:
        print(f"Loading preprocessed graph cache: {cache_path}")
        graph = load_graph_cache(cache_path, config.material)
    else:
        if args.no_cache:
            print("Graph cache disabled by --no_cache.")
        elif args.rebuild_cache:
            print("Rebuilding graph cache because --rebuild_cache was set.")
        else:
            print(f"No graph cache found. It will be saved to: {cache_path}")

        print("Loading VTU data...")
        vtu_data = loader.parse_sequence(verbose=True)
        print(f"Loaded {len(vtu_data)} time steps, {vtu_data[0].coords.shape[0]} nodes.")

        print("Building dynamic graph...")
        graph = DynamicGraph(
            vtu_data,
            material_props=config.material,
            k_neighbors=config.data.k_neighbors,
            use_mesh_edges=config.data.use_mesh_edges,
        )
        del vtu_data
        gc.collect()

        if not args.no_cache:
            print(f"Saving preprocessed graph cache: {cache_path}")
            save_graph_cache(cache_path, graph, {
                "vtu_dir": str(Path(vtu_dir).resolve()),
                "num_vtu_files": loader.num_steps,
                "k_neighbors": config.data.k_neighbors,
                "use_mesh_edges": config.data.use_mesh_edges,
            })

    graph.apply_laser_path_config(config.data)
    if config.data.laser_path_mode != "estimated":
        print(f"Laser path mode: {config.data.laser_path_mode}")

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges.")

    if args.graph_device == "auto":
        graph_device = device if device.type == "cuda" else torch.device("cpu")
    else:
        graph_device = torch.device(args.graph_device)
    if graph_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("graph_device=cuda was requested, but CUDA is not available.")

    print(f"Moving graph tensors to {graph_device}...")
    graph.to(graph_device)
    if graph_device.type == "cuda":
        graph_mem_gb = (
            graph.temperatures.numel() * graph.temperatures.element_size()
            + graph.live.numel() * graph.live.element_size()
            + graph.edge_index.numel() * graph.edge_index.element_size()
            + graph.edge_attr.numel() * graph.edge_attr.element_size()
            + graph.coords.numel() * graph.coords.element_size()
        ) / 1e9
        print(f"Approx graph tensor memory on GPU: {graph_mem_gb:.2f} GB")

    train_idx, val_idx, test_idx = split_indices(
        graph.num_steps,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
    )
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=train_idx,
        use_target_laser_features=config.data.use_target_laser_features,
        laser_feature_radius_mm=config.data.laser_feature_radius_mm,
        laser_feature_along_radius_mm=config.data.laser_feature_along_radius_mm,
        laser_feature_depth_mm=config.data.laser_feature_depth_mm,
        laser_feature_time_scale_to_s=config.data.laser_feature_time_scale_to_s,
        laser_feature_include_laser_coordinates=config.data.laser_feature_include_laser_coordinates,
        laser_feature_include_scan_geometry=config.data.laser_feature_include_scan_geometry,
        laser_feature_include_sweep=config.data.laser_feature_include_sweep,
        laser_feature_include_exposure=config.data.laser_feature_include_exposure,
        laser_feature_exposure_source=config.data.laser_feature_exposure_source,
        laser_feature_exposure_past_steps=config.data.laser_feature_exposure_past_steps,
        laser_feature_exposure_future_steps=config.data.laser_feature_exposure_future_steps,
        laser_feature_exposure_time_decay_s=config.data.laser_feature_exposure_time_decay_s,
        laser_feature_exposure_use_segments=config.data.laser_feature_exposure_use_segments,
        laser_feature_include_exposure_split=config.data.laser_feature_include_exposure_split,
        laser_feature_include_arrival_time=config.data.laser_feature_include_arrival_time,
        laser_feature_arrival_time_decay_s=config.data.laser_feature_arrival_time_decay_s,
        laser_feature_include_neighbor_temp=config.data.laser_feature_include_neighbor_temp,
        laser_feature_include_neighbor_hot_stats=config.data.laser_feature_include_neighbor_hot_stats,
        laser_feature_neighbor_hot_threshold=config.data.laser_feature_neighbor_hot_threshold,
        laser_feature_include_neighbor_warm_stats=config.data.laser_feature_include_neighbor_warm_stats,
        laser_feature_neighbor_warm_threshold=config.data.laser_feature_neighbor_warm_threshold,
        laser_feature_include_body_source=config.data.laser_feature_include_body_source,
        laser_body_radius_mm=config.data.laser_body_radius_mm,
        laser_body_height_mm=config.data.laser_body_height_mm,
        laser_body_radius_front_mm=config.data.laser_body_radius_front_mm,
        laser_body_radius_back_mm=config.data.laser_body_radius_back_mm,
        laser_body_coeff_front=config.data.laser_body_coeff_front,
        laser_body_coeff_back=config.data.laser_body_coeff_back,
        laser_feature_include_path_arrival=config.data.laser_feature_include_path_arrival,
        laser_feature_path_arrival_time_decay_s=config.data.laser_feature_path_arrival_time_decay_s,
        laser_feature_path_arrival_gate_mode=config.data.laser_feature_path_arrival_gate_mode,
        laser_feature_path_arrival_neighbor_tracks=config.data.laser_feature_path_arrival_neighbor_tracks,
        laser_feature_include_path_phase=config.data.laser_feature_include_path_phase,
        laser_feature_include_path_coordinates=config.data.laser_feature_include_path_coordinates,
        laser_feature_include_path_timing=config.data.laser_feature_include_path_timing,
        laser_feature_include_path_body_support=config.data.laser_feature_include_path_body_support,
        laser_feature_include_path_endpoint=config.data.laser_feature_include_path_endpoint,
        laser_feature_endpoint_radius_mm=config.data.laser_feature_endpoint_radius_mm,
        laser_feature_endpoint_time_decay_s=config.data.laser_feature_endpoint_time_decay_s,
        laser_feature_include_path_wake=config.data.laser_feature_include_path_wake,
        laser_feature_wake_cross_radius_mm=config.data.laser_feature_wake_cross_radius_mm,
        laser_feature_wake_tail_decay_mm=config.data.laser_feature_wake_tail_decay_mm,
        laser_feature_wake_lead_decay_mm=config.data.laser_feature_wake_lead_decay_mm,
        laser_feature_wake_time_decay_s=config.data.laser_feature_wake_time_decay_s,
    )
    val_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=val_idx,
        use_target_laser_features=config.data.use_target_laser_features,
        laser_feature_radius_mm=config.data.laser_feature_radius_mm,
        laser_feature_along_radius_mm=config.data.laser_feature_along_radius_mm,
        laser_feature_depth_mm=config.data.laser_feature_depth_mm,
        laser_feature_time_scale_to_s=config.data.laser_feature_time_scale_to_s,
        laser_feature_include_laser_coordinates=config.data.laser_feature_include_laser_coordinates,
        laser_feature_include_scan_geometry=config.data.laser_feature_include_scan_geometry,
        laser_feature_include_sweep=config.data.laser_feature_include_sweep,
        laser_feature_include_exposure=config.data.laser_feature_include_exposure,
        laser_feature_exposure_source=config.data.laser_feature_exposure_source,
        laser_feature_exposure_past_steps=config.data.laser_feature_exposure_past_steps,
        laser_feature_exposure_future_steps=config.data.laser_feature_exposure_future_steps,
        laser_feature_exposure_time_decay_s=config.data.laser_feature_exposure_time_decay_s,
        laser_feature_exposure_use_segments=config.data.laser_feature_exposure_use_segments,
        laser_feature_include_exposure_split=config.data.laser_feature_include_exposure_split,
        laser_feature_include_arrival_time=config.data.laser_feature_include_arrival_time,
        laser_feature_arrival_time_decay_s=config.data.laser_feature_arrival_time_decay_s,
        laser_feature_include_neighbor_temp=config.data.laser_feature_include_neighbor_temp,
        laser_feature_include_neighbor_hot_stats=config.data.laser_feature_include_neighbor_hot_stats,
        laser_feature_neighbor_hot_threshold=config.data.laser_feature_neighbor_hot_threshold,
        laser_feature_include_neighbor_warm_stats=config.data.laser_feature_include_neighbor_warm_stats,
        laser_feature_neighbor_warm_threshold=config.data.laser_feature_neighbor_warm_threshold,
        laser_feature_include_body_source=config.data.laser_feature_include_body_source,
        laser_body_radius_mm=config.data.laser_body_radius_mm,
        laser_body_height_mm=config.data.laser_body_height_mm,
        laser_body_radius_front_mm=config.data.laser_body_radius_front_mm,
        laser_body_radius_back_mm=config.data.laser_body_radius_back_mm,
        laser_body_coeff_front=config.data.laser_body_coeff_front,
        laser_body_coeff_back=config.data.laser_body_coeff_back,
        laser_feature_include_path_arrival=config.data.laser_feature_include_path_arrival,
        laser_feature_path_arrival_time_decay_s=config.data.laser_feature_path_arrival_time_decay_s,
        laser_feature_path_arrival_gate_mode=config.data.laser_feature_path_arrival_gate_mode,
        laser_feature_path_arrival_neighbor_tracks=config.data.laser_feature_path_arrival_neighbor_tracks,
        laser_feature_include_path_phase=config.data.laser_feature_include_path_phase,
        laser_feature_include_path_coordinates=config.data.laser_feature_include_path_coordinates,
        laser_feature_include_path_timing=config.data.laser_feature_include_path_timing,
        laser_feature_include_path_body_support=config.data.laser_feature_include_path_body_support,
        laser_feature_include_path_endpoint=config.data.laser_feature_include_path_endpoint,
        laser_feature_endpoint_radius_mm=config.data.laser_feature_endpoint_radius_mm,
        laser_feature_endpoint_time_decay_s=config.data.laser_feature_endpoint_time_decay_s,
        laser_feature_include_path_wake=config.data.laser_feature_include_path_wake,
        laser_feature_wake_cross_radius_mm=config.data.laser_feature_wake_cross_radius_mm,
        laser_feature_wake_tail_decay_mm=config.data.laser_feature_wake_tail_decay_mm,
        laser_feature_wake_lead_decay_mm=config.data.laser_feature_wake_lead_decay_mm,
        laser_feature_wake_time_decay_s=config.data.laser_feature_wake_time_decay_s,
    )

    full_train_samples = len(train_dataset)
    full_val_samples = len(val_dataset)

    actual_node_feature_dim = train_dataset.input_feature_dim
    configured_node_feature_dim = int(config.model.node_feature_dim)
    if configured_node_feature_dim != actual_node_feature_dim:
        print(
            "Node feature dim: "
            f"config={configured_node_feature_dim}, "
            f"actual={actual_node_feature_dim}; using actual dataset dim."
        )
        config.model.node_feature_dim = actual_node_feature_dim
    else:
        print(f"Node feature dim: {actual_node_feature_dim}")
    print(
        "Laser feature dim: "
        f"{train_dataset.target_laser_feature_dim} "
        f"(base {actual_node_feature_dim - train_dataset.target_laser_feature_dim})"
    )

    train_limit = args.max_train_samples
    if args.benchmark_batches > 0:
        benchmark_samples = args.benchmark_batches * config.training.batch_size
        train_limit = benchmark_samples if train_limit is None else min(
            train_limit, benchmark_samples
        )

    if train_limit is not None and train_limit < len(train_dataset):
        if getattr(config.data, "stratified_sampling", False):
            train_subset_indices = select_stratified_subset_indices(
                train_dataset, train_limit, config, args.seed
            )
        else:
            train_subset_indices = list(range(train_limit))
        train_dataset = Subset(train_dataset, train_subset_indices)
    if args.max_val_samples is not None and args.max_val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, range(args.max_val_samples))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_temporal_batch,
    )

    # --- stratified window sampling (training only) -----------------------
    if getattr(config.data, "stratified_sampling", False):
        stratified_sampler = StratifiedWindowSampler(
            train_dataset,
            solidus=config.material.solidus_temp,
            hot_threshold=getattr(config.loss, "hot_threshold", 500.0),
            normal_ratio=config.data.stratified_normal,
            hot_ratio=config.data.stratified_hot,
            melting_ratio=config.data.stratified_melting,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=stratified_sampler,
            collate_fn=collate_temporal_batch,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
    )

    print(
        f"Train samples: {len(train_dataset)}/{full_train_samples}, "
        f"Val samples: {len(val_dataset)}/{full_val_samples}"
    )

    print("Building model...")
    model = DTSTPINN(config, config.material)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    trainer = Trainer(model, config, config.material, device=device)
    if trainer.amp_enabled:
        print(f"AMP: enabled ({trainer.amp_dtype_name})")
    else:
        print("AMP: disabled")
    print(
        "Physics units: "
        f"coords x {trainer.loss_fn.coordinate_scale_to_m:g} m, "
        f"time x {trainer.loss_fn.time_scale_to_s:g} s"
    )
    print(
        "PDE residual scale: "
        f"{trainer.loss_fn.pde_residual_scale:.4e} W/m^3"
    )

    if args.resume is not None:
        resumed_epoch = trainer.load_checkpoint(args.resume)
        print(
            f"Resumed checkpoint: {args.resume} "
            f"(completed epoch {resumed_epoch}, global step {trainer.global_step})"
        )

    if args.benchmark_batches > 0:
        print(f"Starting benchmark ({len(train_loader)} batch(es))...")
        metrics = trainer.train_epoch(train_loader, epoch=1, total_epochs=1)
        seconds_per_batch = metrics["seconds_per_batch"]
        full_batches = math.ceil(full_train_samples / config.training.batch_size)
        estimated_epoch_seconds = seconds_per_batch * full_batches
        print("\nBenchmark complete:")
        print(f"  Average batch time: {seconds_per_batch:.2f} s")
        print(f"  Peak allocated VRAM: {metrics['peak_vram_gb']:.2f} GB")
        print(f"  Full train batches: {full_batches}")
        print(f"  Estimated full epoch: {estimated_epoch_seconds / 3600:.2f} h")
        trainer.writer.close()
        return

    print("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=config.training.epochs)

    print(f"\nBest validation loss: {trainer.best_val_loss:.6f} at epoch {trainer.best_epoch}")
    if getattr(trainer, "best_hot_epoch", 0) > 0:
        print(
            f"Best {trainer.hot_checkpoint_metric}: "
            f"{trainer.best_hot_score:.6f} at epoch {trainer.best_hot_epoch} "
            f"(checkpoint: {trainer.log_dir / 'best_hot_model.pt'})"
        )

    if args.skip_test:
        print("Skipping test-set evaluation (--skip_test).")
        return

    test_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=test_idx,
        use_target_laser_features=config.data.use_target_laser_features,
            laser_feature_radius_mm=config.data.laser_feature_radius_mm,
            laser_feature_along_radius_mm=config.data.laser_feature_along_radius_mm,
            laser_feature_depth_mm=config.data.laser_feature_depth_mm,
            laser_feature_time_scale_to_s=config.data.laser_feature_time_scale_to_s,
            laser_feature_include_laser_coordinates=config.data.laser_feature_include_laser_coordinates,
            laser_feature_include_scan_geometry=config.data.laser_feature_include_scan_geometry,
            laser_feature_include_sweep=config.data.laser_feature_include_sweep,
            laser_feature_include_exposure=config.data.laser_feature_include_exposure,
            laser_feature_exposure_source=config.data.laser_feature_exposure_source,
            laser_feature_exposure_past_steps=config.data.laser_feature_exposure_past_steps,
            laser_feature_exposure_future_steps=config.data.laser_feature_exposure_future_steps,
            laser_feature_exposure_time_decay_s=config.data.laser_feature_exposure_time_decay_s,
            laser_feature_exposure_use_segments=config.data.laser_feature_exposure_use_segments,
            laser_feature_include_exposure_split=config.data.laser_feature_include_exposure_split,
            laser_feature_include_arrival_time=config.data.laser_feature_include_arrival_time,
            laser_feature_arrival_time_decay_s=config.data.laser_feature_arrival_time_decay_s,
            laser_feature_include_neighbor_temp=config.data.laser_feature_include_neighbor_temp,
            laser_feature_include_neighbor_hot_stats=config.data.laser_feature_include_neighbor_hot_stats,
            laser_feature_neighbor_hot_threshold=config.data.laser_feature_neighbor_hot_threshold,
            laser_feature_include_neighbor_warm_stats=config.data.laser_feature_include_neighbor_warm_stats,
            laser_feature_neighbor_warm_threshold=config.data.laser_feature_neighbor_warm_threshold,
            laser_feature_include_body_source=config.data.laser_feature_include_body_source,
            laser_body_radius_mm=config.data.laser_body_radius_mm,
            laser_body_height_mm=config.data.laser_body_height_mm,
            laser_body_radius_front_mm=config.data.laser_body_radius_front_mm,
            laser_body_radius_back_mm=config.data.laser_body_radius_back_mm,
            laser_body_coeff_front=config.data.laser_body_coeff_front,
            laser_body_coeff_back=config.data.laser_body_coeff_back,
            laser_feature_include_path_arrival=config.data.laser_feature_include_path_arrival,
            laser_feature_path_arrival_time_decay_s=config.data.laser_feature_path_arrival_time_decay_s,
            laser_feature_path_arrival_gate_mode=config.data.laser_feature_path_arrival_gate_mode,
            laser_feature_path_arrival_neighbor_tracks=config.data.laser_feature_path_arrival_neighbor_tracks,
            laser_feature_include_path_phase=config.data.laser_feature_include_path_phase,
            laser_feature_include_path_coordinates=config.data.laser_feature_include_path_coordinates,
            laser_feature_include_path_timing=config.data.laser_feature_include_path_timing,
            laser_feature_include_path_body_support=config.data.laser_feature_include_path_body_support,
            laser_feature_include_path_endpoint=config.data.laser_feature_include_path_endpoint,
            laser_feature_endpoint_radius_mm=config.data.laser_feature_endpoint_radius_mm,
            laser_feature_endpoint_time_decay_s=config.data.laser_feature_endpoint_time_decay_s,
            laser_feature_include_path_wake=config.data.laser_feature_include_path_wake,
            laser_feature_wake_cross_radius_mm=config.data.laser_feature_wake_cross_radius_mm,
            laser_feature_wake_tail_decay_mm=config.data.laser_feature_wake_tail_decay_mm,
            laser_feature_wake_lead_decay_mm=config.data.laser_feature_wake_lead_decay_mm,
            laser_feature_wake_time_decay_s=config.data.laser_feature_wake_time_decay_s,
        )
    full_test_samples = len(test_dataset)
    if args.max_test_samples is not None and args.max_test_samples < len(test_dataset):
        test_dataset = Subset(test_dataset, range(args.max_test_samples))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
    )
    print(
        f"\nEvaluating on test set "
        f"({len(test_dataset)}/{full_test_samples} samples)..."
    )
    test_metrics = trainer.validate_epoch(test_loader)
    print("Test metrics:")
    for k, v in test_metrics.items():
        if k == "metrics":
            for mk, mv in v.items():
                print(f"  {mk}: {mv:.6f}")
        elif k != "worst_case":
            print(f"  {k}: {v:.6f}")

    worst = test_metrics.get("worst_case")
    if worst is not None:
        coord = ", ".join(f"{value:.3f}" for value in worst["coord_mm"])
        raw_time = worst["target_time_raw"]
        raw_time_text = "unknown" if raw_time is None else f"{raw_time:g}"
        seconds = worst["target_time_s"]
        seconds_text = "unknown" if seconds is None else f"{seconds:g} s"
        print("Worst-error node:")
        print(
            f"  Step/time: {worst['target_step']} / "
            f"{raw_time_text} raw ({seconds_text})"
        )
        print(f"  Node index: {worst['node_index']}")
        print(f"  Coordinate (mm): [{coord}]")
        print(
            f"  Prediction/target: {worst['prediction']:.6f} / "
            f"{worst['target']:.6f}"
        )
        print(f"  Absolute error: {worst['abs_error']:.6f}")
        if "hotspot_probability" in worst:
            print(f"  Hotspot probability: {worst['hotspot_probability']:.6f}")
        if "hotspot_specialist_temp" in worst:
            print(f"  Specialist temperature: {worst['hotspot_specialist_temp']:.6f}")
        if "hotspot_specialist_gate" in worst:
            print(f"  Specialist gate: {worst['hotspot_specialist_gate']:.6f}")
        if "process_gate" in worst:
            print(f"  Process gate: {worst['process_gate']:.6f}")
        if "laser_target_heat_gate" in worst:
            print(f"  Laser target heat gate: {worst['laser_target_heat_gate']:.6f}")
        if "laser_body_heat_gate" in worst:
            print(f"  Laser body heat gate: {worst['laser_body_heat_gate']:.6f}")
        if "laser_path_body_heat_gate" in worst:
            print(f"  Laser path body heat gate: {worst['laser_path_body_heat_gate']:.6f}")
        if "laser_path_wake_gate" in worst:
            print(f"  Laser path wake gate: {worst['laser_path_wake_gate']:.6f}")
        if "laser_sweep_heat_gate" in worst:
            print(f"  Laser sweep heat gate: {worst['laser_sweep_heat_gate']:.6f}")
        if "laser_arrival_gate" in worst:
            print(f"  Laser arrival gate: {worst['laser_arrival_gate']:.6f}")
        if "laser_path_active_gate" in worst:
            print(f"  Laser path active gate: {worst['laser_path_active_gate']:.6f}")
        if "laser_path_pre_arrival_gate" in worst:
            print(f"  Laser path pre-arrival gate: {worst['laser_path_pre_arrival_gate']:.6f}")
        if "laser_path_post_arrival_gate" in worst:
            print(f"  Laser path post-arrival gate: {worst['laser_path_post_arrival_gate']:.6f}")
        if "laser_endpoint_gate" in worst:
            print(f"  Laser endpoint gate: {worst['laser_endpoint_gate']:.6f}")
        if "laser_path_program_track_gate" in worst:
            print(f"  Laser path program-track gate: {worst['laser_path_program_track_gate']:.6f}")
        if "laser_path_elapsed_gate" in worst:
            print(f"  Laser path elapsed gate: {worst['laser_path_elapsed_gate']:.6f}")
        if "laser_path_time_until_gate" in worst:
            print(f"  Laser path time-until gate: {worst['laser_path_time_until_gate']:.6f}")
        if "laser_path_scanned_gate" in worst:
            print(f"  Laser path scanned gate: {worst['laser_path_scanned_gate']:.6f}")
        if "laser_path_cooling_tail_gate" in worst:
            print(f"  Laser path cooling tail gate: {worst['laser_path_cooling_tail_gate']:.6f}")
        if "neighbor_hot_gate" in worst:
            print(f"  Neighbor hot gate: {worst['neighbor_hot_gate']:.6f}")
        if "hotspot_laser_prior_temp" in worst:
            print(f"  Laser prior temperature: {worst['hotspot_laser_prior_temp']:.6f}")
        if "hotspot_laser_prior_gate" in worst:
            print(f"  Laser prior gate: {worst['hotspot_laser_prior_gate']:.6f}")
        if "laser_residual_prior_gate" in worst:
            print(f"  Laser residual prior gate: {worst['laser_residual_prior_gate']:.6f}")
        if "laser_residual_learned_gate" in worst:
            print(f"  Laser residual learned gate: {worst['laser_residual_learned_gate']:.6f}")
        if "laser_residual_control_gate" in worst:
            print(f"  Laser residual control gate: {worst['laser_residual_control_gate']:.6f}")
        if "laser_residual_cold_start_gate" in worst:
            print(f"  Laser residual cold-start gate: {worst['laser_residual_cold_start_gate']:.6f}")
        if "laser_residual_gate" in worst:
            print(f"  Laser residual gate: {worst['laser_residual_gate']:.6f}")
        if "laser_residual_delta" in worst:
            print(f"  Laser residual delta: {worst['laser_residual_delta']:.6f}")
        if "laser_residual_boost" in worst:
            print(f"  Laser residual boost: {worst['laser_residual_boost']:.6f}")
        if "laser_residual_post_rescue_cap_gate" in worst:
            print(f"  Laser residual post-rescue cap gate: {worst['laser_residual_post_rescue_cap_gate']:.6f}")
        if "cold_to_hot_gate" in worst:
            print(f"  Cold-to-hot gate: {worst['cold_to_hot_gate']:.6f}")

    report_path = Path(args.test_report) if args.test_report else trainer.log_dir / "test_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_name": config.logging.experiment_name,
        "config": str(Path(args.config)),
        "vtu_dir": str(Path(vtu_dir)),
        "checkpoint_best_val": str(trainer.log_dir / "best_model.pt"),
        "checkpoint_best_hot": str(trainer.log_dir / "best_hot_model.pt"),
        "epochs_requested": config.training.epochs,
        "best_val_loss": trainer.best_val_loss,
        "best_epoch": trainer.best_epoch,
        "best_hot_metric": trainer.hot_checkpoint_metric,
        "best_hot_score": trainer.best_hot_score,
        "best_hot_epoch": trainer.best_hot_epoch,
        "test_samples": len(test_dataset),
        "full_test_samples": full_test_samples,
        "test_metrics": test_metrics,
    }
    report_path.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Test report: {report_path}")


if __name__ == "__main__":
    main()
