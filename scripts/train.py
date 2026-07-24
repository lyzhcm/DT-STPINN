"""Training script for DT-STPINN Paper 1: Temperature prediction."""
from __future__ import annotations

import argparse
import gc
import hashlib
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    )
    val_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=val_idx,
    )

    full_train_samples = len(train_dataset)
    full_val_samples = len(val_dataset)

    train_limit = args.max_train_samples
    if args.benchmark_batches > 0:
        benchmark_samples = args.benchmark_batches * config.training.batch_size
        train_limit = benchmark_samples if train_limit is None else min(
            train_limit, benchmark_samples
        )

    if train_limit is not None and train_limit < len(train_dataset):
        train_dataset = Subset(train_dataset, range(train_limit))
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
        if args.max_train_samples is not None:
            print(
                "WARNING: --max_train_samples disables stratified_sampling "
                "for this run."
            )
        else:
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

    if args.skip_test:
        print("Skipping test-set evaluation (--skip_test).")
        return

    test_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=test_idx,
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


if __name__ == "__main__":
    main()
