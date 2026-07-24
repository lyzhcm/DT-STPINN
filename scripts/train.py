"""Training script for DT-STPINN Paper 1: Temperature prediction."""
from __future__ import annotations

import argparse
import gc
import hashlib
import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, MaterialProps
from src.data.vtu_loader import VTULoader
from src.data.dataset import DEDTemporalDataset, collate_temporal_batch
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
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config.from_yaml(args.config)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_temporal_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print("Building model...")
    model = DTSTPINN(config, config.material)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    trainer = Trainer(model, config, config.material, device=device)

    print("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=config.training.epochs)

    print(f"\nBest validation loss: {trainer.best_val_loss:.6f} at epoch {trainer.best_epoch}")

    test_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=test_idx,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
    )
    print(f"\nEvaluating on test set ({len(test_dataset)} samples)...")
    test_metrics = trainer.validate_epoch(test_loader)
    print("Test metrics:")
    for k, v in test_metrics.items():
        if k == "metrics":
            for mk, mv in v.items():
                print(f"  {mk}: {mv:.6f}")
        else:
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
