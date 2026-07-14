"""Evaluation script for DT-STPINN.

Loads a trained checkpoint and evaluates on test data, with optional
autoregressive prediction and visualization output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.vtu_loader import VTULoader
from src.data.dataset import DEDTemporalDataset, collate_temporal_batch
from src.data.preprocessing import split_indices
from src.graph_builder.dynamic_graph import DynamicGraph
from src.model import DTSTPINN
from src.trainer import Trainer
from src.utils.metrics import compute_metrics
from src.engine.inferencer import Inferencer
from src.utils.visualization import write_prediction_sequence


def main():
    parser = argparse.ArgumentParser(description="Evaluate DT-STPINN")
    parser.add_argument("--config", type=str, default="configs/paper1.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vtu_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--autoregressive_steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    vtu_dir = args.vtu_dir or config.data.vtu_dir

    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")

    print("Loading VTU data...")
    loader = VTULoader(vtu_dir)
    vtu_data = loader.parse_sequence(verbose=True)

    graph = DynamicGraph(
        vtu_data,
        material_props=config.material,
        k_neighbors=config.data.k_neighbors,
        use_mesh_edges=config.data.use_mesh_edges,
    )

    _, _, test_idx = split_indices(
        graph.num_steps,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
    )

    test_dataset = DEDTemporalDataset(
        graph,
        window_size=config.data.window_size,
        predict_steps=config.data.predict_steps,
        time_indices=test_idx,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_temporal_batch,
    )

    model = DTSTPINN(config, config.material)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    print("Running evaluation...")
    trainer = Trainer(model, config, config.material, device=device)
    test_results = trainer.validate_epoch(test_loader)
    print("\nTest Results:")
    for k, v in test_results.items():
        if k == "metrics":
            for mk, mv in v.items():
                print(f"  {mk}: {mv:.6f}")
        else:
            print(f"  {k}: {v:.6f}")

    if args.autoregressive_steps > 0:
        print(f"\nRunning autoregressive prediction ({args.autoregressive_steps} steps)...")
        inferencer = Inferencer(model, device=device)

        sample = test_dataset[0]
        preds = inferencer.autoregressive_predict(
            sample["graph_sequence"],
            steps=args.autoregressive_steps,
            dynamic_graph=graph,
        )

        pred_temps = torch.stack([p.squeeze(-1) for p in preds])
        out_dir = Path(args.output_dir)
        write_prediction_sequence(
            str(out_dir), "pred",
            times=list(range(args.autoregressive_steps)),
            coords=graph.coords,
            temperatures=pred_temps,
            cells=vtu_data[-1].cells,
        )
        print(f"Predictions saved to {out_dir}")


if __name__ == "__main__":
    main()
