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
    graph.apply_laser_path_config(config.data)
    if config.data.laser_path_mode != "estimated":
        print(f"Laser path mode: {config.data.laser_path_mode}")

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
        use_target_laser_features=config.data.use_target_laser_features,
        laser_feature_radius_mm=config.data.laser_feature_radius_mm,
        laser_feature_along_radius_mm=config.data.laser_feature_along_radius_mm,
        laser_feature_depth_mm=config.data.laser_feature_depth_mm,
        laser_feature_time_scale_to_s=config.data.laser_feature_time_scale_to_s,
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
    actual_node_feature_dim = test_dataset.input_feature_dim
    if int(config.model.node_feature_dim) != actual_node_feature_dim:
        print(
            "Node feature dim: "
            f"config={config.model.node_feature_dim}, "
            f"actual={actual_node_feature_dim}; using actual dataset dim."
        )
        config.model.node_feature_dim = actual_node_feature_dim
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
