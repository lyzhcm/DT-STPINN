"""Quick pipeline verification script."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import Config
from src.data.vtu_loader import VTULoader
from src.graph_builder.dynamic_graph import DynamicGraph
from src.model import DTSTPINN
from src.loss import DTSTPINNLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = Config.from_yaml("configs/paper1.yaml")

    print("Loading VTU...")
    loader = VTULoader("data/raw")
    vtu_data = loader.parse_sequence()

    print(f"Building graph ({vtu_data[0].coords.shape[0]} nodes)...")
    graph = DynamicGraph(vtu_data, config.material, use_mesh_edges=True)
    print(f"  Edges: {graph.edge_index.shape[1]}")

    print("Building model...")
    model = DTSTPINN(config, config.material).to(device)
    loss_fn = DTSTPINNLoss(config, config.material)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.2f}M")

    g0 = graph.get_graph_at(0).to(device)
    seq = [g0.clone() for _ in range(4)]

    print("Running forward pass...")
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    t0 = time.time()

    out = model(seq)
    T_pred = out["T_pred"]

    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    print(f"  Done in {elapsed:.1f}s, peak VRAM: {mem:.1f}GB" if device.type == "cuda" else f"  Done in {elapsed:.1f}s")

    total, comps = loss_fn.forward(
        pred=T_pred, target=g0.y, prev_temp=g0.y,
        coords=g0.coords, edge_index=g0.edge_index,
        boundary=graph.boundary.to(device), dt=1.0, mask=g0.mask,
    )
    print(f"  Loss: {total.item():.4f}")
    for k, v in comps.items():
        print(f"    {k}: {v.item():.4f}")

    print("\nPipeline OK - ready to train with full VTU sequence.")

if __name__ == "__main__":
    main()
