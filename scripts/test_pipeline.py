"""Quick pipeline verification for DT-STPINN Paper 1."""
import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
# Use memory-efficient attention instead of flash attention (better for large batch)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
from src.config import Config
from src.data.vtu_loader import VTULoader
from src.graph_builder.dynamic_graph import DynamicGraph
from src.model import DTSTPINN
from src.loss import DTSTPINNLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else f"Device: {device}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if device.type == "cuda" else "")

    config = Config.from_yaml("configs/paper1.yaml")

    print("\n[1/4] Loading VTU...")
    loader = VTULoader(config.data.vtu_dir)
    vtu_data = loader.parse_sequence()
    print(f"  Time steps: {len(vtu_data)}, nodes: {vtu_data[0].coords.shape[0]}")

    print("\n[2/4] Building dynamic graph...")
    t0 = time.time()
    graph = DynamicGraph(vtu_data, config.material,
                         k_neighbors=config.data.k_neighbors,
                         use_mesh_edges=config.data.use_mesh_edges)
    print(f"  Edges: {graph.edge_index.shape[1]:,} ({time.time() - t0:.1f}s)")

    print("\n[3/4] Building model...")
    model = DTSTPINN(config, config.material).to(device)
    model.eval()
    loss_fn = DTSTPINNLoss(config, config.material)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {params:.2f}M")
    print(f"  GNN out dim: {model._out_dim}")
    print(f"  Grad checkpoint: {config.model.spatial.num_layers > 0}")

    print("\n[4/4] Forward pass + Loss...")
    seq_len = min(4, graph.num_steps)
    if seq_len < 2:
        g0 = graph.get_graph_at(0).to(device)
        seq = [g0.clone() for _ in range(4)]
        gt_seq = "repeated x4 (only 1 timestep available)"
    else:
        seq = [graph.get_graph_at(i).to(device) for i in range(seq_len)]
        gt_seq = f"steps 0..{seq_len - 1}"

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        out = model(seq)
    elapsed = time.time() - t0

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Forward: {elapsed:.2f}s, peak VRAM: {peak_mem:.2f} GB")
    else:
        print(f"  Forward: {elapsed:.2f}s")

    T_pred = out["T_pred"]
    print(f"  T_pred shape: {tuple(T_pred.shape)}")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        total, comps = loss_fn.forward(
            pred=T_pred, target=seq[-1].y, prev_temp=seq[-2].y if len(seq) > 1 else seq[-1].y,
            coords=seq[-1].coords, edge_index=seq[-1].edge_index,
            boundary=graph.boundary.to(device), dt=1.0,
            mask=seq[-1].mask,
        )
    print(f"  Total loss: {total.item():.4e}")
    for k, v in comps.items():
        print(f"    {k}: {v.item():.4e}")

    if graph.num_steps < 2:
        print("\n[WARN] Only 1 VTU timestep found. Training needs full simulation sequence.")
        print("       Put all Flow-3D AM Data-*.vtu files into data/raw/")

    print("\nPipeline OK. Ready to train.")


if __name__ == "__main__":
    main()
