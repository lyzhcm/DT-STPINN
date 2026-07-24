"""Quick pipeline verification for DT-STPINN Paper 1."""
import argparse
import sys
import time
import gc
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test DT-STPINN with a small VTU sequence."
    )
    parser.add_argument("--config", type=str, default="configs/paper1.yaml")
    parser.add_argument(
        "--vtu_dir",
        type=str,
        default=None,
        help="VTU directory. Defaults to F:\\VTU when it exists, otherwise config.data.vtu_dir.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=4,
        help="Number of VTU files to parse for the smoke test.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset in the sorted Data-*.vtu list.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Read every Nth VTU file after --start.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length for the forward pass. Defaults to loaded steps, capped by config max length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA autocast during the smoke test.",
    )
    return parser.parse_args()


def resolve_vtu_dir(config, cli_vtu_dir: str | None) -> str:
    if cli_vtu_dir:
        return cli_vtu_dir
    server_dir = Path(r"F:\VTU")
    if server_dir.exists():
        return str(server_dir)
    return config.data.vtu_dir


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    vtu_dir = resolve_vtu_dir(config, args.vtu_dir)
    device = resolve_device(args.device)

    print(f"Device: {device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else f"Device: {device}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if device.type == "cuda" else "")

    print(f"Config: {args.config}")
    print(f"VTU directory: {vtu_dir}")

    print("\n[1/4] Loading VTU...")
    loader = VTULoader(vtu_dir)
    if loader.num_steps == 0:
        raise FileNotFoundError(f"No Data-*.vtu files found in {vtu_dir}")

    print(f"  Found {loader.num_steps} VTU files.")
    print(f"  Using start={args.start}, stride={args.stride}, max_steps={args.max_steps}")
    vtu_data = loader.parse_sequence(
        max_steps=args.max_steps,
        stride=args.stride,
        start=args.start,
    )
    if not vtu_data:
        raise RuntimeError("No VTU files were loaded. Check --start, --stride, and --max_steps.")
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
    requested_seq_len = args.seq_len or graph.num_steps
    seq_len = min(requested_seq_len, graph.num_steps, config.model.temporal.max_seq_len)
    if seq_len < 1:
        raise RuntimeError("Sequence length resolved to zero.")

    if seq_len < 2:
        g0 = graph.get_graph_at(0).to(device)
        repeat_len = min(4, config.model.temporal.max_seq_len)
        seq = [g0.clone() for _ in range(repeat_len)]
        gt_seq = f"repeated x{repeat_len} (only 1 timestep available)"
    else:
        seq = [graph.get_graph_at(i).to(device) for i in range(seq_len)]
        gt_seq = f"steps 0..{seq_len - 1}"
    print(f"  Sequence: {gt_seq}")

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    use_amp = device.type == "cuda" and not args.no_amp
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        out = model(seq)
    elapsed = time.time() - t0

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Forward: {elapsed:.2f}s, peak VRAM: {peak_mem:.2f} GB")
    else:
        print(f"  Forward: {elapsed:.2f}s")

    T_pred = out["T_pred"]
    print(f"  T_pred shape: {tuple(T_pred.shape)}")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
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
        print("       Point --vtu_dir to a directory containing multiple Data-*.vtu files.")

    print("\nPipeline OK. Ready to train.")


if __name__ == "__main__":
    main()
