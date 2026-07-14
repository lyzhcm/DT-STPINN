"""Data preprocessing script.

Converts Flow-3D AM VTU files into HDF5 cached graph data for faster
training iterations.
"""
import argparse
import sys
from pathlib import Path

import torch
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.vtu_loader import VTULoader
from src.data.preprocessing import FeatureNormalizer


def main():
    parser = argparse.ArgumentParser(description="Preprocess VTU data for DT-STPINN")
    parser.add_argument("--config", type=str, default="configs/paper1.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--vtu_dir", type=str, default=None,
                        help="VTU directory (overrides config)")
    parser.add_argument("--output", type=str, default="data/processed/graph_cache.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--skip_normalize", action="store_true",
                        help="Skip feature normalization")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    vtu_dir = args.vtu_dir or config.data.vtu_dir

    print(f"Loading VTU files from: {vtu_dir}")
    loader = VTULoader(vtu_dir)
    print(f"Found {loader.num_steps} time steps.")

    vtu_data = loader.parse_sequence(verbose=True)
    ref = vtu_data[-1]

    all_temps = torch.stack([v.temperature for v in vtu_data])
    all_live = torch.stack([v.live for v in vtu_data])
    all_boundary = torch.stack([v.boundary for v in vtu_data])
    all_times = torch.tensor([v.time for v in vtu_data], dtype=torch.float32)

    normalizer = FeatureNormalizer()
    if not args.skip_normalize:
        normalizer.fit(ref.coords, all_temps)
        print(f"Temperature stats: mean={normalizer.mean['temperature']:.2f}, "
              f"std={normalizer.std['temperature']:.2f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_path}")
    with h5py.File(str(output_path), "w") as f:
        f.create_dataset("coords", data=ref.coords.numpy())
        f.create_dataset("temperatures", data=all_temps.numpy())
        f.create_dataset("live", data=all_live.numpy())
        f.create_dataset("boundary", data=all_boundary.numpy())
        f.create_dataset("times", data=all_times.numpy())

        f.create_dataset("temp_mean", data=normalizer.mean["temperature"].numpy())
        f.create_dataset("temp_std", data=normalizer.std["temperature"].numpy())

        if ref.cells:
            cells_grp = f.create_group("cells")
            for ci, (cell_type, conn) in enumerate(ref.cells):
                cells_grp.create_dataset(f"type_{ci}", data=cell_type.encode())
                cells_grp.create_dataset(f"conn_{ci}", data=conn)

    print("Done.")


if __name__ == "__main__":
    main()
