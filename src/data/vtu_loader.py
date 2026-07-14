"""Flow-3D AM VTU file parser.

Parses VTU unstructured grid files (including appended/zlib-compressed binary)
using meshio. Extracts nodal coordinates, temperature, Live mask, Boundary
labels, and cell connectivity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import torch


@dataclass
class VTUData:
    coords: torch.Tensor
    temperature: torch.Tensor
    live: torch.Tensor
    boundary: torch.Tensor
    time: float
    step_index: int
    cells: list[tuple[str, np.ndarray]]


class VTULoader:
    def __init__(self, vtu_dir: str | Path, temperature_field: str = "Temperature",
                 live_field: str = "Live", boundary_field: str = "Boundary"):
        self.vtu_dir = Path(vtu_dir)
        self.temperature_field = temperature_field
        self.live_field = live_field
        self.boundary_field = boundary_field
        self._files: list[Path] = []
        self._scan()

    def _scan(self):
        pattern = re.compile(r"Data-(\d+)\.vtu", re.IGNORECASE)
        files = []
        for p in self.vtu_dir.glob("*.vtu"):
            m = pattern.match(p.name)
            if m:
                idx = int(m.group(1))
                files.append((idx, p))
        files.sort(key=lambda x: x[0])
        self._files = [f[1] for f in files]

    @property
    def num_steps(self) -> int:
        return len(self._files)

    def parse_single(self, filepath: Path) -> VTUData:
        mesh = meshio.read(str(filepath))

        coords = torch.from_numpy(mesh.points.astype(np.float32))

        step_str = re.search(r"Data-(\d+)", filepath.stem, re.IGNORECASE)
        step_idx = int(step_str.group(1)) if step_str else 0

        point_data = mesh.point_data

        temperature = self._extract_array(point_data, self.temperature_field)
        live = self._extract_array(point_data, self.live_field)
        boundary = self._extract_array(point_data, self.boundary_field)

        if temperature is None:
            T_mean = self._extract_array(point_data, "T_mean")
            T_fluid = self._extract_array(point_data, "Fluid_temperature")
            temperature = T_mean if T_mean is not None else T_fluid

        if temperature is None:
            max_val_check = [v for k, v in point_data.items()
                             if v is not None and np.max(np.abs(v)) > 1.0]
            if max_val_check:
                temperature = torch.from_numpy(
                    max_val_check[0].astype(np.float32).flatten()
                )

        if temperature is None:
            raise KeyError(
                f"Cannot find temperature field in VTU. "
                f"Available PointData: {list(point_data.keys())}"
            )

        temperature = self._to_tensor(temperature)

        if live is None:
            live = np.ones(len(mesh.points), dtype=np.float32)
        live = self._to_tensor(live)

        if boundary is None:
            boundary = np.zeros(len(mesh.points), dtype=np.float32)
        boundary = self._to_tensor(boundary)

        time_val = self._infer_time(mesh, step_idx)

        cells = []
        for cb in mesh.cells:
            cell_type = cb.type if hasattr(cb, "type") else cb[0]
            conn = cb.data if hasattr(cb, "data") else cb[1]
            cells.append((cell_type, np.asarray(conn)))

        return VTUData(
            coords=coords,
            temperature=temperature,
            live=live,
            boundary=boundary,
            time=time_val,
            step_index=step_idx,
            cells=cells,
        )

    def parse_sequence(self, verbose: bool = True) -> list[VTUData]:
        results = []
        for i, fp in enumerate(self._files):
            if verbose and (i == 0 or i == len(self._files) - 1 or i % 50 == 0):
                print(f"  Parsing {i+1}/{len(self._files)}: {fp.name}")
            results.append(self.parse_single(fp))
        if verbose:
            print(f"  Loaded {len(results)} time steps.")
        return results

    @staticmethod
    def _extract_array(point_data: dict, name: str) -> np.ndarray | None:
        if name in point_data and point_data[name] is not None:
            arr = point_data[name]
            if arr.ndim > 1:
                arr = arr[:, 0] if arr.shape[1] == 1 else arr
            return arr.astype(np.float64) if arr.dtype != np.float64 else arr
        return None

    @staticmethod
    def _to_tensor(arr: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if arr is None:
            return torch.empty(0)
        if isinstance(arr, torch.Tensor):
            return arr.float().flatten()
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).flatten()

    @staticmethod
    def _infer_time(mesh, step_idx: int) -> float:
        field_data = getattr(mesh, "field_data", {}) or {}
        for key in ["Time", "TIME", "time", "TimeValue", "time_value"]:
            if key in field_data:
                val = field_data[key]
                if isinstance(val, (list, np.ndarray)):
                    return float(val[0])
                return float(val)
        return float(step_idx)
