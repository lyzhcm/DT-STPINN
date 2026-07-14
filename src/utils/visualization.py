"""VTK output writer for ParaView visualization of predicted temperature fields."""
from __future__ import annotations

import numpy as np
import torch
from meshio import Mesh
from meshio._mesh import CellBlock


def write_temperature_vtu(filepath: str, coords: torch.Tensor,
                          temperature: torch.Tensor,
                          cells: list[tuple[str, np.ndarray]] | None = None,
                          additional_fields: dict | None = None):
    """Write temperature field as VTU file for ParaView.

    Args:
        filepath: output path.
        coords: [N, 3] node coordinates.
        temperature: [N] temperature values.
        cells: optional cell connectivity from original mesh.
        additional_fields: optional dict of additional point data arrays.
    """
    coords_np = coords.detach().cpu().numpy().astype(np.float64)
    temp_np = temperature.detach().cpu().numpy().astype(np.float64).flatten()

    point_data = {"Temperature": temp_np}

    if additional_fields:
        for name, values in additional_fields.items():
            point_data[name] = values.detach().cpu().numpy().flatten().astype(np.float64)

    cell_blocks = []
    if cells:
        for cell_type, connectivity in cells:
            cell_blocks.append(CellBlock(cell_type, np.asarray(connectivity)))
    else:
        cell_blocks = [CellBlock("vertex", np.arange(len(coords_np)).reshape(-1, 1))]

    mesh = Mesh(points=coords_np, cells=cell_blocks, point_data=point_data)
    mesh.write(filepath)


def write_prediction_sequence(output_dir: str, prefix: str, times: list[float],
                              coords: torch.Tensor, temperatures: torch.Tensor,
                              cells: list | None = None):
    """Write a sequence of predicted temperature fields.

    Args:
        output_dir: directory for output VTU files.
        prefix: filename prefix.
        times: list of time values.
        coords: [N, 3] node coordinates.
        temperatures: [T, N] temperature predictions.
        cells: cell connectivity.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for t_idx, time_val in enumerate(times):
        fname = f"{prefix}_{t_idx:05d}.vtu"
        filepath = os.path.join(output_dir, fname)
        write_temperature_vtu(filepath, coords, temperatures[t_idx], cells)
