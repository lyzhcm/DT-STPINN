"""Data preprocessing utilities.

Normalization, log-spaced temporal sampling, dataset splitting,
and laser position estimation.
"""
from __future__ import annotations

import torch
import numpy as np


class FeatureNormalizer:
    def __init__(self):
        self.mean: dict[str, torch.Tensor] = {}
        self.std: dict[str, torch.Tensor] = {}

    def fit(self, coords: torch.Tensor, temperatures: torch.Tensor) -> "FeatureNormalizer":
        self.mean["coords"] = coords.mean(dim=0)
        self.std["coords"] = coords.std(dim=0).clamp(min=1e-8)
        self.mean["temperature"] = temperatures.mean()
        self.std["temperature"] = temperatures.std().clamp(min=1e-8)
        return self

    def transform_coords(self, coords: torch.Tensor) -> torch.Tensor:
        return (coords - self.mean["coords"]) / self.std["coords"]

    def transform_temperature(self, T: torch.Tensor) -> torch.Tensor:
        return (T - self.mean["temperature"]) / self.std["temperature"]

    def inverse_temperature(self, T_norm: torch.Tensor) -> torch.Tensor:
        return T_norm * self.std["temperature"] + self.mean["temperature"]

    def state_dict(self) -> dict:
        return {
            "coords_mean": self.mean["coords"],
            "coords_std": self.std["coords"],
            "temp_mean": self.mean["temperature"],
            "temp_std": self.std["temperature"],
        }

    def load_state_dict(self, state: dict):
        self.mean["coords"] = state["coords_mean"]
        self.std["coords"] = state["coords_std"]
        self.mean["temperature"] = state["temp_mean"]
        self.std["temperature"] = state["temp_std"]


def log_spaced_indices(total_steps: int, num_samples: int) -> list[int]:
    if num_samples >= total_steps:
        return list(range(total_steps))
    indices = np.unique(
        np.geomspace(1, total_steps, num_samples, dtype=int)
    ).tolist()
    return [i - 1 for i in indices]


def estimate_laser_position(temperature: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    max_idx = torch.argmax(temperature)
    return coords[max_idx]


def estimate_scan_direction(laser_pos_curr: torch.Tensor,
                            laser_pos_prev: torch.Tensor) -> float:
    delta = laser_pos_curr - laser_pos_prev
    if delta.norm() < 1e-8:
        return 0.0
    return float(torch.atan2(delta[1], delta[0]))


def split_indices(total_steps: int, train_ratio: float = 0.7,
                  val_ratio: float = 0.15) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(total_steps))
    n_train = int(total_steps * train_ratio)
    n_val = int(total_steps * val_ratio)
    train = indices[:n_train]
    val = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:]
    return train, val, test


def extract_mesh_edges(cells: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique undirected edges from mesh cell connectivity.

    Args:
        cells: list of (cell_type, connectivity) tuples from meshio.

    Returns:
        edge_index: [2, M] unique edges.
    """
    edges_set = set()

    for cell_type, connectivity in cells:
        conn = np.asarray(connectivity)
        if cell_type == "tetra" or cell_type == "tetra4":
            local_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        elif cell_type == "hexahedron" or cell_type == "hexa8":
            local_edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
        elif cell_type == "wedge" or cell_type == "wedge6":
            local_edges = [
                (0, 1), (1, 2), (2, 0),
                (3, 4), (4, 5), (5, 3),
                (0, 3), (1, 4), (2, 5),
            ]
        elif cell_type == "triangle" or cell_type == "triangle3":
            local_edges = [(0, 1), (1, 2), (2, 0)]
        elif cell_type == "quad" or cell_type == "quad4":
            local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        else:
            continue

        for a, b in local_edges:
            nodes_a = conn[:, a]
            nodes_b = conn[:, b]
            for i in range(len(nodes_a)):
                u, v = int(nodes_a[i]), int(nodes_b[i])
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))

    edges = np.array(sorted(edges_set), dtype=np.int64)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    return np.stack([row, col], axis=0)
