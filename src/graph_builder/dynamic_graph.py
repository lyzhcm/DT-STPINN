"""Dynamic Graph Builder for DED process.

Manages the evolving graph structure during additive manufacturing:
- Builds a fixed topology graph from the full computational mesh.
- Applies time-varying node masks based on material deposition (Live field).
- Generates per-time-step PyG Data objects with current features and masks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .node_features import NodeFeatureBuilder
from .edge_features import EdgeFeatureBuilder


class DynamicGraph:
    CACHE_VERSION = 1

    def __init__(self, vtu_data_list: list, material_props,
                 k_neighbors: int = 16, use_mesh_edges: bool = True):
        ref = vtu_data_list[-1]
        self.num_nodes = ref.coords.shape[0]
        self.num_steps = len(vtu_data_list)
        self.device = torch.device("cpu")

        self.coords = ref.coords.clone()
        self.times = torch.tensor([v.time for v in vtu_data_list], dtype=torch.float32)

        self.temperatures = torch.stack([v.temperature for v in vtu_data_list])
        self.live = torch.stack([v.live for v in vtu_data_list])
        self.boundary = ref.boundary.clone()

        self.layer_ids = self._infer_layers(self.coords)

        if use_mesh_edges and ref.cells:
            from ..data.preprocessing import extract_mesh_edges
            self.edge_index = torch.from_numpy(extract_mesh_edges(ref.cells))
        else:
            self.edge_index = self._build_knn_edges(k_neighbors)

        self.edge_index = self.edge_index.to(torch.long)

        self.node_feature_builder = NodeFeatureBuilder(material_props)
        self.edge_feature_builder = EdgeFeatureBuilder(material_props)

        k_ref = material_props.thermal_conductivity
        self.node_k = torch.full((self.num_nodes,), k_ref, dtype=torch.float32)

        self.edge_attr = self.edge_feature_builder.build(
            self.edge_index, self.coords, self.node_k
        )

        self._laser_positions = self._compute_laser_positions()
        self._scan_directions = self._compute_scan_directions()

    def _build_knn_edges(self, k: int) -> torch.Tensor:
        from torch_cluster import knn_graph
        edge = knn_graph(self.coords, k=k, loop=False)
        return edge

    def _infer_layers(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords[:, 2]
        z_min, z_max = z.min().item(), z.max().item()
        if z_max - z_min < 1e-6:
            return torch.zeros(coords.shape[0], dtype=torch.float32)
        z_norm = (z - z_min) / (z_max - z_min)
        n_layers = max(1, int(torch.ceil(z_norm.max() * 20).item()))
        layer_ids = (z_norm * (n_layers - 1)).long().float()
        return layer_ids

    def _compute_laser_positions(self) -> torch.Tensor:
        positions = []
        for t in range(self.num_steps):
            T = self.temperatures[t]
            live = self.live[t]
            if live.sum() > 0:
                T_masked = T.clone()
                T_masked[live < 0.5] = -float("inf")
                max_idx = torch.argmax(T_masked)
            else:
                max_idx = torch.argmax(T)
            positions.append(self.coords[max_idx].clone())
        return torch.stack(positions)

    def _compute_scan_directions(self) -> torch.Tensor:
        directions = [0.0]
        for t in range(1, self.num_steps):
            delta = self._laser_positions[t] - self._laser_positions[t - 1]
            if delta.norm() < 1e-8:
                directions.append(directions[-1])
            else:
                directions.append(float(torch.atan2(delta[1], delta[0])))
        return torch.tensor(directions, dtype=torch.float32)

    def get_active_mask(self, t: int) -> torch.Tensor:
        return self.live[t].clone()

    def get_graph_at(self, t: int) -> Data:
        coords_norm = self.coords.clone()
        T_raw = self.temperatures[t].clone()
        mask = self.get_active_mask(t)
        laser_pos = self._laser_positions[t].clone()
        scan_dir = self._scan_directions[t].item()
        time_val = self.times[t].item()

        dt = (self.times[t] - self.times[t - 1]).item() if t > 0 else 1.0

        x = self.node_feature_builder.build(
            coords_norm, T_raw, mask, self.layer_ids,
            laser_pos, scan_dir, time_val, dt
        )

        return Data(
            x=x,
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone(),
            y=T_raw.view(-1, 1),
            mask=mask.bool(),
            coords=coords_norm,
            boundary=self.boundary.clone(),
            laser_pos=laser_pos,
            dt=dt,
            time=time_val,
            step=t,
        )

    def get_sequence(self, start_t: int, length: int) -> list[Data]:
        return [self.get_graph_at(start_t + i) for i in range(length)]

    def to(self, device: torch.device) -> "DynamicGraph":
        self.device = device
        return self

    def to_cache_dict(self) -> dict:
        """Serialize preprocessed graph tensors for fast reload."""
        return {
            "cache_version": self.CACHE_VERSION,
            "num_nodes": self.num_nodes,
            "num_steps": self.num_steps,
            "coords": self.coords.cpu(),
            "times": self.times.cpu(),
            "temperatures": self.temperatures.cpu(),
            "live": self.live.cpu(),
            "boundary": self.boundary.cpu(),
            "layer_ids": self.layer_ids.cpu(),
            "edge_index": self.edge_index.cpu(),
            "edge_attr": self.edge_attr.cpu(),
            "node_k": self.node_k.cpu(),
            "laser_positions": self._laser_positions.cpu(),
            "scan_directions": self._scan_directions.cpu(),
        }

    @classmethod
    def from_cache_dict(cls, state: dict, material_props) -> "DynamicGraph":
        version = state.get("cache_version")
        if version != cls.CACHE_VERSION:
            raise ValueError(
                f"Unsupported DynamicGraph cache version {version}; "
                f"expected {cls.CACHE_VERSION}."
            )

        graph = cls.__new__(cls)
        graph.num_nodes = int(state["num_nodes"])
        graph.num_steps = int(state["num_steps"])
        graph.device = torch.device("cpu")

        graph.coords = state["coords"].cpu()
        graph.times = state["times"].cpu()
        graph.temperatures = state["temperatures"].cpu()
        graph.live = state["live"].cpu()
        graph.boundary = state["boundary"].cpu()
        graph.layer_ids = state["layer_ids"].cpu()
        graph.edge_index = state["edge_index"].cpu().to(torch.long)
        graph.edge_attr = state["edge_attr"].cpu()
        graph.node_k = state["node_k"].cpu()
        graph._laser_positions = state["laser_positions"].cpu()
        graph._scan_directions = state["scan_directions"].cpu()

        graph.node_feature_builder = NodeFeatureBuilder(material_props)
        graph.edge_feature_builder = EdgeFeatureBuilder(material_props)
        return graph
