"""Dynamic Graph Builder for DED process.

Manages the evolving graph structure during additive manufacturing:
- Builds a fixed topology graph from the full computational mesh.
- Applies time-varying node masks based on material deposition (Live field).
- Generates per-time-step PyG Data objects with current features and masks.
"""
from __future__ import annotations

import math
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
        self._laser_path_segment_starts = None
        self._laser_path_segment_ends = None
        self._laser_path_segment_start_times = None
        self._laser_path_segment_end_times = None
        self._laser_path_params = None

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

    def apply_laser_path_config(self, data_config) -> None:
        """Override estimated laser positions with a prescribed process path."""
        mode = getattr(data_config, "laser_path_mode", "estimated")
        xml_path = getattr(data_config, "laser_xml_path", None)
        if xml_path:
            mode = "additive_z_scan"
        if mode in (None, "", "estimated"):
            return
        if mode != "additive_z_scan":
            raise ValueError(f"Unsupported laser_path_mode: {mode}")

        if xml_path:
            from src.utils.laser_path import AdditiveZScanPath

            path = AdditiveZScanPath.from_xml(
                xml_path,
                time_scale_to_s=getattr(data_config, "laser_path_time_scale_to_s"),
                time_offset_s=getattr(data_config, "laser_path_time_offset_s", 0.0),
                alternate_layer_scan_direction=getattr(
                    data_config, "laser_alternate_layer_scan_direction", False
                ),
                reverse_hatch_order_parity=getattr(
                    data_config, "laser_reverse_hatch_order_parity", -1
                ),
            )
            self.apply_additive_z_scan_path(
                start_point_mm=path.start_point_mm,
                scan_direction=path.scan_direction,
                scan_length_mm=path.scan_length_mm,
                hatch_direction=path.hatch_direction,
                hatch_count=path.hatch_count,
                hatch_spacing_mm=path.hatch_spacing_mm,
                layer_count=path.layer_count,
                layer_thickness_mm=path.layer_thickness_mm,
                velocity_mm_s=path.velocity_mm_s,
                each_path_time_s=path.each_path_time_s,
                each_layer_time_s=path.each_layer_time_s,
                time_scale_to_s=path.time_scale_to_s,
                time_offset_s=path.time_offset_s,
                alternate_layer_scan_direction=path.alternate_layer_scan_direction,
                reverse_hatch_order_parity=path.reverse_hatch_order_parity,
            )
            return

        self.apply_additive_z_scan_path(
            start_point_mm=getattr(data_config, "laser_start_point_mm"),
            scan_direction=getattr(data_config, "laser_scan_direction"),
            scan_length_mm=getattr(data_config, "laser_scan_length_mm"),
            hatch_direction=getattr(data_config, "laser_hatch_direction"),
            hatch_count=getattr(data_config, "laser_hatch_count"),
            hatch_spacing_mm=getattr(data_config, "laser_hatch_spacing_mm"),
            layer_count=getattr(data_config, "laser_layer_count"),
            layer_thickness_mm=getattr(data_config, "laser_layer_thickness_mm"),
            velocity_mm_s=getattr(data_config, "laser_velocity_mm_s"),
            each_path_time_s=getattr(data_config, "laser_each_path_time_s"),
            each_layer_time_s=getattr(data_config, "laser_each_layer_time_s"),
            time_scale_to_s=getattr(data_config, "laser_path_time_scale_to_s"),
            time_offset_s=getattr(data_config, "laser_path_time_offset_s", 0.0),
            alternate_layer_scan_direction=getattr(
                data_config, "laser_alternate_layer_scan_direction", False
            ),
            reverse_hatch_order_parity=getattr(
                data_config, "laser_reverse_hatch_order_parity", -1
            ),
        )

    def apply_additive_z_scan_path(self, *, start_point_mm, scan_direction,
                                   scan_length_mm: float, hatch_direction,
                                   hatch_count: int, hatch_spacing_mm: float,
                                   layer_count: int, layer_thickness_mm: float,
                                    velocity_mm_s: float, each_path_time_s: float,
                                    each_layer_time_s: float,
                                    time_scale_to_s: float = 1.0e-3,
                                    time_offset_s: float = 0.0,
                                    alternate_layer_scan_direction: bool = False,
                                    reverse_hatch_order_parity: int = -1) -> None:
        """Generate the prescribed additive_z_scan serpentine laser path."""
        dtype = self.coords.dtype
        start = torch.tensor(start_point_mm, dtype=dtype)
        scan_dir = torch.tensor(scan_direction, dtype=dtype)
        hatch_dir = torch.tensor(hatch_direction, dtype=dtype)

        scan_dir = scan_dir / scan_dir.norm().clamp_min(1.0e-12)
        hatch_dir = hatch_dir / hatch_dir.norm().clamp_min(1.0e-12)
        hatch_count = max(1, int(hatch_count))
        layer_count = max(1, int(layer_count))

        scan_time = float(scan_length_mm) / max(float(velocity_mm_s), 1.0e-12)
        path_period = scan_time + max(float(each_path_time_s), 0.0)
        layer_period = path_period * hatch_count + max(float(each_layer_time_s), 0.0)
        total_tracks = hatch_count * layer_count

        positions: list[torch.Tensor] = []
        angles: list[float] = []
        segment_starts: list[torch.Tensor] = []
        segment_ends: list[torch.Tensor] = []
        segment_start_times: list[float] = []
        segment_end_times: list[float] = []
        t0 = float(self.times[0].item()) * float(time_scale_to_s)

        for layer_idx in range(layer_count):
            for track_in_layer in range(hatch_count):
                physical_track = track_in_layer
                if int(reverse_hatch_order_parity) in (0, 1):
                    if layer_idx % 2 == int(reverse_hatch_order_parity):
                        physical_track = hatch_count - 1 - track_in_layer

                layer_offset = torch.tensor([0.0, 0.0, layer_idx * float(layer_thickness_mm)],
                                            dtype=dtype)
                line_start = (
                    start
                    + hatch_dir * (physical_track * float(hatch_spacing_mm))
                    + layer_offset
                )

                forward = track_in_layer % 2 == 0
                if alternate_layer_scan_direction and layer_idx % 2 == 1:
                    forward = not forward

                if forward:
                    seg_start = line_start
                    seg_end = line_start + scan_dir * float(scan_length_mm)
                else:
                    seg_start = line_start + scan_dir * float(scan_length_mm)
                    seg_end = line_start

                track_start_s = (
                    layer_idx * layer_period + track_in_layer * path_period
                )
                abs_start_raw = (t0 + float(time_offset_s) + track_start_s) / float(time_scale_to_s)
                abs_end_raw = (t0 + float(time_offset_s) + track_start_s + scan_time) / float(time_scale_to_s)

                segment_starts.append(seg_start)
                segment_ends.append(seg_end)
                segment_start_times.append(abs_start_raw)
                segment_end_times.append(abs_end_raw)

        for raw_time in self.times.detach().cpu().tolist():
            elapsed = max(0.0, float(raw_time) * float(time_scale_to_s) - t0 - float(time_offset_s))
            if layer_period <= 0.0:
                global_track = 0
                local_path_time = 0.0
            else:
                layer_idx = min(int(elapsed // layer_period), layer_count - 1)
                layer_time = elapsed - layer_idx * layer_period
                track_in_layer = min(int(layer_time // path_period), hatch_count - 1)
                global_track = min(layer_idx * hatch_count + track_in_layer,
                                   total_tracks - 1)
                local_path_time = layer_time - track_in_layer * path_period

            layer_idx = global_track // hatch_count
            track_in_layer = global_track % hatch_count
            progress = min(max(local_path_time / scan_time, 0.0), 1.0) if scan_time > 0 else 1.0
            physical_track = track_in_layer
            if int(reverse_hatch_order_parity) in (0, 1):
                if layer_idx % 2 == int(reverse_hatch_order_parity):
                    physical_track = hatch_count - 1 - track_in_layer

            layer_offset = torch.tensor([0.0, 0.0, layer_idx * float(layer_thickness_mm)],
                                        dtype=dtype)
            line_start = (
                start
                + hatch_dir * (physical_track * float(hatch_spacing_mm))
                + layer_offset
            )

            forward = track_in_layer % 2 == 0
            if alternate_layer_scan_direction and layer_idx % 2 == 1:
                forward = not forward

            if forward:
                pos = line_start + scan_dir * (float(scan_length_mm) * progress)
                active_dir = scan_dir
            else:
                pos = line_start + scan_dir * (float(scan_length_mm) * (1.0 - progress))
                active_dir = -scan_dir

            positions.append(pos)
            angles.append(float(math.atan2(float(active_dir[1]), float(active_dir[0]))))

        self._laser_positions = torch.stack(positions).to(self.device)
        self._scan_directions = torch.tensor(angles, dtype=torch.float32, device=self.device)
        self._laser_path_segment_starts = torch.stack(segment_starts).to(self.device)
        self._laser_path_segment_ends = torch.stack(segment_ends).to(self.device)
        self._laser_path_segment_start_times = torch.tensor(
            segment_start_times, dtype=torch.float32, device=self.device
        )
        self._laser_path_segment_end_times = torch.tensor(
            segment_end_times, dtype=torch.float32, device=self.device
        )
        self._laser_path_params = {
            "start_point": start.to(self.device),
            "scan_direction": scan_dir.to(self.device),
            "hatch_direction": hatch_dir.to(self.device),
            "scan_length_mm": float(scan_length_mm),
            "hatch_count": hatch_count,
            "hatch_spacing_mm": float(hatch_spacing_mm),
            "layer_count": layer_count,
            "layer_thickness_mm": float(layer_thickness_mm),
            "velocity_mm_s": float(velocity_mm_s),
            "scan_time_s": scan_time,
            "path_period_s": path_period,
            "layer_period_s": layer_period,
            "time_scale_to_s": float(time_scale_to_s),
            "time_offset_s": float(time_offset_s),
            "alternate_layer_scan_direction": bool(alternate_layer_scan_direction),
            "reverse_hatch_order_parity": int(reverse_hatch_order_parity),
        }

    def get_active_mask(self, t: int) -> torch.Tensor:
        return self.live[t]

    def get_graph_at(self, t: int) -> Data:
        coords_norm = self.coords
        T_raw = self.temperatures[t]
        mask = self.get_active_mask(t)
        laser_pos = self._laser_positions[t]
        scan_dir = self._scan_directions[t].item()
        time_val = self.times[t].item()

        dt = (self.times[t] - self.times[t - 1]).item() if t > 0 else 1.0

        x = self.node_feature_builder.build(
            coords_norm, T_raw, mask, self.layer_ids,
            laser_pos, scan_dir, time_val, dt
        )

        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=T_raw.view(-1, 1),
            mask=mask.bool(),
            coords=coords_norm,
            boundary=self.boundary,
            laser_pos=laser_pos,
            dt=dt,
            time=time_val,
            step=t,
        )

    def get_sequence(self, start_t: int, length: int) -> list[Data]:
        return [self.get_graph_at(start_t + i) for i in range(length)]

    def to(self, device: torch.device) -> "DynamicGraph":
        self.device = device
        self.coords = self.coords.to(device)
        self.times = self.times.to(device)
        self.temperatures = self.temperatures.to(device)
        self.live = self.live.to(device)
        self.boundary = self.boundary.to(device)
        self.layer_ids = self.layer_ids.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.node_k = self.node_k.to(device)
        self._laser_positions = self._laser_positions.to(device)
        self._scan_directions = self._scan_directions.to(device)
        if self._laser_path_segment_starts is not None:
            self._laser_path_segment_starts = self._laser_path_segment_starts.to(device)
            self._laser_path_segment_ends = self._laser_path_segment_ends.to(device)
            self._laser_path_segment_start_times = self._laser_path_segment_start_times.to(device)
            self._laser_path_segment_end_times = self._laser_path_segment_end_times.to(device)
        if self._laser_path_params is not None:
            for key in ("start_point", "scan_direction", "hatch_direction"):
                self._laser_path_params[key] = self._laser_path_params[key].to(device)
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
            "laser_path_segment_starts": (
                None if self._laser_path_segment_starts is None
                else self._laser_path_segment_starts.cpu()
            ),
            "laser_path_segment_ends": (
                None if self._laser_path_segment_ends is None
                else self._laser_path_segment_ends.cpu()
            ),
            "laser_path_segment_start_times": (
                None if self._laser_path_segment_start_times is None
                else self._laser_path_segment_start_times.cpu()
            ),
            "laser_path_segment_end_times": (
                None if self._laser_path_segment_end_times is None
                else self._laser_path_segment_end_times.cpu()
            ),
            "laser_path_params": self._laser_path_params,
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
        graph._laser_path_segment_starts = state.get("laser_path_segment_starts")
        graph._laser_path_segment_ends = state.get("laser_path_segment_ends")
        graph._laser_path_segment_start_times = state.get("laser_path_segment_start_times")
        graph._laser_path_segment_end_times = state.get("laser_path_segment_end_times")
        graph._laser_path_params = state.get("laser_path_params")
        if graph._laser_path_segment_starts is not None:
            graph._laser_path_segment_starts = graph._laser_path_segment_starts.cpu()
            graph._laser_path_segment_ends = graph._laser_path_segment_ends.cpu()
            graph._laser_path_segment_start_times = graph._laser_path_segment_start_times.cpu()
            graph._laser_path_segment_end_times = graph._laser_path_segment_end_times.cpu()
        if graph._laser_path_params is not None:
            for key in ("start_point", "scan_direction", "hatch_direction"):
                graph._laser_path_params[key] = graph._laser_path_params[key].cpu()

        graph.node_feature_builder = NodeFeatureBuilder(material_props)
        graph.edge_feature_builder = EdgeFeatureBuilder(material_props)
        return graph
