"""PyG Temporal Dataset for DED processing.

Creates sliding-window sequences from the DynamicGraph for training
the DT-STPINN model on temperature prediction.  Includes a stratified
window sampler that oversamples high-temperature melting windows to
combat the extreme class imbalance (~0.001 % nodes above solidus).
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset


class DEDTemporalDataset(Dataset):
    def __init__(self, dynamic_graph, window_size: int = 16,
                 predict_steps: int = 1, time_indices: list[int] | None = None,
                 use_target_laser_features: bool = False,
                 laser_feature_radius_mm: float = 0.4,
                 laser_feature_along_radius_mm: float = 0.0,
                 laser_feature_depth_mm: float = 0.1,
                 laser_feature_time_scale_to_s: float = 1.0e-3,
                 laser_feature_include_laser_coordinates: bool = False,
                 laser_feature_include_scan_geometry: bool = False,
                 laser_feature_include_sweep: bool = False,
                 laser_feature_include_exposure: bool = False,
                 laser_feature_exposure_source: str = "samples",
                 laser_feature_exposure_past_steps: int = 4,
                 laser_feature_exposure_future_steps: int = 0,
                 laser_feature_exposure_time_decay_s: float = 0.05,
                 laser_feature_exposure_use_segments: bool = False,
                 laser_feature_include_exposure_split: bool = False,
                 laser_feature_include_arrival_time: bool = False,
                 laser_feature_arrival_time_decay_s: float = 0.08,
                 laser_feature_include_neighbor_temp: bool = False,
                 laser_feature_include_neighbor_hot_stats: bool = False,
                 laser_feature_neighbor_hot_threshold: float = 1604.85,
                 laser_feature_include_neighbor_warm_stats: bool = False,
                 laser_feature_neighbor_warm_threshold: float = 1000.0,
                 laser_feature_include_body_source: bool = False,
                 laser_body_radius_mm: float = 0.1,
                 laser_body_height_mm: float = 0.1,
                 laser_body_radius_front_mm: float = 0.4,
                 laser_body_radius_back_mm: float = 0.4,
                 laser_body_coeff_front: float = 1.0,
                 laser_body_coeff_back: float = 1.0,
                  laser_feature_include_path_arrival: bool = False,
                  laser_feature_path_arrival_time_decay_s: float = 0.14,
                  laser_feature_path_arrival_gate_mode: str = "symmetric",
                  laser_feature_path_arrival_neighbor_tracks: bool = False,
                  laser_feature_include_path_phase: bool = False,
                  laser_feature_include_path_coordinates: bool = False,
                  laser_feature_include_path_timing: bool = False,
                  laser_feature_include_path_body_support: bool = False,
                  laser_feature_include_path_endpoint: bool = False,
                  laser_feature_endpoint_radius_mm: float = 0.75,
                  laser_feature_endpoint_time_decay_s: float = 0.075,
                  laser_feature_include_path_wake: bool = False,
                  laser_feature_wake_cross_radius_mm: float = 0.22,
                  laser_feature_wake_tail_decay_mm: float = 1.25,
                  laser_feature_wake_lead_decay_mm: float = 0.45,
                  laser_feature_wake_time_decay_s: float = 0.08):
        self.graph = dynamic_graph
        self.window_size = window_size
        self.predict_steps = predict_steps
        self.use_target_laser_features = use_target_laser_features
        self.laser_feature_radius_mm = max(float(laser_feature_radius_mm), 1.0e-6)
        along_radius = float(laser_feature_along_radius_mm)
        if along_radius <= 0.0:
            along_radius = self.laser_feature_radius_mm
        self.laser_feature_along_radius_mm = max(along_radius, 1.0e-6)
        self.laser_feature_depth_mm = max(float(laser_feature_depth_mm), 1.0e-6)
        self.laser_feature_time_scale_to_s = float(laser_feature_time_scale_to_s)
        self.laser_feature_include_laser_coordinates = bool(
            laser_feature_include_laser_coordinates
        )
        self.laser_feature_include_scan_geometry = bool(
            laser_feature_include_scan_geometry
        )
        self.laser_feature_include_sweep = bool(laser_feature_include_sweep)
        self.laser_feature_include_exposure = bool(laser_feature_include_exposure)
        self.laser_feature_exposure_source = str(laser_feature_exposure_source).lower()
        self.laser_feature_exposure_past_steps = max(
            0, int(laser_feature_exposure_past_steps)
        )
        self.laser_feature_exposure_future_steps = max(
            0, int(laser_feature_exposure_future_steps)
        )
        self.laser_feature_exposure_time_decay_s = max(
            float(laser_feature_exposure_time_decay_s), 1.0e-6
        )
        self.laser_feature_exposure_use_segments = bool(
            laser_feature_exposure_use_segments
        )
        self.laser_feature_include_exposure_split = bool(
            laser_feature_include_exposure_split
        )
        self.laser_feature_include_arrival_time = bool(
            laser_feature_include_arrival_time
        )
        self.laser_feature_arrival_time_decay_s = max(
            float(laser_feature_arrival_time_decay_s), 1.0e-6
        )
        self.laser_feature_include_neighbor_temp = bool(laser_feature_include_neighbor_temp)
        self.laser_feature_include_neighbor_hot_stats = bool(
            laser_feature_include_neighbor_hot_stats
        )
        self.laser_feature_neighbor_hot_threshold = float(
            laser_feature_neighbor_hot_threshold
        )
        self.laser_feature_include_neighbor_warm_stats = bool(
            laser_feature_include_neighbor_warm_stats
        )
        self.laser_feature_neighbor_warm_threshold = float(
            laser_feature_neighbor_warm_threshold
        )
        self.laser_feature_include_body_source = bool(
            laser_feature_include_body_source
        )
        self.laser_body_radius_mm = max(float(laser_body_radius_mm), 1.0e-6)
        self.laser_body_height_mm = max(float(laser_body_height_mm), 1.0e-6)
        self.laser_body_radius_front_mm = max(
            float(laser_body_radius_front_mm), 1.0e-6
        )
        self.laser_body_radius_back_mm = max(
            float(laser_body_radius_back_mm), 1.0e-6
        )
        self.laser_body_coeff_front = float(laser_body_coeff_front)
        self.laser_body_coeff_back = float(laser_body_coeff_back)
        self.laser_feature_include_path_arrival = bool(
            laser_feature_include_path_arrival
        )
        self.laser_feature_path_arrival_time_decay_s = max(
            float(laser_feature_path_arrival_time_decay_s), 1.0e-6
        )
        self.laser_feature_path_arrival_gate_mode = str(
            laser_feature_path_arrival_gate_mode
        ).lower()
        self.laser_feature_path_arrival_neighbor_tracks = bool(
            laser_feature_path_arrival_neighbor_tracks
        )
        self.laser_feature_include_path_phase = bool(
            laser_feature_include_path_phase
        )
        self.laser_feature_include_path_coordinates = bool(
            laser_feature_include_path_coordinates
        )
        self.laser_feature_include_path_timing = bool(
            laser_feature_include_path_timing
        )
        self.laser_feature_include_path_body_support = bool(
            laser_feature_include_path_body_support
        )
        self.laser_feature_include_path_endpoint = bool(
            laser_feature_include_path_endpoint
        )
        self.laser_feature_endpoint_radius_mm = max(
            float(laser_feature_endpoint_radius_mm), 1.0e-6
        )
        self.laser_feature_endpoint_time_decay_s = max(
            float(laser_feature_endpoint_time_decay_s), 1.0e-6
        )
        self.laser_feature_include_path_wake = bool(
            laser_feature_include_path_wake
        )
        self.laser_feature_wake_cross_radius_mm = max(
            float(laser_feature_wake_cross_radius_mm), 1.0e-6
        )
        self.laser_feature_wake_tail_decay_mm = max(
            float(laser_feature_wake_tail_decay_mm), 1.0e-6
        )
        self.laser_feature_wake_lead_decay_mm = max(
            float(laser_feature_wake_lead_decay_mm), 1.0e-6
        )
        self.laser_feature_wake_time_decay_s = max(
            float(laser_feature_wake_time_decay_s), 1.0e-6
        )
        if self.laser_feature_path_arrival_gate_mode not in {
                "symmetric", "past", "future"}:
            raise ValueError(
                "laser_feature_path_arrival_gate_mode must be "
                "'symmetric', 'past', or 'future'."
            )

        total = dynamic_graph.num_steps
        if time_indices is None:
            time_indices = list(range(total))

        self.valid_starts = []
        for idx in time_indices:
            if idx + window_size + predict_steps <= total:
                self.valid_starts.append(idx)

    def __len__(self) -> int:
        return len(self.valid_starts)

    @property
    def target_laser_feature_dim(self) -> int:
        """Number of laser lookahead features appended to each input frame."""
        if not self.use_target_laser_features:
            return 0

        dim = 9
        if self.laser_feature_include_laser_coordinates:
            dim += 6
        if self.laser_feature_include_scan_geometry:
            dim += 2
        if self.laser_feature_include_body_source:
            dim += 5
        if self.laser_feature_include_sweep:
            dim += 4
        if self.laser_feature_include_exposure:
            dim += 8
            if self.laser_feature_include_body_source:
                dim += 3
            if self.laser_feature_include_exposure_split:
                dim += 3
            if self.laser_feature_include_arrival_time:
                dim += 8
        if self.laser_feature_include_path_arrival:
            dim += 8
        if self.laser_feature_include_path_phase:
            dim += 10
        if self.laser_feature_include_path_coordinates:
            dim += 12
            if self.laser_feature_include_path_timing:
                dim += 4
        if self.laser_feature_include_path_body_support:
            dim += 5
        if self.laser_feature_include_path_endpoint:
            dim += 6
        if self.laser_feature_include_path_wake:
            dim += 6
        if self.laser_feature_include_neighbor_temp:
            dim += 3
        if self.laser_feature_include_neighbor_hot_stats:
            dim += 2
        if self.laser_feature_include_neighbor_warm_stats:
            dim += 2
        return dim

    @property
    def input_feature_dim(self) -> int:
        return self.graph.node_feature_builder.feature_dim + self.target_laser_feature_dim

    def _path_candidate_tracks(self, physical_track: torch.Tensor,
                               hatch_count: int) -> list[torch.Tensor]:
        if not self.laser_feature_path_arrival_neighbor_tracks:
            return [physical_track]
        return [
            (physical_track - 1).clamp(0, hatch_count - 1),
            physical_track,
            (physical_track + 1).clamp(0, hatch_count - 1),
        ]

    @staticmethod
    def _pick_path_candidate(features: list[torch.Tensor],
                             best_idx: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(features, dim=0)
        gather_idx = best_idx.unsqueeze(0).expand_as(stacked[:1])
        return stacked.gather(0, gather_idx).squeeze(0)

    def __getitem__(self, idx: int):
        start_t = self.valid_starts[idx]
        seq = self.graph.get_sequence(start_t, self.window_size)
        target_t = start_t + self.window_size + self.predict_steps - 1
        target_data = self.graph.get_graph_at(target_t)

        if self.use_target_laser_features:
            self._append_target_laser_features(seq, target_t)

        prev_temp = seq[-1].y.clone()
        dt = self.graph.times[target_t].item() - self.graph.times[target_t - 1].item()
        if dt <= 0:
            dt = 1.0

        return {
            "graph_sequence": seq,
            "target": target_data.y,
            "target_mask": target_data.mask,
            "coords": target_data.coords,
            "edge_index": target_data.edge_index,
            "edge_attr": target_data.edge_attr,
            "prev_temp": prev_temp,
            "dt": dt,
            "target_laser_pos": target_data.laser_pos,
            "target_step": target_t,
            "target_time": self.graph.times[target_t].item(),
        }

    def _append_target_laser_features(self, seq: list, target_t: int) -> None:
        """Append known target-step laser trajectory features to input frames.

        The target laser path is a prescribed process input.  It gives the
        model causal lookahead about where energy will be applied without
        exposing the target temperature label.
        """
        target_laser = self.graph._laser_positions[target_t]
        target_scan = self.graph._scan_directions[target_t]
        target_time = self.graph.times[target_t]
        exposure_features_cache = None

        for data in seq:
            coords = data.coords
            dtype = coords.dtype
            device = coords.device
            laser = target_laser.to(device=device, dtype=dtype)

            delta = coords - laser.view(1, 3)
            dxy = torch.linalg.vector_norm(delta[:, :2], dim=1, keepdim=True)
            d3 = torch.linalg.vector_norm(delta, dim=1, keepdim=True)

            radius = torch.as_tensor(self.laser_feature_radius_mm, device=device, dtype=dtype)
            along_radius = torch.as_tensor(
                self.laser_feature_along_radius_mm, device=device, dtype=dtype
            )
            depth = torch.as_tensor(self.laser_feature_depth_mm, device=device, dtype=dtype)
            scan = target_scan.to(device=device, dtype=dtype)
            scan_cos = torch.cos(scan)
            scan_sin = torch.sin(scan)
            along_xy = delta[:, 0:1] * scan_cos + delta[:, 1:2] * scan_sin
            cross_xy = -delta[:, 0:1] * scan_sin + delta[:, 1:2] * scan_cos
            heat_proxy = torch.exp(
                -2.0 * cross_xy.pow(2) / radius.pow(2)
                -2.0 * along_xy.pow(2) / along_radius.pow(2)
                - delta[:, 2:3].pow(2) / depth.pow(2)
            )

            lead_dt = (target_time - torch.as_tensor(data.time, device=target_time.device))
            lead_dt = lead_dt.to(device=device, dtype=dtype) * self.laser_feature_time_scale_to_s
            lead_dt = lead_dt.expand(coords.shape[0], 1)

            target_scan_sin = torch.sin(scan).expand(coords.shape[0], 1)
            target_scan_cos = torch.cos(scan).expand(coords.shape[0], 1)

            features = [
                delta,
                dxy,
                d3,
                heat_proxy,
                lead_dt,
                target_scan_sin,
                target_scan_cos,
            ]

            if self.laser_feature_include_laser_coordinates:
                current_laser = data.laser_pos.to(device=device, dtype=dtype)
                features.extend([
                    current_laser.view(1, 3).expand(coords.shape[0], 3),
                    laser.view(1, 3).expand(coords.shape[0], 3),
                ])

            if self.laser_feature_include_scan_geometry:
                features.extend([along_xy, cross_xy])

            if self.laser_feature_include_body_source:
                body_heat, body_gate, body_along_norm, body_cross_norm, body_depth_norm = (
                    self._ellipsoid_body_source_terms(
                        along=along_xy,
                        cross=cross_xy,
                        z_delta=delta[:, 2:3],
                        device=device,
                        dtype=dtype,
                    )
                )
                features.extend([
                    body_heat,
                    body_gate,
                    body_along_norm,
                    body_cross_norm,
                    body_depth_norm,
                ])

            if self.laser_feature_include_sweep:
                source_laser = data.laser_pos.to(device=device, dtype=dtype)
                segment = laser - source_laser
                seg_len_sq = segment.dot(segment).clamp_min(1.0e-12)
                rel = coords - source_laser.view(1, 3)
                progress = (rel @ segment.view(3, 1) / seg_len_sq).clamp(0.0, 1.0)
                closest = source_laser.view(1, 3) + progress * segment.view(1, 3)
                sweep_delta = coords - closest
                sweep_dxy = torch.linalg.vector_norm(sweep_delta[:, :2], dim=1, keepdim=True)
                sweep_d3 = torch.linalg.vector_norm(sweep_delta, dim=1, keepdim=True)
                sweep_heat_proxy = torch.exp(
                    -2.0 * sweep_dxy.pow(2) / radius.pow(2)
                    - sweep_delta[:, 2:3].pow(2) / depth.pow(2)
                )
                features.extend([sweep_dxy, sweep_d3, sweep_heat_proxy, progress])

            if self.laser_feature_include_exposure:
                if exposure_features_cache is None:
                    exposure_features_cache = self._laser_exposure_features(
                        coords=coords,
                        target_t=target_t,
                        target_time=target_time,
                        radius=radius,
                        along_radius=along_radius,
                        depth=depth,
                        device=device,
                        dtype=dtype,
                    )
                features.extend(exposure_features_cache)

            if self.laser_feature_include_path_arrival:
                features.extend(self._analytic_path_arrival_features(
                    coords=coords,
                    target_time=target_time,
                    radius=radius,
                    along_radius=along_radius,
                    depth=depth,
                    device=device,
                    dtype=dtype,
                ))

            if self.laser_feature_include_path_phase:
                features.extend(self._analytic_path_phase_features(
                    coords=coords,
                    target_time=target_time,
                    radius=radius,
                    depth=depth,
                    device=device,
                    dtype=dtype,
                ))

            if self.laser_feature_include_path_coordinates:
                features.extend(self._analytic_path_coordinate_features(
                    coords=coords,
                    target_time=target_time,
                    depth=depth,
                    device=device,
                    dtype=dtype,
                ))

            if self.laser_feature_include_path_body_support:
                features.extend(self._analytic_path_body_support_features(
                    coords=coords,
                    device=device,
                    dtype=dtype,
                ))

            if self.laser_feature_include_path_endpoint:
                features.extend(self._analytic_path_endpoint_features(
                    coords=coords,
                    target_time=target_time,
                    depth=depth,
                    device=device,
                    dtype=dtype,
                ))

            if self.laser_feature_include_path_wake:
                features.extend(self._analytic_path_wake_features(
                    coords=coords,
                    target_time=target_time,
                    depth=depth,
                    device=device,
                    dtype=dtype,
                ))

            if self.laser_feature_include_neighbor_temp:
                features.extend(self._neighbor_temperature_features(data))
            if self.laser_feature_include_neighbor_hot_stats:
                features.extend(self._neighbor_hot_stats_features(data))
            if self.laser_feature_include_neighbor_warm_stats:
                features.extend(self._neighbor_warm_stats_features(data))

            lookahead = torch.cat(features, dim=1)
            data.x = torch.cat([data.x, lookahead], dim=1)
            data.target_laser_pos = laser

    def _laser_exposure_features(self, *, coords: torch.Tensor, target_t: int,
                                 target_time: torch.Tensor, radius: torch.Tensor,
                                 along_radius: torch.Tensor, depth: torch.Tensor,
                                 device, dtype) -> list[torch.Tensor]:
        """Summarise recent/near-target laser exposure for every node.

        Melt-pool temperatures can lag the laser position by a few saved frames.
        These features make that process input explicit: nearest trajectory
        point, elapsed time since that point, and a decayed heat proxy integral.
        """
        if (
            self.laser_feature_exposure_source in {"path", "path_segments", "xml"}
            and self._laser_path_segments_available()
        ):
            return self._laser_path_segment_exposure_features(
                coords=coords,
                target_t=target_t,
                target_time=target_time,
                radius=radius,
                along_radius=along_radius,
                depth=depth,
                device=device,
                dtype=dtype,
            )

        start = max(0, target_t - self.laser_feature_exposure_past_steps)
        end = min(
            self.graph.num_steps - 1,
            target_t + self.laser_feature_exposure_future_steps,
        )
        laser_positions = self.graph._laser_positions[start:end + 1].to(
            device=device, dtype=dtype,
        )
        laser_times = self.graph.times[start:end + 1].to(device=device, dtype=dtype)

        if self.laser_feature_exposure_use_segments and laser_positions.shape[0] >= 2:
            return self._laser_sampled_segment_exposure_features(
                coords=coords,
                laser_positions=laser_positions,
                laser_times=laser_times,
                target_time=target_time,
                radius=radius,
                along_radius=along_radius,
                depth=depth,
                device=device,
                dtype=dtype,
            )

        delta = coords.unsqueeze(1) - laser_positions.unsqueeze(0)
        dxy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        d3 = torch.linalg.vector_norm(delta, dim=-1)
        z_delta = delta[..., 2]
        body_heat_proxy = None
        heat_proxy = torch.exp(
            -2.0 * dxy.pow(2) / radius.pow(2)
            - z_delta.pow(2) / depth.pow(2)
        )
        if self.laser_feature_include_body_source:
            scan_angles = self.graph._scan_directions[start:end + 1].to(
                device=device, dtype=dtype,
            )
            scan_cos = torch.cos(scan_angles).view(1, -1)
            scan_sin = torch.sin(scan_angles).view(1, -1)
            along = delta[..., 0] * scan_cos + delta[..., 1] * scan_sin
            cross = -delta[..., 0] * scan_sin + delta[..., 1] * scan_cos
            body_heat_proxy = self._ellipsoid_body_source_terms(
                along=along,
                cross=cross,
                z_delta=z_delta,
                device=device,
                dtype=dtype,
            )[0]

        target_time_t = target_time.to(device=device, dtype=dtype)
        rel_dt = (target_time_t - laser_times) * self.laser_feature_time_scale_to_s
        decay = torch.exp(-rel_dt.abs() / self.laser_feature_exposure_time_decay_s)
        decayed_heat = heat_proxy * decay.view(1, -1)
        node_idx = torch.arange(coords.shape[0], device=device)
        body_features = []
        if body_heat_proxy is not None:
            decayed_body_heat = body_heat_proxy * decay.view(1, -1)
            body_best_idx = decayed_body_heat.argmax(dim=1)
            body_best_heat = body_heat_proxy[node_idx, body_best_idx].unsqueeze(1)
            body_exposure_integral = decayed_body_heat.sum(dim=1, keepdim=True)
            body_best_dt = rel_dt[body_best_idx].unsqueeze(1)
            body_features = [body_best_heat, body_exposure_integral, body_best_dt]

        best_idx = decayed_heat.argmax(dim=1)
        best_delta = delta[node_idx, best_idx]
        best_dxy = dxy[node_idx, best_idx].unsqueeze(1)
        best_d3 = d3[node_idx, best_idx].unsqueeze(1)
        best_heat = heat_proxy[node_idx, best_idx].unsqueeze(1)
        best_dt = rel_dt[best_idx].unsqueeze(1)
        exposure_integral = decayed_heat.sum(dim=1, keepdim=True)

        split_features = []
        if self.laser_feature_include_exposure_split:
            rel_dt_by_node = rel_dt.view(1, -1).expand_as(decayed_heat)
            past_integral = decayed_heat.masked_fill(rel_dt_by_node < 0, 0.0).sum(
                dim=1, keepdim=True
            )
            future_integral = decayed_heat.masked_fill(rel_dt_by_node >= 0, 0.0).sum(
                dim=1, keepdim=True
            )
            best_abs_dt = best_dt.abs()
            split_features = [past_integral, future_integral, best_abs_dt]

        arrival_features = []
        if self.laser_feature_include_arrival_time:
            geom_score = (
                dxy.pow(2) / radius.pow(2)
                + z_delta.pow(2) / depth.pow(2)
            )
            arrival_idx = geom_score.argmin(dim=1)
            arrival_dt = rel_dt[arrival_idx].unsqueeze(1)
            arrival_abs_dt = arrival_dt.abs()
            time_until_hit = (-arrival_dt).clamp_min(0.0)
            time_since_hit = arrival_dt.clamp_min(0.0)
            arrival_dxy = dxy[node_idx, arrival_idx].unsqueeze(1)
            arrival_along_outside = torch.zeros_like(arrival_dxy)
            arrival_abs_dz = z_delta[node_idx, arrival_idx].abs().unsqueeze(1)
            arrival_spatial_gate = heat_proxy[node_idx, arrival_idx].unsqueeze(1)
            arrival_gate = arrival_spatial_gate * torch.exp(
                -arrival_abs_dt / self.laser_feature_arrival_time_decay_s
            )
            arrival_features = [
                arrival_dt,
                arrival_abs_dt,
                time_until_hit,
                time_since_hit,
                arrival_dxy,
                arrival_along_outside,
                arrival_abs_dz,
                arrival_gate,
            ]

        return [
            best_delta,
            best_dxy,
            best_d3,
            best_heat,
            exposure_integral,
            best_dt,
            *body_features,
            *split_features,
            *arrival_features,
        ]

    def _ellipsoid_body_source_terms(self, *, along: torch.Tensor,
                                     cross: torch.Tensor,
                                     z_delta: torch.Tensor,
                                     device, dtype) -> tuple[torch.Tensor, ...]:
        radius = torch.as_tensor(self.laser_body_radius_mm, device=device, dtype=dtype)
        height = torch.as_tensor(self.laser_body_height_mm, device=device, dtype=dtype)
        front_radius = torch.as_tensor(
            self.laser_body_radius_front_mm, device=device, dtype=dtype
        )
        back_radius = torch.as_tensor(
            self.laser_body_radius_back_mm, device=device, dtype=dtype
        )
        front_coeff = torch.as_tensor(
            self.laser_body_coeff_front, device=device, dtype=dtype
        )
        back_coeff = torch.as_tensor(
            self.laser_body_coeff_back, device=device, dtype=dtype
        )

        front = along >= 0.0
        along_radius = torch.where(front, front_radius, back_radius)
        coeff = torch.where(front, front_coeff, back_coeff)
        cross_norm = cross.abs() / radius
        along_norm = along.abs() / along_radius
        depth_norm = z_delta.abs() / height
        body_score = cross_norm.pow(2) + along_norm.pow(2) + depth_norm.pow(2)
        body_heat = coeff * torch.exp(
            -2.0 * cross_norm.pow(2)
            -2.0 * along_norm.pow(2)
            -depth_norm.pow(2)
        )
        body_gate = (body_score <= 1.0).to(dtype=dtype)
        return body_heat, body_gate, along_norm, cross_norm, depth_norm

    def _laser_path_segments_available(self) -> bool:
        return (
            getattr(self.graph, "_laser_path_segment_starts", None) is not None
            and getattr(self.graph, "_laser_path_segment_ends", None) is not None
            and getattr(self.graph, "_laser_path_segment_start_times", None) is not None
            and getattr(self.graph, "_laser_path_segment_end_times", None) is not None
        )

    def _laser_path_segment_exposure_features(self, *, coords: torch.Tensor,
                                             target_t: int,
                                             target_time: torch.Tensor,
                                             radius: torch.Tensor,
                                             along_radius: torch.Tensor,
                                             depth: torch.Tensor,
                                             device, dtype) -> list[torch.Tensor]:
        """Summarise exposure to configured additive_z_scan line segments."""
        start_idx = max(0, target_t - self.laser_feature_exposure_past_steps)
        end_idx = min(
            self.graph.num_steps - 1,
            target_t + self.laser_feature_exposure_future_steps,
        )
        window_start = self.graph.times[start_idx].to(device=device, dtype=dtype)
        window_end = self.graph.times[end_idx].to(device=device, dtype=dtype)

        segment_start_times = self.graph._laser_path_segment_start_times.to(
            device=device, dtype=dtype
        )
        segment_end_times = self.graph._laser_path_segment_end_times.to(
            device=device, dtype=dtype
        )
        in_window = (segment_end_times >= window_start) & (segment_start_times <= window_end)

        if not in_window.any():
            segment_mid_times = 0.5 * (segment_start_times + segment_end_times)
            nearest = torch.argmin((segment_mid_times - target_time.to(device=device, dtype=dtype)).abs())
            in_window = torch.zeros_like(segment_mid_times, dtype=torch.bool)
            in_window[nearest] = True

        segment_starts = self.graph._laser_path_segment_starts.to(device=device, dtype=dtype)
        segment_ends = self.graph._laser_path_segment_ends.to(device=device, dtype=dtype)

        return self._timed_segment_exposure_features(
            coords=coords,
            segment_start=segment_starts[in_window],
            segment_end=segment_ends[in_window],
            segment_start_times=segment_start_times[in_window],
            segment_end_times=segment_end_times[in_window],
            target_time=target_time,
            radius=radius,
            along_radius=along_radius,
            depth=depth,
            device=device,
            dtype=dtype,
        )

    def _laser_sampled_segment_exposure_features(self, *, coords: torch.Tensor,
                                                laser_positions: torch.Tensor,
                                                laser_times: torch.Tensor,
                                                target_time: torch.Tensor,
                                                radius: torch.Tensor,
                                                along_radius: torch.Tensor,
                                                depth: torch.Tensor,
                                                device, dtype) -> list[torch.Tensor]:
        """Summarise exposure to swept laser segments between saved frames."""
        return self._timed_segment_exposure_features(
            coords=coords,
            segment_start=laser_positions[:-1],
            segment_end=laser_positions[1:],
            segment_start_times=laser_times[:-1],
            segment_end_times=laser_times[1:],
            target_time=target_time,
            radius=radius,
            along_radius=along_radius,
            depth=depth,
            device=device,
            dtype=dtype,
        )

    def _timed_segment_exposure_features(self, *, coords: torch.Tensor,
                                         segment_start: torch.Tensor,
                                         segment_end: torch.Tensor,
                                         segment_start_times: torch.Tensor,
                                         segment_end_times: torch.Tensor,
                                         target_time: torch.Tensor,
                                         radius: torch.Tensor,
                                         along_radius: torch.Tensor,
                                         depth: torch.Tensor,
                                         device, dtype) -> list[torch.Tensor]:
        segment = segment_end - segment_start
        seg_len_sq = segment.pow(2).sum(dim=1).clamp_min(1.0e-12)

        rel = coords.unsqueeze(1) - segment_start.unsqueeze(0)
        progress = (
            (rel * segment.unsqueeze(0)).sum(dim=-1) / seg_len_sq.view(1, -1)
        ).clamp(0.0, 1.0)
        closest = segment_start.unsqueeze(0) + progress.unsqueeze(-1) * segment.unsqueeze(0)

        delta = coords.unsqueeze(1) - closest
        dxy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        d3 = torch.linalg.vector_norm(delta, dim=-1)
        z_delta = delta[..., 2]

        seg_len = torch.sqrt(seg_len_sq)
        segment_xy = segment[:, :2]
        seg_xy_len = torch.linalg.vector_norm(segment_xy, dim=1).clamp_min(1.0e-12)
        dir_xy = segment_xy / seg_xy_len.unsqueeze(-1)
        raw_along = (rel[..., :2] * dir_xy.unsqueeze(0)).sum(dim=-1)
        along_before = raw_along.clamp_max(0.0)
        along_after = (raw_along - seg_len.view(1, -1)).clamp_min(0.0)
        along_outside = along_before + along_after
        cross_delta_xy = delta[..., :2] - along_outside.unsqueeze(-1) * dir_xy.unsqueeze(0)
        cross_xy = torch.linalg.vector_norm(cross_delta_xy, dim=-1)

        heat_proxy = torch.exp(
            -2.0 * cross_xy.pow(2) / radius.pow(2)
            -2.0 * along_outside.pow(2) / along_radius.pow(2)
            - z_delta.pow(2) / depth.pow(2)
        )
        body_heat_proxy = None
        if self.laser_feature_include_body_source:
            body_heat_proxy = self._ellipsoid_body_source_terms(
                along=along_outside,
                cross=cross_xy,
                z_delta=z_delta,
                device=device,
                dtype=dtype,
            )[0]

        segment_time_delta = (segment_end_times - segment_start_times).view(1, -1)
        closest_times = segment_start_times.view(1, -1) + progress * segment_time_delta
        target_time_t = target_time.to(device=device, dtype=dtype)
        rel_dt = (target_time_t - closest_times) * self.laser_feature_time_scale_to_s
        decay = torch.exp(-rel_dt.abs() / self.laser_feature_exposure_time_decay_s)
        decayed_heat = heat_proxy * decay

        best_idx = decayed_heat.argmax(dim=1)
        node_idx = torch.arange(coords.shape[0], device=device)
        body_features = []
        if body_heat_proxy is not None:
            decayed_body_heat = body_heat_proxy * decay
            body_best_idx = decayed_body_heat.argmax(dim=1)
            body_best_heat = body_heat_proxy[node_idx, body_best_idx].unsqueeze(1)
            body_exposure_integral = decayed_body_heat.sum(dim=1, keepdim=True)
            body_best_dt = rel_dt[node_idx, body_best_idx].unsqueeze(1)
            body_features = [body_best_heat, body_exposure_integral, body_best_dt]
        best_delta = delta[node_idx, best_idx]
        best_dxy = dxy[node_idx, best_idx].unsqueeze(1)
        best_d3 = d3[node_idx, best_idx].unsqueeze(1)
        best_heat = heat_proxy[node_idx, best_idx].unsqueeze(1)
        exposure_integral = decayed_heat.sum(dim=1, keepdim=True)
        best_dt = rel_dt[node_idx, best_idx].unsqueeze(1)

        split_features = []
        if self.laser_feature_include_exposure_split:
            past_integral = decayed_heat.masked_fill(rel_dt < 0, 0.0).sum(
                dim=1, keepdim=True
            )
            future_integral = decayed_heat.masked_fill(rel_dt >= 0, 0.0).sum(
                dim=1, keepdim=True
            )
            best_abs_dt = best_dt.abs()
            split_features = [past_integral, future_integral, best_abs_dt]

        arrival_features = []
        if self.laser_feature_include_arrival_time:
            geom_score = (
                cross_xy.pow(2) / radius.pow(2)
                + along_outside.pow(2) / along_radius.pow(2)
                + z_delta.pow(2) / depth.pow(2)
            )
            arrival_idx = geom_score.argmin(dim=1)
            arrival_dt = rel_dt[node_idx, arrival_idx].unsqueeze(1)
            arrival_abs_dt = arrival_dt.abs()
            time_until_hit = (-arrival_dt).clamp_min(0.0)
            time_since_hit = arrival_dt.clamp_min(0.0)
            arrival_cross_xy = cross_xy[node_idx, arrival_idx].unsqueeze(1)
            arrival_along_outside = along_outside[node_idx, arrival_idx].abs().unsqueeze(1)
            arrival_abs_dz = z_delta[node_idx, arrival_idx].abs().unsqueeze(1)
            arrival_spatial_gate = heat_proxy[node_idx, arrival_idx].unsqueeze(1)
            arrival_gate = arrival_spatial_gate * torch.exp(
                -arrival_abs_dt / self.laser_feature_arrival_time_decay_s
            )
            arrival_features = [
                arrival_dt,
                arrival_abs_dt,
                time_until_hit,
                time_since_hit,
                arrival_cross_xy,
                arrival_along_outside,
                arrival_abs_dz,
                arrival_gate,
            ]

        return [
            best_delta,
            best_dxy,
            best_d3,
            best_heat,
            exposure_integral,
            best_dt,
            *body_features,
            *split_features,
            *arrival_features,
        ]

    def _analytic_path_arrival_features(self, *, coords: torch.Tensor,
                                        target_time: torch.Tensor,
                                        radius: torch.Tensor,
                                        along_radius: torch.Tensor,
                                        depth: torch.Tensor,
                                        device, dtype) -> list[torch.Tensor]:
        """Dense node-wise arrival features from the configured XML scan path."""
        params = getattr(self.graph, "_laser_path_params", None)
        if not params:
            zeros = torch.zeros((coords.shape[0], 1), device=device, dtype=dtype)
            return [zeros] * 8

        start = params["start_point"].to(device=device, dtype=dtype)
        scan_dir = params["scan_direction"].to(device=device, dtype=dtype)
        hatch_dir = params["hatch_direction"].to(device=device, dtype=dtype)
        scan_length = max(float(params["scan_length_mm"]), 1.0e-6)
        hatch_count = max(int(params["hatch_count"]), 1)
        hatch_spacing = max(float(params["hatch_spacing_mm"]), 1.0e-6)
        layer_count = max(int(params["layer_count"]), 1)
        layer_thickness = max(float(params["layer_thickness_mm"]), 1.0e-6)
        velocity = max(float(params["velocity_mm_s"]), 1.0e-6)
        path_period = max(float(params["path_period_s"]), 1.0e-12)
        layer_period = max(float(params["layer_period_s"]), 1.0e-12)
        path_time_scale = max(float(params["time_scale_to_s"]), 1.0e-12)
        path_time_offset = float(params["time_offset_s"])
        alternate_layers = bool(params.get("alternate_layer_scan_direction", False))
        reverse_parity = int(params.get("reverse_hatch_order_parity", -1))

        rel_to_start = coords - start.view(1, 3)
        layer_float = rel_to_start[:, 2] / layer_thickness
        layer_idx = layer_float.round().clamp(0, layer_count - 1).to(torch.long)
        layer_z = start[2] + layer_idx.to(dtype=dtype) * layer_thickness

        hatch_distance = (rel_to_start * hatch_dir.view(1, 3)).sum(dim=1)
        physical_track = (hatch_distance / hatch_spacing).round()
        physical_track = physical_track.clamp(0, hatch_count - 1).to(torch.long)

        target_time_t = target_time.to(device=device, dtype=dtype)
        layer_offset = torch.zeros_like(coords)
        layer_offset[:, 2] = layer_idx.to(dtype=dtype) * layer_thickness
        z_delta_base = (coords[:, 2] - layer_z).view(-1, 1)
        t0_s = self.graph.times[0].to(device=device, dtype=dtype) * path_time_scale

        candidate_features: list[list[torch.Tensor]] = []
        candidate_scores: list[torch.Tensor] = []
        for candidate_track in self._path_candidate_tracks(physical_track, hatch_count):
            track_in_layer = candidate_track.clone()
            if reverse_parity in (0, 1):
                reverse_layer = (layer_idx.remainder(2) == reverse_parity)
                reversed_track = (hatch_count - 1) - candidate_track
                track_in_layer = torch.where(reverse_layer, reversed_track, track_in_layer)

            line_start = (
                start.view(1, 3)
                + hatch_dir.view(1, 3)
                * (candidate_track.to(dtype=dtype).view(-1, 1) * hatch_spacing)
                + layer_offset
            )
            rel_line = coords - line_start
            scan_coord = (rel_line * scan_dir.view(1, 3)).sum(dim=1)
            scan_coord_clamped = scan_coord.clamp(0.0, scan_length)
            closest = line_start + scan_dir.view(1, 3) * scan_coord_clamped.view(-1, 1)
            delta = coords - closest
            cross_xy = torch.linalg.vector_norm(delta[:, :2], dim=1, keepdim=True)
            along_outside = (scan_coord - scan_coord_clamped).abs().view(-1, 1)

            forward = track_in_layer.remainder(2) == 0
            if alternate_layers:
                flip_layer = layer_idx.remainder(2) == 1
                forward = torch.logical_xor(forward, flip_layer)
            scan_elapsed = torch.where(
                forward,
                scan_coord_clamped / velocity,
                (scan_length - scan_coord_clamped) / velocity,
            )
            elapsed_s = (
                layer_idx.to(dtype=dtype) * layer_period
                + track_in_layer.to(dtype=dtype) * path_period
                + scan_elapsed
            )
            hit_raw_time = (t0_s + path_time_offset + elapsed_s) / path_time_scale
            arrival_dt = (target_time_t - hit_raw_time).view(-1, 1)
            arrival_dt = arrival_dt * self.laser_feature_time_scale_to_s
            arrival_abs_dt = arrival_dt.abs()
            time_until_hit = (-arrival_dt).clamp_min(0.0)
            time_since_hit = arrival_dt.clamp_min(0.0)
            spatial_gate = torch.exp(
                -2.0 * cross_xy.pow(2) / radius.pow(2)
                -2.0 * along_outside.pow(2) / along_radius.pow(2)
                -z_delta_base.pow(2) / depth.pow(2)
            )
            if self.laser_feature_path_arrival_gate_mode == "past":
                temporal_gate = torch.exp(
                    -time_since_hit / self.laser_feature_path_arrival_time_decay_s
                ) * (arrival_dt >= 0.0).to(dtype=dtype)
            elif self.laser_feature_path_arrival_gate_mode == "future":
                temporal_gate = torch.exp(
                    -time_until_hit / self.laser_feature_path_arrival_time_decay_s
                ) * (arrival_dt <= 0.0).to(dtype=dtype)
            else:
                temporal_gate = torch.exp(
                    -arrival_abs_dt / self.laser_feature_path_arrival_time_decay_s
                )
            arrival_gate = spatial_gate * temporal_gate
            candidate_features.append([
                arrival_dt,
                arrival_abs_dt,
                time_until_hit,
                time_since_hit,
                cross_xy,
                along_outside,
                z_delta_base.abs(),
                arrival_gate,
            ])
            candidate_scores.append(arrival_gate)

        if len(candidate_features) == 1:
            return candidate_features[0]
        best_idx = torch.stack(candidate_scores, dim=0).argmax(dim=0)
        return [
            self._pick_path_candidate([feat[i] for feat in candidate_features], best_idx)
            for i in range(len(candidate_features[0]))
        ]

    def _analytic_path_phase_features(self, *, coords: torch.Tensor,
                                      target_time: torch.Tensor,
                                      radius: torch.Tensor,
                                      depth: torch.Tensor,
                                      device, dtype) -> list[torch.Tensor]:
        """Structured additive_z_scan phase features for every node.

        These expose the XML scan program more directly than a generic nearest
        segment gate: layer/track identity, progress along the serpentine line,
        signed cross-track offset, and separate before/after arrival gates.
        """
        params = getattr(self.graph, "_laser_path_params", None)
        if not params:
            zeros = torch.zeros((coords.shape[0], 1), device=device, dtype=dtype)
            return [zeros] * 10

        start = params["start_point"].to(device=device, dtype=dtype)
        scan_dir = params["scan_direction"].to(device=device, dtype=dtype)
        hatch_dir = params["hatch_direction"].to(device=device, dtype=dtype)
        scan_length = max(float(params["scan_length_mm"]), 1.0e-6)
        hatch_count = max(int(params["hatch_count"]), 1)
        hatch_spacing = max(float(params["hatch_spacing_mm"]), 1.0e-6)
        layer_count = max(int(params["layer_count"]), 1)
        layer_thickness = max(float(params["layer_thickness_mm"]), 1.0e-6)
        velocity = max(float(params["velocity_mm_s"]), 1.0e-6)
        path_period = max(float(params["path_period_s"]), 1.0e-12)
        layer_period = max(float(params["layer_period_s"]), 1.0e-12)
        path_time_scale = max(float(params["time_scale_to_s"]), 1.0e-12)
        path_time_offset = float(params["time_offset_s"])
        alternate_layers = bool(params.get("alternate_layer_scan_direction", False))
        reverse_parity = int(params.get("reverse_hatch_order_parity", -1))

        rel_to_start = coords - start.view(1, 3)
        layer_float = rel_to_start[:, 2] / layer_thickness
        layer_idx = layer_float.round().clamp(0, layer_count - 1).to(torch.long)
        layer_z = start[2] + layer_idx.to(dtype=dtype) * layer_thickness

        hatch_distance = (rel_to_start * hatch_dir.view(1, 3)).sum(dim=1)
        physical_track_float = (hatch_distance / hatch_spacing).round()
        physical_track = physical_track_float.clamp(0, hatch_count - 1).to(torch.long)

        layer_offset = torch.zeros_like(coords)
        layer_offset[:, 2] = layer_idx.to(dtype=dtype) * layer_thickness
        layer_norm = layer_idx.to(dtype=dtype).view(-1, 1) / max(layer_count - 1, 1)
        z_delta = (coords[:, 2] - layer_z).view(-1, 1)
        t0_s = self.graph.times[0].to(device=device, dtype=dtype) * path_time_scale
        target_time_t = target_time.to(device=device, dtype=dtype)

        candidate_features: list[list[torch.Tensor]] = []
        candidate_scores: list[torch.Tensor] = []
        for candidate_track in self._path_candidate_tracks(physical_track, hatch_count):
            track_in_layer = candidate_track.clone()
            if reverse_parity in (0, 1):
                reverse_layer = layer_idx.remainder(2) == reverse_parity
                reversed_track = (hatch_count - 1) - candidate_track
                track_in_layer = torch.where(reverse_layer, reversed_track, track_in_layer)

            line_start = (
                start.view(1, 3)
                + hatch_dir.view(1, 3)
                * (candidate_track.to(dtype=dtype).view(-1, 1) * hatch_spacing)
                + layer_offset
            )
            rel_line = coords - line_start
            scan_coord = (rel_line * scan_dir.view(1, 3)).sum(dim=1)
            scan_coord_clamped = scan_coord.clamp(0.0, scan_length)
            closest = line_start + scan_dir.view(1, 3) * scan_coord_clamped.view(-1, 1)
            delta = coords - closest
            signed_cross = (rel_line * hatch_dir.view(1, 3)).sum(dim=1, keepdim=True)
            cross_norm = signed_cross / hatch_spacing
            along_norm = (scan_coord / scan_length).view(-1, 1)
            progress_norm = (scan_coord_clamped / scan_length).view(-1, 1)
            track_norm = track_in_layer.to(dtype=dtype).view(-1, 1) / max(hatch_count - 1, 1)
            physical_track_norm = candidate_track.to(dtype=dtype).view(-1, 1) / max(hatch_count - 1, 1)

            forward = track_in_layer.remainder(2) == 0
            if alternate_layers:
                flip_layer = layer_idx.remainder(2) == 1
                forward = torch.logical_xor(forward, flip_layer)
            scan_elapsed = torch.where(
                forward,
                scan_coord_clamped / velocity,
                (scan_length - scan_coord_clamped) / velocity,
            )
            elapsed_s = (
                layer_idx.to(dtype=dtype) * layer_period
                + track_in_layer.to(dtype=dtype) * path_period
                + scan_elapsed
            )
            hit_raw_time = (t0_s + path_time_offset + elapsed_s) / path_time_scale
            arrival_dt = (target_time_t - hit_raw_time).view(-1, 1)
            arrival_dt = arrival_dt * self.laser_feature_time_scale_to_s
            time_until_hit = (-arrival_dt).clamp_min(0.0)
            time_since_hit = arrival_dt.clamp_min(0.0)

            line_gate = torch.exp(
                -2.0 * cross_norm.pow(2)
                -z_delta.pow(2) / depth.pow(2)
            )
            active_scan_gate = torch.exp(
                -2.0 * torch.linalg.vector_norm(delta[:, :2], dim=1, keepdim=True).pow(2)
                / radius.pow(2)
                -z_delta.pow(2) / depth.pow(2)
            )
            pre_arrival_gate = line_gate * torch.exp(
                -time_until_hit / self.laser_feature_path_arrival_time_decay_s
            ) * (arrival_dt <= 0.0).to(dtype=dtype)
            post_arrival_gate = line_gate * torch.exp(
                -time_since_hit / self.laser_feature_path_arrival_time_decay_s
            ) * (arrival_dt >= 0.0).to(dtype=dtype)
            forward_sign = torch.where(
                forward.view(-1, 1),
                torch.ones_like(progress_norm),
                -torch.ones_like(progress_norm),
            )
            candidate_features.append([
                layer_norm,
                track_norm,
                physical_track_norm,
                progress_norm,
                along_norm,
                cross_norm,
                forward_sign,
                active_scan_gate.clamp(0.0, 1.0),
                pre_arrival_gate.clamp(0.0, 1.0),
                post_arrival_gate.clamp(0.0, 1.0),
            ])
            candidate_scores.append(torch.maximum(
                pre_arrival_gate.clamp(0.0, 1.0),
                post_arrival_gate.clamp(0.0, 1.0),
            ))

        if len(candidate_features) == 1:
            return candidate_features[0]
        best_idx = torch.stack(candidate_scores, dim=0).argmax(dim=0)
        return [
            self._pick_path_candidate([feat[i] for feat in candidate_features], best_idx)
            for i in range(len(candidate_features[0]))
        ]

    def _analytic_path_coordinate_features(self, *, coords: torch.Tensor,
                                           target_time: torch.Tensor,
                                           depth: torch.Tensor,
                                           device, dtype) -> list[torch.Tensor]:
        """Explicit additive_z_scan coordinates and timing features.

        These features keep the XML program visible to the model without using
        the target temperature: layer/track identity, scan-line progress,
        endpoint proximity, and signed time relative to laser arrival.
        """
        params = getattr(self.graph, "_laser_path_params", None)
        if not params:
            zeros = torch.zeros((coords.shape[0], 1), device=device, dtype=dtype)
            feature_dim = 16 if self.laser_feature_include_path_timing else 12
            return [zeros] * feature_dim

        start = params["start_point"].to(device=device, dtype=dtype)
        scan_dir = params["scan_direction"].to(device=device, dtype=dtype)
        hatch_dir = params["hatch_direction"].to(device=device, dtype=dtype)
        scan_length = max(float(params["scan_length_mm"]), 1.0e-6)
        hatch_count = max(int(params["hatch_count"]), 1)
        hatch_spacing = max(float(params["hatch_spacing_mm"]), 1.0e-6)
        layer_count = max(int(params["layer_count"]), 1)
        layer_thickness = max(float(params["layer_thickness_mm"]), 1.0e-6)
        velocity = max(float(params["velocity_mm_s"]), 1.0e-6)
        path_period = max(float(params["path_period_s"]), 1.0e-12)
        layer_period = max(float(params["layer_period_s"]), 1.0e-12)
        path_time_scale = max(float(params["time_scale_to_s"]), 1.0e-12)
        path_time_offset = float(params["time_offset_s"])
        alternate_layers = bool(params.get("alternate_layer_scan_direction", False))
        reverse_parity = int(params.get("reverse_hatch_order_parity", -1))

        rel_to_start = coords - start.view(1, 3)
        layer_float = rel_to_start[:, 2] / layer_thickness
        layer_idx = layer_float.round().clamp(0, layer_count - 1).to(torch.long)
        layer_z = start[2] + layer_idx.to(dtype=dtype) * layer_thickness

        hatch_distance = (rel_to_start * hatch_dir.view(1, 3)).sum(dim=1)
        physical_track = (hatch_distance / hatch_spacing).round()
        physical_track = physical_track.clamp(0, hatch_count - 1).to(torch.long)

        layer_offset = torch.zeros_like(coords)
        layer_offset[:, 2] = layer_idx.to(dtype=dtype) * layer_thickness
        z_delta = (coords[:, 2] - layer_z).view(-1, 1)
        t0_s = self.graph.times[0].to(device=device, dtype=dtype) * path_time_scale
        target_time_t = target_time.to(device=device, dtype=dtype)
        total_elapsed_s = max(layer_count * layer_period, 1.0e-12)
        time_decay = self.laser_feature_path_arrival_time_decay_s
        endpoint_radius = self.laser_feature_endpoint_radius_mm

        candidate_features: list[list[torch.Tensor]] = []
        candidate_scores: list[torch.Tensor] = []
        for candidate_track in self._path_candidate_tracks(physical_track, hatch_count):
            track_in_layer = candidate_track.clone()
            if reverse_parity in (0, 1):
                reverse_layer = layer_idx.remainder(2) == reverse_parity
                reversed_track = (hatch_count - 1) - candidate_track
                track_in_layer = torch.where(reverse_layer, reversed_track, track_in_layer)

            line_start = (
                start.view(1, 3)
                + hatch_dir.view(1, 3)
                * (candidate_track.to(dtype=dtype).view(-1, 1) * hatch_spacing)
                + layer_offset
            )
            geom_end = line_start + scan_dir.view(1, 3) * scan_length
            rel_line = coords - line_start
            scan_coord = (rel_line * scan_dir.view(1, 3)).sum(dim=1)
            scan_coord_clamped = scan_coord.clamp(0.0, scan_length)

            forward = track_in_layer.remainder(2) == 0
            if alternate_layers:
                flip_layer = layer_idx.remainder(2) == 1
                forward = torch.logical_xor(forward, flip_layer)
            travel_start = torch.where(forward.view(-1, 1), line_start, geom_end)
            travel_end = torch.where(forward.view(-1, 1), geom_end, line_start)

            progress = (scan_coord_clamped / scan_length).view(-1, 1)
            travel_progress = torch.where(forward.view(-1, 1), progress, 1.0 - progress)
            signed_cross = (rel_line * hatch_dir.view(1, 3)).sum(dim=1, keepdim=True)
            cross_signed_norm = (signed_cross / (3.0 * hatch_spacing)).clamp(-1.0, 1.0)
            centered_along_norm = ((scan_coord.view(-1, 1) / scan_length) * 2.0 - 1.0).clamp(-1.5, 1.5)
            centered_along_norm = centered_along_norm / 1.5

            scan_elapsed = torch.where(
                forward,
                scan_coord_clamped / velocity,
                (scan_length - scan_coord_clamped) / velocity,
            )
            elapsed_s = (
                layer_idx.to(dtype=dtype) * layer_period
                + track_in_layer.to(dtype=dtype) * path_period
                + scan_elapsed
            )
            hit_raw_time = (t0_s + path_time_offset + elapsed_s) / path_time_scale
            arrival_dt = (target_time_t - hit_raw_time).view(-1, 1)
            arrival_dt = arrival_dt * self.laser_feature_time_scale_to_s
            arrival_dt_norm = (arrival_dt / (5.0 * time_decay)).clamp(-1.0, 1.0)
            scanned_gate = (arrival_dt >= 0.0).to(dtype=dtype)
            time_since_s = arrival_dt.clamp_min(0.0)
            time_until_s = (-arrival_dt).clamp_min(0.0)
            time_norm_scale = math.log1p(10.0)
            time_since_log = (
                torch.log1p(time_since_s) / time_norm_scale
            ).clamp(0.0, 1.0)
            time_until_log = (
                torch.log1p(time_until_s) / time_norm_scale
            ).clamp(0.0, 1.0)
            cooling_tail = torch.exp(-time_since_s / 10.0) * scanned_gate

            dist_to_travel_start = torch.linalg.vector_norm(
                coords - travel_start, dim=1, keepdim=True
            )
            dist_to_travel_end = torch.linalg.vector_norm(
                coords - travel_end, dim=1, keepdim=True
            )
            endpoint_side = torch.where(
                dist_to_travel_start <= dist_to_travel_end,
                -torch.ones_like(dist_to_travel_start),
                torch.ones_like(dist_to_travel_end),
            )
            start_gate = torch.exp(
                -dist_to_travel_start.pow(2) / (endpoint_radius * endpoint_radius)
                -z_delta.pow(2) / depth.pow(2)
            )
            end_gate = torch.exp(
                -dist_to_travel_end.pow(2) / (endpoint_radius * endpoint_radius)
                -z_delta.pow(2) / depth.pow(2)
            )
            line_gate = torch.exp(
                -2.0 * (signed_cross / hatch_spacing).pow(2)
                -z_delta.pow(2) / depth.pow(2)
            )
            temporal_score = torch.exp(-arrival_dt.abs() / time_decay)

            layer_signed = (
                2.0 * layer_idx.to(dtype=dtype).view(-1, 1) / max(layer_count - 1, 1)
                - 1.0
            )
            physical_track_signed = (
                2.0 * candidate_track.to(dtype=dtype).view(-1, 1) / max(hatch_count - 1, 1)
                - 1.0
            )
            program_track_signed = (
                2.0 * track_in_layer.to(dtype=dtype).view(-1, 1) / max(hatch_count - 1, 1)
                - 1.0
            )
            path_elapsed_norm = (elapsed_s / total_elapsed_s).view(-1, 1).clamp(0.0, 1.0)

            features = [
                layer_signed,
                physical_track_signed,
                program_track_signed,
                2.0 * progress - 1.0,
                2.0 * travel_progress - 1.0,
                cross_signed_norm,
                centered_along_norm,
                endpoint_side,
                start_gate.clamp(0.0, 1.0),
                end_gate.clamp(0.0, 1.0),
                arrival_dt_norm,
                path_elapsed_norm,
            ]
            if self.laser_feature_include_path_timing:
                features.extend([
                    scanned_gate,
                    time_since_log,
                    time_until_log,
                    cooling_tail.clamp(0.0, 1.0),
                ])
            candidate_features.append(features)
            candidate_scores.append((line_gate * temporal_score).clamp(0.0, 1.0))

        if len(candidate_features) == 1:
            return candidate_features[0]
        best_idx = torch.stack(candidate_scores, dim=0).argmax(dim=0)
        return [
            self._pick_path_candidate([feat[i] for feat in candidate_features], best_idx)
            for i in range(len(candidate_features[0]))
        ]

    def _analytic_path_body_support_features(self, *, coords: torch.Tensor,
                                             device, dtype) -> list[torch.Tensor]:
        """Spatial ellipsoid/corridor support along the XML scan line.

        Unlike the target-center body source, this feature marks nodes close to
        the active hatch corridor even before the moving heat-source center has
        reached them.  It is meant to be combined with path pre-arrival timing.
        """
        params = getattr(self.graph, "_laser_path_params", None)
        if not params:
            zeros = torch.zeros((coords.shape[0], 1), device=device, dtype=dtype)
            return [zeros] * 5

        start = params["start_point"].to(device=device, dtype=dtype)
        scan_dir = params["scan_direction"].to(device=device, dtype=dtype)
        hatch_dir = params["hatch_direction"].to(device=device, dtype=dtype)
        scan_length = max(float(params["scan_length_mm"]), 1.0e-6)
        hatch_count = max(int(params["hatch_count"]), 1)
        hatch_spacing = max(float(params["hatch_spacing_mm"]), 1.0e-6)
        layer_count = max(int(params["layer_count"]), 1)
        layer_thickness = max(float(params["layer_thickness_mm"]), 1.0e-6)

        rel_to_start = coords - start.view(1, 3)
        layer_float = rel_to_start[:, 2] / layer_thickness
        layer_idx = layer_float.round().clamp(0, layer_count - 1).to(torch.long)
        layer_z = start[2] + layer_idx.to(dtype=dtype) * layer_thickness

        hatch_distance = (rel_to_start * hatch_dir.view(1, 3)).sum(dim=1)
        physical_track = (hatch_distance / hatch_spacing).round()
        physical_track = physical_track.clamp(0, hatch_count - 1).to(torch.long)

        layer_offset = torch.zeros_like(coords)
        layer_offset[:, 2] = layer_idx.to(dtype=dtype) * layer_thickness
        line_start = (
            start.view(1, 3)
            + hatch_dir.view(1, 3)
            * (physical_track.to(dtype=dtype).view(-1, 1) * hatch_spacing)
            + layer_offset
        )

        rel_line = coords - line_start
        scan_coord = (rel_line * scan_dir.view(1, 3)).sum(dim=1, keepdim=True)
        scan_coord_clamped = scan_coord.clamp(0.0, scan_length)
        signed_along_outside = scan_coord - scan_coord_clamped
        signed_cross = (rel_line * hatch_dir.view(1, 3)).sum(dim=1, keepdim=True)
        z_delta = (coords[:, 2] - layer_z).view(-1, 1)

        return self._ellipsoid_body_source_terms(
            along=signed_along_outside,
            cross=signed_cross,
            z_delta=z_delta,
            device=device,
            dtype=dtype,
        )

    def _analytic_path_endpoint_features(self, *, coords: torch.Tensor,
                                         target_time: torch.Tensor,
                                         depth: torch.Tensor,
                                         device, dtype) -> list[torch.Tensor]:
        """Endpoint/turnaround support features from additive_z_scan.

        The line-interior gate is intentionally narrow.  This feature family
        keeps support near hatch endpoints where the laser turns around and the
        melt pool can spill slightly beyond the nominal scan segment.
        """
        params = getattr(self.graph, "_laser_path_params", None)
        if not params:
            zeros = torch.zeros((coords.shape[0], 1), device=device, dtype=dtype)
            return [zeros] * 6

        start = params["start_point"].to(device=device, dtype=dtype)
        scan_dir = params["scan_direction"].to(device=device, dtype=dtype)
        hatch_dir = params["hatch_direction"].to(device=device, dtype=dtype)
        scan_length = max(float(params["scan_length_mm"]), 1.0e-6)
        hatch_count = max(int(params["hatch_count"]), 1)
        hatch_spacing = max(float(params["hatch_spacing_mm"]), 1.0e-6)
        layer_count = max(int(params["layer_count"]), 1)
        layer_thickness = max(float(params["layer_thickness_mm"]), 1.0e-6)
        velocity = max(float(params["velocity_mm_s"]), 1.0e-6)
        path_period = max(float(params["path_period_s"]), 1.0e-12)
        layer_period = max(float(params["layer_period_s"]), 1.0e-12)
        path_time_scale = max(float(params["time_scale_to_s"]), 1.0e-12)
        path_time_offset = float(params["time_offset_s"])
        alternate_layers = bool(params.get("alternate_layer_scan_direction", False))
        reverse_parity = int(params.get("reverse_hatch_order_parity", -1))

        rel_to_start = coords - start.view(1, 3)
        layer_float = rel_to_start[:, 2] / layer_thickness
        layer_idx = layer_float.round().clamp(0, layer_count - 1).to(torch.long)
        layer_z = start[2] + layer_idx.to(dtype=dtype) * layer_thickness

        hatch_distance = (rel_to_start * hatch_dir.view(1, 3)).sum(dim=1)
        physical_track = (hatch_distance / hatch_spacing).round()
        physical_track = physical_track.clamp(0, hatch_count - 1).to(torch.long)

        track_in_layer = physical_track.clone()
        if reverse_parity in (0, 1):
            reverse_layer = layer_idx.remainder(2) == reverse_parity
            reversed_track = (hatch_count - 1) - physical_track
            track_in_layer = torch.where(reverse_layer, reversed_track, track_in_layer)

        layer_offset = torch.zeros_like(coords)
        layer_offset[:, 2] = layer_idx.to(dtype=dtype) * layer_thickness
        geom_start = (
            start.view(1, 3)
            + hatch_dir.view(1, 3)
            * (physical_track.to(dtype=dtype).view(-1, 1) * hatch_spacing)
            + layer_offset
        )
        geom_end = geom_start + scan_dir.view(1, 3) * scan_length

        forward = track_in_layer.remainder(2) == 0
        if alternate_layers:
            flip_layer = layer_idx.remainder(2) == 1
            forward = torch.logical_xor(forward, flip_layer)

        travel_start = torch.where(forward.view(-1, 1), geom_start, geom_end)
        travel_end = torch.where(forward.view(-1, 1), geom_end, geom_start)

        base_elapsed_s = (
            layer_idx.to(dtype=dtype) * layer_period
            + track_in_layer.to(dtype=dtype) * path_period
        )
        end_elapsed_s = base_elapsed_s + scan_length / velocity

        t0_s = self.graph.times[0].to(device=device, dtype=dtype) * path_time_scale
        target_time_t = target_time.to(device=device, dtype=dtype)
        start_raw_time = (t0_s + path_time_offset + base_elapsed_s) / path_time_scale
        end_raw_time = (t0_s + path_time_offset + end_elapsed_s) / path_time_scale
        start_dt = (target_time_t - start_raw_time).view(-1, 1)
        end_dt = (target_time_t - end_raw_time).view(-1, 1)
        start_dt = start_dt * self.laser_feature_time_scale_to_s
        end_dt = end_dt * self.laser_feature_time_scale_to_s

        z_delta = (coords[:, 2] - layer_z).view(-1, 1)
        endpoint_radius = torch.as_tensor(
            self.laser_feature_endpoint_radius_mm, device=device, dtype=dtype
        )
        start_dxy = torch.linalg.vector_norm(
            (coords - travel_start)[:, :2], dim=1, keepdim=True
        )
        end_dxy = torch.linalg.vector_norm(
            (coords - travel_end)[:, :2], dim=1, keepdim=True
        )
        start_spatial = torch.exp(
            -2.0 * start_dxy.pow(2) / endpoint_radius.pow(2)
            -z_delta.pow(2) / depth.pow(2)
        )
        end_spatial = torch.exp(
            -2.0 * end_dxy.pow(2) / endpoint_radius.pow(2)
            -z_delta.pow(2) / depth.pow(2)
        )
        start_gate = start_spatial * torch.exp(
            -start_dt.abs() / self.laser_feature_endpoint_time_decay_s
        )
        end_gate = end_spatial * torch.exp(
            -end_dt.abs() / self.laser_feature_endpoint_time_decay_s
        )

        endpoint_gate = torch.maximum(start_gate, end_gate).clamp(0.0, 1.0)
        start_is_best = start_gate >= end_gate
        endpoint_dxy = torch.minimum(start_dxy, end_dxy)
        endpoint_spatial_gate = torch.maximum(start_spatial, end_spatial).clamp(0.0, 1.0)
        endpoint_abs_dt = torch.where(start_is_best, start_dt.abs(), end_dt.abs())

        return [
            endpoint_dxy,
            endpoint_spatial_gate,
            endpoint_abs_dt,
            endpoint_gate,
            start_gate.clamp(0.0, 1.0),
            end_gate.clamp(0.0, 1.0),
        ]

    def _analytic_path_wake_features(self, *, coords: torch.Tensor,
                                     target_time: torch.Tensor,
                                     depth: torch.Tensor,
                                     device, dtype) -> list[torch.Tensor]:
        """Trailing/leading support around the moving XML laser center."""
        params = getattr(self.graph, "_laser_path_params", None)
        if not params:
            zeros = torch.zeros((coords.shape[0], 1), device=device, dtype=dtype)
            return [zeros] * 6

        start = params["start_point"].to(device=device, dtype=dtype)
        scan_dir = params["scan_direction"].to(device=device, dtype=dtype)
        hatch_dir = params["hatch_direction"].to(device=device, dtype=dtype)
        scan_length = max(float(params["scan_length_mm"]), 1.0e-6)
        hatch_count = max(int(params["hatch_count"]), 1)
        hatch_spacing = max(float(params["hatch_spacing_mm"]), 1.0e-6)
        layer_count = max(int(params["layer_count"]), 1)
        layer_thickness = max(float(params["layer_thickness_mm"]), 1.0e-6)
        velocity = max(float(params["velocity_mm_s"]), 1.0e-6)
        path_period = max(float(params["path_period_s"]), 1.0e-12)
        layer_period = max(float(params["layer_period_s"]), 1.0e-12)
        path_time_scale = max(float(params["time_scale_to_s"]), 1.0e-12)
        path_time_offset = float(params["time_offset_s"])
        alternate_layers = bool(params.get("alternate_layer_scan_direction", False))
        reverse_parity = int(params.get("reverse_hatch_order_parity", -1))

        rel_to_start = coords - start.view(1, 3)
        layer_float = rel_to_start[:, 2] / layer_thickness
        layer_idx = layer_float.round().clamp(0, layer_count - 1).to(torch.long)
        layer_z = start[2] + layer_idx.to(dtype=dtype) * layer_thickness

        hatch_distance = (rel_to_start * hatch_dir.view(1, 3)).sum(dim=1)
        physical_track = (hatch_distance / hatch_spacing).round()
        physical_track = physical_track.clamp(0, hatch_count - 1).to(torch.long)

        layer_offset = torch.zeros_like(coords)
        layer_offset[:, 2] = layer_idx.to(dtype=dtype) * layer_thickness
        z_delta = (coords[:, 2] - layer_z).view(-1, 1)
        t0_s = self.graph.times[0].to(device=device, dtype=dtype) * path_time_scale
        target_time_t = target_time.to(device=device, dtype=dtype)

        cross_radius = torch.as_tensor(
            self.laser_feature_wake_cross_radius_mm, device=device, dtype=dtype
        )
        tail_decay = torch.as_tensor(
            self.laser_feature_wake_tail_decay_mm, device=device, dtype=dtype
        )
        lead_decay = torch.as_tensor(
            self.laser_feature_wake_lead_decay_mm, device=device, dtype=dtype
        )
        time_decay = torch.as_tensor(
            self.laser_feature_wake_time_decay_s, device=device, dtype=dtype
        )

        candidate_features: list[list[torch.Tensor]] = []
        candidate_scores: list[torch.Tensor] = []
        for candidate_track in self._path_candidate_tracks(physical_track, hatch_count):
            track_in_layer = candidate_track.clone()
            if reverse_parity in (0, 1):
                reverse_layer = layer_idx.remainder(2) == reverse_parity
                reversed_track = (hatch_count - 1) - candidate_track
                track_in_layer = torch.where(reverse_layer, reversed_track, track_in_layer)

            line_start = (
                start.view(1, 3)
                + hatch_dir.view(1, 3)
                * (candidate_track.to(dtype=dtype).view(-1, 1) * hatch_spacing)
                + layer_offset
            )
            rel_line = coords - line_start
            scan_coord = (rel_line * scan_dir.view(1, 3)).sum(dim=1)
            scan_coord_clamped = scan_coord.clamp(0.0, scan_length)

            forward = track_in_layer.remainder(2) == 0
            if alternate_layers:
                flip_layer = layer_idx.remainder(2) == 1
                forward = torch.logical_xor(forward, flip_layer)

            scan_elapsed = torch.where(
                forward,
                scan_coord_clamped / velocity,
                (scan_length - scan_coord_clamped) / velocity,
            )
            elapsed_s = (
                layer_idx.to(dtype=dtype) * layer_period
                + track_in_layer.to(dtype=dtype) * path_period
                + scan_elapsed
            )
            hit_raw_time = (t0_s + path_time_offset + elapsed_s) / path_time_scale
            arrival_dt = (target_time_t - hit_raw_time).view(-1, 1)
            arrival_dt = arrival_dt * self.laser_feature_time_scale_to_s

            lag_mm = arrival_dt * velocity
            trail_mm = lag_mm.clamp_min(0.0)
            lead_mm = (-lag_mm).clamp_min(0.0)
            signed_lag_norm = torch.where(
                lag_mm >= 0.0,
                lag_mm / tail_decay,
                lag_mm / lead_decay,
            ).clamp(-1.5, 1.5) / 1.5

            signed_cross = (rel_line * hatch_dir.view(1, 3)).sum(dim=1, keepdim=True)
            cross_gate = torch.exp(-(signed_cross / cross_radius).pow(2))
            depth_gate = torch.exp(-(z_delta / depth).pow(2))
            inside_line = torch.logical_and(
                scan_coord.view(-1, 1) >= 0.0,
                scan_coord.view(-1, 1) <= scan_length,
            ).to(dtype=dtype)

            pre_time_gate = torch.exp(-lead_mm / lead_decay) * (
                arrival_dt <= 0.0
            ).to(dtype=dtype)
            post_time_gate = torch.exp(-trail_mm / tail_decay) * (
                arrival_dt >= 0.0
            ).to(dtype=dtype)
            temporal_gate = torch.exp(-arrival_dt.abs() / time_decay)
            pre_wake_gate = pre_time_gate * cross_gate * depth_gate * inside_line
            post_wake_gate = post_time_gate * cross_gate * depth_gate * inside_line
            wake_gate = torch.maximum(pre_wake_gate, post_wake_gate)

            candidate_features.append([
                wake_gate.clamp(0.0, 1.0),
                pre_wake_gate.clamp(0.0, 1.0),
                post_wake_gate.clamp(0.0, 1.0),
                signed_lag_norm,
                cross_gate.clamp(0.0, 1.0),
                temporal_gate.clamp(0.0, 1.0),
            ])
            candidate_scores.append(
                (wake_gate + 0.05 * cross_gate * inside_line).clamp(0.0, 1.0)
            )

        if len(candidate_features) == 1:
            return candidate_features[0]
        best_idx = torch.stack(candidate_scores, dim=0).argmax(dim=0)
        return [
            self._pick_path_candidate([feat[i] for feat in candidate_features], best_idx)
            for i in range(len(candidate_features[0]))
        ]

    def _neighbor_temperature_features(self, data) -> list[torch.Tensor]:
        """Return first-order neighbour thermal statistics for an input frame.

        These are computed only from the current input frame, so they expose
        local diffusion context without leaking the target temperature label.
        """
        temp = data.y.squeeze(-1) if data.y.dim() > 1 else data.y
        temp = temp.to(device=data.x.device, dtype=data.x.dtype)
        edge_index = data.edge_index.to(device=temp.device)
        src, dst = edge_index[0], edge_index[1]
        num_nodes = temp.shape[0]

        neigh_sum = torch.zeros(num_nodes, device=temp.device, dtype=temp.dtype)
        neigh_count = torch.zeros(num_nodes, device=temp.device, dtype=temp.dtype)
        neigh_sum.scatter_add_(0, dst, temp[src])
        neigh_count.scatter_add_(0, dst, torch.ones_like(temp[src]))
        neigh_mean = neigh_sum / neigh_count.clamp_min(1.0)

        fill_value = float(temp.min().detach().item()) if temp.numel() else 0.0
        neigh_max = torch.full((num_nodes,), fill_value, device=temp.device, dtype=temp.dtype)
        neigh_max.scatter_reduce_(0, dst, temp[src], reduce="amax", include_self=False)
        isolated = neigh_count <= 0
        if isolated.any():
            neigh_mean = torch.where(isolated, temp, neigh_mean)
            neigh_max = torch.where(isolated, temp, neigh_max)

        temp_scale = torch.as_tensor(1000.0, device=temp.device, dtype=temp.dtype)
        return [
            (neigh_max / temp_scale).unsqueeze(1),
            (neigh_mean / temp_scale).unsqueeze(1),
            ((neigh_max - temp) / temp_scale).unsqueeze(1),
        ]

    def _neighbor_hot_stats_features(self, data) -> list[torch.Tensor]:
        """Return neighbour melt-pool support features from the input frame."""
        return self._neighbor_threshold_stats_features(
            data,
            threshold=self.laser_feature_neighbor_hot_threshold,
        )

    def _neighbor_warm_stats_features(self, data) -> list[torch.Tensor]:
        """Return neighbour preheating support features from the input frame."""
        return self._neighbor_threshold_stats_features(
            data,
            threshold=self.laser_feature_neighbor_warm_threshold,
        )

    def _neighbor_threshold_stats_features(
            self,
            data,
            threshold: float,
    ) -> list[torch.Tensor]:
        temp = data.y.squeeze(-1) if data.y.dim() > 1 else data.y
        temp = temp.to(device=data.x.device, dtype=data.x.dtype)
        edge_index = data.edge_index.to(device=temp.device)
        src, dst = edge_index[0], edge_index[1]
        num_nodes = temp.shape[0]

        neigh_count = torch.zeros(num_nodes, device=temp.device, dtype=temp.dtype)
        hot_count = torch.zeros(num_nodes, device=temp.device, dtype=temp.dtype)
        threshold = torch.as_tensor(threshold, device=temp.device, dtype=temp.dtype)
        temp_scale = torch.as_tensor(1000.0, device=temp.device, dtype=temp.dtype)

        neigh_count.scatter_add_(0, dst, torch.ones_like(temp[src]))
        src_hot = (temp[src] >= threshold).to(dtype=temp.dtype)
        hot_count.scatter_add_(0, dst, src_hot)

        hot_fraction = hot_count / neigh_count.clamp_min(1.0)
        hot_excess = torch.zeros(num_nodes, device=temp.device, dtype=temp.dtype)
        hot_excess.scatter_reduce_(
            0,
            dst,
            (temp[src] - threshold).clamp_min(0.0),
            reduce="amax",
            include_self=False,
        )
        hot_excess = hot_excess / temp_scale

        return [
            hot_fraction.unsqueeze(1),
            hot_excess.unsqueeze(1),
        ]

    def target_max_temperature(self, idx: int) -> float:
        """Return the maximum temperature in the target frame for *idx*."""
        start_t = self.valid_starts[idx]
        target_t = start_t + self.window_size + self.predict_steps - 1
        if 0 <= target_t < self.graph.num_steps:
            return float(self.graph.temperatures[target_t].max().item())
        return 0.0


class StratifiedWindowSampler(Sampler):
    """Oversamples high-temperature windows while keeping validation untouched.

    Each epoch produces a fixed-size sequence of dataset indices whose
    category proportions match the configured ratios.  Categories are:

    * normal   — target max T <  hot_threshold
    * hot      — target max T ∈ [hot_threshold, solidus)
    * melting  — target max T ≥ solidus

    The sampler only affects training; val / test use plain shuffling.
    """

    def __init__(self, dataset: DEDTemporalDataset, *,
                 solidus: float = 1604.85,
                 hot_threshold: float = 500.0,
                 normal_ratio: float = 0.5,
                 hot_ratio: float = 0.3,
                 melting_ratio: float = 0.2,
                 seed: int = 42):
        if not (0 < normal_ratio + hot_ratio + melting_ratio <= 1.0 + 1e-9):
            raise ValueError("Stratified ratios must sum to 1.0.")

        self.dataset = dataset
        if isinstance(dataset, Subset):
            self.base_dataset = dataset.dataset
            self.dataset_indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            self.dataset_indices = list(range(len(dataset)))
        if not hasattr(self.base_dataset, "target_max_temperature"):
            raise TypeError(
                "StratifiedWindowSampler requires a DEDTemporalDataset or a "
                "Subset of DEDTemporalDataset."
            )
        self.solidus = solidus
        self.hot_threshold = hot_threshold
        self.normal_ratio = normal_ratio
        self.hot_ratio = hot_ratio
        self.melting_ratio = melting_ratio
        self.rng = random.Random(seed)

        # --- classify every dataset index once ---
        buckets: dict[str, list[int]] = {"normal": [], "hot": [], "melting": []}
        for local_i, source_i in enumerate(self.dataset_indices):
            mt = self.base_dataset.target_max_temperature(int(source_i))
            if mt >= solidus:
                buckets["melting"].append(local_i)
            elif mt >= hot_threshold:
                buckets["hot"].append(local_i)
            else:
                buckets["normal"].append(local_i)

        self.buckets = buckets
        counts = {k: len(v) for k, v in buckets.items()}
        total = sum(counts.values())
        print(
            f"StratifiedWindowSampler: "
            f"normal={counts['normal']} ({100*counts['normal']/total:.1f}%), "
            f"hot={counts['hot']} ({100*counts['hot']/total:.1f}%), "
            f"melting={counts['melting']} ({100*counts['melting']/total:.1f}%)"
        )

        # If a category is empty, redistribute its ratio proportionally.
        active_ratios = {}
        active_total = 0.0
        for cat, ratio in [("normal", normal_ratio), ("hot", hot_ratio),
                           ("melting", melting_ratio)]:
            if buckets[cat]:
                active_ratios[cat] = ratio
                active_total += ratio
        for cat in active_ratios:
            active_ratios[cat] /= active_total

        self.active_ratios = active_ratios

    def __len__(self) -> int:
        # One epoch = one pass through every window (same cardinality as
        # the original dataset so epoch boundaries stay comparable).
        return len(self.dataset)

    def __iter__(self):
        n = len(self.dataset)
        indices: list[int] = []

        for cat, ratio in self.active_ratios.items():
            bucket = self.buckets[cat]
            count = max(1, int(round(n * ratio)))
            if bucket:
                # Sample with replacement if bucket is smaller than target.
                if len(bucket) >= count:
                    indices.extend(self.rng.sample(bucket, count))
                else:
                    indices.extend(self.rng.choices(bucket, k=count))

        # Trim or pad to exact length.
        if len(indices) > n:
            indices = indices[:n]
        elif len(indices) < n:
            # Pad from the largest bucket.
            largest = max(self.buckets.values(), key=len) if self.buckets else []
            if largest:
                indices.extend(self.rng.choices(largest, k=n - len(indices)))

        self.rng.shuffle(indices)
        return iter(indices)


def collate_temporal_batch(batch: list[dict]) -> dict:
    if len(batch) == 1:
        return batch[0]
    if len(batch) == 0:
        return {}
    return _collate_multi(batch)


def _collate_multi(batch: list[dict]) -> dict:
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], list):
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], bool):
            result[key] = torch.tensor(values)
        else:
            result[key] = values[0]
    return result
