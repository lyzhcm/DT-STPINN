"""Utilities for the prescribed Flow-3D AM ``additive_z_scan`` path.

The training dataset, diagnostics, and feature-export scripts should use the
same geometry/time convention.  This module keeps that convention explicit and
independent from the graph object so it can be tested without loading all VTU
temperatures.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _vec3(value: str | list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    if isinstance(value, str):
        parts = [float(p.strip()) for p in value.split(",") if p.strip()]
    else:
        parts = [float(p) for p in value]
    if len(parts) != 3:
        raise ValueError(f"Expected a 3-vector, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _unit(value: str | list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    vec = _vec3(value)
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        raise ValueError(f"Cannot normalize zero vector {value!r}")
    return vec / norm


@dataclass(frozen=True)
class AdditiveZScanPath:
    start_point_mm: np.ndarray
    scan_direction: np.ndarray
    scan_length_mm: float
    hatch_direction: np.ndarray
    hatch_count: int
    hatch_spacing_mm: float
    layer_count: int
    layer_thickness_mm: float
    velocity_mm_s: float
    each_path_time_s: float = 0.001
    each_layer_time_s: float = 0.0
    time_scale_to_s: float = 1.0e-3
    time_offset_s: float = 0.0
    alternate_layer_scan_direction: bool = False
    reverse_hatch_order_parity: int = -1
    body_radius_mm: float = 0.1
    body_height_mm: float = 0.1
    body_radius_front_mm: float = 0.4
    body_radius_back_mm: float = 0.4

    @classmethod
    def from_config(cls, config) -> "AdditiveZScanPath":
        data = getattr(config, "data", config)
        return cls(
            start_point_mm=_vec3(getattr(data, "laser_start_point_mm")),
            scan_direction=_unit(getattr(data, "laser_scan_direction")),
            scan_length_mm=float(getattr(data, "laser_scan_length_mm")),
            hatch_direction=_unit(getattr(data, "laser_hatch_direction")),
            hatch_count=int(getattr(data, "laser_hatch_count")),
            hatch_spacing_mm=float(getattr(data, "laser_hatch_spacing_mm")),
            layer_count=int(getattr(data, "laser_layer_count")),
            layer_thickness_mm=float(getattr(data, "laser_layer_thickness_mm")),
            velocity_mm_s=float(getattr(data, "laser_velocity_mm_s")),
            each_path_time_s=float(getattr(data, "laser_each_path_time_s")),
            each_layer_time_s=float(getattr(data, "laser_each_layer_time_s")),
            time_scale_to_s=float(getattr(data, "laser_path_time_scale_to_s")),
            time_offset_s=float(getattr(data, "laser_path_time_offset_s", 0.0)),
            alternate_layer_scan_direction=bool(
                getattr(data, "laser_alternate_layer_scan_direction", False)
            ),
            reverse_hatch_order_parity=int(
                getattr(data, "laser_reverse_hatch_order_parity", -1)
            ),
            body_radius_mm=float(getattr(data, "laser_body_radius_mm", 0.1)),
            body_height_mm=float(getattr(data, "laser_body_height_mm", 0.1)),
            body_radius_front_mm=float(getattr(data, "laser_body_radius_front_mm", 0.4)),
            body_radius_back_mm=float(getattr(data, "laser_body_radius_back_mm", 0.4)),
        )

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        *,
        time_scale_to_s: float = 1.0e-3,
        time_offset_s: float = 0.0,
        alternate_layer_scan_direction: bool = False,
        reverse_hatch_order_parity: int = -1,
    ) -> "AdditiveZScanPath":
        root = ET.parse(str(xml_path)).getroot()
        scan = root.find(".//additive_z_scan")
        if scan is None:
            raise ValueError(f"No <additive_z_scan> block found in {xml_path}")

        def text(name: str, default: str | None = None) -> str:
            node = scan.find(name)
            if node is None or node.text is None:
                if default is None:
                    raise ValueError(f"Missing <{name}> in additive_z_scan")
                return default
            return node.text.strip()

        body_radius = 0.1
        body_height = 0.1
        body_front = 0.4
        body_back = 0.4
        heat_name = text("heatsource", "")
        for heat in root.findall(".//heat_source"):
            name = heat.findtext("name", default="").strip()
            if heat_name and name != heat_name:
                continue
            ellipsoid = heat.find("ellipsoid_body_source")
            if ellipsoid is not None:
                body_radius = float(ellipsoid.findtext("radius", default=str(body_radius)))
                body_height = float(ellipsoid.findtext("height", default=str(body_height)))
                body_front = float(ellipsoid.findtext("radius_front", default=str(body_front)))
                body_back = float(ellipsoid.findtext("radius_back", default=str(body_back)))
            break

        return cls(
            start_point_mm=_vec3(text("start_point")),
            scan_direction=_unit(text("direction")),
            scan_length_mm=float(text("direction_length")),
            hatch_direction=_unit(text("z_direction")),
            hatch_count=int(text("z_direction_num")),
            hatch_spacing_mm=float(text("z_direction_distant")),
            layer_count=int(text("layer_num")),
            layer_thickness_mm=float(text("layer_thick")),
            velocity_mm_s=float(text("velocity")),
            each_path_time_s=float(text("each_path_time", "0")),
            each_layer_time_s=float(text("each_layer_time", "0")),
            time_scale_to_s=float(time_scale_to_s),
            time_offset_s=float(time_offset_s),
            alternate_layer_scan_direction=bool(alternate_layer_scan_direction),
            reverse_hatch_order_parity=int(reverse_hatch_order_parity),
            body_radius_mm=body_radius,
            body_height_mm=body_height,
            body_radius_front_mm=body_front,
            body_radius_back_mm=body_back,
        )

    @property
    def scan_time_s(self) -> float:
        return self.scan_length_mm / max(self.velocity_mm_s, 1.0e-12)

    @property
    def path_period_s(self) -> float:
        return self.scan_time_s + max(self.each_path_time_s, 0.0)

    @property
    def layer_period_s(self) -> float:
        return self.path_period_s * self.hatch_count + max(self.each_layer_time_s, 0.0)

    @property
    def total_tracks(self) -> int:
        return self.hatch_count * self.layer_count

    def physical_track(self, layer_idx: int, track_in_layer: int) -> int:
        physical = int(track_in_layer)
        if self.reverse_hatch_order_parity in (0, 1):
            if int(layer_idx) % 2 == self.reverse_hatch_order_parity:
                physical = self.hatch_count - 1 - physical
        return int(np.clip(physical, 0, self.hatch_count - 1))

    def program_track(self, layer_idx: int, physical_track: int) -> int:
        track = int(physical_track)
        if self.reverse_hatch_order_parity in (0, 1):
            if int(layer_idx) % 2 == self.reverse_hatch_order_parity:
                track = self.hatch_count - 1 - track
        return int(np.clip(track, 0, self.hatch_count - 1))

    def line_start(self, layer_idx: int, physical_track: int) -> np.ndarray:
        return (
            self.start_point_mm
            + self.hatch_direction * (float(physical_track) * self.hatch_spacing_mm)
            + np.asarray([0.0, 0.0, float(layer_idx) * self.layer_thickness_mm])
        )

    def is_forward(self, layer_idx: int, track_in_layer: int) -> bool:
        forward = int(track_in_layer) % 2 == 0
        if self.alternate_layer_scan_direction and int(layer_idx) % 2 == 1:
            forward = not forward
        return forward

    def segment(self, layer_idx: int, track_in_layer: int) -> dict[str, object]:
        physical = self.physical_track(layer_idx, track_in_layer)
        base = self.line_start(layer_idx, physical)
        forward = self.is_forward(layer_idx, track_in_layer)
        if forward:
            start = base
            end = base + self.scan_direction * self.scan_length_mm
            active_dir = self.scan_direction
        else:
            start = base + self.scan_direction * self.scan_length_mm
            end = base
            active_dir = -self.scan_direction
        elapsed_start_s = layer_idx * self.layer_period_s + track_in_layer * self.path_period_s
        return {
            "layer": layer_idx,
            "track_in_layer": track_in_layer,
            "physical_track": physical,
            "forward": forward,
            "active_direction": active_dir,
            "start": start,
            "end": end,
            "elapsed_start_s": elapsed_start_s,
            "elapsed_end_s": elapsed_start_s + self.scan_time_s,
        }

    def iter_segments(self):
        for layer_idx in range(self.layer_count):
            for track_in_layer in range(self.hatch_count):
                yield self.segment(layer_idx, track_in_layer)

    def raw_to_elapsed_s(self, raw_time: float, *, raw_origin: float = 0.0) -> float:
        return max(
            0.0,
            float(raw_time) * self.time_scale_to_s
            - float(raw_origin) * self.time_scale_to_s
            - self.time_offset_s,
        )

    def elapsed_to_raw(self, elapsed_s: float, *, raw_origin: float = 0.0) -> float:
        return (
            float(raw_origin) * self.time_scale_to_s + self.time_offset_s + float(elapsed_s)
        ) / max(self.time_scale_to_s, 1.0e-12)

    def state_at_raw_time(self, raw_time: float, *, raw_origin: float = 0.0) -> dict[str, object]:
        elapsed = self.raw_to_elapsed_s(raw_time, raw_origin=raw_origin)
        layer_idx = min(int(elapsed // self.layer_period_s), self.layer_count - 1)
        layer_time = elapsed - layer_idx * self.layer_period_s
        track_in_layer = min(int(layer_time // self.path_period_s), self.hatch_count - 1)
        local_path_time = layer_time - track_in_layer * self.path_period_s
        progress = float(np.clip(local_path_time / max(self.scan_time_s, 1.0e-12), 0.0, 1.0))
        seg = self.segment(layer_idx, track_in_layer)
        base = self.line_start(layer_idx, int(seg["physical_track"]))
        if bool(seg["forward"]):
            pos = base + self.scan_direction * (self.scan_length_mm * progress)
        else:
            pos = base + self.scan_direction * (self.scan_length_mm * (1.0 - progress))
        active_dir = np.asarray(seg["active_direction"], dtype=np.float64)
        return {
            **seg,
            "raw_time": float(raw_time),
            "elapsed_s": elapsed,
            "local_path_time_s": local_path_time,
            "progress": progress,
            "on_scan": local_path_time <= self.scan_time_s,
            "position": pos,
            "scan_angle_rad": math.atan2(float(active_dir[1]), float(active_dir[0])),
        }

    def node_process_features(
        self,
        coords_mm: np.ndarray,
        raw_time: float,
        *,
        raw_origin: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        coords = np.asarray(coords_mm, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, 3)
        state = self.state_at_raw_time(raw_time, raw_origin=raw_origin)
        laser = np.asarray(state["position"], dtype=np.float64)
        active_dir = np.asarray(state["active_direction"], dtype=np.float64)
        rel = coords - laser.reshape(1, 3)
        along_current = rel @ active_dir.reshape(3, 1)
        cross_vec = rel - along_current * active_dir.reshape(1, 3)
        current_distance = np.linalg.norm(rel, axis=1, keepdims=True)
        current_cross_distance = np.linalg.norm(cross_vec, axis=1, keepdims=True)
        front = along_current >= 0.0
        along_radius = np.where(front, self.body_radius_front_mm, self.body_radius_back_mm)
        cross_norm = current_cross_distance / max(self.body_radius_mm, 1.0e-12)
        along_norm = np.abs(along_current) / np.maximum(along_radius, 1.0e-12)
        depth_norm = np.abs(rel[:, 2:3]) / max(self.body_height_mm, 1.0e-12)
        ellipsoid_body_score = cross_norm**2 + along_norm**2 + depth_norm**2
        ellipsoid_heat_proxy = np.exp(
            -2.0 * cross_norm**2 - 2.0 * along_norm**2 - depth_norm**2
        )
        in_laser_ellipsoid = (
            (ellipsoid_body_score <= 1.0) & bool(state.get("on_scan", True))
        ).astype(np.float64)

        layer_float = (coords[:, 2] - self.start_point_mm[2]) / max(self.layer_thickness_mm, 1.0e-12)
        layer_idx = np.clip(np.rint(layer_float), 0, self.layer_count - 1).astype(np.int64)
        rel_to_start = coords - self.start_point_mm.reshape(1, 3)
        physical_track = np.clip(
            np.rint((rel_to_start @ self.hatch_direction.reshape(3, 1)).reshape(-1) / max(self.hatch_spacing_mm, 1.0e-12)),
            0,
            self.hatch_count - 1,
        ).astype(np.int64)

        arrival_elapsed = np.empty((coords.shape[0], 1), dtype=np.float64)
        line_along = np.empty((coords.shape[0], 1), dtype=np.float64)
        line_cross = np.empty((coords.shape[0], 1), dtype=np.float64)
        direction_sign = np.empty((coords.shape[0], 1), dtype=np.float64)
        program_track = np.empty((coords.shape[0], 1), dtype=np.float64)
        in_track_neighborhood = np.empty((coords.shape[0], 1), dtype=np.float64)
        for i in range(coords.shape[0]):
            layer = int(layer_idx[i])
            physical = int(physical_track[i])
            track = self.program_track(layer, physical)
            base = self.line_start(layer, physical)
            rel_line = coords[i] - base
            s = float(np.dot(rel_line, self.scan_direction))
            s_clamped = float(np.clip(s, 0.0, self.scan_length_mm))
            cross = rel_line - s * self.scan_direction
            forward = self.is_forward(layer, track)
            scan_elapsed = s_clamped / self.velocity_mm_s if forward else (self.scan_length_mm - s_clamped) / self.velocity_mm_s
            arrival = layer * self.layer_period_s + track * self.path_period_s + scan_elapsed
            arrival_elapsed[i, 0] = arrival
            line_along[i, 0] = s
            line_cross[i, 0] = float(np.linalg.norm(cross))
            direction_sign[i, 0] = 1.0 if forward else -1.0
            program_track[i, 0] = float(track)
            in_track_neighborhood[i, 0] = 1.0 if (0.0 <= s <= self.scan_length_mm and line_cross[i, 0] <= self.body_radius_front_mm) else 0.0

        elapsed = self.raw_to_elapsed_s(raw_time, raw_origin=raw_origin)
        arrival_raw = np.asarray(
            [self.elapsed_to_raw(v, raw_origin=raw_origin) for v in arrival_elapsed.reshape(-1)],
            dtype=np.float64,
        ).reshape(-1, 1)
        time_to_arrival_s = arrival_elapsed - elapsed
        layer_norm = layer_idx.reshape(-1, 1).astype(np.float64) / max(self.layer_count - 1, 1)
        physical_track_norm = physical_track.reshape(-1, 1).astype(np.float64) / max(self.hatch_count - 1, 1)
        program_track_norm = program_track / max(self.hatch_count - 1, 1)

        columns = [
            "laser_x_mm", "laser_y_mm", "laser_z_mm",
            "dx_mm", "dy_mm", "dz_mm",
            "distance_to_laser_mm", "current_along_mm", "current_cross_mm",
            "ellipsoid_body_score", "ellipsoid_heat_proxy", "in_laser_ellipsoid",
            "line_along_mm", "line_cross_mm", "time_to_arrival_s", "arrival_raw_time",
            "layer_idx", "track_physical", "track_program", "direction_sign",
            "layer_norm", "track_physical_norm", "track_program_norm",
            "in_track_neighborhood",
        ]
        features = np.concatenate(
            [
                np.repeat(laser.reshape(1, 3), coords.shape[0], axis=0),
                rel,
                current_distance,
                along_current,
                current_cross_distance,
                ellipsoid_body_score,
                ellipsoid_heat_proxy,
                in_laser_ellipsoid,
                line_along,
                line_cross,
                time_to_arrival_s,
                arrival_raw,
                layer_idx.reshape(-1, 1).astype(np.float64),
                physical_track.reshape(-1, 1).astype(np.float64),
                program_track,
                direction_sign,
                layer_norm,
                physical_track_norm,
                program_track_norm,
                in_track_neighborhood,
            ],
            axis=1,
        )
        return features.astype(np.float32), columns
