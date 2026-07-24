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
from torch.utils.data import Dataset, Sampler


class DEDTemporalDataset(Dataset):
    def __init__(self, dynamic_graph, window_size: int = 16,
                 predict_steps: int = 1, time_indices: list[int] | None = None):
        self.graph = dynamic_graph
        self.window_size = window_size
        self.predict_steps = predict_steps

        total = dynamic_graph.num_steps
        if time_indices is None:
            time_indices = list(range(total))

        self.valid_starts = []
        for idx in time_indices:
            if idx + window_size + predict_steps <= total:
                self.valid_starts.append(idx)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        start_t = self.valid_starts[idx]
        seq = self.graph.get_sequence(start_t, self.window_size)
        target_t = start_t + self.window_size + self.predict_steps - 1
        target_data = self.graph.get_graph_at(target_t)

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
            "target_step": target_t,
            "target_time": self.graph.times[target_t].item(),
        }

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
        self.solidus = solidus
        self.hot_threshold = hot_threshold
        self.normal_ratio = normal_ratio
        self.hot_ratio = hot_ratio
        self.melting_ratio = melting_ratio
        self.rng = random.Random(seed)

        # --- classify every dataset index once ---
        buckets: dict[str, list[int]] = {"normal": [], "hot": [], "melting": []}
        for i in range(len(dataset)):
            mt = dataset.target_max_temperature(i)
            if mt >= solidus:
                buckets["melting"].append(i)
            elif mt >= hot_threshold:
                buckets["hot"].append(i)
            else:
                buckets["normal"].append(i)

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
