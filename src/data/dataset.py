"""PyG Temporal Dataset for DED processing.

Creates sliding-window sequences from the DynamicGraph for training
the DT-STPINN model on temperature prediction.
"""
from __future__ import annotations

from torch.utils.data import Dataset


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
        }


def collate_temporal_batch(batch: list[dict]) -> dict:
    if len(batch) == 1:
        return batch[0]
    if len(batch) == 0:
        return {}
    return _collate_multi(batch)


def _collate_multi(batch: list[dict]) -> dict:
    import torch

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
