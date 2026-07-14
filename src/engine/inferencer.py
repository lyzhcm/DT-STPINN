"""Digital Twin Inference Engine.

Handles forward inference and autoregressive prediction of future
temperature states for real-time monitoring and optimization.
"""
from __future__ import annotations

import torch


class Inferencer:
    def __init__(self, model, device: torch.device | None = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_single(self, graph_sequence: list, dt: float = 1.0) -> torch.Tensor:
        self.model.eval()
        seq = self._to_device(graph_sequence)
        output = self.model(seq, dt=dt)
        return output["T_pred"].cpu()

    @torch.no_grad()
    def autoregressive_predict(self, graph_sequence: list,
                                steps: int, dt: float = 1.0,
                                dynamic_graph=None) -> list[torch.Tensor]:
        """Autoregressively predict future temperature states.

        Uses predicted temperatures to construct pseudo-graphs for
        subsequent prediction steps.

        Args:
            graph_sequence: initial L-step observation window.
            steps: number of future steps to predict.
            dt: time step size.
            dynamic_graph: DynamicGraph instance for constructing pseudo-graphs.

        Returns:
            list of predicted temperature tensors, each [N].
        """
        self.model.eval()
        window = list(graph_sequence)
        predictions = []

        for step in range(steps):
            T_pred = self.predict_single(window, dt=dt)
            predictions.append(T_pred)

            if dynamic_graph is not None:
                next_t = len(graph_sequence) + step
                if next_t < dynamic_graph.num_steps:
                    next_graph = dynamic_graph.get_graph_at(next_t)
                else:
                    last_graph = window[-1].clone()
                    last_graph.x = dynamic_graph.node_feature_builder.build(
                        last_graph.coords, T_pred.squeeze(-1),
                        last_graph.mask.float(), dynamic_graph.layer_ids,
                        dynamic_graph._laser_positions[min(next_t, dynamic_graph.num_steps - 1)],
                        dynamic_graph._scan_directions[min(next_t, dynamic_graph.num_steps - 1)].item(),
                        dynamic_graph.times[min(next_t, dynamic_graph.num_steps - 1)].item(),
                        dt,
                    )
                    next_graph = last_graph
            else:
                last_graph = window[-1].clone()
                last_graph.x = last_graph.x
                next_graph = last_graph

            window.append(next_graph)
            window = window[1:]

        return predictions

    def predict_sequence(self, graph_sequence: list, 
                          dynamic_graph, start_t: int, steps: int,
                          dt: float = 1.0) -> list[torch.Tensor]:
        """Predict temperature for consecutive known future time steps
        using ground-truth graph structures from the dynamic graph.

        This is used for evaluation on held-out test data.
        """
        self.model.eval()
        seq = list(graph_sequence)
        predictions = []

        for i in range(steps):
            out = self.model(self._to_device(seq), dt=dt)
            T_pred = out["T_pred"].cpu()
            predictions.append(T_pred)

            next_data = dynamic_graph.get_graph_at(start_t + i + 1)
            seq.append(next_data)
            seq = seq[1:]

        return predictions

    def _to_device(self, graph_sequence: list) -> list:
        result = []
        for data in graph_sequence:
            d = data.clone()
            for attr in ["x", "y", "edge_index", "edge_attr", "mask", "coords"]:
                val = getattr(d, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(d, attr, val.to(self.device))
            result.append(d)
        return result
