"""Evaluation metrics for temperature prediction."""
from __future__ import annotations

import numpy as np
import torch


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        pred: predicted values.
        target: ground truth values.

    Returns:
        dict with RMSE, MAE, R², MaxError, MAPE.
    """
    pred = pred.float()
    target = target.float()

    error = pred - target
    mse = (error ** 2).mean().item()
    rmse = mse ** 0.5
    mae = error.abs().mean().item()
    abs_error = error.abs()
    max_error = abs_error.max().item()
    # torch.quantile rejects tensors above an internal element-count limit.
    # NumPy's partition-based implementation handles the full validation set.
    percentile_values = np.quantile(
        abs_error.detach().reshape(-1).cpu().numpy(),
        [0.50, 0.90, 0.95, 0.99],
        method="linear",
    )

    ss_tot = ((target - target.mean()) ** 2).sum().item()
    ss_res = (error ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    mask = target.abs() > 1e-6
    if mask.any():
        mape = (error[mask].abs() / target[mask].abs()).mean().item() * 100
    else:
        mape = 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MaxError": max_error,
        "AbsErrorP50": float(percentile_values[0]),
        "AbsErrorP90": float(percentile_values[1]),
        "AbsErrorP95": float(percentile_values[2]),
        "AbsErrorP99": float(percentile_values[3]),
        "MAPE": mape,
    }
