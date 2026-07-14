"""Evaluation metrics for temperature prediction."""
from __future__ import annotations

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
    max_error = error.abs().max().item()

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
        "MAPE": mape,
    }
