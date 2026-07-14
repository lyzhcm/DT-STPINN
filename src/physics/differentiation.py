"""Graph-based spatial differentiation for PINN physics losses.

Computes spatial gradients and Laplacians on unstructured graph meshes
using graph finite difference approximations. All operations are
differentiable and compatible with PyTorch autograd.
"""
from __future__ import annotations

import torch
from torch_scatter import scatter_mean


def spatial_gradient(node_values: torch.Tensor, coords: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradient ∇f at each node.

    ∇f_i = mean_j∈N(i) [(f_j - f_i) * (x_j - x_i) / ||x_j - x_i||²]

    Args:
        node_values: [N] or [N, 1] scalar field.
        coords: [N, 3] spatial coordinates.
        edge_index: [2, M] graph connectivity.

    Returns:
        [N, 3] gradient vectors.
    """
    f = node_values.view(-1)
    row, col = edge_index[0], edge_index[1]

    df = f[col] - f[row]
    dx = coords[col] - coords[row]
    d_sq = (dx ** 2).sum(dim=1) + 1e-8

    contrib = (df / d_sq).unsqueeze(-1) * dx

    grad = scatter_mean(contrib, row, dim=0, dim_size=coords.shape[0])
    return grad


def spatial_laplacian(node_values: torch.Tensor, coords: torch.Tensor,
                      edge_index: torch.Tensor) -> torch.Tensor:
    """Compute graph Laplacian ∇²f at each node.

    ∇²f_i = mean_j∈N(i) [(f_j - f_i) / ||x_j - x_i||²] * 2 * d

    where d is the spatial dimension (3 for 3D).

    Args:
        node_values: [N] or [N, 1] scalar field.
        coords: [N, D] spatial coordinates.
        edge_index: [2, M] graph connectivity.

    Returns:
        [N] Laplacian values.
    """
    f = node_values.view(-1)
    row, col = edge_index[0], edge_index[1]

    df = f[col] - f[row]
    d_sq = (coords[col] - coords[row]).pow(2).sum(dim=1) + 1e-8

    contrib = df / d_sq

    laplacian = scatter_mean(contrib, row, dim=0, dim_size=coords.shape[0])
    laplacian = laplacian * 2 * coords.shape[1]
    return laplacian


def divergence(vector_field: torch.Tensor, coords: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
    """Compute divergence ∇·v of a vector field on the graph.

    Args:
        vector_field: [N, D] vector field.
        coords: [N, D] spatial coordinates.
        edge_index: [2, M] graph connectivity.

    Returns:
        [N] divergence values.
    """
    row, col = edge_index[0], edge_index[1]

    dv = vector_field[col] - vector_field[row]
    dx = coords[col] - coords[row]
    d_sq = (dx ** 2).sum(dim=1) + 1e-8

    contrib = (dv * dx).sum(dim=1) / d_sq

    div = scatter_mean(contrib, row, dim=0, dim_size=coords.shape[0])
    return div


def graph_smoothness_loss(node_values: torch.Tensor,
                          edge_index: torch.Tensor,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
    """Graph Laplacian smoothness regularization.

    L_smooth = (1/M) * Σ_{ij} A_ij * (f_i - f_j)²

    Args:
        node_values: [N] or [N, 1] field to regularize.
        edge_index: [2, M] graph connectivity.
        mask: optional [N] active-node mask. Only active-active edges are used.

    Returns:
        scalar smoothness loss.
    """
    f = node_values.view(-1)
    row, col = edge_index[0], edge_index[1]
    if mask is not None:
        active_edges = mask[row] & mask[col]
        row, col = row[active_edges], col[active_edges]
        if row.numel() == 0:
            return torch.tensor(0.0, device=node_values.device, dtype=node_values.dtype)
    loss = ((f[row] - f[col]) ** 2).mean()
    return loss
