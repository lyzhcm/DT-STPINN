# DT-STPINN Paper 1 Architecture

This document describes the implementation used for Paper 1: temperature-field
prediction for DED thin-walled parts. It is intentionally narrower than the full
DT-STPINN roadmap. Stress, displacement, heat-flux heads, MPC, and digital twin
closed-loop optimization are later-stage extensions.

## Scope

Paper 1 focuses on:

- masked dynamic graph construction from VTU/FEA data;
- spatial encoding with physics-aware graph attention;
- temporal encoding of thermal history;
- spatial-temporal cross fusion;
- temperature prediction;
- graph finite-difference physics losses.

## Data Flow

```text
VTU sequence
  -> DynamicGraph
  -> per-step PyG Data(x, edge_index, edge_attr, y, mask, coords)
  -> SpatialEncoder
  -> TemporalEncoder
  -> CrossFusion
  -> TemperatureHead
  -> DTSTPINNLoss
```

The model consumes a list of graph snapshots. For a sequence length `L`, node
count `N`, and hidden dimension `D`, the main tensors are:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `data.x` | `[N, 12]` | node features |
| `edge_index` | `[2, M]` | graph connectivity |
| `edge_attr` | `[M, 5]` | edge features |
| `mask` | `[N]` | active/deposited material mask |
| `z_s` | `[1, L, N, D]` | spatial features |
| `z_t` | `[1, L, N, D]` | temporal features |
| `z_f` | `[1, N, D]` | fused final-step features |
| `T_pred` | `[1, N, 1]` | predicted temperature |

Inside the loss, scalar node fields are normalized to `[N, 1]` and masks to
`[N]`.

## Dynamic Graph

The current implementation uses a full reference mesh plus a time-varying
`live` mask. This is a masked dynamic graph:

```text
G_t = (V_ref, E_ref, mask_t)
```

This is more stable for fixed VTU meshes than physically adding/removing nodes
at runtime. Inactive nodes are masked in the encoder output and loss terms.

Node features are currently 12-dimensional:

| Group | Features |
| --- | --- |
| Geometry | `x, y, z` |
| Thermal state | `T` |
| Process | distance to laser, layer id, sin/cos scan direction |
| Material | density, specific heat, conductivity, expansion coefficient |

Edge features are currently 5-dimensional:

| Feature | Meaning |
| --- | --- |
| distance | Euclidean distance |
| dx, dy, dz | relative direction |
| k_avg | averaged thermal conductivity |

## Spatial Encoder

The spatial encoder uses `GATv2Conv` with edge features. Multi-head attention is
used internally, but the output dimension is fixed to `hidden_dim`.

Important implementation choice:

```text
GATv2Conv(..., heads=heads, concat=False)
```

This keeps:

```text
SpatialEncoder.out_dim = hidden_dim
```

instead of `hidden_dim * heads`. The target Paper 1 hidden dimension is 256, so
the downstream Transformer and cross-attention operate on 256-dimensional
features, not 1024-dimensional features.

## Temporal Encoder

The temporal encoder applies a Transformer along the time axis for each node:

```text
[1, L, N, D] -> chunks over N -> [chunk_nodes, L, D] -> Transformer -> [1, L, N, D]
```

Chunking avoids constructing one giant attention batch for all nodes at once.
The current default is optimized for full-size VTU graphs:

- `D = 256`
- 4 Transformer layers
- 8 attention heads
- FFN dimension 1024

Training quality requires real VTU time sequences. Repeating a single timestep
is acceptable only as a smoke test.

## Cross Fusion

Cross fusion uses the final-step spatial features as queries and the temporal
history as keys/values:

```text
Q = spatial final step: [1, N, D]
K,V = temporal history: [1, L, N, D]
output = [1, N, D]
```

Like the temporal encoder, fusion is chunked over nodes to keep memory bounded.

## Decoder

Paper 1 uses only the temperature head:

```text
TemperatureHead: [1, N, 256] -> [1, N, 1]
```

Stress, displacement, and heat-flux heads are planned for later stages and are
disabled in `configs/paper1.yaml`.

## Loss

The current loss is:

```text
L = lambda_T * L_T
  + lambda_PDE * L_heat
  + lambda_BC * L_boundary
  + lambda_IC * L_initial
  + lambda_smooth * L_smooth
```

Physics terms use graph finite-difference approximations over the unstructured
mesh instead of direct autograd derivatives with respect to coordinates. This is
the intended implementation for VTU graph data.

Active masks are applied to supervised loss, PDE residuals, boundary loss, and
smoothness loss.

## Practical Defaults

Use `configs/paper1.yaml` for the current implementation:

- `node_feature_dim: 12`
- `edge_feature_dim: 5`
- `hidden_dim: 256`
- spatial heads: 4, `concat=False` in code
- temporal heads: 8
- fusion heads: 8
- `k_neighbors: 16` for fallback KNN graph construction
- `use_mesh_edges: true` when VTU mesh cells are available

For debugging, start with fewer VTU files and verify:

```text
GNN out dim: 256
T_pred shape: (1, N, 1)
loss runs without mask shape errors
```

For training, provide a real `Data-*.vtu` sequence. A single repeated timestep
does not validate the temporal-history part of the architecture.
