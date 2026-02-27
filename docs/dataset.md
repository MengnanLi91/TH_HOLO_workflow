# MooseDataset API

`MooseDataset` is a PyTorch `Dataset` that loads processed MOOSE simulation
data from Zarr stores and returns tensors ready for ML training.

**Source:** `src/dataset/moose_dataset.py`

## Constructor

```python
MooseDataset(zarr_dir, mode="graph", time_idx=-1)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `zarr_dir` | `str \| Path` | — | Directory containing `*.zarr` stores written by the ETL |
| `mode` | `str` | `"graph"` | Tensor representation — see [Modes](#modes) |
| `time_idx` | `int` | `-1` | If ≥ 0, return only this time step (removes the T dimension). `-1` returns all time steps. |

Each simulation run is one dataset item (`len(ds)` == number of `.zarr` stores).

## Modes

### `"graph"` — GNN / MeshGraphNet

```python
ds = MooseDataset("data/processed", mode="graph")
sample = ds[0]
```

| Key | Shape | dtype | Description |
|---|---|---|---|
| `coords` | `[N, D]` | float32 | Node spatial coordinates |
| `edge_index` | `[2, M]` | int64 | COO edge list — row 0 = source, row 1 = dest |
| `elem_fields` | `[T, E, F]` | float32 | Normalized solution fields per element |
| `node_fields` | `[T, N, F]` | float32 | Element fields scattered to nodes (centroid average) |
| `probe_data` | `dict[str, Tensor]` | float32 | `probe_name → [Np, C]` CSV probe arrays (empty if no probes) |
| `field_names` | `list[str]` | — | Field name for each index in the F dimension |
| `norm_stats` | `dict` | — | `{field: {"mean": float, "std": float}}` |
| `sim_name` | `str` | — | Simulation identifier |
| `time_steps` | `[T]` | float32 | Simulation time values |

### `"point_cloud"` — PointNet / Transformer

```python
ds = MooseDataset("data/processed", mode="point_cloud")
sample = ds[0]
```

| Key | Shape | dtype | Description |
|---|---|---|---|
| `coords` | `[N, D]` | float32 | Node spatial coordinates |
| `node_fields` | `[T, N, F]` | float32 | Per-node fields |
| `field_names` | `list[str]` | — | |
| `norm_stats` | `dict` | — | |
| `sim_name` | `str` | — | |
| `time_steps` | `[T]` | float32 | |

### `"grid"` — CNN / FNO / U-Net

```python
ds = MooseDataset("data/processed", mode="grid")
sample = ds[0]
```

| Key | Shape | dtype | Description |
|---|---|---|---|
| `grid_x` | `[Nx]` | float32 | x-coordinates of grid columns |
| `grid_y` | `[Ny]` | float32 | y-coordinates of grid rows |
| `grid_fields` | `[T, Nx, Ny, F]` | float32 | Fields interpolated onto a regular grid |
| `field_names` | `list[str]` | — | |
| `norm_stats` | `dict` | — | |
| `sim_name` | `str` | — | |
| `time_steps` | `[T]` | float32 | |

## Selecting a single time step

Pass `time_idx` to remove the T dimension from all field tensors:

```python
ds = MooseDataset("data/processed", mode="graph", time_idx=0)
sample = ds[0]
# elem_fields: [E, F]  (no T dim)
# node_fields: [N, F]
```

## Denormalization

Fields in the Zarr store are z-score normalized. Use `denormalize` to convert
a tensor back to physical units:

```python
ds = MooseDataset("data/processed", mode="grid")
sample = ds[0]

pressure_norm = sample["grid_fields"][..., 0]       # normalized
pressure_phys = ds.denormalize("pressure", pressure_norm)  # Pa (original units)
```

The method uses statistics from the first simulation in the dataset. If
normalization statistics vary per simulation, load them from `sample["norm_stats"]`
and apply manually:

```python
stats = sample["norm_stats"]["vel_x"]
vel_x_phys = vel_x_norm * stats["std"] + stats["mean"]
```

## Usage with DataLoader

```python
from torch.utils.data import DataLoader
from dataset.moose_dataset import MooseDataset

ds = MooseDataset("data/processed", mode="grid")

# Grid mode works directly with the default collate_fn
loader = DataLoader(ds, batch_size=4, shuffle=True)

# Graph mode: use a graph-aware collate (e.g. PyTorch Geometric DataLoader)
# or set batch_size=1 to avoid ragged tensor issues across simulations
# with different mesh sizes.
loader = DataLoader(ds, batch_size=1)

for batch in loader:
    fields = batch["grid_fields"]  # [B, T, Nx, Ny, F]
```

## Node field interpolation

`node_fields` is computed from `elem_fields` by a scatter-mean over the
element→node connectivity map: each node receives the average of the fields
from all elements that share it.

This is equivalent to (but does not require) a full finite-element interpolation
and is sufficient for GNN message-passing architectures where the model learns
its own interpolation weights.
