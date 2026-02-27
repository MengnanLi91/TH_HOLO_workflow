# ETL Pipeline

The ETL pipeline converts raw MOOSE simulation outputs (Exodus + CSV) into
compressed Zarr stores that are ready for ML training.

## Architecture

The pipeline follows the [PhysicsNeMo Curator](https://github.com/NVIDIA/physicsnemo)
ETL pattern:

```
ExodusDataSource ──► MooseDataTransformation ──► MooseZarrSink
     │                                                  │
     │ reads .e + CSV                     writes .zarr  │
     └──────────────────────────────────────────────────┘
                ETLOrchestrator (multiprocessing)
```

| Component | Class | File |
|---|---|---|
| Source | `ExodusDataSource` | `src/moose_etl/data_sources/exodus_source.py` |
| CSV reader | `CSVProbeSource` | `src/moose_etl/data_sources/csv_source.py` |
| Transform | `MooseDataTransformation` | `src/moose_etl/transformations/moose_transform.py` |
| Sink | `MooseZarrSink` | `src/moose_etl/data_sources/zarr_sink.py` |
| Schemas | `MooseRawData`, `MooseProcessedData` | `src/moose_etl/schemas.py` |

## Pipeline stages

### Stage 1 — ExodusDataSource

Reads one Exodus `.e` file and its associated CSV line-probe files.

**Exodus extraction:**
- Node coordinates (`coordx`, `coordy`, optionally `coordz`) → `coords [N, D]`
- Element connectivity (`connect1`) — converted from 1-indexed to 0-indexed → `connectivity [E, K]`
- Time values (`time_whole`) → `time_steps [T]`
- Element variable names (decoded from `name_elem_var`) → `field_names`
- Element variable arrays (`vals_elem_var{i}eb1`) → `fields [T, E, F]`

**CSV co-reading (`CSVProbeSource`):**

Looks for files matching `{sim_name}_out_{probe_name}_{timestep:04d}.csv` in the
configured data directory. Each probe file is read into a `[Np, C]` array where
`Np` is the number of probe points and `C` is the number of columns
(TKE, TKED, id, pressure, vel_x, vel_y, x, y, z).

The Exodus and CSV files use independent filename prefixes and are not required
to match.

### Stage 2 — MooseDataTransformation

Three sub-steps applied in order:

**2a. Per-field z-score normalization**

```
normalized = (field - mean) / (std + eps)
```

Statistics are computed per field across all time steps and elements.
`mean` and `std` are stored in the output Zarr for later denormalization.

**2b. Graph edge construction**

For each element, all pairs of its nodes are connected in both directions
(undirected graph). Duplicate edges (shared between adjacent elements) are
deduplicated.

For a quad mesh with `E` elements and `K=4` nodes per element this produces
`C(4,2) × 2 × E` candidate edges before deduplication.

**2c. Regular-grid interpolation**

Element centroids (mean of node positions) are scattered onto a uniform
`Nx × Ny` grid using `scipy.interpolate.griddata` (linear method, `fill_value=0`).
The default resolution is 64 × 64.

### Stage 3 — MooseZarrSink

Writes one Zarr store per simulation with Blosc/zstd compression (level 3,
shuffle enabled). Chunk sizes target ~1 MB per chunk.

## Output Zarr layout

```
{sim_name}.zarr/
├── mesh/
│   ├── coords          float32 [N, D]    node coordinates
│   ├── connectivity    int32   [E, K]    element→node (0-indexed)
│   ├── edge_src        int32   [M]       graph edge source nodes
│   └── edge_dst        int32   [M]       graph edge destination nodes
├── fields/
│   ├── pressure        float32 [T, E]    normalized element pressure
│   ├── vel_x           float32 [T, E]    normalized x-velocity
│   └── vel_y           float32 [T, E]    normalized y-velocity
├── probes/
│   └── {probe_name}    float32 [Np, C]   CSV line-probe values (raw)
├── grid/
│   ├── x               float32 [Nx]      column x-coordinates
│   ├── y               float32 [Ny]      row y-coordinates
│   ├── pressure        float32 [T,Nx,Ny] interpolated pressure
│   ├── vel_x           float32 [T,Nx,Ny] interpolated x-velocity
│   └── vel_y           float32 [T,Nx,Ny] interpolated y-velocity
└── metadata/
    ├── time_steps      float32 [T]
    ├── attrs: field_names (JSON), probe_columns (JSON), sim_name
    └── norm_stats/
        ├── pressure/   attrs: mean, std
        ├── vel_x/      attrs: mean, std
        └── vel_y/      attrs: mean, std
```

## Configuration

The pipeline is configured via Hydra. The base config lives at
`src/moose_etl/config/moose_etl.yaml`. Any key can be overridden on the CLI.

```bash
python run_etl.py \
    etl.source.input_dir=../data \
    etl.source.data_dir=../data \
    etl.sink.output_dir=../data/processed \
    etl.processing.num_processes=8 \
    etl.transformations.moose_transform.grid_nx=128 \
    etl.transformations.moose_transform.grid_ny=128 \
    etl.sink.compression_level=6
```

### Full config reference

| Key | Default | Description |
|---|---|---|
| `etl.processing.num_processes` | `4` | Worker processes for parallel file processing |
| `etl.source.input_dir` | *(required)* | Directory containing Exodus `.e` files |
| `etl.source.data_dir` | *(required)* | Directory containing CSV probe files |
| `etl.transformations.moose_transform.grid_nx` | `64` | Grid columns for CNN output |
| `etl.transformations.moose_transform.grid_ny` | `64` | Grid rows for CNN output |
| `etl.transformations.moose_transform.eps` | `1e-8` | Stability epsilon for normalization |
| `etl.sink.output_dir` | *(required)* | Output directory for Zarr stores |
| `etl.sink.overwrite_existing` | `true` | Overwrite existing Zarr stores |
| `etl.sink.compression_level` | `3` | Blosc compression level (1–9) |
| `etl.sink.compression_method` | `zstd` | Blosc codec name |
| `etl.sink.chunk_size_mb` | `1.0` | Target chunk size in MB |

## Verified output (lid-driven-segregated_out)

```
mesh/coords        (10201, 2)    101×101 nodes on a unit square
mesh/connectivity  (10000, 4)    10 000 quad elements
mesh/edge_src/dst  (80400,)      graph edges (deduped, both directions)
fields/*           (2, 10000)    2 time steps × 10 000 elements
grid/*             (2, 64, 64)   2 time steps × 64×64 interpolated grid
```
