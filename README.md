# TH_HOLO_workflow

A [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)-based data pipeline for
reading MOOSE thermal-hydraulics simulation outputs and preparing them for ML
training (surrogate modelling, super-resolution, state reconstruction).

## Repository layout

```
TH_HOLO_workflow/
├── data/                        # Raw simulation files (not tracked by git)
│   ├── *.e                      # Exodus mesh + field output files
│   ├── *_out_*_*.csv            # CSV line-probe output files
│   └── processed/               # Generated Zarr stores (written by ETL)
├── docs/
│   ├── etl_pipeline.md          # ETL pipeline deep-dive
│   └── dataset.md               # PyTorch Dataset API reference
├── src/
│   ├── run_etl.py               # ETL entry point (Hydra CLI)
│   ├── read_exdous.py           # Exodus inspection helpers
│   ├── moose_etl/               # ETL package
│   │   ├── schemas.py           # MooseRawData / MooseProcessedData dataclasses
│   │   ├── config/moose_etl.yaml
│   │   ├── data_sources/
│   │   │   ├── exodus_source.py # ExodusDataSource — reads .e files
│   │   │   ├── csv_source.py    # CSVProbeSource  — reads CSV line probes
│   │   │   └── zarr_sink.py     # MooseZarrSink   — writes Zarr stores
│   │   ├── transformations/
│   │   │   └── moose_transform.py  # MooseDataTransformation
│   │   └── validators.py        # MooseDatasetValidator
│   └── dataset/
│       └── moose_dataset.py     # PyTorch Dataset (graph / point_cloud / grid)
├── moose/                       # git submodule — MOOSE framework
└── physicsnemo-curator/         # git submodule — PhysicsNeMo Curator base classes
```

## Environment setup

The pipeline runs inside the `moose-physicsnemo` conda environment.

```bash
conda activate moose-physicsnemo

# Install physicsnemo-curator from the submodule (first time only)
pip install -e physicsnemo-curator/

# netCDF4 is required for reading Exodus files
pip install netCDF4
```

## Quick start

### 1. Run the ETL pipeline

```bash
cd src/
python run_etl.py \
    etl.source.input_dir=../data \
    etl.source.data_dir=../data \
    etl.sink.output_dir=../data/processed
```

This reads every `*.e` file under `data/`, runs the full
normalize → graph → grid-interpolate pipeline, and writes one
`{sim_name}.zarr` store per simulation to `data/processed/`.

See [`docs/etl_pipeline.md`](docs/etl_pipeline.md) for configuration options
and a full description of the pipeline stages.

### 2. Load data for training

```python
import sys
sys.path.insert(0, "src")

from dataset.moose_dataset import MooseDataset

# Graph mode — for GNN / MeshGraphNet models
ds = MooseDataset("data/processed", mode="graph")
sample = ds[0]
# sample keys: coords, edge_index, elem_fields, node_fields,
#              probe_data, field_names, norm_stats, sim_name, time_steps

# Grid mode — for CNN / FNO models
ds_grid = MooseDataset("data/processed", mode="grid")
sample = ds_grid[0]
# sample keys: grid_x, grid_y, grid_fields, field_names, norm_stats, ...

# Select a single time step
ds_t0 = MooseDataset("data/processed", mode="graph", time_idx=0)
```

See [`docs/dataset.md`](docs/dataset.md) for the full Dataset API.

## Data sources

| File pattern | Description |
|---|---|
| `{sim_name}.e` | Exodus II file — mesh geometry + element solution fields |
| `{sim_prefix}_out_{probe_name}_{timestep:04d}.csv` | CSV line-probe — TKE, TKED, pressure, vel_x, vel_y, x, y, z |

The Exodus and CSV files do **not** need to share the same filename prefix —
they are treated as independent data sources.

## Further reading

- [ETL pipeline](docs/etl_pipeline.md) — stages, config keys, Zarr layout
- [Dataset API](docs/dataset.md) — modes, tensor shapes, denormalization
