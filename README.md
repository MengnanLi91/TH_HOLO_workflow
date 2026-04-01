# TH_HOLO_workflow

[![pytest](https://github.com/MengnanLi91/TH_HOLO_workflow/actions/workflows/pytest.yml/badge.svg)](https://github.com/MengnanLi91/TH_HOLO_workflow/actions/workflows/pytest.yml)

TH_HOLO_workflow is a PhysicsNeMo-based ETL pipeline that converts MOOSE
thermal-hydraulics outputs (Exodus + CSV probes) into ML-ready Zarr datasets.

## TH_HOLO_workflow Plot

```mermaid
flowchart LR
    A["MOOSE outputs (.e + CSV probes)"]
    B["ETL pipeline (read, transform, validate)"]
    C["Processed dataset (*.zarr)"]
    D["Training interface: MooseDataset (graph | point_cloud | grid)"]
    A --> B --> C --> D
```

## What It Does

- Reads simulation outputs from Exodus `.e` files and CSV line-probe files.
- Normalizes fields and creates graph and regular-grid representations.
- Writes one compressed `.zarr` store per simulation run.
- Provides a PyTorch dataset interface for graph, point-cloud, and grid training.

## Quick Start

```bash
git submodule update --init --recursive
docker compose build etl-dev
docker compose run --rm etl-dev bash -lc 'cd src && python run_etl.py --config-name lid_driven'
```

The `lid_driven` config is defined in `src/moose_etl/config/lid_driven.yaml` and writes output to `data/processed/lid-driven/*.zarr`.

You can still override values on the command line if needed:

```bash
docker compose run --rm etl-dev bash -lc 'cd src && python run_etl.py --config-name lid_driven \
  etl.processing.num_processes=8'
```

To create a new dataset config, copy `src/moose_etl/config/lid_driven.yaml` to
`src/moose_etl/config/<your_config>.yaml`, update the source/sink paths, then run:

```bash
docker compose run --rm etl-dev bash -lc 'cd src && python run_etl.py --config-name <your_config>'
```

## Train an FNO with PhysicsNeMo

After ETL generates `*.zarr` stores, train with the generic framework using
the FNO example config at `src/config/fno.yaml`.

```bash
docker compose build etl
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name fno'
```

Use `etl-ngc` instead of `etl` if you prefer the NGC PhysicsNeMo base image.
Override config values directly on the CLI, for example:

```bash
docker compose run --rm etl bash -lc 'cd src && python train.py --config-name fno training.epochs=50'
```

## Evaluate an FNO Checkpoint

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-name fno'
```

To save ground-truth vs predicted velocity-field plots during evaluation:

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-name fno \
  output.plot_dir=../data/models/lid_driven_fno_plots'
```

## Train an MLP Surrogate for Darcy Resistance

The alpha-D workflow extracts Darcy resistance coefficient profiles from a
parametric study of flow contraction-expansion simulations, then trains a
PhysicsNeMo `FullyConnected` MLP surrogate.

```bash
# 1. Extract alpha_D profiles from CFD output
cd src && python run_alpha_d_etl.py \
    etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \
    etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed

# 2. Train MLP
cd src && python train.py --config-name alpha_d_mlp

# 3. Evaluate
cd src && python evaluate.py --config-name alpha_d_mlp \
    eval.checkpoint=../data/models/alpha_d_mlp.mdlus
```

See [Alpha-D Surrogate Tutorial](docs/user/alpha_d_surrogate.md) for the full walkthrough.

## Documentation

### User docs

- [Getting Started (Docker setup, run modes, logs, troubleshooting)](docs/user/getting_started.md)
- [Alpha-D Surrogate Tutorial](docs/user/alpha_d_surrogate.md)

### Developer docs

- [ETL Pipeline Internals](docs/dev/etl_pipeline.md)
- [Dataset API](docs/dev/dataset.md)
- [FNO Training and Evaluation](docs/dev/fno_train_eval.md)
