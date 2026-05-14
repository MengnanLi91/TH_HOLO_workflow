# MULTIFID-TH

[![pytest](https://github.com/MengnanLi91/multifid-th/actions/workflows/pytest.yml/badge.svg)](https://github.com/MengnanLi91/multifid-th/actions/workflows/pytest.yml)

MULTIFID-TH — **multifidelity surrogates for thermal-hydraulics** — is a
PhysicsNeMo-based ETL pipeline that converts MOOSE thermal-hydraulics
outputs (Exodus + CSV probes) into ML-ready Zarr datasets and trains
low-fidelity ML surrogates against the high-fidelity simulations.

## Pipeline overview

```mermaid
flowchart LR
    A["MOOSE outputs (.e + CSV probes)"]
    B["ETL pipeline (read, transform, validate)"]
    C["Processed dataset (*.zarr)"]
    D["Training interface: MOOSEDataset (graph | point_cloud | grid)"]
    A --> B --> C --> D
```

## What It Does

- Reads simulation outputs from Exodus `.e` files and CSV line-probe files.
- Normalizes fields and creates graph and regular-grid representations.
- Writes one compressed `.zarr` store per simulation run.
- Provides a PyTorch dataset interface for graph, point-cloud, and grid training.

## Quick Start

```bash
git submodule update --init physicsnemo-curator physicsnemo
docker compose build etl-dev
docker compose run --rm etl-dev bash -lc 'cd src && python cases/moose_grid/run_etl.py'
```

The default config is the lid-driven flow at `src/cases/moose_grid/configs/etl.yaml`, which writes output to `data/processed/lid-driven/*.zarr`.

You can override values on the command line if needed:

```bash
docker compose run --rm etl-dev bash -lc 'cd src && python cases/moose_grid/run_etl.py etl.processing.num_processes=8'
```

To create a new dataset config, copy `src/cases/moose_grid/configs/etl.yaml` to
`src/cases/moose_grid/configs/<your_config>.yaml`, update the source/sink paths, then run:

```bash
docker compose run --rm etl-dev bash -lc 'cd src && python cases/moose_grid/run_etl.py --config-name <your_config>'
```

## Train an FNO with PhysicsNeMo

After ETL generates `*.zarr` stores, train with the generic framework using
the FNO example config at `src/cases/moose_grid/configs/train_fno.yaml`.

```bash
docker compose build etl
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/moose_grid/configs --config-name train_fno'
```

Use `etl-ngc` instead of `etl` if you prefer the NGC PhysicsNeMo base image.
Override config values directly on the CLI, for example:

```bash
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/moose_grid/configs --config-name train_fno training.epochs=50'
```

## Evaluate an FNO Checkpoint

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-path cases/moose_grid/configs --config-name train_fno'
```

To save ground-truth vs predicted velocity-field plots during evaluation:

```bash
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-path cases/moose_grid/configs --config-name train_fno \
  output.plot_dir=../data/models/lid_driven_fno_plots'
```

## Train an MLP Surrogate for Darcy Resistance

The alpha-D workflow extracts Darcy resistance coefficient profiles from a
parametric study of flow contraction-expansion simulations, then trains a
PhysicsNeMo `FullyConnected` MLP surrogate.

```bash
# 1. Extract alpha_D profiles from CFD output
docker compose run --rm etl bash -lc 'cd src && python cases/alpha_d/run_etl.py'

# 2. Train (HPO + retrain best, all in one command)
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'

# 2b. Or skip HPO and train directly
docker compose run --rm etl bash -lc 'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null'

# 3. Evaluate
docker compose run --rm etl bash -lc 'cd src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

See [Alpha-D Surrogate Tutorial](docs/user/alpha_d_surrogate.md) for the full walkthrough.

## Documentation

### User docs

- [Getting Started (Docker setup, run modes, logs, troubleshooting)](docs/user/getting_started.md)
- [Alpha-D Surrogate Tutorial](docs/user/alpha_d_surrogate.md)
- [Hyperparameter Optimization](docs/user/hyperparameter_optimization.md)

### Developer docs

- [ETL Pipeline Internals](docs/dev/etl_pipeline.md)
- [Dataset API](docs/dev/dataset.md)
- [FNO Training and Evaluation](docs/dev/fno_train_eval.md)
- [Building the documentation](docs/dev/building_docs.md)

### Build the docs locally

```bash
pip install -e ".[docs]"
make -C docs html
# Open docs/_build/html/index.html in a browser
```

For live reload while editing: `make -C docs livehtml` and browse to
<http://localhost:8000>. See [docs/dev/building_docs.md](docs/dev/building_docs.md)
for strict mode, the Apptainer fallback, and the full target list.
