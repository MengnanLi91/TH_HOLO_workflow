# MULTIFID-TH

[![pytest](https://github.com/MengnanLi91/multifid-th/actions/workflows/pytest.yml/badge.svg)](https://github.com/MengnanLi91/multifid-th/actions/workflows/pytest.yml)
[![docs](https://github.com/MengnanLi91/multifid-th/actions/workflows/docs.yml/badge.svg)](https://mengnanli91.github.io/multifid-th/)

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

## Run the Alpha-D Coupled Study

The current alpha-D workflow is a resumable, TOML-driven study. It performs
panel construction, leakage-safe feature selection, Conv1D profile-model HPO
and training, closure export, MOOSE coupling, and evidence reporting in one
provenance-tracked run. After building the Python and MOOSE Apptainer images,
run it from the repository root. The checked-in study defaults to a processed
Zarr tree at `data/flow_contraction_expansion/parametric_study/processed`; see
the running guide to create it from raw Exodus output or select another tree:

```bash
uv sync
export MULTIFID_PYTHON_IMAGE=/absolute/path/to/multifid-th.sif
export MULTIFID_MOOSE_IMAGE=/absolute/path/to/moose-dev.sif

uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/coupling_study.toml

uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-001
```

Use [Running the Alpha-D Case](docs/user/running_alpha_d.md) for the complete
raw-data, resume, and publication procedure. The
[Alpha-D surrogate component tutorial](docs/user/alpha_d_surrogate.md)
documents the retained direct ETL, MLP, and evaluation commands for focused
development; those commands do not produce the coupled-study evidence report.

## Documentation

📘 **Rendered site:** <https://mengnanli91.github.io/multifid-th/> — auto-deployed from `main` by [.github/workflows/docs.yml](.github/workflows/docs.yml).

The Markdown files below are the same source the site renders from; they
read correctly both on GitHub and on the hosted site.

### User docs

- [Getting Started (Docker setup, run modes, logs, troubleshooting)](docs/user/getting_started.md)
- [Reusable Workflows and New-Case Template](docs/user/running_workflows.md)
- [Alpha-D Surrogate Tutorial](docs/user/alpha_d_surrogate.md)
- [Hyperparameter Optimization](docs/user/hyperparameter_optimization.md)

### Developer docs

- [Case template (copyable PyCaret + HPO starting point)](templates/case/README.md)
- [ETL Pipeline Internals](docs/dev/etl_pipeline.md)
- [Dataset API](docs/dev/dataset.md)
- [FNO Training and Evaluation](docs/dev/fno_train_eval.md)
- [Building the documentation](docs/dev/building_docs.md)

### Build the docs locally

```bash
uv sync --extra docs
uv run --no-sync sphinx-build -E -b html docs docs/_build/html
# Open docs/_build/html/index.html in a browser
```

For live reload while editing: `uv run --no-sync make -C docs livehtml` and browse to
<http://localhost:8000>. See [docs/dev/building_docs.md](docs/dev/building_docs.md)
for strict mode, the Apptainer fallback, and the full target list.
