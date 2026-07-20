# MULTIFID-TH

MULTIFID-TH is a PhysicsNeMo-based pipeline that turns
**MOOSE thermal-hydraulics outputs** (Exodus `.e` files plus CSV line
probes) into ML-ready Zarr datasets, then trains and evaluates a family
of surrogate models against them through a single generic training core.

```{mermaid}
flowchart LR
    A["MOOSE outputs<br/>(.e + CSV probes)"]
    B["ETL pipeline<br/>(read · transform · validate)"]
    C["Processed dataset<br/>(*.zarr)"]
    D["MOOSEDataset<br/>(graph · point_cloud · grid)"]
    A --> B --> C --> D
```

## What it does

- Reads simulation outputs from Exodus `.e` files and CSV line-probe files.
- Normalizes fields and constructs graph and regular-grid representations.
- Writes one compressed `.zarr` store per simulation run.
- Exposes a PyTorch `Dataset` for graph, point-cloud, and grid training,
  plus per-case tabular Zarr for pointwise / profile surrogates.

## Architecture at a glance

```{mermaid}
flowchart LR
    subgraph ETL["Two ETL pipelines"]
        E1["cases/moose_grid<br/>(run_etl.py)"]
        E2["cases/alpha_d<br/>(run_etl.py)"]
    end
    subgraph DATA["Per-case Zarr"]
        Z1["{sim_name}.zarr<br/>mesh · fields · grid · probes"]
        Z2["{case}.zarr<br/>features · targets · metadata"]
    end
    subgraph TRAIN["One generic trainer"]
        TF["train.py / evaluate.py<br/>adapters: grid · graph · pointwise · profile"]
    end
    E1 --> Z1
    E2 --> Z2
    Z1 --> TF
    Z2 --> TF
    TF --> M["FNO · MeshGraphNet · AFNO · Pix2Pix<br/>MLP · Conv1DProfile"]
```

The whole story in one sentence: **two ETL pipelines feed one trainer
through four adapters**. The deeper structure is on the
[Architecture](architecture.md) page.

## Where do I start?

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`rocket` First time
:link: user/getting_started
:link-type: doc

Container setup, bind mounts, troubleshooting, and log inspection.

Open the Getting Started guide.
:::

:::{grid-item-card} {octicon}`play` Alpha-D surrogate
:link: user/alpha_d_surrogate
:link-type: doc

Reproduce the Darcy-resistance MLP end-to-end, ETL through evaluation.

Open the Alpha-D Surrogate tutorial.
:::

:::{grid-item-card} {octicon}`code` New model
:link: architecture
:link-type: doc

How adapters, datasets, and the runner fit together when adding a model.

Open the Architecture page.
:::

:::{grid-item-card} {octicon}`beaker` Alpha-D coupling demo
:link: demo_cases/alpha_d_coupling_physics
:link-type: doc

An end-to-end demonstration of the α_D pipeline and its MOOSE PINSFV
coupling, including resolved-CFD and surrogate validation.

Open the Alpha-D coupling demo.
:::

::::

## Quick start

::::{tab-set}

:::{tab-item} Docker Compose
```bash
git submodule update --init physicsnemo-curator physicsnemo
docker compose build etl-dev
docker compose run --rm etl-dev bash -lc \
  'cd src && python cases/moose_grid/run_etl.py'
```
:::

:::{tab-item} Apptainer (HPC)
```bash
git submodule update --init physicsnemo-curator physicsnemo
apptainer build multifid-th-cpu.sif docker/physicsnemo-cpu.def
apptainer exec --bind /path/to/project:/path/to/project \
  multifid-th-cpu.sif bash -lc 'cd /path/to/project/src && python cases/moose_grid/run_etl.py'
```
:::

::::

The default config is the lid-driven flow at
`src/cases/moose_grid/configs/etl.yaml`, which writes output to
`data/processed/lid-driven/*.zarr`.

## Train an FNO

After ETL generates `*.zarr` stores, train with the generic framework
using the FNO example config at `src/cases/moose_grid/configs/train_fno.yaml`:

```bash
docker compose run --rm etl bash -lc \
  'cd src && python train.py --config-path cases/moose_grid/configs --config-name train_fno'
```

Use `etl-ngc` if you prefer the NGC PhysicsNeMo base image; CLI flags
override YAML values, for example `training.epochs=50`.

## Train the Alpha-D MLP surrogate

The alpha-D workflow extracts Darcy-resistance coefficient profiles from
a parametric study of flow contraction-expansion simulations and trains
a PhysicsNeMo `FullyConnected` MLP:

```bash
# 1. Extract alpha_D profiles from CFD output
docker compose run --rm etl bash -lc \
  'cd src && python cases/alpha_d/run_etl.py'

# 2. PyCaret feature selection (writes selected_features.txt that
#    train_mlp consumes via data.input_columns_file)
docker compose run --rm etl bash -lc \
  'cd src && python cases/alpha_d/run_feature_selection_pycaret.py'

# 3. Train (HPO + retrain best, one command)
docker compose run --rm etl bash -lc \
  'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'

# 3b. Or skip HPO and train directly
docker compose run --rm etl bash -lc \
  'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null'

# 4. Evaluate
docker compose run --rm etl bash -lc \
  'cd src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

The Conv1D path (`--config-name train_conv1d`) hard-codes its input
columns and does not need step 2.

The full walkthrough is in the
[Alpha-D Surrogate Tutorial](user/alpha_d_surrogate.md).

## Site contents

```{toctree}
:maxdepth: 2
:caption: Overview

architecture
```

```{toctree}
:maxdepth: 2
:caption: Guides

user/index
cases/index
```

```{toctree}
:maxdepth: 2
:caption: Demonstrations

demo_cases/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

dev/index
api/index
```
