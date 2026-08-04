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
    subgraph TRAIN["Training and study execution"]
        TF["train.py / evaluate.py<br/>adapters: grid · graph · pointwise · profile"]
        WF["multifid-workflow<br/>plan · run · status · publish"]
    end
    E1 --> Z1
    E2 --> Z2
    Z1 --> TF
    Z2 --> TF
    TF --> M["FNO · MeshGraphNet · AFNO · Pix2Pix<br/>MLP · Conv1DProfile"]
    WF --> TF
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
:link: user/running_alpha_d
:link-type: doc

Run the tracked alpha-D study from ETL through MOOSE coupling and reporting.

Open the Alpha-D running guide.
:::

:::{grid-item-card} {octicon}`code` New model
:link: dev/workflows
:link-type: doc

Implement a registered or custom model, adapter, experiment, or exporter.

Open the workflow extension guide.
:::

:::{grid-item-card} {octicon}`workflow` Reproducible workflow
:link: user/running_workflows
:link-type: doc

Select an ML method, plan a study, resume it, and publish checked artifacts.

Open the workflow user guide.
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

## Run the Alpha-D coupled study

The current alpha-D path is the provenance-tracked workflow, which runs the
default Conv1D profile method across the configured panels, exports the
Forchheimer closure, executes MOOSE, and writes the evidence report:

```bash
uv sync
export MULTIFID_PYTHON_IMAGE=/absolute/path/to/multifid-th.sif
export MULTIFID_MOOSE_IMAGE=/absolute/path/to/moose-dev.sif

uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/coupling_study.toml
uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-002
```

See [Running the Alpha-D Case](user/running_alpha_d.md) for ETL, resume, and
publication. The [Alpha-D surrogate component tutorial](user/alpha_d_surrogate.md)
retains direct ETL, MLP, and evaluation commands for focused development.

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
