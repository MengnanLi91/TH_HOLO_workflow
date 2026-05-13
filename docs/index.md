# TH_HOLO_workflow

TH_HOLO_workflow is a PhysicsNeMo-based pipeline that turns
**MOOSE thermal-hydraulics outputs** (Exodus `.e` files plus CSV line
probes) into ML-ready Zarr datasets, then trains and evaluates a family
of surrogate models against them through a single generic training core.

```{mermaid}
flowchart LR
    A["MOOSE outputs<br/>(.e + CSV probes)"]
    B["ETL pipeline<br/>(read · transform · validate)"]
    C["Processed dataset<br/>(*.zarr)"]
    D["MooseDataset<br/>(graph · point_cloud · grid)"]
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

## Quick start

::::{tab-set}

:::{tab-item} Docker Compose
```bash
git submodule update --init physicsnemo-curator physicsnemo
docker compose build etl-dev
docker compose run --rm etl-dev bash -lc \
  'cd src && python run_etl.py'
```
:::

:::{tab-item} Apptainer (HPC)
```bash
git submodule update --init physicsnemo-curator physicsnemo
apptainer build th-holo-cpu.sif docker/physicsnemo-cpu.def
apptainer exec --bind /path/to/project:/path/to/project \
  th-holo-cpu.sif bash -lc 'cd /path/to/project/src && python run_etl.py'
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

# 2. Train (HPO + retrain best, one command)
docker compose run --rm etl bash -lc \
  'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'

# 2b. Or skip HPO and train directly
docker compose run --rm etl bash -lc \
  'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null'

# 3. Evaluate
docker compose run --rm etl bash -lc \
  'cd src && python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

The full walkthrough is in the
[Alpha-D Surrogate Tutorial](user/alpha_d_surrogate.md).

## Where do I start?

- **First time using the repo?** Start with the
  [Getting Started guide](user/getting_started.md) — container setup,
  bind mounts, troubleshooting, log inspection.
- **Trying to reproduce the alpha-D surrogate?** Jump to the
  [Alpha-D Surrogate Tutorial](user/alpha_d_surrogate.md) or the
  [Alpha-D case page](cases/alpha_d.md).
- **Wiring up a new model?** Read the
  [Architecture](architecture.md) page, then the
  [FNO training and evaluation guide](dev/fno_train_eval.md) and the
  [Dataset API](dev/dataset.md).
- **Curious about the layout?** See the
  [Repo Layout Refactor Plan](archive/repo_layout_refactor_plan.md)
  (archived — refactor complete) for the design rationale.

## Site contents

```{toctree}
:maxdepth: 2
:caption: Navigation

architecture
user/index
cases/index
dev/index
reference/index
api/index
archive/index
```
