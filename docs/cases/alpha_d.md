# Alpha-D Case

The α_D surrogate predicts a per-station Darcy resistance coefficient along
a pipe contraction-expansion as a function of `(Re, Dr, Lr, z)`. The tracked
study selects the Conv1D profile method and couples its exported closure back
to MOOSE. Two model variants remain available for component development:

- **MLP (`train_mlp`)** — pointwise `FullyConnected` predicting one row
  at a time. HPO over ~10 hyperparameters is enabled by default.
- **Conv1D profile (`train_conv1d`)** — 1D convolutional surrogate that
  consumes the full 50-station profile per case. Its YAML uses the required
  screening-plus-confirmation HPO contract. The study workflow selects one
  nine-feature PyCaret set, then freezes it for HPO and every panel.

## Layout

```{mermaid}
flowchart LR
    R["cases/alpha_d/<br/>README.md"] --> Cfg["configs/<br/>train_mlp · train_conv1d · etl · pycaret"]
    R --> DS["datasets/<br/>profile.py (AlphaDProfileDataset)"]
    R --> ETL["etl/<br/>source · transform · sink"]
    R --> Phys["physics/<br/>baseline · targets"]
    R --> Exp["experiment.py<br/>profile loss · HPO metrics · decode hooks"]
    R --> FD["feature_data.py<br/>ALLOWLIST · engineered features"]
    R --> Met["metrics.py<br/>per-region MSE/RMSE · Δp/HPO metrics"]
    R --> Tr["transforms.py<br/>signed-log1p residual target"]
    R --> Run["study_workflow.py<br/>run_etl.py · train.py"]
```

Tree form (matches the on-disk listing):

```text
cases/alpha_d/
├── configs/           # Hydra YAMLs (train_mlp, train_conv1d, etl, pycaret)
├── datasets/          # AlphaDProfileDataset + build_dataset entry point
├── etl/               # PhysicsNeMo Curator pipeline (source, transform, sink)
├── physics/           # baseline, targets — alpha_D encoding + analytical baseline
├── experiment.py      # AlphaDExperiment — profile loss, HPO metrics, decode hooks
├── feature_data.py    # ALLOWLIST, GROUPED_FEATURES, engineered_features_spec
├── metrics.py         # extended metrics (per-region MSE/RMSE, Δp evaluation)
├── transforms.py      # alpha_d_residual_transform (target = signed-log1p residual)
├── run_etl.py         # ETL entry point
├── train.py           # discoverability wrapper around the shared trainer
└── README.md          # source-of-truth, also rendered here
```

## Current reproducible study

Run the coupled study from the repository root when you need a complete,
resumable result rather than one isolated model artifact:

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

The workflow constructs the configured held-out panels, selects one
outer-case-safe Conv1D feature set, tunes the selected method once, trains and
exports each panel, runs MOOSE coupling, and writes the report under
`data/workflows/alpha_d_coupling/<run-id>/`. See
[Running the Alpha-D Case](../user/running_alpha_d.md) for raw-data ETL,
resume, and publication.

## Component-level development (from `src/`)

The commands below are useful for changing or debugging one ETL, feature
selection, training, or evaluation component. They do not substitute for the
coupled-study workflow above and do not create its provenance or report.

### 1. ETL — MOOSE to per-case Zarr

```bash
python cases/alpha_d/run_etl.py \
  etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \
  etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed
```

Writes one `{case_name}.zarr` per simulation, each with a 50-station
feature/target matrix plus per-case metadata. See the
[Alpha-D Surrogate Tutorial](../user/alpha_d_surrogate.md) for the Zarr
layout and feature reference.

### 2. PyCaret feature selection

```bash
python cases/alpha_d/run_feature_selection_pycaret.py
```

Reads the Zarr stores, runs PyCaret regression with the
`ALLOWLIST`-constrained candidate set, and writes
`selected_features.txt`.

- **MLP** (`train_mlp.yaml`) pulls its input columns from
  `data.input_columns_file: …/selected_features.txt`, so this step
  must run first (or you must override `data.input_columns=[…]` and
  set `data.input_columns_file=null` from the CLI).
- The coupled **Conv1D** workflow runs its own `select_alpha_features` stage
  after outer panel cases have been excluded. It writes
  `features/alpha/selected_features.txt`, then passes that frozen file to HPO
  and every panel. Do not run a separate selector before the full workflow.
- `train_conv1d.yaml` retains its curated nine columns only as a fallback for
  direct component runs. To use PyCaret in such a run, select with
  `--config-name pycaret_conv1d` and pass the resulting file as
  `data.input_columns_file=…/selected_features.txt`.

#### Choosing Conv1D candidate features

The case author sets the candidate pool in `configs/pycaret_conv1d.yaml`.
`data.selected_from_allowlist: null` considers every feature in
`cases.alpha_d.feature_data.ALLOWLIST`; replacing `null` with a list restricts
PyCaret to that ordered subset. For example:

```yaml
data:
  selected_from_allowlist:
    - Dr
    - Lr
    - log10_Re_throat
    - z_hat
    - z_hat_times_Dr
    - z_hat_times_Lr
    - dist_to_throat_start
    - dist_to_throat_end
    - dist_to_nearest_step
```

To add a new candidate, make it available from the ETL or engineered-feature
builder and add it to `ALLOWLIST` in `feature_data.py`; an unknown YAML name
is rejected. The workflow supplies only the data path, outer-case exclusions,
and output path.

### 3. Train

::::{tab-set}

:::{tab-item} MLP (with HPO)
Needs Step 2 output.
```bash
python train.py --config-path cases/alpha_d/configs --config-name train_mlp
```
:::

:::{tab-item} MLP (skip HPO)
Needs Step 2 output.
```bash
python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null
```
:::

:::{tab-item} Conv1D profile (with HPO)
The coupled workflow performs Step 2 automatically. A direct run uses the
curated fallback unless given a selector artifact.
```bash
python train.py --config-path cases/alpha_d/configs --config-name train_conv1d
```
:::

::::

A discoverability wrapper exists for the MLP path. It defaults to
`--config-name train_mlp` for this case but is otherwise equivalent to
the top-level `train.py` — both honour an `hpo` block in the config:

```bash
python cases/alpha_d/train.py                       # MLP with HPO (default config)
python cases/alpha_d/train.py hpo=null              # MLP, skip HPO
python cases/alpha_d/train.py --config-name train_conv1d   # Conv1D
```

### 4. Evaluate

```bash
python evaluate.py --config-path cases/alpha_d/configs --config-name train_mlp
```

Schema-3 `run_meta.json` reconstructs the exact dataset, split,
`target_transform_kwargs`, and acceleration-head choice. Evaluation and export
reject older metadata instead of inferring baseline behavior.

## Further reading

- [Alpha-D Surrogate Tutorial](../user/alpha_d_surrogate.md) — full
  walkthrough with feature reference and config knobs.
- [Hyperparameter Optimization](../user/hyperparameter_optimization.md)
  — HPO study layout and CLI overrides.
- [Case Distribution Analysis](../user/case_distribution_analysis.md) —
  pre-training data audit.
