# Alpha-D surrogate case

This directory packages everything specific to the **α_D surrogate** — a
per-station Darcy resistance coefficient predicted along a pipe
contraction-expansion as a function of (Re, Dr, Lr, z). Two model
variants share the same ETL + feature pipeline:

- **MLP (`train_mlp`)** — pointwise FullyConnected predicting one row at
  a time. HPO over ~10 hyperparameters is enabled by default.
- **Conv1D profile (`train_conv1d`)** — 1D convolutional surrogate that
  consumes the full 50-station profile per case. Its YAML uses the required
  screening-plus-confirmation HPO contract. The study workflow selects one
  nine-feature PyCaret set, then freezes it for HPO and every panel.

For the current reproducible alpha-D result, use
`src/cases/alpha_d/configs/coupling_study.toml` with `multifid-workflow`.
That study selects the Conv1D profile method, constructs held-out panels,
runs HPO once, exports the Forchheimer closure, couples it to MOOSE, and
writes a provenance-tracked evidence report. See
[`docs/user/running_alpha_d.md`](../../../docs/user/running_alpha_d.md) for
the complete run procedure. The commands below are component-level entry
points for focused development and debugging.
The HPO pool excludes the union of every held-out and report panel before
fold construction.

## Layout

```
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
└── README.md          # this file
```

## Component-level development (from `src/`)

### 1. ETL: MOOSE → per-case Zarr

The resumable workflow form, run from the repository root, is:

```bash
uv run multifid-workflow etl \
  --config src/cases/alpha_d/configs/etl_workflow.toml \
  --run-id alpha-d-etl-001 \
  --input-dir /absolute/path/to/parametric_study
```

It writes `data/processed` under
`data/workflows/alpha_d_etl/alpha-d-etl-001/`. The command below remains
useful when directly developing the ETL inside a prepared heavy environment.

```bash
python cases/alpha_d/run_etl.py \
  etl.source.input_dir=../data/flow_contraction_expansion/parametric_study \
  etl.sink.output_dir=../data/flow_contraction_expansion/parametric_study/processed
```

Writes one `{case_name}.zarr` per simulation, each with a 50-station
feature/target matrix plus per-case metadata. See
[`docs/user/alpha_d_surrogate.md`](../../../docs/user/alpha_d_surrogate.md)
for the Zarr layout and feature reference.

### 2. PyCaret feature selection

```bash
python cases/alpha_d/run_feature_selection_pycaret.py
```

Reads the Zarr stores, runs PyCaret regression with the
ALLOWLIST-constrained candidate set, writes `selected_features.txt`.

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

```bash
# MLP with HPO + retrain best  (needs Step 2 output)
python train.py --config-path cases/alpha_d/configs --config-name train_mlp

# MLP, skip HPO  (needs Step 2 output)
python train.py --config-path cases/alpha_d/configs --config-name train_mlp hpo=null

# Conv1D profile model with HPO (uses curated fallback unless given a selector artifact)
python train.py --config-path cases/alpha_d/configs --config-name train_conv1d
```

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

- [`docs/user/alpha_d_surrogate.md`](../../../docs/user/alpha_d_surrogate.md) — full tutorial with feature reference and config knobs.
- [`docs/user/hyperparameter_optimization.md`](../../../docs/user/hyperparameter_optimization.md) — HPO study layout and CLI overrides.
- [`docs/user/case_distribution_analysis.md`](../../../docs/user/case_distribution_analysis.md) — pre-training data audit.
