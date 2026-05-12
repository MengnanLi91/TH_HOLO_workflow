# Repo Layout Refactor — Per-Case Folders + Model-Agnostic Core

Status: planned, not yet implemented.
Author: design discussion, 2026-05-07.
Revision history:
- v3 (2026-05-09) — re-grounded against `HEAD` (commit `0dec187 Add 1D-conv α_D
  profile surrogate`). Folded in `ProfileAdapter` / `AlphaDProfileDataset` /
  `AlphaDConv1D` as a fifth coupling surface. Un-dismissed pickled-checkpoint
  risk (now real because `AlphaDConv1D(physicsnemo.Module)` is our own subclass
  whose `cls.__module__` and `cls.__name__` are embedded by PhysicsNeMo).
  Dropped `eval_delta_p_*.py` from the migration table — those scripts no
  longer exist on disk (their JSON outputs in `data/cases/alpha_d_mlp/` are the
  only remnant). Replaced "narrow vs infrastructure" coupling counts with
  narrow-only (the infra regex was unstable across refactors). Resolved all
  five open questions into concrete decisions. Committed Phase 1.5 to
  "dataset becomes allowlist-agnostic". Committed Phase 3 to per-case
  `hydra.searchpath` rather than a `--case` arg or wrapper scripts. Pulled
  three follow-ups into scope: `moose_etl` → `cases/moose_grid/` (Phase 6),
  `AlphaDConv1D` → `Conv1DProfile` rename + checkpoint-migration utility +
  small design polish (Phase 7), and `pyproject.toml` package-isation with
  setuptools + src layout (Phase 8).
- v2 (2026-05-07) — incorporated plan-and-impl-reviewer findings (external
  importers, fourth coupling surface, recomputed leakage counts,
  `feature_analysis/` is not actually shared, pickled-checkpoint risk softened,
  `case_pressure_drop` Phase 0 scope corrected).
- v1 (2026-05-07) — initial draft.

## Why this matters

A new user wanting to "do alpha-D" today must touch **17 files across 7
directories** to understand a single end-to-end run (alpha-D MLP + the new
1D-conv profile surrogate):

```
src/run_alpha_d_etl.py                         # ETL entry
src/alpha_d_etl/{config/alpha_d_etl.yaml,
                 transform.py, source.py, sink.py}    # ETL implementation
src/run_feature_analysis.py                    # PyCaret entry (sklearn-side)
src/run_feature_selection_pycaret.py           # PyCaret entry (PyCaret-side)
src/feature_analysis/{config/pycaret_feature_analysis.yaml,
                      pycaret_selection.py, data_loader.py,
                      methods.py, manifest.py, plotting.py,
                      __init__.py}                    # PyCaret implementation
src/config/{alpha_d_mlp.yaml, alpha_d_conv1d.yaml}  # Training configs
src/train.py, src/evaluate.py                  # Training entry
src/training/experiments/alpha_d.py            # Throat / Δp experiment
src/training/alpha_d_targets.py                # Encoding helpers
src/training/alpha_d_baseline.py               # Closed-form physics
src/training/datasets_profile.py               # AlphaDProfileDataset (new in 0dec187)
src/training/models/conv1d_profile.py          # AlphaDConv1D (new in 0dec187)
src/training/runner.py                         # extended metrics inline
src/training/datasets_tabular.py               # residual baseline + allowlist inline
src/training/plotting.py                       # alpha-D decode inline
src/training/adapters.py                       # ProfileAdapter (alpha-D-coupled, new in 0dec187)
docs/user/alpha_d_surrogate.md                 # Docs
```

Two diagnostics that v2 of this plan listed (`src/eval_delta_p_baseline.py`,
`src/eval_delta_p_closed_form.py`) **no longer exist** in `HEAD` and have no
git history — only their JSON outputs in `data/cases/alpha_d_mlp/` remain.
v3 drops them from the migration table.

Five structural problems:

1. **Case-specific code lives in supposedly-generic modules.** The
   training core imports alpha-D physics directly. Reproducible counts
   (regex `alpha_d|signed_log1p`, verified at commit `0dec187`):
   - `src/training/runner.py`: 13 hits.
   - `src/training/datasets_tabular.py`: 12 hits.
   - `src/training/plotting.py`: 3 hits.
   - `src/training/adapters.py`: hits via `AlphaDProfileDataset` import (1
     direct import + 5 alpha-D-flavoured kwargs in `ProfileAdapter`).
   - `src/training/datasets_profile.py`: 8 alpha-D-flavoured property
     references that mirror `TabularPairDataset` internals.
   - `src/training/models/conv1d_profile.py`: class name `AlphaDConv1D` (the
     *architecture* is generic — residual dilated 1D conv stack — only the
     class name is case-flavoured).

   v2's "infrastructure" count (a wider regex) drifted unevenly across the
   2026-05-07 → 2026-05-09 refactors (37→22 in `runner.py`, 28→33 in
   `datasets_tabular.py`, 7→4 in `plotting.py`) and is removed here. The
   narrow regex above is the durable one and what the acceptance criteria
   in §"Acceptance criteria" use.

   Beyond the training core, two non-test files import
   `training.alpha_d_targets` directly:
   - `src/feature_analysis/data_loader.py:28`
   - `src/alpha_d_etl/transform.py:26`

   And five test imports across two files:
   - `tests/training/test_alpha_d_baseline.py:8, :125`
   - `tests/training/test_alpha_d_targets.py:16, :35, :62`

   Plus a third test file added in `0dec187`:
   - `tests/training/test_conv1d_profile.py` (352 lines).

2. **`src/feature_analysis/` is not actually a shared library.** Its own
   `__init__.py:1` says "Feature analysis and selection component for the
   alpha_D surrogate"; `BASE_ALLOWLIST` and `ENGINEERED_FEATURES` in
   `data_loader.py` are alpha-D feature names. There is a circular
   case-coupling chain:
   `training/datasets_tabular.py:21` ← `feature_analysis/data_loader.py:28`
   ← `training/alpha_d_targets.py`. The original plan mislabelled this
   directory as "SHARED LIB (cross-case)"; the truth is it's alpha-D
   plumbing that only happens to live at top level.

   `src/feature_analysis/__init__.py:9-22` re-exports `ALLOWLIST`,
   `GROUPED_FEATURES`, `FeatureAnalysisData`, `load_feature_matrix`,
   `build_manifest`, `write_manifest`, `build_dataframe`, `case_level_split`,
   `enforce_allowlist`, `run_pycaret_selection`, `write_selected_features`.
   External callers (`src/run_feature_analysis.py:31`) use this surface.
   The Phase 1 split must keep these names importable from
   `feature_analysis` until Phase 4.

3. **Entry points are at top-level `src/`** even when they only apply to
   one case (`run_alpha_d_etl.py`, `run_case_pressure_drop.py`,
   `run_feature_analysis.py`, `run_feature_selection_pycaret.py`). New
   users cannot tell what belongs to what.

4. **`case_pressure_drop/` is partially-organised.** Its source modules
   live at `src/case_pressure_drop/` (good), but its Hydra config is at
   `src/config/case_pressure_drop.yaml` (separate) and its entry points
   `src/run_case_pressure_drop.py` / `src/evaluate_case_pressure_drop.py`
   are at `src/` top level with hardcoded `config_path="config"`
   (`src/run_case_pressure_drop.py:14`). Finishing the migration requires
   moving the YAML and changing the Hydra `config_path`, which is the same
   Hydra-relocation work needed for alpha-D.

5. **The 1D-conv landing (`0dec187`) added a new coupling surface that
   v2's hook design does not cover.** `ProfileAdapter`
   (`src/training/adapters.py:313-421`) imports `AlphaDProfileDataset`
   directly (`adapters.py:328`) and accepts five alpha-D-flavoured kwargs:
   `throat_weight`, `downstream_weight`, `local_velocity_normalization`,
   `min_Dr`, `target_residual_baseline`. `AlphaDProfileDataset`
   (`src/training/datasets_profile.py`) wraps a `TabularPairDataset` and
   exposes alpha-D-shaped properties (`_raw_z_hat`,
   `_raw_d_local_over_D`, `target_residual_baseline`,
   `local_velocity_normalization`). `AlphaDConv1D`
   (`src/training/models/conv1d_profile.py:52`) is named alpha-D but the
   *architecture* (residual dilated 1D-conv stack) is generic. The
   existing four-hook design in v2 does not generalise this; v3 adds a
   fifth hook (§"Hard part" §5).

## Goals

1. **Discoverability.** Each case has one root folder containing every
   case-specific file. A new user can copy `cases/alpha_d/` to
   `cases/<new_case>/`, change a few names, and have a starting point.
2. **Model-agnostic core.** `src/training/` contains zero references to
   any specific case (no `alpha_d_*`, no `signed_log1p`, no
   `target_residual_baseline`, no `AlphaD*` class names). The core knows
   about adapters, losses, experiments, models, HPO, splits — that's it.
3. **Truthful labelling of cross-case utilities.** Anything labelled
   "shared" must actually be case-agnostic. Today's
   `src/feature_analysis/` does not pass this test and is split: generic
   methods stay shared, case-specific data loaders move into the case.
4. **Data folder unchanged.** `data/cases/<case>/` and
   `data/<dataset>/` continue to be separate from `src/`. The refactor
   only touches code organisation.
5. **Tests stay green at every commit.** No big-bang rewrite — each
   migration phase is independently mergeable.
6. **`src/` becomes a real installable package** (Phase 8) so the
   `tests/conftest.py` `sys.path` hack disappears, Docker images do
   `pip install -e .`, and external callers get clean import errors when
   a path is wrong.

## Non-goals

- Not adding new functionality. This is purely structural cleanup.
- Not changing the Hydra philosophy (still `defaults:` chains, still
  `--config-name` selection).
- Not consolidating ETL pipelines (`alpha_d_etl` and `moose_etl` stay
  as separate Curator pipelines — they handle different raw inputs).
  The `moose_etl` *folder* still moves to `cases/moose_grid/etl/` in
  Phase 6, but the ETL semantics don't change.

## Target structure

```
src/
├── train.py                          # generic Hydra entry — unchanged
├── evaluate.py                       # generic Hydra entry — unchanged
├── run_etl.py                        # generic Curator launcher (already exists)
│
├── training/                         # MODEL-AGNOSTIC CORE
│   ├── runner.py                       # delegates case logic via experiment hooks
│   ├── adapters.py                     # grid, graph, pointwise, profile (dataset-entrypoint-driven)
│   ├── experiment.py                   # base Experiment with no-op hooks
│   ├── losses.py
│   ├── plotting.py                     # generic helpers; takes decode_fn from experiment
│   ├── split_io.py
│   ├── hpo/
│   ├── models/
│   │   ├── mlp.py, fno.py, meshgraphnet.py, afno.py, pix2pix.py
│   │   └── conv1d_profile.py             # was AlphaDConv1D, renamed Conv1DProfile (Phase 7)
│   ├── datasets/                         # split from today's datasets.py + datasets_tabular.py
│   │   ├── grid.py                       # GridPairDataset
│   │   ├── graph.py                      # GraphPairDataset
│   │   └── tabular.py                    # TabularPairDataset, takes target_transform hook
│   └── config/
│       └── default.yaml                  # shared base, was src/config/default.yaml
│
├── feature_selection/                # GENUINELY GENERIC METHODS LIBRARY
│   ├── methods.py                      # was feature_analysis/methods.py (sklearn)
│   ├── pycaret_selection.py            # was feature_analysis/pycaret_selection.py (PyCaret core)
│   ├── manifest.py                     # was feature_analysis/manifest.py
│   └── plotting.py                     # was feature_analysis/plotting.py
│   # No data_loader here — see cases/alpha_d/feature_data.py below.
│
└── cases/                            # ALL case-specific code
    ├── alpha_d/
    │   ├── README.md                       # how to run this case end-to-end
    │   ├── etl/                            # was src/alpha_d_etl/
    │   ├── physics/
    │   │   ├── targets.py                  # was training/alpha_d_targets.py
    │   │   └── baseline.py                 # was training/alpha_d_baseline.py
    │   ├── experiment.py                   # was training/experiments/alpha_d.py
    │   ├── metrics.py                      # extracted from runner.py
    │   ├── plotting.py                     # extracted from training/plotting.py
    │   ├── transforms.py                   # alpha_d_residual_transform (target_transform hook)
    │   ├── feature_data.py                 # was feature_analysis/data_loader.py (case-specific)
    │   ├── feature_selection.py            # case-specific PyCaret config wiring
    │   ├── datasets/
    │   │   └── profile.py                  # was training/datasets_profile.py (AlphaDProfileDataset)
    │   ├── run_etl.py                      # was src/run_alpha_d_etl.py
    │   └── configs/
    │       ├── etl.yaml
    │       ├── train_mlp.yaml              # was src/config/alpha_d_mlp.yaml
    │       ├── train_conv1d.yaml           # was src/config/alpha_d_conv1d.yaml
    │       └── pycaret.yaml
    │
    ├── case_pressure_drop/             # was src/case_pressure_drop/ + entries
    │   ├── data.py
    │   ├── distribution.py
    │   ├── modeling.py
    │   ├── feature_selection.py            # case-specific log1p(delta_p_case) strategy
    │   ├── plotting.py
    │   ├── workflow.py
    │   ├── pycaret_selection.py            # KEEP — different strategy from alpha_d's
    │   ├── run_case_pressure_drop.py       # was src/run_case_pressure_drop.py
    │   ├── evaluate_case_pressure_drop.py  # was src/evaluate_case_pressure_drop.py
    │   └── configs/
    │       └── case_pressure_drop.yaml     # was src/config/case_pressure_drop.yaml
    │
    └── moose_grid/                     # Phase 6 — was src/moose_etl/ + src/config/{fno,…}.yaml
        ├── etl/                            # was src/moose_etl/{data_sources,transformations,…}
        ├── run_etl.py                      # was src/run_etl.py (or thin wrapper)
        └── configs/
            ├── etl.yaml                    # was src/moose_etl/config/lid_driven.yaml
            └── train_fno.yaml              # was src/config/fno.yaml

pyproject.toml                            # Phase 8 — setuptools + src layout
```

Three key shape changes from v2 of this plan:

- **`feature_analysis/` is split into two locations.** Generic methods
  (`methods.py`, `pycaret_selection.py` core, `manifest.py`, `plotting.py`)
  move to `src/feature_selection/`. The alpha-D-coupled
  `data_loader.py` moves to `cases/alpha_d/feature_data.py`. Confronts the
  reality that today's `feature_analysis/` is not actually shared.
- **`case_pressure_drop/pycaret_selection.py` stays.** v1 proposed merging
  it into the shared lib; the reviewer correctly flagged that it
  implements a different strategy (case-granularity `log1p(delta_p_case)`
  with `CasePressureDropDataset`) than the alpha-D one (row-level with
  `GroupKFold(fold_groups='case_id')`). They are different selection
  strategies, not divergent forks of one strategy. Keep both.
- **The 1D-conv stack splits across the boundary** (new in v3). The
  `ProfileAdapter` stays in the generic core but becomes
  dataset-entrypoint-driven (`adapter` no longer imports a case dataset by
  name). The `AlphaDConv1D` model is *renamed* to `Conv1DProfile` and
  stays in `training/models/` because the architecture is generic; a
  backward-compat alias preserves checkpoint round-trips during migration
  (Phase 7). The `AlphaDProfileDataset` is genuinely case-specific (it
  reads alpha-D ETL keys like `_raw_z_hat`, `_raw_d_local_over_D`) and
  moves to `cases/alpha_d/datasets/profile.py`.

## Hard part: extracting case hooks from the core

Five abstraction boundaries (v2 had four; the 1D-conv landing surfaced the
fifth).

### 1. Extended evaluation metrics (`runner.py:298-530`)

Today `_compute_pointwise_extended_metrics` and `_compute_delta_p_metrics`
are inlined in the runner with `is_alpha_d_target()` checks and direct
`field_values_to_physical()` calls. Move to:

```python
# training/experiment.py — base
class Experiment:
    def compute_extended_metrics(
        self,
        eval_dataset,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """Per-experiment extended metrics. Default: empty."""
        return {}

# cases/alpha_d/experiment.py — override
class AlphaDExperiment(Experiment):
    def compute_extended_metrics(self, eval_dataset, preds, targets):
        from cases.alpha_d.metrics import compute_alpha_d_metrics
        return compute_alpha_d_metrics(eval_dataset, preds, targets, ...)
```

The runner calls `experiment.compute_extended_metrics(...)` and writes
the dict into `eval_metrics.json["extended"]`. No knowledge of α_D, Δp,
or signed_log1p anywhere in `runner.py`.

### 2. Tabular dataset target transform (`datasets_tabular.py:228-269`)

The `target_residual_baseline` flag triggers a hardcoded import of
`alpha_d_baseline`. Generalise to an optional callable provided by the
case:

```python
# training/datasets/tabular.py — generic
class TabularPairDataset(Dataset):
    def __init__(self, ..., target_transform=None):
        ...
        if target_transform is not None:
            full_y, baseline_encoded = target_transform(
                full_y, full_x, feature_names, target_names,
                case_meta_list, rows_per_case,
            )
            self._baseline_encoded = baseline_encoded
        ...
```

The case provides the transform via the YAML:

```yaml
data:
  target_transform: cases.alpha_d.transforms:alpha_d_residual_transform
```

(`importlib`-style entrypoint, same pattern as `model.entrypoint`.) The
adapter reads `target_transform`, splits on `:`, calls
`importlib.import_module(...)` and `getattr(...)`.

### 3. Profile plotting (`training/plotting.py:143-258`)

`save_pointwise_profile_plots` decodes via
`field_values_to_physical()`, which is alpha-D-flavoured. Generalise:

```python
# training/plotting.py — generic
def save_pointwise_profile_plots(model, dataset, ..., decode_fn=None):
    """If decode_fn is None, plot raw model outputs."""
    ...
    if decode_fn is not None:
        pred_phys = decode_fn(pred_case[order], dataset, ...)
        target_phys = decode_fn(target_case[order], dataset, ...)
    else:
        pred_phys = pred_case[order]
        target_phys = target_case[order]
```

The runner passes `decode_fn=experiment.decode_for_plotting`; base
returns `None`, alpha-D returns the existing decode + baseline addback.
The physical-RMSE-in-subtitle code lives in the case's decode_fn.

### 4. Training-time experiment lifecycle

The fourth coupling surface lives in `train()`, not `evaluate()`. About
50 lines of alpha-D-specific orchestration in `runner.py`:

- `_build_case_geometry()` at `runner.py:645-690`. Reads
  `delta_p_case`, `Lr`, `D_big`, `outer_height_m`, `buffer_diams`, `rho`,
  `V_bulk` from `cm` (case metadata). All alpha-D ETL keys.
- `delta_p_weight` plumbing at `runner.py:727-729`. Controls whether
  the per-epoch Δp gradient step fires.
- `experiment.alpha_d_target_name = str(dataset.output_columns[0])` at
  `runner.py:740`. Direct attribute injection by name.
- `experiment.case_geometry` and `experiment.val_case_geometry` setup
  at `runner.py:790-797`. Side-channels that bypass the experiment's
  own state.

The hook to add:

```python
# training/experiment.py — base
class Experiment:
    def prepare_for_training(
        self,
        train_dataset,
        val_dataset,
        device: torch.device,
    ) -> None:
        """Per-experiment training-time state setup. Default: no-op."""

# cases/alpha_d/experiment.py — override
class AlphaDExperiment(Experiment):
    def prepare_for_training(self, train_dataset, val_dataset, device):
        # Pull `alpha_d_target_name`, build `case_geometry` and
        # `val_case_geometry`, populate own state.
        ...
```

The runner calls `experiment.prepare_for_training(train_ds, val_ds,
device)` once after experiment construction and stops touching
experiment internals by name. `_build_case_geometry` itself moves to
`cases/alpha_d/experiment.py` as a private method.

`delta_p_weight` is already config-driven in YAML and read by the
experiment constructor; its `runner.py` reference at line 727 just
checks whether to call `compute_delta_p_loss_step` each epoch — that
becomes a generic per-epoch hook on the base `Experiment`:

```python
# training/experiment.py — base
def on_epoch_end_extra_step(self) -> None:
    """Optional extra gradient step per epoch. Default: no-op."""
```

`AlphaDExperiment` overrides this to run `compute_delta_p_loss_step()`.

### 5. Profile adapter dataset binding (NEW in v3)

`ProfileAdapter` in `src/training/adapters.py:313-421` (especially
`build_dataset` at lines 327-373) imports `AlphaDProfileDataset`
directly:

```python
class ProfileAdapter(ModelAdapter):
    def build_dataset(self, data_cfg: dict):
        from training.datasets_profile import AlphaDProfileDataset  # ← case import
        ...
        return AlphaDProfileDataset(
            zarr_dir=data_cfg["zarr_dir"],
            ...
            throat_weight=_opt_float("throat_weight"),
            downstream_weight=_opt_float("downstream_weight"),
            local_velocity_normalization=bool(
                data_cfg.get("local_velocity_normalization", False)
            ),
            min_Dr=_opt_float("min_Dr"),
            target_residual_baseline=bool(
                data_cfg.get("target_residual_baseline", False)
            ),
        )
```

The grid/graph/pointwise adapters do not have this problem because their
datasets (`GridPairDataset`, `GraphPairDataset`, `TabularPairDataset`)
are themselves case-agnostic. `AlphaDProfileDataset` is case-coupled by
construction (it sorts rows by `_raw_z_hat`, an alpha-D ETL key).

The hook: make `ProfileAdapter` dataset-entrypoint-driven, mirroring the
existing `model.entrypoint` pattern.

```python
# training/adapters.py — generic
class ProfileAdapter(ModelAdapter):
    family = "profile"

    def build_dataset(self, data_cfg: dict):
        ep = data_cfg.get("dataset_entrypoint")
        if ep is None:
            raise ValueError(
                "Profile adapter requires data.dataset_entrypoint "
                "(e.g. 'cases.alpha_d.datasets.profile:build_dataset')."
            )
        module_name, fn_name = ep.split(":", 1)
        build = getattr(importlib.import_module(module_name), fn_name)
        return build(data_cfg)
```

```yaml
# cases/alpha_d/configs/train_conv1d.yaml
data:
  dataset_entrypoint: cases.alpha_d.datasets.profile:build_dataset
  # plus the alpha-D-specific kwargs that the case build_dataset reads
  throat_weight: 1.0
  downstream_weight: 1.0
  local_velocity_normalization: true
  min_Dr: 0.333
  target_residual_baseline: true
```

```python
# cases/alpha_d/datasets/profile.py
def build_dataset(data_cfg: dict) -> "AlphaDProfileDataset":
    """Build the case's profile dataset from a Hydra data config."""
    return AlphaDProfileDataset(
        zarr_dir=data_cfg["zarr_dir"],
        throat_weight=_opt_float(data_cfg, "throat_weight"),
        ...
    )
```

Post-Phase-2e, `grep -E "AlphaD|alpha_d" src/training/adapters.py`
returns zero matches.

## Migration path — incremental, behavior-preserving

Each phase is independently mergeable, leaves tests green, and produces
a working system. Each phase has a clear rollback.

### Phase 0 — Establish `src/cases/` and finish case_pressure_drop (2-3 hr)

The cleanest pilot, since `case_pressure_drop/` is already half-organised.

- `mkdir src/cases/`
- `git mv src/case_pressure_drop src/cases/case_pressure_drop`
- `git mv src/run_case_pressure_drop.py src/cases/case_pressure_drop/`
- `git mv src/evaluate_case_pressure_drop.py src/cases/case_pressure_drop/`
- `mkdir src/cases/case_pressure_drop/configs/`
- `git mv src/config/case_pressure_drop.yaml src/cases/case_pressure_drop/configs/`
- Add `src/cases/__init__.py`, `src/cases/case_pressure_drop/configs/__init__.py`
- Update `config_path` argument in the moved entry-point scripts
  (currently hardcoded `"config"` at `src/run_case_pressure_drop.py:14`)
  to the new relative path `"configs"`.
- Update import paths in moved files (`from case_pressure_drop.X` →
  `from cases.case_pressure_drop.X`).
- Update test imports under `tests/case_pressure_drop/`.

**Validates the per-case folder pattern AND the Hydra relocation pattern
on the smaller case before touching alpha-D.**

### Phase 1 — Move alpha-D files with import shims (4-5 hr)

Move files into `src/cases/alpha_d/` but **keep old import paths
working** via re-export modules so nothing breaks during the transition.

```python
# src/training/alpha_d_baseline.py (shim)
from cases.alpha_d.physics.baseline import *  # noqa: F401, F403
import warnings
warnings.warn("training.alpha_d_baseline is deprecated; use "
              "cases.alpha_d.physics.baseline", DeprecationWarning)
```

Files to move + shim (the alpha-D physics moves break `feature_analysis/`
and `datasets_profile.py` simultaneously, so Phase 1 also handles the
`feature_analysis` split and the profile-dataset move; v3 lists 18 rows
where v2 had 14):

| Today                                          | Becomes                                                  |
| ---------------------------------------------- | -------------------------------------------------------- |
| `src/training/alpha_d_baseline.py`             | `src/cases/alpha_d/physics/baseline.py`                  |
| `src/training/alpha_d_targets.py`              | `src/cases/alpha_d/physics/targets.py`                   |
| `src/training/experiments/alpha_d.py`          | `src/cases/alpha_d/experiment.py`                        |
| `src/training/datasets_profile.py`             | `src/cases/alpha_d/datasets/profile.py`                  |
| `src/alpha_d_etl/source.py`                    | `src/cases/alpha_d/etl/source.py`                        |
| `src/alpha_d_etl/transform.py`                 | `src/cases/alpha_d/etl/transform.py`                     |
| `src/alpha_d_etl/sink.py`                      | `src/cases/alpha_d/etl/sink.py`                          |
| `src/alpha_d_etl/config/alpha_d_etl.yaml`      | `src/cases/alpha_d/configs/etl.yaml` (Phase 3)           |
| `src/run_alpha_d_etl.py`                       | `src/cases/alpha_d/run_etl.py`                           |
| `src/feature_analysis/data_loader.py`          | `src/cases/alpha_d/feature_data.py`                      |
| `src/feature_analysis/methods.py`              | `src/feature_selection/methods.py`                       |
| `src/feature_analysis/pycaret_selection.py`    | `src/feature_selection/pycaret_selection.py`             |
| `src/feature_analysis/manifest.py`             | `src/feature_selection/manifest.py`                      |
| `src/feature_analysis/plotting.py`             | `src/feature_selection/plotting.py`                      |
| `src/feature_analysis/__init__.py`             | becomes a thin shim that re-exports from new homes (until Phase 4) |
| `tests/training/test_alpha_d_baseline.py`      | `tests/cases/alpha_d/test_baseline.py`                   |
| `tests/training/test_alpha_d_targets.py`       | `tests/cases/alpha_d/test_targets.py`                    |
| `tests/training/test_conv1d_profile.py`        | `tests/cases/alpha_d/test_profile.py`† and `tests/core/training/test_profile_adapter.py`† |

† Before moving, skim `tests/training/test_conv1d_profile.py` (352
lines) for tests that exercise *generic profile-adapter shape behavior*
(e.g. `[B, C, S]` collation, `accumulate_metrics` shape semantics)
independently of alpha-D physics. Those tests belong in
`tests/core/training/test_profile_adapter.py` so the generic
`ProfileAdapter` keeps test coverage after Phase 2e. Tests that
exercise alpha-D-specific decode paths or `_raw_z_hat` ordering move
into `tests/cases/alpha_d/test_profile.py`. If the split is non-obvious
at execution time, default to keeping everything alpha-D-side and
back-fill the generic suite in Phase 2e.

Also create the `__init__.py` files needed for `pkg://` resolution in
Phase 3:

- `src/cases/__init__.py`
- `src/cases/alpha_d/__init__.py`
- `src/cases/alpha_d/configs/__init__.py`
- `src/cases/alpha_d/etl/__init__.py`
- `src/cases/alpha_d/physics/__init__.py`
- `src/cases/alpha_d/datasets/__init__.py`
- `src/feature_selection/__init__.py`
- `src/training/config/__init__.py`

Without `cases/alpha_d/configs/__init__.py`,
`hydra.searchpath: pkg://cases.alpha_d.configs` (Phase 3) will not
resolve.

Diagnostic scripts in v2's table (`src/eval_delta_p_baseline.py`,
`src/eval_delta_p_closed_form.py`) **are not in the migration list** —
they no longer exist on disk in `HEAD` and have no commits to revert. If
they reappear from an unstaged location at execution time, add a
`cases/alpha_d/diagnostics/` row then.

Add shims at every old path so existing imports keep working until
Phase 4. Tests under `tests/training/test_alpha_d_*.py` continue to use
`from training.alpha_d_*` — these resolve through the shims while the
test files are being relocated.

External importer fixes that were missed in v1:

- `src/feature_analysis/data_loader.py:28` already imports
  `training.alpha_d_targets`. After the move this becomes
  `cases.alpha_d.feature_data` importing
  `cases.alpha_d.physics.targets` — a within-case import, no layering
  violation.
- `src/alpha_d_etl/transform.py:26` imports
  `training.alpha_d_targets`. After the move this becomes
  `cases.alpha_d.etl.transform` importing
  `cases.alpha_d.physics.targets` — within-case, fine.
- `training.datasets_tabular` imports `feature_analysis.data_loader`
  (`datasets_tabular.py:21`). After the move this becomes a
  problem: the generic core would import a case file. Resolution:
  Phase 1.5 below.
- `training.datasets_profile` imports `training.datasets_tabular` (line
  18) and is itself imported by `training.adapters.ProfileAdapter` at
  `adapters.py:328`. After Phase 1 the dataset moves; the adapter still
  imports from its old location via a shim. Phase 2e cuts the adapter →
  case dependency proper.

### Phase 1.5 — Break the `datasets_tabular` ⇄ alpha-D circular import (1 hr)

`datasets_tabular.py:21` imports `BASE_ALLOWLIST` and friends from
`feature_analysis.data_loader`. These are alpha-D feature names.
`datasets_tabular` should not know about them.

**Decision (v3):** the dataset becomes allowlist-agnostic; the case
enforces its own allowlist before passing the feature list in. Concrete
shape:

```python
# Today (datasets_tabular.py:21 + downstream)
from feature_analysis.data_loader import BASE_ALLOWLIST, ENGINEERED_FEATURES
# … input_columns get filtered against BASE_ALLOWLIST internally …

# After Phase 1.5
class TabularPairDataset(Dataset):
    def __init__(self, ..., input_columns: list[str], ...):
        # No allowlist enforcement here. Trust the caller.
        ...

# In the alpha-D config-loading layer (cases/alpha_d/datasets/profile.py
# or its sibling for the MLP path):
from cases.alpha_d.feature_data import BASE_ALLOWLIST, ENGINEERED_FEATURES
input_columns = enforce_allowlist(input_columns, BASE_ALLOWLIST | ENGINEERED_FEATURES)
return TabularPairDataset(input_columns=input_columns, ...)
```

The dataset stops importing case modules; the case enforces its own
contract before constructing the dataset. `enforce_allowlist` already
exists in `feature_analysis/pycaret_selection.py` (`feature_selection/`
post-rename) and stays generic.

**Pre-flight verification.** Before committing Phase 1.5, run:

```bash
grep -rn "feature_analysis\|alpha_d_\|cases\." src/training/datasets_tabular.py
```

Expect: only the imports being removed in this phase. If the grep
surfaces additional case coupling that the v3 audit missed, treat it
as a Phase 1.5 sub-task (extract whichever cycle the grep exposes
using the same allowlist-agnostic-dataset pattern).

After Phase 1.5 lands, the same grep returns zero matches.

Tests pass without modification (shims preserve old paths).

### Phase 2 — Extract case hooks from core (8-10 hr, the surgery)

Five sub-tasks (v2 had three; v3 adds 2d and 2e), each with its own
commit. Order matters: 2a/2b/2c are local; 2d depends on the experiment
move from Phase 1 having shims in place; 2e depends on 2b's
`target_transform` plumbing.

**2a — Extended metrics hook (`compute_extended_metrics`).**
- Add `Experiment.compute_extended_metrics()` no-op base.
- Move `_compute_pointwise_extended_metrics` and
  `_compute_delta_p_metrics` from `runner.py:298-530` to
  `cases/alpha_d/metrics.py`.
- `AlphaDExperiment.compute_extended_metrics` calls the moved
  functions.
- `runner.evaluate()` calls `experiment.compute_extended_metrics()`
  unconditionally, no `is_alpha_d_target` check.

**2b — Tabular target transform.**
- Add `target_transform: Callable | None` parameter to
  `TabularPairDataset`.
- Move residual-baseline logic from `datasets_tabular.py:228-269` to
  `cases/alpha_d/transforms.py::alpha_d_residual_transform`.
- Replace `target_residual_baseline: true` in alpha-D configs with
  `target_transform: cases.alpha_d.transforms:alpha_d_residual_transform`.
  Adapter resolves the entrypoint via `importlib`.

**2c — Plotting decode_fn.**
- Generic `save_pointwise_profile_plots(decode_fn=None)` plots raw
  outputs by default.
- `AlphaDExperiment.decode_for_plotting()` returns the existing alpha-D
  decode.
- Runner passes `decode_fn=experiment.decode_for_plotting` to the
  plotter.

**2d — Training lifecycle hook.**
- Add `Experiment.prepare_for_training(train_ds, val_ds, device)` no-op
  base.
- Move `_build_case_geometry` from `runner.py:645-690` to
  `cases/alpha_d/experiment.py` as a private helper.
- `AlphaDExperiment.prepare_for_training` populates
  `case_geometry`, `val_case_geometry`, `alpha_d_target_name`,
  `local_velocity_normalization` on `self`.
- `runner.train()` calls `experiment.prepare_for_training(...)` once
  after construction; the `delta_p_weight > 0` check at `runner.py:727`,
  the `alpha_d_target_name` injection at `runner.py:740`, and the
  `case_geometry` wiring at `runner.py:790-797` all disappear from
  `runner.py`.
- Add `Experiment.on_epoch_end_extra_step()` no-op base; runner calls
  it once per epoch instead of `if hasattr(experiment,
  "compute_delta_p_loss_step")`. AlphaDExperiment overrides.

**2e — Profile adapter dataset entrypoint (NEW in v3).**
- Add `data.dataset_entrypoint` config key handling to `ProfileAdapter`
  (mirrors `model.entrypoint`).
- Strip the alpha-D-flavoured kwargs (`throat_weight`,
  `downstream_weight`, `local_velocity_normalization`, `min_Dr`,
  `target_residual_baseline`) from `ProfileAdapter.build_dataset` —
  those become opaque pass-through to the case's own `build_dataset(...)`.
- Move construction logic to `cases/alpha_d/datasets/profile.py:build_dataset`
  (reads same kwargs, returns `AlphaDProfileDataset`).
- Update `cases/alpha_d/configs/train_conv1d.yaml` to set
  `data.dataset_entrypoint: cases.alpha_d.datasets.profile:build_dataset`.

After Phase 2, the following acceptance greps return zero matches in
`src/training/`:
- `alpha_d|signed_log1p|delta_p_case|target_residual_baseline`
- `AlphaD\w+`
- `from cases\.|import cases\.`

(Phase 1.5 already broke the `datasets_tabular` ⇄ `feature_analysis.data_loader`
cycle, which was a precondition for the above.)

### Phase 3 — Move configs to case-local locations + Hydra search-path (2 hr)

For alpha-D specifically (case_pressure_drop's configs already moved in
Phase 0).

- `git mv src/config/alpha_d_mlp.yaml src/cases/alpha_d/configs/train_mlp.yaml`
- `git mv src/config/alpha_d_conv1d.yaml src/cases/alpha_d/configs/train_conv1d.yaml`
- `git mv src/alpha_d_etl/config/alpha_d_etl.yaml src/cases/alpha_d/configs/etl.yaml`
- `git mv src/feature_analysis/config/pycaret_feature_analysis.yaml src/cases/alpha_d/configs/pycaret.yaml`
- `git mv src/config/default.yaml src/training/config/default.yaml`.

**Hydra search-path mechanism (decision in v3):** add per-case
`hydra.searchpath` to each case config rather than a `--case=` arg or
wrapper scripts. Two-line per-case change, no `train.py` modification:

```yaml
# src/cases/alpha_d/configs/train_mlp.yaml
defaults:
  - default
  - _self_

hydra:
  searchpath:
    - pkg://training.config        # finds default.yaml
    - pkg://cases.alpha_d.configs  # finds the case's own configs
```

Invocation stays familiar:

```bash
docker compose run --rm etl bash -lc \
  'cd src && python train.py --config-path cases/alpha_d/configs --config-name train_mlp'
```

A thin wrapper script `cases/alpha_d/train.py` (two lines: set
`config_path`, call shared `train`) is added for discoverability per the
recommendation in §"Decisions". This is the only Hydra change to
`train.py` itself: nothing.

### Phase 4 — Drop import shims (30-45 min)

Remove the deprecation shims from Phase 1. By now all internal callers
have been updated to the new paths. Anything still importing the old
paths gets a clean `ImportError` pointing at the new location.

Required green-state assertions:
- `grep -rn "from training.alpha_d_\|import training.alpha_d_" src/ tests/`
  must return zero matches.
- `grep -rn "from feature_analysis\|import feature_analysis" src/ tests/`
  must return zero matches.
- `grep -rn "from training.datasets_profile\|import training.datasets_profile" src/ tests/`
  must return zero matches.

`AlphaDConv1D` retains a backward-compat re-export at its current
class-name path until Phase 7 (the rename + checkpoint migration).

### Phase 5 — Documentation (1 hr)

- New `src/cases/alpha_d/README.md` documents the case end-to-end (ETL,
  feature selection, training, eval).
- Update `docs/user/alpha_d_surrogate.md` to point at the new layout.
- Update `CLAUDE.md` (top-level + `TH_HOLO_workflow/CLAUDE.md`)
  "Common commands" section to use the new `--config-path
  cases/alpha_d/configs --config-name train_*` form.

### Phase 6 — `moose_etl` → `cases/moose_grid/` (2-3 hr)

Mirror Phase 0's mechanics on the second case. Run after the alpha-D
work has stabilised so the per-case-folder pattern is well-trodden.

- `mkdir src/cases/moose_grid/{etl,configs}`
- `git mv src/moose_etl/{data_sources,transformations,…} src/cases/moose_grid/etl/`
- `git mv src/moose_etl/config/lid_driven.yaml src/cases/moose_grid/configs/etl.yaml`
- `git mv src/config/fno.yaml src/cases/moose_grid/configs/train_fno.yaml`
  (and any other grid-model configs that target moose data: `afno.yaml`,
  `meshgraphnet.yaml`, `pix2pix.yaml` — verify each at execution time).
- Decision: `src/run_etl.py` is generic (PhysicsNeMo Curator launcher)
  and stays at top level. The case-specific entry is invoked via
  `python run_etl.py --config-path cases/moose_grid/configs --config-name etl`.
  Add a thin `cases/moose_grid/run_etl.py` wrapper for discoverability,
  matching alpha-D's pattern.
- Add per-case `hydra.searchpath` to each moose_grid config (same
  mechanism as Phase 3).
- Update import paths inside the moved ETL modules
  (`from moose_etl.X` → `from cases.moose_grid.etl.X`).
- Add a Phase 1-style import shim at `src/moose_etl/` for a two-week
  deprecation window, then drop in a Phase-4-style cleanup commit.
  The shim drop is the only thing gating acceptance criterion #6.
- Update tests under `tests/moose_etl/` → `tests/cases/moose_grid/`.

Reuses the Phase 0 + Phase 3 patterns; no new design work. Risk is low
since `moose_etl` does not have the cross-case coupling that alpha-D
had — it's already self-contained.

### Phase 7 — `AlphaDConv1D` → `Conv1DProfile` rename + polish + checkpoint migration (3-4 hr)

The model is named alpha-D but the architecture (residual dilated 1D-conv
stack) is generic. Decoupling completes Goal #2 (model-agnostic core).

PhysicsNeMo's `Module.__new__` records `cls.__module__` and `cls.__name__`
at save time and re-imports via `getattr(module, name)` at load time. Any
saved `.mdlus` for a custom subclass embeds those two strings. Renaming
the class breaks load until either (a) the saved checkpoint is
re-written, or (b) a backward-compat alias of the old class name lives
at the new module path.

**Steps:**

1. Add `class AlphaDConv1D(Conv1DProfile): pass` alias at
   `training/models/conv1d_profile.py` *before* renaming, so the old
   name resolves to the new class. This means existing `.mdlus` load
   correctly and produce a `Conv1DProfile` instance.
2. Rename `AlphaDConv1D` → `Conv1DProfile` in the implementation; the
   alias above keeps the old class name available.
3. Add a small migration utility
   `src/training/models/_migrate_conv1d_checkpoint.py` that takes a
   `.mdlus` path and rewrites the embedded `__class_name__` from
   `AlphaDConv1D` to `Conv1DProfile`. PhysicsNeMo's `.mdlus` is a tar
   with `args.json` inside; the utility extracts, edits, repacks. Single
   shot, ~40 lines.
4. Polish (small design changes per scope decision):
   - Drop the alpha-D-flavoured docstring; replace with a generic
     description of the architecture.
   - Tighten the `try/except ModuleNotFoundError` block around
     `import_physicsnemo_attr("physicsnemo", "Module")`. Today the class
     is conditionally defined; switch to a guard at import time that
     raises a clear error if the module is asked to register without
     PhysicsNeMo present.
   - Accept activation as a config knob (`silu` default, `gelu` as
     option); plumb through `_make_block` and the head.
   - No structural network changes (kernel size, dilation, residual
     pattern stay identical).
5. Add a regression test that loads `data/cases/alpha_d_mlp/`-style
   conv1d checkpoints (if any exist at execution time; otherwise add a
   tiny saved fixture) and verifies the migrated `.mdlus` round-trips.

After Phase 7 the acceptance criterion `AlphaD\w+` returns zero matches
in `src/training/` *except* the documented backward-compat alias, which
is removed in a follow-up release after the migration utility has been
applied to all known checkpoints.

### Phase 8 — `pyproject.toml` package-isation (3-4 hr)

Make `src/` a real installable package; remove the `tests/conftest.py`
`sys.path` hack.

**Build backend:** setuptools, src layout (decision in v3). Lowest churn,
compatible with all four Docker images and the apptainer `.def` files,
no team learning curve.

**Steps:**

1. Author a `pyproject.toml` at the repo root:

   ```toml
   [build-system]
   requires = ["setuptools>=64", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "th-holo-workflow"
   version = "0.1.0"
   description = "Thermal-hydraulic surrogate workflow on MOOSE outputs"
   requires-python = ">=3.11"
   dependencies = []  # runtime deps live in the Dockerfiles for now

   [tool.setuptools.packages.find]
   where = ["src"]
   include = ["training*", "feature_selection*", "cases*"]
   namespaces = false
   ```

   The `include` list above is the **post-Phase-7 state**, which is what
   Phase 8 ships against. By that point Phase 4 has dropped the
   `feature_analysis/` shim and Phase 6 has moved `moose_etl/` into
   `cases/moose_grid/`, so neither needs to appear here. If a Phase 8
   pre-flight finds either of those directories still present (e.g.
   the operator chose to delay the shim removal), extend the list at
   that moment. Do **not** copy this list verbatim during an earlier
   phase — Phase 0 / 1 / 2 do not need a `pyproject.toml`, and the
   first time `pip install -e .` runs is in Phase 8.

2. Update each Dockerfile in `docker/`:
   - `Dockerfile.dev`, `Dockerfile.physicsnemo-cpu`, `Dockerfile.gpu`,
     `Dockerfile.ngc`: add `RUN pip install -e .` after the `COPY` of
     the source tree.
   - Mirror in the `.def` files (`dev.def`, `physicsnemo-cpu.def`,
     `gpu.def`, `ngc.def`) for the apptainer images.

3. Delete `tests/conftest.py`'s `sys.path.insert(0, str(SRC))` block.
   Replace the file with an empty `conftest.py` (or delete entirely if
   no other fixtures live in it — verify at execution time). With the
   package installed in editable mode, `import training`,
   `import cases.alpha_d`, etc. all resolve normally.

4. Verify the CI workflow `.github/workflows/pytest.yml` still passes —
   if it builds the Docker image, the `pip install -e .` is already in
   place; if it sets up Python directly, it needs an explicit
   `pip install -e .` step before `pytest`.

5. Update `CLAUDE.md` to mention `pip install -e .` in the workflow.

This phase has no functional impact on training/eval — it's pure
packaging hygiene. The only risk is a Docker layer cache miss across
rebuilds, which is acceptable.

## Risks

| Risk                                                                                 | Reality                                                                                                                                                                                                                                                                                          | Mitigation                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hydra `defaults:` chain breaks across the move                                       | Real — `default` resolution is search-path-sensitive                                                                                                                                                                                                                                            | Phase 3 adds `hydra.searchpath` per case; Phase 0 validates the same mechanism on the smaller case first; revertable per phase                                                                                                                                                          |
| Pickled checkpoints reference old module paths (`Module.from_checkpoint`)             | **Real now** (un-dismissed in v3). PhysicsNeMo's `Module.__new__` records `cls.__module__` and `cls.__name__` at save time and re-imports via `getattr(module, name)` at load time. `src/training/models/conv1d_profile.py:52` defines `AlphaDConv1D(_PhysicsNeMoModule)` — *our* subclass. The MLP path is still safe (it inherits `physicsnemo.models.mlp.fully_connected:FullyConnected` directly — library path). | Phases 0-6 keep `AlphaDConv1D` at its current class path. Phase 7 (rename) lands an alias `class AlphaDConv1D(Conv1DProfile): pass` at the same module path *and* a checkpoint-migration utility, so existing and new checkpoints both resolve.                                          |
| External notebooks / scripts import `from training.alpha_d_*`                         | Real                                                                                                                                                                                                                                                                                            | Shims (Phase 1) keep them working with deprecation warnings until Phase 4                                                                                                                                                                                                                |
| `case_pressure_drop/pycaret_selection.py` and `feature_analysis/pycaret_selection.py` look like duplicates | **They are different selection strategies, not divergent forks.** `case_pressure_drop`'s targets `log1p(delta_p_case)` at case granularity using `CasePressureDropDataset`; `feature_analysis`'s operates row-level with `GroupKFold`.                                                                                                                | Keep both. The shared `feature_selection/pycaret_selection.py` (post-rename) is the row-level strategy; the case-level strategy stays in `cases/case_pressure_drop/pycaret_selection.py`.                                                                                                |
| CI workflow references old paths                                                     | Likely real — `.github/workflows/pytest.yml` runs `pytest` from repo root                                                                                                                                                                                                                       | Grep `.github/` early, update in Phase 1 if needed; Phase 8 explicitly verifies                                                                                                                                                                                                          |
| Tests under `tests/training/test_alpha_d_*.py` and `test_conv1d_profile.py` mix core and case-specific assertions | Real — five `from training.alpha_d_*` imports across two files, plus `tests/training/test_conv1d_profile.py` (352 lines) imports `training.datasets_profile`                                                                                                                                       | Move all three to `tests/cases/alpha_d/` during Phase 1 alongside the source moves; shims keep old imports working until Phase 4                                                                                                                                                          |
| `feature_analysis/__init__.py` re-exports symbols that callers depend on              | Real — `__init__.py:9-22` re-exports 11 names. `from feature_analysis import build_manifest, load_feature_matrix, write_manifest` is used in `src/run_feature_analysis.py:31`                                                                                                                                                              | The split (Phase 1) keeps a thin `feature_analysis/__init__.py` shim that re-exports from the new locations until Phase 4                                                                                                                                                                |
| `ProfileAdapter` keeps importing `AlphaDProfileDataset` after Phase 1 file moves      | Real — Phase 1 moves the dataset; the adapter still imports its old path through a shim. The shim works but the adapter is still case-coupled in spirit                                                                                                                                                                                                                                | Phase 2e (NEW) replaces the direct import with a `data.dataset_entrypoint` config key; only after 2e is the adapter genuinely generic                                                                                                                                                  |
| `pyproject.toml` editable-install behaves differently than `sys.path` hack             | Low risk. Editable installs see source changes immediately (same as sys.path); the only divergence is import semantics for top-level scripts run as files (`python src/train.py` becomes `python -m train` or stays the same with `cd src && python train.py`).                                                                                              | Phase 8 keeps `cd src && python train.py` invocation form working; the docs in `CLAUDE.md` already use that form.                                                                                                                                                                       |
| `moose_etl` import shim window collides with downstream consumers                      | Low risk — `moose_etl` is consumed only by `src/run_etl.py` and `src/moose_etl/transformations/__init__.py`'s siblings; there are no external importers in the repo today.                                                                                                                                                                                                                | Phase 6 shim mirrors Phase 1's pattern; drop in a follow-up commit after one release.                                                                                                                                                                                                    |

## Decisions (was: open questions in v2)

These each had multiple reasonable answers in v2; v3 commits.

1. **Naming: `cases/`** (not `experiments/`). Matches `data/cases/`,
   unambiguous, no collision with `training/experiments/` (a loss/training-step
   hook concept).

2. **Hydra search-path mechanism: per-case `hydra.searchpath`** (not
   a `--case=` CLI arg, not a wrapper-only solution). Two-line config
   change per case, no `train.py` modification needed. A thin
   `cases/<case>/train.py` wrapper is *also* provided as a discoverability
   shortcut (it just sets `--config-path` and calls the shared
   `train.py`), but is not the load-bearing piece.

3. **PyCaret config lives in the case folder.** `feature_selection/` is
   library-only (no configs); each case's `configs/` directory holds the
   PyCaret YAML it needs.

4. **Migration is incremental**, not big-bang. Eight phases (0 / 1 /
   1.5 / 2 / 3 / 4 / 5 / 6 / 7 / 8), each independently mergeable and
   reversible.

5. **`pyproject.toml` package-isation: in scope.** Phase 8 with
   setuptools + src layout. Removes the `tests/conftest.py` sys.path
   hack and switches Docker images to `pip install -e .`.

6. **Conv1D scope: rename + small design polish**, not architecture
   rework. Phase 7 covers the rename, the checkpoint-migration utility,
   and three small polish items (docstring, import-guard cleanup,
   activation as a config knob). No structural changes to the network.

7. **`moose_etl` move: Phase 6, after alpha-D lands.** Reuses the
   Phase 0 + 3 patterns once the per-case-folder shape is well-trodden.

## Effort estimate (revised v3)

| Phase    | What                                                                          | Time     |
| -------- | ----------------------------------------------------------------------------- | -------- |
| 0        | case_pressure_drop pilot (incl. Hydra relocation)                             | 2-3 hr   |
| 1        | alpha-D file moves + `feature_analysis` split + datasets_profile move         | 4-5 hr   |
| 1.5      | Break `datasets_tabular` ⇄ alpha-D cycle (allowlist-agnostic dataset)         | 1 hr     |
| 2a       | Extended metrics hook                                                         | 2 hr     |
| 2b       | Tabular target transform                                                      | 2 hr     |
| 2c       | Plotting decode_fn                                                            | 1 hr     |
| 2d       | Training lifecycle hook                                                       | 2 hr     |
| 2e       | Profile adapter dataset entrypoint (NEW)                                      | 1.5 hr   |
| 3        | Config relocation + per-case `hydra.searchpath`                               | 2 hr     |
| 4        | Drop shims                                                                    | 30-45 min|
| 5        | Docs                                                                          | 1 hr     |
| 6        | `moose_etl` → `cases/moose_grid/` (NEW)                                       | 2-3 hr   |
| 7        | `AlphaDConv1D` → `Conv1DProfile` + checkpoint migration + polish (NEW)        | 3-4 hr   |
| 8        | `pyproject.toml` package-isation (NEW)                                        | 3-4 hr   |

**Total: ~28-31 hr of focused work** (~4 days). The risky phase remains
Phase 2 (extracted hooks) — if 2a + 2d + 2e land clean, the rest is
mechanical. Phases 6-8 are independently mergeable after the alpha-D
work and can be sequenced flexibly.

## Acceptance criteria

The refactor is done when:

1. `grep -rEn "alpha_d|signed_log1p|delta_p_case|target_residual_baseline" src/training/`
   returns **zero matches**.
2. `grep -rEn "AlphaD[A-Za-z_]+" src/training/`
   returns **zero matches** (catches `AlphaDProfileDataset`,
   `AlphaDExperiment`, `AlphaDConv1D`). The `AlphaDConv1D` alias
   introduced in Phase 7 is documented and removed in a post-release
   cleanup once all known checkpoints have been migrated.
3. `grep -rEn "from training\.alpha_d_|import training\.alpha_d_" src/ tests/`
   returns **zero matches**.
4. `grep -rEn "from feature_analysis|import feature_analysis" src/ tests/`
   returns **zero matches**.
5. `grep -rEn "from training\.datasets_profile|import training\.datasets_profile" src/ tests/`
   returns **zero matches**.
6. `grep -rEn "from moose_etl|import moose_etl" src/ tests/`
   returns **zero matches**. *Deferred*: this criterion gates *only* on
   Phase 6's shim-removal commit, which lands at least two weeks after
   the Phase 6 move (deprecation window) and is not blocked by Phases
   0-5 / 7 / 8. Treat as a release-readiness gate, not a Phase-6 exit
   criterion.
7. `find src/cases/alpha_d/ -type f` enumerates the full alpha-D-specific
   surface; nothing alpha-D-specific outside that folder (modulo the
   `AlphaDConv1D` backward-compat alias).
8. A new user can copy `cases/alpha_d/` → `cases/<new_case>/`, rename a
   handful of identifiers in 3-4 files, and have a working scaffold for
   their own case.
9. `pytest` is green at every commit between Phase 0 and Phase 8.
10. `python train.py --config-path cases/alpha_d/configs --config-name train_mlp`
    reproduces the previous numbers bit-for-bit on the same seed (alpha-D
    MLP has a checkpoint at `data/cases/alpha_d_mlp/model.mdlus` to verify
    against).
11. The 1D-conv path (`--config-name train_conv1d`) trains end-to-end and
    its checkpoints round-trip through `Module.from_checkpoint` — both
    pre- and post-Phase-7 checkpoints load successfully (post-7 via the
    backward-compat alias and/or the migration utility).
12. After Phase 8, `pip install -e .` from a fresh checkout is sufficient
    to make `pytest` pass without any `sys.path` manipulation.

## Out of scope (separate refactors)

- Larger architectural changes to `Conv1DProfile` (attention blocks,
  multi-resolution heads, alternative residual scaling). Phase 7 only
  does rename + cosmetic polish; structural network changes are a
  separate plan.
- Consolidating diagnostics across cases (each case keeps its own).
- Re-introducing `eval_delta_p_baseline.py` and `eval_delta_p_closed_form.py`
  (or whatever replaces them) — those scripts no longer exist on disk
  and are not in any commit. If they reappear, fold them into
  `cases/alpha_d/diagnostics/` at that time.
- Extracting `cases/<x>/` into separate Git repos or installable plugins.
- Replacing Hydra with a different config system.
