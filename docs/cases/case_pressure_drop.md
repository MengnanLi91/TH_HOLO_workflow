# Case Pressure-Drop Surrogate

The `case_pressure_drop` case trains a **case-level** regressor for the
total ROI pressure drop `Δp` produced by a flow contraction-expansion.
Unlike the alpha-D surrogate, which predicts a 50-station profile per
case, this one collapses each simulation to a single feature row and a
single scalar target, then fits a tabular regressor (typically sklearn-
or PyCaret-based) with cross-validation.

The full implementation lives in `src/cases/case_pressure_drop/`. There
is no dedicated long-form tutorial yet; the source modules are
self-describing and the in-tree config is heavily commented.

## Entry points

| Script | Purpose |
| --- | --- |
| `src/cases/case_pressure_drop/run_case_pressure_drop.py` | Train + evaluate via the shared `train_case_pressure_drop` / `evaluate_case_pressure_drop` workflow. |
| `src/cases/case_pressure_drop/evaluate_case_pressure_drop.py` | Standalone evaluation from a saved model directory. |
| `src/cases/case_pressure_drop/pycaret_selection.py` | PyCaret-backed feature selection (alternative to the Borda sklearn ensemble). |

## Config

`src/cases/case_pressure_drop/configs/case_pressure_drop.yaml`
encapsulates:

- `data.zarr_dir` and the `min_Dr` / `exclude_cases` filters.
- A stratified train/test split over the `(Dr, Re, Lr)` design space.
- A two-backend feature-selection block (`method: borda` or `pycaret`)
  with a curated `candidate_features` list.
- A list of regressors to cross-validate.

Use [`analyze_case_distribution.py`](../user/case_distribution_analysis.md)
with `--run-meta <path>/run_meta.json` to audit the resulting split
across bins.

## Related guides

- [Case Distribution Analysis](../user/case_distribution_analysis.md) —
  shares the `Re_*__Dr_*__Lr_*` parsing convention with this case.
- [Hyperparameter Optimization](../user/hyperparameter_optimization.md)
  — applies only to the PyTorch-based surrogates; the case-level
  regression uses sklearn / PyCaret CV instead.
