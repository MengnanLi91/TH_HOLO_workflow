# FNO Training and Evaluation

This page explains how `src/train_fno.py` and `src/eval_fno.py` work with ETL-generated
Zarr datasets.

## Scripts and configs

- Training script: `src/train_fno.py`
- Evaluation script: `src/eval_fno.py`
- Train config template: `src/config/train_fno.yaml`
- Eval config template: `src/config/eval_fno.yaml`

Both scripts support:

- `--config <yaml>` for YAML defaults
- CLI overrides of any YAML value

## Data loading and supervision pairs

Both scripts use `MooseGridPairDataset` (defined in `src/train_fno.py`), which wraps
`MooseDataset(mode="grid", time_idx=-1)`.

How samples are built:

1. All `*.zarr` files in `zarr_dir` are discovered and sorted.
2. For each case, `grid_fields` has shape `[T, Nx, Ny, F]`.
3. Input tensor `x` is selected from one time index (`input_time_idx`) and selected
   channels (`input_fields`), then permuted to `[Cin, Nx, Ny]`.
4. Target tensor `y` is selected from `target_time_idx` and `output_fields`, also
   permuted to `[Cout, Nx, Ny]`.

Notes:

- The first case is used only as a schema reference (field names, grid size, number of
  timesteps). It is not the only reference target used for loss/metrics.
- Every batch target comes from that batch's own Zarr case.
- Time indices support negative indexing (`-1` means last step).

## Training flow (`train_fno.py`)

High-level steps:

1. Parse config/CLI and create `MooseGridPairDataset`.
2. Build `DataLoader` with shuffle enabled.
3. Create PhysicsNeMo `FNO` with:
   - `in_channels = len(input_fields)`
   - `out_channels = len(output_fields)`
4. Train with Adam optimizer and MSE loss (`physicsnemo.metrics.general.mse`).
5. Save checkpoint as `.mdlus` via `model.save(...)`.

Progress reporting:

- If `tqdm` is installed, training shows one progress bar over epochs.
- If `tqdm` is not installed, one log line is printed per epoch.

## Evaluation flow (`eval_fno.py`)

High-level steps:

1. Parse config/CLI and create the same dataset mapping (fields + time indices).
2. Build `DataLoader` with `shuffle=False`.
3. Load trained model from `.mdlus` with `Module.from_checkpoint(...)`.
4. Run inference under `torch.no_grad()`.
5. Accumulate overall and per-field metrics and print/save them.

## Metric definitions

Let `pred` and `y` be `[B, C, Nx, Ny]`.

Per-batch overall MSE:

- `mse_batch = mean((pred - y)^2)` over all elements.

Per-batch per-field MSE:

- `mse_field_batch[c] = mean((pred - y)^2, dim=(0, 2, 3))[c]`
- This averages over batch and spatial dimensions, keeping channel `c`.

Final reported metrics:

- `overall_mse = mean(mse_batch over batches)`
- `overall_rmse = sqrt(overall_mse)`
- `per_field_mse[c] = mean(mse_field_batch[c] over batches)`
- `per_field_rmse[c] = sqrt(per_field_mse[c])`

Important:

- Current aggregation is a mean of batch means. If the final batch has fewer samples,
  this is slightly different from an exact sample-weighted global mean.

## Normalization and interpretation

ETL writes normalized fields, so training/eval losses are in normalized units.
If you need metrics in physical units, denormalize with each field's `mean` and `std`
from metadata before computing final physical-unit errors.

## Recommended workflow

1. Train on one directory (for example `data/processed/lid-driven-train`).
2. Evaluate on a different holdout directory (for example `data/processed/lid-driven-val`).
3. Keep `input_fields`, `output_fields`, `input_time_idx`, and `target_time_idx`
   consistent between training and evaluation unless you intentionally change task setup.
