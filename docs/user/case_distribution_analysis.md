# Case Distribution Analysis

This guide covers the ``analyze_case_distribution.py`` preprocessing tool,
which previews how simulation cases are distributed across the three
design parameters (``Dr``, ``Re``, ``Lr``) before training.

## Why it matters

The alpha-D and case-level pressure-drop surrogates generalise only as
well as their training support permits.  Bins with few training samples
produce unreliable predictions regardless of model capacity.  Running
this tool up-front answers:

- Do I have enough cases in each ``Dr`` / ``Re`` / ``Lr`` bin?
- Which bins will my ``min_Dr`` / ``exclude_cases`` filter remove?
- Does a recorded train / test split cover every bin?

It inspects the Zarr directory (and optionally a ``run_meta.json``) and
prints coloured tables so under-supported regions are obvious.

## How it works

```
data/flow_contraction_expansion/parametric_study/processed/
  Re_*__Dr_*__Lr_*.zarr        <-- discovered by the tool
                                    (parsed from the case name)

data/models/.../run_meta.json   <-- optional; when provided,
                                    Train/Test columns are populated
                                    from split.train_sims / test_sims
```

Support thresholds (based on the **train** count when a split is
provided, or the total count otherwise):

| Marker | Train cases | Meaning |
|--------|-------------|---------|
| ``✗ none`` | 0 | Bin will not be learned at all |
| ``⚠ very low`` | < 3 | Extreme extrapolation risk |
| ``⚠ low`` | < 10 | Generalisation in this bin is unreliable |
| ``◦ ok`` | < 30 | Usable but watch for drift |
| ``✓ good`` | ≥ 30 | Adequate support |

## Quick start

From inside the container:

```bash
cd src && python analyze_case_distribution.py \
    --run-meta ../data/models/case_pressure_drop/run_meta.json
```

From the host with Apptainer:

```bash
apptainer exec th-holo-gpu.sif bash -c \
    'cd src && python analyze_case_distribution.py \
        --run-meta ../data/models/case_pressure_drop/run_meta.json'
```

## Usage examples

### Inspect the raw Zarr directory (before training)

```bash
cd src && python analyze_case_distribution.py \
    --zarr-dir ../data/flow_contraction_expansion/parametric_study/processed
```

Supports the whole dataset with a single ``Total`` column.  Use this to
check the raw simulation inventory.

### Preview filters that will be applied during training

```bash
cd src && python analyze_case_distribution.py \
    --zarr-dir ../data/flow_contraction_expansion/parametric_study/processed \
    --min-Dr 0.333
```

Mirrors the filtering logic in ``TabularPairDataset``.  Useful when
deciding the ``data.min_Dr`` value in ``alpha_d_mlp.yaml``: run it with
different thresholds and see which bins disappear.

### Exclude specific problematic cases

```bash
cd src && python analyze_case_distribution.py \
    --zarr-dir ../data/flow_contraction_expansion/parametric_study/processed \
    --exclude Re_11927__Dr_0p05__Lr_0p052 \
    --exclude Re_7722__Dr_0p05__Lr_0p052
```

``--exclude`` can be repeated to drop any number of case names.  The
filter is applied by exact ``{stem}`` match against the Zarr files.

### Inspect a prior train / test split

```bash
cd src && python analyze_case_distribution.py \
    --run-meta ../data/models/case_pressure_drop/run_meta.json
```

Populates ``Train`` and ``Test`` columns from the recorded split and
classifies Support on the train count.  If ``--zarr-dir`` is omitted,
the tool reads ``data.zarr_dir`` from ``run_meta.json``.

### Restrict to a subset of axes

```bash
cd src && python analyze_case_distribution.py \
    --run-meta ../data/models/case_pressure_drop/run_meta.json \
    --axes Dr
```

Useful when you only care about one parameter (e.g. diagnosing poor
performance at large ``Dr``).

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| ``--zarr-dir`` | from ``run_meta`` if provided | Directory of processed ``*.zarr`` case stores |
| ``--run-meta`` | ``null`` | ``run_meta.json`` to read a recorded train / test split |
| ``--min-Dr`` | ``null`` | Drop cases whose ``Dr`` is below this value |
| ``--exclude`` | ``[]`` | Case name to exclude (repeatable) |
| ``--axes`` | ``Dr Re Lr`` | Which parameter axes to report |

At least one of ``--zarr-dir`` or ``--run-meta`` must be provided.

## Output sections

### Header panel

Summarises the total case count, the Zarr directory in use, and the
train / test split (when a ``run_meta.json`` is supplied).

### Per-axis distribution tables

One table per axis (``Dr``, ``Re``, ``Lr``).  Columns:

- **Axis value** (e.g. ``Dr = 0.900``) -- rounded to 3 decimals, except
  ``Re`` which is shown as an integer.
- **Train / Test** -- present only when a run-meta is provided.
- **Total** -- the union of train, test, and any other cases in the
  Zarr directory.
- **Support** -- coloured marker classifying the training support.

Bins flagged ``⚠ very low`` or ``✗ none`` are likely to show outsized
evaluation errors.  Cross-reference them with the
[version comparison tool](version_comparison.md) to confirm.

## Typical workflow

1. **Before the first ETL→training pass**, run with ``--zarr-dir`` only
   to see the raw simulation inventory.  Look for bins with fewer than
   3 cases and decide whether to gather more simulations or drop them.

2. **Before each HPO run**, run with ``--zarr-dir`` plus the
   ``--min-Dr`` and ``--exclude`` values from your config.  Confirm
   you still have ``◦ ok`` or better support in every bin you care
   about.

3. **After a training run**, run with ``--run-meta`` to verify the
   stratified split gave each bin at least one train and one test case.

4. **When diagnosing a worst-case list** (see
   ``evaluate_case_pressure_drop.py`` output), look up the failing
   cases' ``Dr`` / ``Re`` / ``Lr`` in this table.  If they land in a
   ``⚠ low``-support bin, the fix is data, not model.

## Adding a new axis

The axis set is currently hard-coded to ``("Dr", "Re", "Lr")`` to match
the case-name convention (``Re_*__Dr_*__Lr_*``).  If you add a new
design parameter to the simulation campaign:

1. Extend the case-name pattern in the ETL.
2. Update ``parse_case_params`` in ``src/case_pressure_drop/distribution.py``
   to extract the new key.
3. Add the key to the ``AXES`` tuple and to the ``axis`` index maps
   inside ``bin_by``.

## Related guides

- [Alpha-D Surrogate Tutorial](alpha_d_surrogate.md) -- end-to-end ETL,
  training, and evaluation workflow.
- [Hyperparameter Optimization](hyperparameter_optimization.md) --
  configuring ``data.min_Dr`` and ``data.exclude_cases`` filters that
  this tool previews.
- [Version Comparison](version_comparison.md) -- review evaluation
  metrics across versions; cross-reference worst-case lists with the
  distribution tables produced here.
