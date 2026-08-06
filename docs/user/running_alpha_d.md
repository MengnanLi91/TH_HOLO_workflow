# Run the Alpha-D Case

This is the operational path for running the complete alpha-D study: raw MOOSE
Exodus output to Zarr, profile-model training, MOOSE coupling, and evidence
reports. For the physics and equations, see the
[coupling demo](../demo_cases/alpha_d_coupling_physics.md). For the detailed
ETL feature and target definitions, see the
[alpha-D surrogate tutorial](alpha_d_surrogate.md).

## Before you start

Run commands from the repository root. The large alpha-D campaign is not
tracked in Git: `data/` is ignored. You need either:

- a raw campaign where each immediate case directory contains
  `simulation_out.e` and `case_metadata.txt`, or the campaign root contains
  `cases_manifest.csv`; or
- an existing directory of processed `*.zarr` stores.

The ETL and study execute their heavy Python work in Apptainer. Set the image
paths for the commands you need:

```bash
uv sync
export MULTIFID_PYTHON_IMAGE=/absolute/path/to/multifid-th.sif
export MULTIFID_MOOSE_IMAGE=/absolute/path/to/moose-dev.sif
```

The ETL needs only `MULTIFID_PYTHON_IMAGE`; the full coupled study needs both.

## 1. Build Zarr stores from raw Exodus output

If `inputs.raw_dir` in
`src/cases/alpha_d/configs/etl_workflow.toml` already names your campaign
root, run:

```bash
uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/etl_workflow.toml

uv run multifid-workflow etl \
  --config src/cases/alpha_d/configs/etl_workflow.toml \
  --run-id alpha-d-etl-001
```

To use another campaign directory without editing the TOML, add
`--input-dir`. It overrides `inputs.raw_dir` and binds an external directory
into Apptainer automatically:

```bash
uv run multifid-workflow etl \
  --config src/cases/alpha_d/configs/etl_workflow.toml \
  --run-id alpha-d-etl-002 \
  --input-dir /absolute/path/to/parametric_study
```

The input root must be the directory whose direct children are case folders,
not a parent directory that merely contains `parametric_study/`.

The output is self-contained:

```text
data/workflows/alpha_d_etl/<run-id>/
├── run_manifest.json
├── logs/
├── data/etl_summary.json
└── data/processed/<case>.zarr/
```

Read `data/etl_summary.json` before moving on. A `partial` result means one or
more raw cases did not yield Zarr output. The stage log explains why:

```bash
rg -n "Skipping|No .*files|Writing alpha_D|Traceback|Error" \
  data/workflows/alpha_d_etl/alpha-d-etl-001/logs/extract.01.extract_alpha_d.log
```

## 2. Point the coupling study at processed data

If you created Zarr data in step 1, edit a local copy of
`src/cases/alpha_d/configs/coupling_study.toml` so its input block is:

```toml
[inputs]
mode = "reuse"
zarr_dir = "data/workflows/alpha_d_etl/alpha-d-etl-001/data/processed"
raw_dir = "data/flow_contraction_expansion/parametric_study"
```

If you already have processed data elsewhere, set `zarr_dir` to that directory
instead. When it is outside the repository, add the same absolute directory
to `[executors.python].binds` in the copied TOML.

Choose a new study run ID whenever the TOML, code, or input dataset changes.

## 3. Plan the study, then run inexpensive setup

```bash
uv run multifid-workflow plan \
  --config src/cases/alpha_d/configs/coupling_study.toml

uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-002 \
  --target plan_panels
```

`plan_panels` resolves the seven held-out panels and writes their canonical
case files. Inspect them under:

```text
data/workflows/alpha_d_coupling/alpha-d-conv1d-002/panels/
```

## 4. Run the study

The 27-stage study excludes outer panel cases, selects one PyCaret feature set
for Conv1D, tunes that frozen feature set once, trains direct and alpha-D
models, exports closures, runs MOOSE, and writes the evidence report:

```bash
uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-002
```

For a narrower checkpoint before committing to every panel, run one target:

```bash
uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-002 \
  --target panel.indist_panel.train_alpha
```

This target includes the HPO stage. `plan_panels` writes the union of every
held-out and report panel to `panels/hpo_exclude_cases.txt`; those outer cases
are removed before screening folds are built. Screening runs 40 total trials,
then confirms the five strongest sampled candidates plus the three scientific
controls over folds/seeds 42, 43, and 44.

## 5. Monitor and resume

```bash
uv run multifid-workflow status \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-conv1d-002
```

Rerun the exact same `run` command to resume. Successful stages are reused
only while their artifact checksums, upstream fingerprints, and semantic
validators remain valid. Failed MOOSE cases are retained with their command
and log records; they are not counted as zero-pressure results.

## 6. Review and publish results

The primary outputs are:

```text
report/pressure_drop_comparison.md
report/pressure_drop_comparison.json
report/paired_case_errors.csv
report/moose_paired_case_errors.csv
report/pressure_drop_comparison_errors.svg
moose_matrix.json
```

They are all below `data/workflows/alpha_d_coupling/<run-id>/`. Apply the
publication gates against the previous run first:

```bash
uv run python -m cases.alpha_d.compare_training_runs \
  --baseline-csv data/alpha-d-conv1d-001/report/paired_case_errors.csv \
  --candidate-csv data/workflows/alpha_d_coupling/alpha-d-conv1d-002/report/paired_case_errors.csv \
  --out-json data/workflows/alpha_d_coupling/alpha-d-conv1d-002/report/training_comparison.json \
  --out-markdown data/workflows/alpha_d_coupling/alpha-d-conv1d-002/report/training_comparison.md
```

This passes only when in-domain MARE and p90 improve, Lr-low/high MARE and
absolute signed bias improve, and no other panel MARE regresses by more than
0.25 percentage points. Publish only after it reports `PASS`:

```bash
uv run multifid-workflow publish \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-conv1d-002

uv run multifid-workflow publish \
  --run-dir data/workflows/alpha_d_coupling/alpha-d-conv1d-002 \
  --check
```

`publish --check` verifies published-file drift without rerunning the study.

The acceleration-head-off baseline is implemented but intentionally not part
of the first run. See the [switch physics and consistency contract](alpha_d_surrogate.md#acceleration-head-baseline-switch).
Launch the controlled ablation later with:

```bash
uv run multifid-workflow run \
  --config src/cases/alpha_d/configs/coupling_study.toml \
  --run-id alpha-d-conv1d-002-no-accel \
  --set training.alpha.include_acceleration_head=false
```

Repeatable `--set` arguments replace existing TOML scalar paths. The resolved
configuration is persisted and fingerprinted.
