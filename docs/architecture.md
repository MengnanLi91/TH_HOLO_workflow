# Architecture

MULTIFID-TH has three layers: case-owned **ETL pipelines** turn raw MOOSE
outputs into ML-ready Zarr, the **generic training framework** consumes those
stores through adapters, and the case-agnostic **workflow kernel** composes
ETL, training, evaluation, solvers, and publication into resumable studies.
This page wires them together with the diagrams you need to navigate the
codebase.

## Workflow kernel and case boundary

`src/workflows/` is the reusable study layer. It validates a stage DAG,
executes argument arrays locally or through Apptainer, fingerprints inputs,
updates `run_manifest.json` atomically, and resumes only outputs whose
checksums and case-owned validators still pass. It never imports `cases.*`.

```{mermaid}
flowchart LR
    CLI["multifid-workflow<br/>plan · run · status · publish"] --> CORE["workflows/<br/>DAG · executors · manifest · fingerprints"]
    TOML["tracked workflow TOML"] --> CLI
    CASE["cases/&lt;case&gt;/study_workflow.py<br/>build_workflow(config, repo_root)"] --> CORE
    CORE --> ETL["existing ETL commands"]
    CORE --> TRAIN["existing training/evaluation commands"]
    CORE --> SOLVER["case-owned solver/coupling commands"]
    CORE --> RUN["data/workflows/&lt;workflow-id&gt;/&lt;run-id&gt;"]
    RUN --> PUBLISH["declared docs figures<br/>publish only"]
```

The public integration seam for a new case is one function:

```python
def build_workflow(config, repo_root) -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id=config["workflow"]["id"],
        version=1,
        stages=(Stage("prepare", prepare), Stage("train", train, ("prepare",))),
    )
```

Keep physics, panel construction, model-specific command arguments, solver
validation, and report interpretation in the case package. Keep DAG mechanics,
execution environments, provenance, and resume rules in `workflows`. See the
[workflow extension guide](dev/workflows.md) for the complete contract.

### Configuration and ML-method boundary

The study TOML selects one ML method per run, while the selected Hydra YAML
defines how that method trains. The case builder translates both into stages
and enforces the artifact contract needed by downstream physics.

```{mermaid}
flowchart LR
    TOML["Study TOML<br/>method · runner · contracts"] --> CASE["case study_workflow.py<br/>stage and validator logic"]
    YAML["Hydra YAML<br/>model · adapter · loss · HPO"] --> RUNNER["generic training runner"]
    CASE -->|"runner module + config name"| RUNNER
    RUNNER --> MODEL["registered model<br/>or model.entrypoint"]
    MODEL --> ADAPTER["grid · graph · pointwise · profile"]
    ADAPTER --> ART["checkpoint + run_meta.json"]
    ART --> VALIDATE{"case artifact<br/>contract valid?"}
    VALIDATE -->|yes| EXPORT["configured exporter"]
    EXPORT --> SOLVER["case solver / coupling"]
    VALIDATE -->|no| FAIL["stage failed with contract error"]
```

Configuration ownership is intentionally non-overlapping:

- TOML controls study inputs, panels, executors, selected method, artifact
  names, contracts, and publication.
- Hydra YAML controls architecture, dataset/adapter, split, optimizer, loss,
  schedule, and HPO search space.
- `study_workflow.py` controls case-specific stages, dependencies, physics,
  validators, solver retries, and reports.

Changing the TOML method changes the run's resolved-configuration hash, so an
existing run ID cannot accidentally resume checkpoints from another model.
For alpha-D, `alpha_d_profile_v1` validates the training metadata and profile
shape before `forchheimer_profile_v1` is allowed to feed MOOSE. Other cases may
define different contracts without changing the workflow kernel.

## Two ETLs feed one trainer

```{mermaid}
flowchart LR
    subgraph ETL_grid["cases/moose_grid (run_etl.py)"]
        Aex["MOOSE .e + CSV probes"]
        Aetl["ExodusDataSource → MOOSEDataTransformation → MOOSEZarrSink"]
        Aex --> Aetl
    end

    subgraph ETL_alpha["cases/alpha_d (run_etl.py)"]
        Bex["MOOSE .e + case_metadata.txt"]
        Betl["AlphaDSource → AlphaDTransformation → AlphaDZarrSink"]
        Bex --> Betl
    end

    Aetl --> Az["{sim_name}.zarr<br/>mesh · fields · grid · probes · norm_stats"]
    Betl --> Bz["{case}.zarr<br/>features [50,10] · targets [50,1] · sample_weight · metadata"]

    Az --> Train["train.py / evaluate.py<br/>(generic framework)"]
    Bz --> Train

    Train -->|"grid"| FNO["FNO · AFNO · Pix2Pix"]
    Train -->|"graph"| MGN["MeshGraphNet"]
    Train -->|"pointwise"| MLP["MLP (FullyConnected)"]
    Train -->|"profile"| Conv["Conv1DProfile"]
```

The orchestrator on either side is `physicsnemo_curator.etl.ETLOrchestrator`,
which Hydra wires together from a single `etl.yaml` via `_target_`
keys. Both ETLs share the same Source → Transformation → Sink
convention, only the inputs and per-step transforms differ.

See:

- `cases/moose_grid/` → [MOOSE Grid case page](cases/moose_grid.md) and
  the [ETL pipeline internals](dev/etl_pipeline.md).
- `cases/alpha_d/` → [Alpha-D case page](cases/alpha_d.md) and the
  [Alpha-D surrogate tutorial](user/alpha_d_surrogate.md).

## Training framework: registry → adapter → dataset → runner

```{mermaid}
flowchart TB
    subgraph Reg["Model registry (training/models/__init__.py)"]
        R1["mlp → pointwise"]
        R2["fno · afno · pix2pix → grid"]
        R3["meshgraphnet → graph"]
        R4["conv1d_profile → profile"]
    end

    subgraph Ad["Adapters (training/adapters.py)"]
        A1["GridAdapter<br/>GridPairDataset"]
        A2["GraphAdapter<br/>GraphPairDataset"]
        A3["PointwiseAdapter<br/>TabularPairDataset"]
        A4["ProfileAdapter<br/>AlphaDProfileDataset"]
    end

    subgraph Run["Runner (training/runner.py)"]
        Rn["train() / evaluate()<br/>splits · LR schedule · early stop · run_meta.json"]
    end

    Exp["Experiment hooks<br/>training_step / eval_step"] --> Rn

    R1 --> A3
    R2 --> A1
    R3 --> A2
    R4 --> A4
    A1 --> Rn
    A2 --> Rn
    A3 --> Rn
    A4 --> Rn
```

Swapping the adapter is the only thing that changes between an FNO run
and an MLP run. The runner stays the same.

Key files:

- Registry — {py:mod}`training.models` (`src/training/models/__init__.py`).
- Adapters — {py:mod}`training.adapters`.
- Datasets — {py:mod}`training.datasets`, {py:mod}`training.datasets_tabular`,
  and the case-owned {py:class}`cases.alpha_d.datasets.profile.AlphaDProfileDataset`.
- Runner — {py:mod}`training.runner`.
- Experiments — base {py:class}`training.experiment.Experiment`; case-specific
  overrides live with the case (e.g.
  {py:class}`cases.alpha_d.experiment.AlphaDExperiment`).

## HPO orchestration

```{mermaid}
flowchart LR
    Cfg["train_*.yaml<br/>(hpo section present)"] --> Gate{"hpo=null<br/>on CLI?"}
    Gate -- "yes" --> Direct["train() once<br/>on full pool"]
    Gate -- "no" --> Split["Hold out test set<br/>Split training pool<br/>→ inner train · validation"]
    Split --> Loop["Optuna study<br/>(TPE / pruner)"]
    Loop --> Trial["Trial: build model · train inner · score on val"]
    Trial --> Loop
    Loop --> Best["best_params.json<br/>best_config.yaml<br/>optimization_history.png · ..."]
    Best --> Retrain{"retrain_best?"}
    Retrain -- "true" --> Final["Retrain on full training pool<br/>evaluate on held-out test"]
    Retrain -- "false" --> End["End (HPO artifacts only)"]
    Direct --> Final
```

The held-out test set is **never** used during HPO trials. After
optimization the best trial is retrained on the full training pool and
evaluated on the test set as usual. See the
[Hyperparameter Optimization guide](user/hyperparameter_optimization.md)
for the full search-space format and artifact reference.

## Per-case folder layout

Every case-specific concern lives in a self-contained `src/cases/<case>/`
folder, kept out of the generic training core.

```{mermaid}
flowchart TB
    SRC["src/"] --> CASES["cases/"]
    SRC --> TRAIN["training/<br/>(generic: registry · adapters · runner · experiment)"]
    SRC --> DS["dataset/<br/>(MOOSEDataset public API)"]
    SRC --> ENT["train.py · evaluate.py"]

    CASES --> MG["moose_grid/"]
    CASES --> AD["alpha_d/"]
    CASES --> CP["case_pressure_drop/"]

    MG --> MGc["configs/<br/>etl_base · etl · train_fno"]
    MG --> MGe["etl/<br/>data_sources · transformations"]
    MG --> MGr["run_etl.py"]

    AD --> ADc["configs/<br/>train_mlp · train_conv1d · etl · pycaret"]
    AD --> ADds["datasets/<br/>profile.py"]
    AD --> ADe["etl/<br/>source · transform · sink"]
    AD --> ADp["physics/<br/>baseline · targets"]
    AD --> ADx["experiment.py<br/>feature_data · metrics · transforms"]
    AD --> ADr["run_etl.py · train.py"]

    CP --> CPc["configs/<br/>case_pressure_drop"]
    CP --> CPm["data · modeling · feature_selection · plotting"]
    CP --> CPr["run_case_pressure_drop.py"]
```

For newcomers: pick a case folder, open its README (alpha-D) or top-level entry
script, and follow the imports outward. Neither the training core nor the
workflow core imports from `cases/*`; case builders and adapters point inward
to the generic layers, never the reverse.

## `run_meta.json` round-trip

```{mermaid}
sequenceDiagram
    autonumber
    participant T as train.py
    participant R as runner.train()
    participant FS as Filesystem
    participant E as evaluate.py
    participant Rev as runner.evaluate()

    T->>R: load config · build adapter · build dataset · split
    R->>FS: write model.mdlus
    R->>FS: write run_meta.json<br/>(dataset args · split sims · adapter · model entrypoint · params)
    Note over FS: checkpoint + meta land together

    E->>FS: read run_meta.json (next to ckpt)
    E->>Rev: reconstruct dataset · split · target transform · model
    Rev->>FS: read model.mdlus
    Rev-->>E: per-field MSE / RMSE on held-out test cases
    E->>FS: (optional) metrics.json · plots
```

This is the single invariant that lets `evaluate.py` reproduce training
conditions exactly without re-passing every flag. Don't move or rename
`run_meta.json` without updating both `train.py` and `evaluate.py`.

## Vendored PhysicsNeMo fallback

`training/__init__.py` exposes `import_physicsnemo_module` /
`import_physicsnemo_attr`. If `physicsnemo` is not installed, those
helpers add the checked-out submodule at `physicsnemo/` to `sys.path`
and retry. This means training code works in both the `etl-dev` image
(no PhysicsNeMo) and the `etl` / `etl-gpu` / `etl-ngc` images
(PhysicsNeMo installed from PyPI or NGC) — provided the `physicsnemo`
submodule is initialized.
