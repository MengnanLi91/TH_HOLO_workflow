# MOOSE Grid Case

`cases/moose_grid/` is the canonical MOOSE → grid/graph training case.
It owns the original Exodus-based ETL pipeline that produces the
multi-mode `*.zarr` stores consumed by
{py:class}`dataset.moose_dataset.MooseDataset` (see the
[Dataset API page](../dev/dataset.md)), and the FNO example training
config that downstream grid models clone.

## Layout

```text
cases/moose_grid/
├── configs/
│   ├── etl_base.yaml      # shared ETL defaults
│   ├── etl.yaml           # lid-driven flow (default config)
│   └── train_fno.yaml     # FNO training example
├── etl/
│   ├── data_sources/      # ExodusDataSource, CSVProbeSource, MooseZarrSink
│   ├── transformations/   # MooseDataTransformation
│   ├── schemas.py         # Raw / processed schemas
│   └── validators.py
└── run_etl.py             # ETL entry point
```

## Workflow

```{mermaid}
flowchart LR
    A["MOOSE .e + CSV probes"] --> S["ExodusDataSource<br/>CSVProbeSource"]
    S --> T["MooseDataTransformation<br/>z-score · edges · grid interp"]
    T --> K["MooseZarrSink"]
    K --> Z["*.zarr<br/>(mesh, fields, grid, probes, norm_stats)"]
    Z --> D["MooseDataset (graph / point_cloud / grid)"]
    D --> FNO["FNO / MeshGraphNet / AFNO / Pix2Pix"]
```

## Entry points

```bash
# ETL: MOOSE outputs → Zarr (default = lid-driven flow)
python cases/moose_grid/run_etl.py
# equivalent to:
python run_etl.py --config-path cases/moose_grid/configs --config-name etl

# Train FNO
python train.py --config-path cases/moose_grid/configs --config-name train_fno

# Evaluate
python evaluate.py --config-path cases/moose_grid/configs --config-name train_fno
```

## Further reading

- [ETL Pipeline Internals](../dev/etl_pipeline.md) — stage-by-stage
  description of the Exodus → Zarr transformation.
- [Dataset API](../dev/dataset.md) — what `MooseDataset` returns in
  each mode.
- [FNO Training and Evaluation](../dev/fno_train_eval.md) — config
  walkthrough for the grid surrogate path.
