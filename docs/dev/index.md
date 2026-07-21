# Developer Guide

Internals of the ETL pipeline, the public dataset API, and the
training/evaluation loop. Read these alongside the
[Architecture](../architecture.md) page when you need to extend the
trainer, add a new model, or audit how a particular surrogate is wired
together. Public orchestration signatures are also collected in the
[workflow API reference](../api/workflows.md).

```{toctree}
:maxdepth: 1
:caption: Pipeline internals

etl_pipeline
dataset
fno_train_eval
workflows
```

```{toctree}
:maxdepth: 1
:caption: Contributor reference

code_style
building_docs
```
