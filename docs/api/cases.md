# `cases`

Per-case packages. Each case owns its ETL pipeline, datasets, physics
helpers, experiment hook, and Hydra configs. The training core does
**not** import from `cases/*` — coupling flows in one direction only.

## `cases.alpha_d`

### Experiment

```{eval-rst}
.. automodule:: cases.alpha_d.experiment
   :members:
   :undoc-members:
   :show-inheritance:
```

### Feature data

```{eval-rst}
.. automodule:: cases.alpha_d.feature_data
   :members:
   :undoc-members:
   :show-inheritance:
```

### Transforms and metrics

```{eval-rst}
.. automodule:: cases.alpha_d.transforms
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.alpha_d.metrics
   :members:
   :undoc-members:
   :show-inheritance:
```

### Datasets

```{eval-rst}
.. automodule:: cases.alpha_d.datasets.profile
   :members:
   :undoc-members:
   :show-inheritance:
```

### Physics

```{eval-rst}
.. automodule:: cases.alpha_d.physics.targets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.alpha_d.physics.baseline
   :members:
   :undoc-members:
   :show-inheritance:
```

### ETL (Source · Transform · Sink)

```{eval-rst}
.. automodule:: cases.alpha_d.etl.source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.alpha_d.etl.transform
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.alpha_d.etl.sink
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cases.case_pressure_drop`

```{eval-rst}
.. automodule:: cases.case_pressure_drop.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.case_pressure_drop.modeling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.case_pressure_drop.workflow
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.case_pressure_drop.feature_selection
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.case_pressure_drop.pycaret_selection
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.case_pressure_drop.distribution
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.case_pressure_drop.plotting
   :members:
   :undoc-members:
   :show-inheritance:
```

## `cases.moose_grid`

### ETL data sources

```{eval-rst}
.. automodule:: cases.moose_grid.etl.data_sources.exodus_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.moose_grid.etl.data_sources.csv_source
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.moose_grid.etl.data_sources.zarr_sink
   :members:
   :undoc-members:
   :show-inheritance:
```

### ETL transformations

```{eval-rst}
.. automodule:: cases.moose_grid.etl.transformations.moose_transform
   :members:
   :undoc-members:
   :show-inheritance:
```

### Schemas and validators

```{eval-rst}
.. automodule:: cases.moose_grid.etl.schemas
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cases.moose_grid.etl.validators
   :members:
   :undoc-members:
   :show-inheritance:
```
