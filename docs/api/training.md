# `training`

The generic training framework. One pair of entry points (`train.py`,
`evaluate.py`) drives every model through a model registry, three or
four adapter classes, a shared runner, and pluggable experiment hooks.

## Package

```{eval-rst}
.. automodule:: training
   :members:
   :undoc-members:
   :show-inheritance:
```

## Adapters

```{eval-rst}
.. automodule:: training.adapters
   :members:
   :undoc-members:
   :show-inheritance:
```

## Runner

```{eval-rst}
.. automodule:: training.runner
   :members:
   :undoc-members:
   :show-inheritance:
```

## Experiment

```{eval-rst}
.. automodule:: training.experiment
   :members:
   :undoc-members:
   :show-inheritance:
```

## Losses

```{eval-rst}
.. automodule:: training.losses
   :members:
   :undoc-members:
   :show-inheritance:
```

## Internal datasets

`MOOSEDataset` (under [`dataset`](dataset.md)) is the public Dataset API.
The classes below are the input/output-slicing variants used internally
by `train.py`.

```{eval-rst}
.. automodule:: training.datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.datasets_tabular
   :members:
   :undoc-members:
   :show-inheritance:
```

## Split I/O and plotting

```{eval-rst}
.. automodule:: training.split_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.plotting
   :members:
   :undoc-members:
   :show-inheritance:
```

## Model registry

```{eval-rst}
.. automodule:: training.models
   :members:
   :undoc-members:
   :show-inheritance:
```

### Built-in models

```{eval-rst}
.. automodule:: training.models.mlp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.models.fno
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.models.afno
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.models.pix2pix
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.models.meshgraphnet
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.models.conv1d_profile
   :members:
   :undoc-members:
   :show-inheritance:
```

### Checkpoint migration utility

```{eval-rst}
.. automodule:: training.models._migrate_conv1d_checkpoint
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
```

## Hyperparameter optimization

```{eval-rst}
.. automodule:: training.hpo
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.hpo.study
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.hpo.objective
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.hpo.search_space
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: training.hpo.visualize
   :members:
   :undoc-members:
   :show-inheritance:
```
