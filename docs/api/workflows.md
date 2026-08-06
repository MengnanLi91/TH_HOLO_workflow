# `workflows`

The case-agnostic orchestration API. Case packages construct a
`WorkflowDefinition`; the runner validates and executes it without importing
case code itself.

## Public contracts

```{eval-rst}
.. automodule:: workflows.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## Runner and DAG validation

```{eval-rst}
.. automodule:: workflows.runner
   :members: WorkflowRunner, validate_workflow
   :show-inheritance:
```

## Executors

```{eval-rst}
.. automodule:: workflows.executors
   :members: LocalExecutor, ApptainerExecutor, build_executors
   :show-inheritance:
```

The CLI is documented in the
[workflow user guide](../user/running_workflows.md); case-specific method and
export contracts are documented in the
[workflow developer guide](../dev/workflows.md).

