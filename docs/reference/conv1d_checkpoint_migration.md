# Conv1D Profile Checkpoint Migration

The Phase 7 refactor renamed `AlphaDConv1D` to
{py:class}`training.models.conv1d_profile.Conv1DProfile` so the
1D convolutional profile model could be reused by future cases. The
backward-compat subclass alias keeps old checkpoints loadable, but
PhysicsNeMo embeds the class name (`__name__`) and module path inside
`args.json` of the saved archive, so the alias is consulted every time
the checkpoint is loaded.

{py:mod}`training.models._migrate_conv1d_checkpoint` rewrites the embedded
`__name__` from `AlphaDConv1D` to `Conv1DProfile` so the checkpoint no
longer depends on the alias.

## Usage

```bash
# Inside the container (e.g. with Apptainer):
apptainer exec --pwd /data/lim2/projects/TH_HOLO_workflow \
  /data/lim2/projects/TH_HOLO_workflow/th-holo-gpu.sif \
  python -m training.models._migrate_conv1d_checkpoint path/to/model.mdlus
```

The script:

- Auto-detects whether the `.mdlus` archive is a zip (current format) or
  a tar (legacy PhysicsNeMo format).
- Rewrites `args.json` in place, preserving the original archive format.
- Is idempotent: re-running on an already-migrated checkpoint is a no-op.
- Errors out (without modifying the file) if the embedded class name is
  neither `AlphaDConv1D` nor `Conv1DProfile`.

## When to migrate

Migrate any `.mdlus` written by training runs that pre-date commit
`d02eba6` (`Rename AlphaDConv1D to Conv1DProfile with backward-compat
alias`). New checkpoints are already written with the new name.

## Source

- `src/training/models/_migrate_conv1d_checkpoint.py`
- API reference: {py:mod}`training.models._migrate_conv1d_checkpoint`
