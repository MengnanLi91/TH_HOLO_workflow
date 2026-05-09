"""1D-conv α_D profile model.

The model treats each case as one sample of shape ``[C, S]`` (channels =
features, S = stations) and produces ``[O, S]`` per case. Replicate
padding avoids zero-pad artefacts at z_hat boundaries; dilations widen
the receptive field without increasing depth.

The class subclasses ``physicsnemo.Module`` so checkpoints round-trip
through ``Module.from_checkpoint``. PhysicsNeMo records
``cls.__module__``/``cls.__name__`` in ``Module.__new__`` and re-imports
the class by ``getattr(module, name)`` at load time, so the class must
live at module scope (not inside ``build``).

When ``physicsnemo`` is unavailable in the environment the module
imports cleanly but the model is not registered — mirroring how
``GraphAdapter`` only activates if PyG is importable. This keeps
``_load_builtins`` from hard-failing in envs without physicsnemo.
"""

import torch
import torch.nn as nn

from training import import_physicsnemo_attr
from training.models import register_model


def _make_block(channels: int, kernel_size: int, dilation: int, dropout: float) -> nn.Sequential:
    pad = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, padding_mode="replicate",
        ),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, padding_mode="replicate",
        ),
        nn.SiLU(),
    )


try:
    _PhysicsNeMoModule = import_physicsnemo_attr("physicsnemo", "Module")
except ModuleNotFoundError:
    _PhysicsNeMoModule = None


if _PhysicsNeMoModule is not None:

    class AlphaDConv1D(_PhysicsNeMoModule):
        """Residual dilated 1D-conv stack over the station axis."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden: int,
            num_blocks: int,
            kernel_size: int,
            dilations: list[int],
            dropout: float,
        ):
            super().__init__()
            self.encoder = nn.Conv1d(in_channels, hidden, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    _make_block(
                        hidden, kernel_size,
                        dilations[i % len(dilations)],
                        dropout,
                    )
                    for i in range(num_blocks)
                ]
            )
            self.act = nn.SiLU()
            self.head = nn.Conv1d(hidden, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act(self.encoder(x))
            for block in self.blocks:
                h = h + block(h)
            return self.head(h)

    def build(model_cfg: dict, dataset_info: dict):
        resolved = {
            "in_channels": int(dataset_info["in_channels"]),
            "out_channels": int(dataset_info["out_channels"]),
            "hidden": int(model_cfg.get("hidden_channels", 64)),
            "num_blocks": int(model_cfg.get("num_blocks", 4)),
            "kernel_size": int(model_cfg.get("kernel_size", 5)),
            "dilations": list(model_cfg.get("dilations", [1, 2, 4, 1])),
            "dropout": float(model_cfg.get("dropout", 0.05)),
        }
        model = AlphaDConv1D(**resolved)
        model._resolved_model_params = dict(resolved)
        return model

    register_model("conv1d_profile", build_fn=build, adapter="profile")
