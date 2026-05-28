"""1D-convolutional profile model.

Each case is one sample of shape ``[C, S]`` (channels = features, S =
stations) and the model produces ``[O, S]`` per case. Replicate padding
avoids zero-pad artefacts at the boundary; dilations widen the
receptive field without increasing depth.

The class subclasses ``physicsnemo.Module`` so checkpoints round-trip
through ``Module.from_checkpoint``. PhysicsNeMo records
``cls.__module__``/``cls.__name__`` at save time and re-imports the
class by ``getattr(module, name)`` at load time, so the class must live
at module scope (not inside ``build``).

When ``physicsnemo`` is unavailable in the environment the module
imports cleanly but the model is not registered — mirroring how
``GraphAdapter`` only activates if PyG is importable. This keeps
``_load_builtins`` from hard-failing in envs without physicsnemo.

Backward-compat: ``AlphaDConv1D`` was the original class name. The
alias subclass below keeps old ``.mdlus`` checkpoints loadable, since
PhysicsNeMo embeds ``__name__`` in the saved archive.
"""

import torch
import torch.nn as nn

from training import import_physicsnemo_attr
from training.models import register_model

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    key = str(name).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Expected one of {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]


def _make_block(
    channels: int,
    kernel_size: int,
    dilation: int,
    dropout: float,
    activation_cls: type[nn.Module],
) -> nn.Sequential:
    pad = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            padding_mode="replicate",
        ),
        activation_cls(),
        nn.Dropout(dropout),
        nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
            padding_mode="replicate",
        ),
        activation_cls(),
    )


try:
    _PhysicsNeMoModule = import_physicsnemo_attr("physicsnemo", "Module")
except ModuleNotFoundError:
    _PhysicsNeMoModule = None


if _PhysicsNeMoModule is not None:

    class Conv1DProfile(_PhysicsNeMoModule):
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
            activation: str = "silu",
        ):
            super().__init__()
            activation_cls = _resolve_activation(activation)
            self.encoder = nn.Conv1d(in_channels, hidden, kernel_size=1)
            self.blocks = nn.ModuleList(
                [
                    _make_block(
                        hidden,
                        kernel_size,
                        dilations[i % len(dilations)],
                        dropout,
                        activation_cls,
                    )
                    for i in range(num_blocks)
                ]
            )
            self.act = activation_cls()
            self.head = nn.Conv1d(hidden, out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act(self.encoder(x))
            for block in self.blocks:
                h = h + block(h)
            return self.head(h)

    class AlphaDConv1D(Conv1DProfile):
        """Backward-compat alias for old ``.mdlus`` checkpoints."""

    def build(model_cfg: dict, dataset_info: dict):
        resolved = {
            "in_channels": int(dataset_info["in_channels"]),
            "out_channels": int(dataset_info["out_channels"]),
            "hidden": int(model_cfg.get("hidden_channels", 64)),
            "num_blocks": int(model_cfg.get("num_blocks", 4)),
            "kernel_size": int(model_cfg.get("kernel_size", 5)),
            "dilations": list(model_cfg.get("dilations", [1, 2, 4, 1])),
            "dropout": float(model_cfg.get("dropout", 0.05)),
            "activation": str(model_cfg.get("activation", "silu")),
        }
        model = Conv1DProfile(**resolved)
        model._resolved_model_params = dict(resolved)
        return model

    register_model("conv1d_profile", build_fn=build, adapter="profile")
