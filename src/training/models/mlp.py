"""Built-in FullyConnected MLP model definition."""

import torch

from training import import_physicsnemo_attr
from training.models import register_model


class _DropoutWrapper(torch.nn.Module):
    """Training-time dropout wrapper around a PhysicsNeMo model.

    Applies dropout to the model input during training.  The saved
    checkpoint contains only the inner model so that evaluation works
    with the standard ``Module.from_checkpoint`` path.
    """

    def __init__(self, model: torch.nn.Module, dropout: float):
        super().__init__()
        self.model = model
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.model(self.drop(x))

    # Delegate PhysicsNeMo serialization to the inner model.
    def save(self, path):  # noqa: D102
        return self.model.save(path)

    def state_dict(self, *args, **kwargs):  # noqa: D102
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):  # noqa: D102
        return self.model.load_state_dict(*args, **kwargs)


def build(model_cfg: dict, dataset_info: dict):
    fc_cls = import_physicsnemo_attr(
        "physicsnemo.models.mlp.fully_connected", "FullyConnected"
    )
    resolved = {
        "in_features": dataset_info["in_features"],
        "out_features": dataset_info["out_features"],
        "layer_size": int(model_cfg.get("layer_size", 128)),
        "num_layers": int(model_cfg.get("num_layers", 6)),
        "activation_fn": model_cfg.get("activation_fn", "silu"),
        "skip_connections": bool(model_cfg.get("skip_connections", True)),
        "adaptive_activations": bool(model_cfg.get("adaptive_activations", False)),
        "weight_norm": bool(model_cfg.get("weight_norm", False)),
    }
    model = fc_cls(**resolved)

    dropout = float(model_cfg.get("dropout", 0.0))
    if dropout > 0:
        model = _DropoutWrapper(model, dropout)
        resolved["dropout"] = dropout

    model._resolved_model_params = dict(resolved)
    return model


register_model("mlp", build_fn=build, adapter="pointwise")
