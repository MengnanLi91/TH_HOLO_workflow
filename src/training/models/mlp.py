"""Built-in FullyConnected MLP model definition.

Supports optional inter-layer dropout for improved regularisation in
deeper networks.  Dropout is applied between hidden layers during
training and disabled at eval time, so no extra state needs to be
persisted in the checkpoint.
"""

import torch

from training import import_physicsnemo_attr
from training.models import register_model


class _InterLayerDropout(torch.nn.Module):
    """Wraps a FullyConnected model, adding dropout between hidden layers.

    Unlike the previous input-only ``_DropoutWrapper``, this applies
    dropout *after each hidden layer's activation*, which is the
    standard placement for MLP regularisation.

    The saved checkpoint contains only the inner model so that
    evaluation works with the standard ``Module.from_checkpoint`` path.
    """

    def __init__(self, model: torch.nn.Module, dropout: float):
        super().__init__()
        self.model = model
        self.p = dropout
        n_layers = len(model.layers)
        self.drops = torch.nn.ModuleList(
            [torch.nn.Dropout(dropout) for _ in range(n_layers)]
        )

    def forward(self, x):
        x_skip = None
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            if self.training:
                x = self.drops[i](x)
            if self.model.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x
        return self.model.final_layer(x)

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
        model = _InterLayerDropout(model, dropout)
        resolved["dropout"] = dropout

    model._resolved_model_params = dict(resolved)
    return model


register_model("mlp", build_fn=build, adapter="pointwise")
