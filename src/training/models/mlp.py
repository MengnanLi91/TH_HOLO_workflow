"""Built-in FullyConnected MLP model definition."""

from training import import_physicsnemo_attr
from training.models import register_model


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
    model._resolved_model_params = dict(resolved)
    return model


register_model("mlp", build_fn=build, adapter="pointwise")
