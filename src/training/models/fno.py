"""Built-in FNO model definition."""

from training import import_physicsnemo_attr
from training.models import register_model


def build(model_cfg: dict, dataset_info: dict):
    fno_cls = import_physicsnemo_attr("physicsnemo.models.fno.fno", "FNO")
    resolved = {
        "in_channels": dataset_info["in_channels"],
        "out_channels": dataset_info["out_channels"],
        "dimension": int(model_cfg.get("dimension", 2)),
        "latent_channels": int(model_cfg.get("latent_channels", 32)),
        "num_fno_layers": int(model_cfg.get("num_fno_layers", 4)),
        "num_fno_modes": model_cfg.get("num_fno_modes", 12),
        "padding": int(model_cfg.get("padding", 5)),
        "decoder_layers": int(model_cfg.get("decoder_layers", 1)),
        "decoder_layer_size": int(model_cfg.get("decoder_layer_size", 32)),
    }
    model = fno_cls(**resolved)
    model._resolved_model_params = dict(resolved)
    return model


register_model("fno", build_fn=build, adapter="grid")
