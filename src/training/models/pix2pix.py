"""Built-in Pix2Pix model definition."""

from training import import_physicsnemo_attr
from training.models import register_model


def build(model_cfg: dict, dataset_info: dict):
    pix2pix_cls = import_physicsnemo_attr(
        "physicsnemo.models.pix2pix.pix2pix", "Pix2Pix"
    )
    resolved = {
        "in_channels": dataset_info["in_channels"],
        "out_channels": dataset_info["out_channels"],
        "dimension": int(model_cfg.get("dimension", 2)),
        "conv_layer_size": int(model_cfg.get("conv_layer_size", 64)),
        "n_downsampling": int(model_cfg.get("n_downsampling", 3)),
        "n_upsampling": int(model_cfg.get("n_upsampling", 3)),
        "n_blocks": int(model_cfg.get("n_blocks", 3)),
    }
    model = pix2pix_cls(**resolved)
    model._resolved_model_params = dict(resolved)
    return model


register_model("pix2pix", build_fn=build, adapter="grid")
