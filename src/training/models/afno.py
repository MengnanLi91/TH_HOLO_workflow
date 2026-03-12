"""Built-in AFNO model definition."""

from training import import_physicsnemo_attr
from training.models import register_model


def build(model_cfg: dict, dataset_info: dict):
    afno_cls = import_physicsnemo_attr("physicsnemo.models.afno.afno", "AFNO")
    resolved = {
        "inp_shape": list(dataset_info["spatial_shape"]),
        "in_channels": dataset_info["in_channels"],
        "out_channels": dataset_info["out_channels"],
        "patch_size": list(model_cfg.get("patch_size", [16, 16])),
        "embed_dim": int(model_cfg.get("embed_dim", 256)),
        "depth": int(model_cfg.get("depth", 4)),
        "num_blocks": int(model_cfg.get("num_blocks", 16)),
        "sparsity_threshold": float(model_cfg.get("sparsity_threshold", 0.01)),
        "hard_thresholding_fraction": float(
            model_cfg.get("hard_thresholding_fraction", 1.0)
        ),
    }
    model = afno_cls(**resolved)
    model._resolved_model_params = dict(resolved)
    return model


register_model("afno", build_fn=build, adapter="grid")
