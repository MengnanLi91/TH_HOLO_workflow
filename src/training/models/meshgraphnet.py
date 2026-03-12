"""Built-in MeshGraphNet model definition."""

from training import _require_pyg, import_physicsnemo_attr
from training.models import register_model


def build(model_cfg: dict, dataset_info: dict):
    _require_pyg()
    mesh_graph_net_cls = import_physicsnemo_attr(
        "physicsnemo.models.meshgraphnet.meshgraphnet", "MeshGraphNet"
    )
    resolved = {
        "input_dim_nodes": dataset_info["in_channels"],
        "input_dim_edges": dataset_info["edge_dim"],
        "output_dim": dataset_info["out_channels"],
        "processor_size": int(model_cfg.get("processor_size", 15)),
        "hidden_dim_processor": int(model_cfg.get("hidden_dim_processor", 128)),
        "hidden_dim_node_encoder": int(model_cfg.get("hidden_dim_node_encoder", 128)),
        "num_layers_node_processor": int(
            model_cfg.get("num_layers_node_processor", 2)
        ),
        "num_layers_edge_processor": int(
            model_cfg.get("num_layers_edge_processor", 2)
        ),
    }
    model = mesh_graph_net_cls(**resolved)
    model._resolved_model_params = dict(resolved)
    return model


register_model("meshgraphnet", build_fn=build, adapter="graph")
