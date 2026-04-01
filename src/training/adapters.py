"""Adapter layer to unify grid and graph model families."""

from collections.abc import Callable

import torch

from training import _require_pyg
from training.datasets import GraphPairDataset, GridPairDataset, parse_field_list


class ModelAdapter:
    """Interface for model/data-family-specific behavior."""

    family: str

    def build_dataset(self, data_cfg: dict):
        raise NotImplementedError

    def dataset_info(self, dataset) -> dict:
        raise NotImplementedError

    def build_batch(self, raw_batch, device: torch.device):
        raise NotImplementedError

    def forward_train(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_eval(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def collate_fn(self) -> Callable | None:
        return None

    def accumulate_metrics(
        self,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        raise NotImplementedError


class GridAdapter(ModelAdapter):
    family = "grid"

    def build_dataset(self, data_cfg: dict) -> GridPairDataset:
        return GridPairDataset(
            zarr_dir=data_cfg["zarr_dir"],
            input_fields=parse_field_list(data_cfg.get("input_fields")),
            output_fields=parse_field_list(data_cfg.get("output_fields")),
            input_time_idx=int(data_cfg.get("input_time_idx", 0)),
            target_time_idx=int(data_cfg.get("target_time_idx", -1)),
        )

    def dataset_info(self, dataset: GridPairDataset) -> dict:
        return {
            "in_channels": len(dataset.input_indices),
            "out_channels": len(dataset.output_indices),
            "spatial_shape": dataset.spatial_shape,
        }

    def build_batch(self, raw_batch, device: torch.device):
        pin_memory = device.type == "cuda"
        x, y = raw_batch
        return (
            x.to(device, non_blocking=pin_memory),
            y.to(device, non_blocking=pin_memory),
        )

    def forward_train(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        pred = model(x)
        return pred, y

    def forward_eval(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        pred = model(x)
        return pred, y

    def accumulate_metrics(
        self,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target shape {tuple(target.shape)}."
            )
        if pred.ndim != 4:
            raise ValueError(
                f"Grid predictions must have shape [B, C, Nx, Ny], got {tuple(pred.shape)}."
            )

        squared = (pred - target) ** 2
        field_se = squared.mean(dim=(2, 3)).sum(dim=0)
        num_samples = int(pred.shape[0])
        return field_se, num_samples


class GraphAdapter(ModelAdapter):
    family = "graph"

    def __init__(self):
        _require_pyg()
        from torch_geometric.data import Batch

        self._pyg_batch_cls = Batch
        self._last_batch = None

    def build_dataset(self, data_cfg: dict) -> GraphPairDataset:
        return GraphPairDataset(
            zarr_dir=data_cfg["zarr_dir"],
            input_fields=parse_field_list(data_cfg.get("input_fields")),
            output_fields=parse_field_list(data_cfg.get("output_fields")),
            input_time_idx=int(data_cfg.get("input_time_idx", 0)),
            target_time_idx=int(data_cfg.get("target_time_idx", -1)),
        )

    def dataset_info(self, dataset: GraphPairDataset) -> dict:
        return {
            "in_channels": len(dataset.input_indices),
            "out_channels": len(dataset.output_indices),
            "edge_dim": dataset.edge_dim,
        }

    def collate_fn(self) -> Callable | None:
        def _pyg_collate(items):
            return self._pyg_batch_cls.from_data_list(items)

        return _pyg_collate

    def build_batch(self, raw_batch, device: torch.device):
        pin_memory = device.type == "cuda"
        batch = raw_batch.to(device, non_blocking=pin_memory)
        self._last_batch = batch
        return batch

    def forward_train(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        pred = model(batch.x, batch.edge_attr, batch)
        return pred, batch.y

    def forward_eval(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        pred = model(batch.x, batch.edge_attr, batch)
        return pred, batch.y

    def accumulate_metrics(
        self,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch_obj = self._last_batch
        if batch_obj is None:
            raise RuntimeError("GraphAdapter has no batch metadata for metric accumulation.")

        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match target shape {tuple(target.shape)}."
            )
        if pred.ndim != 2:
            raise ValueError(
                f"Graph predictions must have shape [total_nodes, C], got {tuple(pred.shape)}."
            )
        if not hasattr(batch_obj, "batch"):
            raise ValueError("Graph batch is missing 'batch' graph index tensor.")

        graph_ids = batch_obj.batch
        num_graphs = int(batch_obj.num_graphs)
        squared = (pred - target) ** 2

        graph_sum = torch.zeros(
            (num_graphs, squared.shape[1]), dtype=squared.dtype, device=squared.device
        )
        graph_count = torch.zeros(num_graphs, dtype=squared.dtype, device=squared.device)

        graph_sum.index_add_(0, graph_ids, squared)
        graph_count.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=squared.dtype))
        graph_mean = graph_sum / graph_count.clamp_min(1.0).unsqueeze(-1)

        field_se = graph_mean.sum(dim=0)
        return field_se, num_graphs


class PointwiseAdapter(ModelAdapter):
    """Adapter for tabular / pointwise MLP models.

    Expects datasets producing ``(x, y)`` tuples of shape
    ``(D_in,)`` and ``(D_out,)`` which the default collate batches into
    ``(B, D_in)`` and ``(B, D_out)``.
    """

    family = "pointwise"

    def build_dataset(self, data_cfg: dict):
        from training.datasets_tabular import TabularPairDataset

        return TabularPairDataset(
            zarr_dir=data_cfg["zarr_dir"],
            input_columns=parse_field_list(data_cfg.get("input_columns")),
            output_columns=parse_field_list(data_cfg.get("output_columns")),
        )

    def dataset_info(self, dataset) -> dict:
        return {
            "in_features": dataset.in_features,
            "out_features": dataset.out_features,
        }

    def build_batch(self, raw_batch, device: torch.device):
        pin_memory = device.type == "cuda"
        x, y = raw_batch
        return (
            x.to(device, non_blocking=pin_memory),
            y.to(device, non_blocking=pin_memory),
        )

    def forward_train(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        pred = model(x)
        return pred, y

    def forward_eval(self, model, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        pred = model(x)
        return pred, y

    def accumulate_metrics(
        self,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape)} does not match "
                f"target shape {tuple(target.shape)}."
            )
        squared = (pred - target) ** 2
        field_se = squared.sum(dim=0)
        num_samples = int(pred.shape[0])
        return field_se, num_samples


ADAPTER_REGISTRY = {
    "grid": GridAdapter,
    "graph": GraphAdapter,
    "pointwise": PointwiseAdapter,
}


def get_adapter(name: str) -> ModelAdapter:
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown adapter '{name}'. Available adapters: {available}")
    return ADAPTER_REGISTRY[name]()
