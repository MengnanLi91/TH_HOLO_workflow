"""Tests for the 1D-conv profile pipeline (dataset, adapter, model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")


FEATURE_NAMES = [
    "log10_Re", "Dr", "Lr", "z_hat", "d_local_over_D",
    "V_local_over_V_bulk", "is_throat",
]
TARGET_NAMES = ["log_alpha_D"]
NUM_CASES = 6
ROWS_PER_CASE = 12


@pytest.fixture()
def profile_zarr_dir(tmp_path: Path) -> Path:
    """Synthetic Zarr stores with monotone z_hat per case (deliberately
    written out of order to verify the wrapper sorts on read).
    """
    rng = np.random.default_rng(7)
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    z_idx = FEATURE_NAMES.index("z_hat")
    for i in range(NUM_CASES):
        case_name = f"case_{i:03d}"
        store_path = out_dir / f"{case_name}.zarr"
        root = zarr.open(store=str(store_path), mode="w")

        features = rng.standard_normal(
            (ROWS_PER_CASE, len(FEATURE_NAMES))
        ).astype(np.float32)
        # Shuffle the z_hat column so the wrapper has work to do
        z_hat_sorted = np.linspace(0.0, 1.0, ROWS_PER_CASE, dtype=np.float32)
        perm = rng.permutation(ROWS_PER_CASE)
        features[:, z_idx] = z_hat_sorted[perm]

        targets = rng.standard_normal(
            (ROWS_PER_CASE, len(TARGET_NAMES))
        ).astype(np.float32)

        root.create_array("features", data=features, overwrite=True)
        root.create_array("targets", data=targets, overwrite=True)
        meta = root.require_group("metadata")
        meta.attrs["case_id"] = case_name
        meta.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        meta.attrs["target_names"] = json.dumps(TARGET_NAMES)

    return out_dir


# ---------------------------------------------------------------------------
# Phase 0 — adapter whitelist
# ---------------------------------------------------------------------------


class TestAdapterWhitelist:
    def test_register_model_accepts_profile(self) -> None:
        from training.models import MODEL_REGISTRY, register_model

        def _fake_build(model_cfg, dataset_info):  # noqa: ARG001
            return None

        register_model("__test_profile_only__", build_fn=_fake_build, adapter="profile")
        try:
            assert MODEL_REGISTRY["__test_profile_only__"].adapter == "profile"
        finally:
            del MODEL_REGISTRY["__test_profile_only__"]

    def test_register_model_rejects_unknown_adapter(self) -> None:
        from training.models import register_model

        with pytest.raises(ValueError, match="adapter"):
            register_model(
                "__test_bad_adapter__",
                build_fn=lambda *_: None,
                adapter="not_a_family",
            )

    def test_get_build_fn_validates_entrypoint_adapter(self) -> None:
        from training.models import get_build_fn_and_adapter

        with pytest.raises(ValueError, match="adapter"):
            get_build_fn_and_adapter(
                {"entrypoint": "training.models.mlp:build", "adapter": "bogus"}
            )

    def test_adapter_registry_includes_profile(self) -> None:
        from training.adapters import ADAPTER_REGISTRY, ProfileAdapter

        assert ADAPTER_REGISTRY["profile"] is ProfileAdapter


# ---------------------------------------------------------------------------
# Phase 1 — AlphaDProfileDataset
# ---------------------------------------------------------------------------


class TestProfileDataset:
    def test_per_case_item_shape(self, profile_zarr_dir: Path) -> None:
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["log10_Re", "Dr", "z_hat", "is_throat"],
            output_columns=TARGET_NAMES,
        )
        assert len(ds) == NUM_CASES
        x, y, w, ci = ds[0]
        assert x.shape == (4, ROWS_PER_CASE)
        assert y.shape == (1, ROWS_PER_CASE)
        assert w.shape == (1, ROWS_PER_CASE)
        assert ci.dtype == torch.long
        assert int(ci.item()) == 0

    def test_stations_sorted_by_z_hat(self, profile_zarr_dir: Path) -> None:
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["z_hat"],
            output_columns=TARGET_NAMES,
        )
        x, _, _, _ = ds[0]
        z = x[0].numpy()
        assert np.all(np.diff(z) >= -1e-6)

    def test_subset_by_case_indices_isolates_inner(
        self, profile_zarr_dir: Path
    ) -> None:
        """The subset wrapper must wrap a real subset of the inner — sharing
        the parent's _inner would silently include all cases in the Phase 7
        delta_p loss path, which iterates ds._case_ids_unique.
        """
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["log10_Re", "Dr"],
            output_columns=TARGET_NAMES,
        )
        sub = ds.subset_by_case_indices([0, 2, 4])

        assert len(sub) == 3
        assert len(sub._case_ids_unique) == 3
        assert sub._case_ids_unique == ["case_000", "case_002", "case_004"]
        # Flat row state must reflect the subset, not the parent
        assert sub._x.shape[0] == 3 * ROWS_PER_CASE
        assert sub._row_case_idx.max() == 2
        # Inner must not be the same instance as the parent's
        assert sub._inner is not ds._inner

    def test_runner_dispatch_contract(self, profile_zarr_dir: Path) -> None:
        """Pin the contract used by runner.py to dispatch run_meta writing.

        The runner picks the tabular vs grid data_meta branch by checking
        ``hasattr(dataset, "input_columns")``. The profile dataset must
        satisfy that capability check so its checkpoints reload through
        evaluate.py.
        """
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["log10_Re", "Dr"],
            output_columns=TARGET_NAMES,
            normalize=True,
        )
        assert hasattr(ds, "input_columns")
        assert hasattr(ds, "output_columns")
        assert hasattr(ds, "sim_names")
        # These are what the pointwise/profile run_meta branch records.
        assert ds.normalize is True
        assert ds.target_residual_baseline is False
        assert ds.local_velocity_normalization is False
        assert ds.exclude_cases == []

    def test_subset_preserves_residual_baseline_alignment(
        self, profile_zarr_dir: Path
    ) -> None:
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["log10_Re", "Dr"],
            output_columns=TARGET_NAMES,
            target_residual_baseline=False,
        )
        sub = ds.subset_by_case_indices([1, 3])
        # _baseline_encoded is None when residual baseline is off; delegation
        # must still work without raising AttributeError
        assert sub._baseline_encoded is None
        assert ds._baseline_encoded is None


# ---------------------------------------------------------------------------
# Phase 2 — ProfileAdapter
# ---------------------------------------------------------------------------


class TestProfileAdapter:
    def test_dataset_info_reports_n_stations(self, profile_zarr_dir: Path) -> None:
        from training.adapters import ProfileAdapter
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["log10_Re", "Dr", "z_hat"],
            output_columns=TARGET_NAMES,
        )
        info = ProfileAdapter().dataset_info(ds)
        assert info == {
            "in_channels": 3,
            "out_channels": 1,
            "n_stations": ROWS_PER_CASE,
        }

    def test_forward_train_emits_4_tuple(self, profile_zarr_dir: Path) -> None:
        from training.adapters import ProfileAdapter
        from cases.alpha_d.datasets.profile import AlphaDProfileDataset

        ds = AlphaDProfileDataset(
            zarr_dir=profile_zarr_dir,
            input_columns=["log10_Re", "Dr"],
            output_columns=TARGET_NAMES,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        adapter = ProfileAdapter()
        device = torch.device("cpu")
        raw_batch = next(iter(loader))
        prepared = adapter.build_batch(raw_batch, device)

        # An identity-on-stations stub model just returns the right output shape
        class _Stub(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1, x.shape[-1])

        result = adapter.forward_train(_Stub(), prepared)
        assert len(result) == 4
        pred, y, w, cidx = result
        assert pred.shape == (2, 1, ROWS_PER_CASE)
        assert y.shape == (2, 1, ROWS_PER_CASE)
        assert w.shape == (2, 1, ROWS_PER_CASE)
        assert cidx.shape == (2,)

    def test_accumulate_metrics_uses_b_times_s(
        self, profile_zarr_dir: Path
    ) -> None:
        """RMSE comparable to MLP requires n_samples = batch_cases * stations."""
        from training.adapters import ProfileAdapter

        adapter = ProfileAdapter()
        pred = torch.randn(3, 1, ROWS_PER_CASE)
        target = torch.randn(3, 1, ROWS_PER_CASE)
        field_se, n = adapter.accumulate_metrics(None, pred, target)
        assert field_se.shape == (1,)
        assert n == 3 * ROWS_PER_CASE
        # field_se must be the element-wise sum, not the batch-mean
        expected = ((pred - target) ** 2).sum(dim=(0, 2))
        assert torch.allclose(field_se, expected)

    def test_relative_l2_broadcasts_with_profile_weight(
        self, profile_zarr_dir: Path
    ) -> None:
        from training.losses import relative_l2_loss

        pred = torch.randn(2, 1, ROWS_PER_CASE)
        target = torch.randn(2, 1, ROWS_PER_CASE)
        weight = torch.ones(2, 1, ROWS_PER_CASE) * 2.0
        # Should produce a finite scalar without shape errors
        loss = relative_l2_loss(pred, target, weight)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Phase 3 — Conv1D model
# ---------------------------------------------------------------------------


class TestConv1DModel:
    @pytest.fixture(autouse=True)
    def _need_physicsnemo(self):
        pytest.importorskip("physicsnemo")


    def _build(self, in_c=4, out_c=1, hidden=16, num_blocks=2):
        from training.models.conv1d_profile import build

        return build(
            {
                "hidden_channels": hidden,
                "num_blocks": num_blocks,
                "kernel_size": 3,
                "dilations": [1, 2],
                "dropout": 0.0,
            },
            {"in_channels": in_c, "out_channels": out_c, "n_stations": 10},
        )

    def test_registered_with_profile_adapter(self) -> None:
        from training.models import MODEL_REGISTRY

        entry = MODEL_REGISTRY["conv1d_profile"]
        assert entry.adapter == "profile"

    def test_forward_shape(self) -> None:
        model = self._build(in_c=4, out_c=1)
        model.eval()
        x = torch.randn(2, 4, 10)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1, 10)

    def test_resolved_params_attached(self) -> None:
        model = self._build(hidden=32, num_blocks=3)
        assert model._resolved_model_params["hidden"] == 32
        assert model._resolved_model_params["num_blocks"] == 3
        assert model._resolved_model_params["dilations"] == [1, 2]

    def test_from_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """Module-scope class is the load-bearing fix: from_checkpoint
        re-imports the class via importlib + getattr(module, name).
        """
        import physicsnemo

        from training.models.conv1d_profile import AlphaDConv1D

        model = self._build(in_c=4, out_c=1, hidden=16, num_blocks=2)
        model.eval()
        x = torch.randn(2, 4, 10)
        with torch.no_grad():
            ref = model(x)

        ckpt_path = tmp_path / "model.mdlus"
        model.save(str(ckpt_path))

        reloaded = physicsnemo.Module.from_checkpoint(str(ckpt_path))
        assert isinstance(reloaded, AlphaDConv1D)
        reloaded.eval()
        with torch.no_grad():
            out = reloaded(x)
        assert torch.allclose(out, ref, atol=1e-6)
