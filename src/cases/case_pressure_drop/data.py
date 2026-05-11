"""Case-level data loading for scalar pressure-drop regression."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_FEATURES: tuple[str, ...] = (
    "Re",
    "log10_Re",
    "Dr",
    "Lr",
    "inv_Dr",
    "inv_Lr",
    "Re_times_Dr",
    "Re_times_Lr",
    "Dr_times_Lr",
)


def _require_positive(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive for all cases.")
    return arr


def _load_metadata_attrs(store_path: Path) -> dict[str, Any]:
    """Read per-case metadata attributes from a processed zarr store."""
    json_path = store_path / "metadata" / "zarr.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        attrs = payload.get("attributes")
        if not isinstance(attrs, dict):
            raise ValueError(f"Malformed metadata JSON: {json_path}")
        return attrs

    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find {json_path} and zarr is not installed for fallback loading."
        ) from exc

    root = zarr.open(store=str(store_path), mode="r")
    meta = root["metadata"]
    return dict(meta.attrs)


@dataclass(frozen=True)
class CasePressureDropDataset:
    """One scalar sample per simulation case."""

    zarr_dir: Path
    sim_names: list[str]
    Re: np.ndarray
    Dr: np.ndarray
    Lr: np.ndarray
    delta_p_case: np.ndarray

    @classmethod
    def from_zarr_dir(
        cls,
        zarr_dir: str | Path,
        *,
        exclude_cases: list[str] | None = None,
        min_Dr: float | None = None,
    ) -> "CasePressureDropDataset":
        zarr_dir = Path(zarr_dir).expanduser().resolve()
        sim_paths = sorted(zarr_dir.glob("*.zarr"))
        if not sim_paths:
            raise FileNotFoundError(f"No .zarr stores found in {zarr_dir}")

        exclude_set = set(exclude_cases or [])
        sim_names: list[str] = []
        re_vals: list[float] = []
        dr_vals: list[float] = []
        lr_vals: list[float] = []
        dp_vals: list[float] = []

        for store_path in sim_paths:
            attrs = _load_metadata_attrs(store_path)
            case_id = str(attrs.get("case_id", store_path.stem))
            if case_id in exclude_set or store_path.stem in exclude_set:
                continue

            try:
                re_v = float(attrs["Re"])
                dr_v = float(attrs["Dr"])
                lr_v = float(attrs["Lr"])
                dp_v = float(attrs["delta_p_case"])
            except KeyError as exc:
                raise KeyError(
                    f"Missing required metadata attribute {exc!s} in {store_path}"
                ) from exc

            if min_Dr is not None and dr_v < min_Dr:
                continue

            re_vals.append(re_v)
            dr_vals.append(dr_v)
            lr_vals.append(lr_v)
            dp_vals.append(dp_v)
            sim_names.append(case_id)

        if not sim_names:
            raise ValueError("All cases were filtered out; no training data remains.")

        return cls(
            zarr_dir=zarr_dir,
            sim_names=sim_names,
            Re=_require_positive(np.asarray(re_vals, dtype=np.float64), "Re"),
            Dr=_require_positive(np.asarray(dr_vals, dtype=np.float64), "Dr"),
            Lr=_require_positive(np.asarray(lr_vals, dtype=np.float64), "Lr"),
            delta_p_case=_require_positive(
                np.asarray(dp_vals, dtype=np.float64),
                "delta_p_case",
            ),
        )

    def __len__(self) -> int:
        return len(self.sim_names)

    def subset_by_case_indices(self, indices: list[int]) -> "CasePressureDropDataset":
        idx = np.asarray(indices, dtype=np.int64)
        return CasePressureDropDataset(
            zarr_dir=self.zarr_dir,
            sim_names=[self.sim_names[i] for i in idx.tolist()],
            Re=self.Re[idx].copy(),
            Dr=self.Dr[idx].copy(),
            Lr=self.Lr[idx].copy(),
            delta_p_case=self.delta_p_case[idx].copy(),
        )

    def subset_by_case_names(self, names: list[str]) -> "CasePressureDropDataset":
        name_to_idx = {name: idx for idx, name in enumerate(self.sim_names)}
        missing = [name for name in names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Unknown case names requested: {missing}")
        return self.subset_by_case_indices([name_to_idx[name] for name in names])

    def build_feature_matrix(
        self,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ) -> np.ndarray:
        requested = list(feature_names or CANDIDATE_FEATURES)
        available = self._feature_map()
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(f"Unknown candidate feature(s): {missing}")
        return np.column_stack([available[name] for name in requested]).astype(np.float64)

    def groups(self) -> np.ndarray:
        return np.arange(len(self.sim_names), dtype=np.int32)

    def target_log1p(self) -> np.ndarray:
        return np.log1p(self.delta_p_case).astype(np.float64)

    def _feature_map(self) -> dict[str, np.ndarray]:
        return {
            "Re": self.Re,
            "log10_Re": np.log10(self.Re),
            "Dr": self.Dr,
            "Lr": self.Lr,
            "inv_Dr": 1.0 / self.Dr,
            "inv_Lr": 1.0 / self.Lr,
            "Re_times_Dr": self.Re * self.Dr,
            "Re_times_Lr": self.Re * self.Lr,
            "Dr_times_Lr": self.Dr * self.Lr,
        }
