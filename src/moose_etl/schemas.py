"""Data schemas for MOOSE simulation ETL pipeline.

MooseRawData   -- raw data extracted directly from Exodus + CSV files.
MooseProcessedData -- normalized, graph-ready, grid-ready data for ML training.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MooseRawData:
    """Raw data extracted from a single MOOSE simulation run.

    coords       : [N, D]     node coordinates (D=2 for 2-D, D=3 for 3-D)
    connectivity : [E, K]     element→node connectivity (0-indexed, K nodes/element)
    field_names  : list[str]  ordered list of element solution field names
    fields       : [T, E, F]  element solution values (time, element, field)
    time_steps   : [T]        simulation time values
    probe_data   : dict       probe_name → [Np, C] CSV line probe arrays
    probe_columns: list[str]  column names shared by all CSV probes
    sim_name     : str        unique simulation identifier (stem of the .e file)
    """

    coords: np.ndarray
    connectivity: np.ndarray
    field_names: list[str]
    fields: np.ndarray
    time_steps: np.ndarray
    probe_data: dict[str, np.ndarray]
    probe_columns: list[str]
    sim_name: str


@dataclass
class NormStats:
    """Per-field normalization statistics."""

    mean: float
    std: float


@dataclass
class MooseProcessedData:
    """Processed, normalized data ready for PhysicsNeMo ML training.

    coords       : [N, D]        node coordinates
    connectivity : [E, K]        element→node connectivity (0-indexed)
    edge_src     : [M]           graph edge source node indices
    edge_dst     : [M]           graph edge destination node indices
    fields       : [T, E, F]     normalized element solution fields
    field_names  : list[str]     ordered field names matching last dim of `fields`
    norm_stats   : dict          field_name → NormStats(mean, std)
    probe_data   : dict          probe_name → [Np, C] CSV probe arrays (raw)
    probe_columns: list[str]     column names for probe arrays
    grid_fields  : [T, Nx, Ny, F] fields interpolated onto a regular grid
    grid_x       : [Nx]          x-coordinates of grid columns
    grid_y       : [Ny]          y-coordinates of grid rows
    time_steps   : [T]           simulation time values
    sim_name     : str           unique simulation identifier
    """

    coords: np.ndarray
    connectivity: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    fields: np.ndarray
    field_names: list[str]
    norm_stats: dict[str, NormStats]
    probe_data: dict[str, np.ndarray]
    probe_columns: list[str]
    grid_fields: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    time_steps: np.ndarray
    sim_name: str
