from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TrajectoryStats:
    median_displacement: float
    moving_fraction: float
    reset_fraction: float
    finite: bool


@dataclass(frozen=True)
class DensityStats:
    median_nn_ratio: float
    occupied_bin_ratio: float
    normalized_l1: float


def deterministic_cell_center_seeds(flow, n_particles=512):
    """Return deterministic in-domain seeds from cell centers."""
    centers = flow.active_mesh.cell_centers().points
    idx = np.linspace(0, len(centers) - 1, n_particles, dtype=int)
    return np.ascontiguousarray(centers[idx])


def trajectory_stats(result, initial_positions, movement_eps=1e-4):
    final = result.positions[-1]
    displacement = np.linalg.norm(final - initial_positions, axis=1)
    return TrajectoryStats(
        median_displacement=float(np.median(displacement)),
        moving_fraction=float(np.mean(displacement > movement_eps)),
        reset_fraction=float(np.mean(result.reset)),
        finite=bool(np.isfinite(result.positions).all()),
    )


def density_stats(initial_positions, final_positions, bounds, bins=(6, 6, 6)):
    nn0 = cKDTree(initial_positions).query(initial_positions, k=2)[0][:, 1]
    nn1 = cKDTree(final_positions).query(final_positions, k=2)[0][:, 1]

    hist0, _ = np.histogramdd(initial_positions, bins=bins, range=bounds)
    hist1, _ = np.histogramdd(final_positions, bins=bins, range=bounds)

    occupied0 = np.count_nonzero(hist0)
    occupied1 = np.count_nonzero(hist1)
    p0 = hist0 / hist0.sum()
    p1 = hist1 / hist1.sum()

    return DensityStats(
        median_nn_ratio=float(np.median(nn1) / np.median(nn0)),
        occupied_bin_ratio=float(occupied1 / occupied0),
        normalized_l1=float(np.abs(p1 - p0).sum()),
    )
