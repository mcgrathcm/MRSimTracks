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


def motion_stats(result, initial_positions, dt):
    """JSON-safe motion statistics for regression artifacts."""
    steps = np.diff(np.concatenate([initial_positions[None], result.positions], axis=0),
                    axis=0)
    step_distance = np.linalg.norm(steps, axis=2)
    speed = step_distance / dt
    path_length = step_distance.sum(axis=0)
    displacement = np.linalg.norm(result.positions[-1] - initial_positions, axis=1)
    reset_count = result.reset.sum(axis=0)

    return {
        "speed_mean": float(speed.mean()),
        "speed_median": float(np.median(speed)),
        "speed_p95": float(np.percentile(speed, 95)),
        "speed_max": float(speed.max()),
        "path_length_mean": float(path_length.mean()),
        "path_length_median": float(np.median(path_length)),
        "path_length_p95": float(np.percentile(path_length, 95)),
        "displacement_mean": float(displacement.mean()),
        "displacement_median": float(np.median(displacement)),
        "displacement_p95": float(np.percentile(displacement, 95)),
        "reset_count_mean": float(reset_count.mean()),
        "reset_count_p95": float(np.percentile(reset_count, 95)),
        "reset_count_max": float(reset_count.max()),
    }


def subset_particle_stats(result, initial_positions, dt, indices):
    """Per-particle trajectory summaries for a stable subset."""
    steps = np.diff(np.concatenate([initial_positions[None], result.positions], axis=0),
                    axis=0)
    step_distance = np.linalg.norm(steps, axis=2)
    out = []
    for i in indices:
        path_length = step_distance[:, i].sum()
        displacement = np.linalg.norm(result.positions[-1, i] - initial_positions[i])
        out.append({
            "particle": int(i),
            "path_length": float(path_length),
            "displacement": float(displacement),
            "mean_speed": float(path_length / (result.n_steps * dt)),
            "max_step_speed": float(step_distance[:, i].max() / dt),
            "reset_count": int(result.reset[:, i].sum()),
            "final_position": [float(v) for v in result.positions[-1, i]],
        })
    return out
