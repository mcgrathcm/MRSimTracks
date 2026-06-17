from pathlib import Path

import numpy as np

import particle_tracking as pt
from diagnostics import (
    density_stats,
    deterministic_cell_center_seeds,
    trajectory_stats,
)


DATA = Path(__file__).parent / "data"
SMALL_FLOW = DATA / "CFD_velocity_00190_00210.vtu"
INLET = DATA / "Inlet.vtp"
OUTLET = DATA / "Outlet.vtp"
RNG_SEED = 1234
DT = 0.002
TMAX = 0.012
N_PARTICLES = 512


def _track_small_case():
    flow = pt.load_flow(SMALL_FLOW, active_key="Velocity", pbar=False)
    seeds = deterministic_cell_center_seeds(flow, N_PARTICLES)
    reseeder = pt.BoundaryReseeder(
        [INLET, OUTLET],
        flow,
        rng=np.random.default_rng(RNG_SEED),
        dt=DT,
    )
    result = pt.track(
        flow,
        seeds=seeds,
        dt=DT,
        tmax=TMAX,
        reseeder=reseeder,
        pbar=False,
    )
    return flow, seeds, result


def test_small_case_particles_move_and_resets_stay_low():
    _, seeds, result = _track_small_case()

    stats = trajectory_stats(result, seeds)

    assert result.positions.shape == (6, N_PARTICLES, 3)
    assert stats.finite
    assert stats.moving_fraction > 0.95
    assert stats.median_displacement > 0.05
    assert stats.reset_fraction < 0.05


def test_small_case_density_stays_roughly_stable():
    flow, seeds, result = _track_small_case()
    bounds = np.array(flow.active_mesh.bounds).reshape(3, 2).tolist()

    stats = density_stats(seeds, result.positions[-1], bounds)

    assert 0.5 < stats.median_nn_ratio < 1.75
    assert stats.occupied_bin_ratio > 0.75
    assert stats.normalized_l1 < 0.75
