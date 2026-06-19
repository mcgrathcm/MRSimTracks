"""Tests for track_parallel: batched, multi-process tracking.

track_parallel partitions seeds across worker processes (each reloads the flow),
tracks each batch, and concatenates the results. Because batching permutes the
seeds, the output particle order differs from the input -- so equivalence with
serial tracking is checked order-independently over the set of trajectories.
Particles that never leave the domain integrate identically whether tracked
alone, in a batch, or all together, so a clean (no-reset) run must reproduce
serial tracking exactly.
"""

import numpy as np
import pytest

import mrsimtracks as pt
from diagnostics import deterministic_cell_center_seeds
from fixture_paths import ACTIVE_KEY, SMALL_FLOW

DT = 0.002
TMAX = 0.006            # 3 steps; short enough that interior seeds never recycle
N_PARTICLES = 64


def _sorted_rows(a):
    return a[np.lexsort(a.T[::-1])]


@pytest.fixture(scope="module")
def flow():
    return pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, pbar=False)


@pytest.fixture(scope="module")
def seeds(flow):
    return deterministic_cell_center_seeds(flow, N_PARTICLES)


@pytest.fixture(scope="module")
def inlet(flow):
    return np.ascontiguousarray(flow.active_mesh.points[:500])


@pytest.fixture(scope="module")
def serial_result(flow, seeds, inlet):
    return pt.track(flow, seeds=seeds, dt=DT, tmax=TMAX, inlet=inlet,
                    pbar=False, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def parallel_result(seeds, inlet):
    return pt.track_parallel(
        str(SMALL_FLOW), seeds, dt=DT, tmax=TMAX, inlet=inlet, n_workers=3,
        active_key=ACTIVE_KEY, pbar=False, rng=np.random.default_rng(0),
        return_metrics=True)


def test_track_parallel_tracks_every_particle(parallel_result, serial_result):
    result, _ = parallel_result
    assert result.positions.shape == serial_result.positions.shape
    assert result.positions.shape == (3, N_PARTICLES, 3)
    assert np.isfinite(result.positions).all()


def test_track_parallel_matches_serial_for_clean_particles(parallel_result, serial_result):
    result, _ = parallel_result
    # Guard the regime: no recycling, so the set of trajectories is well-defined.
    assert result.reset.sum() == 0
    assert serial_result.reset.sum() == 0

    par_final = _sorted_rows(np.asarray(result.positions[-1]))
    ser_final = _sorted_rows(np.asarray(serial_result.positions[-1]))
    np.testing.assert_allclose(par_final, ser_final, atol=1e-10)


def test_track_parallel_is_deterministic_with_seeded_rng(seeds, inlet):
    kw = dict(dt=DT, tmax=TMAX, inlet=inlet, n_workers=2, active_key=ACTIVE_KEY,
              pbar=False)
    a = pt.track_parallel(str(SMALL_FLOW), seeds, rng=np.random.default_rng(7), **kw)
    b = pt.track_parallel(str(SMALL_FLOW), seeds, rng=np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.reset, b.reset)


def test_track_parallel_aggregates_worker_metrics(parallel_result):
    _, metrics = parallel_result
    assert metrics["n_workers"] == 3
    assert len(metrics["workers"]) == 3
    assert metrics["particle_steps_per_s"] > 0
    # aggregate is the sum of the per-worker throughputs
    assert metrics["particle_steps_per_s"] == pytest.approx(
        sum(w["particle_steps_per_s"] for w in metrics["workers"]))


def test_track_parallel_rejects_invalid_arguments(seeds, inlet):
    with pytest.raises(ValueError, match="provide seeds"):
        pt.track_parallel(str(SMALL_FLOW), None, inlet=inlet, pbar=False)

    with pytest.raises(ValueError, match="n_workers must be"):
        pt.track_parallel(str(SMALL_FLOW), seeds, n_workers=0, inlet=inlet, pbar=False)

    # pass tmax + active_key so shape validation is reached without a flow load
    with pytest.raises(ValueError, match="seeds must have shape"):
        pt.track_parallel(str(SMALL_FLOW), np.zeros((4, 2)), tmax=TMAX, inlet=inlet,
                          active_key=ACTIVE_KEY, pbar=False)
