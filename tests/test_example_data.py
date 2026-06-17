from pathlib import Path

import numpy as np
import pytest

import particle_tracking as pt


DATA = Path(__file__).parent / "data"
SMALL_FLOW = DATA / "CFD_velocity_00190_00210.vtu"
INLET = DATA / "Inlet.vtp"
OUTLET = DATA / "Outlet.vtp"
FULL_FLOW = Path(__file__).parents[1] / "example" / "CFD_velocity.vtu"
FULL_INLET = Path(__file__).parents[1] / "example" / "Inlet.vtp"
FULL_OUTLET = Path(__file__).parents[1] / "example" / "Outlet.vtp"


def test_small_fixture_loads_expected_time_window():
    flow = pt.load_flow(SMALL_FLOW, active_key="Velocity", pbar=False)

    assert flow.times[0] == 190
    assert flow.times[-1] == 210
    assert len(flow.times) == 11
    np.testing.assert_allclose(flow.times_shift_s[[0, -1]], [0.0, 0.02])


def test_small_fixture_tracks_with_boundary_reseeding():
    flow = pt.load_flow(SMALL_FLOW, active_key="Velocity", pbar=False)
    reseeder = pt.BoundaryReseeder([INLET, OUTLET], flow, dt=0.002)

    result = pt.track(
        flow,
        seeds=flow.active_mesh.points[:10],
        dt=0.002,
        tmax=0.004,
        reseeder=reseeder,
        pbar=False,
    )

    assert result.positions.shape == (2, 10, 3)
    assert result.reset.shape == (2, 10)
    assert np.isfinite(result.positions).all()


def test_boundary_reseeder_flux_waveform_is_finite_and_balanced():
    flow = pt.load_flow(SMALL_FLOW, active_key="Velocity", pbar=False)
    reseeder = pt.BoundaryReseeder([INLET, OUTLET], flow, dt=0.002)

    times, flux = reseeder.flux_waveform()
    imbalance = np.abs(flux.sum(axis=1))
    total_flux = np.abs(flux).sum(axis=1)

    assert times.shape == (11,)
    assert flux.shape == (11, 2)
    assert np.isfinite(times).all()
    assert np.isfinite(flux).all()
    assert np.max(imbalance / total_flux) < 0.01


def test_boundary_reseeder_is_repeatable_with_seeded_rng():
    flow = pt.load_flow(SMALL_FLOW, active_key="Velocity", pbar=False)
    reseeder1 = pt.BoundaryReseeder(
        [INLET, OUTLET],
        flow,
        rng=np.random.default_rng(1234),
        dt=0.002,
    )
    reseeder2 = pt.BoundaryReseeder(
        [INLET, OUTLET],
        flow,
        rng=np.random.default_rng(1234),
        dt=0.002,
    )

    seeds1 = reseeder1.reseed(100, t=0.006)
    seeds2 = reseeder2.reseed(100, t=0.006)

    np.testing.assert_allclose(seeds1, seeds2)
    assert np.isfinite(seeds1).all()


def test_boundary_reseeder_accepts_string_paths():
    flow = pt.load_flow(SMALL_FLOW, active_key="Velocity", pbar=False)
    reseeder = pt.BoundaryReseeder([str(INLET), str(OUTLET)], flow, dt=0.002)

    seeds = reseeder.reseed(8, t=0.0)

    assert seeds.shape == (8, 3)
    assert np.isfinite(seeds).all()


@pytest.mark.large
def test_full_lfs_fixture_tracks_with_boundary_reseeding():
    if FULL_FLOW.stat().st_size < 100_000_000:
        pytest.skip("full example VTU was not fetched through Git LFS")

    flow = pt.load_flow(FULL_FLOW, active_key="Velocity", pbar=False)
    reseeder = pt.BoundaryReseeder([FULL_INLET, FULL_OUTLET], flow, dt=0.002)

    result = pt.track(
        flow,
        seeds=flow.active_mesh.points[:10],
        dt=0.002,
        tmax=0.004,
        reseeder=reseeder,
        pbar=False,
    )

    assert result.positions.shape == (2, 10, 3)
    assert np.isfinite(result.positions).all()
