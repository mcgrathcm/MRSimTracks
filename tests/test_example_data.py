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
