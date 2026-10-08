import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt
import mrsimtracks.io as mt_io
import mrsimtracks.sampler as mt_sampler


def _hex():
    return pv.UnstructuredGrid(
        np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]),
        np.array([pv.CellType.HEXAHEDRON], np.uint8),
        np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=float),
    )


def _steady_flow(tmp_path, key="Velocity_00000"):
    mesh = _hex()
    mesh.point_data[key] = np.tile([1.0, 0.0, 0.0], (mesh.n_points, 1))
    path = tmp_path / "steady.vtu"
    mesh.save(path)
    return mt.load_flow(path, active_key="Velocity", conform_mesh=False)


def test_native_hex_interpolation_and_locator_reuse(tmp_path, monkeypatch):
    mesh = _hex()
    x, y, z = mesh.points.T
    # Cross terms distinguish native Q1 interpolation from split-tet P1.
    field = np.column_stack((x * y, y * z, x * z))
    mesh.point_data["Velocity_00000"] = field
    mesh.point_data["Velocity_01000"] = 3 * field
    path = tmp_path / "hex.vtu"
    mesh.save(path)

    def forbidden(*args, **kwargs):
        pytest.fail("native static sampling must not tetrahedralize or use the walker")

    monkeypatch.setattr(mt_io, "_TetSampler", forbidden)
    monkeypatch.setattr(pv.UnstructuredGrid, "triangulate", forbidden)
    if mt_sampler._HAVE_NUMBA:
        monkeypatch.setattr(mt_sampler, "_walk_interp_kernel", forbidden)
    flow = mt.load_flow(path, active_key="Velocity", conform_mesh=False)
    sampler = flow._sampler
    assert isinstance(sampler, mt_sampler._VTKSampler)
    np.testing.assert_array_equal(flow.active_mesh.cells, mesh.cells)
    np.testing.assert_array_equal(flow.active_mesh.celltypes, mesh.celltypes)
    assert flow.locator.GetUseExistingSearchStructure()
    assert sampler._probe.GetCellLocatorPrototype() is None
    assert sampler._probe.GetFindCellStrategy().GetCellLocator() is flow.locator

    points = np.array([[0.2, 0.4, 0.6], [0.7, 0.3, 0.5], [2, 2, 2]])
    expected = np.column_stack((points[:, 0] * points[:, 1],
                                points[:, 1] * points[:, 2],
                                points[:, 0] * points[:, 2]))
    for time, scale in [(0.0, 1), (0.25, 1.5), (0.5, 2), (1.25, 1.5)]:
        velocity, valid, cells = flow.sample_v(points, time, guess=np.zeros(3, int))
        np.testing.assert_allclose(velocity[:2], scale * expected[:2])
        np.testing.assert_array_equal(velocity[2], 0)
        assert valid.tolist() == [True, True, False]
        assert cells is None
        assert flow._frame_runtime(0).sampler is sampler
        assert sampler._probe.GetFindCellStrategy().GetCellLocator() is flow.locator
    assert sampler.locate(points).tolist() == [0, 0, -1]
    assert len(flow._runtime_cache) == 1


@pytest.mark.parametrize("key", ["Velocity_00000", "Velocity"])
def test_single_frame_is_steady_and_requires_tracking_duration(tmp_path, key):
    flow = _steady_flow(tmp_path, key)
    assert len(flow.times) == 1
    assert flow.tmax == np.inf
    point = np.array([[0.5, 0.5, 0.5]])
    for time in [0.0, 0.5, 123.0]:
        velocity, valid, cells = flow.sample_v(point, time)
        np.testing.assert_allclose(velocity, [[1, 0, 0]])
        assert valid.all()
        assert cells is None
        np.testing.assert_allclose(flow.get_mesh(time)["Velocity"], flow._frame_vel(0))
    with pytest.raises(ValueError, match="finite tmax"):
        mt.track(flow, seeds=point, inlet=point, pbar=False)


def test_native_rk4_tracking_and_outside_reset(tmp_path):
    flow = _steady_flow(tmp_path)
    seeds = np.array([[0.25, 0.5, 0.5], [0.95, 0.5, 0.5]])
    inlet = np.array([[0.1, 0.5, 0.5]])
    result = mt.track(flow, seeds=seeds, inlet=inlet, dt=0.2, tmax=0.4, pbar=False)
    np.testing.assert_allclose(result.positions[:, 0],
                               [[0.25, 0.5, 0.5], [0.45, 0.5, 0.5], [0.65, 0.5, 0.5]])
    np.testing.assert_allclose(result.positions[1:, 1],
                               [[0.1, 0.5, 0.5], [0.3, 0.5, 0.5]])
    assert result.reset.tolist() == [[False, False], [False, True], [False, False]]


def test_native_tracking_uses_time_varying_field_on_static_geometry(tmp_path):
    mesh = _hex()
    mesh.point_data["Velocity_00000"] = np.tile([1.0, 0, 0], (8, 1))
    mesh.point_data["Velocity_01000"] = np.tile([3.0, 0, 0], (8, 1))
    path = tmp_path / "unsteady.vtu"
    mesh.save(path)
    flow = mt.load_flow(path, active_key="Velocity", conform_mesh=False)
    seeds = np.array([[0.1, 0.5, 0.5]])
    result = mt.track(flow, seeds=seeds, inlet=seeds, dt=0.05, tmax=0.2, pbar=False)
    times = result.times
    np.testing.assert_allclose(result.positions[:, 0, 0], 0.1 + times + times**2)
    assert not result.reset.any()
    assert len(flow._runtime_cache) == 1


def test_native_static_cap_reseeding(tmp_path):
    flow = _steady_flow(tmp_path)
    caps = pv.PolyData(
        flow.active_mesh.points.copy(),
        np.array([4, 0, 3, 7, 4, 4, 1, 2, 6, 5]),
    )
    caps.cell_data["region_id"] = np.array([0, 1])
    reseeder = mt.BoundaryReseeder(
        caps, flow, inward_eps=0.05, dt=0.01, rng=np.random.default_rng(4)
    )
    np.testing.assert_allclose(reseeder.flux_waveform()[1], [[-1.0, 1.0]])
    points = reseeder.reseed(100, 123.0)
    assert np.all(flow._sampler.locate(points) >= 0)
    assert np.all((points[:, 0] > 0) & (points[:, 0] < 1))
    result = mt.track(
        flow, seeds=np.array([[0.95, 0.5, 0.5]]), reseeder=reseeder,
        dt=0.1, tmax=0.1, pbar=False,
    )
    assert result.reset[1, 0]
    assert flow._sampler.locate(result.positions[1])[0] >= 0


def test_wall_slip_still_rejects_native_non_tet_geometry(tmp_path):
    flow = _steady_flow(tmp_path)
    with pytest.raises(ValueError, match="all-tetrahedral"):
        mt.WallSlip(flow)
