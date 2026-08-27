import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt
from mrsimtracks.core import _step_count
from mrsimtracks.sampler import _TetSampler


def _tetra(
    *, points=None, velocity=0.0, displacement=0.0, mesh_velocity=0.0
):
    if points is None:
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
    mesh = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA], np.uint8),
        points,
    )
    mesh.point_data["Velocity"] = np.full((len(points), 3), velocity)
    mesh.point_data["Displacement"] = np.full((len(points), 3), displacement)
    mesh.point_data["Mesh_velocity"] = np.full(
        (len(points), 3), mesh_velocity
    )
    return mesh


def test_step_count_preserves_almost_integral_cycle():
    assert _step_count(0.69, 0.001) == 690
    assert _step_count(0.69, 0.0005) == 1380
    assert _step_count(0.69, 0.004) == 172


def _save_series(tmp_path, meshes, times=None):
    times = np.arange(len(meshes), dtype=float) if times is None else times
    entries = []
    for index, (time, mesh) in enumerate(zip(times, meshes, strict=True)):
        path = tmp_path / f"flow_{index:02d}.vtu"
        mesh.save(path)
        entries.append((time, path))
    datasets = "\n".join(
        f'    <DataSet timestep="{time}" file="{path.name}"/>'
        for time, path in entries
    )
    pvd = tmp_path / "dataset.pvd"
    pvd.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1">\n'
        '  <Collection>\n'
        f'{datasets}\n'
        '  </Collection>\n'
        '</VTKFile>\n'
    )
    return pvd, [path for _, path in entries]


def _two_tetra_mesh(*, stretch=0.0):
    first = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    second = first + np.array([3, 0, 0])
    points = np.vstack((first, second))
    cells = np.array(
        [4, 0, 1, 2, 3, 4, 4, 5, 6, 7], dtype=np.int64
    )
    mesh = pv.UnstructuredGrid(
        cells,
        np.full(2, pv.CellType.TETRA, np.uint8),
        points,
    )
    velocity = np.zeros((8, 3))
    velocity[:4, 2] = 2.0
    velocity[4:, 2] = 1.0
    displacement = np.zeros((8, 3))
    displacement[:4, 0] = stretch * first[:, 0]
    mesh.point_data["Velocity"] = velocity
    mesh.point_data["Displacement"] = displacement
    mesh.point_data["Mesh_velocity"] = np.zeros((8, 3))
    return mesh


def _two_tetra_caps(mesh):
    node_ids = np.array([0, 2, 1, 4, 6, 5])
    caps = pv.PolyData(
        mesh.points[node_ids],
        np.array([3, 0, 1, 2, 3, 3, 4, 5]),
    )
    caps.point_data["volume_point_id"] = node_ids
    caps.cell_data["region_id"] = np.array([0, 1], np.int32)
    return caps


def test_load_ale_flow_loads_all_ale_fields_on_one_mesh(tmp_path):
    pvd, _ = _save_series(
        tmp_path,
        [
            _tetra(velocity=1, displacement=0.1),
            _tetra(velocity=2, displacement=0.2),
            _tetra(velocity=3, displacement=0.3),
        ],
        times=(2.76, 2.77, 2.78),
    )

    flow = mt.load_ale_flow(pvd, precision="f32")

    assert flow.n_frames == 3
    assert flow.data.geometry_mode == "static"
    assert len(flow.data.topologies) == 1
    assert len(flow.data.coordinates) == 1
    np.testing.assert_allclose(flow.times, [2.76, 2.77, 2.78])
    np.testing.assert_allclose(flow.times_shift_s, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(flow.velocity(2), 3)
    np.testing.assert_allclose(flow.displacement(2), 0.3)
    assert flow.velocity(0).dtype == np.float32
    assert set(flow.frame(1).point_data) == {
        "Velocity",
        "Displacement",
        "Mesh_velocity",
    }


def test_load_ale_flow_accepts_file_subset_and_dt(tmp_path):
    _, files = _save_series(
        tmp_path,
        [_tetra(velocity=1), _tetra(velocity=2), _tetra(velocity=3)],
    )
    renamed = []
    for step, path in zip((5520, 5540, 5560), files, strict=True):
        target = path.with_name(f"result_{step}.vtu")
        path.rename(target)
        renamed.append(target)

    flow = mt.load_ale_flow(renamed, dt=0.0005)

    np.testing.assert_allclose(flow.times, [5520, 5540, 5560])
    np.testing.assert_allclose(flow.times_shift_s, [0.0, 0.01, 0.02])


def test_load_ale_flow_rejects_changed_node_locations_in_any_frame(tmp_path):
    points = _tetra().points.copy()
    moved = points.copy()
    moved[0, 0] = 0.1
    pvd, _ = _save_series(
        tmp_path,
        [_tetra(points=points), _tetra(points=points), _tetra(points=moved)],
    )

    with pytest.raises(ValueError, match="static node locations"):
        mt.load_ale_flow(pvd)


def test_load_ale_flow_rejects_changed_topology_in_any_frame(tmp_path):
    changed = pv.UnstructuredGrid(
        np.array([4, 1, 0, 2, 3]),
        np.array([pv.CellType.TETRA], np.uint8),
        _tetra().points,
    )
    changed.point_data["Velocity"] = np.zeros((4, 3))
    changed.point_data["Displacement"] = np.zeros((4, 3))
    changed.point_data["Mesh_velocity"] = np.zeros((4, 3))
    pvd, _ = _save_series(tmp_path, [_tetra(), _tetra(), changed])

    with pytest.raises(ValueError, match="static topology"):
        mt.load_ale_flow(pvd)


def test_load_ale_flow_requires_displacement(tmp_path):
    first = _tetra()
    second = _tetra()
    del second.point_data["Displacement"]
    pvd, _ = _save_series(tmp_path, [first, second])

    with pytest.raises(ValueError, match="Displacement.*not found"):
        mt.load_ale_flow(pvd)


def test_load_ale_flow_requires_mesh_velocity(tmp_path):
    first = _tetra()
    second = _tetra()
    del second.point_data["Mesh_velocity"]
    pvd, _ = _save_series(tmp_path, [first, second])

    with pytest.raises(ValueError, match="Mesh_velocity.*not found"):
        mt.load_ale_flow(pvd)


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_load_ale_flow_rejects_invalid_velocity_scale(tmp_path, scale):
    pvd, _ = _save_series(tmp_path, [_tetra(), _tetra()])

    with pytest.raises(ValueError, match="velocity_scale"):
        mt.load_ale_flow(pvd, velocity_scale=scale)


def test_load_ale_flow_scales_both_velocity_fields(tmp_path):
    pvd, _ = _save_series(
        tmp_path,
        [
            _tetra(velocity=1.0, mesh_velocity=0.25),
            _tetra(velocity=2.0, mesh_velocity=0.5),
        ],
    )

    flow = mt.load_ale_flow(pvd, velocity_scale=100)

    np.testing.assert_allclose(flow.velocity(0), 100)
    np.testing.assert_allclose(flow.mesh_velocity(0), 25)
    np.testing.assert_allclose(flow.relative_velocity(0), 75)
    assert flow.velocity_scale == 100


def test_ale_relative_sampling_uses_velocity_minus_mesh_velocity(tmp_path):
    pvd, _ = _save_series(
        tmp_path,
        [
            _tetra(velocity=2, mesh_velocity=0.5),
            _tetra(velocity=4, mesh_velocity=1.0),
        ],
    )
    flow = mt.load_ale_flow(pvd)

    relative, valid, _ = flow.sample_relative_v([[0.1, 0.1, 0.1]], 0.5)

    assert valid.tolist() == [True]
    np.testing.assert_allclose(relative, 2.25)


def test_ale_sampling_uses_interpolated_deformed_mesh(tmp_path):
    displacement = np.array([1.0, 0.0, 0.0])
    pvd, _ = _save_series(
        tmp_path,
        [
            _tetra(velocity=0, displacement=0),
            _tetra(velocity=2, displacement=displacement),
            _tetra(velocity=4, displacement=2 * displacement),
        ],
    )
    flow = mt.load_ale_flow(pvd)

    velocity, valid, _ = flow.sample_v([[0.6, 0.05, 0.05]], 0.5)

    assert valid.tolist() == [True]
    np.testing.assert_allclose(velocity, 1.0)
    np.testing.assert_allclose(flow.active_mesh.points[0], [0.5, 0.0, 0.0])


def test_ale_runtime_cache_holds_three_time_states(tmp_path):
    pvd, _ = _save_series(
        tmp_path,
        [_tetra(), _tetra(), _tetra()],
    )
    flow = mt.load_ale_flow(pvd)
    point = np.array([[0.1, 0.1, 0.1]])

    for time in (0.0, 0.5, 0.5, 1.0):
        flow.sample_v(point, time)

    assert flow._runtime_build_count == 3
    assert len(flow._runtime_cache) == 3
    runtimes = list(flow._runtime_cache.values())
    assert all(runtime.locator is None for runtime in runtimes)
    assert runtimes[1].sampler._adj is runtimes[0].sampler._adj

    flow.sample_v(point, 1.5)
    assert flow._runtime_build_count == 4
    assert len(flow._runtime_cache) == 3


def test_track_accepts_ale_flow_without_core_special_case(tmp_path):
    points = 10 * _tetra().points
    velocity = np.array([0.1, 0.0, 0.0])
    pvd, _ = _save_series(
        tmp_path,
        [
            _tetra(points=points, velocity=velocity, displacement=0),
            _tetra(points=points, velocity=velocity, displacement=velocity),
            _tetra(points=points, velocity=velocity, displacement=2 * velocity),
        ],
    )
    flow = mt.load_ale_flow(pvd)
    seeds = np.array([[0.5, 0.5, 0.5]])

    result = mt.track(
        flow,
        seeds=seeds,
        inlet=seeds,
        dt=0.1,
        tmax=0.2,
        pbar=False,
    )

    np.testing.assert_allclose(result.positions[:, 0, 0], [0.51, 0.52])
    assert not result.reset.any()
    assert flow._runtime_build_count == 5
    assert len(flow._runtime_cache) == 3


def test_ale_reseeding_uses_deformed_area_and_relative_inflow(tmp_path):
    meshes = [
        _two_tetra_mesh(stretch=0),
        _two_tetra_mesh(stretch=1),
        _two_tetra_mesh(stretch=1),
    ]
    pvd, _ = _save_series(tmp_path, meshes)
    flow = mt.load_ale_flow(pvd)
    caps = _two_tetra_caps(meshes[0])
    reseeder = mt.ALEBoundaryReseeder(
        caps,
        flow,
        inward_eps=0.01,
        rng=np.random.default_rng(1234),
    )

    initial = reseeder._cap_state(0.0)
    stretched = reseeder._cap_state(1.0)

    np.testing.assert_allclose(initial.area, [0.5, 0.5])
    np.testing.assert_allclose(initial.inward_speed, [2.0, 1.0])
    np.testing.assert_allclose(
        np.diff(np.r_[0, initial.cumulative_flux]), [1, 0.5]
    )
    np.testing.assert_allclose(stretched.area, [1.0, 0.5])
    np.testing.assert_allclose(
        np.diff(np.r_[0, stretched.cumulative_flux]), [2, 0.5]
    )

    points, cells = reseeder.reseed_with_cells(1000, 1.0)
    runtime = flow._runtime(1.0)
    assert np.all(cells >= 0)
    assert np.all(runtime.sampler._bary(points, cells) >= -runtime.sampler.tol)
    assert len(reseeder._state_cache) == 2


def test_track_reseeds_invalid_ale_particle_on_deformed_cap(tmp_path, monkeypatch):
    meshes = [
        _two_tetra_mesh(stretch=0),
        _two_tetra_mesh(stretch=0.5),
        _two_tetra_mesh(stretch=1),
    ]
    pvd, _ = _save_series(tmp_path, meshes)
    flow = mt.load_ale_flow(pvd)
    reseeder = mt.ALEBoundaryReseeder(
        _two_tetra_caps(meshes[0]),
        flow,
        inward_eps=0.01,
        rng=np.random.default_rng(1234),
    )

    probe_calls = 0
    original_probe = _TetSampler._locate_probe

    def counted_probe(self, points):
        nonlocal probe_calls
        probe_calls += 1
        return original_probe(self, points)

    monkeypatch.setattr(_TetSampler, "_locate_probe", counted_probe)

    result = mt.track(
        flow,
        seeds=np.array([[100.0, 100.0, 100.0]]),
        reseeder=reseeder,
        dt=0.1,
        tmax=0.2,
        pbar=False,
    )

    assert result.reset[0].tolist() == [True]
    assert probe_calls == 1  # initial arbitrary seed bootstrap only
    runtime = flow._runtime(0.0)
    assert np.all(runtime.sampler.locate(result.positions[0]) >= 0)
