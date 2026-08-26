import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt


def _tetra(*, points=None, velocity=0.0, displacement=0.0):
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
    return mesh


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


def test_load_ale_flow_loads_both_fields_on_one_mesh(tmp_path):
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
    assert set(flow.frame(1).point_data) == {"Velocity", "Displacement"}


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
