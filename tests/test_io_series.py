import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt
import mrsimtracks.io as mt_io


def _tetra(points=None, *, velocity=0.0):
    if points is None:
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
    mesh = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([10], np.uint8),
        points,
    )
    mesh.point_data["velocity"] = np.full((len(points), 3), velocity)
    return mesh


def _write_pvd(path, files):
    datasets = "\n".join(
        f'    <DataSet timestep="{time}" file="{file.name}"/>'
        for time, file in files
    )
    path.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"Collection\" version=\"0.1\">\n"
        "  <Collection>\n"
        f"{datasets}\n"
        "  </Collection>\n"
        "</VTKFile>\n"
    )


def _save_series(tmp_path, meshes, times=(0.0, 1.0)):
    files = []
    for index, (time, mesh) in enumerate(zip(times, meshes, strict=True)):
        file = tmp_path / f"flow_{index:02d}.vtu"
        mesh.save(file)
        files.append((time, file))
    pvd = tmp_path / "data.pvd"
    _write_pvd(pvd, files)
    return pvd, [file for _, file in files]


def test_static_series_deduplicates_mesh(tmp_path):
    pvd, _ = _save_series(tmp_path, [_tetra(velocity=1), _tetra(velocity=2)])

    flow = mt.load_flow(pvd)

    assert flow.geometry_mode == "static"
    assert len(flow.data.topologies) == 1
    assert len(flow.data.coordinates) == 1
    assert flow.data.topology_ids.tolist() == [0, 0]
    assert flow.data.coordinate_ids.tolist() == [0, 0]
    np.testing.assert_allclose(flow._frame_vel(1), 2)


def test_moving_node_series_shares_topology(tmp_path):
    points = _tetra().points.copy()
    moved = points + np.array([0.1, 0.0, 0.0])
    pvd, _ = _save_series(
        tmp_path,
        [_tetra(points, velocity=1), _tetra(moved, velocity=2)],
    )

    flow = mt.load_flow(pvd, mesh_mode="moving-node")

    assert flow.geometry_mode == "moving"
    assert len(flow.data.topologies) == 1
    assert len(flow.data.coordinates) == 2
    assert flow.data.topology_ids.tolist() == [0, 0]
    np.testing.assert_allclose(flow.data.points(1), moved)
    velocity, valid, _ = flow.sample_v(np.array([[0.2, 0.1, 0.1]]), 0.5)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(velocity, 1.5)


def test_center_mesh_uses_initial_frame_for_all_flow_frames(tmp_path):
    offset = np.array([10.0, -4.0, 2.0])
    translation = np.array([0.2, 0.3, -0.1])
    base = _tetra().points
    first = _tetra(base + offset, velocity=1)
    second = _tetra(base + offset + translation, velocity=2)
    pvd, _ = _save_series(tmp_path, [first, second])

    flow = mt.load_flow(pvd, mesh_mode="moving", center_mesh=True)
    expected_shift = -(offset + 0.5)

    np.testing.assert_allclose(flow.origin_shift, expected_shift)
    np.testing.assert_allclose(flow.data.points(0), first.points + expected_shift)
    np.testing.assert_allclose(flow.data.points(1), second.points + expected_shift)
    np.testing.assert_allclose(flow.data.points(0).min(axis=0), -0.5)
    np.testing.assert_allclose(flow.data.points(0).max(axis=0), 0.5)
    np.testing.assert_allclose(flow._sampler.node_xyz, flow.data.points(0))
    np.testing.assert_allclose(flow._frame_vel(1), 2)


def test_boundary_reseeder_shifts_vtp_caps_with_centered_flow(tmp_path):
    offset = np.array([10.0, -4.0, 2.0])
    base = _tetra().points
    mesh = _tetra(base + offset)
    mesh.point_data["velocity"] = np.zeros((4, 3))
    pvd, _ = _save_series(tmp_path, [mesh, mesh])
    flow = mt.load_flow(pvd, center_mesh=True)

    cap = pv.PolyData(
        mesh.points[[1, 2, 3]],
        np.array([3, 0, 1, 2]),
    )
    cap.cell_data["region_id"] = np.array([0], dtype=np.int32)
    cap_path = tmp_path / "cap.vtp"
    cap.save(cap_path)

    reseeder = mt.BoundaryReseeder(cap_path, flow, inward_eps=0.01)
    np.testing.assert_allclose(
        reseeder._a[0], mesh.points[1] + flow.origin_shift
    )


def test_declared_static_checks_midpoint_node_locations(tmp_path):
    points = _tetra().points.copy()
    moved = points + np.array([0.1, 0.0, 0.0])
    pvd, _ = _save_series(
        tmp_path,
        [
            _tetra(points, velocity=1),
            _tetra(moved, velocity=2),
            _tetra(points, velocity=3),
        ],
        times=(0.0, 0.5, 1.0),
    )

    with pytest.raises(ValueError, match="midpoint frame.*node locations"):
        mt.load_flow(pvd, mesh_mode="static")


@pytest.mark.parametrize("mesh_mode", ["static", "moving"])
def test_declared_shared_topology_checks_midpoint_connectivity(tmp_path, mesh_mode):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=float,
    )
    first = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4]),
        np.array([10, 10], np.uint8),
        points,
    )
    changed = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 4, 4, 0, 2, 3, 4]),
        np.array([10, 10], np.uint8),
        points,
    )
    for mesh in (first, changed):
        mesh.point_data["velocity"] = np.ones((5, 3))
    pvd, _ = _save_series(
        tmp_path,
        [first, changed, first],
        times=(0.0, 0.5, 1.0),
    )

    with pytest.raises(ValueError, match="midpoint frame.*different topology"):
        mt.load_flow(pvd, mesh_mode=mesh_mode)


@pytest.mark.parametrize("mesh_mode", ["static", "moving"])
def test_declared_shared_topology_rejects_changed_point_count(tmp_path, mesh_mode):
    first = _tetra(velocity=1)
    points = np.vstack((first.points, [[1, 1, 1]]))
    second = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4]),
        np.array([10, 10], np.uint8),
        points,
    )
    second.point_data["velocity"] = np.full((5, 3), 2.0)
    pvd, _ = _save_series(tmp_path, [first, second])

    with pytest.raises(ValueError, match="point values per frame"):
        mt.load_flow(pvd, mesh_mode=mesh_mode)


def test_declared_static_skips_geometry_classification(tmp_path, monkeypatch):
    pvd, _ = _save_series(tmp_path, [_tetra(velocity=1), _tetra(velocity=2)])

    def fail_if_called(_):
        raise AssertionError("geometry classification should be skipped")

    monkeypatch.setattr(mt_io, "_geometry_signatures", fail_if_called)
    flow = mt.load_flow(pvd, mesh_mode="static")

    assert flow.geometry_mode == "static"


def test_invalid_mesh_mode_is_rejected(tmp_path):
    pvd, _ = _save_series(tmp_path, [_tetra(velocity=1), _tetra(velocity=2)])

    with pytest.raises(ValueError, match="mesh_mode must be one of"):
        mt.load_flow(pvd, mesh_mode="unknown")


def test_topology_changing_series_retains_each_topology(tmp_path):
    first = _tetra(velocity=1)
    points = np.vstack((first.points, [[1, 1, 1]]))
    second = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4]),
        np.array([10, 10], np.uint8),
        points,
    )
    second.point_data["velocity"] = np.full((5, 3), 2.0)
    pvd, _ = _save_series(tmp_path, [first, second])

    flow = mt.load_flow(pvd)

    assert flow.geometry_mode == "changing_topology"
    assert len(flow.data.topologies) == 2
    assert flow.data.topology_ids.tolist() == [0, 1]
    assert flow.data.mesh(1).n_cells == 2
    velocity, valid, cells = flow.sample_v(np.array([[0.1, 0.1, 0.1]]), 0.5)
    assert valid.tolist() == [True]
    assert cells is None
    np.testing.assert_allclose(velocity, 1.5)


def test_directory_and_file_list_are_series_sources(tmp_path):
    _, files = _save_series(tmp_path, [_tetra(velocity=1), _tetra(velocity=2)])

    from_directory = mt.load_flow(tmp_path, dt=0.25)
    from_list = mt.load_flow(list(reversed(files)), dt=0.25)

    np.testing.assert_allclose(from_directory.times_shift_s, [0.0, 0.25])
    np.testing.assert_allclose(from_list.times_shift_s, [0.0, 0.25])
    np.testing.assert_allclose(from_directory._frame_vel(1), 2)
    np.testing.assert_allclose(from_list._frame_vel(1), 2)


def test_single_vtu_still_uses_time_labeled_fields(tmp_path):
    mesh = _tetra()
    del mesh.point_data["velocity"]
    mesh.point_data["velocity_00100"] = np.full((4, 3), 1.0)
    mesh.point_data["velocity_00120"] = np.full((4, 3), 2.0)
    mesh.point_data["velocity_00140"] = np.full((4, 3), 3.0)
    path = tmp_path / "flow.vtu"
    mesh.save(path)

    flow = mt.load_flow(path)

    assert flow.geometry_mode == "static"
    np.testing.assert_array_equal(flow.times, [100, 120, 140])
    np.testing.assert_allclose(flow.times_shift_s, [0.0, 0.02, 0.04])

    subsampled = mt.load_flow(path, subsamp=2)
    np.testing.assert_array_equal(subsampled.times, [100, 140])
    np.testing.assert_allclose(subsampled._frame_vel(1), 3)
