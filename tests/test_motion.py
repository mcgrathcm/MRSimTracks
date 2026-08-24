import h5py
import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt


def _tetra(points=None):
    if points is None:
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
    return pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA], np.uint8),
        points,
    )


def _hexa(points=None):
    if points is None:
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
    return pv.UnstructuredGrid(
        np.array([8, *range(8)]),
        np.array([pv.CellType.HEXAHEDRON], np.uint8),
        points,
    )


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


def _save_series(tmp_path, meshes, times):
    files = []
    for index, (time, mesh) in enumerate(zip(times, meshes, strict=True)):
        file = tmp_path / f"motion_{index:02d}.vtu"
        mesh.save(file)
        files.append((time, file))
    pvd = tmp_path / "motion.pvd"
    _write_pvd(pvd, files)
    return pvd


def test_displacement_motion_seeds_and_translates_material_points(tmp_path):
    meshes = []
    for shift in (0.0, 1.0, 0.0):
        mesh = _tetra()
        mesh.point_data["displacement"] = np.tile([shift, 0.0, 0.0], (4, 1))
        meshes.append(mesh)
    pvd = _save_series(tmp_path, meshes, times=(0.0, 1.0, 2.0))

    motion = mt.load_mesh_motion(pvd, displacement_key="displacement")
    particles = motion.seed(500, rng=np.random.default_rng(4))
    initial = motion.positions(particles, 0.0)

    assert motion.source == "displacement"
    assert particles.n_particles == 500
    assert particles.weights.shape == (500, 4)
    np.testing.assert_allclose(particles.weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        motion.positions(particles, 1.0), initial + [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        motion.positions(particles, 0.5), initial + [0.5, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        motion.positions(particles, 2.5), initial + [0.5, 0.0, 0.0]
    )
    cloud = motion.point_cloud(particles, 1.0)
    assert cloud.n_points == 500
    np.testing.assert_array_equal(cloud["material_cell_id"], particles.cell_ids)
    trajectory = motion.trajectory(particles)
    assert trajectory.shape == (3, 500, 3)
    np.testing.assert_allclose(trajectory.times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(trajectory.positions[1], initial + [1.0, 0.0, 0.0])


def test_material_trajectory_streams_to_hdf5(tmp_path):
    meshes = []
    for shift in (0.0, 1.0, 0.0):
        mesh = _tetra()
        mesh.point_data["displacement"] = np.tile([shift, 0.0, 0.0], (4, 1))
        meshes.append(mesh)
    pvd = _save_series(tmp_path, meshes, times=(0.0, 1.0, 2.0))
    motion = mt.load_mesh_motion(pvd, displacement_key="displacement")
    particles = motion.seed(50, rng=np.random.default_rng(7))
    expected = motion.trajectory(particles, times=[0.0, 0.5, 1.0])
    path = tmp_path / "material_motion.h5"

    result = motion.trajectory(
        particles,
        times=[0.0, 0.5, 1.0],
        output_path=path,
    )

    assert result.is_file_backed
    assert result.shape == (3, 50, 3)
    with h5py.File(path, "r") as file:
        assert file["position"].shape == (3, 50, 3)
        assert file["position"].chunks == (1, 50, 3)
        np.testing.assert_allclose(file["time"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(file["position"], expected.positions)
        assert file.attrs["kind"] == "fixed_topology_material_motion"
        assert bool(file.attrs["periodic"])

    opened = mt.MaterialTrajectory.open(path)
    assert opened.is_file_backed
    np.testing.assert_allclose(opened.times, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(opened.positions, expected.positions)


def test_coordinate_motion_splits_hex_once_and_reuses_barycentric_weights(tmp_path):
    translation = np.array([0.2, -0.1, 0.3])
    first = _hexa()
    second = _hexa(first.points + translation)
    pvd = _save_series(tmp_path, [first, second], times=(0.0, 1.0))

    motion = mt.load_mesh_motion(pvd)
    particles = motion.seed(800, rng=np.random.default_rng(5))
    initial = motion.positions(particles, 0.0)

    assert motion.source == "coordinates"
    assert motion.mesh().n_cells == 6
    assert np.all(motion.mesh().celltypes == pv.CellType.TETRA)
    assert particles.node_ids.shape == (800, 4)
    assert np.all((initial >= 0.0) & (initial <= 1.0))
    np.testing.assert_allclose(
        motion.positions(particles, 0.5), initial + 0.5 * translation
    )


def test_single_vtu_supports_time_labeled_displacement_fields(tmp_path):
    mesh = _tetra()
    mesh.point_data["disp_00000"] = np.zeros((4, 3))
    mesh.point_data["disp_00010"] = np.tile([0.1, 0.0, 0.0], (4, 1))
    path = tmp_path / "motion.vtu"
    mesh.save(path)

    motion = mt.load_mesh_motion(path, displacement_key="disp")

    np.testing.assert_allclose(motion.times_shift_s, [0.0, 0.01])
    np.testing.assert_allclose(motion.node_positions[1], mesh.points + [0.1, 0, 0])


def test_coordinate_motion_rejects_changed_mesh_size(tmp_path):
    first = _tetra()
    points = np.vstack((first.points, [[1, 1, 1]]))
    second = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4]),
        np.array([pv.CellType.TETRA, pv.CellType.TETRA], np.uint8),
        points,
    )
    pvd = _save_series(tmp_path, [first, second], times=(0.0, 1.0))

    with pytest.raises(ValueError, match="fixed-topology.*points and.*cells"):
        mt.load_mesh_motion(pvd)


def test_coordinate_motion_checks_midpoint_connectivity(tmp_path):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=float,
    )
    first = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4]),
        np.array([pv.CellType.TETRA, pv.CellType.TETRA], np.uint8),
        points,
    )
    changed = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 4, 4, 0, 2, 3, 4]),
        np.array([pv.CellType.TETRA, pv.CellType.TETRA], np.uint8),
        points,
    )
    pvd = _save_series(
        tmp_path,
        [first, changed, first],
        times=(0.0, 0.5, 1.0),
    )

    with pytest.raises(ValueError, match="midpoint frame.*different topology"):
        mt.load_mesh_motion(pvd)


def test_nonperiodic_motion_rejects_times_outside_loaded_interval(tmp_path):
    first = _tetra()
    second = _tetra(first.points + [0.1, 0.0, 0.0])
    pvd = _save_series(tmp_path, [first, second], times=(0.0, 1.0))
    motion = mt.load_mesh_motion(pvd, periodic=False)
    particles = motion.seed(2, rng=np.random.default_rng(6))

    with pytest.raises(ValueError, match="time must be within"):
        motion.positions(particles, 1.1)
