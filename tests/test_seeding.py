import numpy as np
import pyvista as pv

from mrsimtracks.seeding import seed_mesh, seed_region


def _unit_tet_mesh():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([pv.CellType.TETRA])
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data["density"] = np.array([0.1, 0.5, 0.75, 1.0])
    mesh.point_data["velocity"] = points
    return mesh


def _assert_inside_unit_tet(mesh, points):
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert np.isfinite(points).all()
    assert np.all(points >= 0.0)
    valid = pv.PolyData(points).sample(mesh)["vtkValidPointMask"]
    assert np.all(valid)


def test_seed_mesh_returns_repeatable_points_inside_domain():
    mesh = _unit_tet_mesh()

    seeds1 = seed_mesh(mesh, 200, rng=np.random.default_rng(1234))
    seeds2 = seed_mesh(mesh, 200, rng=np.random.default_rng(1234))

    assert 0 < seeds1.shape[0] <= 200
    np.testing.assert_allclose(seeds1, seeds2)
    _assert_inside_unit_tet(mesh, seeds1)


def test_seed_region_supports_scalar_normalization():
    mesh = _unit_tet_mesh()

    seeds = seed_region(
        mesh,
        400,
        mesh.bounds,
        normalization="density",
        rng=np.random.default_rng(1234),
    )

    assert 0 < seeds.shape[0] <= 400
    _assert_inside_unit_tet(mesh, seeds)


def test_seed_region_supports_vector_normalization():
    mesh = _unit_tet_mesh()

    seeds = seed_region(
        mesh,
        400,
        mesh.bounds,
        normalization="velocity",
        rng=np.random.default_rng(1234),
    )

    assert 0 < seeds.shape[0] <= 400
    _assert_inside_unit_tet(mesh, seeds)
