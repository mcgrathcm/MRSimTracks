"""Unit tests for the tetrahedral velocity sampler.

These isolate interpolation correctness from the CFD data: on an all-tet mesh
carrying a *linear* velocity field, barycentric interpolation is exact, so the
sampler must reproduce the field to ~machine precision at arbitrary interior
points -- on the cold (probe) path, the fused numba walk path, and the numpy
walk fallback alike. Out-of-domain points must report invalid.
"""

import numpy as np
import pyvista as pv
import pytest

from mrsimtracks.sampler import _TetSampler

# Linear field v(x) = A x + b: exactly representable by P1 barycentric interp.
_A = np.array([[2.0, -1.0, 0.5],
               [0.0, 3.0, -2.0],
               [1.0, 0.5, -1.0]])
_B = np.array([0.3, -0.7, 1.1])


def linear_field(points):
    return points @ _A.T + _B


@pytest.fixture(scope="module")
def tet_mesh():
    """A unit-cube tetrahedralization (all tets) from a jittered grid."""
    rng = np.random.default_rng(0)
    lin = np.linspace(0.0, 1.0, 5)
    grid = np.array([[x, y, z] for x in lin for y in lin for z in lin])
    # small interior jitter avoids degenerate co-planar slivers
    interior = (grid > 0) & (grid < 1)
    grid[interior] += rng.uniform(-0.05, 0.05, size=interior.sum())
    mesh = pv.PolyData(grid).delaunay_3d()
    return mesh.cast_to_unstructured_grid()


@pytest.fixture(scope="module")
def sampler(tet_mesh):
    s = _TetSampler(tet_mesh)
    assert s.ok                      # mesh is all-tetrahedral
    return s


@pytest.fixture(scope="module")
def interior_points():
    rng = np.random.default_rng(1)
    return np.ascontiguousarray(rng.uniform(0.2, 0.8, size=(200, 3)))


def test_cold_path_interpolates_linear_field_exactly(sampler, tet_mesh, interior_points):
    vel = linear_field(np.asarray(tet_mesh.points))
    v, valid, cells = sampler.sample(interior_points, vel, guess=None)

    assert valid.all()
    assert (cells >= 0).all()
    np.testing.assert_allclose(v, linear_field(interior_points), rtol=1e-9, atol=1e-9)


def test_walk_path_interpolates_linear_field_exactly(sampler, tet_mesh, interior_points):
    vel = linear_field(np.asarray(tet_mesh.points))
    # seed the walk with each point's true cell (cold result) -> exercises the
    # fused numba walk + interpolation path.
    cells = sampler.locate(interior_points, guess=None)
    v, valid, _ = sampler.sample(interior_points, vel, guess=cells)

    assert valid.all()
    np.testing.assert_allclose(v, linear_field(interior_points), rtol=1e-9, atol=1e-9)


def test_dynamic_walk_reuses_topology_on_deformed_mesh(sampler, tet_mesh,
                                                       interior_points):
    moved = tet_mesh.copy()
    nodes = np.asarray(moved.points)
    moved.points = nodes + 0.05 * np.column_stack((
        nodes[:, 0] * nodes[:, 1],
        nodes[:, 1] * nodes[:, 2],
        nodes[:, 2] * nodes[:, 0],
    ))
    dynamic = _TetSampler(moved, dynamic=True, topology=sampler)
    guesses = sampler.locate(interior_points)
    weights = sampler._bary(interior_points, guesses)
    query = np.einsum(
        "nij,ni->nj", np.asarray(moved.points)[sampler.conn[guesses]], weights
    )
    static = _TetSampler(moved)

    v, valid, cells = dynamic.sample(
        query, linear_field(np.asarray(moved.points)), guess=guesses
    )
    expected, expected_valid, _ = static.sample(
        query, linear_field(np.asarray(moved.points)), guess=guesses
    )

    assert valid.all()
    np.testing.assert_array_equal(valid, expected_valid)
    assert dynamic.conn is sampler.conn
    assert dynamic._adj is sampler._adj
    np.testing.assert_allclose(v, expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(v, linear_field(query), rtol=1e-9, atol=1e-9)


def test_dynamic_walk_uses_geometry_tolerance_without_boundary_probe(monkeypatch):
    mesh = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA], np.uint8),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
    )
    topology = _TetSampler(mesh)
    dynamic = _TetSampler(mesh, dynamic=True, topology=topology)
    points = np.array([
        [-5e-4, 0.2, 0.2],  # inside VTK's 1e-3 * cell-length tolerance
        [-1e-2, 0.2, 0.2],  # unambiguously outside
        [0.2, 0.2, 0.2],    # valid point, but deliberately missing its guess
    ])

    def unexpected_probe(_):
        raise AssertionError("dynamic boundary/no-guess path used the locator")

    monkeypatch.setattr(dynamic, "_locate_probe", unexpected_probe)
    velocity = linear_field(mesh.points)
    sampled, valid, cells = dynamic.sample(
        points, velocity, guess=np.array([0, 0, -1])
    )

    assert valid.tolist() == [True, False, False]
    assert cells.tolist() == [0, -1, -1]
    np.testing.assert_allclose(sampled[0], linear_field(points[:1])[0])
    np.testing.assert_array_equal(sampled[1:], 0.0)

    static_probe_calls = 0
    original_probe = topology._locate_probe

    def counted_probe(query):
        nonlocal static_probe_calls
        static_probe_calls += 1
        return original_probe(query)

    monkeypatch.setattr(topology, "_locate_probe", counted_probe)
    _, static_valid, _ = topology.sample(
        points[1:2], velocity, guess=np.array([0])
    )
    assert static_valid.tolist() == [False]
    assert static_probe_calls == 1


def test_numpy_walk_locate_finds_containing_cell(sampler, interior_points):
    # locate(guess=...) is the vectorized numpy walk fallback. A point on a shared
    # face may resolve to either adjacent tet, so the invariant is not "same id as
    # the probe" but "the located cell genuinely contains the point" (all
    # barycentric coords >= -tol).
    probe_cells = sampler.locate(interior_points, guess=None)
    walk_cells = sampler.locate(interior_points, guess=probe_cells)

    assert (walk_cells >= 0).all()
    weights = sampler._bary(interior_points, walk_cells)
    assert (weights >= -1e-9).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)


def test_out_of_domain_points_are_invalid(sampler, tet_mesh):
    vel = linear_field(np.asarray(tet_mesh.points))
    outside = np.array([[5.0, 5.0, 5.0], [-2.0, 0.5, 0.5], [0.5, 0.5, 9.0]])
    v, valid, cells = sampler.sample(outside, vel, guess=None)

    assert not valid.any()
    assert (cells < 0).all()
    np.testing.assert_array_equal(v, 0.0)


def test_sampler_rejects_non_tetrahedral_mesh():
    # An ImageData/structured grid is hexahedral -> sampler declines (ok=False).
    hexes = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    assert not _TetSampler(hexes).ok
