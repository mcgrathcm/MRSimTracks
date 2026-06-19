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
