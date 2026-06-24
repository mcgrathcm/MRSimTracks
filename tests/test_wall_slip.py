"""Tests for the near-wall no-penetration (slip) projection."""

import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt
from mrsimtracks.wall_slip import WallSlip


class _FakeSampler:
    """Minimal stand-in exposing the geometry WallSlip reads off the sampler."""

    def __init__(self, mesh):
        self.ok = True
        self.node_xyz = np.asarray(mesh.points, dtype=np.float64)
        self.conn = mesh.cells.reshape(-1, 5)[:, 1:]
        from mrsimtracks.sampler import _TetSampler
        self._adj = _TetSampler._build_adjacency(self.conn, self.node_xyz.shape[0])


class _FakeFlow:
    def __init__(self, mesh):
        self._sampler = _FakeSampler(mesh)


def _box_mesh(n=4):
    lin = np.linspace(0.0, 1.0, n)
    pts = np.array([[x, y, z] for x in lin for y in lin for z in lin])
    return pv.PolyData(pts).delaunay_3d().cast_to_unstructured_grid()


@pytest.fixture(scope="module")
def flow():
    return _FakeFlow(_box_mesh())


def test_build_finds_walls_and_diameter(flow):
    ws = WallSlip(flow, caps=None, band_frac=0.02)
    assert ws._centroid.shape[0] > 0
    assert ws.d_hydraulic > 0
    assert ws.band == pytest.approx(0.02 * ws.d_hydraulic)


def test_removes_into_wall_component_near_wall(flow):
    ws = WallSlip(flow, caps=None, band_frac=0.25)   # wide band so a point qualifies
    # a point just inside the z=1 wall, velocity heading out through it (+z)
    pos = np.array([[0.5, 0.5, 0.98]])
    vel = np.array([[1.0, 0.0, 2.0]])
    out = ws.apply(pos, vel.copy())
    # tangential (x) preserved; the outward-normal (+z) component is removed
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 2] <= 1e-9
    # never adds inward motion or flips tangential
    assert np.linalg.norm(out) <= np.linalg.norm(vel) + 1e-9


def test_interior_particle_untouched(flow):
    ws = WallSlip(flow, caps=None, band_frac=0.02)   # thin band
    pos = np.array([[0.5, 0.5, 0.5]])                # mesh centre, far from walls
    vel = np.array([[1.0, -2.0, 0.5]])
    out = ws.apply(pos, vel.copy())
    np.testing.assert_allclose(out, vel)


def test_inward_velocity_preserved(flow):
    ws = WallSlip(flow, caps=None, band_frac=0.25)
    pos = np.array([[0.5, 0.5, 0.98]])               # near z=1 wall
    vel = np.array([[0.0, 0.0, -1.0]])               # heading back into the fluid
    out = ws.apply(pos, vel.copy())
    np.testing.assert_allclose(out, vel)             # inward motion is not touched


def test_band_scales_with_fraction(flow):
    a = WallSlip(flow, caps=None, band_frac=0.01)
    b = WallSlip(flow, caps=None, band_frac=0.02)
    assert b.band == pytest.approx(2.0 * a.band)


def test_rejects_non_tet_flow():
    class Bad:
        _sampler = type("S", (), {"ok": False})()
    with pytest.raises(ValueError, match="all-tetrahedral"):
        WallSlip(Bad(), caps=None)


def test_track_accepts_wall_slip(flow):
    # smoke test: track() threads wall_slip through without error
    seeds = np.array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]])
    # a trivial flow object with a sample_v that returns a constant velocity
    mesh = _box_mesh()

    class ConstFlow(_FakeFlow):
        dtype = np.float64
        tmax = 1.0

        def sample_v(self, pts, t, guess=None):
            n = pts.shape[0]
            return (np.tile([0.1, 0.0, 0.0], (n, 1)), np.ones(n, bool),
                    np.zeros(n, np.int64))

    cf = ConstFlow(mesh)
    ws = WallSlip(cf, caps=None, band_frac=0.02)
    res = mt.track(cf, seeds=seeds, dt=0.1, tmax=0.5, inlet=seeds,
                   wall_slip=ws, pbar=False)
    assert res.positions.shape[1:] == (2, 3)
    assert np.isfinite(res.positions).all()
