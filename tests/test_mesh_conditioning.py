"""Tests for load-time mesh conditioning and the degenerate-cell guard.

Synthetic meshes (no external data): a hybrid tet+wedge mesh to exercise the
wedge->tet split, and an all-tet mesh with a zero-volume sliver to exercise the
degenerate-cell drop and the _TetSampler guard.
"""

import numpy as np
import pyvista as pv
import pytest

from mrsimtracks.sampler import VTK_TETRA, _TetSampler, _condition_mesh

VTK_WEDGE = 13


def _hybrid_mesh():
    """One wedge + one tet (mixed cell types), with a 'velocity' array."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1],  # wedge
        [2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1],                        # tet
    ], dtype=float)
    cells = np.array([6, 0, 1, 2, 3, 4, 5, 4, 6, 7, 8, 9])
    ctypes = np.array([VTK_WEDGE, VTK_TETRA], np.uint8)
    g = pv.UnstructuredGrid(cells, ctypes, pts)
    g.point_data["velocity"] = pts * 0.5
    return g


def _degenerate_mesh():
    """One good tet + one flat (coplanar, zero-volume) tet."""
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],        # good tet
        [2, 0, 0], [3, 0, 0], [2, 1, 0], [2.4, 0.4, 0],    # flat tet (all z=0)
    ], dtype=float)
    cells = np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7])
    ctypes = np.array([VTK_TETRA, VTK_TETRA], np.uint8)
    g = pv.UnstructuredGrid(cells, ctypes, pts)
    g.point_data["velocity"] = pts * 1.0
    return g


def test_condition_mesh_splits_wedges_into_tets():
    g = _hybrid_mesh()
    out = _condition_mesh(g, verbose=False)

    assert np.all(np.asarray(out.celltypes) == VTK_TETRA)   # all tet now
    assert out.n_cells == 4                                  # wedge -> 3 tets, + 1 tet
    assert out.n_points == g.n_points                        # no points added
    assert "velocity" in out.point_data                      # field preserved


def test_condition_mesh_drops_degenerate_cells():
    g = _degenerate_mesh()
    out = _condition_mesh(g, verbose=False)

    assert np.all(np.asarray(out.celltypes) == VTK_TETRA)
    assert out.n_cells == 1                                  # flat tet dropped
    assert "velocity" in out.point_data


def test_condition_mesh_is_noop_on_clean_all_tet():
    g = _degenerate_mesh().extract_cells([0])               # the single good tet
    g = pv.UnstructuredGrid(g)                              # ensure UnstructuredGrid
    out = _condition_mesh(g, verbose=False)
    assert out is g                                          # untouched (no copy)


def test_condition_mesh_prints_summary(capsys):
    _condition_mesh(_hybrid_mesh(), verbose=True)
    msg = capsys.readouterr().out
    assert "mesh conditioning" in msg
    assert "non-tetrahedral" in msg


def test_tetsampler_guard_handles_degenerate_without_crashing(capsys):
    g = _degenerate_mesh()
    s = _TetSampler(g)                                       # must not raise
    assert s.ok
    assert int(s._degenerate.sum()) == 1
    assert "degenerate" in capsys.readouterr().out          # notified

    # sampling still works and never returns the degenerate cell
    pts = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])      # inside the good tet
    vel = np.asarray(g.point_data["velocity"], np.float64)
    v, valid, cells = s.sample(np.ascontiguousarray(pts), vel, guess=None)
    assert valid.all()
    assert np.isfinite(v).all()
    assert not s._degenerate[cells[valid]].any()


def test_load_flow_conform_toggle(tmp_path):
    g = _hybrid_mesh()
    g.point_data["Velocity_00000"] = g.points * 0.0
    g.point_data["Velocity_00001"] = g.points * 1.0
    path = tmp_path / "hybrid.vtu"
    g.save(path)

    import mrsimtracks as mt
    conformed = mt.load_flow(path, active_key="Velocity", conform_mesh=True)
    assert conformed._sampler.ok                             # fast path enabled

    raw = mt.load_flow(path, active_key="Velocity", conform_mesh=False)
    assert not raw._sampler.ok                               # hybrid -> fallback
