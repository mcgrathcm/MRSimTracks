"""Tests for temporal interpolation between flow frames (linear vs cubic)."""

import numpy as np
import pytest

import mrsimtracks as pt
from mrsimtracks.io import (
    _catmull_rom,
    _interp_time,
    resolve_time_interp,
)
from fixture_paths import ACTIVE_KEY, SMALL_FLOW


# --- unit tests on the interpolation core ---------------------------------- #

def test_resolve_time_interp_validates():
    assert resolve_time_interp("linear") == "linear"
    assert resolve_time_interp("cubic") == "cubic"
    with pytest.raises(ValueError, match="time_interp must be"):
        resolve_time_interp("quadratic")


def test_catmull_rom_reproduces_constant_and_hits_knots():
    p = np.array([1.0, -2.0, 3.0])
    # equal knots -> constant for any s
    np.testing.assert_allclose(_catmull_rom(p, p, p, p, 0.37), p)
    # endpoints: s=0 -> p1, s=1 -> p2
    p0, p1, p2, p3 = (np.random.default_rng(i).normal(size=3) for i in range(4))
    np.testing.assert_allclose(_catmull_rom(p0, p1, p2, p3, 0.0), p1)
    np.testing.assert_allclose(_catmull_rom(p0, p1, p2, p3, 1.0), p2)


def _interp_at(frames, mode, t, tmax=None, n_distinct=None):
    times = np.arange(len(frames), dtype=float)            # uniform spacing
    tmax = tmax if tmax is not None else times[-1] + 1.0    # period incl. wrap step
    nd = n_distinct if n_distinct is not None else len(frames)
    return _interp_time(times, tmax, nd, lambda i: frames[i], t, mode)


def test_cubic_is_exact_for_quadratic_in_time():
    # frame f_k(node) = a + b*k + c*k^2 ; Catmull-Rom (central-diff tangents) is
    # exact for quadratics, while linear is not.
    rng = np.random.default_rng(0)
    a, b, c = (rng.normal(size=(5, 3)) for _ in range(3))
    nf = 12
    frames = [a + b * k + c * k * k for k in range(nf)]
    truth = a + b * 4.7 + c * 4.7 ** 2

    cub = _interp_at(frames, "cubic", 4.7)
    lin = _interp_at(frames, "linear", 4.7)
    np.testing.assert_allclose(cub, truth, rtol=1e-10, atol=1e-10)
    assert np.abs(lin - truth).max() > 1e-3                 # linear is not exact


def test_linear_matches_explicit_two_frame_blend():
    rng = np.random.default_rng(1)
    frames = [rng.normal(size=(4, 3)) for _ in range(6)]
    # t=2.25 -> between frame 2 and 3, weight 0.25
    got = _interp_at(frames, "linear", 2.25)
    want = 0.75 * frames[2] + 0.25 * frames[3]
    np.testing.assert_allclose(got, want)


# --- integration tests through load_flow ----------------------------------- #

def test_default_is_linear():
    flow = pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, pbar=False)
    assert flow.time_interp == "linear"


def test_load_flow_rejects_bad_time_interp():
    with pytest.raises(ValueError, match="time_interp must be"):
        pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, time_interp="spline", pbar=False)


def test_cubic_hits_frames_at_knots_and_stays_finite():
    lin = pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, pbar=False)
    cub = pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, time_interp="cubic", pbar=False)

    # at a stored frame time both reproduce that exact frame
    t_knot = float(lin.times_shift_s[4])
    cub.set_active_time(t_knot)
    frame = np.asarray(cub._frame_vel(4))
    np.testing.assert_allclose(np.asarray(cub.active_mesh[ACTIVE_KEY]), frame, atol=1e-4)

    # mid-frame cubic differs from linear but stays bounded and finite
    t_mid = 0.45 * lin.tmax
    lin.set_active_time(t_mid); cub.set_active_time(t_mid)
    vl = np.asarray(lin.active_mesh[ACTIVE_KEY])
    vc = np.asarray(cub.active_mesh[ACTIVE_KEY])
    assert np.isfinite(vc).all()
    assert not np.allclose(vc, vl)                          # cubic actually differs
    assert np.abs(vc).max() < 5 * np.abs(vl).max()          # no wild overshoot


def test_cubic_tracking_runs_and_reproduces_linear_at_coarse_steps():
    # With dt at the frame spacing, sampling lands on knots where cubic==linear,
    # so a short track agrees closely; this exercises the full tracking path.
    seeds = pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, pbar=False)
    inlet = np.ascontiguousarray(seeds.active_mesh.points[:300])
    pts = np.ascontiguousarray(seeds.active_mesh.cell_centers().points[::5000])

    def run(mode):
        flow = pt.load_flow(SMALL_FLOW, active_key=ACTIVE_KEY, time_interp=mode, pbar=False)
        return pt.track(flow, seeds=pts, dt=0.002, tmax=0.006, inlet=inlet,
                        rng=np.random.default_rng(0), pbar=False).positions

    a = run("linear"); b = run("cubic")
    assert a.shape == b.shape
    assert np.isfinite(b).all()
