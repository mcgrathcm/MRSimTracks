"""Regression guard against silent numerical drift in tracking.

Committed ground-truth (GT) trajectory sets are re-tracked here on the *same*
seeds and config; any change to the sampler, integrator, or interpolation that
moves trajectories shows up as error against the GT.

Two references:

* ``ground_truth.h5`` -- short window (small fixture, 0.02 s, dt=5e-5). Runs in
  the normal suite; catches per-step drift quickly.
* ``ground_truth_full.h5`` -- full cardiac cycle (LFS ``example`` flow, 0.858 s,
  integrated at dt=5e-5, stored every 1 ms). Marked ``large``; exercises
  long-term drift over ~17k integration steps.

Both GTs integrate at the same small dt, in f64 with RK4. A one-off
dt-convergence study showed convergence is only ~1st order -- the C0 field
interpolation (piecewise-linear in space, linear in time), not RK4 truncation,
is the accuracy ceiling -- so the GT is the dt->0 trajectory of the interpolated
field, with residual ~3e-6 of domain scale at this dt.

Tolerances sit far above same-machine float noise (the GT reproduces essentially
exactly) and far below any real behavioral change. If a legitimate algorithm
change shifts results, regenerate with ``scripts/ground_truth.py generate``.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import mrsimtracks as pt
from diagnostics import aligned_trajectory_error

DATA = Path(__file__).parent / "data"
SMALL_GT = DATA / "ground_truth.h5"
FULL_GT = DATA / "ground_truth_full.h5"
SMALL_FLOW = DATA / "CFD_velocity_00190_00210.vtu"
FULL_FLOW = Path(__file__).parents[1] / "example" / "CFD_velocity.vtu"
CAPS = [DATA / "Inlet.vtp", DATA / "Outlet.vtp"]

# Same-machine reproduction is ~0; these bound cross-platform float noise while
# staying orders of magnitude below the error of any real behavioral change.
MEDIAN_REL_TOL = 1e-6   # median clean-particle error, relative to domain scale
MAX_REL_TOL = 1e-4      # worst single clean particle, relative to domain scale


def _load_gt(path):
    with h5py.File(path, "r") as f:
        return f["position"][...], f["reset"][...], f["seeds"][...], dict(f.attrs)


def _retrack_to_stored_grid(flow_path, attrs, seeds):
    """Re-run tracking at the GT's integration dt, subsampled to its stored grid.

    Returns ``(positions, reset)`` on the same time grid as the committed GT so
    they compare element-wise (stored every ``store_every`` integration steps,
    reset OR-ed within each window to mirror the generator).
    """
    int_dt = float(attrs.get("integration_dt", attrs["dt"]))
    store_every = int(attrs.get("store_every", 1))
    tmax = float(attrs["tmax"])
    seed = int(attrs["seed"])

    flow = pt.load_flow(flow_path, active_key=attrs["active_key"], pbar=False)
    reseeder = pt.BoundaryReseeder(
        [str(c) for c in CAPS], flow, dt=int_dt, rng=np.random.default_rng(seed))
    result = pt.track(
        flow, seeds=seeds, dt=int_dt, tmax=tmax, method=attrs["method"],
        reseeder=reseeder, rng=np.random.default_rng(seed), pbar=False)

    pos = np.asarray(result.positions)
    reset = np.asarray(result.reset)
    if store_every > 1:
        n_s = pos.shape[0] // store_every
        pos = pos[store_every - 1::store_every][:n_s]
        reset = reset[:n_s * store_every].reshape(n_s, store_every, -1).any(axis=1)
    return pos, reset


def _assert_reproduces(gt_pos, gt_reset, cand_pos, cand_reset, seeds):
    assert cand_pos.shape == gt_pos.shape
    err = aligned_trajectory_error(gt_pos, gt_reset, cand_pos, cand_reset, k=1)
    scale = float(np.linalg.norm(np.ptp(seeds, axis=0)))

    final = err[-1]
    n_clean = int(np.isfinite(final).sum())
    assert n_clean > 0.5 * seeds.shape[0]                  # a real sample remains
    assert np.nanmedian(final) < MEDIAN_REL_TOL * scale
    assert np.nanmax(err) < MAX_REL_TOL * scale            # over all stored steps


def test_small_ground_truth_is_well_formed():
    gt_pos, gt_reset, seeds, attrs = _load_gt(SMALL_GT)
    assert gt_pos.shape == (400, 256, 3)
    assert gt_reset.shape == (400, 256)
    assert seeds.shape == (256, 3)
    assert gt_pos.dtype == np.float64
    assert attrs["method"] == "RK4" and attrs["precision"] == "f64"
    assert np.isfinite(gt_pos).all()


def test_full_ground_truth_is_well_formed():
    gt_pos, gt_reset, seeds, attrs = _load_gt(FULL_GT)
    assert gt_pos.shape == (858, 256, 3)
    assert seeds.shape == (256, 3)
    assert float(attrs["dt"]) == pytest.approx(1e-3)       # stored every 1 ms
    assert float(attrs["integration_dt"]) == pytest.approx(5e-5)
    assert float(attrs["tmax"]) == pytest.approx(0.858)
    assert np.isfinite(gt_pos).all()


def test_tracking_reproduces_small_ground_truth():
    gt_pos, gt_reset, seeds, attrs = _load_gt(SMALL_GT)
    cand_pos, cand_reset = _retrack_to_stored_grid(SMALL_FLOW, attrs, seeds)
    _assert_reproduces(gt_pos, gt_reset, cand_pos, cand_reset, seeds)


@pytest.mark.large
def test_tracking_reproduces_full_cycle_ground_truth():
    if FULL_FLOW.stat().st_size < 100_000_000:
        pytest.skip("full example VTU was not fetched through Git LFS")

    gt_pos, gt_reset, seeds, attrs = _load_gt(FULL_GT)
    cand_pos, cand_reset = _retrack_to_stored_grid(FULL_FLOW, attrs, seeds)
    _assert_reproduces(gt_pos, gt_reset, cand_pos, cand_reset, seeds)
