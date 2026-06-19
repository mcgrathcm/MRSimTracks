"""Regression guard against silent numerical drift in tracking.

A committed ground-truth (GT) trajectory set -- a small-dt, f64, RK4 reference
over the committed small fixture -- is re-tracked here on the *same* seeds and
config. Any change to the sampler, integrator, or interpolation that moves
trajectories shows up as error against the GT.

The GT's own numerical uncertainty was characterized by a one-off dt-convergence
study (field interpolation limits convergence to ~1st order; residual ~3e-6 of
domain scale at this dt). The tolerances below sit far above same-machine float
noise (which reproduces the GT essentially exactly) yet far below the error any
real behavioral change would produce, so the test is sensitive without being
flaky. If a legitimate algorithm change shifts results, regenerate the GT with
``scripts/ground_truth.py generate``.
"""

from pathlib import Path

import h5py
import numpy as np

import mrsimtracks as pt
from diagnostics import aligned_trajectory_error

DATA = Path(__file__).parent / "data"
GT_FILE = DATA / "ground_truth.h5"
FLOW = DATA / "CFD_velocity_00190_00210.vtu"
CAPS = [DATA / "Inlet.vtp", DATA / "Outlet.vtp"]

# Same-machine reproduction is ~0; these bound cross-platform float noise while
# staying orders of magnitude below the error of any real behavioral change.
MEDIAN_REL_TOL = 1e-6   # median clean-particle error, relative to domain scale
MAX_REL_TOL = 1e-4      # worst single clean particle, relative to domain scale


def _load_gt():
    with h5py.File(GT_FILE, "r") as f:
        return (f["position"][...], f["reset"][...], f["seeds"][...], dict(f.attrs))


def test_ground_truth_fixture_is_well_formed():
    gt_pos, gt_reset, seeds, attrs = _load_gt()

    assert gt_pos.shape == (400, 256, 3)
    assert gt_reset.shape == (400, 256)
    assert seeds.shape == (256, 3)
    assert gt_pos.dtype == np.float64
    assert attrs["method"] == "RK4"
    assert attrs["precision"] == "f64"
    assert np.isfinite(gt_pos).all()


def test_tracking_reproduces_ground_truth_within_tolerance():
    gt_pos, gt_reset, seeds, attrs = _load_gt()
    dt = float(attrs["dt"])
    tmax = float(attrs["tmax"])
    seed = int(attrs["seed"])

    flow = pt.load_flow(FLOW, active_key=attrs["active_key"], pbar=False)
    reseeder = pt.BoundaryReseeder(
        [str(c) for c in CAPS], flow, dt=dt, rng=np.random.default_rng(seed))
    result = pt.track(
        flow, seeds=seeds, dt=dt, tmax=tmax, method=attrs["method"],
        reseeder=reseeder, rng=np.random.default_rng(seed), pbar=False)

    cand_pos = np.asarray(result.positions)
    cand_reset = np.asarray(result.reset)
    assert cand_pos.shape == gt_pos.shape

    # Same dt -> k = 1 (direct, step-for-step comparison).
    err = aligned_trajectory_error(gt_pos, gt_reset, cand_pos, cand_reset, k=1)
    scale = float(np.linalg.norm(np.ptp(seeds, axis=0)))

    final = err[-1]
    final_clean = final[np.isfinite(final)]
    assert final_clean.size > 0.9 * seeds.shape[0]          # most particles compared
    assert np.nanmedian(final) < MEDIAN_REL_TOL * scale
    assert np.nanmax(err) < MAX_REL_TOL * scale             # over all steps
