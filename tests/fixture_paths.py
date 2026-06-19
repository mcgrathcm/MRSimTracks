"""Canonical paths to committed test fixtures (single source of truth).

The inlet/outlet cap surfaces live only in ``example/`` -- they are the same
geometry for the reduced ``tests/data`` flow and the full LFS flow, so keeping a
single copy avoids the two-locations drift risk. The ``.vtp`` caps are committed
normally (not LFS), so they are available even when the large flow is not
fetched.
"""

from pathlib import Path

ROOT = Path(__file__).parents[1]
DATA = Path(__file__).parent / "data"
EXAMPLE = ROOT / "example"

ACTIVE_KEY = "Velocity"

# Flows: reduced fixture (committed) vs full cardiac cycle (Git LFS).
SMALL_FLOW = DATA / "CFD_velocity_00190_00210.vtu"
FULL_FLOW = EXAMPLE / "CFD_velocity.vtu"

# Canonical cap surfaces (shared by both flows).
INLET = EXAMPLE / "Inlet.vtp"
OUTLET = EXAMPLE / "Outlet.vtp"
CAPS = [INLET, OUTLET]

# Committed ground-truth references (see test_ground_truth_regression).
SMALL_GT = DATA / "ground_truth.h5"
FULL_GT = DATA / "ground_truth_full.h5"

# Size below which the LFS flow is just a pointer (not fetched).
LFS_MIN_BYTES = 100_000_000


def full_flow_available():
    """True when the large example flow has been fetched through Git LFS."""
    return FULL_FLOW.stat().st_size >= LFS_MIN_BYTES
