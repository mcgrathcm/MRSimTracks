"""MRSimTracks -- CFD-derived particle trajectories for MR flow simulation.

Typical use:

    import mrsimtracks as mt

    flow = mt.load_flow("case.pvd", active_key="Velocity")            # .vtu or .pvd
    reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"],
                                   flow, dt=0.002)                    # backflow-aware
    result = mt.track(flow, seeds=seeds, dt=0.002, reseeder=reseeder)
    result.save("tracks.h5")
"""

from .core import TrackingResult, track
from .io import load_flow
from .parallel import track_parallel
from .reseeding import BoundaryReseeder

__all__ = [
    "load_flow",
    "track",
    "track_parallel",
    "TrackingResult",
    "BoundaryReseeder",
]
