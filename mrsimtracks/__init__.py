"""MRSimTracks -- CFD-derived particle trajectories for MR flow simulation.

Typical use:

    import mrsimtracks as mt

    flow = mt.load_flow("case.pvd", active_key="Velocity")   # VTU/PVD/dir/list
    reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"],
                                   flow, dt=0.002)                    # backflow-aware
    result = mt.track(flow, seeds=seeds, dt=0.002, reseeder=reseeder)
    result.save("tracks.h5")
"""

from .core import TrackingResult, track
from .imaging import VelocityImage, sample_velocity_image
from .io import load_flow
from .motion import MaterialPoints, MaterialTrajectory, MeshMotion, load_mesh_motion
from .parallel import track_parallel
from .reseeding import BoundaryReseeder
from .wall_slip import WallSlip

__all__ = [
    "load_flow",
    "load_mesh_motion",
    "sample_velocity_image",
    "track",
    "track_parallel",
    "TrackingResult",
    "MeshMotion",
    "MaterialPoints",
    "MaterialTrajectory",
    "VelocityImage",
    "BoundaryReseeder",
    "WallSlip",
]
