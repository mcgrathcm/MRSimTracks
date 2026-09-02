"""MRSimTracks -- CFD-derived particle trajectories for MR flow simulation.

Typical use:

    import mrsimtracks as mt

    flow = mt.load_flow("case.pvd", active_key="Velocity")   # VTU/PVD/dir/list
    reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"],
                                   flow, dt=0.002)                    # backflow-aware
    result = mt.track(flow, seeds=seeds, dt=0.002, reseeder=reseeder)
    result.save("tracks.h5")
"""

from .ale import ALEFlow, load_ale_flow
from .core import TrackingResult, track
from .cmm import CMMFlow, CMMMeshMotion, load_cmm_flow
from .imaging import VelocityImage, sample_velocity_image
from .io import load_flow
from .mapping import periodic_mapping
from .motion import MaterialPoints, MaterialTrajectory, MeshMotion, load_mesh_motion
from .parallel import track_parallel
from .reseeding import ALEBoundaryReseeder, BoundaryReseeder
from .wall_slip import WallSlip

__all__ = [
    "load_flow",
    "load_ale_flow",
    "load_cmm_flow",
    "load_mesh_motion",
    "periodic_mapping",
    "sample_velocity_image",
    "track",
    "track_parallel",
    "TrackingResult",
    "ALEFlow",
    "MeshMotion",
    "MaterialPoints",
    "MaterialTrajectory",
    "VelocityImage",
    "BoundaryReseeder",
    "ALEBoundaryReseeder",
    "WallSlip",
    "CMMFlow",
    "CMMMeshMotion",
]
