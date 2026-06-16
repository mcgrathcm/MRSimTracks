"""particle_tracking -- Lagrangian particle tracking in time-resolved CFD meshes.

Typical use:

    import particle_tracking as pt

    flow = pt.load_flow("case.pvd", active_key="Velocity")            # .vtu or .pvd
    reseeder = pt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"],       # optional,
                                   flow, dt=0.002)                    # backflow-aware
    result = pt.track(flow, n_particles=1e5, dt=0.002, reseeder=reseeder)
    result.save("tracks.h5")

For large runs across processes (each worker reloads the field):

    result = pt.track_parallel("case.pvd", n_particles=2e6, dt=0.002,
                               caps=["Inlet.vtp", "Outlet.vtp"],
                               active_key="Velocity", n_workers=3)

Lower-level building blocks (flow classes, seeding, the tracking loop, the
reseeder, cap extraction) are re-exported below for advanced use.
"""

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from .tracking import (
    timeMeshSingleVTU, timeMeshPVD, timeMeshStaticPVD,
    seed_mesh, seed_region, tracking, tracking_parallel, batched_particles,
)
from .reseeding import BoundaryReseeder
from .caps import extract_caps

__all__ = [
    "load_flow", "seed_volume", "track", "track_parallel", "TrackingResult",
    "BoundaryReseeder", "extract_caps",
    "timeMeshSingleVTU", "timeMeshPVD", "timeMeshStaticPVD",
    "seed_mesh", "seed_region", "tracking", "tracking_parallel", "batched_particles",
]


def load_flow(path, active_key="velocity", subsamp=1, only_active_key=True, pbar=False, dt=None):
    """Load a time-resolved flow field, picking the right reader for the file type.

    .vtu -> timeMeshSingleVTU (one file, one field array per timestep)
    .pvd -> timeMeshStaticPVD (a series; stores geometry once + one field per frame)

    `active_key` is the velocity array name ("velocity", "Velocity", ...).
    `subsamp` keeps every Nth frame (.pvd only) to trade temporal resolution for memory.
    """
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "vtu":
        return timeMeshSingleVTU(path, active_key=active_key, pbar=pbar,
                                 only_active_key=only_active_key)
    if ext == "pvd":
        return timeMeshStaticPVD(path, active_key=active_key, pbar=pbar,
                                 subsamp=subsamp, dt=dt)
    raise ValueError(f"unsupported flow file type: .{ext} (expected .vtu or .pvd)")


def seed_volume(flow, n_particles):
    """Seed ~n_particles uniformly inside the flow domain (returns an (m, 3) array)."""
    return seed_mesh(flow.active_mesh, n_particles)


@dataclass
class TrackingResult:
    """Particle tracks. ``positions`` is (n_steps, n_particles, 3); ``reset`` flags
    which particles were recycled at each step; ``dt`` is the time step."""
    positions: np.ndarray
    reset: np.ndarray
    dt: float

    @property
    def n_steps(self):
        return self.positions.shape[0]

    @property
    def n_particles(self):
        return self.positions.shape[1]

    @property
    def times(self):
        return np.arange(self.n_steps) * self.dt

    def save(self, path, time_subsample=1):
        """Write positions/reset/dt to an HDF5 file (optionally thinned in time)."""
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("position", data=self.positions[::time_subsample])
            f.create_dataset("reset", data=self.reset[::time_subsample])
            f.attrs["dt"] = self.dt * time_subsample


def track(flow, n_particles=None, dt=1e-3, tmax=None, reseeder=None, seeds=None,
          inlet=None, method="RK4", pbar=True):
    """Track particles through a loaded ``flow`` (single process).

    Provide ``seeds`` ((m,3) array or PolyData) or ``n_particles`` to seed the
    volume. ``reseeder`` (a BoundaryReseeder) recycles out-of-bounds particles to
    currently-inflow caps; otherwise pass ``inlet`` (an (m,3) cloud) for the legacy
    static reseeding. ``tmax`` defaults to one period (flow.tmax).
    """
    if seeds is None:
        if n_particles is None:
            raise ValueError("provide either seeds or n_particles")
        seeds = seed_volume(flow, n_particles)
    seeds = seeds if isinstance(seeds, pv.PolyData) else pv.PolyData(np.asarray(seeds, float))
    if tmax is None:
        tmax = flow.tmax
    inlet_arr = np.empty((0, 3)) if inlet is None else np.asarray(inlet, float)

    pos, reset, _ = tracking(flow, seeds, inlet_arr, dt, tmax, method=method,
                             pbar=pbar, reseeder=reseeder)
    return TrackingResult(pos, reset, dt)


def track_parallel(path, n_particles=None, dt=1e-3, tmax=None, caps=None, inlet=None,
                   n_workers=3, active_key="velocity", method="RK4", subsamp=1,
                   only_active_key=True, seeds=None, pbar=True):
    """Track particles across ``n_workers`` processes, each reloading ``path``.

    Best for large runs: seeds are split into batches tracked in parallel and
    recombined. ``caps`` (cap surface path(s) or a labeled surface) enables
    backflow-aware inflow reseeding; ``inlet`` is the legacy static cloud.
    """
    from joblib import Parallel, delayed

    # One light load to obtain seeds and/or the period, then free before workers.
    if seeds is None or tmax is None:
        flow = load_flow(path, active_key=active_key, subsamp=subsamp,
                         only_active_key=only_active_key)
        if tmax is None:
            tmax = flow.tmax
        if seeds is None:
            if n_particles is None:
                raise ValueError("provide either seeds or n_particles")
            seeds = seed_volume(flow, n_particles)
        del flow

    seeds = np.asarray(seeds, float)
    inlet_arr = np.empty((0, 3)) if inlet is None else np.asarray(inlet, float)
    batch_size = int(np.ceil(len(seeds) / n_workers))
    batches = batched_particles(seeds, batch_size)

    results = Parallel(n_jobs=len(batches))(
        delayed(tracking_parallel)(
            path, batch, inlet_arr, dt, tmax, method=method, active_key=active_key,
            only_active_key=only_active_key, caps=caps, subsamp=subsamp,
            pbar=(pbar and i == 0))
        for i, batch in enumerate(batches)
    )

    pos = np.concatenate([r[0] for r in results], axis=1)
    reset = np.concatenate([r[1] for r in results], axis=1)
    return TrackingResult(pos, reset, dt)
