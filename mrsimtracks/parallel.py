import numpy as np
import pyvista as pv

from .core import TrackingResult, tracking
from .io import load_flow, timeMeshPVD, timeMeshSingleVTU, timeMeshStaticPVD


def tracking_parallel(fn, seeds, inlet, dt, tmax, method="RK4", active_key="velocity",
                      pbar=False, dt_pvd=None, only_active_key=True, caps=None,
                      static_pvd=True, subsamp=1, rng_seed=None):
    # Tracking only ever reads active_key, so skip pressure (etc.) by default to
    # speed up the per-worker reload and cut memory.
    if fn.split(".")[-1] == "vtu":
        flow = timeMeshSingleVTU(fn, active_key=active_key, pbar=pbar, only_active_key=only_active_key)
    elif fn.split(".")[-1] == "pvd":
        # static_pvd: store one geometry + per-frame fields (memory-efficient).
        # Set False to fall back to the full-mesh-per-frame timeMeshPVD.
        cls = timeMeshStaticPVD if static_pvd else timeMeshPVD
        flow = cls(fn, active_key=active_key, pbar=pbar, dt=dt_pvd, subsamp=subsamp)

    # `caps` (path or labeled surface) enables backflow-aware inflow reseeding.
    # Built per worker since the reseeder samples this worker's own flow field.
    reseeder = None
    if caps is not None:
        from .reseeding import BoundaryReseeder
        # dt enables the volumetric inflow layer (avoids density striping).
        reseeder = BoundaryReseeder(caps, flow, dt=dt)

    rng = None if rng_seed is None else np.random.default_rng(rng_seed)
    r_res, m_reset_flag, oob_loc_list = tracking(
        flow, pv.PolyData(seeds), inlet, dt, tmax, method=method, pbar=pbar,
        reseeder=reseeder, rng=rng)

    return r_res, m_reset_flag, oob_loc_list

def batched_particles(particles, batch_size, rng=None):
    result = []
    rng = rng if rng is not None else np.random.default_rng()
    p = rng.permutation(particles)
    for i in range(0, len(p), batch_size):
        result.append(p[i:i + batch_size])
    return result


def track_parallel(path, seeds, dt=1e-3, tmax=None, caps=None, inlet=None,
                   n_workers=3, active_key="velocity", method="RK4", subsamp=1,
                   only_active_key=True, pbar=True, rng=None):
    """Track particles across processes, each reloading ``path``."""
    from joblib import Parallel, delayed

    if seeds is None:
        raise ValueError("provide seeds for tracking")

    if tmax is None:
        flow = load_flow(path, active_key=active_key, subsamp=subsamp,
                         only_active_key=only_active_key)
        tmax = flow.tmax
        del flow

    rng = rng if rng is not None else np.random.default_rng()
    seeds = np.asarray(seeds, float)
    inlet_arr = np.empty((0, 3)) if inlet is None else np.asarray(inlet, float)
    batch_size = int(np.ceil(len(seeds) / n_workers))
    batches = batched_particles(seeds, batch_size, rng=rng)
    rng_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=len(batches))

    results = Parallel(n_jobs=len(batches))(
        delayed(tracking_parallel)(
            path, batch, inlet_arr, dt, tmax, method=method, active_key=active_key,
            only_active_key=only_active_key, caps=caps, subsamp=subsamp,
            pbar=(pbar and i == 0), rng_seed=int(rng_seeds[i]))
        for i, batch in enumerate(batches)
    )

    pos = np.concatenate([r[0] for r in results], axis=1)
    reset = np.concatenate([r[1] for r in results], axis=1)
    return TrackingResult(pos, reset, dt)
