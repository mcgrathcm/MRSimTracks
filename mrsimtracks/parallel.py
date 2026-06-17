import numpy as np
import pyvista as pv

from .core import TrackingResult, _track_particles
from .io import PVDFlow, SingleVTUFlow, StaticPVDFlow, load_flow


def _track_particle_batch(path, seeds, inlet, dt, tmax, method="RK4",
                          active_key="velocity", pbar=False, dt_pvd=None,
                          only_active_key=True, caps=None, static_pvd=True,
                          subsamp=1, rng_seed=None, collect_metrics=False):
    # Tracking only ever reads active_key, so skip pressure (etc.) by default to
    # speed up the per-worker reload and cut memory.
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "vtu":
        flow = SingleVTUFlow(
            path, active_key=active_key, pbar=pbar,
            only_active_key=only_active_key)
    elif ext == "pvd":
        # static_pvd: store one geometry + per-frame fields (memory-efficient).
        # Set False to fall back to the full-mesh-per-frame PVDFlow.
        cls = StaticPVDFlow if static_pvd else PVDFlow
        flow = cls(path, active_key=active_key, pbar=pbar, dt=dt_pvd, subsamp=subsamp)
    else:
        raise ValueError(f"unsupported flow file type: .{ext} (expected .vtu or .pvd)")

    # `caps` (path or labeled surface) enables backflow-aware inflow reseeding.
    # Built per worker since the reseeder samples this worker's own flow field.
    reseeder = None
    if caps is not None:
        from .reseeding import BoundaryReseeder
        # dt enables the volumetric inflow layer (avoids density striping).
        reseeder = BoundaryReseeder(caps, flow, dt=dt)

    rng = None if rng_seed is None else np.random.default_rng(rng_seed)
    metrics = {} if collect_metrics else None
    positions, reset_flags = _track_particles(
        flow, pv.PolyData(seeds), inlet, dt, tmax, method=method, pbar=pbar,
        reseeder=reseeder, rng=rng, metrics=metrics)

    return positions, reset_flags, metrics

def batched_particles(particles, batch_size, rng=None):
    result = []
    rng = rng if rng is not None else np.random.default_rng()
    p = rng.permutation(particles)
    for i in range(0, len(p), batch_size):
        result.append(p[i:i + batch_size])
    return result


def track_parallel(path, seeds, dt=1e-3, tmax=None, caps=None, inlet=None,
                   n_workers=3, active_key="velocity", method="RK4", subsamp=1,
                   only_active_key=True, pbar=True, rng=None,
                   return_metrics=False):
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
        delayed(_track_particle_batch)(
            path, batch, inlet_arr, dt, tmax, method=method, active_key=active_key,
            only_active_key=only_active_key, caps=caps, subsamp=subsamp,
            pbar=(pbar and i == 0), rng_seed=int(rng_seeds[i]),
            collect_metrics=return_metrics)
        for i, batch in enumerate(batches)
    )

    pos = np.concatenate([r[0] for r in results], axis=1)
    reset = np.concatenate([r[1] for r in results], axis=1)
    metrics = None
    if return_metrics:
        workers = [r[2] for r in results]
        metrics = {
            "n_workers": len(workers),
            "workers": workers,
            "particle_steps_per_s": sum(
                w["particle_steps_per_s"] for w in workers if w),
        }
    result = TrackingResult(pos, reset, dt, metrics=metrics)
    if return_metrics:
        return result, metrics
    return result
