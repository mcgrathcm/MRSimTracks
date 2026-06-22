import numpy as np
import pyvista as pv

from .core import TrackingResult, _normalize_method, _track_particles
from .io import PVDFlow, SingleVTUFlow, StaticPVDFlow, load_flow


def _track_particle_batch(path, seeds, inlet, dt, tmax, method="RK4",
                          active_key="velocity", pbar=False, dt_pvd=None,
                          only_active_key=True, caps=None, static_pvd=True,
                          subsamp=1, rng_seed=None, collect_metrics=False,
                          precision="f64", time_interp="linear"):
    # Tracking only ever reads active_key, so skip pressure (etc.) by default to
    # speed up the per-worker reload and cut memory.
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "vtu":
        flow = SingleVTUFlow(
            path, active_key=active_key, pbar=pbar,
            only_active_key=only_active_key, precision=precision,
            time_interp=time_interp)
    elif ext == "pvd":
        # static_pvd: store one geometry + per-frame fields (memory-efficient).
        # Set False to fall back to the full-mesh-per-frame PVDFlow.
        cls = StaticPVDFlow if static_pvd else PVDFlow
        flow = cls(path, active_key=active_key, pbar=pbar, dt=dt_pvd,
                   subsamp=subsamp, precision=precision, time_interp=time_interp)
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
                   return_metrics=False, precision="f64", time_interp="linear"):
    """Track particles in parallel, with each worker reloading the flow field.

    Args:
        path (str | pathlib.Path): Path to a supported flow file (``.vtu`` or
            ``.pvd``).
        seeds (np.ndarray): Initial particle positions as an ``(n, 3)`` array.
        dt (float): Tracking time step in seconds.
        tmax (float | None): Total tracking duration. Defaults to one flow
            period.
        caps (str | pathlib.Path | list | None): Cap surface path(s) used to
            build a ``BoundaryReseeder`` per worker.
        inlet (np.ndarray | None): Static reset points used when ``caps`` is
            omitted.
        n_workers (int): Number of particle batches/processes.
        active_key (str): Velocity array prefix in the flow files.
        method (str): Integration method, either ``"RK4"`` or ``"Euler"``.
        subsamp (int): Keep every Nth frame when loading ``.pvd`` data.
        only_active_key (bool): Load only velocity arrays for ``.vtu`` inputs.
        pbar (bool): Show a progress bar for the first worker.
        rng (numpy.random.Generator | None): Optional generator for deterministic
            batching and reset draws.
        return_metrics (bool): When ``True``, return ``(result, metrics)``.
        precision (str): Working precision for the sampling/advection math,
            ``"f64"`` (default) or ``"f32"`` (single, faster but less accurate).
        time_interp (str): Temporal interpolation between frames, ``"linear"``
            (default) or ``"cubic"`` (Catmull-Rom; requires uniform spacing).

    Returns:
        (Union[TrackingResult, tuple]): ``TrackingResult`` by
            default, or ``(TrackingResult, metrics)`` when
            ``return_metrics=True``.
    """
    from joblib import Parallel, delayed

    if seeds is None:
        raise ValueError("provide seeds for tracking")
    method = _normalize_method(method)
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    if tmax is None:
        flow = load_flow(path, active_key=active_key, subsamp=subsamp,
                         only_active_key=only_active_key)
        tmax = flow.tmax
        del flow

    rng = rng if rng is not None else np.random.default_rng()
    seeds = np.asarray(seeds, float)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or seeds.shape[0] == 0:
        raise ValueError("seeds must have shape (n_particles, 3)")
    inlet_arr = np.empty((0, 3)) if inlet is None else np.asarray(inlet, float)
    batch_size = int(np.ceil(len(seeds) / n_workers))
    batches = batched_particles(seeds, batch_size, rng=rng)
    rng_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=len(batches))

    results = Parallel(n_jobs=len(batches))(
        delayed(_track_particle_batch)(
            path, batch, inlet_arr, dt, tmax, method=method, active_key=active_key,
            only_active_key=only_active_key, caps=caps, subsamp=subsamp,
            pbar=(pbar and i == 0), rng_seed=int(rng_seeds[i]),
            collect_metrics=return_metrics, precision=precision,
            time_interp=time_interp)
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
