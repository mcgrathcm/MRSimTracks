import time
from pathlib import Path

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm


class TrackingResult:
    """Particle trajectories and reset flags from a tracking run.

    Args:
        positions (np.ndarray | None): In-memory particle positions with shape
            ``(n_steps, n_particles, 3)``.
        reset (np.ndarray | None): In-memory reset flags with shape
            ``(n_steps, n_particles)``.
        dt (float | None): Tracking time step in seconds.
        path (str | pathlib.Path | None): HDF5 file path for a file-backed
            result.
        shape (tuple[int, int, int] | None): Position dataset shape for
            file-backed results.
        metrics (dict | None): Optional timing/throughput metrics returned by
            tracking.
    """

    def __init__(self, positions=None, reset=None, dt=None, *, path=None,
                 shape=None, metrics=None):
        if positions is None and path is None:
            raise ValueError("TrackingResult requires positions or an HDF5 path")
        if positions is not None and reset is None:
            raise ValueError("TrackingResult requires reset flags with positions")

        self._positions = None if positions is None else np.asarray(positions)
        self._reset = None if reset is None else np.asarray(reset)
        self.dt = self._read_dt(path) if dt is None and path is not None else dt
        self.path = None if path is None else Path(path)
        self.metrics = dict(metrics or {})

        if shape is not None:
            self._shape = tuple(shape)
        elif self._positions is not None:
            self._shape = self._positions.shape
        elif self.path is not None:
            self._shape = self._read_shape(self.path)
        else:
            self._shape = None

    @property
    def is_file_backed(self):
        return self.path is not None and self._positions is None

    @property
    def positions(self):
        if self._positions is None:
            import h5py
            with h5py.File(self.path, "r") as f:
                self._positions = f["position"][...]
        return self._positions

    @property
    def reset(self):
        if self._reset is None:
            import h5py
            with h5py.File(self.path, "r") as f:
                self._reset = f["reset"][...]
        return self._reset

    @property
    def n_steps(self):
        return self._shape[0]

    @property
    def n_particles(self):
        return self._shape[1]

    @property
    def times(self):
        return np.arange(self.n_steps) * self.dt

    @classmethod
    def open(cls, path):
        """Open a file-backed result without loading positions into memory."""
        return cls(path=path)

    def save(self, path, time_subsample=1):
        """Write positions/reset/dt to an HDF5 file."""
        import h5py

        if time_subsample < 1:
            raise ValueError("time_subsample must be >= 1")

        if self.path is not None and Path(path) == self.path and time_subsample == 1:
            return

        if self.path is None:
            with h5py.File(path, "w") as f:
                f.create_dataset("position", data=self.positions[::time_subsample])
                f.create_dataset("reset", data=self.reset[::time_subsample])
                f.attrs["dt"] = self.dt * time_subsample
            return

        with h5py.File(self.path, "r") as src, h5py.File(path, "w") as dst:
            src_pos = src["position"]
            src_reset = src["reset"]
            n_out = len(range(0, src_pos.shape[0], time_subsample))
            pos = dst.create_dataset(
                "position",
                shape=(n_out, src_pos.shape[1], 3),
                dtype=src_pos.dtype,
                chunks=_hdf5_chunks((n_out, src_pos.shape[1], 3)),
            )
            reset = dst.create_dataset(
                "reset",
                shape=(n_out, src_reset.shape[1]),
                dtype=src_reset.dtype,
                chunks=_hdf5_chunks((n_out, src_reset.shape[1])),
            )
            for out_i, src_i in enumerate(range(0, src_pos.shape[0], time_subsample)):
                pos[out_i] = src_pos[src_i]
                reset[out_i] = src_reset[src_i]
            dst.attrs["dt"] = self.dt * time_subsample

    @staticmethod
    def _read_shape(path):
        import h5py
        with h5py.File(path, "r") as f:
            return f["position"].shape

    @staticmethod
    def _read_dt(path):
        import h5py
        with h5py.File(path, "r") as f:
            return float(f.attrs["dt"])


class _HDF5TrackWriter:
    def __init__(self, path, n_steps, n_particles, dt):
        import h5py

        self.path = Path(path)
        self.file = h5py.File(self.path, "w")
        self.positions = self.file.create_dataset(
            "position",
            shape=(n_steps, n_particles, 3),
            dtype=np.float64,
            chunks=_hdf5_chunks((n_steps, n_particles, 3)),
        )
        self.reset = self.file.create_dataset(
            "reset",
            shape=(n_steps, n_particles),
            dtype=bool,
            chunks=_hdf5_chunks((n_steps, n_particles)),
        )
        self.file.attrs["dt"] = dt
        self.file.attrs["n_steps"] = n_steps
        self.file.attrs["n_particles"] = n_particles

    def write_step(self, index, positions, reset_flags):
        self.positions[index] = positions
        self.reset[index] = reset_flags

    def close(self):
        self.file.close()


def _hdf5_chunks(shape):
    if shape[0] == 0 or shape[1] == 0:
        return None
    if len(shape) == 3:
        return (1, min(shape[1], 65_536), shape[2])
    return (1, min(shape[1], 65_536))


def _normalize_method(method):
    methods = {
        "rk4": "RK4",
        "euler": "Euler",
    }
    try:
        return methods[method.lower()]
    except AttributeError as exc:
        raise TypeError("method must be a string") from exc
    except KeyError as exc:
        valid = ", ".join(sorted(set(methods.values())))
        raise ValueError(f"unsupported tracking method {method!r}; expected one of: {valid}") from exc


def _track_particles(flow_mesh, initial_seeds: pv.PolyData, reset_points: np.ndarray,
                     dt, tmax, method="RK4", pbar=True, metrics=None,
                     reseeder=None, rng=None, step_writer=None):
    # `metrics` is filled in place with a wall-time breakdown when provided.
    # reseeder: optional object with reseed(n, t) -> (n, 3); when given, OOB
    # particles are recycled to currently-inflow boundary faces (handles
    # backflow) instead of uniformly to the static `reset_points`.

    if reseeder is None and reset_points.shape[0] == 0:
        raise ValueError(
            "provide a BoundaryReseeder or non-empty inlet points for recycled "
            "out-of-bounds particles"
        )

    rng = rng if rng is not None else np.random.default_rng()
    method = _normalize_method(method)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if tmax <= 0:
        raise ValueError("tmax must be > 0")
    n_steps = int(tmax/dt)
    if n_steps < 1:
        raise ValueError("tmax must be at least one dt")

    # Positions are carried as a plain (n, 3) array; sample_v works on numpy
    # directly, so we avoid wrapping/unwrapping a PolyData every substep.
    r = np.ascontiguousarray(initial_seeds.points, dtype=float).copy()
    n_particles = r.shape[0]
    if n_particles < 1:
        raise ValueError("provide at least one seed point")

    if step_writer is None:
        positions = np.zeros((n_steps, n_particles, 3))
        reset_flags = np.zeros((n_steps, n_particles), dtype=bool)
    else:
        positions = None
        reset_flags = None
    n_oob = 0

    # Optional profiling: accumulate time spent in field sampling vs everything else.
    profile = metrics is not None
    n_samples = 0
    t_sample = 0.0
    t0_loop = time.perf_counter()

    def _sample(points, t, guess):
        nonlocal t_sample, n_samples
        if profile:
            _t = time.perf_counter()
            out = flow_mesh.sample_v(points, t, guess=guess)
            t_sample += time.perf_counter() - _t
            n_samples += 1
            return out
        return flow_mesh.sample_v(points, t, guess=guess)

    # Per-particle cell guess for the temporal-coherence walk; None on the first
    # step (cold locator) and reset to -1 for recycled particles (forces a probe).
    cells = None

    pbar = tqdm(range(n_steps), disable=not pbar)

    for i in pbar:

        # Sample current time and position
        k1, valid, cells = _sample(r, i*dt, cells)

        # Reset OOB points
        oob = ~valid
        if reset_flags is not None:
            reset_flags[i, oob] = True

        # Get velocity step
        if method == "RK4":
            # Substep positions stay within ~1 cell, so reuse the running cell
            # guess to seed each walk.
            k2, _, c = _sample(k1*dt/2 + r, i*dt + dt/2, cells)
            k3, _, c = _sample(k2*dt/2 + r, i*dt + dt/2, c)
            k4, _, _ = _sample(k3*dt + r, i*dt + dt, c)

            v = (k1 + 2*k2 + 2*k3 + k4)/6
        else:
            v = k1

        # Advect
        r = r + v*dt

        # Move OOB back to the inlet. With a reseeder, recycle to currently-inflow
        # boundary faces (backflow-aware); otherwise draw from the static cloud.
        n_reset = int(np.sum(oob))
        if reseeder is not None:
            newpos = reseeder.reseed(n_reset, i*dt)
        else:
            newpos = reset_points[rng.integers(
                low=0, high=reset_points.shape[0], size=n_reset), :]
        r[oob, :] = newpos

        # Recycled particles jumped to the inlet -> their cell guess is stale.
        if cells is not None:
            cells[oob] = -1

        if step_writer is None:
            positions[i, ...] = r
        else:
            step_writer.write_step(i, r, oob)
        n_oob = np.sum(oob)

        pbar.set_postfix_str(f"n_oob={n_oob}")

    if profile:
        t_total = time.perf_counter() - t0_loop
        metrics.update(
            n_particles=n_particles,
            n_steps=n_steps,
            method=method,
            t_total=t_total,
            t_sample=t_sample,
            t_other=t_total - t_sample,
            n_sample_calls=n_samples,
            sample_frac=(t_sample / t_total) if t_total else 0.0,
            s_per_step=t_total / n_steps if n_steps else 0.0,
            # Throughput in particle-steps per second (the headline number to beat).
            particle_steps_per_s=(n_particles * n_steps / t_total) if t_total else 0.0,
        )

    return positions, reset_flags


def track(flow, seeds=None, dt=1e-3, tmax=None, reseeder=None, inlet=None,
          method="RK4", pbar=True, rng=None, output_path=None,
          return_metrics=False):
    """Track particles through a loaded flow field.

    Args:
        flow (object): Loaded flow field from :func:`mrsimtracks.load_flow`.
        seeds (np.ndarray | pyvista.PolyData): Initial particle positions as an ``(n, 3)`` array or
            ``pyvista.PolyData``.
        dt (float): Tracking time step in seconds.
        tmax (float | None): Total tracking duration. Defaults to one flow
            period.
        reseeder (BoundaryReseeder | None): Boundary reseeder used to recycle
            out-of-bounds particles. If omitted, ``inlet`` must provide static
            reset points.
        inlet (np.ndarray | None): Static reset points used when ``reseeder`` is
            omitted.
        method (str): Integration method, either ``"RK4"`` or ``"Euler"``.
        pbar (bool): Show a progress bar.
        rng (numpy.random.Generator | None): Optional generator for
            deterministic reset draws.
        output_path (str | pathlib.Path | None): Optional HDF5 path. When
            provided, positions/reset flags are streamed to disk and the
            returned result is file-backed until arrays are accessed.
        return_metrics (bool): When ``True``, return ``(result, metrics)`` with
            loop timing metrics.

    Returns:
        (Union[TrackingResult, tuple]): ``TrackingResult`` by
            default, or ``(TrackingResult, metrics)`` when
            ``return_metrics=True``.
    """
    if seeds is None:
        raise ValueError("provide seeds for tracking")
    method = _normalize_method(method)
    if not isinstance(seeds, pv.PolyData):
        seed_arr = np.asarray(seeds, float)
        if seed_arr.ndim != 2 or seed_arr.shape[1] != 3 or seed_arr.shape[0] == 0:
            raise ValueError("seeds must have shape (n_particles, 3)")
        seeds = pv.PolyData(seed_arr)
    elif seeds.n_points == 0:
        raise ValueError("provide at least one seed point")
    if tmax is None:
        tmax = flow.tmax
    inlet_arr = np.empty((0, 3)) if inlet is None else np.asarray(inlet, float)
    metrics = {} if return_metrics else None
    n_steps = int(tmax / dt)

    writer = None
    try:
        if output_path is not None:
            writer = _HDF5TrackWriter(output_path, n_steps, seeds.n_points, dt)
        pos, reset = _track_particles(
            flow, seeds, inlet_arr, dt, tmax, method=method, pbar=pbar,
            metrics=metrics, reseeder=reseeder, rng=rng, step_writer=writer)
    finally:
        if writer is not None:
            writer.close()

    if output_path is None:
        result = TrackingResult(pos, reset, dt, metrics=metrics)
    else:
        result = TrackingResult(
            dt=dt,
            path=output_path,
            shape=(n_steps, seeds.n_points, 3),
            metrics=metrics,
        )
    if return_metrics:
        return result, metrics
    return result
