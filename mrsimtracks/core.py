import time
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm


@dataclass
class TrackingResult:
    """Particle tracks from a tracking run."""
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
        """Write positions/reset/dt to an HDF5 file."""
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("position", data=self.positions[::time_subsample])
            f.create_dataset("reset", data=self.reset[::time_subsample])
            f.attrs["dt"] = self.dt * time_subsample


def tracking(flow_mesh, initial_seeds: pv.PolyData, seeding_points: np.ndarray, dt,
             tmax, method="RK4", pbar=True, timings=None, reseeder=None, rng=None):
    # Pass a dict as `timings` to collect a wall-time breakdown of the loop.
    # It is filled in place (non-breaking: the 3-tuple return is unchanged).
    #
    # reseeder: optional object with reseed(n, t) -> (n, 3); when given, OOB
    # particles are recycled to currently-inflow boundary faces (handles
    # backflow) instead of uniformly to the static `seeding_points`.

    if reseeder is None and seeding_points.shape[0] == 0:
        raise ValueError(
            "provide a BoundaryReseeder or non-empty inlet points for recycled "
            "out-of-bounds particles"
        )

    rng = rng if rng is not None else np.random.default_rng()
    nstep = int(tmax/dt)

    # Positions are carried as a plain (n, 3) array; sample_v works on numpy
    # directly, so we avoid wrapping/unwrapping a PolyData every substep.
    r = np.ascontiguousarray(initial_seeds.points, dtype=float).copy()
    n_particles = r.shape[0]

    r_res = np.zeros((nstep, n_particles, 3))
    m_reset_flag = np.zeros((nstep, n_particles))
    n_oob = 0

    # For debugging
    oob_loc_list = []

    # Optional profiling: accumulate time spent in field sampling vs everything else.
    profile = timings is not None
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

    pbar = tqdm(range(nstep), disable=not pbar)

    for i in pbar:

        # Sample current time and position
        k1, valid, cells = _sample(r, i*dt, cells)

        # Reset OOB points
        oob = ~valid
        m_reset_flag[i,oob] = 1
        # Save oob locations
        oob_loc_list.append(r[oob,:])

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
            newpos = seeding_points[rng.integers(
                low=0, high=seeding_points.shape[0], size=n_reset), :]
        r[oob,:] = newpos

        # Recycled particles jumped to the inlet -> their cell guess is stale.
        if cells is not None:
            cells[oob] = -1

        r_res[i,...] = r
        n_oob = np.sum(oob)

        pbar.set_postfix_str(f"n_oob={n_oob}")

    if profile:
        t_total = time.perf_counter() - t0_loop
        timings.update(
            n_particles=n_particles,
            nstep=nstep,
            method=method,
            t_total=t_total,
            t_sample=t_sample,
            t_other=t_total - t_sample,
            n_sample_calls=n_samples,
            sample_frac=(t_sample / t_total) if t_total else 0.0,
            s_per_step=t_total / nstep if nstep else 0.0,
            # Throughput in particle-steps per second (the headline number to beat).
            particle_steps_per_s=(n_particles * nstep / t_total) if t_total else 0.0,
        )

    return r_res, m_reset_flag, oob_loc_list


def track(flow, seeds=None, dt=1e-3, tmax=None, reseeder=None, inlet=None,
          method="RK4", pbar=True, rng=None):
    """Track particles through a loaded flow field.

    Parameters
    ----------
    flow
        Loaded flow field from :func:`mrsimtracks.load_flow`.
    seeds
        Initial particle positions as an ``(n, 3)`` array or ``pyvista.PolyData``.
    reseeder
        Boundary reseeder used to recycle out-of-bounds particles. If omitted,
        ``inlet`` must provide static reset points.
    """
    if seeds is None:
        raise ValueError("provide seeds for tracking")
    seeds = seeds if isinstance(seeds, pv.PolyData) else pv.PolyData(np.asarray(seeds, float))
    if tmax is None:
        tmax = flow.tmax
    inlet_arr = np.empty((0, 3)) if inlet is None else np.asarray(inlet, float)

    pos, reset, _ = tracking(
        flow, seeds, inlet_arr, dt, tmax, method=method, pbar=pbar,
        reseeder=reseeder, rng=rng)
    return TrackingResult(pos, reset, dt)
