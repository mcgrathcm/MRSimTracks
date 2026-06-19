"""Benchmark harness for the particle tracking pipeline.

Establishes baseline numbers so performance work can be measured. It times three
things independently:

  1. mesh load        -- one-time cost of reading the flow file
  2. raw sampling     -- the core hotspot: flow.sample() vs sample_v() throughput
  3. full RK4 step    -- end-to-end tracking throughput (sample + advect + reset)

Timing methodology: every measurement runs one or more warmup calls first (which
also trigger numba JIT compilation of the sampler kernel), then reports the
**median** of `--repeats` timed runs plus the fastest run. Median resists the
occasional GC/scheduler outlier that skews a mean, and the min approximates the
noise floor.

Run:
    uv run python benchmark.py                         # defaults (small fixture)
    uv run python benchmark.py --counts 1e4,1e5,1e6    # sweep particle counts
    uv run python benchmark.py --nsteps 20 --json out.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from mrsimtracks.core import _track_particles
from mrsimtracks.io import SingleVTUFlow

# Committed small fixture (190-210 ms window of the example case): ~197k points,
# ~1.1M tet cells, 11 timesteps. Small enough to run out of the box, large
# enough to be representative of the sampler's per-cell costs.
DEFAULT_FLOW = Path(__file__).parent / "tests" / "data" / "CFD_velocity_00190_00210.vtu"
DEFAULT_ACTIVE_KEY = "Velocity"


def _fmt(n):
    return f"{n:,.0f}" if n >= 100 else f"{n:.3g}"


def make_particles(flow, n, rng):
    """Sample n points uniformly from the mesh's own node coordinates.

    Cheap and guaranteed in-domain, so the benchmark measures sampling/advection
    cost rather than seed_region's enclosed-point geometry (timed separately).
    """
    pts = flow.active_mesh.points
    idx = rng.integers(0, pts.shape[0], size=int(n))
    return pv.PolyData(pts[idx].copy())


def _bench(fn, repeats, warmups=1):
    """Return (median_s, min_s) over `repeats` timed calls of ``fn()``.

    ``warmups`` untimed calls run first to prime locator/field caches and to
    JIT-compile the numba kernel, so neither is charged to the measurement.
    """
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def bench_sampling(flow, t_bench, counts, repeats, rng):
    """Microbenchmark: old flow.sample() vs new flow.sample_v() per particle count."""
    print("\n=== Raw sampling throughput: sample (old) vs sample_v (fast) ===")
    print(f"{'particles':>12} {'old ms':>9} {'fast ms':>9} {'speedup':>9} "
          f"{'Mpts/s':>9} {'best ms':>9}")
    rows = []
    for n in counts:
        pts = make_particles(flow, n, rng)
        xyz = pts.points
        old_med, _ = _bench(lambda: flow.sample(pts, t_bench), repeats)
        new_med, new_min = _bench(lambda: flow.sample_v(xyz, t_bench), repeats)
        print(f"{_fmt(n):>12} {old_med*1e3:>9.1f} {new_med*1e3:>9.1f} "
              f"{old_med/new_med:>8.1f}x {n/new_med/1e6:>9.2f} {new_min*1e3:>9.1f}")
        rows.append(dict(n=int(n), old_ms=old_med * 1e3, fast_ms=new_med * 1e3,
                         fast_best_ms=new_min * 1e3, speedup=old_med / new_med,
                         fast_mpts_per_s=n / new_med / 1e6))
    return rows


def bench_tracking(flow, inlet, counts, nsteps, dt, method, repeats, rng):
    """End-to-end tracking throughput with the in-loop timing breakdown.

    Reports the median particle-steps/s over `repeats` runs (after a warmup that
    also compiles the numba kernel). The sample-fraction breakdown is taken from
    a representative (median-throughput) run.
    """
    print(f"\n=== Full tracking step ({method}, {nsteps} steps, dt={dt}) ===")
    print(f"{'particles':>12} {'ms/step':>10} {'Mpstep/s':>10} {'best Mpstep/s':>14} "
          f"{'sample%':>9} {'#samp/step':>11}")
    rows = []
    tmax = nsteps * dt
    for n in counts:
        seeds = make_particles(flow, n, rng)

        def run():
            m = {}
            _track_particles(flow, seeds, inlet, dt, tmax=tmax,
                             method=method, pbar=False, metrics=m)
            return m

        run()  # warmup + numba JIT
        runs = [run() for _ in range(repeats)]
        runs.sort(key=lambda m: m["particle_steps_per_s"])
        rep = runs[len(runs) // 2]                 # median-throughput run
        best = runs[-1]
        thr = rep["particle_steps_per_s"]
        ms_step = n / thr * 1e3                     # t_total/nsteps = n/throughput
        print(f"{_fmt(n):>12} {ms_step:>10.2f} {thr/1e6:>10.3f} "
              f"{best['particle_steps_per_s']/1e6:>14.3f} "
              f"{rep['sample_frac']*100:>8.1f}% {rep['n_sample_calls']/nsteps:>11.1f}")
        rows.append(dict(n=int(n),
                         particle_steps_per_s=thr,
                         best_particle_steps_per_s=best["particle_steps_per_s"],
                         s_per_step=n / thr,
                         sample_frac=rep["sample_frac"],
                         n_sample_calls=rep["n_sample_calls"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", default=str(DEFAULT_FLOW))
    ap.add_argument("--active-key", default=DEFAULT_ACTIVE_KEY,
                    help="velocity array prefix in the flow file")
    ap.add_argument("--counts", default="1e3,1e4,1e5",
                    help="comma-separated particle counts, e.g. 1e4,1e5,1e6")
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--method", default="RK4", choices=["RK4", "Euler"])
    ap.add_argument("--repeats", type=int, default=5,
                    help="timed repeats per measurement (median reported)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sampling", action="store_true", help="skip sampling microbenchmark")
    ap.add_argument("--json", default=None, help="write results to this JSON file")
    args = ap.parse_args()

    counts = [float(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)

    print(f"flow file: {args.flow}")
    t0 = time.perf_counter()
    flow = SingleVTUFlow(args.flow, active_key=args.active_key)
    t_load = time.perf_counter() - t0
    print(f"load: {t_load:.1f}s | {flow.mesh.n_points:,} points | "
          f"{flow.mesh.n_cells:,} cells | {len(flow.times)} timesteps | "
          f"repeats={args.repeats}")

    # A handful of inlet points is enough; OOB reset just needs somewhere to go.
    inlet = flow.active_mesh.points[:1000].copy()

    # Sample at a time strictly between two frames so the field time-blend (not
    # the keyframe shortcut) is exercised -- the realistic per-call path.
    t_bench = 0.45 * flow.tmax

    results = {"flow": args.flow, "n_points": int(flow.mesh.n_points),
               "n_cells": int(flow.mesh.n_cells), "n_timesteps": len(flow.times),
               "t_load_s": t_load, "repeats": args.repeats, "method": args.method,
               "nsteps": args.nsteps, "dt": args.dt}

    if not args.no_sampling:
        results["sampling"] = bench_sampling(flow, t_bench, counts, args.repeats, rng)
    results["tracking"] = bench_tracking(flow, inlet, counts, args.nsteps,
                                         args.dt, args.method, args.repeats, rng)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
