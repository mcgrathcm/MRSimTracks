"""Benchmark harness for the particle tracking pipeline.

Establishes baseline numbers before performance work so changes can be measured.
It times three things independently:

  1. mesh load        -- one-time cost of reading the flow file
  2. raw sampling     -- the core hotspot: flow.sample() throughput vs particle count
  3. full RK4 step    -- end-to-end tracking throughput (sample + advect + reset)

Run:
    uv run python benchmark.py                         # defaults (small, fast)
    uv run python benchmark.py --counts 1e4,1e5,1e6    # sweep particle counts
    uv run python benchmark.py --nsteps 20 --json out.json
"""

import argparse
import json
import time

import numpy as np
import pyvista as pv

import tracking


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


def _time_call(fn, pts, repeats):
    fn(pts, 0.0)  # warm up locator / caches
    t0 = time.perf_counter()
    for r in range(repeats):
        fn(pts, r * 1e-4)
    return (time.perf_counter() - t0) / repeats


def bench_sampling(flow, counts, repeats, rng):
    """Microbenchmark: old flow.sample() vs new flow.sample_v() per particle count."""
    print("\n=== Raw sampling throughput: sample (old) vs sample_v (fast) ===")
    print(f"{'particles':>12} {'old ms':>9} {'fast ms':>9} {'speedup':>9} {'Mpts/s':>9}")
    rows = []
    for n in counts:
        pts = make_particles(flow, n, rng)
        t_old = _time_call(lambda p, t: flow.sample(p, t), pts, repeats)
        t_new = _time_call(lambda p, t: flow.sample_v(p.points, t), pts, repeats)
        print(f"{_fmt(n):>12} {t_old*1e3:>9.1f} {t_new*1e3:>9.1f} "
              f"{t_old/t_new:>8.1f}x {n/t_new/1e6:>9.2f}")
        rows.append(dict(n=int(n), old_ms=t_old * 1e3, fast_ms=t_new * 1e3,
                         speedup=t_old / t_new))
    return rows


def bench_tracking(flow, inlet, counts, nsteps, dt, method, rng):
    """End-to-end tracking throughput with the in-loop timing breakdown."""
    print(f"\n=== Full tracking step ({method}, {nsteps} steps, dt={dt}) ===")
    print(f"{'particles':>12} {'ms/step':>10} {'Mpstep/s':>10} {'sample%':>9} {'#samp/step':>11}")
    rows = []
    for n in counts:
        seeds = make_particles(flow, n, rng)
        timings = {}
        tracking.tracking(flow, seeds, inlet, dt, tmax=nsteps * dt,
                          method=method, pbar=False, timings=timings)
        mpstep = timings["particle_steps_per_s"] / 1e6
        print(f"{_fmt(n):>12} {timings['s_per_step']*1e3:>10.2f} {mpstep:>10.3f} "
              f"{timings['sample_frac']*100:>8.1f}% {timings['n_sample_calls']/nsteps:>11.1f}")
        rows.append(dict(n=int(n), **{k: v for k, v in timings.items()
                                      if k not in ("method",)}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", default="P015_pulsatile_rigid_nobackflow.vtu")
    ap.add_argument("--inlet", default="P015_inlet7670.vtu")
    ap.add_argument("--counts", default="1e3,1e4,1e5",
                    help="comma-separated particle counts, e.g. 1e4,1e5,1e6")
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--method", default="RK4", choices=["RK4", "Euler"])
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats for the sampling microbenchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sampling", action="store_true", help="skip sampling microbenchmark")
    ap.add_argument("--json", default=None, help="write results to this JSON file")
    args = ap.parse_args()

    counts = [float(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)

    print(f"flow file: {args.flow}")
    t0 = time.perf_counter()
    flow = tracking.timeMeshSingleVTU(args.flow)
    t_load = time.perf_counter() - t0
    print(f"load: {t_load:.1f}s | {flow.mesh.n_points:,} points | "
          f"{flow.mesh.n_cells:,} cells | {len(flow.times)} timesteps")

    # A handful of inlet points is enough; OOB reset just needs somewhere to go.
    inlet = flow.active_mesh.points[:1000].copy()

    results = {"flow": args.flow, "n_points": int(flow.mesh.n_points),
               "n_cells": int(flow.mesh.n_cells), "n_timesteps": len(flow.times),
               "t_load_s": t_load}

    if not args.no_sampling:
        results["sampling"] = bench_sampling(flow, counts, args.repeats, rng)
    results["tracking"] = bench_tracking(flow, inlet, counts, args.nsteps,
                                         args.dt, args.method, rng)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
