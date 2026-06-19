"""Generate and compare against a high-quality ground-truth (GT) trajectory set.

The GT is a reference particle-tracking run at a small timestep in double
precision (f64, RK4) -- accurate enough that cheaper configurations (larger dt,
single precision, Euler) can be measured against it to quantify their error.

Two modes:

    # 1. produce the reference (small dt, f64) and save it self-describing
    uv run python scripts/ground_truth.py generate --dt 1e-4 --out ground_truth.h5

    # 2. run a candidate config on the SAME seeds and report its error vs the GT
    uv run python scripts/ground_truth.py compare ground_truth.h5 --dt 5e-4
    uv run python scripts/ground_truth.py compare ground_truth.h5 --dt 1e-4 --method Euler

The candidate dt must be an integer multiple of the GT dt so their stored output
times line up; the GT is subsampled to the candidate's times for the comparison.

Error is reported only over particles that have not been recycled (gone
out-of-bounds) in either run up to the step in question, so the boundary
reseeding draw -- which is not identical across dt -- never pollutes the accuracy
numbers. The masked-out fraction is reported alongside.
"""

import argparse
import inspect
import json
from pathlib import Path

import h5py
import numpy as np

import mrsimtracks as mt

DATA = Path(__file__).parents[1] / "tests" / "data"
DEFAULT_FLOW = DATA / "CFD_velocity_00190_00210.vtu"
DEFAULT_CAPS = [DATA / "Inlet.vtp", DATA / "Outlet.vtp"]
DEFAULT_ACTIVE_KEY = "Velocity"


def _load_flow(path, active_key, precision):
    """Load a flow, forwarding ``precision`` only if this build supports it.

    The f32/f64 ``precision`` knob may not be present on every checkout. f64 is
    always available; requesting f32 without support is a clear error rather than
    a silent f64 run.
    """
    if "precision" in inspect.signature(mt.load_flow).parameters:
        return mt.load_flow(path, active_key=active_key, precision=precision)
    if precision not in ("f64", "float64", "double"):
        raise SystemExit(
            f"precision={precision!r} is not supported by this mrsimtracks build "
            "(only f64). Merge the single-precision PR to compare f32.")
    return mt.load_flow(path, active_key=active_key)


def cell_center_seeds(flow, n):
    """Deterministic, in-domain seeds: evenly spaced tet cell centers."""
    centers = flow.active_mesh.cell_centers().points
    idx = np.linspace(0, len(centers) - 1, n, dtype=int)
    return np.ascontiguousarray(centers[idx], dtype=np.float64)


def _make_reseeder(flow, caps, dt, seed):
    return mt.BoundaryReseeder(
        [str(c) for c in caps], flow, dt=dt, rng=np.random.default_rng(seed))


def run_tracking(flow, seeds, dt, tmax, method, caps, seed):
    """Track ``seeds`` and return (positions, reset) as numpy arrays."""
    reseeder = _make_reseeder(flow, caps, dt, seed) if caps else None
    inlet = None if caps else flow.active_mesh.points[:1000].copy()
    result = mt.track(flow, seeds=seeds, dt=dt, tmax=tmax, method=method,
                      reseeder=reseeder, inlet=inlet,
                      rng=np.random.default_rng(seed), pbar=False)
    return np.asarray(result.positions), np.asarray(result.reset)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #

def generate(args):
    flow = _load_flow(args.flow, args.active_key, args.precision)
    tmax = flow.tmax if args.tmax is None else args.tmax
    caps = [] if args.no_reseed else args.caps
    seeds = cell_center_seeds(flow, args.n)

    print(f"GT: dt={args.dt:g} method={args.method} precision={args.precision} "
          f"tmax={tmax:g} -> {int(round(tmax/args.dt))} steps, {args.n} particles")
    pos, reset = run_tracking(flow, seeds, args.dt, tmax, args.method, caps, args.seed)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("position", data=pos, compression="gzip")
        f.create_dataset("reset", data=reset, compression="gzip")
        f.create_dataset("seeds", data=seeds)
        f.attrs.update(dt=args.dt, tmax=tmax, method=args.method,
                       precision=args.precision, n_particles=args.n,
                       active_key=args.active_key, flow=str(args.flow),
                       reseeded=not args.no_reseed, seed=args.seed)
    reset_frac = float(reset.mean())
    print(f"wrote {args.out}  ({pos.shape[0]} steps x {pos.shape[1]} particles, "
          f"reset_fraction={reset_frac:.3f})")


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def trajectory_error(gt_pos, gt_reset, cand_pos, cand_reset, k):
    """Per-step displacement error of a candidate vs GT, masking recycled particles.

    GT output step ``i`` holds the position at time ``(i+1)*dt_gt``; candidate
    step ``j`` holds ``(j+1)*dt_cand = (j+1)*k*dt_gt`` -- i.e. GT step
    ``(j+1)*k - 1``. We subsample the GT on that stride and compare element-wise.

    A particle is excluded from a step once it has reset in either run at or
    before that step (cumulative), so reseeding nondeterminism is removed.
    Returns ``(err, clean)`` arrays of shape ``(n_cand, n_particles)``; ``err``
    is NaN where masked.
    """
    n_cand = cand_pos.shape[0]
    gt_al = gt_pos[k - 1::k][:n_cand]
    gt_rst = gt_reset[k - 1::k][:n_cand]
    if gt_al.shape != cand_pos.shape:
        raise SystemExit(
            f"shape mismatch after alignment: GT {gt_al.shape} vs candidate "
            f"{cand_pos.shape}; check that tmax and dt ratios match.")
    clean = (np.cumsum(gt_rst, axis=0) + np.cumsum(cand_reset, axis=0)) == 0
    err = np.linalg.norm(cand_pos - gt_al, axis=2)
    return np.where(clean, err, np.nan), clean


def _stats(err_row, scale):
    vals = err_row[np.isfinite(err_row)]
    if vals.size == 0:
        return dict(n=0, median=float("nan"), p95=float("nan"), max=float("nan"),
                    median_rel=float("nan"))
    return dict(n=int(vals.size),
                median=float(np.median(vals)),
                p95=float(np.percentile(vals, 95)),
                max=float(vals.max()),
                median_rel=float(np.median(vals) / scale))


def compare(args):
    with h5py.File(args.gt, "r") as f:
        gt_pos = f["position"][...]
        gt_reset = f["reset"][...]
        seeds = f["seeds"][...]
        gt = dict(f.attrs)

    dt_gt = float(gt["dt"])
    tmax = float(gt["tmax"])
    ratio = args.dt / dt_gt
    k = int(round(ratio))
    if abs(ratio - k) > 1e-6 or k < 1:
        raise SystemExit(
            f"candidate dt ({args.dt:g}) must be an integer multiple of the GT dt "
            f"({dt_gt:g}); got ratio {ratio:.4f}.")

    flow = _load_flow(gt["flow"], gt["active_key"], args.precision)
    caps = [] if args.no_reseed else args.caps
    cand_pos, cand_reset = run_tracking(
        flow, seeds, args.dt, tmax, args.method, caps, args.seed)

    err, clean = trajectory_error(gt_pos, gt_reset, cand_pos, cand_reset, k)
    diag = float(np.linalg.norm(np.ptp(seeds, axis=0)))  # domain scale for rel error

    n_steps = err.shape[0]
    checkpoints = sorted(set([max(1, n_steps // 4), n_steps // 2,
                              (3 * n_steps) // 4, n_steps]))
    print(f"\nGT: dt={dt_gt:g} {gt['method']} {gt['precision']}  vs  "
          f"candidate: dt={args.dt:g} (k={k}) {args.method} {args.precision}")
    print(f"domain scale (seed bbox diagonal) = {diag:.3g}\n")
    print(f"{'step':>6} {'time':>9} {'clean':>7} {'med err':>10} {'p95 err':>10} "
          f"{'max err':>10} {'med/scale':>10}")
    rows = []
    for s in checkpoints:
        st = _stats(err[s - 1], diag)
        t = s * args.dt
        print(f"{s:>6} {t:>9.4g} {st['n']:>7} {st['median']:>10.3e} "
              f"{st['p95']:>10.3e} {st['max']:>10.3e} {st['median_rel']:>10.2e}")
        rows.append(dict(step=s, time=t, **st))

    final = _stats(err[-1], diag)
    masked_frac = float(1.0 - np.isfinite(err[-1]).mean())
    summary = {
        "gt": {k_: (v.item() if hasattr(v, "item") else v) for k_, v in gt.items()},
        "candidate": dict(dt=args.dt, method=args.method, precision=args.precision,
                          k=k, seed=args.seed, reseeded=not args.no_reseed),
        "domain_scale": diag,
        "final": final,
        "final_masked_fraction": masked_frac,
        "checkpoints": rows,
    }
    print(f"\nfinal-step median error {final['median']:.3e} "
          f"({final['median_rel']:.2e} of domain scale), "
          f"{masked_frac*100:.1f}% of particles masked (recycled).")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json}")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def _add_common(p):
    p.add_argument("--method", default="RK4", choices=["RK4", "Euler"])
    p.add_argument("--precision", default="f64",
                   help="f64 (default) or f32 (requires single-precision support)")
    p.add_argument("--caps", nargs="+", type=Path, default=DEFAULT_CAPS,
                   help="boundary cap surfaces for flux-weighted reseeding")
    p.add_argument("--no-reseed", action="store_true",
                   help="disable boundary reseeding (static inlet for OOB resets)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json", default=None, help="write results/metadata to JSON")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="produce a GT reference run")
    g.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    g.add_argument("--active-key", default=DEFAULT_ACTIVE_KEY)
    g.add_argument("--dt", type=float, default=1e-4, help="GT timestep (small)")
    g.add_argument("--tmax", type=float, default=None, help="default: one flow period")
    g.add_argument("--n", type=int, default=256, help="number of particles")
    g.add_argument("--out", type=Path, default="ground_truth.h5")
    _add_common(g)
    g.set_defaults(func=generate)

    c = sub.add_parser("compare", help="run a candidate config and report error vs GT")
    c.add_argument("gt", type=Path, help="GT .h5 produced by `generate`")
    c.add_argument("--dt", type=float, required=True,
                   help="candidate timestep (integer multiple of GT dt)")
    _add_common(c)
    c.set_defaults(func=compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
