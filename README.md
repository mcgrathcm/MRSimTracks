# MRSimTracks

Generate CFD-derived particle trajectories for MR flow simulation.

MRSimTracks performs Lagrangian particle tracking in time-resolved (pulsatile)
CFD meshes. It operates on mesh velocity fields, not MR image data. It seeds
particles in a tetrahedral flow domain, advects them through a time-periodic
velocity field (RK4), and recycles out-of-bounds particles back to the inflow
boundaries with optional **backflow-aware** reseeding.

## Install

```bash
uv sync          # installs the package (editable) and dependencies
```

The distribution name and Python import package are both `mrsimtracks`.

## Quick start

```python
import numpy as np

import mrsimtracks as mt
from mrsimtracks.seeding import seed_mesh

# 1. Load a time-resolved flow field (.vtu single-file series or .pvd collection)
flow = mt.load_flow("case.pvd", active_key="Velocity")

# 2. (optional) Backflow-aware inflow reseeder from labeled cap surfaces
reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)

# 3. Seed and track
seeds = seed_mesh(flow.active_mesh, 200_000, rng=np.random.default_rng(0))
result = mt.track(flow, seeds=seeds, dt=0.002, reseeder=reseeder)

# 4. Use / save
result.positions      # (n_steps, n_particles, 3)
result.reset          # (n_steps, n_particles) reseed flags
result.times          # (n_steps,)
result.save("tracks.h5")
```

Run `example.py` for a complete version using local example data. The large
example flow file is not tracked in normal Git; see `example/README.md`.

## Large runs (multiple processes)

Each worker reloads the field, so memory scales with `n_workers`:

```python
result = mt.track_parallel(
    "case.pvd", seeds=seeds, dt=0.002,
    caps=["Inlet.vtp", "Outlet.vtp"], active_key="Velocity",
    n_workers=3, subsamp=1,
)
```

## Key functions

| Function | Purpose |
|---|---|
| `load_flow(path, active_key=...)` | Load `.vtu` (one geometry, many time fields) or `.pvd` (series); auto-selects the memory-efficient reader. `subsamp=N` keeps every Nth frame. |
| `track(flow, seeds=..., dt=..., reseeder=...)` | Single-process tracking → `TrackingResult`. |
| `track_parallel(path, ..., caps=..., n_workers=...)` | Multi-process tracking → `TrackingResult`. |
| `BoundaryReseeder(caps, flow, dt=...)` | Flux-weighted, time-resolved inflow reseeder. `caps` = cap surface path(s) or a surface with a `region_id` cell array. |

## Reseeding notes

- The reseeder weights every cap face by `max(-v·n, 0)·area` at the nearest flow
  frame, so particles re-enter only through currently-inflow faces — handling
  backflow and partial inflow/outflow on a single cap.
- `BoundaryReseeder(..., dt=dt)` spreads new particles over a thin inflow
  *volume* (depth `~v_n·dt`) so successive reseeds overlap, keeping spatial
  density smooth (important for MR-style uniform-density use). Omit `dt` for
  plane seeding.
- `flux_waveform()` returns per-cap net flux over the cycle — a conservation /
  validation diagnostic (`Σ caps ≈ 0` for a well-resolved incompressible field).

## Performance

Sampling reuses a cell-tree locator built once, a temporal-coherence tet walk
(numba) seeded from each particle's previous cell, and fused walk+interpolation.
`benchmark.py` measures sampling and tracking throughput.

## Development

Normal CI runs against a reduced real-data fixture:

```bash
uv sync --group dev
uv run pytest -m "not large" --cov=mrsimtracks --cov-report=term-missing
```

Full-data validation uses the Git LFS example file and runs only for release
validation:

```bash
git lfs pull --include="example/CFD_velocity.vtu"
uv run pytest -m large
```

See `CONTRIBUTING.md` for the full development workflow.

## Documentation

Documentation is built with MkDocs:

```bash
uv sync --group docs
uv run --group docs mkdocs build --strict
```

The GitHub Pages site is deployed from `main`.

## Notes

- Flow meshes are assumed all-tetrahedral and static in time (the field varies,
  the geometry does not).
- `.pvd` loading stores one geometry plus one field per frame for static-mesh
  series, so long time series fit in a few GB instead of tens.

## License

MIT.
