# particle_tracking

Lagrangian particle tracking in time-resolved (pulsatile) CFD meshes. Seeds
particles in a tetrahedral flow domain, advects them through a time-periodic
velocity field (RK4), and recycles out-of-bounds particles back to the inflow
boundaries — with optional **backflow-aware** reseeding that only re-injects
where flow is actually entering the domain.

## Install

```bash
uv sync          # installs the package (editable) and dependencies
```

## Quick start

```python
import particle_tracking as pt

# 1. Load a time-resolved flow field (.vtu single-file series or .pvd collection)
flow = pt.load_flow("case.pvd", active_key="Velocity")

# 2. (optional) Backflow-aware inflow reseeder from labeled cap surfaces
reseeder = pt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)

# 3. Track (seeds the volume, advects, recycles to inflow caps)
result = pt.track(flow, n_particles=2e5, dt=0.002, reseeder=reseeder)

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
result = pt.track_parallel(
    "case.pvd", n_particles=2e6, dt=0.002,
    caps=["Inlet.vtp", "Outlet.vtp"], active_key="Velocity",
    n_workers=3, subsamp=1,
)
```

## Key functions

| Function | Purpose |
|---|---|
| `load_flow(path, active_key=...)` | Load `.vtu` (one geometry, many time fields) or `.pvd` (series); auto-selects the memory-efficient reader. `subsamp=N` keeps every Nth frame. |
| `track(flow, n_particles=..., dt=..., reseeder=...)` | Single-process tracking → `TrackingResult`. |
| `track_parallel(path, ..., caps=..., n_workers=...)` | Multi-process tracking → `TrackingResult`. |
| `BoundaryReseeder(caps, flow, dt=...)` | Flux-weighted, time-resolved inflow reseeder. `caps` = cap surface path(s) or a surface with a `region_id` cell array. |
| `extract_caps("case.vtu", active_key="Velocity")` | Reconstruct labeled inlet/outlet caps from a volume mesh when none are provided (uses the no-slip-wall signal: walls have v≈0, caps carry flow). |

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
uv run pytest -m "not large"
```

Full-data validation uses the Git LFS example file and runs only for release
validation:

```bash
git lfs pull --include="example/CFD_velocity.vtu"
uv run pytest -m large
```

See `CONTRIBUTING.md` for the full development workflow.

## Notes

- Flow meshes are assumed all-tetrahedral and static in time (the field varies,
  the geometry does not).
- `timeMeshStaticPVD` stores one geometry + one field per frame, so a long `.pvd`
  series fits in a few GB instead of tens; `timeMeshPVD` (full mesh per frame) is
  retained for reference.

## License

MIT.
