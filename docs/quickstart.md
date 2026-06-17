# Quick Start

```python
import particle_tracking as pt

flow = pt.load_flow("case.pvd", active_key="Velocity")
reseeder = pt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)

result = pt.track(
    flow,
    n_particles=200_000,
    dt=0.002,
    reseeder=reseeder,
)

result.save("tracks.h5")
```

The saved HDF5 file contains:

- `position`: particle positions with shape `(n_steps, n_particles, 3)`
- `reset`: reset flags with shape `(n_steps, n_particles)`
- `dt`: time step attribute

## Parallel Tracking

For larger runs:

```python
result = pt.track_parallel(
    "case.pvd",
    n_particles=2_000_000,
    dt=0.002,
    caps=["Inlet.vtp", "Outlet.vtp"],
    active_key="Velocity",
    n_workers=3,
)
```

Each worker reloads the field, so memory use scales with `n_workers`.
