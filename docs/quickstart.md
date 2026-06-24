# Quick Start

```python
import numpy as np

import mrsimtracks as mt
from mrsimtracks.seeding import seed_mesh

flow = mt.load_flow("case.pvd", active_key="Velocity")
reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)
wall_slip = mt.WallSlip(flow, caps=["Inlet.vtp", "Outlet.vtp"])
seeds = seed_mesh(flow.active_mesh, 200_000, rng=np.random.default_rng(0))

result = mt.track(
    flow,
    seeds=seeds,
    dt=0.002,
    reseeder=reseeder,
    wall_slip=wall_slip,
)

result.save("tracks.h5")
```

The saved HDF5 file contains:

- `position`: particle positions with shape `(n_steps, n_particles, 3)`
- `reset`: reset flags with shape `(n_steps, n_particles)`
- `dt`: time step attribute

## Streaming Output

For large single-process runs, pass `output_path` so timesteps are written
directly to HDF5 instead of accumulated in memory:

```python
result, metrics = mt.track(
    flow,
    seeds=seeds,
    dt=0.002,
    reseeder=reseeder,
    output_path="tracks.h5",
    return_metrics=True,
)

result.is_file_backed
metrics["particle_steps_per_s"]
```

## Parallel Tracking

For larger runs:

```python
result = mt.track_parallel(
    "case.pvd",
    seeds=seeds,
    dt=0.002,
    caps=["Inlet.vtp", "Outlet.vtp"],
    active_key="Velocity",
    n_workers=3,
)
```

Each worker reloads the field, so memory use scales with `n_workers`.

## Wall Slip

Use `WallSlip` when interpolation near no-slip walls deposits particles into a
thin stuck layer. It removes only the into-wall velocity component for particles
inside a narrow wall band and leaves tangential motion unchanged:

```python
wall_slip = mt.WallSlip(flow, caps=["Inlet.vtp", "Outlet.vtp"], band_frac=0.02)
result = mt.track(
    flow,
    seeds=seeds,
    dt=0.002,
    reseeder=reseeder,
    wall_slip=wall_slip,
)
```

Pass the cap surfaces so inlet/outlet faces are not treated as walls. For
parallel tracking, set `wall_slip=True` and optionally `wall_slip_band=0.02`:

```python
result = mt.track_parallel(
    "case.pvd",
    seeds=seeds,
    dt=0.002,
    caps=["Inlet.vtp", "Outlet.vtp"],
    active_key="Velocity",
    n_workers=3,
    wall_slip=True,
)
```
