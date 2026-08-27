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
    time_subsample=10,
    return_metrics=True,
)

result.is_file_backed
metrics["particle_steps_per_s"]
```

With `time_subsample=N`, every Nth integration state is stored, `dt` records
the stored-state interval, and reset flags are accumulated over each interval.

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

## Ground-Truth Velocity Images

Sample a Cartesian image directly from the loaded CFD field:

```python
image = mt.sample_velocity_image(
    flow,
    fov=(12.6, 2.6, 16.4),       # full widths in native (x,y,z) mesh units
    resolution=0.2,              # isotropic voxel size in mesh units
    temporal_spacing=0.030,      # output time spacing in seconds
    temporal_width=0.030,        # exact boxcar average of the linear waveform
    grid_subsampling=2,          # 2 per axis = 8 samples per voxel
    reorder_by_extent=False,     # True for largest-to-smallest spatial axes
)

image.axis_order  # "x,y,z", matching track output
image.velocity    # (time, x, y, z, component)
image.occupancy   # (x, y, z)
image.save("velocity_image.h5")  # sparse spatial support by default
```

Use `fov=None` for the native mesh bounds. Three FOV widths expand symmetrically
about the mesh bounding-box center; three explicit `(minimum, maximum)` pairs
are also accepted. Neither form shifts the coordinate origin.
Set `temporal_width=0` to linearly interpolate the exact output time instead of
averaging a window.

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
