# Quick Start

```python
import numpy as np

import mrsimtracks as mt
from mrsimtracks.seeding import seed_mesh

flow = mt.load_flow("case.pvd", active_key="Velocity")
reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)
seeds = seed_mesh(flow.active_mesh, 200_000, rng=np.random.default_rng(0))

result = mt.track(
    flow,
    seeds=seeds,
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
