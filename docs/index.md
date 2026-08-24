# MRSimTracks

MRSimTracks generates CFD-derived particle trajectories and ground-truth
velocity images for MR flow simulation. It can also generate material-particle
trajectories directly from fixed-topology mesh deformation.

It operates on time-resolved mesh velocity fields. The core tracking workflow is:

1. Load a static, moving-node, or topology-changing mesh series with velocity fields.
2. Seed particles in the flow domain.
3. Advect particles with RK4 or Euler integration.
4. Recycle particles that leave the domain through user-provided inlet/outlet
   cap surfaces.
5. Save trajectories for downstream simulation.

## Highlights

- Supports single-file `.vtu` time series, `.pvd` collections, directories, and
  explicit VTU path lists.
- Uses a fast tetrahedral sampler with temporal-coherence cell walking.
- Provides flux-weighted, backflow-aware boundary reseeding.
- Can stream large tracking outputs directly to HDF5.
- Includes small normal-CI fixtures and full Git LFS release validation.
- Samples native-coordinate Cartesian velocity images with temporal and spatial averaging.
- Uniformly seeds fixed-topology deforming meshes and moves particles by fixed
  tetrahedral barycentric coordinates without velocity integration.

## Import Name

The package distribution and Python import package are both `mrsimtracks`:

```python
import mrsimtracks as mt
```
