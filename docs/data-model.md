# Data Model

MRSimTracks uses CFD mesh data for particle tracking and for sampling
ground-truth Cartesian velocity images.

## Flow Mesh

Supported inputs:

- `.vtu`: one static mesh with velocity arrays named by timestep, such as
  `Velocity_00190`.
- `.pvd`: a collection of static-geometry `.vtu` frames.

The current fast path assumes:

- tetrahedral cells
- static mesh geometry
- point-data velocity fields
- a consistent velocity array name prefix, such as `Velocity`

## Ground-Truth Velocity Images

`sample_velocity_image` uses the same unshifted native `(x,y,z)` coordinates as
`track`. Velocity components remain `(vx,vy,vz)` and no mesh-dependent axis
permutation is applied by default. `reorder_by_extent=True` optionally applies a
stable largest-to-smallest permutation to both image axes and vector components,
without shifting the coordinate origin.

The dense in-memory velocity array has shape `(time,x,y,z,component)`.
Spatial occupancy records the fraction of regular subvoxel samples inside the
CFD domain. Sparse HDF5 output stores only voxels with nonzero occupancy while
retaining the dense shape, FOV, resolution, times, selected axis permutation,
and zero origin shift as metadata.

## Cap Surfaces

Boundary reseeding uses user-provided cap surfaces from the CFD setup:

```python
import mrsimtracks as mt

reseeder = mt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)
```

A list of surfaces is interpreted as one cap per file. A single labeled surface
may also be used if it contains a `region_id` cell array.

The development-only cap extraction helper is intentionally not part of the
recommended workflow. Prefer cap surfaces exported directly from the CFD setup.
