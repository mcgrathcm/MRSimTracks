# Data Model

MRSimTracks expects CFD mesh data, not MR image data.

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

## Cap Surfaces

Boundary reseeding uses user-provided cap surfaces from the CFD setup:

```python
reseeder = pt.BoundaryReseeder(["Inlet.vtp", "Outlet.vtp"], flow, dt=0.002)
```

A list of surfaces is interpreted as one cap per file. A single labeled surface
may also be used if it contains a `region_id` cell array.

The development-only cap extraction helper is intentionally not part of the
recommended workflow. Prefer cap surfaces exported directly from the CFD setup.
