# Data Model

MRSimTracks uses CFD mesh data for particle tracking and for sampling
ground-truth Cartesian velocity images.

## Flow Mesh

Supported inputs:

- One `.vtu` with velocity arrays named by timestep, such as
  `Velocity_00190`.
- A `.pvd` collection with one `.vtu` per frame.
- A directory or explicit path list containing one `.vtu` per frame. A
  `TimeValue` field is used when present; otherwise the trailing number in each
  filename determines order. Pass `dt` to scale filename-derived indices.

All sources populate the same internal representation. Mesh topology, node
coordinates, and point fields are stored separately and indexed by frame:

- Static series retain one topology and one coordinate array.
- Moving-node series retain one topology and the distinct coordinate arrays.
- Topology-changing series retain only the distinct topologies and coordinates.

For separate files, encoded geometry payloads are checked at every frame. A
changed payload is decoded and compared semantically before a new topology or
coordinate array is retained. Exact field-only frames therefore avoid loading
or decompressing repeated connectivity.

Pass `mesh_mode` when the layout is known:

- `"auto"` (default) classifies every frame.
- `"static"` reuses the first topology and coordinates. Every point field must
  retain the first frame's length, and the node coordinates are checked exactly
  at frame `N // 2` to catch likely motion half a cycle away. Connectivity and
  cell types are also compared at that frame.
- `"moving"` (also `"moving-node"` or `"moving_node"`) reuses the first
  topology and loads node coordinates and the active point field for every
  frame. Both must retain the first frame's point count, and topology is compared
  at frame `N // 2`.
- `"changing_topology"` loads each frame's geometry.

The fast sampler is used for tetrahedral cells. Other cell types use PyVista's
fallback sampler unless `conform_mesh=True` can condition them to tetrahedra.
Velocity must be a three-component point-data field with a consistent name
(matching is case-insensitive).

`load_ale_flow` uses the same reference mesh for all frames and reconstructs
absolute deformed states as `reference coordinates + displacement`. With
`center_mesh=True`, the translation is computed from the initial absolute ALE
state and applied to the reference coordinates before any state samplers are
constructed. Velocity, displacement, and mesh-velocity fields are unchanged.

## Fixed-Topology Mesh Motion

`load_mesh_motion` handles material particles attached to a deforming mesh. It
accepts either a series whose VTU point coordinates move or a static-coordinate
series with a three-component `displacement_key`. Both inputs are normalized at
load time to absolute node positions for each frame. Pass `center_mesh=True` to
translate every absolute frame by the same vector, computed from the initial
frame's axis-aligned bounds, before seeding or trajectory generation.

The first frame's topology is tetrahedralized once. Existing nodes and their
motion are preserved exactly; hex and wedge interiors use the resulting
piecewise-linear tetrahedral interpolation rather than their native trilinear
shape functions. Each seeded particle stores one tetrahedron id, four node ids,
and four barycentric weights. Its position is therefore

`x(t) = sum(weight[i] * node_position[t, node_id[i]])`.

Seeding selects tetrahedra in proportion to reference volume and draws uniform
barycentric coordinates, so it returns exactly the requested particle count.
Weights and connectivity never change. `periodic=True` wraps interpolated query
times over the loaded duration; `periodic=False` restricts queries to the loaded
time interval. The coordinate-series loader compares midpoint topology to the
reference frame; mesh conditioning and periodic closure remain input-data
responsibilities.

`MeshMotion.trajectory` returns a `MaterialTrajectory`. Without `output_path`,
positions are held in memory. With an HDF5 output path, each time frame is
evaluated and written immediately, keeping working memory independent of the
number of output frames. The file contains `position` with shape
`(time, particle, xyz)` and an explicit `time` dataset, which preserves
nonuniform stored or requested times. `MaterialTrajectory.open(path)` reopens
the result lazily.

## Ground-Truth Velocity Images

`sample_velocity_image` uses the same native `(x,y,z)` coordinates as `track`.
The loaders preserve the source origin by default. With `center_mesh=True`,
`load_flow` applies one translation computed from the initial mesh frame to all
stored mesh frames before constructing the sampler; velocity components remain
`(vx,vy,vz)`. `reorder_by_extent=True` optionally applies a stable
largest-to-smallest permutation to both image axes and vector components,
without adding another coordinate shift.

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
