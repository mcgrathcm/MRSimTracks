# Changelog

All notable changes to this project will be documented here.

This project uses semantic versioning while the public API stabilizes.

## [Unreleased]

## [0.3.0] - 2026-10-29

### Changed

- Tracking results now include the initial seeds at `t=0`, with false reset
  flags. For N integration steps, default output contains N+1 states;
  `TrackingResult.n_steps` counts stored states. Consumers of 0.2.x output must
  account for the additional initial frame and use `result.times` for timing.
- Recycle particles at the end-of-step time and retain RK4's final cell guess
  for the next step. An invalid intermediate RK4 query triggers a reset.
- Add `time_subsample` to streamed tracking output. Both streaming and
  `TrackingResult.save` preserve the initial state, accumulate reset flags over
  skipped steps, and record the stored-state interval in the HDF5 `dt` attribute.
- Preserve the final integration step when floating-point rounding makes an
  integral duration/time-step ratio slightly smaller than its integer value.

### Added

- Add `sample_velocity_image` for Cartesian ground-truth velocity images in the
  same native `(x,y,z)` coordinates as particle tracking, with exact piecewise-
  linear temporal averaging, regular subvoxel averaging, occupancy maps, and
  sparse spatial HDF5 storage. Optional extent-based reordering applies the same
  largest-to-smallest permutation to image axes and velocity components.
- Unify flow loading from single VTUs, PVD collections, directories, and explicit
  VTU lists. Retain static, moving-node, and topology-changing geometry without
  storing identical meshes for every frame; expose `mesh_mode` for known layouts.
- Add `load_mesh_motion` and material-particle trajectories on fixed-topology
  deforming meshes, with uniform reference-volume seeding, fixed barycentric
  weights, and streamed HDF5 positions with exact evaluation times.
- Add `load_ale_flow` and `ALEBoundaryReseeder` for physical-space tracking on
  displaced reference meshes. Reuse tetrahedral connectivity across deformed
  states and weight reseeding by current face area and relative inflow velocity.
- Add `periodic_mapping` with 1-based destination-to-source nearest-neighbor
  indices for KomaMRI periodic `FlowPath` motion.
- Add `center_mesh` to flow, ALE, and material-motion loading, using one
  frame-0-based translation; translate reseeding caps consistently.
- Add `reverse_velocity` to flow/ALE loading and parallel tracking. ALE mesh
  velocity and displacement are preserved.
- Support native hex, wedge, and mixed cells on static geometry with
  `conform_mesh=False`. Reuse one VTK locator for field sampling and boundary
  reseeding, preserving native shape functions without tetrahedral cell walking.
  `WallSlip` continues to require tetrahedral cells.
- Treat a single velocity frame, including an unsuffixed field, as steady flow.
  Tracking requires an explicit finite `tmax` for these inputs.
- Allow the development cap extractor to accept a loaded flow, preserve source
  cap coordinates for centered flows, and report empty threshold selections or
  undersized components clearly.

### Release tooling

- Fix the strict API documentation build for material-trajectory progress.
- Validate both bare version tags and `v`-prefixed tags.
- Limit source distributions to the package, documentation, and release files
  so untracked example scripts and local outputs cannot enter release artifacts.

## [0.2.1] - 2026-08-18

- Fix mesh seeding for tetrahedra with reversed vertex orientation by probing
  the volume mesh directly instead of relying on orientation-sensitive surface
  normals.

## [0.2.0] - 2026-06-24

- Add a `precision` option (`"f64"` default, `"f32"` single) to `load_flow` and
  `track_parallel` that runs the field sampling and advection in single
  precision for a speedup, carrying positions/output at the same precision.
- Add a `time_interp` option (`"linear"` default, `"cubic"`) to `load_flow` and
  `track_parallel`. Cubic uses a uniform Catmull-Rom spline over four frames to
  interpolate the velocity field between stored timesteps, removing the
  per-frame velocity kink and reducing temporal reconstruction error (~38% on
  the example cardiac waveform). Requires uniformly spaced frames.
- Add load-time mesh conditioning (`conform_mesh`, default `True`): split
  non-tetrahedral cells (e.g. boundary-layer wedges/prisms) into tets and drop
  degenerate (near-zero-volume) cells so the fast tet sampler can run on hybrid
  or imperfect meshes. A no-op for already-clean all-tet meshes; pass
  `conform_mesh=False` to load as-is. `_TetSampler` now also guards against
  degenerate cells defensively (no crash; reports the count).
- Add `WallSlip`, an optional near-wall no-penetration projection passed to
  `track`/`track_parallel` (`wall_slip=`). It strips the into-wall velocity
  component within a band of the wall (a fraction of the vessel hydraulic
  diameter, default 2%) so particles slide along walls instead of being
  deposited and trapped by the spurious near-wall divergence of the
  interpolated field. Open boundaries (caps) are excluded so flux still passes.

## [0.1.0rc1] - 2026-06-18

- Rename the import package to `mrsimtracks` and narrow the top-level public
  API.
- Split tracking, parallel execution, flow loading, sampling, seeding, and
  reseeding into focused modules.
- Add explicit seeding control to `track` and `track_parallel`.
- Add streamed HDF5 tracking output with file-backed `TrackingResult` loading.
- Add structured timing metrics via `return_metrics=True`.
- Add input validation for tracking methods, seed shapes, flow array names, and
  seeding arguments.
- Move cap extraction to `mrsimtracks.dev` as a development helper.
- Add MkDocs GitHub Pages documentation.
- Add full-cycle Git LFS release validation with JSON metrics artifacts.
- Add normal-commit coverage reporting and small behavioral regression tests.
- Add GitHub Actions CI for normal tests and release validation.
- Add Git LFS tracking for the full example CFD dataset.
- Add a reduced real-data fixture for fast CI.
- Add initial publication metadata and MIT license.

## [0.1.0rc0] - 2026-06-17

- Initial packaged version of MRSimTracks.
- Add high-level APIs for loading flow fields, tracking particles, parallel
  tracking, cap extraction, and boundary-aware reseeding.
- Add time-resolved static-mesh `.pvd` loading and fast tetrahedral sampling.
