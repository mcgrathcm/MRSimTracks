# Changelog

All notable changes to this project will be documented here.

This project uses semantic versioning while the public API stabilizes.

## [Unreleased]

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

## [0.1.0] - 2026-06-17

- Initial packaged version of MRSimTracks.
- Add high-level APIs for loading flow fields, tracking particles, parallel
  tracking, cap extraction, and boundary-aware reseeding.
- Add time-resolved static-mesh `.pvd` loading and fast tetrahedral sampling.
