# Changelog

All notable changes to this project will be documented here.

This project uses semantic versioning while the public API stabilizes.

## [Unreleased]

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
