# Changelog

All notable changes to this project will be documented here.

This project uses semantic versioning while the public API stabilizes.

## [Unreleased]

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
