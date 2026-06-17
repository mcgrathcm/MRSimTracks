# Testing

MRSimTracks uses two validation tiers.

## Normal CI

Normal CI runs on pull requests and pushes to `main`. It uses the reduced
fixture in `tests/data/` and does not fetch Git LFS data:

```bash
uv run pytest -m "not large" --cov=particle_tracking --cov-report=term-missing
```

These tests check:

- public API behavior
- reduced fixture loading
- boundary reseeding construction
- particle movement
- reset fraction sanity
- coarse density stability
- HDF5 output schema

## Release Validation

Release validation runs on tags, releases, and manual dispatch. It checks out
Git LFS data and runs:

```bash
uv run pytest -m large
```

The full-data validation writes a JSON metrics artifact with:

- loading and tracking runtime
- particle-steps per second
- density statistics
- reset statistics
- average and percentile speed statistics
- fixed-subset particle trajectory summaries

All test random operations use explicit seeds.
