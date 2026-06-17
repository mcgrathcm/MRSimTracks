# Contributing

## Setup

Use the repository root as the project environment:

```bash
uv sync --group dev
```

The full example dataset is stored with Git LFS. After cloning, fetch it with:

```bash
git lfs pull --include="example/CFD_velocity.vtu"
```

## Tests

Run the normal test suite before committing:

```bash
uv run pytest -m "not large"
```

The normal suite uses the reduced fixture in `tests/data/`, so it does not need
the full LFS dataset.

Run full-data validation before releases:

```bash
uv run pytest -m large
```

Those tests require `example/CFD_velocity.vtu` to be present through Git LFS.

## Package Checks

Before a release, verify the package builds:

```bash
uv build
```

The wheel and source distribution intentionally exclude `.vtu` and `.vtp` data
fixtures. Those files are repository test assets, not package data.

## Data Policy

- Keep generated outputs, exploratory data, and scratch files under `ignore/`.
- Keep small CI fixtures under `tests/data/`.
- Keep large canonical fixtures under `example/` and track them with Git LFS.
- Do not commit private simulation outputs unless they are intended to be public.
