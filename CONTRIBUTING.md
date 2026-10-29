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
uv run pytest -m "not large" --cov=mrsimtracks --cov-report=term-missing
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
The source distribution includes only the package, documentation, and release
files listed in `pyproject.toml`; local example scripts and scratch outputs are
excluded.

## Docs

Build the documentation locally with:

```bash
uv sync --group docs
uv run --group docs mkdocs build --strict
```

## Releases

Prepare releases on a branch from the latest `main`:

1. Update the version in `pyproject.toml`, run `uv lock`, and update the version
   pins in `README.md` and `docs/install.md`.
2. Move the unreleased notes into a dated version section in `CHANGELOG.md`.
   Document behavior changes and migration steps in the user guides.
3. Run the normal tests, full-data tests with Git LFS available, and the strict
   documentation build above. Confirm the full-data tests pass without skips.
4. Build from a clean checkout with `uv build`, inspect the wheel and source
   distribution for local files or data, and validate their metadata:

   ```bash
   uv run --with twine python -m twine check dist/*
   ```

   Smoke-test the wheel from outside the checkout so the source package cannot
   shadow the installed distribution:

   ```bash
   uv run --isolated --no-project --directory /tmp --with "$PWD"/dist/mrsimtracks-*.whl python -c "import mrsimtracks as mt; assert 'track' in mt.__all__"
   ```

5. After the release changes are merged, run the **Release Validation** workflow
   on that commit before publishing. Use a bare version tag such as `0.3.0`,
   matching existing releases. Tag pushes also trigger validation; `v`-prefixed
   tags remain supported, and publishing a GitHub release reruns validation.
6. Once validation succeeds, manually run the **Publish** workflow for the
   release tag to upload to PyPI. Tagging or creating a GitHub release does not
   publish the package to PyPI automatically. Confirm the published version
   before announcing the release.

## Data Policy

- Keep generated outputs, exploratory data, and scratch files under `ignore/`.
- Keep small CI fixtures under `tests/data/`.
- Keep large canonical fixtures under `example/` and track them with Git LFS.
- Do not commit private simulation outputs unless they are intended to be public.
