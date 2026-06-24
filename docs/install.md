# Install

MRSimTracks is published on PyPI:

```bash
uv add "mrsimtracks==0.2.0"
```

or with pip:

```bash
python -m pip install "mrsimtracks==0.2.0"
```

To install the latest source from GitHub instead:

```bash
uv add "mrsimtracks @ git+https://github.com/mcgrathcm/MRSimTracks.git"
```

For development from a clone:

```bash
uv sync
```

For tests and development tooling:

```bash
uv sync --group dev
```

For documentation builds:

```bash
uv sync --group docs
```

The project depends on PyVista/VTK and expects Python 3.12 or newer.

## Git LFS Data

The full example CFD file is stored with Git LFS. Fetch it when running
full-data validation:

```bash
git lfs pull --include="example/CFD_velocity.vtu"
```

Normal tests use the reduced fixture in `tests/data/` and do not require LFS.
