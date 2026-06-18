# Full-Cycle Example Data

This folder contains the full-cycle CFD example used by release validation and
long-stability checks.

`CFD_velocity.vtu` is tracked with Git LFS because it is about 816 MB. Fetch it
after cloning when you need the full dataset:

```bash
git lfs pull --include="example/CFD_velocity.vtu"
```

The quick `example.py` script uses the reduced fixture under `tests/data/` so it
can run without Git LFS. To run against the full cycle, point `FLOW` and `CAPS`
in `example.py` at the files in this directory.

Expected files:

- `CFD_velocity.vtu`
- `Inlet.vtp`
- `Outlet.vtp`
