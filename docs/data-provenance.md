# Data Provenance

The repository includes CFD mesh fixtures so examples and tests are
reproducible.

- The full-cycle fixture is `example/CFD_velocity.vtu` and is tracked with Git
  LFS.
- The reduced normal-CI fixture is `tests/data/CFD_velocity_00190_00210.vtu`,
  derived from frames 190-210 of the full example.
- Cap surfaces are stored as `.vtp` files and used by `BoundaryReseeder`.

These files are CFD velocity-field data, not MR image data. See
[DATA_LICENSE.md](https://github.com/mcgrathcm/MRSimTracks/blob/main/DATA_LICENSE.md)
for licensing and citation notes.
