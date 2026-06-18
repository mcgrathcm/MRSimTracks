# Example Data Provenance and License

MRSimTracks includes CFD mesh fixtures for examples, normal CI, and release
validation.

## Included Files

- `example/CFD_velocity.vtu`: full-cycle time-resolved CFD velocity field,
  tracked with Git LFS.
- `example/Inlet.vtp`, `example/Outlet.vtp`, `example/Wall.vtp`: boundary
  surfaces for the full example case.
- `tests/data/CFD_velocity_00190_00210.vtu`: reduced fixture derived from
  frames 190-210 of the full example velocity field for normal CI.
- `tests/data/Inlet.vtp`, `tests/data/Outlet.vtp`: cap surfaces used with the
  reduced fixture.

## Intended Use

These files are included to make tests, examples, and validation reproducible.
They are CFD mesh/velocity data, not MR image data.

## License

The included example data are distributed for use with MRSimTracks under the
same MIT license as the repository unless a future release adds a more specific
data license.

If you use the data in publications or derived benchmarks, cite the MRSimTracks
repository and record the commit or release tag used.
