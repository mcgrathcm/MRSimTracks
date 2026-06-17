import h5py
import numpy as np
import pytest

import particle_tracking as pt


def test_public_api_exports_expected_names():
    expected = {
        "load_flow",
        "seed_volume",
        "track",
        "track_parallel",
        "TrackingResult",
        "BoundaryReseeder",
        "extract_caps",
    }

    assert expected <= set(pt.__all__)
    for name in expected:
        assert hasattr(pt, name)


def test_load_flow_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="unsupported flow file type"):
        pt.load_flow("case.txt")


def test_track_requires_seeds_or_particle_count():
    with pytest.raises(ValueError, match="provide either seeds or n_particles"):
        pt.track(flow=None, pbar=False)


def test_tracking_result_save_writes_expected_hdf5_schema(tmp_path):
    result = pt.TrackingResult(
        positions=np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3),
        reset=np.zeros((4, 3), dtype=bool),
        dt=0.002,
    )

    path = tmp_path / "tracks.h5"
    result.save(path, time_subsample=2)

    with h5py.File(path, "r") as f:
        assert f["position"].shape == (2, 3, 3)
        assert f["reset"].shape == (2, 3)
        assert f.attrs["dt"] == pytest.approx(0.004)


def test_tracking_result_properties():
    result = pt.TrackingResult(
        positions=np.zeros((4, 3, 3)),
        reset=np.zeros((4, 3), dtype=bool),
        dt=0.002,
    )

    assert result.n_steps == 4
    assert result.n_particles == 3
    np.testing.assert_allclose(result.times, [0.0, 0.002, 0.004, 0.006])
