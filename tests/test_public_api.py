import h5py
import numpy as np
import pytest

import mrsimtracks as pt
from mrsimtracks.seeding import seed_region


def test_public_api_exports_expected_names():
    expected = {
        "load_flow",
        "track",
        "track_parallel",
        "TrackingResult",
        "BoundaryReseeder",
    }

    assert expected <= set(pt.__all__)
    for name in expected:
        assert hasattr(pt, name)


def test_load_flow_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="unsupported flow file type"):
        pt.load_flow("case.txt")


def test_track_requires_seeds_or_particle_count():
    with pytest.raises(ValueError, match="provide seeds"):
        pt.track(flow=None, seeds=None, pbar=False)


def test_track_rejects_unknown_method_before_running():
    with pytest.raises(ValueError, match="unsupported tracking method"):
        pt.track(flow=None, seeds=np.zeros((1, 3)), method="not-a-method", pbar=False)


def test_track_rejects_invalid_seed_shape():
    with pytest.raises(ValueError, match="seeds must have shape"):
        pt.track(flow=None, seeds=np.zeros((3,)), pbar=False)


def test_seed_region_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="npoints must be"):
        seed_region(mesh=None, npoints=0, bounds=(0, 1, 0, 1, 0, 1))

    with pytest.raises(ValueError, match="bounds must contain"):
        seed_region(mesh=None, npoints=1, bounds=(0, 1))

    with pytest.raises(ValueError, match="positive volume"):
        seed_region(mesh=None, npoints=1, bounds=(0, 0, 0, 1, 0, 1))


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


def test_tracking_result_open_is_file_backed(tmp_path):
    result = pt.TrackingResult(
        positions=np.arange(4 * 3 * 3, dtype=float).reshape(4, 3, 3),
        reset=np.zeros((4, 3), dtype=bool),
        dt=0.002,
    )
    path = tmp_path / "tracks.h5"
    result.save(path)

    opened = pt.TrackingResult.open(path)

    assert opened.is_file_backed
    assert opened.n_steps == 4
    assert opened.n_particles == 3
    assert opened.dt == pytest.approx(0.002)
    np.testing.assert_allclose(opened.positions, result.positions)
    assert not opened.is_file_backed


def test_tracking_result_properties():
    result = pt.TrackingResult(
        positions=np.zeros((4, 3, 3)),
        reset=np.zeros((4, 3), dtype=bool),
        dt=0.002,
    )

    assert result.n_steps == 4
    assert result.n_particles == 3
    np.testing.assert_allclose(result.times, [0.0, 0.002, 0.004, 0.006])
