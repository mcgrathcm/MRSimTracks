import numpy as np
import pytest

import mrsimtracks as mt


def test_periodic_mapping_is_one_based_destination_to_source():
    initial = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [9.0, 0.0, 0.0], [9.5, 0.0, 0.0]]
    )
    final = np.array(
        [[0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
    )

    mapping = mt.periodic_mapping(initial, final)

    np.testing.assert_array_equal(mapping, [1, 1, 2, 2])
    assert mapping.dtype == np.int64


def test_periodic_mapping_matches_brute_force():
    rng = np.random.default_rng(4)
    initial = rng.normal(size=(50, 3))
    final = rng.normal(size=(50, 3))
    distances_squared = np.sum(
        (initial[:, np.newaxis, :] - final[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    expected = np.argmin(distances_squared, axis=1) + 1

    np.testing.assert_array_equal(mt.periodic_mapping(initial, final), expected)


def test_periodic_mapping_accepts_float32_noncontiguous_inputs():
    initial_base = np.array(
        [[0, 0, 0], [99, 99, 99], [2, 0, 0], [99, 99, 99]], dtype=np.float32
    )
    final_base = np.array(
        [[0.1, 0, 0], [99, 99, 99], [2.1, 0, 0], [99, 99, 99]], dtype=np.float32
    )

    mapping = mt.periodic_mapping(initial_base[::2], final_base[::2])

    np.testing.assert_array_equal(mapping, [1, 2])


@pytest.mark.parametrize(
    ("initial", "final", "message"),
    [
        (np.zeros((3,)), np.zeros((1, 3)), "initial_positions must have shape"),
        (np.zeros((1, 3)), np.zeros((1, 2)), "final_positions must have shape"),
        (np.zeros((0, 3)), np.zeros((0, 3)), "at least one particle"),
        (np.zeros((2, 3)), np.zeros((1, 3)), "same number of particles"),
        (
            np.array([[np.nan, 0.0, 0.0]]),
            np.zeros((1, 3)),
            "initial_positions must contain only finite values",
        ),
        (
            np.zeros((1, 3)),
            np.array([[np.inf, 0.0, 0.0]]),
            "final_positions must contain only finite values",
        ),
    ],
)
def test_periodic_mapping_rejects_invalid_positions(initial, final, message):
    with pytest.raises(ValueError, match=message):
        mt.periodic_mapping(initial, final)
