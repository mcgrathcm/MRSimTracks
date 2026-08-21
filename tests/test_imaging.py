import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt
from mrsimtracks.imaging import _temporal_weights
from mrsimtracks.sampler import _TetSampler


class AnalyticFlow:
    def __init__(self, spans=(1.0, 1.0, 1.0)):
        corners = np.array(
            [
                [x, y, z]
                for x in (0.0, spans[0])
                for y in (0.0, spans[1])
                for z in (0.0, spans[2])
            ]
        )
        self.active_mesh = pv.PolyData(corners).delaunay_3d().cast_to_unstructured_grid()
        self.active_key = "Velocity"
        self.dtype = np.dtype(np.float64)
        self.times_shift_s = np.array([0.0, 1.0])
        self.tmax = 1.0
        centered = np.asarray(self.active_mesh.points) - np.asarray(spans) / 2
        self.fields = [centered, centered.copy()]
        self._sampler = _TetSampler(self.active_mesh)

    def _frame_vel(self, index):
        return self.fields[index]


def test_zero_width_uses_linear_interpolation_at_exact_time():
    weights = _temporal_weights(np.array([0.0, 1.0, 2.0]), center=0.25, width=0)
    np.testing.assert_allclose(weights, [0.75, 0.25, 0.0])


def test_finite_width_exactly_integrates_interpolant_and_uses_outside_frames():
    # Window [0.5, 1.9]: frames at t=0 and t=2 sit outside the window but both
    # contribute through interpolation of its two boundaries.
    weights = _temporal_weights(
        np.array([0.0, 1.0, 2.0, 3.0]), center=1.2, width=1.4
    )
    np.testing.assert_allclose(weights, np.array([0.125, 0.870, 0.405, 0.0]) / 1.4)
    assert weights[0] > 0 and weights[2] > 0
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_temporal_window_wraps_periodically():
    weights = _temporal_weights(
        np.array([0.0, 1.0, 2.0, 3.0]), center=0.0, width=1.0
    )
    np.testing.assert_allclose(weights, [0.375, 0.125, 0.125, 0.375])


def test_public_sampler_supports_off_grid_timing_and_window_edges():
    flow = AnalyticFlow()
    base = flow.fields[0]
    flow.times_shift_s = np.array([0.0, 0.7, 2.0])
    flow.tmax = 2.0
    flow.fields = [0.0 * base, 1.0 * base, 0.0 * base]
    fov = ((0.5, 1.0), (0.25, 0.75), (0.25, 0.75))

    point = mt.sample_velocity_image(
        flow,
        fov=fov,
        resolution=1.0,
        temporal_spacing=2.0,
        temporal_width=0.0,
        start_time=0.35,
    )
    window = mt.sample_velocity_image(
        flow,
        fov=fov,
        resolution=1.0,
        temporal_spacing=2.0,
        temporal_width=0.5,
        start_time=0.35,
    )

    np.testing.assert_allclose(point.times, [0.35])
    np.testing.assert_allclose(window.times, [0.35])
    point_weights = _temporal_weights(flow.times_shift_s, 0.35, 0.0)
    window_weights = _temporal_weights(flow.times_shift_s, 0.35, 0.5)
    np.testing.assert_allclose(point_weights, [0.5, 0.5, 0.0])
    assert window_weights[0] > 0 and window_weights[1] > 0
    np.testing.assert_allclose(point.velocity[0, 0, 0, 0], [0.125, 0.0, 0.0])
    np.testing.assert_allclose(
        window.velocity[0, 0, 0, 0], [0.25 * window_weights[1], 0.0, 0.0]
    )
    np.testing.assert_allclose(point.occupancy, 1.0)
    np.testing.assert_allclose(window.occupancy, 1.0)


def test_grid_matches_native_tracking_origin_and_axis_order():
    flow = AnalyticFlow(spans=(1.0, 3.0, 2.0))
    result = mt.sample_velocity_image(
        flow,
        fov=None,
        resolution=1.0,
        temporal_spacing=1.0,
        temporal_width=0.0,
    )

    assert result.axis_order == "x,y,z"
    assert result.velocity.shape == (1, 1, 3, 2, 3)
    np.testing.assert_allclose(result.fov, [[0.0, 1.0], [0.0, 3.0], [0.0, 2.0]])
    np.testing.assert_allclose(result.resolution, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.occupancy, 1.0)
    axes = [np.array([0.0]), np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5])]
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    np.testing.assert_allclose(result.velocity[0], expected, atol=1e-12)


def test_extent_reordering_is_explicit_and_does_not_shift_origin():
    result = mt.sample_velocity_image(
        AnalyticFlow(spans=(1.0, 3.0, 2.0)),
        fov=None,
        resolution=1.0,
        temporal_spacing=1.0,
        temporal_width=0.0,
        reorder_by_extent=True,
    )

    assert result.axis_order == "y,z,x"
    np.testing.assert_array_equal(result.axis_permutation, [1, 2, 0])
    assert result.velocity.shape == (1, 3, 2, 1, 3)
    np.testing.assert_allclose(result.fov, [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0]])
    axes = [np.array([-1.0, 0.0, 1.0]), np.array([-0.5, 0.5]), np.array([0.0])]
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    np.testing.assert_allclose(result.velocity[0], expected, atol=1e-12)


def test_factor_two_averages_eight_regular_subvoxel_samples():
    result = mt.sample_velocity_image(
        AnalyticFlow(),
        fov=(1.0, 1.0, 1.0),
        resolution=0.5,
        temporal_spacing=1.0,
        temporal_width=0.0,
        grid_subsampling=2,
    )

    axes = [np.array([-0.25, 0.25])] * 3
    expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    np.testing.assert_allclose(result.velocity[0], expected, atol=1e-12)
    np.testing.assert_allclose(result.occupancy, 1.0)


def test_exact_resolution_multiple_is_not_rounded_to_an_extra_voxel():
    widths = np.array([1.0, 0.6, 0.2])
    center = np.array([0.5, 0.5, 0.5])
    fov = np.column_stack((center - widths / 2, center + widths / 2))
    result = mt.sample_velocity_image(
        AnalyticFlow(),
        fov=fov,
        resolution=0.2,
        temporal_spacing=1.0,
        temporal_width=0.0,
    )

    assert result.velocity.shape == (1, 5, 3, 1, 3)
    np.testing.assert_allclose(result.resolution, 0.2, atol=1e-15)


def test_partial_voxels_average_valid_samples_and_report_occupancy():
    result = mt.sample_velocity_image(
        AnalyticFlow(),
        fov=(2.0, 1.0, 1.0),
        resolution=1.0,
        temporal_spacing=1.0,
        temporal_width=0.0,
        grid_subsampling=2,
    )

    np.testing.assert_allclose(result.occupancy, 0.5)
    np.testing.assert_allclose(result.velocity[0, 0, ..., 0], -0.25, atol=1e-12)
    np.testing.assert_allclose(result.velocity[0, 1, ..., 0], 0.25, atol=1e-12)


def test_sparse_save_load_round_trip(tmp_path):
    flow = AnalyticFlow()
    result = mt.sample_velocity_image(
        flow,
        fov=(2.0, 1.0, 1.0),
        resolution=0.5,
        temporal_spacing=0.5,
        temporal_width=0.0,
    )
    path = tmp_path / "velocity_image.h5"
    result.save(path)
    loaded = mt.VelocityImage.load(path)

    np.testing.assert_allclose(loaded.velocity, result.velocity)
    np.testing.assert_allclose(loaded.occupancy, result.occupancy)
    np.testing.assert_allclose(loaded.times, result.times)
    np.testing.assert_allclose(loaded.fov, result.fov)
    np.testing.assert_allclose(loaded.resolution, result.resolution)
    np.testing.assert_array_equal(loaded.axis_permutation, result.axis_permutation)
    assert loaded.axis_order == result.axis_order
    assert loaded.grid_subsampling == result.grid_subsampling


def test_imaging_argument_validation():
    flow = AnalyticFlow()
    for kwargs, match in [
        ({"fov": (1.0, -1.0, 1.0)}, "FOV widths"),
        ({"resolution": 0.0}, "resolution values"),
        ({"temporal_spacing": 0.0}, "temporal_spacing"),
        ({"temporal_width": -1.0}, "temporal_width"),
        ({"grid_subsampling": 0}, "grid_subsampling"),
    ]:
        args = dict(
            fov=(1.0, 1.0, 1.0),
            resolution=0.5,
            temporal_spacing=1.0,
            temporal_width=0.0,
        )
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            mt.sample_velocity_image(flow, **args)
