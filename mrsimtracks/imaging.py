"""Sample time-resolved CFD velocity fields onto Cartesian images."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

_VOXEL_CHUNK = 65_536
SOURCE_AXES = np.array(["x", "y", "z"])


@dataclass
class VelocityImage:
    """Dense velocity image sampled from a CFD flow.

    The coordinate origin is never shifted. Spatial axes and vector components
    are native ``(x,y,z)`` by default or carry the same optional extent-based
    permutation recorded in ``axis_permutation``. ``occupancy`` is the fraction
    of spatial sub-samples inside the CFD domain.
    """

    velocity: np.ndarray
    occupancy: np.ndarray
    times: np.ndarray
    fov: np.ndarray
    requested_resolution: np.ndarray
    resolution: np.ndarray
    grid_subsampling: int
    temporal_spacing: float
    temporal_width: float
    axis_permutation: np.ndarray

    @property
    def axis_order(self):
        return ",".join(SOURCE_AXES[self.axis_permutation])

    def save(self, path, sparse=True):
        """Save to HDF5, sparsifying the fixed spatial support by default."""
        path = Path(path)
        with h5py.File(path, "w") as f:
            f.attrs["storage"] = "spatial-coo" if sparse else "dense"
            f.attrs["axis_order"] = self.axis_order
            f.attrs["axis_permutation"] = self.axis_permutation
            f.attrs["origin_shift"] = np.zeros(3)
            f.attrs["dense_shape"] = self.velocity.shape
            f.attrs["grid_subsampling"] = self.grid_subsampling
            f.attrs["temporal_spacing"] = self.temporal_spacing
            f.attrs["temporal_width"] = self.temporal_width
            f.create_dataset("times", data=self.times)
            f.create_dataset("fov", data=self.fov)
            f.create_dataset("requested_resolution", data=self.requested_resolution)
            f.create_dataset("resolution", data=self.resolution)

            if not sparse:
                f.create_dataset(
                    "velocity", data=self.velocity, compression="gzip", shuffle=True
                )
                f.create_dataset(
                    "occupancy", data=self.occupancy, compression="gzip", shuffle=True
                )
                return

            indices = np.argwhere(self.occupancy > 0).astype(np.int32)
            f.create_dataset("voxel_indices", data=indices, compression="gzip")
            n_active = len(indices)
            if n_active == 0:
                f.create_dataset(
                    "velocity", shape=(len(self.times), 0, 3), dtype=self.velocity.dtype
                )
                f.create_dataset("occupancy", shape=(0,), dtype=self.occupancy.dtype)
                return

            chunks = (1, min(n_active, _VOXEL_CHUNK), 3)
            velocity = f.create_dataset(
                "velocity",
                shape=(len(self.times), n_active, 3),
                dtype=self.velocity.dtype,
                chunks=chunks,
                compression="gzip",
                shuffle=True,
            )
            spatial_index = tuple(indices.T)
            for i in range(len(self.times)):
                velocity[i] = self.velocity[i][spatial_index]
            f.create_dataset(
                "occupancy",
                data=self.occupancy[spatial_index],
                compression="gzip",
                shuffle=True,
            )

    @classmethod
    def load(cls, path):
        """Load an HDF5 image saved by :meth:`save` into dense memory."""
        with h5py.File(path, "r") as f:
            storage = f.attrs["storage"]
            if isinstance(storage, bytes):
                storage = storage.decode()
            dense_shape = tuple(int(v) for v in f.attrs["dense_shape"])
            if storage == "dense":
                velocity = f["velocity"][...]
                occupancy = f["occupancy"][...]
            elif storage == "spatial-coo":
                velocity = np.zeros(dense_shape, dtype=f["velocity"].dtype)
                occupancy = np.zeros(dense_shape[1:4], dtype=f["occupancy"].dtype)
                indices = f["voxel_indices"][...]
                if len(indices):
                    spatial_index = tuple(indices.T)
                    velocity[(slice(None), *spatial_index, slice(None))] = f[
                        "velocity"
                    ][...]
                    occupancy[spatial_index] = f["occupancy"][...]
            else:
                raise ValueError(f"unsupported velocity-image storage {storage!r}")

            return cls(
                velocity=velocity,
                occupancy=occupancy,
                times=f["times"][...],
                fov=f["fov"][...],
                requested_resolution=f["requested_resolution"][...],
                resolution=f["resolution"][...],
                grid_subsampling=int(f.attrs["grid_subsampling"]),
                temporal_spacing=float(f.attrs["temporal_spacing"]),
                temporal_width=float(f.attrs["temporal_width"]),
                axis_permutation=np.asarray(f.attrs["axis_permutation"], dtype=int),
            )


def _linear_coefficients(times, time):
    """Coefficients of the piecewise-linear interpolant at one in-cycle time."""
    weights = np.zeros(len(times), dtype=float)
    if time <= times[0]:
        weights[0] = 1.0
        return weights
    if time >= times[-1]:
        weights[-1] = 1.0
        return weights

    right = int(np.searchsorted(times, time, side="left"))
    if np.isclose(time, times[right], rtol=0.0, atol=1e-12):
        weights[right] = 1.0
        return weights
    left = right - 1
    fraction = (time - times[left]) / (times[right] - times[left])
    weights[left] = 1.0 - fraction
    weights[right] = fraction
    return weights


def _integrate_linear_segment(times, start, stop):
    """Exact integral weights over ``start:stop`` within one flow period."""
    inside = times[(times > start) & (times < stop)]
    knots = np.concatenate(([start], inside, [stop]))
    weights = np.zeros(len(times), dtype=float)
    left_coeff = _linear_coefficients(times, knots[0])
    for left, right in zip(knots[:-1], knots[1:]):
        right_coeff = _linear_coefficients(times, right)
        weights += 0.5 * (right - left) * (left_coeff + right_coeff)
        left_coeff = right_coeff
    return weights


def _temporal_weights(times, center, width):
    """Exact boxcar-average weights for a periodic piecewise-linear waveform."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("flow times must be a strictly increasing one-dimensional array")
    if not np.isclose(times[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("flow times must start at zero")
    period = float(times[-1])
    if period <= 0:
        raise ValueError("flow period must be positive")
    if width < 0 or not np.isfinite(width):
        raise ValueError("temporal_width must be finite and >= 0")

    if width == 0:
        return _linear_coefficients(times, float(center) % period)

    start = float(center) - width / 2
    stop = float(center) + width / 2
    weights = np.zeros(len(times), dtype=float)
    cursor = start
    while cursor < stop:
        cycle = np.floor(cursor / period)
        cycle_stop = (cycle + 1) * period
        segment_stop = min(stop, cycle_stop)
        local_start = cursor - cycle * period
        local_stop = segment_stop - cycle * period
        if np.isclose(local_stop, 0.0, rtol=0.0, atol=1e-12):
            local_stop = period
        weights += _integrate_linear_segment(times, local_start, local_stop)
        if segment_stop <= cursor:
            raise RuntimeError("failed to advance through temporal averaging window")
        cursor = segment_stop

    total = weights.sum()
    if total <= 0:
        raise RuntimeError("temporal averaging produced zero total weight")
    return weights / total


def _normalize_fov(flow, fov, permutation):
    mesh_bounds = np.asarray(flow.active_mesh.bounds, dtype=float).reshape(3, 2)
    output_bounds = mesh_bounds[permutation]

    if fov is None:
        return output_bounds

    fov = np.asarray(fov, dtype=float)
    if fov.shape == (3,):
        if np.any(fov <= 0):
            raise ValueError("FOV widths must be positive")
        center = output_bounds.mean(axis=1)
        fov = np.column_stack((center - fov / 2, center + fov / 2))
    elif fov.shape != (3, 2):
        raise ValueError("fov must be three widths or three (minimum, maximum) pairs")
    if not np.isfinite(fov).all() or np.any(fov[:, 1] <= fov[:, 0]):
        raise ValueError("fov bounds must be finite and define positive extents")
    return fov


def _normalize_resolution(resolution):
    resolution = np.asarray(resolution, dtype=float)
    if resolution.ndim == 0:
        resolution = np.repeat(resolution, 3)
    if resolution.shape != (3,) or not np.isfinite(resolution).all():
        raise ValueError("resolution must be one value or three finite values")
    if np.any(resolution <= 0):
        raise ValueError("resolution values must be positive")
    return resolution


def _subvoxel_offsets(resolution, factor):
    local = (np.arange(factor) + 0.5) / factor - 0.5
    offsets = np.stack(np.meshgrid(local, local, local, indexing="ij"), axis=-1)
    return offsets.reshape(-1, 3) * resolution


def _source_points(base, offsets, permutation):
    output = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    source = np.empty_like(output)
    source[:, permutation] = output
    return source


def _sample_field(flow, points, field, guess=None):
    sampler = flow._sampler
    if sampler.ok:
        return sampler.sample(points, field, guess=guess)

    flow.active_mesh.point_data[flow.active_key] = field
    sampled = pv.PolyData(points).sample(
        flow.active_mesh,
        locator=flow.locator,
        pass_cell_data=False,
        pass_point_data=False,
        pass_field_data=False,
    )
    valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
    velocity = np.asarray(sampled[flow.active_key]).copy()
    velocity[~valid] = 0
    return velocity, valid, None


def _weighted_field(flow, weights):
    first = np.asarray(flow._frame_vel(0))
    if first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("active CFD field must have three velocity components")
    field = np.zeros_like(first)
    for index in np.flatnonzero(weights):
        field += weights[index] * np.asarray(flow._frame_vel(index))
    return np.ascontiguousarray(field)


def sample_velocity_image(
    flow,
    *,
    fov=None,
    resolution,
    temporal_spacing,
    temporal_width,
    grid_subsampling=1,
    start_time=0.0,
    reorder_by_extent=False,
):
    """Sample a CFD flow onto a time-resolved Cartesian velocity image.

    Args:
        flow: Flow returned by :func:`mrsimtracks.load_flow`.
        fov: FOV in the selected output axis order: native ``(x,y,z)`` by
            default, or extent order when ``reorder_by_extent=True``. Pass three
            widths centered on the mesh bounding-box center, three explicit
            ``(minimum, maximum)`` pairs, or ``None`` for the mesh bounds.
        resolution: Requested voxel size in mesh spatial units, as one isotropic
            value or three values in the selected output axis order. The FOV is
            tiled exactly with voxel sizes no larger than requested.
        temporal_spacing: Spacing between output time points in seconds.
        temporal_width: Boxcar averaging width in seconds. Zero performs linear
            interpolation at each exact output time.
        grid_subsampling: Number of regular sub-samples per voxel axis. A value
            of two averages eight samples per nominal voxel.
        start_time: First output time in seconds. Output continues periodically
            at ``temporal_spacing`` until the end of the CFD period.
        reorder_by_extent: When ``True``, reorder spatial axes and velocity
            components by descending mesh extent. The default ``False`` preserves
            native tracking order ``(x,y,z)``. Neither mode shifts the origin.

    Returns:
        VelocityImage: Dense ``(time, x, y, z, component)`` velocity and spatial
            occupancy arrays in the same coordinate convention as ``track``.
    """
    requested_resolution = _normalize_resolution(resolution)
    mesh_bounds = np.asarray(flow.active_mesh.bounds, dtype=float).reshape(3, 2)
    spans = np.diff(mesh_bounds, axis=1).ravel()
    permutation = (
        np.argsort(-spans, kind="stable")
        if reorder_by_extent
        else np.arange(3, dtype=np.int64)
    )
    fov = _normalize_fov(flow, fov, permutation)
    if not isinstance(grid_subsampling, (int, np.integer)) or grid_subsampling < 1:
        raise ValueError("grid_subsampling must be a positive integer")
    if temporal_spacing <= 0 or not np.isfinite(temporal_spacing):
        raise ValueError("temporal_spacing must be finite and > 0")

    frame_times = np.asarray(flow.times_shift_s, dtype=float)
    period = float(frame_times[-1])
    if start_time < 0 or start_time >= period or not np.isfinite(start_time):
        raise ValueError("start_time must be finite and within one flow period")
    output_times = np.arange(start_time, period, temporal_spacing, dtype=float)
    if len(output_times) == 0:
        raise ValueError("temporal settings produced no output time points")
    temporal_weights = [
        _temporal_weights(frame_times, time, temporal_width) for time in output_times
    ]

    widths = np.diff(fov, axis=1).ravel()
    ratios = np.nextafter(widths / requested_resolution, -np.inf)
    shape = np.maximum(1, np.ceil(ratios).astype(int))
    actual_resolution = widths / shape
    n_voxels = int(np.prod(shape))
    offsets = _subvoxel_offsets(actual_resolution, grid_subsampling)
    n_subsamples = len(offsets)

    dtype = np.dtype(getattr(flow, "dtype", np.float64))
    velocity = np.zeros((len(output_times), *shape, 3), dtype=dtype)
    occupancy = np.zeros(tuple(shape), dtype=dtype)
    flat_occupancy = occupancy.reshape(-1)
    cell_cache = [None] * ((n_voxels + _VOXEL_CHUNK - 1) // _VOXEL_CHUNK)

    for time_index, weights in enumerate(temporal_weights):
        field = _weighted_field(flow, weights)
        flat_velocity = velocity[time_index].reshape(-1, 3)
        for chunk_index, start in enumerate(range(0, n_voxels, _VOXEL_CHUNK)):
            stop = min(start + _VOXEL_CHUNK, n_voxels)
            ijk = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
            base = fov[:, 0] + (ijk + 0.5) * actual_resolution
            points = _source_points(base, offsets, permutation)
            values, valid, cells = _sample_field(
                flow, points, field, guess=cell_cache[chunk_index]
            )
            if time_index == 0:
                cell_cache[chunk_index] = cells

            values = values[:, permutation].reshape(-1, n_subsamples, 3)
            valid = valid.reshape(-1, n_subsamples)
            counts = valid.sum(axis=1)
            sums = np.where(valid[..., None], values, 0).sum(axis=1)
            np.divide(
                sums,
                counts[:, None],
                out=flat_velocity[start:stop],
                where=counts[:, None] > 0,
            )
            if time_index == 0:
                flat_occupancy[start:stop] = counts / n_subsamples

    return VelocityImage(
        velocity=velocity,
        occupancy=occupancy,
        times=output_times,
        fov=fov,
        requested_resolution=requested_resolution,
        resolution=actual_resolution,
        grid_subsampling=int(grid_subsampling),
        temporal_spacing=float(temporal_spacing),
        temporal_width=float(temporal_width),
        axis_permutation=permutation,
    )
