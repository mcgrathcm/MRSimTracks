import os

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from .io import (
    MeshFieldSeries,
    MeshTopology,
    _UnsupportedFastPath,
    _geometry_signatures,
    _interp_weights,
    _read_array,
    _read_vtu,
    _read_vtu_metadata,
    _resolve_point_array,
    _series_source,
)
from .sampler import _TetSampler, _condition_mesh, resolve_float_dtype


@dataclass
class _ALEFrameRuntime:
    mesh: pv.UnstructuredGrid
    locator: vtkStaticCellLocator
    sampler: _TetSampler
    velocity: np.ndarray
    mesh_velocity: np.ndarray


class ALEFlow:
    """Velocity and displacement fields on one fixed reference mesh.

    Sampling occurs in physical coordinates. At each requested time, the
    reference nodes are displaced, a locator/interpolator is built for that
    deformed mesh, and physical velocity is sampled there. The three most recent
    time states are cached, matching the ``t``, ``t + dt/2``, and ``t + dt``
    states used by RK4.
    """

    def __init__(
        self,
        data,
        velocity_key,
        displacement_key,
        mesh_velocity_key,
        times_shift_s,
        velocity_scale,
    ):
        if data.geometry_mode != "static":
            raise ValueError("ALEFlow requires one static reference mesh")
        self.data = data
        self.velocity_key = velocity_key
        self.displacement_key = displacement_key
        self.mesh_velocity_key = mesh_velocity_key
        self.velocity_scale = float(velocity_scale)
        self.active_key = velocity_key
        self.dtype = self.velocity(0).dtype
        self.time_interp = "linear"
        self.times = np.asarray(data.times)
        self.times_shift_s = np.asarray(times_shift_s, dtype=float)
        if np.any(np.diff(self.times_shift_s) <= 0):
            raise ValueError("ALE flow timesteps must be strictly increasing")
        self.tmax = float(self.times_shift_s[-1])
        if self.tmax <= 0:
            raise ValueError("ALE flow duration must be positive")

        self.reference_mesh = data.mesh(0)
        self.geometry_mode = "ale"
        self.fields = list(data.point_fields[velocity_key])
        self._runtime_cache = OrderedDict()
        self._runtime_build_count = 0

        self.active_mesh = self.get_mesh(0.0)
        self.mesh = self.active_mesh
        self._sampler = None
        self.locator = None

        reference = np.asarray(self.reference_mesh.points)
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for displacement in data.point_fields[displacement_key]:
            points = reference + displacement
            lower = np.minimum(lower, points.min(axis=0))
            upper = np.maximum(upper, points.max(axis=0))
        self.bounds = tuple(np.column_stack((lower, upper)).ravel())

    @property
    def n_frames(self):
        return len(self.times)

    def velocity(self, frame):
        return self.data.field(self.velocity_key, frame)

    def _frame_vel(self, frame):
        return self.velocity(frame)

    def displacement(self, frame):
        return self.data.field(self.displacement_key, frame)

    def mesh_velocity(self, frame):
        return self.data.field(self.mesh_velocity_key, frame)

    def relative_velocity(self, frame):
        return self.velocity(frame) - self.mesh_velocity(frame)

    def frame(self, frame):
        """Return the static reference mesh with one frame's ALE fields."""
        mesh = self.data.mesh(frame)
        mesh.point_data[self.velocity_key] = self.velocity(frame)
        mesh.point_data[self.displacement_key] = self.displacement(frame)
        mesh.point_data[self.mesh_velocity_key] = self.mesh_velocity(frame)
        return mesh

    def _time_state(self, time):
        indices, weights = _interp_weights(
            self.times_shift_s,
            self.tmax,
            len(self.times),
            time,
            self.time_interp,
        )
        key = (
            tuple(int(index) for index in indices),
            tuple(round(float(weight), 14) for weight in weights),
        )
        return key, indices, weights

    @staticmethod
    def _interpolate(get_frame, indices, weights):
        output = weights[0] * get_frame(indices[0])
        for index, weight in zip(indices[1:], weights[1:], strict=True):
            output = output + weight * get_frame(index)
        return np.ascontiguousarray(output)

    def _mesh_at_state(self, indices, weights):
        displacement = self._interpolate(self.displacement, indices, weights)
        velocity = self._interpolate(self.velocity, indices, weights)
        mesh_velocity = self._interpolate(
            self.mesh_velocity, indices, weights
        )
        points = np.asarray(self.reference_mesh.points) + displacement
        topology = self.data.topologies[0]
        mesh = pv.UnstructuredGrid(
            topology.cells,
            topology.cell_types,
            np.ascontiguousarray(points),
            deep=False,
        )
        mesh.point_data[self.velocity_key] = velocity
        mesh.point_data[self.displacement_key] = displacement
        mesh.point_data[self.mesh_velocity_key] = mesh_velocity
        return mesh, velocity, mesh_velocity

    def _runtime(self, time):
        key, indices, weights = self._time_state(time)
        runtime = self._runtime_cache.pop(key, None)
        if runtime is not None:
            self._runtime_cache[key] = runtime
            return runtime

        mesh, velocity, mesh_velocity = self._mesh_at_state(indices, weights)
        locator = vtkStaticCellLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        runtime = _ALEFrameRuntime(
            mesh,
            locator,
            _TetSampler(mesh, dtype=self.dtype),
            velocity,
            mesh_velocity,
        )
        self._runtime_cache[key] = runtime
        self._runtime_build_count += 1
        if len(self._runtime_cache) > 3:
            self._runtime_cache.popitem(last=False)
        return runtime

    def get_mesh(self, time):
        """Return the deformed mesh and interpolated fields at ``time``."""
        _, indices, weights = self._time_state(time)
        mesh, _, _ = self._mesh_at_state(indices, weights)
        return mesh

    def set_active_time(self, time):
        runtime = self._runtime(time)
        self.active_mesh = runtime.mesh
        self.mesh = runtime.mesh
        self._sampler = runtime.sampler
        self.locator = runtime.locator

    def _sample_runtime_field(self, points_xyz, runtime, field, guess):
        points_xyz = np.ascontiguousarray(points_xyz, dtype=self.dtype)
        if runtime.sampler.ok:
            return runtime.sampler.sample(points_xyz, field, guess=guess)

        name = "_sample_field"
        runtime.mesh.point_data[name] = field
        sampled = pv.PolyData(points_xyz).sample(
            runtime.mesh,
            locator=runtime.locator,
            pass_cell_data=False,
            pass_point_data=False,
            pass_field_data=False,
        )
        valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
        velocity = np.asarray(sampled[name]).copy()
        velocity[~valid] = 0
        return velocity, valid, None

    def sample_v(self, points_xyz, time, guess=None):
        """Sample physical velocity on the deformed mesh at ``time``."""
        runtime = self._runtime(time)
        self.active_mesh = runtime.mesh
        self.mesh = runtime.mesh
        self._sampler = runtime.sampler
        self.locator = runtime.locator
        return self._sample_runtime_field(
            points_xyz, runtime, runtime.velocity, guess
        )

    def sample_relative_v(self, points_xyz, time, guess=None):
        """Sample ``Velocity - Mesh_velocity`` on the deformed mesh."""
        runtime = self._runtime(time)
        relative = runtime.velocity - runtime.mesh_velocity
        return self._sample_runtime_field(points_xyz, runtime, relative, guess)

    def sample(self, points, time):
        velocity, valid, _ = self.sample_v(np.asarray(points.points), time)
        output = points.copy()
        output.point_data[self.velocity_key] = velocity
        output.point_data["vtkValidPointMask"] = valid.astype(np.uint8)
        return output


def _geometry_matches(reference_topology, reference_points, file, keys):
    mesh = _read_vtu(file, keys, pbar=False)
    if not reference_topology.matches(mesh):
        raise ValueError(f"ALE flow requires static topology; mismatch in {file}")
    if not np.array_equal(reference_points, np.asarray(mesh.points)):
        raise ValueError(f"ALE flow requires static node locations; mismatch in {file}")
    return mesh


def _load_static_ale_series(
    entries,
    *,
    velocity_key,
    displacement_key,
    mesh_velocity_key,
    velocity_scale,
    subsamp,
    dt,
    pbar,
    dtype,
    conform_mesh,
):
    if subsamp < 1:
        raise ValueError("subsamp must be >= 1")
    entries = entries[::subsamp]
    if len(entries) < 2:
        raise ValueError("ALE flow input must contain at least two timesteps")

    raw_times = np.asarray([time for time, _ in entries])
    if len(np.unique(raw_times)) != len(raw_times):
        raise ValueError("ALE flow timesteps must be unique")

    velocities = []
    displacements = []
    mesh_velocities = []
    canonical_velocity = None
    canonical_displacement = None
    canonical_mesh_velocity = None
    reference_topology = None
    reference_points = None
    reference_signature = None
    conditioned = None

    for frame, (_, file) in enumerate(
        tqdm(entries, total=len(entries), disable=not pbar)
    ):
        info = _read_vtu_metadata(file)
        velocity_name = _resolve_point_array(info, velocity_key)
        displacement_name = _resolve_point_array(info, displacement_key)
        mesh_velocity_name = _resolve_point_array(info, mesh_velocity_key)
        if len({velocity_name, displacement_name, mesh_velocity_name}) != 3:
            raise ValueError(
                "velocity, displacement, and mesh-velocity keys must name "
                "different arrays"
            )
        keys = [velocity_name, displacement_name, mesh_velocity_name]
        mesh = None

        if frame == 0:
            mesh = _read_vtu(file, keys, pbar=False)
            reference_topology = MeshTopology.from_mesh(mesh)
            reference_points = np.ascontiguousarray(mesh.points).copy()
            try:
                reference_signature = _geometry_signatures(info)
            except _UnsupportedFastPath:
                reference_signature = None
            conditioned = _condition_mesh(mesh) if conform_mesh else mesh
            canonical_velocity = velocity_name
            canonical_displacement = displacement_name
            canonical_mesh_velocity = mesh_velocity_name
        else:
            if (
                info.n_points != len(reference_points)
                or info.n_cells != len(reference_topology.cell_types)
            ):
                raise ValueError(
                    "ALE flow requires one static mesh; "
                    f"point or cell count differs in {file}"
                )
            try:
                signature = _geometry_signatures(info)
            except _UnsupportedFastPath:
                signature = None
            if reference_signature is None or signature != reference_signature:
                mesh = _geometry_matches(
                    reference_topology, reference_points, file, keys
                )

        if mesh is None:
            try:
                velocity = _read_array(
                    info, info.array("point", velocity_name), n_tuples=info.n_points
                )
                displacement = _read_array(
                    info,
                    info.array("point", displacement_name),
                    n_tuples=info.n_points,
                )
                mesh_velocity = _read_array(
                    info,
                    info.array("point", mesh_velocity_name),
                    n_tuples=info.n_points,
                )
            except _UnsupportedFastPath:
                mesh = _read_vtu(file, keys, pbar=False)
        if mesh is not None:
            velocity = np.asarray(mesh.point_data[velocity_name])
            displacement = np.asarray(mesh.point_data[displacement_name])
            mesh_velocity = np.asarray(mesh.point_data[mesh_velocity_name])

        for role, name, field in (
            ("velocity", velocity_name, velocity),
            ("displacement", displacement_name, displacement),
            ("mesh velocity", mesh_velocity_name, mesh_velocity),
        ):
            if field.shape != (len(reference_points), 3):
                raise ValueError(
                    f"ALE {role} array {name!r} in {file} has shape {field.shape}; "
                    f"expected {(len(reference_points), 3)}"
                )
        velocities.append(
            np.ascontiguousarray(velocity * velocity_scale, dtype=dtype)
        )
        displacements.append(np.ascontiguousarray(displacement, dtype=dtype).copy())
        mesh_velocities.append(
            np.ascontiguousarray(mesh_velocity * velocity_scale, dtype=dtype)
        )

    times_shift_s = raw_times - raw_times[0]
    if dt is not None:
        times_shift_s = times_shift_s * dt
    n_frames = len(entries)
    data = MeshFieldSeries(
        times=raw_times,
        topologies=(MeshTopology.from_mesh(conditioned),),
        topology_ids=np.zeros(n_frames, dtype=np.int64),
        coordinates=(np.ascontiguousarray(conditioned.points).copy(),),
        coordinate_ids=np.zeros(n_frames, dtype=np.int64),
        point_fields={
            canonical_velocity: tuple(velocities),
            canonical_displacement: tuple(displacements),
            canonical_mesh_velocity: tuple(mesh_velocities),
        },
    )
    return (
        data,
        canonical_velocity,
        canonical_displacement,
        canonical_mesh_velocity,
        times_shift_s,
    )


def load_ale_flow(
    path: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    velocity_key: str = "Velocity",
    displacement_key: str = "Displacement",
    mesh_velocity_key: str = "Mesh_velocity",
    *,
    subsamp: int = 1,
    pbar: bool = False,
    dt: float | None = None,
    precision: str = "f64",
    conform_mesh: bool = True,
    velocity_scale: float = 1.0,
) -> ALEFlow:
    """Load ALE velocity, mesh velocity, and displacement on a static mesh.

    ``path`` may be a PVD collection, a directory of VTUs, or an explicit VTU
    path iterable. Every selected frame must have identical node coordinates and
    cell connectivity. All three requested point fields are loaded eagerly.

    Args:
        path: PVD path, VTU directory, or explicit VTU path iterable.
        velocity_key: Three-component physical velocity point field.
        displacement_key: Three-component nodal displacement point field.
        mesh_velocity_key: Three-component mesh-velocity point field.
        subsamp: Keep every Nth frame.
        pbar: Show load progress.
        dt: Optional multiplier for PVD, directory, or file-list time labels.
        precision: Stored field precision, ``"f64"`` or ``"f32"``.
        conform_mesh: Split supported non-tetrahedral cells and remove
            degenerate tetrahedra on the shared reference mesh.
        velocity_scale: Multiplier converting both velocity fields to reference
            mesh spatial units per second. Displacement is not scaled because it
            must already use the same units as the reference coordinates.

    Returns:
        ALEFlow: Static reference mesh with time-resolved velocity and
        displacement fields.
    """
    dtype = resolve_float_dtype(precision)
    velocity_scale = float(velocity_scale)
    if not np.isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("velocity_scale must be finite and > 0")
    entries, _ = _series_source(path, velocity_key)
    (
        data,
        velocity_name,
        displacement_name,
        mesh_velocity_name,
        times_shift_s,
    ) = (
        _load_static_ale_series(
            entries,
            velocity_key=velocity_key,
            displacement_key=displacement_key,
            mesh_velocity_key=mesh_velocity_key,
            velocity_scale=velocity_scale,
            subsamp=subsamp,
            dt=dt,
            pbar=pbar,
            dtype=dtype,
            conform_mesh=conform_mesh,
        )
    )
    return ALEFlow(
        data,
        velocity_name,
        displacement_name,
        mesh_velocity_name,
        times_shift_s,
        velocity_scale,
    )
