import os

from collections.abc import Iterable

import numpy as np

from tqdm.auto import tqdm

from .io import (
    MeshFieldSeries,
    MeshTopology,
    _UnsupportedFastPath,
    _geometry_signatures,
    _read_array,
    _read_vtu,
    _read_vtu_metadata,
    _resolve_point_array,
    _series_source,
)
from .sampler import _condition_mesh, resolve_float_dtype


class ALEFlow:
    """Velocity and displacement fields on one fixed reference mesh."""

    def __init__(self, data, velocity_key, displacement_key, times_shift_s):
        if data.geometry_mode != "static":
            raise ValueError("ALEFlow requires one static reference mesh")
        self.data = data
        self.velocity_key = velocity_key
        self.displacement_key = displacement_key
        self.times = np.asarray(data.times)
        self.times_shift_s = np.asarray(times_shift_s, dtype=float)
        if np.any(np.diff(self.times_shift_s) <= 0):
            raise ValueError("ALE flow timesteps must be strictly increasing")
        self.tmax = float(self.times_shift_s[-1])
        self.reference_mesh = data.mesh(0)
        self.reference_mesh.point_data[velocity_key] = self.velocity(0).copy()
        self.reference_mesh.point_data[displacement_key] = (
            self.displacement(0).copy()
        )

    @property
    def n_frames(self):
        return len(self.times)

    def velocity(self, frame):
        return self.data.field(self.velocity_key, frame)

    def displacement(self, frame):
        return self.data.field(self.displacement_key, frame)

    def frame(self, frame):
        """Return the static reference mesh with one frame's two fields."""
        mesh = self.data.mesh(frame)
        mesh.point_data[self.velocity_key] = self.velocity(frame)
        mesh.point_data[self.displacement_key] = self.displacement(frame)
        return mesh


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
    canonical_velocity = None
    canonical_displacement = None
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
        if velocity_name == displacement_name:
            raise ValueError(
                "velocity_key and displacement_key must name different arrays"
            )
        keys = [velocity_name, displacement_name]
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
            except _UnsupportedFastPath:
                mesh = _read_vtu(file, keys, pbar=False)
        if mesh is not None:
            velocity = np.asarray(mesh.point_data[velocity_name])
            displacement = np.asarray(mesh.point_data[displacement_name])

        for role, name, field in (
            ("velocity", velocity_name, velocity),
            ("displacement", displacement_name, displacement),
        ):
            if field.shape != (len(reference_points), 3):
                raise ValueError(
                    f"ALE {role} array {name!r} in {file} has shape {field.shape}; "
                    f"expected {(len(reference_points), 3)}"
                )
        velocities.append(np.ascontiguousarray(velocity, dtype=dtype).copy())
        displacements.append(np.ascontiguousarray(displacement, dtype=dtype).copy())

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
        },
    )
    return data, canonical_velocity, canonical_displacement, times_shift_s


def load_ale_flow(
    path: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    velocity_key: str = "Velocity",
    displacement_key: str = "Displacement",
    *,
    subsamp: int = 1,
    pbar: bool = False,
    dt: float | None = None,
    precision: str = "f64",
    conform_mesh: bool = True,
) -> ALEFlow:
    """Load ALE velocity and displacement on a verified static mesh.

    ``path`` may be a PVD collection, a directory of VTUs, or an explicit VTU
    path iterable. Every selected frame must have identical node coordinates and
    cell connectivity. Both requested point fields are loaded eagerly.

    Args:
        path: PVD path, VTU directory, or explicit VTU path iterable.
        velocity_key: Three-component physical velocity point field.
        displacement_key: Three-component nodal displacement point field.
        subsamp: Keep every Nth frame.
        pbar: Show load progress.
        dt: Optional multiplier for PVD, directory, or file-list time labels.
        precision: Stored field precision, ``"f64"`` or ``"f32"``.
        conform_mesh: Split supported non-tetrahedral cells and remove
            degenerate tetrahedra on the shared reference mesh.

    Returns:
        ALEFlow: Static reference mesh with time-resolved velocity and
        displacement fields.
    """
    dtype = resolve_float_dtype(precision)
    entries, _ = _series_source(path, velocity_key)
    data, velocity_name, displacement_name, times_shift_s = (
        _load_static_ale_series(
            entries,
            velocity_key=velocity_key,
            displacement_key=displacement_key,
            subsamp=subsamp,
            dt=dt,
            pbar=pbar,
            dtype=dtype,
            conform_mesh=conform_mesh,
        )
    )
    return ALEFlow(data, velocity_name, displacement_name, times_shift_s)
