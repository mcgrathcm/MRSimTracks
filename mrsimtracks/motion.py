import os

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm

from .io import (
    MeshTopology,
    _UnsupportedFastPath,
    _filename_time,
    _interp_weights,
    _load_single_vtu,
    _load_vtu_series,
    _metadata_time,
    _parse_pvd,
    _read_array,
    _read_vtu,
    _read_vtu_metadata,
    _series_source,
    _center_mesh_frames,
)
from .sampler import _condition_mesh, _tet_volumes, resolve_float_dtype


@dataclass(frozen=True)
class MaterialPoints:
    """Fixed cell-local coordinates for particles attached to a mesh."""

    cell_ids: np.ndarray
    node_ids: np.ndarray
    weights: np.ndarray

    @property
    def n_particles(self):
        return len(self.cell_ids)


class MaterialTrajectory:
    """In-memory or HDF5-backed fixed-topology material trajectories."""

    def __init__(self, positions=None, times=None, *, path=None, shape=None):
        if positions is None and path is None:
            raise ValueError("MaterialTrajectory requires positions or an HDF5 path")
        self._positions = None if positions is None else np.asarray(positions)
        self.path = None if path is None else Path(path)
        self._times = None if times is None else np.asarray(times, dtype=float)
        if shape is not None:
            self._shape = tuple(shape)
        elif self._positions is not None:
            self._shape = self._positions.shape
        else:
            self._shape = self._read_shape()

    @property
    def is_file_backed(self):
        return self.path is not None and self._positions is None

    @property
    def positions(self):
        if self._positions is None:
            import h5py

            with h5py.File(self.path, "r") as file:
                self._positions = file["position"][...]
        return self._positions

    @property
    def times(self):
        if self._times is None:
            import h5py

            with h5py.File(self.path, "r") as file:
                self._times = file["time"][...]
        return self._times

    @property
    def shape(self):
        return self._shape

    @property
    def n_frames(self):
        return self._shape[0]

    @property
    def n_particles(self):
        return self._shape[1]

    @classmethod
    def open(cls, path):
        """Open a streamed material trajectory without loading positions."""
        return cls(path=path)

    def _read_shape(self):
        import h5py

        with h5py.File(self.path, "r") as file:
            return file["position"].shape


class MeshMotion:
    """Fixed-topology mesh deformation represented by absolute node positions."""

    def __init__(self, times, times_shift_s, topology, node_positions, *,
                 dtype, periodic, source, origin_shift=None):
        self.times = np.asarray(times)
        self.times_shift_s = np.asarray(times_shift_s, dtype=float)
        self.node_positions = tuple(
            np.ascontiguousarray(points, dtype=dtype) for points in node_positions
        )
        self.dtype = np.dtype(dtype)
        self.periodic = bool(periodic)
        self.source = source
        self.origin_shift = np.zeros(3) if origin_shift is None else np.asarray(
            origin_shift, dtype=float
        )

        if len(self.times) < 2:
            raise ValueError("mesh motion requires at least two frames")
        if len(self.node_positions) != len(self.times):
            raise ValueError("times and node-position frames must have equal length")
        if np.any(np.diff(self.times_shift_s) <= 0):
            raise ValueError("mesh motion timesteps must be strictly increasing")
        self.tmax = float(self.times_shift_s[-1])
        if self.tmax <= 0:
            raise ValueError("mesh motion duration must be positive")

        n_points = len(self.node_positions[0])
        for frame, points in enumerate(self.node_positions):
            if points.shape != (n_points, 3):
                raise ValueError(
                    f"node-position frame {frame} has shape {points.shape}; "
                    f"expected {(n_points, 3)}"
                )
            if not np.isfinite(points).all():
                raise ValueError(f"node-position frame {frame} contains non-finite values")

        reference_mesh = pv.UnstructuredGrid(
            topology.cells,
            topology.cell_types,
            self.node_positions[0],
            deep=False,
        )
        conditioned = _condition_mesh(reference_mesh)
        if conditioned.n_points != n_points:
            raise ValueError("mesh tetrahedralization must preserve the original nodes")
        self.topology = MeshTopology.from_mesh(conditioned)

    @property
    def n_frames(self):
        return len(self.node_positions)

    @property
    def n_points(self):
        return len(self.node_positions[0])

    def mesh(self, frame=0):
        """Return one mesh frame with the shared topology."""
        return pv.UnstructuredGrid(
            self.topology.cells,
            self.topology.cell_types,
            self.node_positions[frame],
            deep=False,
        )

    def seed(self, n_particles, rng=None):
        """Uniformly seed exactly ``n_particles`` material points by volume.

        Hexahedra, wedges, and other supported volume cells are split once when
        the motion is loaded. Particles retain the resulting tetrahedron id and
        barycentric weights for every subsequent frame.
        """
        n_particles = int(n_particles)
        if n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        rng = rng if rng is not None else np.random.default_rng()

        mesh = self.mesh(0)
        if mesh.n_cells == 0 or not np.all(mesh.celltypes == pv.CellType.TETRA):
            raise ValueError("mesh volume could not be tetrahedralized for seeding")

        connectivity = mesh.cells.reshape(-1, 5)[:, 1:]
        volumes = _tet_volumes(np.asarray(mesh.points), connectivity)
        total_volume = float(volumes.sum())
        if not np.isfinite(total_volume) or total_volume <= 0:
            raise ValueError("mesh must contain positive-volume cells")

        tet_ids = rng.choice(
            mesh.n_cells, size=n_particles, p=volumes / total_volume
        )
        barycentric = rng.exponential(size=(n_particles, 4))
        barycentric /= barycentric.sum(axis=1, keepdims=True)
        seeds = np.einsum(
            "pi,pij->pj",
            barycentric,
            np.asarray(mesh.points)[connectivity[tet_ids]],
        )
        particles = MaterialPoints(
            np.ascontiguousarray(tet_ids, dtype=np.int64),
            np.ascontiguousarray(connectivity[tet_ids], dtype=np.int64),
            np.ascontiguousarray(barycentric, dtype=self.dtype),
        )

        reconstructed = self._frame_positions(particles, 0)
        scale = float(np.ptp(self.node_positions[0], axis=0).max()) or 1.0
        tolerance = 2e-6 * scale if self.dtype == np.float32 else 1e-10 * scale
        if not np.allclose(reconstructed, seeds, rtol=0, atol=tolerance):
            raise ValueError("failed to reconstruct seeded material coordinates")
        return particles

    def _frame_positions(self, particles, frame):
        nodes = self.node_positions[frame][particles.node_ids]
        return np.einsum("pi,pij->pj", particles.weights, nodes)

    def _time_weights(self, time):
        time = float(time)
        if self.periodic:
            return _interp_weights(
                self.times_shift_s,
                self.tmax,
                len(self.times),
                time,
                "linear",
            )
        if time < 0 or time > self.tmax:
            raise ValueError(f"time must be within [0, {self.tmax}]")
        index = int(np.searchsorted(self.times_shift_s, time, side="left"))
        if index == len(self.times_shift_s):
            return (index - 1,), (1.0,)
        if self.times_shift_s[index] == time:
            return (index,), (1.0,)
        previous = index - 1
        fraction = (
            (time - self.times_shift_s[previous])
            / (self.times_shift_s[index] - self.times_shift_s[previous])
        )
        return (previous, index), (1.0 - fraction, fraction)

    def positions(self, particles, time):
        """Evaluate material-particle positions at one time."""
        indices, weights = self._time_weights(time)
        output = weights[0] * self._frame_positions(particles, indices[0])
        for index, weight in zip(indices[1:], weights[1:], strict=True):
            output += weight * self._frame_positions(particles, index)
        return np.ascontiguousarray(output, dtype=self.dtype)

    def point_cloud(self, particles, time):
        """Return material-particle positions as a PyVista point cloud."""
        cloud = pv.PolyData(self.positions(particles, time))
        cloud.point_data["material_cell_id"] = particles.cell_ids
        return cloud

    def trajectory(self, particles, times=None, output_path=None):
        """Evaluate material positions in memory or stream them to HDF5."""
        stored_frames = times is None
        evaluation_times = (
            self.times_shift_s.copy()
            if stored_frames
            else np.asarray(times, dtype=float)
        )
        if evaluation_times.ndim != 1 or len(evaluation_times) == 0:
            raise ValueError("times must be a non-empty one-dimensional array")
        shape = (len(evaluation_times), particles.n_particles, 3)

        def evaluate(index, time):
            if stored_frames:
                return self._frame_positions(particles, index)
            return self.positions(particles, time)

        if output_path is None:
            positions = np.empty(shape, dtype=self.dtype)
            for index, time in enumerate(evaluation_times):
                positions[index] = evaluate(index, time)
            return MaterialTrajectory(positions, evaluation_times)

        import h5py

        output_path = Path(output_path)
        particle_chunks = int(np.ceil(particles.n_particles / 65_536))
        chunk_particles = int(np.ceil(particles.n_particles / particle_chunks))
        with h5py.File(output_path, "w") as file:
            dataset = file.create_dataset(
                "position",
                shape=shape,
                dtype=self.dtype,
                chunks=(1, chunk_particles, 3),
            )
            file.create_dataset("time", data=evaluation_times)
            file.attrs["kind"] = "fixed_topology_material_motion"
            file.attrs["periodic"] = self.periodic
            for index, time in enumerate(evaluation_times):
                dataset[index] = evaluate(index, time)
        return MaterialTrajectory(
            path=output_path,
            shape=shape,
            times=evaluation_times,
        )


def _coordinate_series_source(path):
    if isinstance(path, (str, os.PathLike)):
        source = Path(path)
        if source.is_file() and source.suffix.lower() == ".pvd":
            return [(time, file, None) for time, file in _parse_pvd(source)]
        if source.is_dir():
            candidates = sorted(source.glob("*.vtu"))
        elif source.is_file() and source.suffix.lower() == ".vtu":
            candidates = [source]
        else:
            raise ValueError(
                f"unsupported mesh-motion source: {source} "
                "(expected .vtu, .pvd, a directory, or a VTU file list)"
            )
    else:
        candidates = [Path(file) for file in path]

    entries = []
    for file in candidates:
        if file.suffix.lower() != ".vtu":
            continue
        metadata = _read_vtu_metadata(file)
        time = _metadata_time(metadata)
        entries.append(
            (time if time is not None else _filename_time(file), file, metadata)
        )
    entries.sort(key=lambda entry: entry[0])
    return entries


def _load_coordinate_motion(path, *, subsamp, pbar, dt, dtype):
    entries = _coordinate_series_source(path)[::subsamp]
    if len(entries) < 2:
        raise ValueError("coordinate mesh motion requires at least two VTU frames")
    raw_times = np.asarray([time for time, _, _ in entries])
    if len(set(raw_times)) != len(raw_times):
        raise ValueError("mesh-motion timesteps must be unique")

    topology = None
    node_positions = []
    n_points = n_cells = None
    midpoint = len(entries) // 2
    for frame, (_, file, metadata) in enumerate(
        tqdm(entries, total=len(entries), disable=not pbar)
    ):
        info = metadata if metadata is not None else _read_vtu_metadata(file)
        if frame == 0:
            mesh = _read_vtu(file, [], pbar=False)
            topology = MeshTopology.from_mesh(mesh)
            n_points, n_cells = mesh.n_points, mesh.n_cells
            points = np.asarray(mesh.points)
        else:
            if info.n_points != n_points or info.n_cells != n_cells:
                raise ValueError(
                    "fixed-topology mesh motion requires every frame to have "
                    f"{n_points} points and {n_cells} cells; {file} has "
                    f"{info.n_points} points and {info.n_cells} cells"
                )
            if frame == midpoint:
                mesh = _read_vtu(file, [], pbar=False)
                if not topology.matches(mesh):
                    raise ValueError(
                        "fixed-topology mesh motion midpoint frame has different "
                        f"topology: {file}"
                    )
                points = np.asarray(mesh.points)
            else:
                point_arrays = info.arrays_in("points")
                if len(point_arrays) != 1:
                    raise ValueError(f"{file} does not contain one points array")
                try:
                    points = _read_array(info, point_arrays[0], n_tuples=n_points)
                except _UnsupportedFastPath:
                    points = np.asarray(_read_vtu(file, [], pbar=False).points)
        node_positions.append(np.ascontiguousarray(points, dtype=dtype).copy())

    times_shift_s = raw_times - raw_times[0]
    if dt is not None:
        times_shift_s *= dt
    return raw_times, times_shift_s, topology, node_positions


def load_mesh_motion(
    path: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    displacement_key: str | None = None,
    *,
    subsamp: int = 1,
    pbar: bool = False,
    dt: float | None = None,
    precision: str = "f64",
    periodic: bool = True,
    center_mesh: bool = False,
) -> MeshMotion:
    """Load fixed-topology mesh motion from coordinates or nodal displacement.

    Args:
        path: A VTU/PVD path, directory, or explicit VTU path iterable.
        displacement_key: Three-component nodal displacement field. When
            omitted, every VTU frame's point coordinates are treated as the
            absolute node positions.
        subsamp: Keep every Nth frame.
        pbar: Show load progress.
        dt: Optional multiplier for PVD, directory, or file-list time labels.
        precision: Stored node-position precision, ``"f64"`` or ``"f32"``.
        periodic: Wrap evaluation times over the loaded motion duration.
        center_mesh: Translate every loaded mesh frame by the same vector so
            the initial frame's axis-aligned bounds are centered at the origin.
            The default is ``False``. Displacement fields are unchanged.

    Returns:
        MeshMotion: Fixed-topology deformation ready for material seeding.
    """
    if subsamp < 1:
        raise ValueError("subsamp must be >= 1")
    dtype = resolve_float_dtype(precision)

    if displacement_key is None:
        times, times_shift_s, topology, node_positions = _load_coordinate_motion(
            path,
            subsamp=subsamp,
            pbar=pbar,
            dt=dt,
            dtype=dtype,
        )
        source = "coordinates"
    else:
        single_vtu = (
            isinstance(path, (str, os.PathLike))
            and Path(path).suffix.lower() == ".vtu"
        )
        if single_vtu:
            data, key, times_shift_s = _load_single_vtu(
                path,
                displacement_key,
                subsamp=subsamp,
                only_active_key=True,
                pbar=pbar,
                dtype=dtype,
                conform_mesh=False,
            )
        else:
            entries, metadata = _series_source(path, displacement_key)
            data, key, times_shift_s = _load_vtu_series(
                entries,
                metadata,
                displacement_key,
                subsamp=subsamp,
                dt=dt,
                pbar=pbar,
                dtype=dtype,
                conform_mesh=False,
                mesh_mode="static",
            )
        reference_points = np.asarray(data.coordinates[0], dtype=dtype)
        node_positions = tuple(
            np.ascontiguousarray(displacement, dtype=dtype)
            for displacement in data.point_fields[key]
        )
        for points in node_positions:
            np.add(points, reference_points, out=points)
        times = data.times
        topology = data.topologies[0]
        source = "displacement"

    origin_shift = np.zeros(3)
    if center_mesh:
        node_positions, origin_shift = _center_mesh_frames(node_positions)

    return MeshMotion(
        times,
        times_shift_s,
        topology,
        node_positions,
        dtype=dtype,
        periodic=periodic,
        source=source,
        origin_shift=origin_shift,
    )
