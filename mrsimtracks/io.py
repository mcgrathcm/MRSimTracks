import os
import re
import xml.etree.ElementTree as ET

from copy import deepcopy

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from .sampler import _TetSampler, _sample_v_fallback


def _read_vtu(filepath, active_key, pbar):
    """Read a .vtu, loading only point arrays whose name contains active_key.

    The pulsatile files store pressure_NNNNN alongside velocity_NNNNN for every
    timestep; tracking never uses pressure, so skipping it cuts both read time
    (~40%) and peak memory (~25%) -- important when each worker reloads the file.
    """
    reader = pv.get_reader(filepath)
    reader.disable_all_point_arrays()
    matches = [n for n in reader.point_array_names if active_key in n]
    if not matches:
        raise ValueError(
            f"no point-data arrays containing {active_key!r} found in {filepath}"
        )
    for n in matches:
        if active_key in n:
            reader.enable_point_array(n)
    if pbar:
        reader.show_progress()
    return reader.read()


class SingleVTUFlow:
    """Single ``.vtu`` flow file with one static mesh and time-indexed arrays.

    Velocity arrays must use names like ``Velocity_00190`` where ``Velocity`` is
    the chosen ``active_key``.
    """

    def __init__(self, filepath, active_key="velocity", pbar=False, only_active_key=False):
        self.filepath = filepath
        if only_active_key:
            self.mesh = _read_vtu(filepath, active_key, pbar)
        else:
            self.mesh = pv.read(filepath, progress_bar=pbar)

        # Extract all time steps from the point data keys
        self.active_key = active_key
        time_keys = []
        pattern = re.compile(rf"^{re.escape(active_key)}_(\d+)$")
        for key in self.mesh.point_data.keys():
            match = pattern.match(key)
            if match is not None:
                time_keys.append((int(match.group(1)), key))

        if len(time_keys) < 2:
            raise ValueError(
                f"{filepath} must contain at least two point-data arrays named "
                f"{active_key}_NNNNN"
            )

        time_keys.sort(key=lambda item: item[0])
        self.times = np.array([item[0] for item in time_keys])
        self._mesh_keys = [item[1] for item in time_keys]
        self.times_shift_s = (self.times - self.times[0])/1000
        self.tmax = self.times_shift_s.max()

        # Set up active mesh
        self.active_mesh = self.mesh.copy()
        self.active_mesh.clear_point_data()
        self.active_mesh.clear_cell_data()

        self.active_mesh.point_data[self.active_key] = self.mesh.point_data[self._get_mesh_key(0)]

        # Build cell locator
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.active_mesh)
        self.locator.BuildLocator()

        # Fast tet sampler (locator built once, reused across all calls)
        self._sampler = _TetSampler(self.active_mesh)

    def _get_mesh_key(self, index):
        return self._mesh_keys[index]

    def set_active_time(self, time):
        time_wrapped = time % self.tmax

        ind_next = np.argmax((self.times_shift_s - time_wrapped) > 0)
        ind_prev = ind_next - 1
        weight_next = (time_wrapped - self.times_shift_s[ind_prev]) / (self.times_shift_s[ind_next] - self.times_shift_s[ind_prev])

        tol = 0.001 # Tolerance of 0.1% of dt

        if weight_next < tol:
            key = self._get_mesh_key(ind_prev)
            self.active_mesh[self.active_key] = self.mesh[key]
        elif weight_next > 1 - tol:
            key = self._get_mesh_key(ind_next)
            self.active_mesh[self.active_key] = self.mesh[key]
        else:
            key_prev = self._get_mesh_key(ind_prev)
            key_next = self._get_mesh_key(ind_next)
            v_prev = self.mesh[key_prev]
            v_next = self.mesh[key_next]
            self.active_mesh[self.active_key] = (1 - weight_next)*v_prev + weight_next*v_next

    def get_mesh(self, time):
        self.set_active_time(time)
        return self.active_mesh

    def sample(self, points:pv.PolyData, time):
        self.set_active_time(time)
        return points.sample(self.active_mesh, locator=self.locator, pass_cell_data=False, pass_point_data=False, pass_field_data=False)

    def sample_v(self, points_xyz, time, guess=None):
        """Fast path: return (velocity (n,3), valid (n,), cells (n,)) numpy arrays."""
        self.set_active_time(time)
        if not self._sampler.ok:
            return _sample_v_fallback(self, points_xyz, time)
        vel = np.asarray(self.active_mesh.point_data[self.active_key])
        return self._sampler.sample(points_xyz, vel, guess=guess)

class PVDFlow:
    """Time-resolved ``.pvd`` flow that stores one full mesh per timestep."""

    def __init__(self, filepath, dt=None, active_key="velocity", pbar=True, subsamp=1):
        if subsamp < 1:
            raise ValueError("subsamp must be >= 1")

        self.reader = pv.get_reader(filepath)
        self.times = np.array(self.reader.time_values)
        if subsamp > 1:
            self.times = self.times[::subsamp]
        if len(self.times) < 2:
            raise ValueError(f"{filepath} must contain at least two timesteps")
        self.active_key = active_key

        self.meshes = []
        # Preload all meshes
        for t in tqdm(self.times, disable=not pbar):
            self.reader.set_active_time_value(t)
            mesh = self.reader.read()[0]
            if active_key not in mesh.point_data:
                raise ValueError(
                    f"point-data array {active_key!r} not found at timestep {t}"
                )
            self.meshes.append(mesh)

        self.tmax = np.max(self.times)
        self.times_shift_s = (self.times - self.times[0])
        if dt is not None:
            self.times_shift_s = self.times_shift_s * dt
        self.tmax = self.times_shift_s.max()
        
        self.active_mesh = deepcopy(self.meshes[0])
        self.active_mesh[self.active_key] = self.active_mesh[self.active_key]*0.

        # Build cell locator for only first mesh - assumes static mesh
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.active_mesh)
        self.locator.BuildLocator()

        # Fast tet sampler (locator built once, reused across all calls)
        self._sampler = _TetSampler(self.active_mesh)

    def set_active_time(self, time):

        time_wrapped = time % self.tmax

        # Get indices

        ind_next = np.argmax((self.times_shift_s - time_wrapped) > 0)
        ind_prev = ind_next - 1
        weight_next = (time_wrapped - self.times_shift_s[ind_prev]) / (self.times_shift_s[ind_next] - self.times_shift_s[ind_prev])

        tol = 0.001

        if weight_next < tol:
            self.active_mesh[self.active_key] = self.meshes[ind_prev][self.active_key]
        elif weight_next > 1 - tol:
            self.active_mesh[self.active_key] = self.meshes[ind_next][self.active_key]
        else:
            v_prev = self.meshes[ind_prev][self.active_key]
            v_next = self.meshes[ind_next][self.active_key]
            self.active_mesh[self.active_key] = (1 - weight_next)*v_prev + weight_next*v_next

    def get_mesh(self, time):
        self.set_active_time(time)
        return self.active_mesh


    def sample(self, points:pv.PolyData, time):
        self.set_active_time(time)
        return points.sample(self.active_mesh, locator=self.locator, pass_cell_data=False, pass_point_data=False, pass_field_data=False)

    def sample_v(self, points_xyz, time, guess=None):
        """Fast path: return (velocity (n,3), valid (n,), cells (n,)) numpy arrays."""
        self.set_active_time(time)
        if not self._sampler.ok:
            return _sample_v_fallback(self, points_xyz, time)
        vel = np.asarray(self.active_mesh.point_data[self.active_key])
        return self._sampler.sample(points_xyz, vel, guess=guess)


def _parse_pvd(filepath):
    """Return [(time, vtu_abspath), ...] sorted by time from a .pvd collection."""
    base = os.path.dirname(os.path.abspath(filepath))
    root = ET.parse(filepath).getroot()
    out = [(float(ds.get("timestep")), os.path.join(base, ds.get("file")))
           for ds in root.iter("DataSet")]
    out.sort(key=lambda e: e[0])
    if not out:
        raise ValueError(f"{filepath} does not contain any DataSet entries")
    return out


class StaticPVDFlow:
    """Time-resolved flow over a STATIC mesh from a .pvd series.

    Unlike ``PVDFlow``, which keeps a full mesh copy per timestep (so memory
    scales as one-mesh-per-frame, with the connectivity duplicated hundreds of
    times), this stores the geometry once and only the active field per frame --
    cutting memory from ~one-mesh-per-frame to ~one-field-per-frame (e.g. 430
    frames of a 1.1M-cell tet mesh: tens of GB -> ~1-2 GB). It assumes the mesh
    does not change in time (``PVDFlow`` already assumes this for its single
    locator). Only the active field is read from each file (pressure etc. skipped).

    Drop-in for the other flow classes: same set_active_time / sample /
    sample_v / get_mesh interface.
    """

    def __init__(self, filepath, dt=None, active_key="velocity", pbar=True, subsamp=1):
        if subsamp < 1:
            raise ValueError("subsamp must be >= 1")
        entries = _parse_pvd(filepath)
        self.times = np.array([e[0] for e in entries])
        files = [e[1] for e in entries]
        if subsamp > 1:
            self.times = self.times[::subsamp]
            files = files[::subsamp]
        if len(self.times) < 2:
            raise ValueError(f"{filepath} must contain at least two timesteps")
        self.active_key = active_key

        # One geometry (from the first frame) + only the active field per frame.
        self.fields = []
        geom = None
        for f in tqdm(files, disable=not pbar):
            m = _read_vtu(f, active_key, pbar=False)
            if active_key not in m.point_data:
                raise ValueError(f"point-data array {active_key!r} not found in {f}")
            if geom is None:
                geom = m
            self.fields.append(np.ascontiguousarray(m.point_data[active_key]))

        self.times_shift_s = self.times - self.times[0]
        if dt is not None:
            self.times_shift_s = self.times_shift_s * dt
        self.tmax = self.times_shift_s.max()

        self.active_mesh = geom
        self.active_mesh.point_data[self.active_key] = self.fields[0].copy()

        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.active_mesh)
        self.locator.BuildLocator()

        # Fast tet sampler (locator built once, reused across all calls)
        self._sampler = _TetSampler(self.active_mesh)

    def set_active_time(self, time):
        time_wrapped = time % self.tmax

        ind_next = np.argmax((self.times_shift_s - time_wrapped) > 0)
        ind_prev = ind_next - 1
        weight_next = (time_wrapped - self.times_shift_s[ind_prev]) / (self.times_shift_s[ind_next] - self.times_shift_s[ind_prev])

        tol = 0.001

        if weight_next < tol:
            self.active_mesh.point_data[self.active_key] = self.fields[ind_prev]
        elif weight_next > 1 - tol:
            self.active_mesh.point_data[self.active_key] = self.fields[ind_next]
        else:
            self.active_mesh.point_data[self.active_key] = (1 - weight_next)*self.fields[ind_prev] + weight_next*self.fields[ind_next]

    def get_mesh(self, time):
        self.set_active_time(time)
        return self.active_mesh

    def sample(self, points:pv.PolyData, time):
        self.set_active_time(time)
        return points.sample(self.active_mesh, locator=self.locator, pass_cell_data=False, pass_point_data=False, pass_field_data=False)

    def sample_v(self, points_xyz, time, guess=None):
        """Fast path: return (velocity (n,3), valid (n,), cells (n,)) numpy arrays."""
        self.set_active_time(time)
        if not self._sampler.ok:
            return _sample_v_fallback(self, points_xyz, time)
        vel = np.asarray(self.active_mesh.point_data[self.active_key])
        return self._sampler.sample(points_xyz, vel, guess=guess)





def load_flow(path, active_key="velocity", subsamp=1, only_active_key=True, pbar=False, dt=None):
    """Load a time-resolved flow field, picking the right reader for the file type.

    ``.vtu`` inputs are interpreted as one static mesh with one velocity array
    per timestep, named ``{active_key}_NNNNN``. ``.pvd`` inputs are interpreted
    as a static-geometry series and loaded with one geometry plus one active
    velocity field per frame.

    Args:
        path (str | pathlib.Path): Flow file path ending in ``.vtu`` or
            ``.pvd``.
        active_key (str): Velocity array prefix/name.
        subsamp (int): Keep every Nth frame for ``.pvd`` inputs.
        only_active_key (bool): For ``.vtu`` files, skip unrelated point-data
            arrays.
        pbar (bool): Show reader progress.
        dt (float | None): Optional timestep scale for ``.pvd`` time values.

    Returns:
        (Union[SingleVTUFlow, StaticPVDFlow]): A flow object compatible with
            ``track`` and ``BoundaryReseeder``.
    """
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "vtu":
        return SingleVTUFlow(path, active_key=active_key, pbar=pbar,
                             only_active_key=only_active_key)
    if ext == "pvd":
        return StaticPVDFlow(path, active_key=active_key, pbar=pbar,
                             subsamp=subsamp, dt=dt)
    raise ValueError(f"unsupported flow file type: .{ext} (expected .vtu or .pvd)")
