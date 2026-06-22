import os
import re
import xml.etree.ElementTree as ET

from copy import deepcopy

import numpy as np
import pyvista as pv

from tqdm.auto import tqdm
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from .sampler import (
    _TetSampler,
    _condition_mesh,
    _sample_v_fallback,
    resolve_float_dtype,
)


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


_TIME_INTERP = ("linear", "cubic")


def resolve_time_interp(time_interp):
    """Validate the temporal interpolation mode (``"linear"`` or ``"cubic"``)."""
    if time_interp not in _TIME_INTERP:
        raise ValueError(
            f"time_interp must be one of {_TIME_INTERP}, got {time_interp!r}")
    return time_interp


def _catmull_rom(p0, p1, p2, p3, s):
    """Uniform Catmull-Rom spline value at local parameter ``s`` in [0, 1].

    Interpolates the smooth segment between knots ``p1`` and ``p2`` using the
    neighbours ``p0`` and ``p3`` to estimate the endpoint tangents. Reproduces
    each knot exactly (s=0 -> p1, s=1 -> p2). ``s`` is a python float so an f32
    field stays f32.

    Uses scalar basis weights (not array-valued coefficients) so each frame is
    touched by a single scalar-times-array multiply -- only ~2x the linear blend
    rather than the many full-field temporaries a Horner form allocates.
    """
    s2 = s * s
    s3 = s2 * s
    w0 = 0.5 * (-s + 2.0 * s2 - s3)
    w1 = 0.5 * (2.0 - 5.0 * s2 + 3.0 * s3)
    w2 = 0.5 * (s + 4.0 * s2 - 3.0 * s3)
    w3 = 0.5 * (-s2 + s3)
    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


def _periodic_distinct_count(get_frame, n_frames):
    """Number of distinct frames per period.

    A pulsatile series often stores the period's closing frame as a duplicate of
    the opening one; if so it is dropped from the periodic wrap so cubic
    interpolation across the cycle boundary doesn't see a repeated knot.
    """
    f0 = np.asarray(get_frame(0))
    fn = np.asarray(get_frame(n_frames - 1))
    scale = float(np.abs(f0).max()) or 1.0
    dup = np.allclose(f0, fn, rtol=1e-3, atol=1e-3 * scale)
    return n_frames - 1 if dup else n_frames


def _require_uniform_spacing(times_shift_s, mode):
    """Cubic interpolation assumes uniform frame spacing -- check it once."""
    if mode == "cubic":
        d = np.diff(np.asarray(times_shift_s, dtype=float))
        if d.size and not np.allclose(d, d[0], rtol=1e-6, atol=1e-12):
            raise ValueError(
                "time_interp='cubic' requires uniformly spaced time frames")


def _interp_time(times_shift_s, tmax, n_distinct, get_frame, time, mode,
                 tol=1e-3):
    """Interpolate the nodal field at ``time`` (periodic) with ``mode``.

    ``get_frame(i)`` returns the nodal velocity array for frame index ``i``.
    Linear reproduces the legacy two-frame blend exactly; cubic uses a uniform
    Catmull-Rom across four frames, wrapping neighbours periodically.
    """
    tw = time % tmax
    inext = int(np.argmax((times_shift_s - tw) > 0))
    iprev = inext - 1
    s = float((tw - times_shift_s[iprev])
              / (times_shift_s[inext] - times_shift_s[iprev]))

    if s < tol:
        return get_frame(iprev)
    if s > 1 - tol:
        return get_frame(inext)
    if mode == "cubic":
        nd = n_distinct
        return _catmull_rom(get_frame((iprev - 1) % nd), get_frame(iprev % nd),
                            get_frame(inext % nd), get_frame((inext + 1) % nd), s)
    return (1.0 - s) * get_frame(iprev) + s * get_frame(inext)


class SingleVTUFlow:
    """Single ``.vtu`` flow file with one static mesh and time-indexed arrays.

    Velocity arrays must use names like ``Velocity_00190`` where ``Velocity`` is
    the chosen ``active_key``.
    """

    def __init__(self, filepath, active_key="velocity", pbar=False,
                 only_active_key=False, precision="f64", time_interp="linear",
                 conform_mesh=True):
        self.filepath = filepath
        self.dtype = resolve_float_dtype(precision)
        self.time_interp = resolve_time_interp(time_interp)
        if only_active_key:
            self.mesh = _read_vtu(filepath, active_key, pbar)
        else:
            self.mesh = pv.read(filepath, progress_bar=pbar)

        # Condition the mesh to clean all-tet (split wedges, drop degenerate
        # cells) so the fast sampler can run; no-op when already clean all-tet.
        if conform_mesh:
            self.mesh = _condition_mesh(self.mesh)

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
        _require_uniform_spacing(self.times_shift_s, self.time_interp)
        self._n_distinct = _periodic_distinct_count(self._frame_vel, len(self._mesh_keys))

        # Store the per-timestep velocity arrays at the working precision so the
        # sampler reads them without a per-call cast (the field dominates the
        # loop's memory bandwidth).
        for key in self._mesh_keys:
            arr = self.mesh.point_data[key]
            if arr.dtype != self.dtype:
                self.mesh.point_data[key] = arr.astype(self.dtype)

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
        self._sampler = _TetSampler(self.active_mesh, dtype=self.dtype)

    def _get_mesh_key(self, index):
        return self._mesh_keys[index]

    def _frame_vel(self, index):
        return self.mesh.point_data[self._mesh_keys[index]]

    def set_active_time(self, time):
        self.active_mesh.point_data[self.active_key] = _interp_time(
            self.times_shift_s, self.tmax, self._n_distinct, self._frame_vel,
            time, self.time_interp)

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

    def __init__(self, filepath, dt=None, active_key="velocity", pbar=True,
                 subsamp=1, precision="f64", time_interp="linear",
                 conform_mesh=True):
        if subsamp < 1:
            raise ValueError("subsamp must be >= 1")

        self.dtype = resolve_float_dtype(precision)
        self.time_interp = resolve_time_interp(time_interp)
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
            # Carry the velocity field at the working precision.
            arr = mesh.point_data[active_key]
            if arr.dtype != self.dtype:
                mesh.point_data[active_key] = arr.astype(self.dtype)
            self.meshes.append(mesh)

        self.tmax = np.max(self.times)
        self.times_shift_s = (self.times - self.times[0])
        if dt is not None:
            self.times_shift_s = self.times_shift_s * dt
        self.tmax = self.times_shift_s.max()
        _require_uniform_spacing(self.times_shift_s, self.time_interp)
        self._n_distinct = _periodic_distinct_count(self._frame_vel, len(self.meshes))

        self.active_mesh = deepcopy(self.meshes[0])
        self.active_mesh[self.active_key] = self.active_mesh[self.active_key]*0.
        # Clean geometry for the sampler; per-frame fields stay node-aligned.
        if conform_mesh:
            self.active_mesh = _condition_mesh(self.active_mesh)

        # Build cell locator for only first mesh - assumes static mesh
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.active_mesh)
        self.locator.BuildLocator()

        # Fast tet sampler (locator built once, reused across all calls)
        self._sampler = _TetSampler(self.active_mesh, dtype=self.dtype)

    def _frame_vel(self, index):
        return self.meshes[index].point_data[self.active_key]

    def set_active_time(self, time):
        self.active_mesh.point_data[self.active_key] = _interp_time(
            self.times_shift_s, self.tmax, self._n_distinct, self._frame_vel,
            time, self.time_interp)

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

    def __init__(self, filepath, dt=None, active_key="velocity", pbar=True,
                 subsamp=1, precision="f64", time_interp="linear",
                 conform_mesh=True):
        if subsamp < 1:
            raise ValueError("subsamp must be >= 1")
        self.dtype = resolve_float_dtype(precision)
        self.time_interp = resolve_time_interp(time_interp)
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
            self.fields.append(
                np.ascontiguousarray(m.point_data[active_key], dtype=self.dtype))

        self.times_shift_s = self.times - self.times[0]
        if dt is not None:
            self.times_shift_s = self.times_shift_s * dt
        self.tmax = self.times_shift_s.max()
        _require_uniform_spacing(self.times_shift_s, self.time_interp)
        self._n_distinct = _periodic_distinct_count(self._frame_vel, len(self.fields))

        # Clean geometry for the sampler; per-frame fields stay node-aligned
        # (conditioning preserves all points).
        if conform_mesh:
            geom = _condition_mesh(geom)
        self.active_mesh = geom
        self.active_mesh.point_data[self.active_key] = self.fields[0].copy()

        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.active_mesh)
        self.locator.BuildLocator()

        # Fast tet sampler (locator built once, reused across all calls)
        self._sampler = _TetSampler(self.active_mesh, dtype=self.dtype)

    def _frame_vel(self, index):
        return self.fields[index]

    def set_active_time(self, time):
        self.active_mesh.point_data[self.active_key] = _interp_time(
            self.times_shift_s, self.tmax, self._n_distinct, self._frame_vel,
            time, self.time_interp)

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





def load_flow(path, active_key="velocity", subsamp=1, only_active_key=True,
              pbar=False, dt=None, precision="f64", time_interp="linear",
              conform_mesh=True):
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
        precision (str): Working precision for the sampling/advection math,
            ``"f64"`` (default, double) or ``"f32"`` (single -- roughly halves
            the field's memory bandwidth for a speedup, at a looser geometric
            tolerance and reduced trajectory accuracy).
        time_interp (str): Temporal interpolation between stored frames,
            ``"linear"`` (default) or ``"cubic"`` (uniform Catmull-Rom over four
            frames -- 2nd-order-consistent with the solver's time integration,
            removing the per-frame velocity kink; requires uniform frame
            spacing).
        conform_mesh (bool): When ``True`` (default), condition the mesh to clean
            all-tetrahedral at load -- split non-tet cells (e.g. boundary-layer
            wedges) into tets and drop degenerate (near-zero-volume) cells -- so
            the fast sampler can run. A no-op for already-clean all-tet meshes.
            Set ``False`` to load the mesh as-is (for debugging).

    Returns:
        (Union[SingleVTUFlow, StaticPVDFlow]): A flow object compatible with
            ``track`` and ``BoundaryReseeder``.
    """
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "vtu":
        return SingleVTUFlow(path, active_key=active_key, pbar=pbar,
                             only_active_key=only_active_key, precision=precision,
                             time_interp=time_interp, conform_mesh=conform_mesh)
    if ext == "pvd":
        return StaticPVDFlow(path, active_key=active_key, pbar=pbar,
                             subsamp=subsamp, dt=dt, precision=precision,
                             time_interp=time_interp, conform_mesh=conform_mesh)
    raise ValueError(f"unsupported flow file type: .{ext} (expected .vtu or .pvd)")
