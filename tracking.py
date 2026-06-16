import time

import numpy as np
import pyvista as pv
import scipy

from copy import deepcopy
from tqdm.auto import tqdm

from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator, vtkCellTreeLocator
from vtkmodules.vtkFiltersCore import vtkProbeFilter

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:                       # numba is optional; numpy path still works
    _HAVE_NUMBA = False

VTK_TETRA = 10


if _HAVE_NUMBA:
    @njit(parallel=True, cache=True)
    def _walk_interp_kernel(points, Minv, dd, conn, adj, vel, guess,
                            tol, slack, max_iter, out_v, out_cells, out_status):
        """Per-particle tet walk + barycentric interpolation.

        For each particle, walk from its guess cell toward the query point and,
        on success, interpolate ``vel`` in place. ``out_status``: 0 located &
        interpolated; 1 hit a domain boundary; 2 did not converge; 3 no guess.
        Non-zero statuses are resolved by the caller's locator fallback.
        """
        n = points.shape[0]
        ncomp = vel.shape[1]
        for p in prange(n):
            c = guess[p]
            if c < 0:
                out_cells[p] = -1
                out_status[p] = 3
                continue
            prev = -1
            l0 = l1 = l2 = l3 = 0.0
            accepted = False
            boundary = False
            it = 0
            while it < max_iter:
                px = points[p, 0] - dd[c, 0]
                py = points[p, 1] - dd[c, 1]
                pz = points[p, 2] - dd[c, 2]
                l0 = Minv[c, 0, 0]*px + Minv[c, 0, 1]*py + Minv[c, 0, 2]*pz
                l1 = Minv[c, 1, 0]*px + Minv[c, 1, 1]*py + Minv[c, 1, 2]*pz
                l2 = Minv[c, 2, 0]*px + Minv[c, 2, 1]*py + Minv[c, 2, 2]*pz
                l3 = 1.0 - l0 - l1 - l2
                if l0 >= -tol and l1 >= -tol and l2 >= -tol and l3 >= -tol:
                    accepted = True
                    break
                # step across the most-negative face, never back to prev
                minval = -tol
                face = -1
                if l0 < minval and adj[c, 0] != prev:
                    minval = l0; face = 0
                if l1 < minval and adj[c, 1] != prev:
                    minval = l1; face = 1
                if l2 < minval and adj[c, 2] != prev:
                    minval = l2; face = 2
                if l3 < minval and adj[c, 3] != prev:
                    minval = l3; face = 3
                if face == -1:
                    # backtrack-stuck on a face/edge: accept if essentially inside
                    worst = min(l0, l1, l2, l3)
                    accepted = worst >= -slack
                    break
                nb = adj[c, face]
                if nb < 0:
                    boundary = True
                    break
                prev = c
                c = nb
                it += 1
            else:
                # ran out of iterations: recompute weights for the final cell and
                # accept only if the point sits essentially inside it
                px = points[p, 0] - dd[c, 0]
                py = points[p, 1] - dd[c, 1]
                pz = points[p, 2] - dd[c, 2]
                l0 = Minv[c, 0, 0]*px + Minv[c, 0, 1]*py + Minv[c, 0, 2]*pz
                l1 = Minv[c, 1, 0]*px + Minv[c, 1, 1]*py + Minv[c, 1, 2]*pz
                l2 = Minv[c, 2, 0]*px + Minv[c, 2, 1]*py + Minv[c, 2, 2]*pz
                l3 = 1.0 - l0 - l1 - l2
                accepted = min(l0, l1, l2, l3) >= -slack

            if accepted:
                n0 = conn[c, 0]; n1 = conn[c, 1]; n2 = conn[c, 2]; n3 = conn[c, 3]
                for k in range(ncomp):
                    out_v[p, k] = (l0*vel[n0, k] + l1*vel[n1, k]
                                   + l2*vel[n2, k] + l3*vel[n3, k])
                out_cells[p] = c
                out_status[p] = 0
            else:
                out_cells[p] = c
                out_status[p] = 1 if boundary else 2


class _TetSampler:
    """Fast velocity sampler for a static all-tetrahedral mesh.

    pyvista's ``DataSet.sample`` passes the cell locator only as a *prototype*,
    so VTK rebuilds it (~140 ms over ~2M cells) on every call -- and because the
    tracking loop rewrites the velocity array each substep, that rebuild fires
    4x per RK4 step. The locator depends only on geometry, which never changes.

    Cell location uses two strategies:

    * Cold path (``locate``) -- a probe whose source geometry never changes, so
      VTK builds its ``vtkCellTreeLocator`` once and reuses it; the containing
      cell id comes back as a passed-through cell-data array.
    * Temporal-coherence walk (``locate``, with a ``guess``) -- particles move
      far less than a cell per substep, so starting from each particle's previous
      cell and walking across tet faces toward the query point locates it in ~1-2
      vectorized iterations, several times faster than a fresh locator query.
      Particles that leave the domain or fail to converge fall back to the probe.

    Interpolation is a vectorized barycentric weighting using affine transforms
    precomputed once per cell. Falls back to ``ok=False`` for non-tet meshes.
    """

    def __init__(self, mesh):
        self.ok = bool(np.all(np.asarray(mesh.celltypes) == VTK_TETRA))
        if not self.ok:
            return

        # Tet connectivity (n_cells, 4) and node coordinates -- precomputed once.
        self.conn = mesh.cells.reshape(-1, 5)[:, 1:].copy()
        self.node_xyz = np.asarray(mesh.points)

        # Per-cell affine map xyz -> barycentric: l123 = Minv @ (p - d), where d
        # is the 4th vertex. Precomputed once so both the walk and interpolation
        # are matrix-vector products (no per-call linear solve).
        vx = self.node_xyz[self.conn]                       # (nc, 4, 3)
        self._d = np.ascontiguousarray(vx[:, 3, :])         # (nc, 3)
        T = np.stack([vx[:, 0] - self._d, vx[:, 1] - self._d,
                      vx[:, 2] - self._d], axis=2)           # (nc, 3, 3)
        self._Minv = np.ascontiguousarray(np.linalg.inv(T))

        # Tet face adjacency: adj[c, i] is the cell sharing the face opposite
        # local vertex i of cell c (-1 on a domain boundary). Built by matching
        # faces (sorted node triples) that appear in exactly two cells.
        self._adj = self._build_adjacency(self.conn, self.node_xyz.shape[0])
        # Contiguous int64 connectivity for the numba kernel.
        self._conn64 = np.ascontiguousarray(self.conn, dtype=np.int64)

        # Geometry-only source carrying the cell id as cell data. We never mutate
        # it, so its MTime stays fixed and the probe reuses its built locator.
        geom = pv.UnstructuredGrid()
        geom.copy_structure(mesh)
        geom.cell_data["cid"] = np.arange(geom.n_cells, dtype=np.int64)
        self._geom = geom  # keep a reference alive for the probe

        # vtkCellTreeLocator resolves interior-point FindCell ~3.5x faster than
        # vtkStaticCellLocator on this tet mesh (the cold-path cost at scale).
        probe = vtkProbeFilter()
        probe.SetCellLocatorPrototype(vtkCellTreeLocator())
        probe.SetSourceData(geom)
        probe.SetPassCellArrays(True)
        probe.SetPassPointArrays(False)
        probe.SetPassFieldArrays(False)
        self._probe = probe

    @staticmethod
    def _build_adjacency(conn, n_nodes):
        nc = conn.shape[0]
        # face i is opposite local vertex i
        faces = np.stack([conn[:, [1, 2, 3]], conn[:, [0, 2, 3]],
                          conn[:, [0, 1, 3]], conn[:, [0, 1, 2]]], axis=1)
        fs = np.sort(faces, axis=2).reshape(-1, 3)          # (4nc, 3)
        maxn = n_nodes + 1
        key = (fs[:, 0].astype(np.int64) * maxn + fs[:, 1]) * maxn + fs[:, 2]
        cell_id = np.repeat(np.arange(nc), 4)
        local_f = np.tile(np.arange(4), nc)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        # interior faces appear exactly twice -> consecutive after sorting
        same = np.where(ks[:-1] == ks[1:])[0]
        a, b = order[same], order[same + 1]
        adj = np.full((nc, 4), -1, dtype=np.int32)   # cell ids < 2^31
        adj[cell_id[a], local_f[a]] = cell_id[b]
        adj[cell_id[b], local_f[b]] = cell_id[a]
        return adj

    def _bary(self, points_xyz, cells):
        """Barycentric weights (n, 4) of points within their given cells."""
        l123 = np.einsum("nij,nj->ni", self._Minv[cells], points_xyz - self._d[cells])
        return np.concatenate([l123, 1 - l123.sum(1, keepdims=True)], axis=1)

    def _locate_probe(self, points_xyz):
        """Cold-path location: returns (cell id, valid mask) via the reused locator."""
        self._probe.SetInputData(pv.PolyData(np.ascontiguousarray(points_xyz)))
        self._probe.Update()
        out = pv.wrap(self._probe.GetOutput())
        cid = np.asarray(out.point_data["cid"])
        valid = np.asarray(out.point_data["vtkValidPointMask"]).astype(bool)
        return cid, valid

    def locate(self, points_xyz, guess=None, tol=1e-10, max_iter=20):
        """Return the containing cell id per point (-1 if outside the domain).

        With ``guess`` (previous cell per particle), walk across tet faces from
        the guess; otherwise do a full locator query. Particles that exit the
        domain or do not converge fall back to the locator.
        """
        n = points_xyz.shape[0]
        if guess is None:
            cid, valid = self._locate_probe(points_xyz)
            return np.where(valid, cid, -1)

        cells = guess.astype(np.int64, copy=True)
        need_probe = cells < 0           # no usable guess (e.g. reset particles)
        active = ~need_probe
        prev = np.full(n, -1, dtype=np.int64)   # previous cell, to avoid backtracking
        for _ in range(max_iter):
            idx = np.where(active)[0]
            if idx.size == 0:
                break
            w = self._bary(points_xyz[idx], cells[idx])
            inside = (w >= -tol).all(axis=1)
            active[idx[inside]] = False
            out = idx[~inside]
            if out.size:
                # Cross the face opposite the most-negative barycentric coord, but
                # never step straight back to the cell we just came from -- that is
                # the 2-cycle that traps points sitting on a shared face.
                wn = w[~inside]
                back = self._adj[cells[out]] == prev[out][:, None]
                face = np.argmin(np.where(back, np.inf, wn), axis=1)
                nb = self._adj[cells[out], face]
                boundary = nb < 0
                need_probe[out[boundary]] = True   # left domain -> confirm w/ probe
                active[out[boundary]] = False
                mv = out[~boundary]
                prev[mv] = cells[mv]
                cells[mv] = nb[~boundary]
        need_probe[active] = True        # hit max_iter -> confirm w/ probe

        if need_probe.any():
            pidx = np.where(need_probe)[0]
            cid_p, valid_p = self._locate_probe(points_xyz[pidx])
            cells[pidx] = np.where(valid_p, cid_p, -1)
        return cells

    def _interp(self, points_xyz, cells_safe, vel):
        w = self._bary(points_xyz, cells_safe)
        return np.einsum("nij,ni->nj", vel[self.conn[cells_safe]], w)

    def sample(self, points_xyz, vel, guess=None, tol=1e-10, slack=1e-7, max_iter=20):
        """Return (velocity (n,3), valid (n,), cells (n,)) for points in the field.

        ``cells`` is the resolved containing cell per point (-1 if outside),
        suitable to feed back as ``guess`` on the next call for the walk.
        """
        points_xyz = np.ascontiguousarray(points_xyz, dtype=np.float64)

        # Cold path (no guess) or no numba: locate via the probe, interpolate in numpy.
        if guess is None or not _HAVE_NUMBA:
            cells = self.locate(points_xyz, guess=guess, tol=tol, max_iter=max_iter)
            valid = cells >= 0
            v = self._interp(points_xyz, np.where(valid, cells, 0), vel)
            v[~valid] = 0.0
            return v, valid, cells

        # Fast path: fused walk + interpolation in one numba kernel; only the
        # particles it couldn't resolve (boundary/non-converged/no-guess) hit the
        # locator probe.
        vel = np.ascontiguousarray(vel, dtype=np.float64)
        n = points_xyz.shape[0]
        v = np.zeros((n, vel.shape[1]))
        cells = np.empty(n, dtype=np.int64)
        status = np.empty(n, dtype=np.int8)
        _walk_interp_kernel(points_xyz, self._Minv, self._d, self._conn64, self._adj,
                            vel, guess.astype(np.int64), tol, slack, max_iter,
                            v, cells, status)

        need = status != 0
        if need.any():
            pidx = np.where(need)[0]
            cid_p, valid_p = self._locate_probe(points_xyz[pidx])
            vv = self._interp(points_xyz[pidx], np.where(valid_p, cid_p, 0), vel)
            vv[~valid_p] = 0.0
            v[pidx] = vv
            cells[pidx] = np.where(valid_p, cid_p, -1)

        return v, cells >= 0, cells



def seed_mesh(mesh, npoints):
    bounds = mesh.bounds
    return seed_region(mesh, npoints, bounds)

def seed_region(mesh, npoints, bounds, normalization=None):

    vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
    # Aim for 10x less inital points than desired
    npoints_initial = npoints//10
    dx_init = (vol / npoints_initial) ** (1/3)

    x = np.arange(bounds[0], bounds[1], dx_init)
    x = np.insert(x, -1, x[-1]+dx_init)
    y = np.arange(bounds[2], bounds[3], dx_init)
    y = np.insert(y, -1, y[-1]+dx_init)
    z = np.arange(bounds[4], bounds[5], dx_init)
    z = np.insert(z, -1, z[-1]+dx_init)

        
    points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    point_cloud = pv.PolyData(points)

    # Extract surface
    surf = mesh.extract_geometry()

    # Initial guess of points  inside the surface
    inside = point_cloud.select_enclosed_points(surf)["SelectedPoints"]
    # This is like a binary mask of the mesh
    inside_array = inside.reshape(len(z), len(x), len(y))
    # dilate (to account for boundary regions)
    inside_array = scipy.ndimage.binary_dilation(
        inside_array, structure=np.ones((5, 5, 5)), iterations=1
    )
    inside = inside_array.flatten()


    valid = points[np.argwhere(inside)[:, 0], :]

    # # Refine with sample operation
    # valid = valid0[np.argwhere(point_valid.sample(mesh)['vtkValidPointMask'])[:,0],:]
    # point_valid = pv.PolyData(valid)

    # Now we have a good distribution of points inside the surface
    # We can add even more (randomly) and then refine again
    n_subsample = (
        int(np.ceil(npoints / valid.shape[0])) * 2
    )  # Assume about half will be outside
    rand_pts = np.random.rand(valid.shape[0], n_subsample, 3) - 0.5
    # Randomly between -0.5 and 0.5

    subsampled = rand_pts * dx_init + valid[:, np.newaxis, :]
    subsampled = subsampled.reshape(-1, 3)

    # Refine with sample operation
    point_subsampled = pv.PolyData(subsampled)
    subsampled = subsampled[
        np.argwhere(point_subsampled.sample(mesh)["vtkValidPointMask"])[:, 0], :]

    if normalization is not None:
        # Sample absolute velocity for the seeded points, and do stoastic subsampling based on normalization field
        point_subsampled = pv.PolyData(subsampled)
        samp = point_subsampled.sample(mesh)

        if samp[normalization].ndim > 1:
            normag = np.sum(samp[normalization]**2, axis=1)**0.5
            density = normag / np.max(normag)
        else:
            density = samp[normalization] / np.max(samp[normalization])

        # Density is now between 0 and 1, giving the probability of keeping each point
        rand = np.random.rand(density.shape[0])
        keep = rand < density
        subsampled = subsampled[keep, :]

    return subsampled

def _read_vtu(filepath, active_key, pbar):
    """Read a .vtu, loading only point arrays whose name contains active_key.

    The pulsatile files store pressure_NNNNN alongside velocity_NNNNN for every
    timestep; tracking never uses pressure, so skipping it cuts both read time
    (~40%) and peak memory (~25%) -- important when each worker reloads the file.
    """
    reader = pv.get_reader(filepath)
    reader.disable_all_point_arrays()
    for n in reader.point_array_names:
        if active_key in n:
            reader.enable_point_array(n)
    if pbar:
        reader.show_progress()
    return reader.read()


class timeMeshSingleVTU:
    def __init__(self, filepath, active_key="velocity", pbar = False, only_active_key=False):
        self.filepath = filepath
        if only_active_key:
            self.mesh = _read_vtu(filepath, active_key, pbar)
        else:
            self.mesh = pv.read(filepath, progress_bar=pbar)

        # Extract all time steps from the point data keys
        self.active_key = active_key
        self.times = []
        for key in self.mesh.point_data.keys():
            if self.active_key in key:
                self.times.append(int(key[-5:]))

        self.times = np.array(self.times)
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
        return f"{self.active_key}_{self.times[index]:05d}"

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


def _sample_v_fallback(flow, points_xyz, time):
    """Generic (slower) sampler used when the mesh is not all-tetrahedral."""
    samp = flow.sample(pv.PolyData(np.ascontiguousarray(points_xyz)), time)
    valid = np.asarray(samp["vtkValidPointMask"]).astype(bool)
    v = np.asarray(samp[flow.active_key]).copy()
    v[~valid] = 0.0
    return v, valid, None


class timeMeshPVD:
    def __init__(self, filepath, dt=None, active_key="velocity", pbar = True, subsamp = 1):

        self.reader = pv.get_reader(filepath)
        self.times = np.array(self.reader.time_values)
        if subsamp > 1:
            self.times = self.times[::subsamp]
        self.active_key = active_key

        self.meshes = []
        # Preload all meshes
        for t in tqdm(self.times, disable=not pbar):
            self.reader.set_active_time_value(t)
            mesh = self.reader.read()[0]
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


def tracking(flow_mesh, initial_seeds:pv.PolyData, seeding_points:np.ndarray, dt, tmax, method = "RK4", pbar = True, key="velocity", timings=None):
    # Pass a dict as `timings` to collect a wall-time breakdown of the loop.
    # It is filled in place (non-breaking: the 3-tuple return is unchanged).

    nstep = int(tmax/dt)

    # Positions are carried as a plain (n, 3) array; sample_v works on numpy
    # directly, so we avoid wrapping/unwrapping a PolyData every substep.
    r = np.ascontiguousarray(initial_seeds.points, dtype=float).copy()
    n_particles = r.shape[0]

    r_res = np.zeros((nstep, n_particles, 3))
    m_reset_flag = np.zeros((nstep, n_particles))
    n_oob = 0

    # For debugging
    oob_loc_list = []

    # Optional profiling: accumulate time spent in field sampling vs everything else.
    profile = timings is not None
    n_samples = 0
    t_sample = 0.0
    t0_loop = time.perf_counter()

    def _sample(points, t, guess):
        nonlocal t_sample, n_samples
        if profile:
            _t = time.perf_counter()
            out = flow_mesh.sample_v(points, t, guess=guess)
            t_sample += time.perf_counter() - _t
            n_samples += 1
            return out
        return flow_mesh.sample_v(points, t, guess=guess)

    # Per-particle cell guess for the temporal-coherence walk; None on the first
    # step (cold locator) and reset to -1 for recycled particles (forces a probe).
    cells = None

    pbar = tqdm(range(nstep), disable=not pbar)

    for i in pbar:

        # Sample current time and position
        k1, valid, cells = _sample(r, i*dt, cells)

        # Reset OOB points
        oob = ~valid
        m_reset_flag[i,oob] = 1
        # Save oob locations
        oob_loc_list.append(r[oob,:])

        # Get velocity step
        if method == "RK4":
            # Substep positions stay within ~1 cell, so reuse the running cell
            # guess to seed each walk.
            k2, _, c = _sample(k1*dt/2 + r, i*dt + dt/2, cells)
            k3, _, c = _sample(k2*dt/2 + r, i*dt + dt/2, c)
            k4, _, _ = _sample(k3*dt + r, i*dt + dt, c)

            v = (k1 + 2*k2 + 2*k3 + k4)/6
        else:
            v = k1

        # Advect
        r = r + v*dt

        # Move OOB back to inlet
        newpos = seeding_points[np.random.randint(low = 0, high = seeding_points.shape[0], size = (np.sum(oob),)),:]
        r[oob,:] = newpos

        # Recycled particles jumped to the inlet -> their cell guess is stale.
        if cells is not None:
            cells[oob] = -1

        r_res[i,...] = r
        n_oob = np.sum(oob)

        pbar.set_postfix_str(f"n_oob={n_oob}")

    if profile:
        t_total = time.perf_counter() - t0_loop
        timings.update(
            n_particles=n_particles,
            nstep=nstep,
            method=method,
            t_total=t_total,
            t_sample=t_sample,
            t_other=t_total - t_sample,
            n_sample_calls=n_samples,
            sample_frac=(t_sample / t_total) if t_total else 0.0,
            s_per_step=t_total / nstep if nstep else 0.0,
            # Throughput in particle-steps per second (the headline number to beat).
            particle_steps_per_s=(n_particles * nstep / t_total) if t_total else 0.0,
        )

    return r_res, m_reset_flag, oob_loc_list

def tracking_parallel(fn, seeds, inlet, dt, tmax, method = "RK4", active_key="velocity", pbar = False, dt_pvd = None, only_active_key=True):
    # Tracking only ever reads active_key, so skip pressure (etc.) by default to
    # speed up the per-worker reload and cut memory.
    if fn.split(".")[-1] == "vtu":
        flow = timeMeshSingleVTU(fn, active_key=active_key, pbar=pbar, only_active_key=only_active_key)
    elif fn.split(".")[-1] == "pvd":
        flow = timeMeshPVD(fn, active_key=active_key, pbar=pbar, dt=dt_pvd)

    r_res, m_reset_flag, oob_loc_list = tracking(flow, pv.PolyData(seeds), inlet, dt, tmax, method=method, pbar = pbar, key=active_key)

    return r_res, m_reset_flag, oob_loc_list

def batched_particles(particles, batch_size):
    result = []
    # Random shuffle
    p = np.random.permutation(particles)
    for i in range(0, len(p), batch_size):
        result.append(p[i:i + batch_size])
    return result