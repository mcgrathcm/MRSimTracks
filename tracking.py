import time

import numpy as np
import pyvista as pv
import scipy

from copy import deepcopy
from tqdm.auto import tqdm

from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator, vtkCellTreeLocator
from vtkmodules.vtkFiltersCore import vtkProbeFilter

VTK_TETRA = 10


class _TetSampler:
    """Fast velocity sampler for a static all-tetrahedral mesh.

    pyvista's ``DataSet.sample`` passes the cell locator only as a *prototype*,
    so VTK rebuilds it (~140 ms over ~2M cells) on every call -- and because the
    tracking loop rewrites the velocity array each substep, that rebuild fires
    4x per RK4 step. The locator depends only on geometry, which never changes.

    This sampler builds the locator once against a fixed geometry source (whose
    MTime never changes, so VTK reuses the locator) and recovers the containing
    cell id per query point via a passed-through cell-data array. Interpolation
    of the *current* velocity is then a vectorized barycentric weighting in numpy.
    Falls back to ``None`` (caller uses the generic path) for non-tet meshes.
    """

    def __init__(self, mesh):
        self.ok = bool(np.all(np.asarray(mesh.celltypes) == VTK_TETRA))
        if not self.ok:
            return

        # Tet connectivity (n_cells, 4) and node coordinates -- precomputed once.
        self.conn = mesh.cells.reshape(-1, 5)[:, 1:].copy()
        self.node_xyz = np.asarray(mesh.points)

        # Geometry-only source carrying the cell id as cell data. We never mutate
        # it, so its MTime stays fixed and the probe reuses its built locator.
        geom = pv.UnstructuredGrid()
        geom.copy_structure(mesh)
        geom.cell_data["cid"] = np.arange(geom.n_cells, dtype=np.int64)
        self._geom = geom  # keep a reference alive for the probe

        # vtkCellTreeLocator resolves interior-point FindCell ~3.5x faster than
        # vtkStaticCellLocator on this tet mesh (the dominant cost at scale).
        probe = vtkProbeFilter()
        probe.SetCellLocatorPrototype(vtkCellTreeLocator())
        probe.SetSourceData(geom)
        probe.SetPassCellArrays(True)
        probe.SetPassPointArrays(False)
        probe.SetPassFieldArrays(False)
        self._probe = probe

    def sample(self, points_xyz, vel):
        """Return (velocity (n,3), valid mask (n,)) for points in the current field."""
        points_xyz = np.ascontiguousarray(points_xyz)
        pd = pv.PolyData(points_xyz)
        self._probe.SetInputData(pd)
        self._probe.Update()
        out = pv.wrap(self._probe.GetOutput())

        cid = np.asarray(out.point_data["cid"])
        valid = np.asarray(out.point_data["vtkValidPointMask"]).astype(bool)
        cid_safe = np.where(valid, cid, 0)  # invalid -> dummy cell, zeroed below

        nodes = self.conn[cid_safe]          # (n, 4) node ids of containing tet
        vx = self.node_xyz[nodes]            # (n, 4, 3) tet vertex coords

        # Barycentric coords: solve [a-d, b-d, c-d] @ [l1,l2,l3] = p - d.
        d = vx[:, 3, :]
        T = np.stack([vx[:, 0] - d, vx[:, 1] - d, vx[:, 2] - d], axis=2)
        l123 = np.linalg.solve(T, (points_xyz - d)[..., None])[..., 0]
        w = np.concatenate([l123, 1 - l123.sum(1, keepdims=True)], axis=1)

        v = np.einsum("nij,ni->nj", vel[nodes], w)
        v[~valid] = 0.0
        return v, valid



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

class timeMeshSingleVTU:
    def __init__(self, filepath, active_key="velocity", pbar = False):
        self.filepath = filepath
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

    def sample_v(self, points_xyz, time):
        """Fast path: return (velocity (n,3), valid (n,)) numpy arrays."""
        self.set_active_time(time)
        if not self._sampler.ok:
            return _sample_v_fallback(self, points_xyz, time)
        vel = np.asarray(self.active_mesh.point_data[self.active_key])
        return self._sampler.sample(points_xyz, vel)


def _sample_v_fallback(flow, points_xyz, time):
    """Generic (slower) sampler used when the mesh is not all-tetrahedral."""
    samp = flow.sample(pv.PolyData(np.ascontiguousarray(points_xyz)), time)
    valid = np.asarray(samp["vtkValidPointMask"]).astype(bool)
    v = np.asarray(samp[flow.active_key]).copy()
    v[~valid] = 0.0
    return v, valid


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

    def sample_v(self, points_xyz, time):
        """Fast path: return (velocity (n,3), valid (n,)) numpy arrays."""
        self.set_active_time(time)
        if not self._sampler.ok:
            return _sample_v_fallback(self, points_xyz, time)
        vel = np.asarray(self.active_mesh.point_data[self.active_key])
        return self._sampler.sample(points_xyz, vel)


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

    def _sample(points, t):
        nonlocal t_sample, n_samples
        if profile:
            _t = time.perf_counter()
            out = flow_mesh.sample_v(points, t)
            t_sample += time.perf_counter() - _t
            n_samples += 1
            return out
        return flow_mesh.sample_v(points, t)

    pbar = tqdm(range(nstep), disable=not pbar)

    for i in pbar:

        # Sample current time and position
        k1, valid = _sample(r, i*dt)

        # Reset OOB points
        oob = ~valid
        m_reset_flag[i,oob] = 1
        # Save oob locations
        oob_loc_list.append(r[oob,:])

        # Get velocity step
        if method == "RK4":

            k2 = _sample(k1*dt/2 + r, i*dt + dt/2)[0]
            k3 = _sample(k2*dt/2 + r, i*dt + dt/2)[0]
            k4 = _sample(k3*dt + r, i*dt + dt)[0]

            v = (k1 + 2*k2 + 2*k3 + k4)/6
        else:
            v = k1

        # Advect
        r = r + v*dt

        # Move OOB back to inlet
        newpos = seeding_points[np.random.randint(low = 0, high = seeding_points.shape[0], size = (np.sum(oob),)),:]
        r[oob,:] = newpos

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

def tracking_parallel(fn, seeds, inlet, dt, tmax, method = "RK4", active_key="velocity", pbar = False, dt_pvd = None):
    if fn.split(".")[-1] == "vtu":
        flow = timeMeshSingleVTU(fn, active_key=active_key, pbar=pbar)
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