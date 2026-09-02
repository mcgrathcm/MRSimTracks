"""Moving-wall handling for CMM fixed-mesh FSI flow fields.

Currently supports linear tetrahedral tracking meshes whose wall and cap
surfaces share points with the loaded volume mesh. Caps are treated as fixed
open boundaries.
"""

import os
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from .io import load_flow


def _as_surface(obj):
    if isinstance(obj, (str, bytes, os.PathLike)):
        obj = pv.read(obj)
    return obj.extract_surface(algorithm="dataset_surface").triangulate()


def _as_surfaces(obj):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return [_as_surface(item) for item in obj]
    return [_as_surface(obj)]


def _tet_laplacian(node_xyz, conn):
    """Assemble the P1 tetrahedral FEM stiffness matrix."""
    nc = conn.shape[0]
    n_nodes = node_xyz.shape[0]
    P = node_xyz[conn]
    A = np.concatenate([np.ones((nc, 4, 1)), P], axis=2)
    Ainv = np.linalg.inv(A)
    grad = Ainv[:, 1:, :]
    vol = np.abs(np.linalg.det(A)) / 6.0
    K = vol[:, None, None] * np.einsum("nki,nkj->nij", grad, grad)

    rows = np.broadcast_to(conn[:, :, None], (nc, 4, 4)).reshape(-1)
    cols = np.broadcast_to(conn[:, None, :], (nc, 4, 4)).reshape(-1)
    return coo_matrix((K.reshape(-1), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()


class CMMMeshMotion:
    """Reference-to-physical mesh motion for CMM fixed-mesh FSI fields.

    Args:
        flow (object): Loaded flow field from :func:`mrsimtracks.load_flow`.
            Its mesh must be a linear tetrahedral tracking mesh.
        exterior (str | pathlib.Path | pyvista.PolyData): CLOSED exterior
            surface (walls + caps).
        walls (str | pathlib.Path | pyvista.PolyData): Wall-only surface; nodes
            must be a subset of the volume mesh.
        caps (list): Cap surfaces (required -- the interior displacement solve
            treats them as fixed open boundaries).
        verbose (bool): Print reconstruction/solve diagnostics.
        cache_path (str | pathlib.Path | None): Optional ``.npz`` cache for the
            interior displacement solve.

    Attributes:
        displacement (np.ndarray): ``(n_frames, n_wall, 3)`` reconstructed
            *wall* displacement (float32, mesh units) -- the Dirichlet data fed
            to the interior solve.
        drift_ratio (float): End-of-window residual over peak ``|d|`` before
            de-drift; small (<~0.2) confirms a whole-cycle export.
        peak_displacement (float): Max wall-node ``|d|`` over the cycle.
    """

    def __init__(self, flow, exterior, walls, caps=None, verbose=True, cache_path=None):
        self._flow = flow
        self._times = np.asarray(flow.times_shift_s, dtype=np.float64)
        self._tmax = float(flow.tmax)
        self._remap_guess = None
        self._last_t = None
        self._last_D_t = None

        sampler = flow._sampler
        if getattr(flow, "geometry_mode", "static") != "static":
            raise ValueError("CMMMeshMotion requires a fixed reference mesh")
        if not sampler.ok:
            raise ValueError("CMMMeshMotion requires a linear tetrahedral tracking mesh")

        n_nodes = sampler.node_xyz.shape[0]
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                cached = np.load(cache_path)
                if int(cached["n_nodes"]) == n_nodes:
                    self._load_cache(cached)
                    if verbose:
                        print(f"CMMMeshMotion: loaded cached interior solve from {cache_path}")
                    return
                if verbose:
                    print(f"CMMMeshMotion: cache at {cache_path} is for a different mesh "
                          f"({int(cached['n_nodes'])} nodes vs {n_nodes} now) -- rebuilding")

        cap_surfaces = _as_surfaces(caps)
        if not cap_surfaces:
            raise ValueError(
                "caps is required: the interior displacement solve needs a "
                "fixed (zero-displacement) boundary at the inlet/outlet, or "
                "the harmonic extension is not well-posed")

        mesh_pts = np.asarray(flow.mesh.points)
        tree = cKDTree(mesh_pts)

        walls = _as_surface(walls)
        wall_pts = np.asarray(walls.points, dtype=np.float64)

        d, self._wall_rows = tree.query(wall_pts)
        if d.max() > 1e-9:
            raise ValueError("wall surface nodes are not a subset of the "
                             f"volume mesh (max mismatch {d.max():g})")

        cap_pts = np.concatenate([np.asarray(cap.points) for cap in cap_surfaces])
        d_cap, self._cap_rows = tree.query(cap_pts)
        if d_cap.max() > 1e-9:
            raise ValueError("cap surface nodes are not a subset of the "
                             f"volume mesh (max mismatch {d_cap.max():g})")
        self._build_combined_tree(wall_pts, cap_pts)

        ext = _as_surface(exterior).triangulate()

        d_ext, ext_rows = tree.query(np.asarray(ext.points))
        if d_ext.max() > 1e-9:
            raise ValueError("exterior surface nodes are not a subset of the "
                             f"volume mesh (max mismatch {d_ext.max():g})")
        covered = np.zeros(mesh_pts.shape[0], dtype=bool)
        covered[self._wall_rows] = True
        covered[self._cap_rows] = True
        if not covered[ext_rows].all():
            raise ValueError(
                "some exterior-surface nodes are neither in `walls` nor "
                "`caps` -- the interior displacement solve needs every "
                "boundary node classified as one or the other")

        self.displacement = self._reconstruct(verbose)
        self.peak_displacement = float(
            np.linalg.norm(self.displacement, axis=2).max())
        self._solve_interior(sampler, verbose)

        if cache_path is not None:
            self._save_cache(cache_path, n_nodes)

    def _build_combined_tree(self, wall_pts, cap_pts):
        self._n_wall_pts = wall_pts.shape[0]
        self._combined_tree = cKDTree(np.concatenate([wall_pts, cap_pts]))

    def _load_cache(self, cached):
        """Restore a cached interior displacement solve."""
        self._wall_rows = cached["wall_rows"]
        self._cap_rows = cached["cap_rows"]
        self.displacement = cached["displacement"]
        self.peak_displacement = float(cached["peak_displacement"])
        self.drift_ratio = float(cached["drift_ratio"])
        self._D = cached["D"]

        node_xyz = self._flow._sampler.node_xyz
        wall_pts = node_xyz[self._wall_rows]
        cap_pts = node_xyz[self._cap_rows]
        self._build_combined_tree(wall_pts, cap_pts)

    def _save_cache(self, cache_path, n_nodes):
        """Persist the interior displacement solve."""
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp{os.getpid()}.npz")
        np.savez(
            tmp_path,
            n_nodes=n_nodes,
            D=self._D,
            displacement=self.displacement,
            wall_rows=self._wall_rows,
            cap_rows=self._cap_rows,
            times=self._times,
            tmax=self._tmax,
            peak_displacement=self.peak_displacement,
            drift_ratio=self.drift_ratio,
        )
        tmp_path.replace(cache_path)

    def _reconstruct(self, verbose):
        """Trapezoidal wall-velocity integration + linear de-drift."""
        n_frames = self._times.size
        d = np.zeros((n_frames, self._wall_rows.size, 3))
        v_prev = np.asarray(self._flow._frame_vel(0), float)[self._wall_rows]
        for k in range(1, n_frames):
            v_k = np.asarray(self._flow._frame_vel(k), float)[self._wall_rows]
            d[k] = d[k - 1] + 0.5 * (self._times[k] - self._times[k - 1]) * (v_prev + v_k)
            v_prev = v_k

        # Remove any end-of-cycle residual as a linear drift.
        N = n_frames - 1
        resid = d[N].copy()
        peak = np.linalg.norm(d, axis=2).max()
        self.drift_ratio = float(np.linalg.norm(resid, axis=1).max() / max(peak, 1e-30))
        d -= (np.arange(n_frames) / N)[:, None, None] * resid[None]

        if verbose or self.drift_ratio >= 0.2:
            warn = ("" if self.drift_ratio < 0.2 else
                    " -- WARNING: large residual; export may not be a whole cycle")
            print(f"CMMMeshMotion: wall displacement over {n_frames} frames, peak |d| "
                  f"{np.linalg.norm(d, axis=2).max():.5g}, drift "
                  f"{self.drift_ratio:.2%}{warn}")
        return d.astype(np.float32)

    def _solve_interior(self, sampler, verbose):
        """Harmonic extension of the wall displacement into the interior, zero
        at the caps: one FEM Laplacian factorization, reused across frames."""
        n_nodes = sampler.node_xyz.shape[0]
        is_boundary = np.zeros(n_nodes, dtype=bool)
        is_boundary[self._wall_rows] = True
        is_boundary[self._cap_rows] = True
        interior_rows = np.where(~is_boundary)[0]
        boundary_rows = np.where(is_boundary)[0]
        wall_local = np.searchsorted(boundary_rows, self._wall_rows)

        L = _tet_laplacian(sampler.node_xyz, sampler.conn)
        L_rows = L[interior_rows].tocsc()
        L_II = L_rows[:, interior_rows]
        L_IB = L_rows[:, boundary_rows]
        lu = splu(L_II.tocsc())

        n_frames = self._times.size
        self._D = np.zeros((n_frames, n_nodes, 3), dtype=np.float32)
        d_b = np.zeros((boundary_rows.size, 3))
        for k in range(n_frames):
            d_b[wall_local] = self.displacement[k]
            d_full = np.empty((n_nodes, 3))
            d_full[interior_rows] = lu.solve(-(L_IB @ d_b))
            d_full[boundary_rows] = d_b
            self._D[k] = d_full

        if verbose:
            print(f"CMMMeshMotion: interior displacement solved over {n_nodes} nodes "
                  f"({interior_rows.size} interior unknowns), {n_frames} frames")

    def _weights(self, t):
        """Periodic linear-interpolation frame indices/weight for time ``t``."""
        tw = t % self._tmax
        inext = int(np.searchsorted(self._times, tw, side="right"))
        inext = min(max(inext, 1), self._times.size - 1)
        s = float((tw - self._times[inext - 1]) / (self._times[inext] - self._times[inext - 1]))
        return inext - 1, inext, s

    def displacement_at(self, t):
        """Interpolated volume-mesh displacement at time ``t``."""
        if t == self._last_t and self._last_D_t is not None:
            return self._last_D_t
        i0, i1, s = self._weights(t)
        self._last_t = t
        self._last_D_t = np.ascontiguousarray((1.0 - s) * self._D[i0] + s * self._D[i1])
        return self._last_D_t

    def remap(self, points, t):
        """Map physical particle positions back onto the reference mesh."""
        points = np.asarray(points, dtype=np.float64)
        combined_idx = self._combined_tree.query(points, workers=-1)[1]
        near_cap = combined_idx >= self._n_wall_pts
        j = np.where(near_cap, 0, combined_idx)
        D_t = self.displacement_at(t)

        X = np.where(near_cap[:, None], points, points - D_t[self._wall_rows[j]])

        sampler = self._flow._sampler
        guess = self._remap_guess
        if guess is not None and guess.shape[0] != points.shape[0]:
            guess = None
        for _ in range(2):
            Dx, valid, guess = sampler.sample(X, D_t, guess=guess)
            update = valid & ~near_cap
            X = np.where(update[:, None], points - Dx, X)
        self._remap_guess = guess
        return X


class CMMFlow:
    """Fixed-mesh CMM flow with moving-wall remapping.

    ``sample_v`` accepts physical particle coordinates and samples the wrapped
    fixed-mesh flow at the corresponding reference coordinates.
    """

    def __init__(self, flow, mesh_motion):
        self.base_flow = flow
        self.mesh_motion = mesh_motion
        self.active_key = flow.active_key
        self.dtype = flow.dtype
        self.times = flow.times
        self.times_shift_s = flow.times_shift_s
        self.tmax = flow.tmax
        self.fields = flow.fields
        self.geometry_mode = "cmm"
        points = np.asarray(flow.mesh.points)
        deformed = points[None, :, :] + mesh_motion._D
        lower = deformed.min(axis=(0, 1))
        upper = deformed.max(axis=(0, 1))
        self.bounds = tuple(np.column_stack((lower, upper)).ravel())
        self._sampler = flow._sampler
        self.locator = getattr(flow, "locator", None)
        self.active_mesh = self.get_mesh(0.0)
        self.mesh = self.active_mesh

    def _frame_vel(self, frame):
        return self.base_flow._frame_vel(frame)

    def _frame_runtime(self, frame):
        return self.base_flow._frame_runtime(frame)

    def set_active_time(self, time):
        if hasattr(self.base_flow, "set_active_time"):
            self.base_flow.set_active_time(time)
            self.active_mesh = self.get_mesh(time)
            self.mesh = self.active_mesh
            self._sampler = self.base_flow._sampler
            self.locator = getattr(self.base_flow, "locator", None)

    def get_mesh(self, time):
        """Return the deformed mesh at ``time``."""
        mesh = self.base_flow.get_mesh(time)
        displacement = self.mesh_motion.displacement_at(time)
        mesh.points = np.ascontiguousarray(np.asarray(mesh.points) + displacement)
        return mesh

    def sample_v(self, points_xyz, time, guess=None):
        sample_xyz = self.mesh_motion.remap(points_xyz, time)
        return self.base_flow.sample_v(sample_xyz, time, guess=guess)

    def sample(self, points, time):
        velocity, valid, _ = self.sample_v(np.asarray(points.points), time)
        output = points.copy()
        output.point_data[self.active_key] = velocity
        output.point_data["vtkValidPointMask"] = valid.astype(np.uint8)
        return output


def load_cmm_flow(path, exterior, walls, caps=None, *, active_key="velocity",
                   subsamp=1, only_active_key=True, pbar=False, dt=None,
                   precision="f64", time_interp="linear", conform_mesh=True,
                   cache_path=None, verbose=True) -> CMMFlow:
    """Load a CMM flow field with moving-wall remapping.

    Args:
        path: Flow file path ending in ``.vtu`` or ``.pvd``.
        exterior: Closed exterior surface.
        walls: Wall-only surface.
        caps: Cap surfaces treated as fixed open boundaries.
        active_key: Velocity array prefix/name.
        subsamp: Keep every Nth frame for ``.pvd`` inputs.
        only_active_key: For ``.vtu`` files, skip unrelated point-data arrays.
        pbar: Show reader/solve progress.
        dt: Optional timestep scale for ``.pvd`` time values.
        precision: Working precision for the sampling/advection math,
            ``"f64"`` (default) or ``"f32"``.
        time_interp: Temporal interpolation between stored frames,
            ``"linear"`` (default) or ``"cubic"``.
        conform_mesh: Condition the mesh to clean all-tetrahedral at load.
        cache_path: Reuse/write the displacement solve cache.
        verbose: Print reconstruction/solve diagnostics.

    Returns:
        CMMFlow: Fixed reference mesh flow with moving-wall remapping.
    """
    flow = load_flow(path, active_key=active_key, subsamp=subsamp,
                     only_active_key=only_active_key, pbar=pbar, dt=dt,
                     precision=precision, time_interp=time_interp,
                     conform_mesh=conform_mesh)
    mesh_motion = CMMMeshMotion(flow, exterior=exterior, walls=walls, caps=caps,
                                verbose=verbose, cache_path=cache_path)
    return CMMFlow(flow, mesh_motion)
