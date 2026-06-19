import numpy as np
import pyvista as pv

from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator
from vtkmodules.vtkFiltersCore import vtkProbeFilter

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:                       # numba is optional; numpy path still works
    _HAVE_NUMBA = False

VTK_TETRA = 10

# Single (f32) vs double (f64) precision for the sampling/advection math. f32
# roughly halves the memory bandwidth of the velocity field and per-cell affine
# maps -- the loop's dominant cost -- at the price of a looser geometric
# tolerance. Geometry-only precompute (the matrix inverse) stays in f64.
_FLOAT_DTYPES = {
    "f32": np.dtype(np.float32), "float32": np.dtype(np.float32),
    "single": np.dtype(np.float32),
    "f64": np.dtype(np.float64), "float64": np.dtype(np.float64),
    "double": np.dtype(np.float64),
}

# Walk/inside-test tolerances scaled to each dtype's machine epsilon: f64
# barycentric coords are good to ~1e-15, f32 only to ~1e-7, so f32 needs a much
# looser band to count points sitting on a shared face as "inside" (otherwise
# they spuriously fall through to the locator probe every step).
_WALK_TOL = {np.dtype(np.float64): 1e-10, np.dtype(np.float32): 1e-5}
_WALK_SLACK = {np.dtype(np.float64): 1e-7, np.dtype(np.float32): 1e-4}


def resolve_float_dtype(precision):
    """Map a precision spec to a numpy float dtype (``np.float32``/``np.float64``).

    Accepts ``"f32"``/``"f64"`` (and ``float32``/``single``/``float64``/``double``
    aliases) or any numpy float32/float64 dtype-like.
    """
    if isinstance(precision, str):
        try:
            return _FLOAT_DTYPES[precision.lower()]
        except KeyError:
            raise ValueError(
                f"precision must be one of {sorted(_FLOAT_DTYPES)}, got {precision!r}"
            ) from None
    dt = np.dtype(precision)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"precision must be float32 or float64, got {precision!r}")
    return dt


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

    def __init__(self, mesh, dtype=np.float64):
        # Working precision for the sampling/advection math (points, velocity,
        # per-cell affine maps). Geometry precompute below stays in f64.
        self.dtype = np.dtype(dtype)
        self.tol = _WALK_TOL[self.dtype]
        self.slack = _WALK_SLACK[self.dtype]

        self.ok = bool(np.all(np.asarray(mesh.celltypes) == VTK_TETRA))
        if not self.ok:
            return

        # Tet connectivity (n_cells, 4) and node coordinates -- precomputed once.
        self.conn = mesh.cells.reshape(-1, 5)[:, 1:].copy()
        self.node_xyz = np.asarray(mesh.points, dtype=np.float64)

        # Per-cell affine map xyz -> barycentric: l123 = Minv @ (p - d), where d
        # is the 4th vertex. Precomputed once so both the walk and interpolation
        # are matrix-vector products (no per-call linear solve). The inverse is
        # taken in f64 for conditioning, then stored at the working precision.
        vx = self.node_xyz[self.conn]                       # (nc, 4, 3)
        d = np.ascontiguousarray(vx[:, 3, :])               # (nc, 3)
        T = np.stack([vx[:, 0] - d, vx[:, 1] - d,
                      vx[:, 2] - d], axis=2)                 # (nc, 3, 3)
        self._d = np.ascontiguousarray(d, dtype=self.dtype)
        self._Minv = np.ascontiguousarray(np.linalg.inv(T), dtype=self.dtype)

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

    def locate(self, points_xyz, guess=None, tol=None, max_iter=20):
        """Return the containing cell id per point (-1 if outside the domain).

        With ``guess`` (previous cell per particle), walk across tet faces from
        the guess; otherwise do a full locator query. Particles that exit the
        domain or do not converge fall back to the locator.
        """
        if tol is None:
            tol = self.tol
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

    def sample(self, points_xyz, vel, guess=None, tol=None, slack=None, max_iter=20):
        """Return (velocity (n,3), valid (n,), cells (n,)) for points in the field.

        ``cells`` is the resolved containing cell per point (-1 if outside),
        suitable to feed back as ``guess`` on the next call for the walk.
        """
        if tol is None:
            tol = self.tol
        if slack is None:
            slack = self.slack
        points_xyz = np.ascontiguousarray(points_xyz, dtype=self.dtype)
        vel = np.ascontiguousarray(vel, dtype=self.dtype)

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
        n = points_xyz.shape[0]
        v = np.zeros((n, vel.shape[1]), dtype=self.dtype)
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






def _sample_v_fallback(flow, points_xyz, time):
    """Generic (slower) sampler used when the mesh is not all-tetrahedral."""
    samp = flow.sample(pv.PolyData(np.ascontiguousarray(points_xyz)), time)
    valid = np.asarray(samp["vtkValidPointMask"]).astype(bool)
    v = np.asarray(samp[flow.active_key]).copy()
    v[~valid] = 0.0
    return v, valid, None
