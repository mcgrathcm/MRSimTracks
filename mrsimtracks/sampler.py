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

# vtkProbeFilter's computed tolerance is 1e-3 of the candidate cell length.
# Use the same geometry-scaled band for dynamic ALE walks, where a fixed
# barycentric tolerance is strongly cell-shape dependent. Static walks keep the
# existing dtype-scaled tolerances and locator fallback.
_ALE_CELL_TOL_FACTOR = 1e-3


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
    @njit(inline="always")
    def _bary_coords(point, cell, Minv, dd, node_xyz, conn, dynamic):
        if not dynamic:
            px = point[0] - dd[cell, 0]
            py = point[1] - dd[cell, 1]
            pz = point[2] - dd[cell, 2]
            l0 = (Minv[cell, 0, 0]*px + Minv[cell, 0, 1]*py
                  + Minv[cell, 0, 2]*pz)
            l1 = (Minv[cell, 1, 0]*px + Minv[cell, 1, 1]*py
                  + Minv[cell, 1, 2]*pz)
            l2 = (Minv[cell, 2, 0]*px + Minv[cell, 2, 1]*py
                  + Minv[cell, 2, 2]*pz)
            return l0, l1, l2, 1.0 - l0 - l1 - l2

        n0 = conn[cell, 0]; n1 = conn[cell, 1]
        n2 = conn[cell, 2]; n3 = conn[cell, 3]
        dx = node_xyz[n3, 0]; dy = node_xyz[n3, 1]; dz = node_xyz[n3, 2]
        ax = node_xyz[n0, 0] - dx
        ay = node_xyz[n0, 1] - dy
        az = node_xyz[n0, 2] - dz
        bx = node_xyz[n1, 0] - dx
        by = node_xyz[n1, 1] - dy
        bz = node_xyz[n1, 2] - dz
        cx = node_xyz[n2, 0] - dx
        cy = node_xyz[n2, 1] - dy
        cz = node_xyz[n2, 2] - dz
        qx = point[0] - dx; qy = point[1] - dy; qz = point[2] - dz

        bxcx = by*cz - bz*cy
        bxcy = bz*cx - bx*cz
        bxcz = bx*cy - by*cx
        det = ax*bxcx + ay*bxcy + az*bxcz
        scale = max(abs(ax), abs(ay), abs(az), abs(bx), abs(by), abs(bz),
                    abs(cx), abs(cy), abs(cz))
        if abs(det) <= 32*np.finfo(np.float64).eps*scale*scale*scale:
            return np.nan, np.nan, np.nan, np.nan

        l0 = (qx*bxcx + qy*bxcy + qz*bxcz) / det
        l1 = (ax*(qy*cz - qz*cy) + ay*(qz*cx - qx*cz)
              + az*(qx*cy - qy*cx)) / det
        l2 = (ax*(by*qz - bz*qy) + ay*(bz*qx - bx*qz)
              + az*(bx*qy - by*qx)) / det
        return l0, l1, l2, 1.0 - l0 - l1 - l2


    @njit(inline="always")
    def _point_triangle_distance2(p, a, b, c):
        """Squared distance from a point to a triangle."""
        ab0 = b[0] - a[0]; ab1 = b[1] - a[1]; ab2 = b[2] - a[2]
        ac0 = c[0] - a[0]; ac1 = c[1] - a[1]; ac2 = c[2] - a[2]
        ap0 = p[0] - a[0]; ap1 = p[1] - a[1]; ap2 = p[2] - a[2]
        d1 = ab0*ap0 + ab1*ap1 + ab2*ap2
        d2 = ac0*ap0 + ac1*ap1 + ac2*ap2
        if d1 <= 0.0 and d2 <= 0.0:
            return ap0*ap0 + ap1*ap1 + ap2*ap2

        bp0 = p[0] - b[0]; bp1 = p[1] - b[1]; bp2 = p[2] - b[2]
        d3 = ab0*bp0 + ab1*bp1 + ab2*bp2
        d4 = ac0*bp0 + ac1*bp1 + ac2*bp2
        if d3 >= 0.0 and d4 <= d3:
            return bp0*bp0 + bp1*bp1 + bp2*bp2

        vc = d1*d4 - d3*d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            q0 = ap0 - v*ab0; q1 = ap1 - v*ab1; q2 = ap2 - v*ab2
            return q0*q0 + q1*q1 + q2*q2

        cp0 = p[0] - c[0]; cp1 = p[1] - c[1]; cp2 = p[2] - c[2]
        d5 = ab0*cp0 + ab1*cp1 + ab2*cp2
        d6 = ac0*cp0 + ac1*cp1 + ac2*cp2
        if d6 >= 0.0 and d5 <= d6:
            return cp0*cp0 + cp1*cp1 + cp2*cp2

        vb = d5*d2 - d1*d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            q0 = ap0 - w*ac0; q1 = ap1 - w*ac1; q2 = ap2 - w*ac2
            return q0*q0 + q1*q1 + q2*q2

        va = d3*d6 - d5*d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            bc0 = c[0] - b[0]; bc1 = c[1] - b[1]; bc2 = c[2] - b[2]
            q0 = bp0 - w*bc0; q1 = bp1 - w*bc1; q2 = bp2 - w*bc2
            return q0*q0 + q1*q1 + q2*q2

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        q0 = ap0 - v*ab0 - w*ac0
        q1 = ap1 - v*ab1 - w*ac1
        q2 = ap2 - v*ab2 - w*ac2
        return q0*q0 + q1*q1 + q2*q2


    @njit(inline="always")
    def _within_ale_cell_tolerance(point, cell, node_xyz, conn):
        """Whether a point is within VTK's cell-length-scaled boundary band."""
        n0 = conn[cell, 0]; n1 = conn[cell, 1]
        n2 = conn[cell, 2]; n3 = conn[cell, 3]
        x0 = node_xyz[n0]; x1 = node_xyz[n1]
        x2 = node_xyz[n2]; x3 = node_xyz[n3]

        distance2 = min(
            _point_triangle_distance2(point, x1, x2, x3),
            _point_triangle_distance2(point, x0, x2, x3),
            _point_triangle_distance2(point, x0, x1, x3),
            _point_triangle_distance2(point, x0, x1, x2),
        )
        xmin = min(x0[0], x1[0], x2[0], x3[0])
        xmax = max(x0[0], x1[0], x2[0], x3[0])
        ymin = min(x0[1], x1[1], x2[1], x3[1])
        ymax = max(x0[1], x1[1], x2[1], x3[1])
        zmin = min(x0[2], x1[2], x2[2], x3[2])
        zmax = max(x0[2], x1[2], x2[2], x3[2])
        dx = xmax - xmin; dy = ymax - ymin; dz = zmax - zmin
        cell_length2 = dx*dx + dy*dy + dz*dz
        return distance2 <= (_ALE_CELL_TOL_FACTOR * _ALE_CELL_TOL_FACTOR
                             * cell_length2)


    @njit(parallel=True, cache=True)
    def _walk_interp_kernel(points, Minv, dd, node_xyz, conn, adj, vel, guess,
                            dynamic,
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
                l0, l1, l2, l3 = _bary_coords(
                    points[p], c, Minv, dd, node_xyz, conn, dynamic
                )
                if l0 >= -tol and l1 >= -tol and l2 >= -tol and l3 >= -tol:
                    accepted = True
                    break
                if dynamic and _within_ale_cell_tolerance(
                        points[p], c, node_xyz, conn):
                    accepted = True
                    break
                # step across the most-negative face, never back to prev
                minval = -tol
                face = -1
                if l0 < minval and (prev < 0 or adj[c, 0] != prev):
                    minval = l0; face = 0
                if l1 < minval and (prev < 0 or adj[c, 1] != prev):
                    minval = l1; face = 1
                if l2 < minval and (prev < 0 or adj[c, 2] != prev):
                    minval = l2; face = 2
                if l3 < minval and (prev < 0 or adj[c, 3] != prev):
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
                l0, l1, l2, l3 = _bary_coords(
                    points[p], c, Minv, dd, node_xyz, conn, dynamic
                )
                accepted = (min(l0, l1, l2, l3) >= -slack
                            if not dynamic else _within_ale_cell_tolerance(
                                points[p], c, node_xyz, conn))

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


def _tet_volumes(node_xyz, conn):
    """Per-cell volume of tets given (n,3) node coords and (nc,4) connectivity.

    Uses the scalar triple product directly -- far faster than ``np.linalg.det``
    over a million 3x3 matrices (no per-matrix LAPACK overhead).
    """
    p = node_xyz[conn]
    e0, e1, e2 = p[:, 0] - p[:, 3], p[:, 1] - p[:, 3], p[:, 2] - p[:, 3]
    return np.abs(np.einsum("ci,ci->c", e0, np.cross(e1, e2))) / 6.0


# A tet is "degenerate" when its volume is this far below the median cell volume
# -- effectively zero (coplanar/duplicate nodes). Such cells hold no interior, so
# no particle is ever inside one; they only break the affine precompute.
_DEGENERATE_VOL_FRAC = 1e-8


def _condition_mesh(mesh, verbose=True):
    """Return a clean all-tetrahedral mesh, or the input unchanged if already so.

    Splits any non-tet cells (e.g. boundary-layer wedges/prisms) into tets and
    drops near-zero-volume slivers that would otherwise make the per-cell affine
    map singular. Points and point-data are preserved (so node-indexed velocity
    fields stay aligned), and an already-clean all-tet mesh is returned untouched.
    """
    celltypes = np.asarray(mesh.celltypes)
    n_nontet = int(np.count_nonzero(celltypes != VTK_TETRA))
    work = mesh.triangulate() if n_nontet else mesh   # wedges/prisms -> tets

    conn = work.cells.reshape(-1, 5)[:, 1:]           # all-tet after triangulate
    node = np.asarray(work.points, dtype=np.float64)
    vol = _tet_volumes(node, conn)
    pos = vol[vol > 0]
    med = np.median(pos) if pos.size else 1.0
    good = vol > _DEGENERATE_VOL_FRAC * med
    n_degen = int(good.size - np.count_nonzero(good))

    if n_nontet == 0 and n_degen == 0:
        return mesh                                   # already clean -> no-op

    kept = conn[good]
    cells = np.empty((kept.shape[0], 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = kept
    out = pv.UnstructuredGrid(cells.ravel(),
                              np.full(kept.shape[0], VTK_TETRA, np.uint8),
                              np.asarray(work.points))
    out.point_data.update(work.point_data)            # node-aligned arrays preserved

    if verbose:
        parts = []
        if n_nontet:
            parts.append(f"split {n_nontet} non-tetrahedral cell(s) into tets")
        if n_degen:
            parts.append(f"dropped {n_degen} degenerate (near-zero-volume) cell(s)")
        print(f"[mrsimtracks] mesh conditioning: {'; '.join(parts)} "
              f"({mesh.n_cells} -> {out.n_cells} cells)")
    return out


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
      vectorized iterations, several times faster than a fresh locator query. A
      static walk falls back to the probe when unresolved; an ALE walk directly
      classifies boundary exits and only probes a genuine convergence failure.

    Static meshes precompute affine transforms for every cell. Dynamic ALE meshes
    share topology and compute barycentric coordinates only for visited cells.
    Falls back to ``ok=False`` for non-tet meshes.
    """

    def __init__(self, mesh, dtype=np.float64, *, dynamic=False, topology=None):
        # Working precision for sampling/advection. Geometry calculations stay
        # in f64 before interpolation into the requested output precision.
        self.dtype = np.dtype(dtype)
        self.tol = _WALK_TOL[self.dtype]
        self.slack = _WALK_SLACK[self.dtype]
        self.dynamic = bool(dynamic)

        self.ok = bool(np.all(np.asarray(mesh.celltypes) == VTK_TETRA))
        if not self.ok:
            return

        # Connectivity and face adjacency depend only on topology. ALE samplers
        # share them across all deformed time states.
        if topology is None:
            self.conn = mesh.cells.reshape(-1, 5)[:, 1:].copy()
        else:
            if (not topology.ok or mesh.n_cells != topology.conn.shape[0]
                    or mesh.n_points != topology.node_xyz.shape[0]):
                raise ValueError("shared sampler topology does not match mesh")
            self.conn = topology.conn
        self.node_xyz = np.asarray(mesh.points, dtype=np.float64)

        # Static meshes precompute the affine map xyz -> barycentric. Dynamic ALE
        # meshes instead evaluate barycentrics only in cells visited by a walk.
        nc = self.conn.shape[0]
        if self.dynamic:
            self._d = np.empty((0, 3), dtype=self.dtype)
            self._Minv = np.empty((0, 3, 3), dtype=self.dtype)
            self._degenerate = np.zeros(nc, dtype=bool)
        else:
            vx = self.node_xyz[self.conn]                   # (nc, 4, 3)
            d = np.ascontiguousarray(vx[:, 3, :])           # (nc, 3)
            T = np.stack([vx[:, 0] - d, vx[:, 1] - d,
                          vx[:, 2] - d], axis=2)             # (nc, 3, 3)
            self._d = np.ascontiguousarray(d, dtype=self.dtype)

            # A degenerate tet makes the batch inverse singular. On failure,
            # exclude those cells from the static fast path and use the probe.
            try:
                Minv = np.linalg.inv(T)
                self._degenerate = np.zeros(nc, dtype=bool)
            except np.linalg.LinAlgError:
                vol = _tet_volumes(self.node_xyz, self.conn)
                pos = vol[vol > 0]
                med = np.median(pos) if pos.size else 1.0
                self._degenerate = vol <= _DEGENERATE_VOL_FRAC * med
                n_degen = int(np.count_nonzero(self._degenerate))
                print(f"[mrsimtracks] _TetSampler: caught {n_degen} degenerate "
                      f"(near-zero-volume) cell(s); excluded from the fast walk "
                      f"(resolved via the locator probe). Consider mesh conditioning.")
                T = T.copy()
                T[self._degenerate] = np.eye(3)             # avoid singular inverse
                Minv = np.linalg.inv(T)
                Minv[self._degenerate] = np.nan             # never accepted by the walk
            self._Minv = np.ascontiguousarray(Minv, dtype=self.dtype)

        # Tet face adjacency: adj[c, i] is the cell sharing the face opposite
        # local vertex i of cell c (-1 on a domain boundary). Built by matching
        # faces (sorted node triples) that appear in exactly two cells.
        self._adj = (self._build_adjacency(self.conn, self.node_xyz.shape[0])
                     if topology is None else topology._adj)
        if self._degenerate.any():
            # Treat a step into a degenerate cell as a domain boundary so the
            # walk falls back to the probe instead of reading a NaN affine map.
            nbr = self._adj.copy()
            into_degen = (nbr >= 0) & self._degenerate[np.where(nbr >= 0, nbr, 0)]
            nbr[into_degen] = -1
            self._adj = nbr
        # Contiguous int64 connectivity for the numba kernel.
        self._conn64 = (np.ascontiguousarray(self.conn, dtype=np.int64)
                        if topology is None else topology._conn64)

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
        if self.dynamic:
            vertices = self.node_xyz[self.conn[cells]]
            d = vertices[:, 3]
            e0 = vertices[:, 0] - d
            e1 = vertices[:, 1] - d
            e2 = vertices[:, 2] - d
            q = points_xyz - d
            cross12 = np.cross(e1, e2)
            det = np.einsum("ni,ni->n", e0, cross12)
            with np.errstate(divide="ignore", invalid="ignore"):
                l0 = np.einsum("ni,ni->n", q, cross12) / det
                l1 = np.einsum("ni,ni->n", e0, np.cross(q, e2)) / det
                l2 = np.einsum("ni,ni->n", e0, np.cross(e1, q)) / det
            l123 = np.column_stack((l0, l1, l2))
            return np.column_stack((l123, 1 - l123.sum(axis=1)))
        l123 = np.einsum("nij,nj->ni", self._Minv[cells], points_xyz - self._d[cells])
        return np.concatenate([l123, 1 - l123.sum(1, keepdims=True)], axis=1)

    def _locate_probe(self, points_xyz):
        """Cold-path location: returns (cell id, valid mask) via the reused locator."""
        self._probe.SetInputData(pv.PolyData(np.ascontiguousarray(points_xyz)))
        self._probe.Update()
        out = pv.wrap(self._probe.GetOutput())
        cid = np.asarray(out.point_data["cid"])
        valid = np.asarray(out.point_data["vtkValidPointMask"]).astype(bool)
        if self.dynamic and valid.any():
            idx = np.where(valid)[0]
            valid[idx] &= np.isfinite(self._bary(points_xyz[idx], cid[idx])).all(axis=1)
        if self._degenerate.any():
            # A point landing exactly on a zero-volume sliver must not resolve to
            # it (its affine map is NaN); treat as outside the domain.
            nc = self.conn.shape[0]
            safe = np.where((cid >= 0) & (cid < nc), cid, 0)
            valid &= ~self._degenerate[safe]
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
            finite = np.isfinite(w).all(axis=1)
            inside = finite & (w >= -tol).all(axis=1)
            active[idx[inside]] = False
            failed = idx[~finite]
            need_probe[failed] = True
            active[failed] = False
            out = idx[~inside]
            out = out[finite[~inside]]
            if out.size:
                # Cross the face opposite the most-negative barycentric coord, but
                # never step straight back to the cell we just came from -- that is
                # the 2-cycle that traps points sitting on a shared face.
                wn = w[~inside]
                back = self._adj[cells[out]] == prev[out][:, None]
                face = np.argmin(np.where(back, np.inf, wn), axis=1)
                nb = self._adj[cells[out], face]
                boundary = nb < 0
                need_probe[out[boundary]] = True
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

    def walk_locate(self, points_xyz, guess, tol=None, slack=None, max_iter=20):
        """Locate from known topology cells without invoking the probe fallback.

        This is the ALE reseeding path: status 0 is located, while boundary,
        missing-guess, and non-converged results are returned as cell ``-1``.
        """
        if tol is None:
            tol = self.tol
        if slack is None:
            slack = self.slack
        points_xyz = np.ascontiguousarray(points_xyz, dtype=self.dtype)
        guess = np.asarray(guess, dtype=np.int64)
        if not _HAVE_NUMBA:
            cells = self.locate(points_xyz, guess=guess, tol=tol,
                                max_iter=max_iter)
            return cells, np.where(cells >= 0, 0, 2).astype(np.int8)

        n = len(points_xyz)
        cells = np.empty(n, dtype=np.int64)
        status = np.empty(n, dtype=np.int8)
        scratch = np.empty((n, 1), dtype=self.dtype)
        zeros = np.zeros((self.node_xyz.shape[0], 1), dtype=self.dtype)
        _walk_interp_kernel(points_xyz, self._Minv, self._d, self.node_xyz,
                            self._conn64, self._adj, zeros, guess, self.dynamic,
                            tol, slack, max_iter, scratch, cells, status)
        cells[status != 0] = -1
        return cells, status

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

        # Fast path: fused walk + interpolation in one numba kernel. Dynamic ALE
        # walks classify boundary/no-guess directly; only a genuine walk stall
        # still needs the lazy locator. Static sampling retains the prior fallback
        # behavior for every unresolved point.
        n = points_xyz.shape[0]
        v = np.zeros((n, vel.shape[1]), dtype=self.dtype)
        cells = np.empty(n, dtype=np.int64)
        status = np.empty(n, dtype=np.int8)
        _walk_interp_kernel(points_xyz, self._Minv, self._d, self.node_xyz,
                            self._conn64, self._adj, vel, guess.astype(np.int64),
                            self.dynamic, tol, slack, max_iter, v, cells, status)

        if self.dynamic:
            direct_invalid = (status == 1) | (status == 3)
            cells[direct_invalid] = -1
            need = status == 2
        else:
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
