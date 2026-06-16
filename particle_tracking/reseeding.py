"""Time-resolved, flux-weighted inflow reseeding over boundary cap patches.

The original tracker recycles out-of-bounds particles to random points in a
single static inlet volume. That cannot represent backflow: a cap that is partly
(or, over the cycle, intermittently) an outflow should only receive new particles
where and when flow is actually entering the domain.

``BoundaryReseeder`` takes a set of labeled boundary caps (inlets/outlets) and,
for each flow time frame, computes the inflow flux ``max(-v.n, 0) * area`` on
every cap face. Reseeding at time ``t`` draws faces with probability proportional
to the inflow flux at the nearest frame, so particles only enter through faces
that are currently inflow -- correct for backflow and partial inflow/outflow on a
single cap. Flux weighting also makes seed density proportional to local inflow,
generalizing the old velocity-magnitude weighting.

Velocity on the (static) cap faces is sampled once per frame by locating the face
sample points in the volume mesh a single time and interpolating each frame's
field, reusing the fast tet sampler.
"""

import numpy as np
import pyvista as pv


def _frame_velocity(flow, k):
    """Node velocity array for flow frame index k (handles all timeMesh* classes)."""
    if hasattr(flow, "fields"):                 # timeMeshStaticPVD (one geom + fields)
        return np.asarray(flow.fields[k])
    if hasattr(flow, "meshes"):                 # timeMeshPVD (full mesh per frame)
        return np.asarray(flow.meshes[k].point_data[flow.active_key])
    return np.asarray(flow.mesh.point_data[flow._get_mesh_key(k)])  # timeMeshSingleVTU


class BoundaryReseeder:
    def __init__(self, caps, flow, rng=None, region_key="region_id",
                 inward_eps=None, dt=None, verify=True):
        """
        caps : pv.PolyData with a per-cell ``region_key`` array, a path to such a
               file, or a list of surface meshes/paths (one cap each).
        flow : a timeMesh* object exposing an all-tet ``_sampler``.
        inward_eps : minimum distance to offset seed points inside the domain
               along the inward normal. Defaults to ~half the median cap edge.
        dt : tracking time step. When given, seeds are spread over a random
               inward depth (a thin inflow *volume*) instead of a single plane,
               so successive per-step reseeds overlap rather than forming
               advecting density stripes -- important for uniform-density MR use.
               When None, a fixed ``inward_eps`` offset is used (plane seeding).
        """
        self.flow = flow
        self.rng = rng if rng is not None else np.random.default_rng()
        self.region_key = region_key
        self.dt = dt
        self.verify = verify
        if not getattr(flow, "_sampler", None) or not flow._sampler.ok:
            raise ValueError("BoundaryReseeder requires an all-tetrahedral flow mesh")

        caps = self._load_caps(caps)
        self.region = np.asarray(caps.cell_data[region_key]).astype(np.int64)
        self.n_caps = int(self.region.max()) + 1

        # Triangle geometry: store base vertex + edge vectors for fast sampling.
        tris = caps.faces.reshape(-1, 4)[:, 1:]
        pts = np.asarray(caps.points)
        self._a = pts[tris[:, 0]]
        self._e1 = pts[tris[:, 1]] - self._a
        self._e2 = pts[tris[:, 2]] - self._a
        cross = np.cross(self._e1, self._e2)
        self.area = 0.5 * np.linalg.norm(cross, axis=1)
        unit = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)
        centroid = self._a + (self._e1 + self._e2) / 3.0
        # per-face characteristic length (mean edge) -> minimum seed-layer depth
        self._charlen = (np.linalg.norm(self._e1, axis=1)
                         + np.linalg.norm(self._e2, axis=1)
                         + np.linalg.norm(self._e2 - self._e1, axis=1)) / 3.0

        if inward_eps is None:
            inward_eps = 0.5 * np.median(np.linalg.norm(self._e1, axis=1))
        self.inward_eps = float(inward_eps)

        # Orient normals outward and build interior sample points, using the
        # locator: whichever side of the face finds a containing cell is inside.
        self.normal, self._sample_pt = self._orient(centroid, unit)
        self._sample_cells = flow._sampler.locate(
            np.ascontiguousarray(self._sample_pt), guess=None)

        self._build_flux_tables()

    # ---- construction helpers ------------------------------------------------

    def _load_caps(self, caps):
        if isinstance(caps, (str, bytes)):
            caps = pv.read(caps)
        elif isinstance(caps, (list, tuple)):
            # one surface (or path) per cap -> stitch with a region id each
            blocks = []
            for i, c in enumerate(caps):
                s = pv.read(c) if isinstance(c, (str, bytes)) else c
                s = s.extract_surface().triangulate()
                s.cell_data[self.region_key] = np.full(s.n_cells, i, np.int32)
                blocks.append(s)
            caps = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
        caps = caps.extract_surface().triangulate()
        if self.region_key not in caps.cell_data:
            caps.cell_data[self.region_key] = np.zeros(caps.n_cells, np.int32)
        return caps

    def _orient(self, centroid, unit):
        loc = self.flow._sampler
        plus = np.ascontiguousarray(centroid + self.inward_eps * unit)
        inside_plus = loc.locate(plus, guess=None) >= 0
        # outward normal points away from the interior side
        normal = np.where(inside_plus[:, None], -unit, unit)
        sample_pt = centroid - self.inward_eps * normal
        return normal, sample_pt

    def _build_flux_tables(self):
        """Per-frame inflow flux per face and its cumulative sum (for sampling)."""
        loc = self.flow._sampler
        valid = self._sample_cells >= 0
        cells_safe = np.where(valid, self._sample_cells, 0)
        nframes = len(self.flow.times)
        self.frame_t = np.asarray(self.flow.times_shift_s)
        self.tmax = self.flow.tmax

        self._vn = np.zeros((nframes, self.area.shape[0]))      # signed normal vel
        for k in range(nframes):
            v = loc._interp(self._sample_pt, cells_safe, _frame_velocity(self.flow, k))
            v[~valid] = 0.0
            self._vn[k] = np.einsum("ij,ij->i", v, self.normal)

        inflow = np.maximum(-self._vn, 0.0) * self.area        # q >= 0
        self._cum = np.cumsum(inflow, axis=1)                  # (nframes, nfaces)
        self._total = self._cum[:, -1].copy()

    def _frame_index(self, t):
        tw = t % self.tmax
        return int(np.argmin(np.abs(self.frame_t - tw)))

    # ---- public API ----------------------------------------------------------

    def reseed(self, n, t):
        """Return ``(n, 3)`` seed points just inside currently-inflow cap faces."""
        if n <= 0:
            return np.empty((0, 3))
        k = self._frame_index(t)
        cum, total = self._cum[k], self._total[k]
        if total <= 0:           # no inflow at this instant: fall back to area weighting
            cum = np.cumsum(self.area)
            total = cum[-1]

        u = self.rng.random(n) * total
        f = np.searchsorted(cum, u, side="right")
        np.clip(f, 0, self.area.shape[0] - 1, out=f)

        # uniform point within each chosen triangle (reflection method)
        r1 = self.rng.random(n)
        r2 = self.rng.random(n)
        over = r1 + r2 > 1.0
        r1[over] = 1.0 - r1[over]
        r2[over] = 1.0 - r2[over]
        p_surf = self._a[f] + r1[:, None] * self._e1[f] + r2[:, None] * self._e2[f]

        # Offset inward. With dt, randomize the depth over a layer thick enough
        # that consecutive reseeds overlap: a particle penetrates ~v_n*dt per
        # step, so spreading new seeds over U(0, max(v_n*dt, cell)) makes discrete
        # plane seeding equivalent to continuous volumetric inflow (no striping).
        if self.dt is not None:
            vn_in = np.maximum(-self._vn[k][f], 0.0)          # inflow normal speed
            layer = np.maximum(vn_in * self.dt, self._charlen[f])
            depth = self.inward_eps + self.rng.random(n) * layer
        else:
            depth = np.full(n, self.inward_eps)
        p = p_surf - depth[:, None] * self.normal[f]

        if self.verify:
            cells = self.flow._sampler.locate(np.ascontiguousarray(p), guess=None)
            bad = cells < 0
            if bad.any():        # rare: fall back to the known-valid face sample point
                p[bad] = self._sample_pt[f[bad]]
        return p

    def flux_waveform(self):
        """Net signed flux per cap over the cycle (positive = outflow).

        Returns ``(frame_times, flux[nframes, n_caps])``. Summing across caps per
        frame should be ~0 by mass conservation -- a useful correctness check.
        """
        face_flux = self._vn * self.area                       # (nframes, nfaces)
        out = np.zeros((face_flux.shape[0], self.n_caps))
        for r in range(self.n_caps):
            out[:, r] = face_flux[:, self.region == r].sum(axis=1)
        return self.frame_t, out
