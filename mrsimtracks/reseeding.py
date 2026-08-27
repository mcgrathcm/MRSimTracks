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

Velocity on the cap faces is sampled once per frame using that frame's geometry
and field, so moving-node flow series use the correct spatial interpolation.
"""

from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


@dataclass
class _ALECapState:
    a: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    area: np.ndarray
    charlen: np.ndarray
    normal: np.ndarray
    sample_point: np.ndarray
    cells: np.ndarray
    signed_speed: np.ndarray
    inward_speed: np.ndarray
    cumulative_flux: np.ndarray
    total_flux: float


def _frame_velocity(flow, k):
    """Node velocity array for flow frame index k."""
    return np.asarray(flow._frame_vel(k))


class BoundaryReseeder:
    """Flux-weighted boundary reseeder for particles that leave the domain.

    The reseeder samples currently inflowing cap faces using per-face weights
    ``max(-v . n, 0) * area``. This handles backflow and caps that are partly
    inflow and partly outflow at the same timestep.

    Args:
        caps (pyvista.PolyData | str | pathlib.Path | list): A cap surface with
            a per-cell ``region_id`` array, a path to such a file, or a list of
            cap surface paths/meshes. A list is interpreted as one cap per item.
        flow (object): Loaded flow object returned by ``mrsimtracks.load_flow``.
        rng (numpy.random.Generator | None): Optional generator for repeatable
            reseeding.
        region_key (str): Cell-data array name used to identify cap regions.
        inward_eps (float | None): Minimum inward offset for reseeded points.
        dt (float | None): Tracking time step. When provided, reseeded points
            are spread over a thin inward volume instead of a single plane.
        verify (bool): Check reseeded points with the mesh locator and fall back
            to known-valid face sample points if needed.
    """

    def __init__(self, caps, flow, rng=None, region_key="region_id",
                 inward_eps=None, dt=None, verify=True):
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

        self._build_flux_tables()

    # ---- construction helpers ------------------------------------------------

    def _load_caps(self, caps):
        path_types = (str, bytes, PathLike)
        if isinstance(caps, path_types):
            caps = pv.read(caps)
        elif isinstance(caps, (list, tuple)):
            # one surface (or path) per cap -> stitch with a region id each
            blocks = []
            for i, c in enumerate(caps):
                s = pv.read(c) if isinstance(c, path_types) else c
                s = s.extract_surface(algorithm="dataset_surface").triangulate()
                s.cell_data[self.region_key] = np.full(s.n_cells, i, np.int32)
                blocks.append(s)
            caps = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
        caps = caps.extract_surface(algorithm="dataset_surface").triangulate()
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
        nframes = len(self.flow.times)
        self.frame_t = np.asarray(self.flow.times_shift_s)
        self.tmax = self.flow.tmax

        self._vn = np.zeros((nframes, self.area.shape[0]))      # signed normal vel
        for k in range(nframes):
            loc = self.flow._frame_runtime(k).sampler
            cells = loc.locate(np.ascontiguousarray(self._sample_pt), guess=None)
            valid = cells >= 0
            cells_safe = np.where(valid, cells, 0)
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


class ALEBoundaryReseeder(BoundaryReseeder):
    """Flux-weighted reseeding on deforming ALE boundary caps.

    Cap vertices must correspond to nodes of the ALE reference mesh. At each
    reseeding time the cap is displaced, triangle geometry is recomputed, and
    face probability is proportional to
    ``area * max(-(Velocity - Mesh_velocity) . outward_normal, 0)``.
    """

    def __init__(
        self,
        caps,
        flow,
        rng=None,
        region_key="region_id",
        inward_eps=None,
        dt=None,
        verify=True,
    ):
        if not hasattr(flow, "sample_relative_v"):
            raise TypeError("ALEBoundaryReseeder requires an ALEFlow")
        self.flow = flow
        self.rng = rng if rng is not None else np.random.default_rng()
        self.region_key = region_key
        self.dt = dt
        self.verify = verify
        if inward_eps is not None and inward_eps <= 0:
            raise ValueError("inward_eps must be > 0")
        self.inward_eps = inward_eps
        self._state_cache = OrderedDict()

        caps = self._load_caps(caps)
        self.region = np.asarray(caps.cell_data[region_key]).astype(np.int64)
        self.n_caps = int(self.region.max()) + 1
        self._faces = caps.faces.reshape(-1, 4)[:, 1:]

        reference = np.asarray(flow.reference_mesh.points)
        if "volume_point_id" in caps.point_data:
            node_ids = np.asarray(caps.point_data["volume_point_id"]).astype(
                np.int64
            )
            if np.any(node_ids < 0) or np.any(node_ids >= len(reference)):
                raise ValueError("cap volume_point_id contains invalid node ids")
            distance = np.linalg.norm(reference[node_ids] - caps.points, axis=1)
        else:
            distance, node_ids = cKDTree(reference).query(caps.points)
        tolerance = 1e-6 * (float(np.linalg.norm(np.ptp(reference, axis=0))) or 1.0)
        if np.any(distance > tolerance):
            raise ValueError(
                "ALE cap vertices must match reference volume-mesh nodes; "
                f"maximum mismatch is {float(distance.max()):.6g}"
            )
        self._node_ids = np.asarray(node_ids, dtype=np.int64)

        # ALE topology is invariant: map every cap triangle to its unique
        # adjacent volume tetrahedron once, in the same cell-id namespace used
        # by every deformed runtime sampler.
        sampler = flow._runtime(0.0).sampler
        cap_nodes = self._node_ids[self._faces]
        conn = sampler.conn
        tet_faces = np.stack(
            [conn[:, [1, 2, 3]], conn[:, [0, 2, 3]],
             conn[:, [0, 1, 3]], conn[:, [0, 1, 2]]],
            axis=1,
        )
        boundary_cells, boundary_local = np.where(sampler._adj < 0)
        boundary_nodes = tet_faces[boundary_cells, boundary_local]
        maxn = sampler.node_xyz.shape[0] + 1

        def face_key(faces):
            ordered = np.sort(faces, axis=1).astype(np.int64)
            return (ordered[:, 0] * maxn + ordered[:, 1]) * maxn + ordered[:, 2]

        boundary_key = face_key(boundary_nodes)
        order = np.argsort(boundary_key)
        cap_key = face_key(cap_nodes)
        found = np.searchsorted(boundary_key[order], cap_key)
        safe = np.minimum(found, len(order) - 1)
        if (len(order) == 0
                or np.any(found == len(order))
                or np.any(boundary_key[order[safe]] != cap_key)):
            raise ValueError("ALE cap contains a face not found on the volume boundary")
        match = order[found]
        self._face_cells = boundary_cells[match].astype(np.int64)
        self._face_local = boundary_local[match].astype(np.int8)

        triangles = reference[cap_nodes]
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
        centroid = triangles.mean(axis=1)
        opposite = reference[conn[self._face_cells, self._face_local]]
        orientation = np.einsum("ij,ij->i", cross, opposite - centroid)
        if np.any(orientation == 0):
            raise ValueError("could not orient an ALE cap face from its adjacent cell")
        # Multiply the cap-order normal by this fixed sign to obtain the outward
        # normal at every non-inverted ALE time state.
        self._outward_sign = np.where(orientation > 0, -1.0, 1.0)

    def _deformed_geometry(self, runtime):
        points = runtime.sampler.node_xyz[self._node_ids]
        triangles = points[self._faces]
        a = triangles[:, 0]
        e1 = triangles[:, 1] - a
        e2 = triangles[:, 2] - a
        cross = np.cross(e1, e2)
        twice_area = np.linalg.norm(cross, axis=1)
        if np.any(twice_area == 0):
            raise ValueError("deformed ALE cap contains a zero-area triangle")
        area = 0.5 * twice_area
        unit = cross / twice_area[:, None]
        charlen = (
            np.linalg.norm(e1, axis=1)
            + np.linalg.norm(e2, axis=1)
            + np.linalg.norm(e2 - e1, axis=1)
        ) / 3.0
        centroid = a + (e1 + e2) / 3.0
        return a, e1, e2, area, unit, charlen, centroid

    def _orient(self, runtime, centroid, unit, charlen):
        eps = (
            np.full(len(centroid), float(self.inward_eps))
            if self.inward_eps is not None
            else 0.05 * charlen
        )
        normal = self._outward_sign[:, None] * unit
        opposite = runtime.sampler.node_xyz[
            runtime.sampler.conn[self._face_cells, self._face_local]
        ]
        inward = opposite - centroid
        distance = np.linalg.norm(inward, axis=1)
        if np.any(distance == 0):
            raise ValueError("deformed ALE cap face collapsed onto its opposite node")
        # A convex combination of the face centroid and opposite tetra vertex is
        # guaranteed to lie inside the known adjacent cell. It is the locator-free
        # fallback if a deeper randomized inlet point cannot be walked.
        fraction = np.minimum(eps / distance, 0.25)
        sample_point = centroid + fraction[:, None] * inward
        return normal, sample_point, self._face_cells

    def _cap_state(self, time):
        key, _, _ = self.flow._time_state(time)
        state = self._state_cache.pop(key, None)
        if state is not None:
            self._state_cache[key] = state
            return state

        runtime = self.flow._runtime(time)
        a, e1, e2, area, unit, charlen, centroid = self._deformed_geometry(runtime)
        normal, sample_point, cells = self._orient(
            runtime, centroid, unit, charlen
        )
        relative = runtime.velocity - runtime.mesh_velocity
        face_velocity = relative[self._node_ids][self._faces].mean(axis=1)
        signed_speed = np.einsum("ij,ij->i", face_velocity, normal)
        inward_speed = np.maximum(-signed_speed, 0.0)
        cumulative_flux = np.cumsum(area * inward_speed)
        state = _ALECapState(
            a,
            e1,
            e2,
            area,
            charlen,
            normal,
            sample_point,
            cells,
            signed_speed,
            inward_speed,
            cumulative_flux,
            float(cumulative_flux[-1]),
        )
        self._state_cache[key] = state
        if len(self._state_cache) > 3:
            self._state_cache.popitem(last=False)
        return state

    def _reseed(self, n, t, resolve_cells):
        if n <= 0:
            points = np.empty((0, 3))
            cells = np.empty(0, dtype=np.int64)
            return points, cells
        state = self._cap_state(t)
        cumulative = state.cumulative_flux
        total = state.total_flux
        if total <= 0:
            cumulative = np.cumsum(state.area)
            total = float(cumulative[-1])

        faces = np.searchsorted(
            cumulative, self.rng.random(n) * total, side="right"
        )
        np.clip(faces, 0, len(state.area) - 1, out=faces)
        r1 = self.rng.random(n)
        r2 = self.rng.random(n)
        over = r1 + r2 > 1.0
        r1[over] = 1.0 - r1[over]
        r2[over] = 1.0 - r2[over]
        points = (
            state.a[faces]
            + r1[:, None] * state.e1[faces]
            + r2[:, None] * state.e2[faces]
        )

        if self.dt is None:
            depth = (
                np.full(n, float(self.inward_eps))
                if self.inward_eps is not None
                else 0.05 * state.charlen[faces]
            )
        else:
            layer = np.maximum(
                state.inward_speed[faces] * self.dt,
                state.charlen[faces],
            )
            base = (
                float(self.inward_eps)
                if self.inward_eps is not None
                else 0.05 * state.charlen[faces]
            )
            depth = base + self.rng.random(n) * layer
        points -= depth[:, None] * state.normal[faces]

        cells = state.cells[faces].copy()
        if resolve_cells:
            runtime = self.flow._runtime(t)
            cells, _ = runtime.sampler.walk_locate(
                np.ascontiguousarray(points), cells
            )
            bad = cells < 0
            valid = np.where(~bad)[0]
            if valid.size:
                weights = runtime.sampler._bary(points[valid], cells[valid])
                bad[valid] = (weights < -runtime.sampler.tol).any(axis=1)
            if bad.any():
                points[bad] = state.sample_point[faces[bad]]
                cells[bad] = state.cells[faces[bad]]
        return points, cells

    def reseed(self, n, t):
        """Return ALE-aware seed locations, preserving the public points API."""
        points, _ = self._reseed(n, t, resolve_cells=self.verify)
        return points

    def reseed_with_cells(self, n, t):
        """Return seed locations and their flow-topology cell IDs."""
        return self._reseed(n, t, resolve_cells=True)

    def flux_waveform(self):
        flux = np.zeros((self.flow.n_frames, self.n_caps))
        for frame, time in enumerate(self.flow.times_shift_s):
            state = self._cap_state(time)
            face_flux = state.signed_speed * state.area
            for region in range(self.n_caps):
                flux[frame, region] = face_flux[self.region == region].sum()
        return self.flow.times_shift_s.copy(), flux
