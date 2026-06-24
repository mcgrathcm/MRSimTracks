"""Near-wall no-penetration (slip) projection for particle tracking.

Discrete velocity fields interpolated from stabilized CFD output are not exactly
divergence-free; near no-slip walls that leaves a small spurious wall-normal
velocity that pushes tracers into the wall, where ``v -> 0`` traps them (a
one-way "ratchet" that deposits a growing layer of stuck particles).

``WallSlip`` removes the *into-wall* component of a particle's velocity when it
is within a thin band of a wall, so it slides along instead of being deposited:

    v* = v - max(v . n_out, 0) n_out          (n_out = outward wall normal)

Only the wall-normal component is removed; tangential (and back-into-fluid)
motion is untouched, and interior particles are unaffected.

The band is a fixed fraction of the vessel's hydraulic diameter
(``D_h = 4 V / A_wall``). The deposition layer is a boundary-layer-scale feature
that tracks the vessel diameter rather than the local mesh size, so a fraction of
``D_h`` is the robust, predictable choice; ~2% suppresses the deposition while
keeping the particle-to-wall gap small. Open boundaries (inlet/outlet caps) are
excluded so flux still passes.

This is a particle-level boundary condition (a modelling choice), not a fix to
the field; it is the targeted, minimally-invasive way to suppress the wall
deposition without re-meshing.
"""

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from .sampler import _tet_volumes

# local node indices of the face opposite each local vertex of a tet
_FACE = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


class WallSlip:
    """Near-wall no-penetration projection sized as a fraction of vessel diameter.

    Args:
        flow (object): Loaded flow with an all-tetrahedral mesh (load with the
            default ``conform_mesh=True``); uses its fast sampler's geometry.
        caps (list | None): Open-boundary cap surfaces (paths or meshes) to
            exclude from the wall set so inflow/outflow is not blocked. If
            ``None``, every domain-boundary face is treated as a wall.
        band_frac (float): Band thickness as a fraction of the vessel hydraulic
            diameter ``D_h = 4 V / A_wall``. Default ``0.02`` (2%).

    Attributes:
        d_hydraulic (float): Estimated vessel diameter.
        band (float): Absolute band thickness used (``band_frac * d_hydraulic``).
    """

    def __init__(self, flow, caps=None, band_frac=0.02):
        sampler = getattr(flow, "_sampler", None)
        if sampler is None or not getattr(sampler, "ok", False):
            raise ValueError("WallSlip requires an all-tetrahedral flow mesh "
                             "(load with conform_mesh=True)")
        node = np.asarray(sampler.node_xyz, dtype=np.float64)
        conn = sampler.conn

        cells, faces = np.where(sampler._adj == -1)        # boundary faces
        fnodes = conn[cells[:, None], _FACE[faces]]        # (nb, 3) face node ids
        opp = conn[cells, faces]                           # (nb,) interior vertex

        p = node[fnodes]                                   # (nb, 3, 3)
        centroid = p.mean(axis=1)
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = 0.5 * np.linalg.norm(cross, axis=1)
        normal = cross / (2.0 * area[:, None])
        # orient outward: point away from the cell's interior (opposite vertex)
        flip = np.einsum("ij,ij->i", normal, centroid - node[opp]) < 0
        normal[flip] *= -1.0

        wall = self._wall_mask(node, fnodes, caps)
        if not wall.any():
            raise ValueError("no wall faces found (all boundary faces matched the "
                             "caps); check the cap surfaces")

        # vessel hydraulic diameter D_h = 4 V / A_wall (== diameter for a tube)
        total_volume = float(_tet_volumes(node, conn).sum())
        self.d_hydraulic = 4.0 * total_volume / float(area[wall].sum())
        self.band = float(band_frac * self.d_hydraulic)

        self._centroid = np.ascontiguousarray(centroid[wall])
        self._normal = np.ascontiguousarray(normal[wall])
        self._tree = cKDTree(self._centroid)

    @staticmethod
    def _wall_mask(node, fnodes, caps):
        """Boundary faces that are NOT entirely on an open-boundary cap."""
        if not caps:
            return np.ones(fnodes.shape[0], dtype=bool)
        tol = 1e-6 * float(np.linalg.norm(np.ptp(node, axis=0)))
        tree = cKDTree(node)
        is_cap = np.zeros(node.shape[0], dtype=bool)
        for cap in caps:
            surf = cap if isinstance(cap, pv.DataSet) else pv.read(cap)
            dist, idx = tree.query(np.asarray(surf.points, dtype=float))
            is_cap[idx[dist <= tol]] = True
        return ~is_cap[fnodes].all(axis=1)                 # wall unless all 3 nodes are cap

    def apply(self, positions, velocity):
        """Remove the into-wall velocity for particles within the wall band.

        ``velocity`` is modified in place and returned. ``positions`` is the
        current particle position used to find the nearest wall face.
        """
        dist, face = self._tree.query(positions)
        within = dist < self.band
        if within.any():
            n = self._normal[face[within]]
            vn = np.einsum("ij,ij->i", velocity[within], n)
            velocity[within] -= np.maximum(vn, 0.0)[:, None] * n
        return velocity
