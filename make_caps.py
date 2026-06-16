"""Extract labeled inlet/outlet cap patches from a CFD volume mesh.

Development fixture for the inflow-reseeding feature: the real workflow will use
boundary surfaces exported (named) from the CFD setup. Until those exist, this
reconstructs equivalent labeled caps from the volume file.

How it works: viscous no-slip walls have velocity ~0 on the boundary, while
inlet/outlet caps carry flow. Taking the max |velocity| over all time frames at
each boundary node cleanly separates caps (flow at some phase) from walls
(always ~0). Connected cap faces are then labeled into separate patches.

Output: a PolyData surface of just the caps with an integer ``region_id`` cell
array (0..n_caps-1), saved to caps_labeled.vtp.
"""

import numpy as np
import pyvista as pv

import tracking


def extract_caps(flow_file, out="caps_labeled.vtp", vmag_thresh=0.5, min_faces=20):
    flow = tracking.timeMeshSingleVTU(flow_file, only_active_key=True)
    full = flow.mesh
    surf = full.extract_surface().triangulate()
    orig = surf.point_data["vtkOriginalPointIds"]

    # max |velocity| over all frames at each boundary node (walls stay ~0)
    vmax = np.zeros(surf.n_points)
    for t in flow.times:
        v = full.point_data[f"{flow.active_key}_{t:05d}"][orig]
        vmax = np.maximum(vmax, np.linalg.norm(v, axis=1))

    # a face is a cap face if any of its nodes carries flow -- this keeps the
    # rim ring of faces straddling the cap/wall edge (one or two no-slip nodes),
    # which still carry real flux through their interior. Mirrors a user-provided
    # cap, where every face is a reseeding candidate regardless of nodal values.
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    cap_node = vmax > vmag_thresh
    cap_face = cap_node[faces].any(axis=1)
    print(f"{surf.n_cells} boundary faces -> {cap_face.sum()} cap faces "
          f"({cap_node.sum()} cap nodes of {surf.n_points})")

    # split the cap faces into separate connected patches
    caps = surf.extract_cells(np.where(cap_face)[0]).extract_surface()
    caps = caps.connectivity("all")
    region = np.asarray(caps.cell_data["RegionId"])

    # drop tiny spurious patches, renumber 0..n-1
    keep_ids, counts = np.unique(region, return_counts=True)
    keep_ids = keep_ids[counts >= min_faces]
    mask = np.isin(region, keep_ids)
    caps = caps.extract_cells(np.where(mask)[0]).extract_surface()
    region = np.asarray(caps.cell_data["RegionId"])
    _, region = np.unique(region, return_inverse=True)
    caps.cell_data["region_id"] = region.astype(np.int32)
    if "RegionId" in caps.cell_data:
        del caps.cell_data["RegionId"]

    area = caps.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
    cent = caps.cell_centers().points
    print(f"\n{region.max() + 1} caps:")
    for r in range(region.max() + 1):
        m = region == r
        c = cent[m].mean(0)
        print(f"  cap {r}: faces={m.sum():4d} area={area[m].sum():6.2f} "
              f"centroid=[{c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}]")

    caps.save(out)
    print(f"\nsaved {out}")
    return caps


if __name__ == "__main__":
    extract_caps("P015_pulsatile_rigid_nobackflow.vtu")
