"""Development helper for extracting labeled inlet/outlet cap patches.

This is not the recommended production workflow. MRSimTracks expects users to
provide cap surfaces exported from the CFD setup. This helper exists for
development and exploratory cases where such surfaces are unavailable.

How it works: viscous no-slip walls have zero fluid velocity relative to the
boundary, while inlet/outlet caps carry flow. Taking the maximum velocity (or
ALE relative velocity) over all frames separates caps from walls. Connected cap
faces are then labeled into separate patches.

Output: a PolyData surface of just the caps with an integer ``region_id`` cell
array (0..n_caps-1), saved to caps_labeled.vtp.
"""

import numpy as np
import pyvista as pv

from ..io import load_flow


def extract_caps(flow_file, out="caps_labeled.vtp", vmag_thresh=0.5, min_faces=20,
                 active_key="velocity"):
    flow = load_flow(flow_file, active_key=active_key, only_active_key=True)
    full = flow.active_mesh
    surf = full.extract_surface(algorithm="dataset_surface").triangulate()
    orig = surf.point_data["vtkOriginalPointIds"]

    # max |velocity| over all frames at each boundary node (walls stay ~0)
    vmax = np.zeros(surf.n_points)
    for index in range(len(flow.times)):
        v = flow._frame_vel(index)[orig]
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
    caps = surf.extract_cells(np.where(cap_face)[0]).extract_surface(
        algorithm="dataset_surface")
    caps = caps.connectivity("all")
    region = np.asarray(caps.cell_data["RegionId"])

    # drop tiny spurious patches, renumber 0..n-1
    keep_ids, counts = np.unique(region, return_counts=True)
    keep_ids = keep_ids[counts >= min_faces]
    mask = np.isin(region, keep_ids)
    caps = caps.extract_cells(np.where(mask)[0]).extract_surface(
        algorithm="dataset_surface")
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


def extract_ale_caps(flow, out="caps_labeled.vtp", relative_tol=1e-8,
                     min_faces=20):
    """Extract reference-configuration caps from an :class:`ALEFlow`.

    Boundary nodes whose maximum ``|Velocity - Mesh_velocity|`` exceeds
    ``relative_tol`` times the boundary maximum are treated as flowing nodes.
    A boundary triangle touching any flowing node is retained so triangles at a
    no-slip cap rim are not lost.
    """
    if not hasattr(flow, "relative_velocity"):
        raise TypeError("extract_ale_caps requires an ALEFlow")
    if relative_tol < 0:
        raise ValueError("relative_tol must be >= 0")

    surface = flow.reference_mesh.extract_surface(
        algorithm="dataset_surface"
    ).triangulate()
    original = np.asarray(surface.point_data["vtkOriginalPointIds"])
    surface.point_data["volume_point_id"] = original

    vmax = np.zeros(surface.n_points)
    for frame in range(flow.n_frames):
        relative = flow.relative_velocity(frame)[original]
        vmax = np.maximum(vmax, np.linalg.norm(relative, axis=1))
    maximum = float(vmax.max())
    if maximum <= 0:
        raise ValueError("relative velocity is zero on the entire boundary")
    threshold = relative_tol * maximum

    faces = surface.faces.reshape(-1, 4)[:, 1:]
    cap_node = vmax > threshold
    cap_face = cap_node[faces].any(axis=1)
    if not cap_face.any():
        raise ValueError("no cap faces exceed the relative-velocity threshold")
    print(
        f"{surface.n_cells} boundary faces -> {cap_face.sum()} cap faces "
        f"({cap_node.sum()} cap nodes of {surface.n_points}, "
        f"threshold={threshold:.3g})"
    )

    caps = surface.extract_cells(np.flatnonzero(cap_face)).extract_surface(
        algorithm="dataset_surface"
    )
    caps = caps.connectivity("all")
    region = np.asarray(caps.cell_data["RegionId"])
    region_ids, counts = np.unique(region, return_counts=True)
    keep_ids = region_ids[counts >= min_faces]
    if len(keep_ids) == 0:
        raise ValueError(f"no cap component contains at least {min_faces} faces")
    caps = caps.extract_cells(
        np.flatnonzero(np.isin(region, keep_ids))
    ).extract_surface(algorithm="dataset_surface")
    region = np.asarray(caps.cell_data["RegionId"])
    _, region = np.unique(region, return_inverse=True)
    caps.cell_data["region_id"] = region.astype(np.int32)
    del caps.cell_data["RegionId"]

    area = caps.compute_cell_sizes(
        length=False, area=True, volume=False
    ).cell_data["Area"]
    centers = caps.cell_centers().points
    print(f"\n{region.max() + 1} ALE caps:")
    for cap_id in range(region.max() + 1):
        selected = region == cap_id
        center = centers[selected].mean(axis=0)
        print(
            f"  cap {cap_id}: faces={selected.sum():4d} "
            f"area={area[selected].sum():6.2f} "
            f"centroid=[{center[0]:+.1f}, {center[1]:+.1f}, "
            f"{center[2]:+.1f}]"
        )

    caps.save(out)
    print(f"\nsaved {out}")
    return caps


if __name__ == "__main__":
    extract_caps("P015_pulsatile_rigid_nobackflow.vtu")
