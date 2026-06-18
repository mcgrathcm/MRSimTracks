import numpy as np
import pyvista as pv
import scipy


def seed_mesh(mesh, npoints, rng=None):
    """Seed points inside the full mesh bounds.

    Args:
        mesh: PyVista mesh defining the flow domain.
        npoints: Target number of seed points. The stochastic refinement can
            return fewer points when the domain is sparse or normalization
            rejects samples.
        rng: Optional ``numpy.random.Generator`` for repeatable seeding.

    Returns:
        Array of seed points with shape ``(n, 3)``.
    """
    bounds = mesh.bounds
    return seed_region(mesh, npoints, bounds, rng=rng)

def seed_region(mesh, npoints, bounds, normalization=None, rng=None):
    """Seed points inside a bounded region of a mesh.

    Args:
        mesh: PyVista mesh defining the flow domain.
        npoints: Target number of seed points.
        bounds: Region bounds in PyVista order
            ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        normalization: Optional point-data array name used for stochastic
            density weighting.
        rng: Optional ``numpy.random.Generator`` for repeatable seeding.

    Returns:
        Array of seed points with shape ``(n, 3)``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    npoints = int(npoints)
    if npoints < 1:
        raise ValueError("npoints must be >= 1")
    if len(bounds) != 6:
        raise ValueError("bounds must contain six values")

    vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
    if vol <= 0:
        raise ValueError("bounds must define a positive volume")
    # Aim for 10x less inital points than desired
    npoints_initial = max(1, npoints//10)
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
    surf = mesh.extract_surface(algorithm=None)

    # Initial guess of points  inside the surface
    selected = point_cloud.select_interior_points(surf)
    selected_key = "selected_points" if "selected_points" in selected.point_data else "SelectedPoints"
    inside = selected[selected_key]
    # This is like a binary mask of the mesh
    inside_array = inside.reshape(len(z), len(x), len(y))
    # dilate (to account for boundary regions)
    inside_array = scipy.ndimage.binary_dilation(
        inside_array, structure=np.ones((5, 5, 5)), iterations=1
    )
    inside = inside_array.flatten()


    valid = points[np.argwhere(inside)[:, 0], :]
    if valid.shape[0] == 0:
        raise ValueError("no candidate seed points found inside the requested region")

    # # Refine with sample operation
    # valid = valid0[np.argwhere(point_valid.sample(mesh)['vtkValidPointMask'])[:,0],:]
    # point_valid = pv.PolyData(valid)

    # Now we have a good distribution of points inside the surface
    # We can add even more (randomly) and then refine again
    n_subsample = (
        int(np.ceil(npoints / valid.shape[0])) * 2
    )  # Assume about half will be outside
    rand_pts = rng.random((valid.shape[0], n_subsample, 3)) - 0.5
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
        if normalization not in samp.point_data:
            raise ValueError(f"normalization array {normalization!r} not found")

        if samp[normalization].ndim > 1:
            normag = np.sum(samp[normalization]**2, axis=1)**0.5
            max_density = np.max(normag)
            density = normag / max_density if max_density > 0 else np.zeros_like(normag)
        else:
            values = samp[normalization]
            max_density = np.max(values)
            density = values / max_density if max_density > 0 else np.zeros_like(values)

        # Density is now between 0 and 1, giving the probability of keeping each point
        rand = rng.random(density.shape[0])
        keep = rand < density
        subsampled = subsampled[keep, :]

    return subsampled
