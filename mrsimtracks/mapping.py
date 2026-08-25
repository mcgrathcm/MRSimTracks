import numpy as np

from scipy.spatial import cKDTree


def periodic_mapping(initial_positions, final_positions) -> np.ndarray:
    """Map each initial particle location to its nearest final particle.

    The result is a destination-to-source map for KomaMRI ``FlowPath`` motion:
    particle ``i`` receives the state of particle ``mapping[i]`` at the cycle
    boundary. Indices are 1-based for direct use as Koma's ``cycle_map``.
    Neighbors use ordinary Euclidean distance without spatial boundary
    wrapping. Equidistant ties follow ``scipy.spatial.cKDTree`` selection.

    Args:
        initial_positions (array-like): Initial coordinates with shape ``(n, 3)``.
        final_positions (array-like): Final coordinates with shape ``(n, 3)``.

    Returns:
        np.ndarray: One-based nearest-neighbor indices with shape ``(n,)`` and
            dtype ``int64``. Multiple initial positions may map to the same
            final particle.
    """
    initial = np.asarray(initial_positions)
    final = np.asarray(final_positions)

    for name, positions in (
        ("initial_positions", initial),
        ("final_positions", final),
    ):
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n_particles, 3)")
        if (
            not np.issubdtype(positions.dtype, np.number)
            or np.iscomplexobj(positions)
        ):
            raise ValueError(f"{name} must contain real numeric coordinates")
        if not np.isfinite(positions).all():
            raise ValueError(f"{name} must contain only finite values")

    if len(initial) == 0:
        raise ValueError("periodic mapping requires at least one particle")
    if len(initial) != len(final):
        raise ValueError(
            "initial_positions and final_positions must contain the same "
            "number of particles"
        )

    _, indices = cKDTree(final).query(initial, k=1, workers=-1)
    return np.ascontiguousarray(indices + 1, dtype=np.int64)
