import numpy as np
import pyvista as pv
import pytest

import mrsimtracks as mt
from mrsimtracks.dev import extract_caps


@pytest.fixture
def slow_flow(tmp_path):
    mesh = pv.ImageData(
        dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25)
    ).cast_to_unstructured_grid()
    points = np.asarray(mesh.points)
    interior = ((points[:, 1] > 0) & (points[:, 1] < 1)
                & (points[:, 2] > 0) & (points[:, 2] < 1))
    ends = (points[:, 0] == 0) | (points[:, 0] == 1)
    velocity = np.zeros((mesh.n_points, 3))
    velocity[interior & ends, 0] = 0.2
    mesh.point_data["Velocity_00000"] = velocity
    path = tmp_path / "flow.vtu"
    mesh.save(path)
    return mt.load_flow(path, active_key="Velocity", conform_mesh=False)


def test_no_caps_reports_threshold_and_boundary_maximum(slow_flow, tmp_path):
    output = tmp_path / "caps.vtp"
    with pytest.raises(ValueError, match="threshold 0.5; maximum boundary speed is 0.2"):
        extract_caps(slow_flow, output)
    assert not output.exists()


def test_lower_threshold_extracts_two_native_caps(slow_flow, tmp_path):
    original_cells = slow_flow.active_mesh.cells.copy()
    caps = extract_caps(slow_flow, tmp_path / "caps.vtp", vmag_thresh=0.1)
    assert np.unique(caps.cell_data["region_id"]).tolist() == [0, 1]
    np.testing.assert_array_equal(slow_flow.active_mesh.cells, original_cells)
    reseeder = mt.BoundaryReseeder(caps, slow_flow, rng=np.random.default_rng(2))
    seeds = reseeder.reseed(50, 0)
    assert np.all(slow_flow._sampler.locate(seeds) >= 0)


def test_filtering_all_cap_components_reports_minimum_size(slow_flow, tmp_path):
    output = tmp_path / "caps.vtp"
    with pytest.raises(ValueError, match="at least 1000 faces"):
        extract_caps(slow_flow, output, vmag_thresh=0.1, min_faces=1000)
    assert not output.exists()
