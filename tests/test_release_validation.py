import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import mrsimtracks as pt
from diagnostics import (
    density_stats,
    deterministic_cell_center_seeds,
    motion_stats,
    subset_particle_stats,
    trajectory_stats,
)
from fixture_paths import FULL_FLOW, full_flow_available
from fixture_paths import INLET as FULL_INLET
from fixture_paths import OUTLET as FULL_OUTLET

RNG_SEED = 1234
DT = 0.002
N_PARTICLES = 256


def _metrics_path(name):
    out_dir = Path(os.environ.get("MRSIMTRACKS_METRICS_DIR", "release-metrics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


@pytest.mark.large
def test_full_cycle_stability_and_metrics():
    if not full_flow_available():
        pytest.skip("full example VTU was not fetched through Git LFS")

    t0 = time.perf_counter()
    flow = pt.load_flow(FULL_FLOW, active_key="Velocity", pbar=False)
    load_s = time.perf_counter() - t0

    seeds = deterministic_cell_center_seeds(flow, N_PARTICLES)
    reseeder = pt.BoundaryReseeder(
        [FULL_INLET, FULL_OUTLET],
        flow,
        rng=np.random.default_rng(RNG_SEED),
        dt=DT,
    )

    t1 = time.perf_counter()
    result = pt.track(
        flow,
        seeds=seeds,
        dt=DT,
        tmax=flow.tmax,
        reseeder=reseeder,
        pbar=False,
    )
    tracking_s = time.perf_counter() - t1

    bounds = np.array(flow.active_mesh.bounds).reshape(3, 2).tolist()
    traj = trajectory_stats(result, seeds)
    density = density_stats(seeds, result.positions[-1], bounds, bins=(6, 6, 6))
    motion = motion_stats(result, seeds, DT)
    subset = subset_particle_stats(result, seeds, DT, indices=range(8))
    frame_t, flux = reseeder.flux_waveform()
    flux_imbalance = np.abs(flux.sum(axis=1))
    total_flux = np.abs(flux).sum(axis=1)
    peak_total_flux = total_flux.max()

    metrics = {
        "case": "full_cycle",
        "rng_seed": RNG_SEED,
        "n_particles": N_PARTICLES,
        "dt": DT,
        "tmax": float(flow.tmax),
        "n_steps": int(result.n_steps),
        "load_s": float(load_s),
        "tracking_s": float(tracking_s),
        "particle_steps_per_s": float(N_PARTICLES * result.n_steps / tracking_s),
        "trajectory": asdict(traj),
        "density": asdict(density),
        "motion": motion,
        "flux": {
            "n_frames": int(len(frame_t)),
            "n_caps": int(flux.shape[1]),
            "max_relative_imbalance": float(flux_imbalance.max() / peak_total_flux),
            "rms_relative_imbalance": float(
                np.sqrt(np.mean(flux_imbalance**2)) / peak_total_flux),
        },
        "subset_particles": subset,
    }
    _metrics_path("full_cycle_metrics.json").write_text(json.dumps(metrics, indent=2))

    assert result.positions.shape == (429, N_PARTICLES, 3)
    assert traj.finite
    assert traj.moving_fraction > 0.98
    assert traj.median_displacement > 1.0
    assert traj.reset_fraction < 0.02
    assert 0.4 < density.median_nn_ratio < 2.5
    assert density.occupied_bin_ratio > 0.7
    assert density.normalized_l1 < 1.4
    assert motion["speed_mean"] > 0.0
    assert motion["speed_p95"] < 200.0
    assert motion["reset_count_p95"] <= 5.0
    assert flux.shape == (430, 2)
    assert np.isfinite(flux).all()
    assert metrics["flux"]["max_relative_imbalance"] < 0.01
    assert metrics["flux"]["rms_relative_imbalance"] < 0.005
