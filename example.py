"""Minimal end-to-end example of the mrsimtracks package.

Tracks particles through the example pulsatile U-bend with backflow-aware inflow
reseeding from the provided cap surfaces, then saves the tracks.
"""

import numpy as np

import mrsimtracks as mt
from mrsimtracks.seeding import seed_mesh

FLOW = "example/CFD_velocity.vtu"   # single mesh + time-resolved velocity fields
CAPS = ["example/Inlet.vtp", "example/Outlet.vtp"]

# 1. Load the time-resolved flow field (.vtu single-file or .pvd series; auto-detected).
flow = mt.load_flow(FLOW, active_key="Velocity")

# 2. Backflow-aware inflow reseeder from the labeled inlet/outlet surfaces.
#    Passing dt spreads new particles over a thin inflow volume (no density
#    striping) instead of a single plane.
reseeder = mt.BoundaryReseeder(CAPS, flow, dt=0.002)

# 3. Seed and track. Seeding is explicit so users control the initial particle
#    distribution. tmax defaults to one period (flow.tmax).
seeds = seed_mesh(flow.active_mesh, 2e5, rng=np.random.default_rng(0))
result = mt.track(flow, seeds=seeds, dt=0.002, reseeder=reseeder)

# 4. Inspect / save.
print(f"positions {result.positions.shape}  (n_steps, n_particles, 3)")
print(f"total resets: {int(result.reset.sum())}")
result.save("tracks.h5")
print("saved tracks.h5")

# --- Large runs: spread across processes (each worker reloads the field) ---
# result = mt.track_parallel(
#     FLOW, seeds=seeds, dt=0.002, caps=CAPS,
#     active_key="Velocity", n_workers=3,
# )

# --- No labeled caps? Reconstruct them from a volume mesh (no-slip walls -> v~0):
# from mrsimtracks.dev import extract_caps
# extract_caps("case.vtu", out="caps_labeled.vtp", active_key="Velocity")
# reseeder = mt.BoundaryReseeder("caps_labeled.vtp", flow, dt=0.002)
