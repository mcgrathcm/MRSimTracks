"""Minimal end-to-end example of the particle_tracking package.

Tracks particles through the example pulsatile U-bend with backflow-aware inflow
reseeding from the provided cap surfaces, then saves the tracks.
"""

import particle_tracking as pt

FLOW = "example/CFD_velocity.vtu"   # single mesh + time-resolved velocity fields
CAPS = ["example/Inlet.vtp", "example/Outlet.vtp"]

# 1. Load the time-resolved flow field (.vtu single-file or .pvd series; auto-detected).
flow = pt.load_flow(FLOW, active_key="Velocity")

# 2. Backflow-aware inflow reseeder from the labeled inlet/outlet surfaces.
#    Passing dt spreads new particles over a thin inflow volume (no density
#    striping) instead of a single plane.
reseeder = pt.BoundaryReseeder(CAPS, flow, dt=0.002)

# 3. Track. Seeds the volume with ~n_particles, advects with RK4, and recycles
#    out-of-bounds particles to currently-inflow cap faces. tmax defaults to one
#    period (flow.tmax).
result = pt.track(flow, n_particles=2e5, dt=0.002, reseeder=reseeder)

# 4. Inspect / save.
print(f"positions {result.positions.shape}  (n_steps, n_particles, 3)")
print(f"total resets: {int(result.reset.sum())}")
result.save("tracks.h5")
print("saved tracks.h5")

# --- Large runs: spread across processes (each worker reloads the field) ---
# result = pt.track_parallel(
#     FLOW, n_particles=2e6, dt=0.002, caps=CAPS,
#     active_key="Velocity", n_workers=3,
# )

# --- No labeled caps? Reconstruct them from a volume mesh (no-slip walls -> v~0):
# pt.extract_caps("case.vtu", out="caps_labeled.vtp")
# reseeder = pt.BoundaryReseeder("caps_labeled.vtp", flow, dt=0.002)
