# Boundary Reseeding

Particles that leave the flow domain can be recycled through inlet/outlet cap
surfaces.

`BoundaryReseeder` computes a time-resolved inflow weight for every cap face:

```text
max(-v . n, 0) * area
```

This means particles are only reseeded where the current flow is entering the
domain. The approach handles backflow and caps that are partially inflow and
partially outflow at the same time.

## Volumetric Seed Layer

Pass the tracking time step to spread reseeded particles over a thin inward
layer:

```python
reseeder = pt.BoundaryReseeder(caps, flow, dt=dt)
```

This reduces repeated plane-seeding artifacts and helps maintain smoother
particle density for downstream MR simulation.

## Flux Diagnostics

Use `flux_waveform()` to inspect net signed cap flux over the cycle:

```python
times, flux = reseeder.flux_waveform()
```

The sum across caps should be small relative to the total cap flux for a
well-resolved incompressible field.
