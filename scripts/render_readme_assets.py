"""Render README visualization assets from the full U-bend example.

This script intentionally uses the full Git LFS example:

    uv run python scripts/render_readme_assets.py

The default settings match the README assets:

- inlet-seeded speed-colored particle WebP
- selected inlet-seeded trajectory PNG, cut at reseed boundaries
- full-volume-seeded grayscale center-slice density WebP

Use ``--quick`` for a lower-cost smoke render. Use ``--include-projection`` to
also render the grayscale all-particle x-z projection-density WebP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageFilter, ImageSequence

import mrsimtracks as mt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", type=Path, default=Path("example/CFD_velocity.vtu"))
    parser.add_argument("--inlet", type=Path, default=Path("example/Inlet.vtp"))
    parser.add_argument("--outlet", type=Path, default=Path("example/Outlet.vtp"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--frames", type=int, default=110)
    parser.add_argument("--duration-ms", type=int, default=40)
    parser.add_argument(
        "--animation-scale",
        type=float,
        default=0.55,
        help="Scale animation frames before writing to reduce README asset size.",
    )
    parser.add_argument(
        "--gif-colors",
        type=int,
        default=96,
        help="Adaptive palette size when writing GIF output.",
    )
    parser.add_argument(
        "--webp-quality",
        type=int,
        default=70,
        help="Quality setting when writing animated WebP output.",
    )
    parser.add_argument("--inlet-particles", type=int, default=3_200)
    parser.add_argument("--density-particles", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-projection", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller particle/frame counts for a fast smoke render.",
    )
    return parser.parse_args()


def set_camera(plotter: pv.Plotter, zoom: float = 1.25) -> None:
    plotter.view_xz()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(zoom)


def crop_white_image(image: Image.Image, pad: int = 34, threshold: int = 248) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    box = (
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, rgb.width),
        min(int(ys.max()) + pad + 1, rgb.height),
    )
    return image.crop(box)


def crop_white_animation(
    path: Path,
    *,
    pad: int = 18,
    threshold: int = 248,
    duration_ms: int = 30,
    gif_colors: int = 96,
    webp_quality: int = 80,
) -> None:
    im = Image.open(path)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]
    mask = None
    for frame in frames:
        arr = np.asarray(frame)
        current = np.any(arr < threshold, axis=2)
        mask = current if mask is None else mask | current

    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        box = (
            max(int(xs.min()) - pad, 0),
            max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad + 1, frames[0].width),
            min(int(ys.max()) + pad + 1, frames[0].height),
        )
        frames = [frame.crop(box) for frame in frames]

    save_animation(
        path,
        frames,
        duration_ms=duration_ms,
        gif_colors=gif_colors,
        webp_quality=webp_quality,
    )


def save_animation(
    path: Path,
    frames: list[Image.Image],
    *,
    duration_ms: int,
    scale: float = 1.0,
    gif_colors: int = 96,
    webp_quality: int = 80,
) -> None:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale != 1.0:
        resized = []
        for frame in frames:
            width = max(1, int(round(frame.width * scale)))
            height = max(1, int(round(frame.height * scale)))
            resized.append(frame.resize((width, height), Image.Resampling.LANCZOS))
        frames = resized
    if path.suffix.lower() == ".webp":
        frames = [frame.convert("RGB") for frame in frames]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            format="WEBP",
            quality=webp_quality,
            method=6,
            # Force every frame to be a keyframe. Otherwise libwebp encodes long
            # chains of delta frames, and a decoder (e.g. macOS Preview) must
            # replay the whole chain to draw a late frame, so playback lags more
            # and more until the loop resets it. For sparse particle-on-white
            # content keyframes cost essentially no extra size.
            kmin=1,
            kmax=1,
        )
    else:
        paletted = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=gif_colors)
            for frame in frames
        ]
        paletted[0].save(
            path,
            save_all=True,
            append_images=paletted[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )


def speed_from_positions(positions: np.ndarray, reset: np.ndarray, dt: float) -> np.ndarray:
    speed = np.zeros((positions.shape[0], positions.shape[1]), dtype=float)
    step_speed = np.linalg.norm(np.diff(positions, axis=0), axis=2) / dt
    step_speed = np.where(reset[1:], np.nan, step_speed)
    speed[1:] = step_speed
    speed[0] = speed[1]
    return speed


def load_surfaces(flow, inlet_path: Path, outlet_path: Path):
    surface = flow.active_mesh.extract_surface(algorithm=None).triangulate()
    inlet = pv.read(inlet_path).extract_surface(algorithm="dataset_surface").triangulate()
    outlet = pv.read(outlet_path).extract_surface(algorithm="dataset_surface").triangulate()
    return surface, inlet, outlet


def render_particle_gif(
    flow,
    surface,
    inlet,
    outlet,
    reseeder,
    out_path: Path,
    *,
    dt: float,
    cycles: int,
    n_particles: int,
    frames: int,
    duration_ms: int,
    seed: int,
    animation_scale: float,
    gif_colors: int,
    webp_quality: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    seeds = reseeder.reseed(n_particles, t=0.0)
    result = mt.track(
        flow,
        seeds=seeds,
        dt=dt,
        tmax=flow.tmax * cycles,
        reseeder=reseeder,
        rng=np.random.default_rng(seed + 1),
        pbar=False,
    )
    positions = result.positions
    reset = result.reset
    speed = speed_from_positions(positions, reset, dt)
    vmax = float(np.percentile(speed[np.isfinite(speed)], 98.5))
    frame_indices = np.linspace(0, positions.shape[0] - 1, frames, dtype=int)

    images = []
    for frame_i, step_i in enumerate(frame_indices, start=1):
        if frame_i == 1 or frame_i % 25 == 0:
            print(f"particle frame {frame_i}/{frames}")
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 700))
        plotter.set_background("white")
        plotter.add_mesh(surface, color="#d9d9d9", opacity=0.22, smooth_shading=True)
        plotter.add_mesh(inlet, color="#2ca25f", opacity=0.50)
        plotter.add_mesh(outlet, color="#2b6cb0", opacity=0.50)

        cloud = pv.PolyData(positions[step_i])
        cloud.point_data["speed"] = np.nan_to_num(
            speed[step_i], nan=0.0, posinf=vmax, neginf=0.0
        )
        plotter.add_mesh(
            cloud,
            scalars="speed",
            cmap="jet",
            clim=(0.0, vmax),
            point_size=3.0,
            render_points_as_spheres=True,
            opacity=0.92,
            show_scalar_bar=False,
        )
        set_camera(plotter)
        images.append(Image.fromarray(plotter.screenshot(return_img=True)))
        plotter.close()

    save_animation(
        out_path,
        images,
        duration_ms=duration_ms,
        scale=animation_scale,
        gif_colors=gif_colors,
        webp_quality=webp_quality,
    )
    crop_white_animation(
        out_path,
        duration_ms=duration_ms,
        gif_colors=gif_colors,
        webp_quality=webp_quality,
    )
    return positions, reset, speed, vmax


def select_track_segments(
    positions: np.ndarray,
    reset: np.ndarray,
    speed: np.ndarray,
    flow,
    *,
    max_tracks: int = 16,
) -> list[tuple[np.ndarray, np.ndarray]]:
    bounds = np.array(flow.active_mesh.bounds, dtype=float)
    _, _, _, _, zmin, zmax = bounds
    z_mid = 0.5 * (zmin + zmax)
    reset_count = np.cumsum(reset, axis=0)

    segments = []
    for particle_id in range(positions.shape[1]):
        splits = np.r_[
            0,
            np.flatnonzero(np.diff(reset_count[:, particle_id]) != 0) + 1,
            positions.shape[0],
        ]
        for start, stop in zip(splits[:-1], splits[1:]):
            if stop - start < 180:
                continue
            indices = np.arange(start, stop, 2)
            seg = positions[indices, particle_id]
            seg_speed = speed[indices, particle_id]
            xspan = np.ptp(seg[:, 0])
            zspan = np.ptp(seg[:, 2])
            if xspan < 7.0 or zspan < 10.0 or seg[:, 2].max() < z_mid:
                continue
            path_len = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
            displacement = np.linalg.norm(seg[-1] - seg[0])
            score = path_len + 1.5 * displacement + xspan + zspan
            segments.append((score, seg, seg_speed))

    segments.sort(key=lambda item: item[0], reverse=True)
    selected = []
    starts = []
    endpoints = []
    for _, seg, seg_speed in segments:
        start = seg[0, [0, 2]]
        endpoint = seg[-1, [0, 2]]
        if endpoints:
            d_end = np.min([np.linalg.norm(endpoint - other) for other in endpoints])
            d_start = np.min([np.linalg.norm(start - other) for other in starts])
            if d_end < 0.32 and d_start < 0.32:
                continue
        selected.append((seg, seg_speed))
        starts.append(start)
        endpoints.append(endpoint)
        if len(selected) >= max_tracks:
            break

    if len(selected) < 12:
        selected = [(seg, seg_speed) for _, seg, seg_speed in segments[:max_tracks]]
    print(f"track candidates={len(segments)}, selected={len(selected)}")
    return selected


def render_tracks_png(
    flow,
    surface,
    inlet,
    outlet,
    out_path: Path,
    positions: np.ndarray,
    reset: np.ndarray,
    speed: np.ndarray,
    vmax: float,
) -> None:
    selected = select_track_segments(positions, reset, speed, flow)
    plotter = pv.Plotter(off_screen=True, window_size=(2600, 1800))
    plotter.set_background("white")
    plotter.add_mesh(surface, color="#d9d9d9", opacity=0.22, smooth_shading=True)
    plotter.add_mesh(inlet, color="#2ca25f", opacity=0.50)
    plotter.add_mesh(outlet, color="#2b6cb0", opacity=0.50)

    for seg, seg_speed in selected:
        line = pv.lines_from_points(seg)
        line.point_data["speed"] = np.nan_to_num(
            seg_speed, nan=0.0, posinf=vmax, neginf=0.0
        )
        plotter.add_mesh(
            line,
            scalars="speed",
            cmap="jet",
            clim=(0.0, vmax),
            opacity=0.88,
            line_width=4,
            show_scalar_bar=False,
        )

    set_camera(plotter, zoom=1.17)
    plotter.show(screenshot=str(out_path))
    plotter.close()
    crop_white_image(Image.open(out_path)).save(out_path)


def render_slice_frame(
    points: np.ndarray,
    *,
    y_mid: float,
    y_half_thickness: float,
    surface,
    inlet,
    outlet,
) -> Image.Image:
    keep = np.abs(points[:, 1] - y_mid) <= y_half_thickness
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 700))
    plotter.set_background("white")
    plotter.add_mesh(surface, color="#d9d9d9", opacity=0.13, smooth_shading=True)
    plotter.add_mesh(inlet, color="#9ca3af", opacity=0.28)
    plotter.add_mesh(outlet, color="#9ca3af", opacity=0.28)
    if np.any(keep):
        plotter.add_mesh(
            pv.PolyData(points[keep]),
            color="#111827",
            point_size=1.5,
            render_points_as_spheres=True,
            opacity=0.55,
            show_scalar_bar=False,
        )
    set_camera(plotter)
    frame = Image.fromarray(plotter.screenshot(return_img=True))
    plotter.close()
    return frame


def render_histogram_frame(
    points: np.ndarray,
    *,
    x0: float,
    x1: float,
    z0: float,
    z1: float,
    hist_size: tuple[int, int] = (780, 700),
) -> Image.Image:
    width, height = hist_size
    x = points[:, 0]
    z = points[:, 2]
    valid = (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
    xi = ((x[valid] - x0) / (x1 - x0) * (width - 1)).astype(np.int32)
    yi = ((z1 - z[valid]) / (z1 - z0) * (height - 1)).astype(np.int32)

    hist = np.zeros((height, width), dtype=np.float32)
    np.add.at(hist, (yi, xi), 1.0)
    image = Image.fromarray(
        np.clip(hist / max(float(hist.max()), 1.0) * 255, 0, 255).astype(np.uint8),
        "L",
    )
    image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 0:
        arr = np.log1p(arr)
        arr /= arr.max()
    arr = np.clip((arr**0.55) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def rk4_step(flow, positions: np.ndarray, cells, time: float, dt: float):
    k1, valid, cells = flow.sample_v(positions, time, guess=cells)
    k2, _, c2 = flow.sample_v(k1 * dt / 2 + positions, time + dt / 2, guess=cells)
    k3, _, c3 = flow.sample_v(k2 * dt / 2 + positions, time + dt / 2, guess=c2)
    k4, _, _ = flow.sample_v(k3 * dt + positions, time + dt, guess=c3)
    velocity = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return positions + velocity * dt, valid, cells


def render_density_gifs(
    flow,
    surface,
    inlet,
    outlet,
    reseeder,
    out_path: Path,
    projection_path: Path | None,
    *,
    dt: float,
    cycles: int,
    n_particles: int,
    frames: int,
    duration_ms: int,
    seed: int,
    animation_scale: float,
    gif_colors: int,
    webp_quality: int,
) -> None:
    rng = np.random.default_rng(seed)
    # Seed volume-uniformly, weighting each cell by its volume. Sampling cell
    # centers uniformly would make seed density proportional to cell count per
    # unit volume (i.e. 1/cell_volume), oversampling the refined boundary-layer
    # and refinement regions of the CFD mesh and producing spurious density
    # banding at walls. Volume weighting cancels that so density tracks the
    # physical distribution, independent of mesh resolution.
    sized = flow.active_mesh.compute_cell_sizes(length=False, area=False, volume=True)
    centers = np.asarray(sized.cell_centers().points)
    cell_volume = np.abs(np.asarray(sized.cell_data["Volume"], dtype=float))
    total_volume = cell_volume.sum()
    if total_volume <= 0:
        raise ValueError("mesh has no positive-volume cells to seed from")
    cell_choice = rng.choice(
        centers.shape[0], size=n_particles, replace=True, p=cell_volume / total_volume
    )
    positions = np.ascontiguousarray(centers[cell_choice], dtype=float)
    print(f"full-volume seeds={positions.shape[0]}")

    bounds = np.array(flow.active_mesh.bounds, dtype=float)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y_mid = 0.5 * (ymin + ymax)
    y_half_thickness = 0.10 * (ymax - ymin)

    xpad = 0.02 * (xmax - xmin)
    zpad = 0.03 * (zmax - zmin)
    x0, x1 = xmin - xpad, xmax + xpad
    z0, z1 = zmin - zpad, zmax + zpad

    n_steps = int((flow.tmax * cycles) / dt)
    frame_steps = np.linspace(0, n_steps - 1, frames, dtype=int)
    frame_set = set(int(i) for i in frame_steps)
    slice_frames = []
    projection_frames = [] if projection_path is not None else None

    cells = None
    for step_i in range(n_steps):
        positions, valid, cells = rk4_step(flow, positions, cells, step_i * dt, dt)

        oob = ~valid
        n_oob = int(oob.sum())
        if n_oob:
            positions[oob] = reseeder.reseed(n_oob, step_i * dt)
            if cells is not None:
                cells[oob] = -1

        if step_i in frame_set:
            frame_i = len(slice_frames) + 1
            print(f"density frame {frame_i}/{frames} at step {step_i}/{n_steps}, oob={n_oob}")
            # Tracking occurs for the full volume. The slice is applied only here,
            # when making a rendered frame.
            slice_frames.append(
                render_slice_frame(
                    positions,
                    y_mid=y_mid,
                    y_half_thickness=y_half_thickness,
                    surface=surface,
                    inlet=inlet,
                    outlet=outlet,
                )
            )
            if projection_frames is not None:
                projection_frames.append(
                    render_histogram_frame(positions, x0=x0, x1=x1, z0=z0, z1=z1)
                )

    save_animation(
        out_path,
        slice_frames,
        duration_ms=duration_ms,
        scale=animation_scale,
        gif_colors=gif_colors,
        webp_quality=webp_quality,
    )
    crop_white_animation(
        out_path,
        duration_ms=duration_ms,
        gif_colors=gif_colors,
        webp_quality=webp_quality,
    )
    if projection_path is not None and projection_frames is not None:
        save_animation(
            projection_path,
            projection_frames,
            duration_ms=duration_ms,
            scale=animation_scale,
            gif_colors=gif_colors,
            webp_quality=webp_quality,
        )


def main() -> None:
    args = parse_args()
    if args.quick:
        args.frames = min(args.frames, 30)
        args.inlet_particles = min(args.inlet_particles, 800)
        args.density_particles = min(args.density_particles, 12_000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pv.OFF_SCREEN = True

    flow = mt.load_flow(args.flow, active_key="Velocity", pbar=False)
    surface, inlet, outlet = load_surfaces(flow, args.inlet, args.outlet)

    inlet_reseeder = mt.BoundaryReseeder(
        [args.inlet, args.outlet],
        flow,
        rng=np.random.default_rng(args.seed),
        dt=args.dt,
    )
    positions, reset, speed, vmax = render_particle_gif(
        flow,
        surface,
        inlet,
        outlet,
        inlet_reseeder,
        args.out_dir / "ubend_particles.webp",
        dt=args.dt,
        cycles=args.cycles,
        n_particles=args.inlet_particles,
        frames=args.frames,
        duration_ms=args.duration_ms,
        seed=args.seed,
        animation_scale=args.animation_scale,
        gif_colors=args.gif_colors,
        webp_quality=args.webp_quality,
    )
    render_tracks_png(
        flow,
        surface,
        inlet,
        outlet,
        args.out_dir / "ubend_tracks.png",
        positions,
        reset,
        speed,
        vmax,
    )

    density_reseeder = mt.BoundaryReseeder(
        [args.inlet, args.outlet],
        flow,
        rng=np.random.default_rng(args.seed + 100),
        dt=args.dt,
    )
    render_density_gifs(
        flow,
        surface,
        inlet,
        outlet,
        density_reseeder,
        args.out_dir / "ubend_density_slice.webp",
        args.out_dir / "ubend_density_projection.webp" if args.include_projection else None,
        dt=args.dt,
        cycles=args.cycles,
        n_particles=args.density_particles,
        frames=args.frames,
        duration_ms=args.duration_ms,
        seed=args.seed + 200,
        animation_scale=args.animation_scale,
        gif_colors=args.gif_colors,
        webp_quality=args.webp_quality,
    )


if __name__ == "__main__":
    main()
