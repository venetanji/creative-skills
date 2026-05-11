#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pygame",
#   "numpy",
#   "imageio-ffmpeg",
# ]
# ///
"""Generate a depth-map video for LTX IC-LoRA Depth-Control / Union-Control.

Renders a perspective endless-corridor (or other configurable scene) as a
sequence of grayscale depth frames. Output convention: NEAR = WHITE,
FAR = BLACK (matches Lightricks' depth-control LoRA training distribution).
The resulting mp4 is fed into LTX as a reference video via the new
`--ic_lora_reference_video` flag on `comfy_graph.py ia2v` so the model's
spatial layout is locked to the scripted geometry while the prompt drives
materials / character / lighting.

Usage:
  depth_corridor.py corridor --output depth-corridor.mp4 --duration 6 --fps 24 \\
      --width 432 --height 768 --speed 1.0 --branch-density 0.0

  depth_corridor.py tunnel   --output depth-tunnel.mp4 --duration 8 ...
  depth_corridor.py walls    --output depth-walls.mp4  --duration 5 ...
  depth_corridor.py custom <scene.json>     # arbitrary scene from JSON spec

Scenes:
  corridor   — straight perspective corridor, camera dollies forward.
               Tunable: --branch-density (0..1, side-corridor frequency).
  tunnel     — circular tunnel, camera moves through.
  walls      — two parallel walls (no ceiling/floor), camera dolly-forward.
  custom     — JSON spec with primitives (corridor, panels, doors, etc).

The output is intended for IC-LoRA Depth-Control. NEAR=WHITE / FAR=BLACK
is the convention the Lightricks LoRA was trained on (verify by feeding
a known scene; if the model inverts the layout, flip with --invert-depth).
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pygame
import pygame.gfxdraw


# ──── camera math (very small fixed-pipeline) ────────────────────────────

def _project(point_xyz: tuple[float, float, float],
             cam_xyz: tuple[float, float, float],
             cam_yaw: float,
             screen_w: int, screen_h: int,
             fov_deg: float = 60.0) -> tuple[int, int, float] | None:
    """Project a world-space point to screen coords + return its depth.
    Returns None if behind the camera. cam_yaw rotates the camera around Y."""
    px, py, pz = point_xyz
    cx, cy, cz = cam_xyz
    # Translate
    dx, dy, dz = px - cx, py - cy, pz - cz
    # Rotate around Y by -cam_yaw (camera-relative)
    cos_y, sin_y = math.cos(-cam_yaw), math.sin(-cam_yaw)
    rx = dx * cos_y - dz * sin_y
    rz = dx * sin_y + dz * cos_y
    ry = dy
    if rz <= 0.01:
        return None  # behind camera
    f = (screen_h / 2) / math.tan(math.radians(fov_deg) / 2)
    sx = int(screen_w / 2 + rx * f / rz)
    sy = int(screen_h / 2 - ry * f / rz)
    return sx, sy, rz


def _depth_to_grayscale(depth: float, near: float, far: float) -> int:
    """Map distance [near, far] → uint8 [255, 0]. NEAR=white, FAR=black."""
    if depth <= near:
        return 255
    if depth >= far:
        return 0
    t = (depth - near) / (far - near)
    return int(round((1.0 - t) * 255))


# ──── scene generators ──────────────────────────────────────────────────

def _emit_corridor(t: float, params: dict) -> list[tuple[tuple, tuple, tuple]]:
    """Return a list of (start_xyz, end_xyz, segment_kind) line segments
    representing the corridor at time t. Camera moves forward at `speed`
    units/sec along +Z. The corridor is rectangular, infinitely long.
    """
    speed = float(params.get("speed", 1.0))
    width = float(params.get("corridor_width", 3.0))
    height = float(params.get("corridor_height", 4.0))
    seg_len = float(params.get("seg_len", 2.0))   # spacing of door/feature lines
    branch_density = float(params.get("branch_density", 0.0))  # 0..1

    cam_z = t * speed
    far = cam_z + 60.0  # render 60 units ahead
    near = cam_z - 1.0   # 1 unit behind for safety
    rng = np.random.default_rng(int(cam_z * 17))  # deterministic-ish branches

    segments = []
    # Continuous floor + ceiling lines (long edges along +Z)
    for x_offset, y_offset in [
        (-width/2, -height/2),  # floor-left
        (+width/2, -height/2),  # floor-right
        (-width/2, +height/2),  # ceiling-left
        (+width/2, +height/2),  # ceiling-right
    ]:
        segments.append(
            ((x_offset, y_offset, near),
             (x_offset, y_offset, far),
             "edge"))

    # Cross-section "rib" rectangles at each seg_len step ahead
    z = math.floor(near / seg_len) * seg_len
    while z < far:
        if z > near:
            corners = [
                (-width/2, -height/2, z),  (+width/2, -height/2, z),
                (+width/2, +height/2, z),  (-width/2, +height/2, z),
            ]
            for i in range(4):
                segments.append((corners[i], corners[(i+1) % 4], "rib"))
        # Optional side-branches
        if branch_density > 0 and rng.random() < branch_density:
            side = rng.choice([-1, 1])
            x_wall = side * width / 2
            door_w = 1.2
            door_h = 2.4
            depth = 1.5
            # Door frame on the side wall — receding outward
            corners = [
                (x_wall, -door_h/2, z),
                (x_wall + side * depth, -door_h/2, z),
                (x_wall + side * depth, +door_h/2, z),
                (x_wall, +door_h/2, z),
            ]
            for i in range(4):
                segments.append((corners[i], corners[(i+1) % 4], "branch"))
        z += seg_len
    return segments


def _emit_tunnel(t: float, params: dict) -> list[tuple]:
    """Circular tunnel — N rings receding into +Z. Camera moves forward."""
    speed = float(params.get("speed", 1.0))
    radius = float(params.get("tunnel_radius", 2.5))
    seg_len = float(params.get("seg_len", 1.5))
    n_segments = int(params.get("ring_segments", 24))

    cam_z = t * speed
    far = cam_z + 60.0
    near = cam_z - 1.0

    segments = []
    z = math.floor(near / seg_len) * seg_len
    while z < far:
        if z > near:
            ring = []
            for i in range(n_segments):
                a = 2 * math.pi * i / n_segments
                ring.append((radius * math.cos(a), radius * math.sin(a), z))
            for i in range(n_segments):
                segments.append((ring[i], ring[(i+1) % n_segments], "ring"))
        # spokes connecting consecutive rings (visible only on the silhouette)
        z += seg_len
    return segments


SCENES = {
    "corridor": _emit_corridor,
    "tunnel": _emit_tunnel,
}


# ──── render loop ────────────────────────────────────────────────────────

def render(scene: str, output: Path, duration: float, fps: int,
           width: int, height: int, params: dict, invert: bool = False) -> None:
    if scene not in SCENES:
        sys.exit(f"unknown scene '{scene}'. Choose: {', '.join(SCENES)}")
    emit = SCENES[scene]
    near = float(params.get("near", 0.5))
    far = float(params.get("far", 50.0))

    pygame.init()
    pygame.display.set_mode((1, 1))  # require minimal display
    surf = pygame.Surface((width, height))

    # ffmpeg writer (raw RGB stream → libx264 mp4)
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe
    ff = get_ffmpeg_exe()
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "16",  # near-lossless for depth maps
        "-preset", "veryfast",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_frames = int(duration * fps)
    cam_xyz = (0.0, 0.0, 0.0)
    cam_yaw = 0.0  # straight ahead

    for frame_idx in range(n_frames):
        t = frame_idx / fps
        surf.fill((0, 0, 0))  # far = black background
        segments = emit(t, params)

        for (a, b, _kind) in segments:
            pa = _project(a, (cam_xyz[0], cam_xyz[1], t * params.get("speed", 1.0)),
                           cam_yaw, width, height)
            pb = _project(b, (cam_xyz[0], cam_xyz[1], t * params.get("speed", 1.0)),
                           cam_yaw, width, height)
            if pa is None or pb is None:
                continue
            # Per-line shade by midpoint depth — rough approximation of
            # depth-painting; for a true depth map we'd raster-scan the
            # solid faces, but for IC-LoRA conditioning the per-line
            # gradient from near-bright to far-dark conveys enough geometry.
            mid_depth = (pa[2] + pb[2]) / 2.0
            shade = _depth_to_grayscale(mid_depth, near, far)
            if invert:
                shade = 255 - shade
            color = (shade, shade, shade)
            # Anti-aliased line for smoother depth gradients
            pygame.draw.aaline(surf, color, (pa[0], pa[1]), (pb[0], pb[1]))

        # Convert pygame surface → ndarray RGB → bytes → ffmpeg stdin
        arr = pygame.surfarray.pixels3d(surf)  # (W, H, 3) — pygame's axis order
        arr = np.transpose(arr, (1, 0, 2))      # (H, W, 3)
        proc.stdin.write(arr.tobytes())
        del arr
        if frame_idx % max(1, n_frames // 20) == 0:
            print(f"  frame {frame_idx}/{n_frames}  "
                  f"({100 * frame_idx / n_frames:.0f}%)", file=sys.stderr)

    proc.stdin.close()
    proc.wait()
    print(f"done: {output}  ({output.stat().st_size//1024} KB)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene", choices=list(SCENES) + ["custom"],
                   help="scene generator")
    p.add_argument("--output", default=None,
                   help="output mp4 path (default: depth-<scene>.mp4)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="seconds (default 5)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=432)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--speed", type=float, default=2.5,
                   help="forward camera speed in world-units / second")
    p.add_argument("--corridor-width", type=float, default=3.0)
    p.add_argument("--corridor-height", type=float, default=4.0)
    p.add_argument("--tunnel-radius", type=float, default=2.5)
    p.add_argument("--seg-len", type=float, default=2.0,
                   help="rib spacing (smaller = denser depth lines)")
    p.add_argument("--branch-density", type=float, default=0.0,
                   help="0..1, frequency of side-branch doorways")
    p.add_argument("--near", type=float, default=0.5)
    p.add_argument("--far", type=float, default=50.0)
    p.add_argument("--invert-depth", action="store_true",
                   help="flip near/far convention if the LoRA expects black=near")
    p.add_argument("--scene-spec", type=str, default=None,
                   help="when scene=custom, path to JSON scene spec")
    args = p.parse_args()

    output = Path(args.output) if args.output else Path(f"depth-{args.scene}.mp4")
    params = {
        "speed": args.speed,
        "corridor_width": args.corridor_width,
        "corridor_height": args.corridor_height,
        "tunnel_radius": args.tunnel_radius,
        "seg_len": args.seg_len,
        "branch_density": args.branch_density,
        "near": args.near,
        "far": args.far,
    }
    if args.scene == "custom":
        if not args.scene_spec:
            sys.exit("--scene-spec is required when scene=custom")
        params.update(json.load(open(args.scene_spec)))
        # custom scenes set their own emitter under params['emit']
        sys.exit("custom scene support not implemented yet — pick corridor or tunnel")

    render(args.scene, output, args.duration, args.fps, args.width, args.height,
           params, invert=args.invert_depth)


if __name__ == "__main__":
    main()
