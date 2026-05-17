#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mido",
#   "numpy",
#   "pygame",
#   "imageio-ffmpeg",
# ]
# ///
"""3D tunnel ride with MIDI-driven wall bumps. Software-rendered port of
the operator's midi2canny.html mode3d.js (three.js depth-mode tunnel).

Pygame software 3D — perspective projection + wireframe rendering of a
cylinder mesh that the camera flies through. Per-vertex amplitude offsets
spawn from MIDI onsets and decay over time, animating the tunnel walls.

Output: grayscale wireframe (NEAR=white, FAR=black via projected-z shade)
intended as the depth/canny conditioning video for LTX IC-LoRA Union-Control.

What's ported:
- Cylinder mesh (NUM_SEGMENTS × RING_RADIAL verts) with per-vertex bump
  state (age, amplitude). Same defaults as the JS (60 × 48).
- Camera: forward speed modulated by drum kick onsets + bar-locked
  beat-pulse. Roll torque from snare. Subtle position bob.
- bumpTerrain: spawns a vertex bump at the right segment ahead of camera
  (~22 units), with neighbor-smearing.
- Per-stem onset routing — drums (kick/snare/hat/cymbal sub-types),
  bass, guitar, synth, vocals, backing, fx.

What's deliberately NOT ported (yet):
- Spawned obstacles (guitar/synth icosahedrons / cylinders ahead of camera)
- Pulse rings (kick spawns ring sprite that travels toward camera)
- 3D color/normal-shading mode (depth-only output is what feeds IC-LoRA)
- Star points (depth cues — easy to add as a follow-up)
- Z-buffer / hidden-surface (wireframe only — visible geometry is the
  back-facing inside of the tunnel which the camera always sees from inside)

Usage:
  tunnel_3d.py --midi-dir stems/ --output tunnel.mp4 \\
      --duration 6 --fps 24 --width 448 --height 768 \\
      --bpm 128 --start-sec 58.38

The output is the conditioning video. Feed via:
  comfy_graph.py ia2v --image anchor.png --audio slice.mp3 \\
      --prompt "endless flickering hospital corridor" \\
      --width 448 --height 768 --seconds 6 --fast \\
      --ic_loras "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors:1.0" \\
      --ic_lora_reference_video tunnel.mp4
"""
from __future__ import annotations
import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

import mido
import numpy as np
import pygame
from imageio_ffmpeg import get_ffmpeg_exe


# ──── glob conventions (shared with midi2canny.py) ───────────────────────

TRACK_GLOBS = {
    "drums":   ["*Drums*.mid", "*drums*.mid"],
    "bass":    ["*Bass*.mid", "*bass*.mid"],
    "guitar":  ["*Guitar*.mid", "*guitar*.mid"],
    "synth":   ["*Synth*.mid", "*synth*.mid"],
    "vocals":  ["*Vocals*.mid", "*vocals*.mid", "*Vocal*.mid", "*vocal*.mid"],
    "backing": ["*Backing*.mid", "*backing*.mid"],
    "fx":      ["*FX*.mid", "*fx*.mid"],
}


def parse_track(midi_path: Path, start_sec: float = 0.0):
    mid = mido.MidiFile(str(midi_path))
    notes = []
    abs_t = 0.0
    pending = {}
    for msg in mid:
        abs_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending[msg.note] = (abs_t, msg.velocity)
        elif (msg.type == "note_off"
              or (msg.type == "note_on" and msg.velocity == 0)) and msg.note in pending:
            on_t, vel = pending.pop(msg.note)
            dur = max(0.05, abs_t - on_t)
            if on_t >= start_sec:
                notes.append({
                    "time": on_t - start_sec, "duration": dur,
                    "midi": msg.note, "velocity": vel / 127.0,
                })
    notes.sort(key=lambda n: n["time"])
    if not notes:
        return [], {"min": 36, "max": 84}, 0.0
    rng = {"min": min(n["midi"] for n in notes),
           "max": max(n["midi"] for n in notes)}
    end = max(n["time"] + n["duration"] for n in notes)
    return notes, rng, end


def norm_pitch(midi, rng):
    if rng["max"] == rng["min"]:
        return 0.5
    return (midi - rng["min"]) / (rng["max"] - rng["min"])


# ──── Engine: walk per-stem cursors, emit onsets ────────────────────────

class Engine:
    def __init__(self, stems, bpm):
        self.stems = stems
        self.bpm = bpm
        self.cursors = {sid: 0 for sid in stems}
        self.last_t = -1.0

    def reset(self, t):
        self.cursors = {}
        for sid, s in self.stems.items():
            i = 0
            ns = s["notes"]
            while i < len(ns) and ns[i]["time"] < t:
                i += 1
            self.cursors[sid] = i
        self.last_t = t

    def tick(self, t, dt):
        if dt < 0 or dt > 0.5:
            self.reset(t)
        onsets = {sid: [] for sid in self.stems}
        for sid, s in self.stems.items():
            ns = s["notes"]
            i = self.cursors.get(sid, 0)
            while i < len(ns) and ns[i]["time"] <= t:
                onsets[sid].append(ns[i])
                i += 1
            self.cursors[sid] = i
        self.last_t = t
        beat = (t * self.bpm) / 60.0
        return {"time": t, "dt": dt, "beat": beat, "bpm": self.bpm,
                "onsets": onsets}


# ──── Tunnel mesh (port of buildTunnelMesh + per-vertex bump state) ─────

class Tunnel:
    def __init__(self, num_segments=60, ring_radial=48, ring_radius=6.0,
                 segment_spacing=4.0):
        self.NUM_SEGMENTS = num_segments
        self.RING_RADIAL = ring_radial
        self.RING_RADIUS = ring_radius
        self.SEGMENT_SPACING = segment_spacing
        # Per-segment baseline z and per-vertex bump state
        self.base_z = np.array([-i * segment_spacing for i in range(num_segments)],
                                dtype=np.float64)
        # bumpAge / bumpAmp: shape (NUM_SEGMENTS, RING_RADIAL)
        self.bump_age = np.full((num_segments, ring_radial), 999.0)
        self.bump_amp = np.zeros((num_segments, ring_radial))
        # Pre-compute static angle for each radial slot
        self.angles = np.linspace(0, 2 * np.pi, ring_radial, endpoint=False)
        self.cos_a = np.cos(self.angles)
        self.sin_a = np.sin(self.angles)

    def recycle(self, camera_z):
        """Move segments behind the camera to the far end, like the JS does."""
        thresh = camera_z + self.SEGMENT_SPACING * 1.5
        for s in range(self.NUM_SEGMENTS):
            if self.base_z[s] > thresh:
                min_z = float(self.base_z.min())
                self.base_z[s] = min_z - self.SEGMENT_SPACING
                self.bump_age[s, :] = 999.0
                self.bump_amp[s, :] = 0.0

    def step_bumps(self, dt):
        self.bump_age += dt

    def bump(self, angle, amplitude, camera_z, spawn_ahead=22.0):
        """Spawn a vertex bump at the segment closest to (camera_z - spawn_ahead)
        and the radial slot matching `angle`."""
        target_z = camera_z - spawn_ahead
        seg = int(np.argmin(np.abs(self.base_z - target_z)))
        radial = ((angle / (2 * np.pi)) % 1 + 1) % 1
        idx = int(radial * self.RING_RADIAL) % self.RING_RADIAL
        cap = self.RING_RADIUS - 0.8
        self.bump_age[seg, idx] = 0.0
        self.bump_amp[seg, idx] = min(cap,
                                      max(self.bump_amp[seg, idx], amplitude))
        # Smear to neighbors ±2 indices (with falloff)
        for k in (1, 2):
            for n in (idx + k, idx - k):
                a = n % self.RING_RADIAL
                self.bump_age[seg, a] = min(self.bump_age[seg, a], 0.0)
                self.bump_amp[seg, a] = min(cap,
                                            max(self.bump_amp[seg, a],
                                                amplitude * (1 - k / 3)))

    def vertex_positions(self, camera_z):
        """Return (NUM_SEGMENTS, RING_RADIAL, 3) world-space vertex coords."""
        decay = np.exp(-self.bump_age * 0.7)
        offset = self.bump_amp * decay  # (S, R)
        r = np.maximum(0.6, self.RING_RADIUS - offset)  # (S, R)
        # Broadcast to xyz
        x = r * self.cos_a[None, :]
        y = r * self.sin_a[None, :]
        z = np.broadcast_to(self.base_z[:, None], r.shape)
        return np.stack([x, y, z], axis=-1)  # (S, R, 3)


# ──── Software 3D projection + wireframe rasterizer ─────────────────────

def project_points(points, camera, W, H, fov_deg=75.0):
    """Project (N, 3) world-space points → (N, 3) screen [x_px, y_px, depth].
    `camera` has fields x, y, z, roll. Returns NaN for points behind camera.
    """
    # Camera-relative
    rel = points - np.array([camera["x"], camera["y"], camera["z"]])
    # Apply roll around z-axis (camera-relative)
    cos_r, sin_r = math.cos(-camera["roll"]), math.sin(-camera["roll"])
    rx = rel[..., 0] * cos_r - rel[..., 1] * sin_r
    ry = rel[..., 0] * sin_r + rel[..., 1] * cos_r
    rz = rel[..., 2]
    # Camera looks down -Z. Behind = rz > 0.
    # Mask points behind camera (NaN out).
    behind = rz >= -0.01
    f = (H / 2) / math.tan(math.radians(fov_deg) / 2)
    sx = W / 2 + rx * f / -rz
    sy = H / 2 - ry * f / -rz
    out = np.stack([sx, sy, -rz], axis=-1)  # depth = positive distance
    out[behind] = np.nan
    return out


def shade_for_depth(depth, near=0.5, far=60.0):
    """Map depth to grayscale 0..255 (near=255, far=0). NaN→0."""
    if not np.isfinite(depth):
        return 0
    if depth <= near:
        return 255
    if depth >= far:
        return 0
    t = (depth - near) / (far - near)
    return int(round((1 - t) * 255))


def draw_wire_quads(surf, projected, num_segments, ring_radial,
                    near=0.5, far=60.0, line_w=1):
    """Draw the tunnel quads as edge wireframes with depth-shaded color.
    `projected`: (S, R, 3) screen-space xyz_depth.
    """
    W, H = surf.get_size()
    # For each cell (s, i): edges to (s, i+1) and (s+1, i)
    for s in range(num_segments - 1):
        for i in range(ring_radial):
            i_next = (i + 1) % ring_radial
            v_a = projected[s, i]
            v_b = projected[s, i_next]
            v_c = projected[s + 1, i]
            # Cross-section (a-b on same ring)
            if np.isfinite(v_a[0]) and np.isfinite(v_b[0]):
                d = (v_a[2] + v_b[2]) / 2
                shade = shade_for_depth(d, near, far)
                if shade > 5:
                    pygame.draw.aaline(surf, (shade, shade, shade),
                                        (v_a[0], v_a[1]),
                                        (v_b[0], v_b[1]))
            # Longitudinal (a-c connecting consecutive rings)
            if np.isfinite(v_a[0]) and np.isfinite(v_c[0]):
                d = (v_a[2] + v_c[2]) / 2
                shade = shade_for_depth(d, near, far)
                if shade > 5:
                    pygame.draw.aaline(surf, (shade, shade, shade),
                                        (v_a[0], v_a[1]),
                                        (v_c[0], v_c[1]))


# ──── per-stem MIDI handlers (port of handleOnsets) ─────────────────────

def handle_drums(notes, tunnel, camera_state):
    for n in notes:
        m = n["midi"]
        v = n["velocity"]
        if m <= 37:
            # KICK → camera speed boost (additive, decays in main loop)
            camera_state["speed_boost"] = max(camera_state["speed_boost"],
                                               v * 60)
        elif m <= 41:
            # SNARE → roll impulse
            camera_state["roll_vel"] += (1 if random.random() < 0.5 else -1) \
                                        * 1.6 * v
        elif m <= 49:
            # HAT → tiny terrain bump on a random angle
            tunnel.bump(random.random() * 2 * math.pi,
                        0.5 + v * 1.5, camera_state["z"])
        else:
            # CYMBAL → wide bump (smear several radials)
            a0 = random.random() * 2 * math.pi
            for k in range(-3, 4):
                tunnel.bump(a0 + k * 0.2, 1.5 + v * 2, camera_state["z"])


def handle_bass(notes, meta, tunnel, camera_state):
    for n in notes:
        v = n["velocity"]
        p = norm_pitch(n["midi"], meta["range"])
        a0 = -math.pi / 2 + (p - 0.5) * 1.4  # bottom-anchored, bias by pitch
        amp = 1.5 + v * 3.2
        for k in range(-2, 3):
            tunnel.bump(a0 + k * 0.18, amp, camera_state["z"])


def handle_guitar(notes, meta, tunnel, camera_state):
    # Spawning obstacles is deferred — for now we just nudge bumps mildly.
    for n in notes:
        v = n["velocity"]
        p = norm_pitch(n["midi"], meta["range"])
        a0 = (p - 0.5) * 2 * math.pi
        for k in range(-2, 3):
            tunnel.bump(a0 + k * 0.15, 0.8 + v * 1.5, camera_state["z"])


def handle_synth(notes, meta, tunnel, camera_state):
    for n in notes:
        v = n["velocity"]
        p = norm_pitch(n["midi"], meta["range"])
        a0 = (p - 0.5) * 2 * math.pi + math.pi / 4  # offset for visual variety
        for k in range(-1, 2):
            tunnel.bump(a0 + k * 0.2, 0.6 + v * 1.2, camera_state["z"])


def handle_vocals(notes, meta, tunnel, camera_state):
    for n in notes:
        v = n["velocity"]
        p = norm_pitch(n["midi"], meta["range"])
        a0 = (p - 0.5) * math.pi  # pitch → side
        amp = 1.0 + v * 2.5
        for k in range(-4, 5):
            tunnel.bump(a0 + k * 0.15, amp * (1 - abs(k) / 5),
                        camera_state["z"])
        # Pitch-driven tunnel spin — higher vocal notes accelerate the
        # camera roll. Operator request 2026-05-16: "the fractal LoRA at
        # the end of the tunnel should spin faster driven by the pitch
        # of the voice". Normalized pitch in [0,1] × velocity becomes a
        # positive impulse on roll_vel, accumulated per vocal onset.
        # Sustained high vocals = sustained CW spin acceleration that
        # the snare's existing ± wobble rides on top of. The natural
        # decay in the main loop pulls roll_vel back down when vocals
        # fall silent (e.g., during the post-drop drum-only beats).
        camera_state["roll_vel"] += 3.0 * p * v


def handle_backing(notes, tunnel, camera_state):
    for n in notes:
        v = n["velocity"]
        for _ in range(6):
            tunnel.bump(random.random() * 2 * math.pi,
                        0.4 + v, camera_state["z"])


def handle_fx(notes, tunnel, camera_state):
    for n in notes:
        v = n["velocity"]
        count = 6 + int(v * 12)
        for _ in range(count):
            tunnel.bump(random.random() * 2 * math.pi,
                        0.6 + random.random() * 2, camera_state["z"])


# ──── render loop ────────────────────────────────────────────────────────

def render(stems, bpm, output: Path, *,
           duration: float, fps: int, width: int, height: int,
           start_sec: float = 0.0, base_speed: float = 24.0,
           # When set, the camera speed linearly interpolates from
           # base_speed at t=0 to end_speed at t=duration. Useful for
           # transition-window conds where the tunnel-flight should ease
           # out before scene B takes over.
           end_speed: float | None = None,
           fov_deg: float = 75.0, near: float = 0.5, far: float = 60.0,
           seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    pygame.init()
    pygame.display.set_mode((1, 1))
    surf = pygame.Surface((width, height))
    tunnel = Tunnel()
    engine = Engine(stems, bpm)

    camera_state = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "roll": 0.0, "roll_vel": 0.0,
        "speed_boost": 0.0,
    }
    meta = {sid: {"range": s["range"]} for sid, s in stems.items()}

    n_frames = int(duration * fps)
    ff = get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{width}x{height}",
           "-framerate", str(fps),
           "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "16", "-preset", "veryfast",
           str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    last_t = 0.0
    for fidx in range(n_frames):
        t = fidx / fps
        dt = (t - last_t) if fidx > 0 else 1.0 / fps
        last_t = t

        state = engine.tick(t, dt)

        # Route onsets
        for sid in state["onsets"]:
            ns = state["onsets"][sid]
            if not ns:
                continue
            if sid == "drums":
                handle_drums(ns, tunnel, camera_state)
            elif sid == "bass":
                handle_bass(ns, meta[sid], tunnel, camera_state)
            elif sid == "guitar":
                handle_guitar(ns, meta[sid], tunnel, camera_state)
            elif sid == "synth":
                handle_synth(ns, meta[sid], tunnel, camera_state)
            elif sid == "vocals":
                handle_vocals(ns, meta[sid], tunnel, camera_state)
            elif sid == "backing":
                handle_backing(ns, tunnel, camera_state)
            elif sid == "fx":
                handle_fx(ns, tunnel, camera_state)

        # Camera step
        beat = state["beat"]
        beat_pulse = math.sin(beat * 2 * math.pi)
        if end_speed is None:
            cur_base = base_speed
        else:
            ramp = t / max(duration, 1e-6)
            cur_base = base_speed + (end_speed - base_speed) * ramp
        speed = (cur_base + camera_state["speed_boost"]) + beat_pulse * 4
        camera_state["z"] -= speed * dt
        camera_state["speed_boost"] *= math.pow(0.05, dt)
        camera_state["roll_vel"] *= math.pow(0.1, dt)
        camera_state["roll"] += camera_state["roll_vel"] * dt
        # Subtle bob
        camera_state["x"] = math.sin(beat * math.pi) * 0.3
        camera_state["y"] = math.cos(beat * math.pi * 0.5) * 0.2

        # Tunnel state update
        tunnel.recycle(camera_state["z"])
        tunnel.step_bumps(dt)

        # Build verts + project
        verts = tunnel.vertex_positions(camera_state["z"])  # (S, R, 3)
        flat = verts.reshape(-1, 3)
        projected_flat = project_points(flat, camera_state, width, height, fov_deg)
        projected = projected_flat.reshape(verts.shape)

        # Draw
        surf.fill((0, 0, 0))
        draw_wire_quads(surf, projected,
                         tunnel.NUM_SEGMENTS, tunnel.RING_RADIAL,
                         near=near, far=far)

        arr = pygame.surfarray.pixels3d(surf)
        arr = np.transpose(arr, (1, 0, 2))  # (H, W, 3)
        proc.stdin.write(arr.tobytes())
        del arr
        if fidx % max(1, n_frames // 20) == 0:
            print(f"  frame {fidx}/{n_frames}  "
                  f"({100 * fidx / n_frames:.0f}%)", file=sys.stderr)

    proc.stdin.close()
    proc.wait()
    print(f"done: {output}  ({output.stat().st_size//1024} KB)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi-dir", default=None)
    for sid in TRACK_GLOBS:
        p.add_argument(f"--{sid}", default=None)
    p.add_argument("--output", default="tunnel-3d.mp4")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--bpm", type=float, default=128.0)
    p.add_argument("--end-speed", type=float, default=None,
                   help="Linear ramp from --base-speed at t=0 to --end-speed "
                        "at t=duration. Default = no ramp (constant base_speed).")
    p.add_argument("--base-speed", type=float, default=24.0,
                   help="forward camera speed in world units / second")
    p.add_argument("--fov", type=float, default=75.0)
    p.add_argument("--near", type=float, default=0.5)
    p.add_argument("--far", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stems = {}
    midi_dir = Path(args.midi_dir).resolve() if args.midi_dir else None
    for sid, globs in TRACK_GLOBS.items():
        explicit = getattr(args, sid)
        if explicit:
            path = Path(explicit).resolve()
        elif midi_dir:
            path = None
            for g in globs:
                cands = list(midi_dir.glob(g))
                if sid == "vocals":
                    cands = [c for c in cands if "backing" not in c.name.lower()]
                if cands:
                    path = cands[0]
                    break
            if path is None:
                continue
        else:
            continue
        notes, rng, end = parse_track(path, start_sec=args.start_sec)
        stems[sid] = {"notes": notes, "range": rng, "end": end}
        print(f"  loaded {sid}: {path.name}  "
              f"{len(notes)} notes", file=sys.stderr)

    if not stems:
        sys.exit("no MIDI tracks loaded — pass --midi-dir or per-stem flags")

    duration = args.duration
    if duration is None:
        duration = min(30.0, max(s["end"] for s in stems.values()) + 1.0)

    render(stems, args.bpm, Path(args.output).resolve(),
           duration=duration, fps=args.fps,
           width=args.width, height=args.height,
           start_sec=args.start_sec, base_speed=args.base_speed,
           end_speed=args.end_speed,
           fov_deg=args.fov, near=args.near, far=args.far,
           seed=args.seed)


if __name__ == "__main__":
    main()
