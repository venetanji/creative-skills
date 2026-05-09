#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mido",
#   "numpy",
#   "opencv-python-headless",
#   "imageio-ffmpeg",
# ]
# ///
"""MIDI → canny-edge motion video. Python port of the operator's
midi2canny.html canvas project (js/mode2d.js semantics preserved).

Each stem becomes a different visual system on a black canvas:
  drums    → kick=ring + camera shake + bg flash, snare=slash, hats=ticks,
             cymbal=radial rays
  bass     → horizontal bar at pitch-derived y, with vertical tick marks
  guitar   → drives a shared polygon-field underlay (the "friendly polygons"
             pattern — N polygons orbiting on a ring, fractional side count
             so the closing edge sweeps as sides drifts, rotation +
             scale modulated by guitar pitch + bar-locked sine waves)
  synth    → orbiting arc from frame center, radius = pitch
  vocals   → breathing sine wave at pitch-derived y
  backing  → horizontal line at pitch-derived y
  fx       → scattered glitch rectangles

White-on-black by default (canny-friendly). Flip via --color-mode for
debug.

Usage:
  midi2canny.py --midi-dir stems-midi/ \\
      --output canny-full.mp4 --duration 6 --fps 24 \\
      --width 432 --height 768

  # The stems/ dir must contain the per-stem .mid files. Lookup happens
  # by glob pattern per stem (see TRACK_GLOBS) so naming is flexible:
  #   *Drums*.mid, *Bass*.mid, *Guitar*.mid, *Synth*.mid,
  #   *Vocal*.mid (lead), *Backing*.mid, *FX*.mid

  # Or pass tracks explicitly:
  midi2canny.py --drums drums.mid --guitar guitar.mid --vocals vocals.mid \\
      --output canny.mp4 --duration 8

Then feed the output as the IC-LoRA reference video:
  comfy_graph.py ia2v --image anchor.png --audio slice.mp3 \\
      --prompt "..." --width 448 --height 768 --seconds 6 --fast \\
      --ic_loras "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors:1.0" \\
      --ic_lora_reference_video canny.mp4
"""
from __future__ import annotations
import argparse
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable

import cv2
import mido
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe


# ──── stem registry (mirror of MV.TRACKS in engine.js) ───────────────────

# Stem → list of glob patterns tried in order. First match wins. Case-
# sensitive on the host's filesystem, so we list both forms — `Drums.mid`
# (Lightricks/operator stems-zip convention) and `drums.mid` (the
# midi2canny canvas project convention).
TRACK_GLOBS = {
    "drums":   ["*Drums*.mid", "*drums*.mid"],
    "bass":    ["*Bass*.mid", "*bass*.mid"],
    "guitar":  ["*Guitar*.mid", "*guitar*.mid"],
    "synth":   ["*Synth*.mid", "*synth*.mid"],
    # Lead vocals — exclude any "Backing Vocals" file by listing it as a
    # negative match in glob form is messy, so we match exact-prefix
    # names first then fall back broader.
    "vocals":  ["*Vocals*.mid", "*vocals*.mid", "*Vocal*.mid", "*vocal*.mid"],
    "backing": ["*Backing*.mid", "*backing*.mid"],
    "fx":      ["*FX*.mid", "*fx*.mid"],
}


# ──── MIDI parsing ───────────────────────────────────────────────────────

def parse_track(midi_path: Path, start_sec: float = 0.0
                ) -> tuple[list[dict], dict, float]:
    """Returns (notes, range, end_sec). Each note dict has keys
    time, duration, midi, velocity (mirroring the JS shape)."""
    mid = mido.MidiFile(str(midi_path))
    notes: list[dict] = []
    abs_t = 0.0
    pending: dict[int, tuple[float, int]] = {}
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
                    "time": on_t - start_sec,
                    "duration": dur,
                    "midi": msg.note,
                    "velocity": vel / 127.0,
                })
    notes.sort(key=lambda n: n["time"])
    if not notes:
        return [], {"min": 36, "max": 84}, 0.0
    rng = {
        "min": min(n["midi"] for n in notes),
        "max": max(n["midi"] for n in notes),
    }
    end = max(n["time"] + n["duration"] for n in notes)
    return notes, rng, end


def norm_pitch(midi: int, rng: dict) -> float:
    """Normalize midi pitch into 0..1 of the track's observed range."""
    if rng["max"] == rng["min"]:
        return 0.5
    return (midi - rng["min"]) / (rng["max"] - rng["min"])


# ──── particle system + shared polygon-field state ───────────────────────

class Engine:
    """One-shot stateful engine: walks notes of all stems against time t,
    emits onsets that fired in [last_t, t] and the set of active notes
    (those whose [time, time+duration] envelopes contain t)."""
    def __init__(self, stems: dict, bpm: float):
        self.stems = stems  # {id: {notes, range}}
        self.bpm = bpm
        self.cursors = {sid: 0 for sid in stems}
        self.last_t = -1.0

    def reset(self, t: float):
        self.cursors = {}
        for sid, s in self.stems.items():
            i = 0
            ns = s["notes"]
            while i < len(ns) and ns[i]["time"] < t:
                i += 1
            self.cursors[sid] = i
        self.last_t = t

    def tick(self, t: float, dt: float) -> dict:
        if dt < 0 or dt > 0.5:
            self.reset(t)
        onsets = {sid: [] for sid in self.stems}
        active = {sid: [] for sid in self.stems}
        for sid, s in self.stems.items():
            ns = s["notes"]
            i = self.cursors.get(sid, 0)
            # Onsets that fire AT-OR-BEFORE t
            while i < len(ns) and ns[i]["time"] <= t:
                onsets[sid].append(ns[i])
                i += 1
            self.cursors[sid] = i
            # Active notes: scan back ~4s for sustains touching t
            for j in range(i - 1, -1, -1):
                n = ns[j]
                if n["time"] < t - 4.0:
                    break
                if n["time"] <= t and n["time"] + n["duration"] >= t:
                    active[sid].append(n)
        self.last_t = t
        beat = (t * self.bpm) / 60.0
        return {"time": t, "dt": dt, "beat": beat, "bpm": self.bpm,
                "onsets": onsets, "active": active}


class PolyField:
    """Port of the 2D mode polygon-field state. Drawn underneath the
    particle layer every frame, driven by guitar onsets + global volume."""
    def __init__(self, w: int, h: int, count: int = 14, rings: int = 1,
                 line_width: float = 2.0):
        self.W, self.H = w, h
        self.count = count
        self.rings = rings
        self.line_width = line_width
        self.rot = 0.0
        self.rotV = 0.0
        self.sidesBase = 6.0
        self.scaleK = 1.0
        self.energy = 0.4
        self.countMod = 0
        self.volume = 0.0
        self.offsets: list[dict] = []
        self.offsetsFor = 0
        self._rebuild_offsets(count)

    def _rebuild_offsets(self, n: int):
        self.offsets = []
        for _ in range(n):
            self.offsets.append({
                "ang": random.random() * math.pi * 2,
                "radJitter": random.random() * 2 - 1,
                "phase": random.random() * math.pi * 2,
            })
        self.offsetsFor = n

    def on_guitar_note(self, note: dict, rng: dict):
        v = note["velocity"]
        pitchN = norm_pitch(note["midi"], rng)
        self.energy = min(1.0, self.energy + 0.3 + v * 0.5)
        self.rotV += (pitchN - 0.5) * 6 * (0.5 + v)
        self.sidesBase = 5 + pitchN * 6
        self.scaleK = 0.7 + v * 0.6
        self.countMod = int(pitchN * 6) - 3

    def render(self, canvas: np.ndarray, state: dict):
        dt = max(0.001, min(0.1, state["dt"]))
        # Decay rotV + energy
        self.rotV *= math.pow(0.6, dt)
        self.rot += self.rotV * dt
        self.energy = max(0.15, self.energy * math.pow(0.4, dt))
        # Smoothed volume from active notes (sum velocities)
        vol = 0.0
        n = 0
        for sid, notes in (state.get("active") or {}).items():
            for note in notes:
                vol += note["velocity"]
                n += 1
        instant_vol = min(1.0, vol * 0.15)
        self.volume += (instant_vol - self.volume) * min(1.0, dt * 6)
        # Bar-locked sine waves
        beat = state["beat"]
        bar = beat / 4.0
        tau = math.pi * 2
        wSlow = math.sin(bar * tau / 4)
        wMed = math.sin(bar * tau / 2)
        wFast = math.sin(bar * tau)
        sides = self.sidesBase + wSlow * 4 + wFast * 1.2
        baseScale = min(self.W, self.H) * 0.42 * self.scaleK
        scale = baseScale * (0.55 + 0.45 * wMed)
        target_count = max(3, round(self.count + self.countMod + self.volume * 10))
        if target_count != self.offsetsFor:
            self._rebuild_offsets(target_count)
        cx, cy = self.W * 0.5, self.H * 0.5
        ring_radius = min(self.W, self.H) * 0.22
        line_w = max(1, int(self.line_width * 0.5))
        max_jitter = min(self.W, self.H) * 0.18 * self.volume
        for r_idx in range(self.rings):
            for d in range(target_count):
                off = self.offsets[d]
                ringAng = (d / target_count) * tau + self.rot + r_idx * 0.3
                px = cx + math.cos(ringAng) * ring_radius * (1 + r_idx * 0.5)
                py = cy + math.sin(ringAng) * ring_radius * (1 + r_idx * 0.5)
                jitter = max_jitter * off["radJitter"] * math.sin(bar * tau + off["phase"])
                px += math.cos(off["ang"]) * jitter
                py += math.sin(off["ang"]) * jitter
                a = self.energy * (0.5 + 0.5 * (1 - r_idx / self.rings)
                                   if self.rings > 1 else 1.0)
                shade = int(255 * a)
                if shade < 1:
                    continue
                _draw_fractional_polygon(canvas, px, py, scale, sides,
                                          shade=shade, thickness=line_w)


def _draw_fractional_polygon(canvas, cx, cy, scale, sides, shade=255, thickness=1):
    """Polygon with a fractional `sides` count. The closing segment back
    to the start sweeps as `sides` drifts — the signature breathing/morph
    that gives the field its life. Mirrors drawFractionalPolygon in JS."""
    n = max(3, math.ceil(sides))
    pts = []
    for i in range(n):
        a = (math.pi * 2 * i) / sides
        x = cx + math.cos(a) * scale
        y = cy + math.sin(a) * scale
        pts.append((int(x), int(y)))
    # Close the polygon (back to first point)
    poly = np.array(pts + [pts[0]], dtype=np.int32)
    cv2.polylines(canvas, [poly], isClosed=False,
                  color=(shade, shade, shade), thickness=thickness,
                  lineType=cv2.LINE_AA)


# ──── particle types — one render fn per kind ────────────────────────────

def _draw_particle(canvas, p, alpha: float, state: dict):
    shade = int(255 * alpha)
    if shade <= 0:
        return
    color = (shade, shade, shade)
    lw = int(p.get("lw", 2))
    kind = p["kind"]
    t = p["age"] / max(p["life"], 0.01)
    W = canvas.shape[1]
    H = canvas.shape[0]
    if kind == "ring":
        r = int(p["r"] + (p["maxR"] - p["r"]) * t)
        cv2.circle(canvas, (int(p["x"]), int(p["y"])), max(1, r), color, max(1, lw),
                   lineType=cv2.LINE_AA)
    elif kind == "slash":
        x0, y0 = int(p["x"]), int(p["y"])
        ang = p["angle"]
        x1 = int(x0 + math.cos(ang) * p["len"])
        y1 = int(y0 + math.sin(ang) * p["len"])
        cv2.line(canvas, (x0, y0), (x1, y1), color, max(1, lw), lineType=cv2.LINE_AA)
    elif kind == "tick":
        cv2.line(canvas,
                 (int(p["x"]), int(p["y"] - p["len"]/2)),
                 (int(p["x"]), int(p["y"] + p["len"]/2)),
                 color, max(1, lw), lineType=cv2.LINE_AA)
    elif kind == "ray":
        length = p["len"] * (0.2 + 0.8 * (1 - t))
        x2 = int(p["x"] + math.cos(p["angle"]) * length)
        y2 = int(p["y"] + math.sin(p["angle"]) * length)
        cv2.line(canvas, (int(p["x"]), int(p["y"])), (x2, y2),
                 color, max(1, lw), lineType=cv2.LINE_AA)
    elif kind == "bar":
        y = int(p["y"])
        cv2.line(canvas, (0, y), (W, y), color, max(1, lw), lineType=cv2.LINE_AA)
        for x in range(0, W, 24):
            cv2.line(canvas,
                     (x, int(y - p["h"]/2)), (x, int(y + p["h"]/2)),
                     color, 1, lineType=cv2.LINE_AA)
    elif kind == "arc":
        # Animated arc — center at (x, y), radius r, swept from start_angle by sweep
        start_deg = math.degrees(p["startAngle"] + p["rotV"] * p["age"])
        end_deg = start_deg + math.degrees(p["sweep"])
        cv2.ellipse(canvas, (int(p["x"]), int(p["y"])),
                    (int(p["r"]), int(p["r"])),
                    0, start_deg, end_deg,
                    color, max(1, lw), lineType=cv2.LINE_AA)
    elif kind == "wave":
        # breathing sine wave across width at y, decreasing amp with age
        prev = None
        phase = p["phase"] + state["time"] * 4
        for x in range(0, W + 1, 4):
            y = int(p["y"] + math.sin(x * 0.012 * p["freq"] + phase)
                    * p["amp"] * (1 - t * 0.5))
            if prev is not None:
                cv2.line(canvas, prev, (x, y), color, max(1, lw),
                         lineType=cv2.LINE_AA)
            prev = (x, y)
    elif kind == "hline":
        y = int(p["y"])
        cv2.line(canvas, (0, y), (W, y), color, max(1, lw), lineType=cv2.LINE_AA)
    elif kind == "glitch":
        cv2.rectangle(canvas,
                      (int(p["x"]), int(p["y"])),
                      (int(p["x"] + p["w"]), int(p["y"] + p["h"])),
                      color, 1, lineType=cv2.LINE_AA)


# ──── per-stem spawners — port of the JS spawners object ────────────────

def _spawn_drums(p_list, note, meta, state, W, H, shake_state):
    m = note["midi"]
    v = note["velocity"]
    if m <= 37:
        # KICK → camera shake + big ring + bg flash
        shake_state["shake"] = max(shake_state["shake"], v * 24)
        shake_state["bgFlash"] = max(shake_state["bgFlash"], v * 0.5)
        p_list.append({
            "kind": "ring", "x": W/2, "y": H/2, "r": 20,
            "maxR": min(W, H) * 0.55, "life": 0.7, "age": 0,
            "lw": 3 + v * 3,
        })
    elif m <= 41:
        # SNARE / CLAP → 2 horizontal slashes
        for _ in range(2):
            p_list.append({
                "kind": "slash",
                "x": 0, "y": H * (0.3 + random.random() * 0.4),
                "angle": (random.random() - 0.5) * 0.2,
                "len": W, "life": 0.35, "age": 0, "lw": 2 + v * 4,
            })
    elif m <= 49:
        # HATS → small vertical ticks scattered along horizontal
        n = 5 + int(v * 8)
        for _ in range(n):
            p_list.append({
                "kind": "tick",
                "x": random.random() * W,
                "y": H * 0.2 + random.random() * H * 0.6,
                "len": 6 + v * 12,
                "life": 0.18 + random.random() * 0.15, "age": 0, "lw": 1,
            })
    else:
        # CYMBAL / TOM → radial burst
        cx = W/2 + (random.random() - 0.5) * W * 0.4
        cy = H/2 + (random.random() - 0.5) * H * 0.4
        rays = 8 + int(v * 8)
        for i in range(rays):
            a = (i / rays) * math.pi * 2 + random.random() * 0.3
            p_list.append({
                "kind": "ray", "x": cx, "y": cy, "angle": a,
                "len": 40 + v * 200, "life": 0.5, "age": 0, "lw": 1 + v * 2,
            })


def _spawn_bass(p_list, note, meta, state, W, H):
    v = note["velocity"]
    dur = max(0.2, note["duration"])
    pitchN = norm_pitch(note["midi"], meta["range"])
    y = H - 30 - pitchN * (H * 0.5)
    p_list.append({
        "kind": "bar", "y": y, "h": 20 + v * 40,
        "life": dur, "age": 0, "lw": 2,
    })


def _spawn_synth(p_list, note, meta, state, W, H):
    v = note["velocity"]
    pitchN = norm_pitch(note["midi"], meta["range"])
    cx, cy = W/2, H/2
    r = 80 + pitchN * min(W, H) * 0.45
    p_list.append({
        "kind": "arc", "x": cx, "y": cy, "r": r,
        "startAngle": random.random() * math.pi * 2,
        "sweep": math.pi * (0.3 + v * 1.3),
        "rotV": (random.random() - 0.5) * 2.2,
        "life": max(0.5, note["duration"]), "age": 0, "lw": 2,
    })


def _spawn_vocals(p_list, note, meta, state, W, H):
    pitchN = norm_pitch(note["midi"], meta["range"])
    y = H * (0.85 - pitchN * 0.7)
    p_list.append({
        "kind": "wave", "y": y, "amp": 20 + note["velocity"] * 60,
        "freq": 2 + pitchN * 6,
        "phase": random.random() * math.pi * 2,
        "life": max(0.6, note["duration"]), "age": 0, "lw": 2,
    })


def _spawn_backing(p_list, note, meta, state, W, H):
    pitchN = norm_pitch(note["midi"], meta["range"])
    y = H * (0.1 + pitchN * 0.8)
    p_list.append({
        "kind": "hline", "y": y,
        "life": max(0.5, note["duration"]), "age": 0, "lw": 1,
    })


def _spawn_fx(p_list, note, meta, state, W, H):
    v = note["velocity"]
    n = 8 + int(v * 18)
    for _ in range(n):
        p_list.append({
            "kind": "glitch",
            "x": random.random() * W,
            "y": random.random() * H,
            "w": 8 + random.random() * 80,
            "h": 1 + random.random() * 4,
            "life": 0.15 + random.random() * 0.4, "age": 0, "lw": 1,
        })


# ──── render loop ────────────────────────────────────────────────────────

def render(stems: dict, bpm: float, output: Path, *,
           duration: float, fps: int, width: int, height: int,
           start_sec: float = 0.0, line_width: float = 2.0,
           poly_count: int = 14, poly_rings: int = 1, show_beat: bool = True,
           seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    engine = Engine(stems, bpm)
    polyfield = PolyField(width, height, count=poly_count, rings=poly_rings,
                          line_width=line_width)
    particles: list[dict] = []
    shake_state = {"shake": 0.0, "bgFlash": 0.0}

    # Pre-resolve per-stem meta dicts that include range
    meta = {sid: {"range": s["range"], "color": "#ffffff"} for sid, s in stems.items()}

    # Frame writer
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
        dt = t - last_t if fidx > 0 else 1.0 / fps
        last_t = t

        state = engine.tick(t, dt)

        # Process onsets — each stem fires its spawner
        for sid, ns in state["onsets"].items():
            if not ns:
                continue
            for note in ns:
                if sid == "drums":
                    _spawn_drums(particles, note, meta[sid], state, width, height, shake_state)
                elif sid == "bass":
                    _spawn_bass(particles, note, meta[sid], state, width, height)
                elif sid == "guitar":
                    polyfield.on_guitar_note(note, meta[sid]["range"])
                elif sid == "synth":
                    _spawn_synth(particles, note, meta[sid], state, width, height)
                elif sid == "vocals":
                    _spawn_vocals(particles, note, meta[sid], state, width, height)
                elif sid == "backing":
                    _spawn_backing(particles, note, meta[sid], state, width, height)
                elif sid == "fx":
                    _spawn_fx(particles, note, meta[sid], state, width, height)

        # Build canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if shake_state["bgFlash"] > 0.01:
            v = int(shake_state["bgFlash"] * 40)
            canvas[:] = (v, v, v)
            shake_state["bgFlash"] *= math.pow(0.001, dt)

        # Camera shake — apply translation when drawing if active.
        # We render to a slightly larger canvas, then translate-crop. For a
        # fast first cut, shift particle positions instead:
        shake_x = shake_y = 0
        if shake_state["shake"] > 0.5:
            shake_x = int((random.random() - 0.5) * shake_state["shake"])
            shake_y = int((random.random() - 0.5) * shake_state["shake"])
            shake_state["shake"] *= math.pow(0.001, dt)

        # Translate canvas via affine matrix at the end (cheap)
        # Polygon-field underlay
        polyfield.render(canvas, state)

        # Particles
        live: list[dict] = []
        dt_clamped = max(0.001, min(0.1, dt))
        for p in particles:
            p["age"] += dt_clamped
            a = 1 - p["age"] / max(p["life"], 0.01)
            if a <= 0:
                continue
            _draw_particle(canvas, p, a, state)
            live.append(p)
        particles = live

        # Beat tick markers in corners
        if show_beat:
            beat_phase = state["beat"] - math.floor(state["beat"])
            if beat_phase < 0.08:
                s = int(6 + (1 - beat_phase / 0.08) * 14)
                for x, y in [(20, 20), (width - 20 - s, 20),
                              (20, height - 20 - s),
                              (width - 20 - s, height - 20 - s)]:
                    cv2.rectangle(canvas, (x, y), (x + s, y + s),
                                  (255, 255, 255), -1)

        # Apply camera shake as a final translate
        if shake_x or shake_y:
            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
            canvas = cv2.warpAffine(canvas, M, (width, height),
                                     borderMode=cv2.BORDER_CONSTANT)

        proc.stdin.write(canvas.tobytes())
        if fidx % max(1, n_frames // 20) == 0:
            print(f"  frame {fidx}/{n_frames}  "
                  f"({100 * fidx / n_frames:.0f}%)", file=sys.stderr)

    proc.stdin.close()
    proc.wait()
    print(f"done: {output}  ({output.stat().st_size//1024} KB)", file=sys.stderr)


# ──── entry point ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi-dir", default=None,
                   help="directory holding per-stem .mid files (resolved by glob)")
    # Per-stem explicit overrides
    for sid in TRACK_GLOBS:
        p.add_argument(f"--{sid}", default=None,
                       help=f"explicit path to {sid}.mid (overrides --midi-dir glob)")
    p.add_argument("--output", default="midi2canny.mp4")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds (default = max-end of any stem, capped at 30s)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="trim midi start (offset all notes by -start_sec)")
    p.add_argument("--bpm", type=float, default=128.0,
                   help="tempo for bar/beat-locked sine waves in the polygon field")
    p.add_argument("--line-width", type=float, default=2.0)
    p.add_argument("--poly-count", type=int, default=14)
    p.add_argument("--poly-rings", type=int, default=1)
    p.add_argument("--no-beat-tick", action="store_true",
                   help="hide the corner beat markers")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Resolve per-stem MIDI paths
    stems = {}
    midi_dir = Path(args.midi_dir).resolve() if args.midi_dir else None
    for sid, globs in TRACK_GLOBS.items():
        explicit = getattr(args, sid)
        if explicit:
            path = Path(explicit).resolve()
        elif midi_dir:
            path = None
            # Special-case: lead-vocals patterns can match "Backing Vocals"
            # files, which would steal the slot. Exclude any path containing
            # "Backing"/"backing" when we're resolving the vocals stem.
            for g in globs:
                cands = list(midi_dir.glob(g))
                if sid == "vocals":
                    cands = [p for p in cands
                             if "backing" not in p.name.lower()]
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
              f"{len(notes)} notes, range {rng['min']}-{rng['max']}, "
              f"{end:.2f}s", file=sys.stderr)

    if not stems:
        sys.exit("no MIDI tracks loaded — pass --midi-dir or per-stem flags")

    duration = args.duration
    if duration is None:
        duration = min(30.0, max(s["end"] for s in stems.values()) + 1.0)

    render(stems, args.bpm, Path(args.output).resolve(),
           duration=duration, fps=args.fps,
           width=args.width, height=args.height,
           start_sec=args.start_sec,
           line_width=args.line_width,
           poly_count=args.poly_count, poly_rings=args.poly_rings,
           show_beat=not args.no_beat_tick,
           seed=args.seed)


if __name__ == "__main__":
    main()
