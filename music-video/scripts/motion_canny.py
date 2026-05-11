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
"""Generate a canny-edge motion video from MIDI for LTX IC-LoRA Canny-Control
(or Union-Control). Each MIDI note becomes an animated edge-only shape
(circle / pulse-ring / fading dot) on a black canvas. The IC-LoRA reads
those edges as motion guides and the prompt-described thing ("balloons",
"fish", "petals", "sparks") inherits the per-note motion.

Note → screen mapping:
  pitch     → vertical position (high pitch = top, low = bottom)
              + deterministic horizontal scatter from a hash of the pitch
  velocity  → base size (loud notes draw bigger shapes)
  duration  → lifetime (the shape fades / persists for the note's length)
  shape     → see --shape

Usage:
  motion_canny.py --midi stems/Stable\\ Altitude\\ \\(Drums\\).mid \\
      --output canny-drums.mp4 --duration 6 --fps 24 \\
      --width 432 --height 768 --shape pulse

Then feed the output as the IC-LoRA reference video:
  comfy_graph.py ia2v --image anchor.png --audio slice.mp3 --prompt "balloons" \\
      --ic_loras "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors:1.0" \\
      --ic_lora_reference_video canny-drums.mp4 \\
      --width 432 --height 768 --seconds 6 --fast

The Union-Control LoRA accepts canny-edge maps natively; if you have the
dedicated Canny-Control LoRA installed instead, swap the --ic_loras name.
"""
from __future__ import annotations
import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import mido
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe


def parse_notes(midi_path: Path, start_sec: float = 0.0) -> list[tuple[float, int, int, float]]:
    """Walk a MIDI file and return [(on_time_sec, pitch, velocity, duration_sec), ...].

    `mido.MidiFile.__iter__` yields messages with their delta-times accumulated
    in real seconds, which is what we want for direct frame-time mapping.
    Returns notes whose on_time >= start_sec, with on_time shifted by
    -start_sec so frame 0 of the canvas = the chosen song-time origin.
    """
    mid = mido.MidiFile(str(midi_path))
    notes: list[tuple[float, int, int, float]] = []
    abs_t = 0.0
    pending: dict[int, tuple[float, int]] = {}  # pitch -> (on_t, velocity)
    for msg in mid:
        abs_t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending[msg.note] = (abs_t, msg.velocity)
        elif (msg.type == "note_off"
              or (msg.type == "note_on" and msg.velocity == 0)) and msg.note in pending:
            on_t, vel = pending.pop(msg.note)
            dur = max(0.05, abs_t - on_t)
            if on_t >= start_sec:
                notes.append((on_t - start_sec, msg.note, vel, dur))
    return notes


def _draw_circle(canvas, cx, cy, size, age, dur, shade=255, thickness=2):
    """Edge-only circle that fades out toward the end of the note's lifetime."""
    fade = max(0.0, 1.0 - age / max(dur, 0.05))
    s = int(shade * fade)
    if s > 0:
        cv2.circle(canvas, (cx, cy), int(size), (s, s, s), int(thickness),
                   lineType=cv2.LINE_AA)


def _draw_pulse(canvas, cx, cy, size, age, dur, shade=255):
    """Expanding ring on each onset — radius grows with age, thins out."""
    radius = int(size * (1 + age * 4))
    thickness = max(1, 3 - int(age * 3))
    fade = max(0.0, 1.0 - age / max(dur * 1.5, 0.05))
    s = int(shade * fade)
    if s > 0:
        cv2.circle(canvas, (cx, cy), radius, (s, s, s), thickness,
                   lineType=cv2.LINE_AA)


def _draw_dot(canvas, cx, cy, size, age, dur, shade=255):
    """Filled disc that fades — for shape='dot' (the model sees a moving dot)."""
    fade = max(0.0, 1.0 - age / max(dur, 0.05))
    s = int(shade * fade)
    if s > 0:
        cv2.circle(canvas, (cx, cy), int(size), (s, s, s), -1,
                   lineType=cv2.LINE_AA)


SHAPES = {"circle": _draw_circle, "pulse": _draw_pulse, "dot": _draw_dot,
          "ring": _draw_circle}  # ring = circle without fade — alias


def render(midi_path: Path, output: Path, *,
           duration: float | None, fps: int, width: int, height: int,
           shape: str, start_sec: float, min_pitch: int, max_pitch: int,
           x_jitter: float = 0.30, base_size: int = 18, scale_velocity: bool = True) -> None:
    notes = parse_notes(midi_path, start_sec=start_sec)
    if not notes:
        sys.exit(f"no notes found in {midi_path} after start_sec={start_sec}")
    end_t = max(t + d for t, _, _, d in notes) + 1.0
    if duration is None:
        duration = min(end_t, 30.0)
    n_frames = int(duration * fps)
    print(f"  {len(notes)} notes; rendering {n_frames} frames "
          f"({duration:.2f}s @ {fps}fps)", file=sys.stderr)

    drawer = SHAPES[shape]
    pitch_range = max(1, max_pitch - min_pitch)

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

    for fidx in range(n_frames):
        t = fidx / fps
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for on_t, pitch, vel, dur in notes:
            age = t - on_t
            # 'pulse' rings keep expanding for ~1.5x note duration; others gate
            # at note duration exactly. Keep notes alive a hair past the tail
            # so the IC-LoRA sees the shape decay rather than vanish.
            window = dur * (1.5 if shape == "pulse" else 1.0)
            if age < 0 or age > window:
                continue
            yt = (pitch - min_pitch) / pitch_range
            yt = max(0.0, min(1.0, yt))
            cy = int((1 - yt) * (height - 40) + 20)
            # Deterministic horizontal scatter from a hash of pitch — keeps
            # the same MIDI pitch always at the same X position so the model
            # sees a coherent spatial layout per stem.
            jitter = math.sin(pitch * 0.7) * x_jitter * width
            cx = int(width / 2 + jitter)
            cx = max(20, min(width - 20, cx))
            size = base_size + (vel // 4 if scale_velocity else 0)
            drawer(canvas, cx, cy, size, age, dur)
        proc.stdin.write(canvas.tobytes())
        if fidx % max(1, n_frames // 20) == 0:
            print(f"  frame {fidx}/{n_frames}  "
                  f"({100 * fidx / n_frames:.0f}%)", file=sys.stderr)

    proc.stdin.close()
    proc.wait()
    print(f"done: {output}  ({output.stat().st_size//1024} KB)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi", required=True, help="input MIDI track")
    p.add_argument("--output", default="canny-motion.mp4")
    p.add_argument("--duration", type=float, default=None,
                   help="seconds; default = full midi (capped at 30s)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=432)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--shape", default="pulse",
                   choices=list(SHAPES))
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="trim midi start (offset note times by -start_sec)")
    p.add_argument("--min-pitch", type=int, default=24)
    p.add_argument("--max-pitch", type=int, default=108)
    p.add_argument("--x-jitter", type=float, default=0.30,
                   help="horizontal scatter amplitude as fraction of width")
    p.add_argument("--base-size", type=int, default=18)
    p.add_argument("--no-velocity-scale", action="store_true",
                   help="don't scale shape size by note velocity")
    args = p.parse_args()

    render(Path(args.midi).resolve(), Path(args.output).resolve(),
           duration=args.duration, fps=args.fps,
           width=args.width, height=args.height,
           shape=args.shape, start_sec=args.start_sec,
           min_pitch=args.min_pitch, max_pitch=args.max_pitch,
           x_jitter=args.x_jitter, base_size=args.base_size,
           scale_velocity=not args.no_velocity_scale)


if __name__ == "__main__":
    main()
