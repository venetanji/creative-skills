#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "librosa>=0.10",
#   "numpy",
#   "imageio-ffmpeg",
#   "opencv-python-headless",
# ]
# ///
"""Expanding-circles-from-center cond video, audio-onset timed.

Each onset (typically a drum kick from the drums stem) spawns a circle
that expands radially from the frame center over a fixed lifetime,
fading out as it grows. Output is grayscale wireframe (canny-friendly)
intended as the IC-LoRA Union-Control reference video for an ia2v /
transition render — pairs well with IC-LoRA reference_strength ~0.5-0.7
where the rings drive structural rhythm without overriding the anchor.

Why this exists: midi2canny.py's polyfield is good for a sustained
fractal field with subtle rotation, but at scene boundaries (drum
dropout → drum slam, transitions, climactic pulses) we want a
ring-shockwave-from-center effect that's audio-onset-locked instead
of MIDI-driven. Audio onsets are sample-accurate; MIDI may drift.

Usage:
  expanding_circles.py --audio drums.wav --start-sec 128 --duration 4 \\
      --output cond.mp4 --width 448 --height 768
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg as iio
import librosa
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--audio", required=True,
                   help="Source audio (typically the drums stem). The "
                        "script runs librosa.onset.onset_detect on this "
                        "to find ring-spawn times.")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--ring-life", type=float, default=0.8,
                   help="seconds each ring expands+fades (default 0.8)")
    p.add_argument("--max-radius", type=float, default=None,
                   help="max ring radius in pixels (default = "
                        "max(W,H) * 0.95 — rings sweep past the frame edge)")
    p.add_argument("--line-width", type=int, default=4,
                   help="ring line width in pixels (default 4)")
    p.add_argument("--onset-delta", type=float, default=0.07,
                   help="librosa onset threshold (default 0.07)")
    p.add_argument("--easing", choices=["linear", "ease-out"], default="ease-out",
                   help="radius growth curve (default 'ease-out' — fast "
                        "expansion then deceleration, reads as an impact "
                        "shockwave)")
    args = p.parse_args()

    y, sr = librosa.load(args.audio, sr=22050, mono=True,
                          offset=args.start_sec, duration=args.duration)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, delta=args.onset_delta, hop_length=512, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr,
                                           hop_length=512)
    # Also extract onset strengths so we can scale velocity-like.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    strengths = []
    for f in onset_frames:
        strengths.append(float(onset_env[min(f, len(onset_env) - 1)]))
    if strengths:
        s_max = max(strengths)
        strengths = [s / s_max for s in strengths] if s_max > 0 else strengths
    print(f"detected {len(onset_times)} onsets in [{args.start_sec:.2f}, "
          f"{args.start_sec + args.duration:.2f}]; "
          f"first 5: {[round(t,3) for t in onset_times[:5]]}",
          file=sys.stderr)

    W, H = args.width, args.height
    cx, cy = W // 2, H // 2
    max_r = args.max_radius or (max(W, H) * 0.95)
    n_frames = int(args.duration * args.fps)

    out_path = Path(args.output).resolve()
    ff = iio.get_ffmpeg_exe()
    proc = subprocess.Popen([
        ff, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(args.fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", str(out_path)
    ], stdin=subprocess.PIPE)

    for fi in range(n_frames):
        t = fi / args.fps
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for ot, strength in zip(onset_times, strengths):
            age = t - ot
            if age < 0 or age > args.ring_life:
                continue
            p = age / args.ring_life  # 0 → 1
            if args.easing == "ease-out":
                # 1 - (1-p)^2 → fast expansion then deceleration.
                eased_p = 1.0 - (1.0 - p) ** 2
            else:
                eased_p = p
            radius = int(max_r * eased_p)
            alpha = (1.0 - p) * (0.5 + 0.5 * strength)
            shade = int(np.clip(255 * alpha, 0, 255))
            lw = max(1, int(args.line_width * (0.7 + 0.6 * strength)))
            if radius > 2:
                cv2.circle(canvas, (cx, cy), radius, (shade, shade, shade),
                           lw, lineType=cv2.LINE_AA)
        proc.stdin.write(canvas.tobytes())
        if fi % 24 == 0:
            print(f"  frame {fi}/{n_frames}", file=sys.stderr)

    proc.stdin.close()
    proc.wait()
    print(f"done: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
