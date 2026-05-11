#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "librosa",
#   "numpy",
#   "opencv-python-headless",
#   "imageio-ffmpeg",
#   "pyyaml",
# ]
# ///
"""Audio-reactive post-process for an assembled music-video.

Reads <project>/final.mp4 (or any video), reads stems from <project>/stems/
(or paths in --react-spec), detects per-stem onset events via librosa, and
renders a reactive variant via OpenCV with effects pinned to those events.

Effects (per-stem, configurable via --react-spec react.yaml):

  drums    → flash      ← gamma + brightness pulse on each strong drum onset,
                          1-2 frames wide. Use for kick/snare-driven punch.
  guitar   → hue_shift  ← rotate hue toward red on guitar phrase onsets,
                          decaying back to neutral over ~200ms.
  bass     → vignette   ← saturation vignette pulse on low-freq energy,
                          smoother than drum flash.
  vocals   → chroma     ← chromatic aberration on vocal phrase onsets, fades.
  any      → cut        ← split-frame "hard cut" tween (pseudo) — duplicate
                          frame jitter to simulate jump-cut feel on big hits.

Usage:
  audio_react.py <project_dir> [--input final.mp4] [--output final_react.mp4]
                 [--react-spec react.yaml] [--no-audio-passthrough]

The default react.yaml pulled from <project>/react.yaml if present; otherwise
the built-in DEFAULTS apply (drums-flash + guitar-hue, modest intensities).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import yaml

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG = get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"


DEFAULTS = {
    "stems": {
        # Each stem: a relative path glob within <project>/stems/ + an effect
        # block. The first matching wav file in the project's stems/ dir is
        # used. Effect kwargs are passed to the effect's apply() — see EFFECTS.
        #
        # `active_ranges`: optional list of [start_sec, end_sec] song-time
        # spans during which the effect fires. Outside the spans, onset
        # events are silently dropped (no effect applied). Default = always
        # active. Use this to keep flashes/hue concentrated in high-intensity
        # sections (chorus crashes, riff explosions) instead of spraying
        # them across the whole song.
        #
        # `min_strength`: optional 0..1 floor on normalized onset strength —
        # only fire when the onset is stronger than this fraction of the
        # song's loudest hit. Use to filter out ghost notes.
        "drums": {
            "wav": "*Drums*.wav",
            "effect": "flash",
            "onset_delta": 0.5,
            "onset_wait": 4,
            "intensity": 0.18,         # subtle; was 0.40 (eye-burn)
            "duration_frames": 2,
            "min_strength": 0.45,      # only the louder half of detected hits
            # active_ranges: leave unset = always-on; or set per-section
        },
        "guitar": {
            "wav": "*Guitar*.wav",
            "effect": "hue_shift",
            "onset_delta": 0.7,
            "onset_wait": 8,
            "hue_degrees": -10,        # subtle red push; was -25
            "decay_ms": 260,
            "intensity": 0.35,         # was 0.65
            "min_strength": 0.5,
        },
    },
    "passthrough_audio": True,
    "preserve_codec": False,           # True = stream-copy audio (faster); False = re-encode
}


# ── effect registry ──────────────────────────────────────────────────────

class _Effect:
    """Base class. Each effect.apply(frame, t, events) returns the modified
    frame in BGR uint8. `events` is a list of (event_time, weight) tuples
    relevant to the current effect; the apply() decides how to use them.
    """
    name = "base"
    def __init__(self, **kw):
        self.kw = kw
    def apply(self, frame: np.ndarray, t: float, events: list[tuple[float, float]]) -> np.ndarray:
        return frame


class _FlashEffect(_Effect):
    """Brightness pulse on every onset within `duration_frames` of t."""
    name = "flash"
    def apply(self, frame, t, events):
        intensity = float(self.kw.get("intensity", 0.4))
        dur_frames = int(self.kw.get("duration_frames", 2))
        fps = float(self.kw.get("_fps", 24))
        window = dur_frames / fps  # seconds
        # find any onset within (t - window, t]
        boost = 0.0
        for ev_t, weight in events:
            if t - window < ev_t <= t:
                # Linear ramp: max at the onset, fades over `window` seconds
                age = t - ev_t
                ramp = max(0.0, 1.0 - age / window)
                boost = max(boost, ramp * weight * intensity)
        if boost <= 0:
            return frame
        # Brightness lift in YCrCb keeps colour stable
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.int16)
        ycc[..., 0] = np.clip(ycc[..., 0] + int(boost * 255), 0, 255)
        return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


class _HueShiftEffect(_Effect):
    """Rotate the hue ring on each onset, decaying exponentially."""
    name = "hue_shift"
    def apply(self, frame, t, events):
        decay_ms = float(self.kw.get("decay_ms", 220))
        max_deg = float(self.kw.get("hue_degrees", -25))
        intensity = float(self.kw.get("intensity", 0.6))
        # cumulative hue offset from any recent onsets
        offset_deg = 0.0
        for ev_t, weight in events:
            if 0 <= t - ev_t <= 4 * decay_ms / 1000.0:
                age_ms = (t - ev_t) * 1000.0
                ramp = np.exp(-age_ms / decay_ms)
                offset_deg += max_deg * ramp * weight * intensity
        if abs(offset_deg) < 0.5:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        # OpenCV HSV: H in [0, 179]  → 360° / 2 = 180°
        hsv[..., 0] = (hsv[..., 0] + int(offset_deg / 2)) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class _VignetteEffect(_Effect):
    """Saturation vignette pulse on bass energy."""
    name = "vignette"
    def apply(self, frame, t, events):
        decay_ms = float(self.kw.get("decay_ms", 320))
        intensity = float(self.kw.get("intensity", 0.5))
        boost = 0.0
        for ev_t, weight in events:
            if 0 <= t - ev_t <= 4 * decay_ms / 1000.0:
                age_ms = (t - ev_t) * 1000.0
                boost = max(boost, np.exp(-age_ms / decay_ms) * weight * intensity)
        if boost <= 0.01:
            return frame
        h, w = frame.shape[:2]
        # Build a radial vignette mask once per call (cheap; OK for proto)
        cy, cx = h / 2, w / 2
        Y, X = np.ogrid[:h, :w]
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        d_norm = d / max(cx, cy)
        mask = np.clip(1.0 - d_norm * 0.7, 0, 1).astype(np.float32)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + boost * mask * 0.6), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class _ChromaEffect(_Effect):
    """Chromatic aberration on vocal onsets — RGB channels split horizontally."""
    name = "chroma"
    def apply(self, frame, t, events):
        decay_ms = float(self.kw.get("decay_ms", 180))
        max_offset = int(self.kw.get("max_offset_px", 4))
        intensity = float(self.kw.get("intensity", 0.7))
        boost = 0.0
        for ev_t, weight in events:
            if 0 <= t - ev_t <= 4 * decay_ms / 1000.0:
                age_ms = (t - ev_t) * 1000.0
                boost = max(boost, np.exp(-age_ms / decay_ms) * weight * intensity)
        if boost <= 0.05:
            return frame
        offset = max(1, int(max_offset * boost))
        b, g, r = cv2.split(frame)
        # shift R right, B left → magenta/cyan fringe
        r_shift = np.roll(r, offset, axis=1)
        b_shift = np.roll(b, -offset, axis=1)
        return cv2.merge([b_shift, g, r_shift])


EFFECTS: dict[str, type[_Effect]] = {
    "flash": _FlashEffect,
    "hue_shift": _HueShiftEffect,
    "vignette": _VignetteEffect,
    "chroma": _ChromaEffect,
}


# ── onset detection ─────────────────────────────────────────────────────

def _find_stem(stems_dir: Path, glob: str) -> Path | None:
    matches = list(stems_dir.glob(glob))
    return matches[0] if matches else None


def _onsets(wav_path: Path, delta: float, wait: int,
            active_ranges: list[list[float]] | None = None,
            min_strength: float = 0.0) -> list[tuple[float, float]]:
    """Returns list of (onset_time_sec, normalized_strength_in_0_1).

    Filters by:
      - active_ranges: keep onset only if its time falls inside any
        [start, end] span (song-time seconds). None = no filtering.
      - min_strength: drop onsets whose normalized strength is below
        this floor.
    """
    print(f"  loading {wav_path.name}...", file=sys.stderr)
    y, sr = librosa.load(str(wav_path), sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        delta=delta, wait=wait,
    )
    if len(frames) == 0:
        return []
    times = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    strengths = onset_env[frames]
    s_max = float(np.max(strengths)) or 1.0
    weights = (strengths / s_max).astype(float)

    raw_count = len(frames)
    keep_mask = weights >= float(min_strength)
    if active_ranges:
        in_range = np.zeros_like(times, dtype=bool)
        for span in active_ranges:
            if len(span) != 2:
                continue
            lo, hi = float(span[0]), float(span[1])
            in_range |= (times >= lo) & (times <= hi)
        keep_mask &= in_range
    times = times[keep_mask]
    weights = weights[keep_mask]
    print(f"  → {raw_count} raw → {len(times)} after filters "
          f"(min_strength={min_strength}, "
          f"active_ranges={'all' if not active_ranges else len(active_ranges)})",
          file=sys.stderr)
    return list(zip(times.tolist(), weights.tolist()))


# ── main render loop ─────────────────────────────────────────────────────

def react(project: Path, input_video: Path, output_video: Path,
          react_spec: dict, audio_passthrough: bool = True) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        sys.exit(f"could not open {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"input: {w}x{h} @ {fps:.2f}fps  {n_frames} frames  ({n_frames/fps:.2f}s)",
          file=sys.stderr)

    # Compile effects by reading react_spec
    stems_dir = project / "stems"
    effects: list[tuple[_Effect, list[tuple[float, float]]]] = []
    for stem_name, cfg in react_spec.get("stems", {}).items():
        wav = _find_stem(stems_dir, cfg.get("wav", f"*{stem_name}*.wav"))
        if wav is None:
            print(f"  skipping {stem_name}: no wav matching {cfg.get('wav')}",
                  file=sys.stderr)
            continue
        eff_name = cfg.get("effect")
        eff_cls = EFFECTS.get(eff_name)
        if eff_cls is None:
            print(f"  skipping {stem_name}: unknown effect {eff_name}",
                  file=sys.stderr)
            continue
        events = _onsets(wav,
                         delta=float(cfg.get("onset_delta", 0.5)),
                         wait=int(cfg.get("onset_wait", 4)),
                         active_ranges=cfg.get("active_ranges"),
                         min_strength=float(cfg.get("min_strength", 0.0)))
        kw = {k: v for k, v in cfg.items()
              if k not in ("wav", "effect", "onset_delta", "onset_wait",
                           "active_ranges", "min_strength")}
        kw["_fps"] = fps
        effects.append((eff_cls(**kw), events))
        print(f"  registered: {stem_name} → {eff_name}  events={len(events)}",
              file=sys.stderr)

    # Render frame-by-frame to a tmp .mp4 (no audio), then mux
    tmp_video = output_video.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (w, h))

    t0 = time.time()
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        for effect, events in effects:
            frame = effect.apply(frame, t, events)
        writer.write(frame)
        if i % max(1, n_frames // 40) == 0:
            print(f"  frame {i}/{n_frames}  ({i/n_frames*100:5.1f}%)  "
                  f"elapsed {time.time()-t0:.1f}s",
                  file=sys.stderr)
    cap.release()
    writer.release()
    print(f"  rendered {n_frames} frames in {time.time()-t0:.1f}s", file=sys.stderr)

    # Mux original audio onto the rendered frames
    if audio_passthrough:
        cmd = [FFMPEG, "-y",
               "-i", str(tmp_video),
               "-i", str(input_video),
               "-map", "0:v:0",
               "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "18",
               "-c:a", "aac", "-b:a", "192k",
               "-shortest",
               str(output_video)]
    else:
        cmd = [FFMPEG, "-y",
               "-i", str(tmp_video),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               str(output_video)]
    print(f"  muxing → {output_video}", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    tmp_video.unlink(missing_ok=True)
    print(f"done: {output_video}  ({output_video.stat().st_size//1024} KB)",
          file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("project", help="project dir (containing stems/ and final.mp4)")
    p.add_argument("--input", default="final.mp4",
                   help="input video filename (relative to project; default: final.mp4)")
    p.add_argument("--output", default="final_react.mp4",
                   help="output video filename (relative to project)")
    p.add_argument("--react-spec", default=None,
                   help="path to react.yaml (default: <project>/react.yaml or built-in DEFAULTS)")
    p.add_argument("--no-audio-passthrough", action="store_true",
                   help="omit audio mux from input video")
    args = p.parse_args()

    project = Path(args.project).resolve()
    if not project.is_dir():
        sys.exit(f"not a directory: {project}")
    in_v = (project / args.input).resolve()
    out_v = (project / args.output).resolve()
    if not in_v.exists():
        sys.exit(f"input video missing: {in_v}")

    if args.react_spec:
        spec = yaml.safe_load(open(args.react_spec))
    elif (project / "react.yaml").exists():
        spec = yaml.safe_load(open(project / "react.yaml"))
        print(f"using react spec: {project / 'react.yaml'}", file=sys.stderr)
    else:
        spec = DEFAULTS
        print("using built-in DEFAULTS (drums-flash + guitar-hue)", file=sys.stderr)

    react(project, in_v, out_v, spec,
          audio_passthrough=not args.no_audio_passthrough)


if __name__ == "__main__":
    main()
