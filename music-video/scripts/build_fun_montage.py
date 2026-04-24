#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["imageio-ffmpeg>=0.6"]
# ///
"""Build the "fun post-chorus" rapid-cut montage for snakebird-meets.

Premise: at the song's 3-bar post-chorus break (68.13 – 76.31s @ 88.1 BPM),
show one still from each of the 12 character sheets for exactly 1 beat
(0.681s), with a subtle ken-burns zoom to keep the frame alive, hard-cut
between them, over the corresponding slice of the song.

The generated mp4 replaces whatever ltx2 would have produced for scene S8.
`music_video.py scenes` skips a scene whose mp4 already exists, so drop
this into `<project>/scenes/008-fun_montage.mp4` and the rest of the
pipeline picks it up as-is.

Usage:
    build_fun_montage.py <project_dir>
        [--bpm 88.1] [--start-sec 68.13] [--beats 12] [--fps 24]
        [--width 448] [--height 832]
        [--order snakebird,turtle,raven,badger,owl,cat,fox,hare,deer,rabbit,fish,bear]

Outputs:
    <project>/scenes/008-fun_montage.mp4
    <project>/song_slices/008-fun_montage.mp3  (matching audio slice)
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

try:
    import imageio_ffmpeg as iim
except ImportError:
    sys.exit("imageio-ffmpeg missing. Re-run via `uv run --script` (auto-installs).")

FFMPEG = iim.get_ffmpeg_exe()

DEFAULT_ORDER = [
    "snakebird", "turtle", "raven", "badger", "owl", "cat",
    "fox", "hare", "deer", "rabbit", "fish", "bear",
]


def pick_sheet(sheets_dir: Path, name: str) -> Path:
    """First existing sheet for a character (prefer 00001, else lowest number,
    else `<name>_canon.png`)."""
    canon = sheets_dir / f"sheet_{name}_canon.png"
    if canon.exists():
        return canon
    candidates = sorted(sheets_dir.glob(f"sheet_{name}_*.png"))
    if not candidates:
        sys.exit(f"no sheet found for character '{name}' in {sheets_dir}")
    # Prefer _00001_ when present
    for c in candidates:
        if "_00001_" in c.name:
            return c
    return candidates[0]


def slice_audio(song_mp3: Path, start_sec: float, duration_sec: float,
                out_mp3: Path) -> None:
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
        "-i", str(song_mp3),
        "-c:a", "libmp3lame", "-q:a", "2",
        str(out_mp3),
    ], check=True)


def build_cut(sheet: Path, out_mp4: Path, beat_sec: float,
              fps: int, width: int, height: int, zoom_dir: str) -> None:
    """Render a single-character 1-beat cut with a gentle ken-burns zoom.

    zoom_dir: "in" or "out" — alternating between cuts keeps the rapid-fire
    sequence from feeling static."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(2, int(round(beat_sec * fps)))
    # Pre-upscale source 4× so zoompan's per-frame step is smooth; then cover
    # the target (width x height) preserving aspect and filling with centered
    # crop (scale=w:h:force_original_aspect_ratio=increase,crop).
    if zoom_dir == "in":
        z_expr = f"zoom+{(1.12 - 1.0) / total_frames:.6f}"
    else:
        z_expr = f"if(eq(on,0),1.12,zoom-{(1.12 - 1.0) / total_frames:.6f})"
    vf = (
        f"scale=iw*4:ih*4,"
        f"zoompan=z='{z_expr}':d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-t", f"{beat_sec:.3f}", "-i", str(sheet),
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-r", str(fps),
        str(out_mp4),
    ], check=True)


def concat_cuts(cut_mp4s: list[Path], audio_mp3: Path, out_mp4: Path) -> None:
    """Concat the per-beat cuts with no crossfade and mux the audio slice."""
    # Use ffmpeg's concat demuxer
    list_file = out_mp4.parent / "_concat_list.txt"
    list_file.write_text("".join(f"file '{c.resolve()}'\n" for c in cut_mp4s))
    # Two-step: concat video -> mux audio
    tmp_silent = out_mp4.with_suffix(".silent.mp4")
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(tmp_silent),
    ], check=True)
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-i", str(tmp_silent), "-i", str(audio_mp3),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_mp4),
    ], check=True)
    tmp_silent.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--bpm", type=float, default=88.1)
    ap.add_argument("--start-sec", type=float, default=68.13)
    ap.add_argument("--beats", type=int, default=12)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--order", default=",".join(DEFAULT_ORDER),
                    help="comma-separated character names, one per beat")
    ap.add_argument("--song", default="song.mp3",
                    help="song mp3 path relative to project (default: song.mp3)")
    ap.add_argument("--out", default="scenes/008-fun_montage.mp4",
                    help="output mp4 path relative to project")
    args = ap.parse_args()

    project = args.project.resolve()
    sheets_dir = project / "sheets"
    song = project / args.song
    out_mp4 = project / args.out
    if not sheets_dir.is_dir():
        sys.exit(f"no sheets/ dir in {project}")
    if not song.exists():
        sys.exit(f"song not found: {song}")

    order = [n.strip() for n in args.order.split(",") if n.strip()]
    if len(order) != args.beats:
        sys.exit(f"order has {len(order)} names but --beats={args.beats}")

    beat_sec = 60.0 / args.bpm
    duration = beat_sec * args.beats
    print(f"montage: {args.beats} beats × {beat_sec:.3f}s = {duration:.3f}s "
          f"@ {args.bpm} BPM, {args.fps}fps")

    # 1. slice audio
    audio_slice = project / "song_slices" / "008-fun_montage.mp3"
    slice_audio(song, args.start_sec, duration, audio_slice)
    print(f"  audio  → {audio_slice.relative_to(project)}")

    # 2. render per-character cuts
    cuts_dir = project / "scenes" / "_fun_montage_cuts"
    cuts_dir.mkdir(parents=True, exist_ok=True)
    cut_mp4s: list[Path] = []
    for i, name in enumerate(order):
        sheet = pick_sheet(sheets_dir, name)
        cut = cuts_dir / f"{i:02d}-{name}.mp4"
        zoom_dir = "in" if i % 2 == 0 else "out"
        build_cut(sheet, cut, beat_sec, args.fps, args.width, args.height, zoom_dir)
        cut_mp4s.append(cut)
        print(f"  cut {i+1:>2} [{name}]  ← {sheet.name}  ({zoom_dir} zoom)")

    # 3. concat + mux
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat_cuts(cut_mp4s, audio_slice, out_mp4)
    print(f"\nmontage ready: {out_mp4.relative_to(project)}")
    print(f"  (music_video.py scenes will skip ltx2 for S8 — picks up this mp4)")


if __name__ == "__main__":
    main()
