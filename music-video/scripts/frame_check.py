#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["imageio-ffmpeg>=0.6", "Pillow>=10.0", "numpy>=1.24"]
# ///
"""Frame-similarity checker for scene-to-scene stitching.

Extracts the LAST frame of scene N and the FIRST frame of scene N+1 from their
MP4s, computes pixel MSE and a mean-abs diff, and prints a per-boundary report.
Also writes side-by-side comparison PNGs so you can eyeball the cut.

Usage:
  frame_check.py <project_dir>              # use scenes/NNN-*.mp4 in order
  frame_check.py <project_dir> --save       # also write PNG pairs to stitch_check/

High scores (MSE > ~2000, mean diff > ~30) = jarring cuts. Low scores = smooth.
"""
from __future__ import annotations
import argparse, subprocess, sys, tempfile, re
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image
import numpy as np


def ffmpeg() -> str:
    return get_ffmpeg_exe()


def extract_frame(mp4: Path, position: str, out: Path) -> None:
    """position: 'first' or 'last'."""
    if position == "first":
        cmd = [ffmpeg(), "-y", "-i", str(mp4), "-frames:v", "1", "-q:v", "2", str(out)]
    else:
        cmd = [ffmpeg(), "-y", "-sseof", "-1", "-i", str(mp4), "-update", "1",
               "-frames:v", "1", "-q:v", "2", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        sys.exit(f"ffmpeg frame extract failed for {mp4} ({position}):\n{r.stderr[-500:]}")


def img_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.float32)


def score(a: np.ndarray, b: np.ndarray) -> dict:
    # Resize b to match a if dimensions differ (they shouldn't, but be safe).
    if a.shape != b.shape:
        from PIL import Image as _I
        h, w = a.shape[:2]
        bi = _I.fromarray(b.astype(np.uint8))
        bi = bi.resize((w, h), _I.LANCZOS)
        b = np.asarray(bi, dtype=np.float32)
    diff = a - b
    mse = float(np.mean(diff * diff))
    mad = float(np.mean(np.abs(diff)))
    # Normalized cross-correlation (1.0 = identical, lower = more different).
    a_mean, b_mean = a.mean(), b.mean()
    num = np.sum((a - a_mean) * (b - b_mean))
    den = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2)) + 1e-9
    ncc = float(num / den)
    return {"mse": mse, "mad": mad, "ncc": ncc}


def verdict(s: dict) -> str:
    if s["mad"] < 12:   return "seamless"
    if s["mad"] < 25:   return "smooth"
    if s["mad"] < 45:   return "visible cut"
    return "jarring cut"


def main() -> None:
    p = argparse.ArgumentParser(prog="frame_check")
    p.add_argument("project", help="music-video project dir (contains scenes/)")
    p.add_argument("--save", action="store_true", help="write PNG pairs to stitch_check/")
    args = p.parse_args()

    project = Path(args.project).resolve()
    scenes_dir = project / "scenes"
    mp4s = sorted(p for p in scenes_dir.glob("[0-9][0-9][0-9]-*.mp4") if "-last" not in p.name)
    if len(mp4s) < 2:
        sys.exit(f"need at least 2 scene MP4s; found {len(mp4s)}")

    out_dir = project / "stitch_check"
    if args.save:
        out_dir.mkdir(exist_ok=True)

    print(f"{'boundary':35}  {'mad':>6} {'mse':>8} {'ncc':>6}  verdict")
    print("-" * 78)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i in range(len(mp4s) - 1):
            a, b = mp4s[i], mp4s[i + 1]
            a_last = tmp / f"{a.stem}-last.png"
            b_first = tmp / f"{b.stem}-first.png"
            extract_frame(a, "last", a_last)
            extract_frame(b, "first", b_first)
            s = score(img_array(a_last), img_array(b_first))
            v = verdict(s)
            label = f"{a.stem} → {b.stem}"[:35]
            print(f"{label:35}  {s['mad']:>6.2f} {s['mse']:>8.1f} {s['ncc']:>6.3f}  {v}")
            if args.save:
                # Save a side-by-side comparison PNG
                ai = Image.open(a_last); bi = Image.open(b_first).resize(ai.size)
                combo = Image.new("RGB", (ai.width * 2 + 4, ai.height), (32, 32, 32))
                combo.paste(ai, (0, 0))
                combo.paste(bi, (ai.width + 4, 0))
                combo.save(out_dir / f"boundary_{i+1:02d}.png")


if __name__ == "__main__":
    main()
