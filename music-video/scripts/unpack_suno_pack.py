#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []  # stdlib only
# ///
"""Unpack a Suno stems+midi zip into `stems/` and `midi/` subfolders.

Suno's standard pack contains paired `<Title> (<Stem>).wav` and
`<Title> (<Stem>).mid` files. This script normalizes the names to
`<stem_lowercase>.wav` / `<stem_lowercase>.mid` and drops them into
`<project>/stems/` and `<project>/midi/`.

Usage:
    unpack_suno_pack.py <zip_path> [--out <project_dir>]

If --out is omitted, uses the zip's parent directory.
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

# Windows cp1252 default stdout chokes on the ✓ glyph below. Reconfigure
# stdout/stderr to utf-8 so the same script runs on POSIX and Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

STEM_RE = re.compile(r"\(([^)]+)\)\.([a-zA-Z0-9]+)$")


def normalize_stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Project dir (default: zip's parent)")
    args = ap.parse_args()

    if not args.zip_path.exists():
        sys.exit(f"zip not found: {args.zip_path}")
    out = (args.out or args.zip_path.parent).resolve()
    stems_dir = out / "stems"
    midi_dir = out / "midi"
    stems_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            m = STEM_RE.search(info.filename)
            if not m:
                print(f"  ? skipping (unrecognized format): {info.filename}",
                      file=sys.stderr)
                continue
            stem_label, ext = m.group(1), m.group(2).lower()
            stem_slug = normalize_stem(stem_label)
            if ext in {"wav", "mp3", "flac"}:
                target = stems_dir / f"{stem_slug}.{ext}"
            elif ext in {"mid", "midi"}:
                target = midi_dir / f"{stem_slug}.mid"
            else:
                print(f"  ? skipping (unknown ext .{ext}): {info.filename}",
                      file=sys.stderr)
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                print(f"  · {target.relative_to(out)} (already present)")
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  ✓ {target.relative_to(out)}  ({info.file_size/1024/1024:.2f} MB)")

    print(f"\nDone — stems in {stems_dir}, midi in {midi_dir}")


if __name__ == "__main__":
    main()
