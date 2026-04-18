#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["imageio-ffmpeg>=0.6"]
# ///
"""Join / trim / extract-frame utilities for video clips via ffmpeg (uv-managed).

Runs under `uv` which auto-installs `imageio-ffmpeg` (bundles a static ffmpeg
binary) on first invocation. No apt or system install needed — works on host
and inside the sandbox identically.

Commands:
  concat  --inputs a.mp4,b.mp4,c.mp4 --output joined.mp4
  concat  --inputs @list.txt --output joined.mp4             # list of paths, one per line
  trim    --input v.mp4 --start 0.0 --duration 3.0 --output v_trim.mp4
  last-frame --input v.mp4 --output last.png                 # extract the final frame
  first-frame --input v.mp4 --output first.png

Concat uses ffmpeg's `concat` demuxer (stream-copy when the inputs share codec /
resolution / framerate — fast, no re-encode). Falls back to re-encode when the
copy attempt fails (different codecs / sizes).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _ffmpeg() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stderr or p.stdout)[-2000:]


def _resolve_inputs(arg: str) -> list[Path]:
    if arg.startswith("@"):
        lines = Path(arg[1:]).read_text().splitlines()
        paths = [Path(line.strip()) for line in lines if line.strip()]
    else:
        paths = [Path(p.strip()) for p in arg.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            sys.exit(f"input not found: {p}")
    return paths


def cmd_concat(args) -> None:
    inputs = _resolve_inputs(args.inputs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    # Try stream-copy concat first (fast).
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in inputs:
            f.write(f"file '{p.resolve()}'\n")
        listfile = f.name
    copy_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                "-c", "copy", str(out)]
    rc, log = _run(copy_cmd)
    if rc == 0:
        print(f"ok: {out} (stream-copy, {len(inputs)} clips)")
        Path(listfile).unlink(missing_ok=True)
        return

    # Fallback: re-encode to a common target.
    print(f"stream-copy failed (mismatched codecs/size); re-encoding…", file=sys.stderr)
    reenc_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                 "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(out)]
    rc, log = _run(reenc_cmd)
    Path(listfile).unlink(missing_ok=True)
    if rc != 0:
        sys.exit(f"concat failed:\n{log}")
    print(f"ok: {out} (re-encoded, {len(inputs)} clips)")


def cmd_trim(args) -> None:
    ffmpeg = _ffmpeg()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-ss", str(args.start)]
    if args.duration:
        cmd += ["-t", str(args.duration)]
    cmd += ["-i", args.input, "-c", "copy", str(out)]
    rc, log = _run(cmd)
    if rc != 0:
        # fallback with re-encode if copy fails (non-keyframe cut)
        cmd2 = [ffmpeg, "-y", "-ss", str(args.start)]
        if args.duration: cmd2 += ["-t", str(args.duration)]
        cmd2 += ["-i", args.input, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(out)]
        rc, log = _run(cmd2)
        if rc != 0:
            sys.exit(f"trim failed:\n{log}")
    print(f"ok: {out}")


def cmd_frame(args, which: str) -> None:
    ffmpeg = _ffmpeg()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if which == "first":
        cmd = [ffmpeg, "-y", "-i", args.input, "-frames:v", "1", "-q:v", "2", str(out)]
    else:  # last
        cmd = [ffmpeg, "-y", "-sseof", "-1", "-i", args.input, "-update", "1",
               "-frames:v", "1", "-q:v", "2", str(out)]
    rc, log = _run(cmd)
    if rc != 0:
        sys.exit(f"frame extract failed:\n{log}")
    print(f"ok: {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="video_join")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("concat", help="concatenate clips (stream-copy when possible)")
    c.add_argument("--inputs", required=True,
                   help="comma-separated paths OR @/path/to/list.txt")
    c.add_argument("--output", required=True)

    t = sub.add_parser("trim", help="trim a clip")
    t.add_argument("--input", required=True)
    t.add_argument("--start", type=float, default=0.0, help="seconds")
    t.add_argument("--duration", type=float, help="seconds (omit for until-end)")
    t.add_argument("--output", required=True)

    ff = sub.add_parser("first-frame", help="extract the first frame as PNG/JPG")
    ff.add_argument("--input", required=True)
    ff.add_argument("--output", required=True)

    lf = sub.add_parser("last-frame", help="extract the last frame as PNG/JPG")
    lf.add_argument("--input", required=True)
    lf.add_argument("--output", required=True)

    args = p.parse_args()
    if args.cmd == "concat":
        cmd_concat(args)
    elif args.cmd == "trim":
        cmd_trim(args)
    elif args.cmd == "first-frame":
        cmd_frame(args, "first")
    elif args.cmd == "last-frame":
        cmd_frame(args, "last")


if __name__ == "__main__":
    main()
