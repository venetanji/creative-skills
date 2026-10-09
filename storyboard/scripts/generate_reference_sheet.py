#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]   # for the tiler; flux2 calls go via comfy_graph subprocess
# ///
"""generate_reference_sheet.py — build a project "ingredients" reference sheet
from a recurring character + named props/location, for the LTX-2.3 ingredients
IC-LoRA (comfyui `ingredients` command / `ltx2_ingredients_to_video`).

What it does:
  1. Generates clean element PANELS via flux2 (through the comfyui skill's
     comfy_graph.py):
       - character turnaround: `multiprompt` (1 ref + N angle prompts) when a
         --character image is given, else a single `t2i` from --character-desc;
       - each prop: `t2i` from its description (or `i2i` / pass-through if the
         value is an image path);
       - location: `t2i` from its description.
     Every panel is forced onto a solid black background with no text (the
     ingredients LoRA wants clean panels, black bg, no captions).
  2. Tiles the panels onto ONE sheet at the project render WxH via the comfyui
     skill's make_reference_sheet.build_sheet (contain-fit, auto-grid).

The sheet is then fed to `comfy_graph.py ingredients --sheet <sheet> --prompt ...`
(or wired per-scene via music-video / drama-video `mode: ingredients`).

Why WxH matters: the ingredients graph stretches the sheet to the render size, so
emit the sheet at the project's render aspect (default 768x448 trained bucket;
pass --width/--height for portrait etc.).

Exit codes: 0 success, 1 bad args, 2 a flux2 panel render failed.

Usage:
    generate_reference_sheet.py --out project_sheet.png \
        --character marco.png \
        --character-desc "Marco, early-30s man, tan skin, short black hair, stubble, lean build" \
        --angle "front-facing head-and-shoulders portrait, neutral expression" \
        --angle "full-body three-quarter view, neutral stance" \
        --prop "trident=ornate golden three-pronged trident" \
        --prop "soda=a red soda can" \
        --location "sunny tropical beach, turquoise water, palm trees, bright daylight" \
        --width 768 --height 448
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SCRIPT_DIR = Path(__file__).resolve().parent

# Force every generated panel onto a clean black background with no text — the
# ingredients LoRA is trained on sheets of isolated elements on black, no
# captions. Appended to every flux2 panel prompt.
PANEL_TAIL = (", isolated subject centered in frame, full subject visible, "
              "solid pure black background, studio reference shot, sharp focus, "
              "no text, no caption, no watermark, no logo, no border")

DEFAULT_ANGLES = [
    "front-facing head-and-shoulders portrait, neutral expression, looking at camera",
    "full-body three-quarter view, neutral stance",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _find_comfy_graph(cli_override: str | None) -> Path:
    if cli_override:
        p = Path(cli_override)
        if p.exists():
            return p
        sys.exit(f"--comfy-graph path not found: {cli_override}")
    candidates = [
        Path("/home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py"),
        Path.home() / ".openclaw/skills/comfyui/scripts/comfy_graph.py",
        SCRIPT_DIR.parent.parent / "comfyui/scripts/comfy_graph.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit("could not locate comfy_graph.py — pass --comfy-graph <path>")


def _load_build_sheet(comfy_graph: Path):
    """Import build_sheet from the comfyui skill's make_reference_sheet.py
    (sibling of comfy_graph.py)."""
    ms = comfy_graph.parent / "make_reference_sheet.py"
    if not ms.exists():
        sys.exit(f"make_reference_sheet.py not found next to comfy_graph: {ms}")
    spec = importlib.util.spec_from_file_location("make_reference_sheet", ms)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_sheet


def _run_comfy(comfy: Path, sub: str, prefix: str, work_dir: Path,
               extra: list[str], steps: int) -> list[Path]:
    """Run a comfy_graph flux2 subcommand and return the PNGs it produced
    (globbed by prefix, newest set)."""
    before = {p: p.stat().st_mtime for p in work_dir.glob(f"{prefix}_*.png")}
    cmd = ["python3", str(comfy), sub,
           "--prefix", prefix,
           "--output-dir", str(work_dir),
           "--steps", str(steps),
           "--timeout", "600"] + extra
    print(f"[sheet] flux2 {sub} -> {prefix}…", file=sys.stderr)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(2)
    produced = sorted(
        (p for p in work_dir.glob(f"{prefix}_*.png")
         if p.stat().st_mtime > before.get(p, 0)),
        key=lambda p: (p.stat().st_mtime, p.name))
    if not produced:
        sys.exit(f"comfy produced no {prefix}_*.png in {work_dir}")
    return produced


def _is_image_path(val: str) -> bool:
    p = Path(val)
    return p.suffix.lower() in IMAGE_EXTS and p.exists()


def _parse_props(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"--prop must be 'name=description_or_path', got {p!r}")
        name, val = p.split("=", 1)
        out.append((name.strip(), val.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--character", default=None,
                    help="character reference image (turnaround via flux2 multiprompt)")
    ap.add_argument("--character-desc", default=None,
                    help="character description (prepended to angle prompts; "
                         "required if --character is omitted)")
    ap.add_argument("--angle", action="append", default=[],
                    help="character angle prompt, repeatable (defaults to a "
                         "portrait + a full-body view)")
    ap.add_argument("--prop", action="append", default=[],
                    help="name=description (t2i) or name=path.png (use image), repeatable")
    ap.add_argument("--location", default=None, help="location description (t2i panel)")
    ap.add_argument("--width", type=int, default=768, help="sheet = render width")
    ap.add_argument("--height", type=int, default=448, help="sheet = render height")
    ap.add_argument("--style", default=None,
                    help="optional style tail appended to every panel prompt "
                         "(free text, e.g. 'cinematic photoreal')")
    ap.add_argument("--steps", type=int, default=8, help="flux2 steps per panel")
    ap.add_argument("--grid", default=None, help="force sheet grid RxC")
    ap.add_argument("--pad", type=int, default=10, help="tiler inner cell padding px")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where panels are rendered (default <out>_panels/)")
    ap.add_argument("--comfy-graph", default=None, help="override comfy_graph.py path")
    ap.add_argument("--force", action="store_true", help="re-render even if --out exists")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(str(args.out))
        return
    if not args.character and not args.character_desc:
        sys.exit("need --character (image) or --character-desc (text)")

    comfy = _find_comfy_graph(args.comfy_graph)
    build_sheet = _load_build_sheet(comfy)
    work_dir = (args.work_dir or args.out.with_name(args.out.stem + "_panels"))
    work_dir.mkdir(parents=True, exist_ok=True)

    style_tail = f", {args.style.strip().lstrip(', ')}" if args.style else ""
    tail = style_tail + PANEL_TAIL
    panels: list[Path] = []

    # ── 1. character ────────────────────────────────────────────────
    angles = args.angle or DEFAULT_ANGLES
    if args.character:
        if not Path(args.character).exists():
            sys.exit(f"--character image not found: {args.character}")
        # multiprompt: 1 ref + N angle prompts -> N consistent panels.
        prompts = "\n".join(angles)
        prepend = (args.character_desc.strip() + ", ") if args.character_desc else ""
        panels += _run_comfy(
            comfy, "multiprompt", "sheet_char", work_dir,
            ["--image", args.character, "--prompts", prompts,
             "--prepend", prepend, "--append", tail],
            steps=args.steps)
    else:
        # No ref image — one t2i identity panel from the description.
        panels += _run_comfy(
            comfy, "t2i", "sheet_char", work_dir,
            ["--prompt", args.character_desc.strip() + ", " + angles[0] + tail,
             "--width", "768", "--height", "768"],
            steps=args.steps)

    # ── 2. props ────────────────────────────────────────────────────
    for i, (name, val) in enumerate(_parse_props(args.prop)):
        if _is_image_path(val):
            panels.append(Path(val))                 # use the provided image as-is
            continue
        panels += _run_comfy(
            comfy, "t2i", f"sheet_prop{i:02d}", work_dir,
            ["--prompt", f"{val}{tail}", "--width", "768", "--height", "768"],
            steps=args.steps)

    # ── 3. location ─────────────────────────────────────────────────
    if args.location:
        if _is_image_path(args.location):
            panels.append(Path(args.location))
        else:
            panels += _run_comfy(
                comfy, "t2i", "sheet_loc", work_dir,
                ["--prompt", f"{args.location.strip()}{tail}",
                 "--width", "1024", "--height", "576"],
                steps=args.steps)

    # ── 4. tile ─────────────────────────────────────────────────────
    rows = cols = None
    if args.grid:
        s = args.grid.lower().replace(" ", "")
        for sep in ("x", ",", "*"):
            if sep in s:
                rows, cols = (int(x) for x in s.split(sep, 1))
                break
    out = build_sheet([str(p) for p in panels], args.out,
                      width=args.width, height=args.height,
                      rows=rows, cols=cols, pad=args.pad)
    print(f"[sheet] {len(panels)} panels -> {out} ({args.width}x{args.height})",
          file=sys.stderr)
    print(str(out))


if __name__ == "__main__":
    main()
