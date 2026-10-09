#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""make_reference_sheet.py — tile element panels onto one black "ingredients"
reference sheet for the LTX-2.3 ingredients IC-LoRA (comfy_graph.py ingredients).

The ingredients LoRA conditions a clip on a reference sheet: one clean panel per
recurring element (character face + turnaround, each prop, one location), on a
**black background, with no text**. This script is the low-level assembler — it
takes already-rendered panels and lays them out; the panels themselves are made
upstream (storyboard/generate_reference_sheet.py composes them via flux2).

Why aspect matters: the ingredients graph scales the sheet to the exact output
WxH with crop disabled (a straight stretch), so the **sheet aspect must equal
the render aspect** or the reference is distorted. This tool therefore emits the
sheet at exactly --width x --height — pass the project's render dimensions.
Trained buckets are landscape (768x448, 960x544); portrait (e.g. 896x1536 for a
vertical music-video) is supported here but is off the trained bucket — author
the sheet at the project aspect and evaluate.

Layout: panels are contain-fit (no crop, no distortion — fidelity matters for a
reference) and centered in a uniform grid on black. The grid is chosen to
maximise total panel area for the canvas aspect, so it adapts to portrait vs
landscape automatically; override with --cols / --rows / --grid RxC. Bigger
panels carry over better (per the Lightricks tips), so fewer/larger elements
beats cramming many tiny ones.

Usage:
    make_reference_sheet.py --out sheet.png [--width 768 --height 448] \
        face.png turnaround.png trident.png soda.png beach.png
    make_reference_sheet.py --out sheet.png --panels a.png,b.png,c.png --grid 2x2
    make_reference_sheet.py --out sheet.png --width 896 --height 1536 a.png b.png

Importable: `build_sheet(panels, out, width, height, ...)` returns the out Path.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image


def _contain(panel_w: int, panel_h: int, box_w: float, box_h: float) -> tuple[int, int]:
    """Largest (w, h) that fits panel into box preserving aspect (letterbox)."""
    if panel_w <= 0 or panel_h <= 0:
        return 0, 0
    ar = panel_w / panel_h
    if ar >= box_w / box_h:          # panel is wider than the box -> fit width
        w = box_w
        h = box_w / ar
    else:                            # taller than the box -> fit height
        h = box_h
        w = box_h * ar
    return max(1, int(round(w))), max(1, int(round(h)))


def _best_grid(aspects: list[float], canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """Pick (rows, cols) covering len(aspects) cells that maximises the total
    contained panel area for this canvas — naturally adapts to the canvas
    aspect (wide canvas -> more columns, tall canvas -> more rows)."""
    n = len(aspects)
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = canvas_w / cols
        cell_h = canvas_h / rows
        cell_ar = cell_w / cell_h
        area = 0.0
        for ar in aspects:
            if ar >= cell_ar:
                w = cell_w; h = cell_w / ar
            else:
                h = cell_h; w = cell_h * ar
            area += w * h
        # Tie-break toward fewer empty cells, then toward squarer cells.
        empties = rows * cols - n
        score = (area, -empties, -abs(cell_ar - 1.0))
        if best is None or score > best[0]:
            best = (score, rows, cols)
    return best[1], best[2]


def build_sheet(panels, out, width: int = 768, height: int = 448,
                cols: int | None = None, rows: int | None = None,
                pad: int = 10, margin: int = 0,
                bg: tuple[int, int, int] = (0, 0, 0),
                center_last_row: bool = True) -> Path:
    """Tile `panels` (paths) onto a black `width`x`height` sheet and save to `out`.

    pad: inner padding inside each cell (px). margin: border around the whole
    canvas (px). center_last_row: horizontally center a partial final row."""
    paths = [Path(p) for p in panels]
    if not paths:
        raise ValueError("no panels given")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"panel not found: {p}")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    n = len(imgs)

    inner_w = width - 2 * margin
    inner_h = height - 2 * margin
    if inner_w < 1 or inner_h < 1:
        raise ValueError("margin too large for canvas")

    if rows and cols:
        if rows * cols < n:
            raise ValueError(f"grid {rows}x{cols} too small for {n} panels")
    elif cols:
        rows = math.ceil(n / cols)
    elif rows:
        cols = math.ceil(n / rows)
    else:
        rows, cols = _best_grid([im.width / im.height for im in imgs], inner_w, inner_h)

    cell_w = inner_w / cols
    cell_h = inner_h / rows

    canvas = Image.new("RGB", (width, height), bg)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        # How many panels share this row (for optional last-row centering).
        in_row = min(cols, n - r * cols)
        row_shift = ((cols - in_row) * cell_w / 2) if (center_last_row and in_row < cols) else 0.0
        box_w = cell_w - 2 * pad
        box_h = cell_h - 2 * pad
        w, h = _contain(im.width, im.height, box_w, box_h)
        resized = im.resize((w, h), Image.LANCZOS)
        cell_x = margin + c * cell_w + row_shift
        cell_y = margin + r * cell_h
        ox = int(round(cell_x + (cell_w - w) / 2))
        oy = int(round(cell_y + (cell_h - h) / 2))
        canvas.paste(resized, (ox, oy))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def _parse_grid(s: str | None) -> tuple[int | None, int | None]:
    if not s:
        return None, None
    s = s.lower().replace(" ", "")
    for sep in ("x", ",", "*"):
        if sep in s:
            a, b = s.split(sep, 1)
            return int(a), int(b)
    raise ValueError(f"--grid must look like RxC, got {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panels", nargs="*", help="panel image paths (positional)")
    ap.add_argument("--panels", dest="panels_csv", default=None,
                    help="comma-separated panel paths (alternative to positional)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=768,
                    help="sheet width = render width (default 768)")
    ap.add_argument("--height", type=int, default=448,
                    help="sheet height = render height (default 448)")
    ap.add_argument("--grid", default=None, help="explicit grid RxC (e.g. 2x3)")
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--pad", type=int, default=10, help="inner cell padding px")
    ap.add_argument("--margin", type=int, default=0, help="canvas border px")
    ap.add_argument("--bg", default="0,0,0", help="background r,g,b (default black)")
    ap.add_argument("--no-center-last-row", action="store_true")
    args = ap.parse_args()

    panels = list(args.panels)
    if args.panels_csv:
        panels += [p.strip() for p in args.panels_csv.split(",") if p.strip()]
    if not panels:
        ap.error("no panels given (positional or --panels)")

    grid_rows, grid_cols = _parse_grid(args.grid)
    rows = args.rows if args.rows is not None else grid_rows
    cols = args.cols if args.cols is not None else grid_cols
    try:
        bg = tuple(int(x) for x in args.bg.split(","))
        assert len(bg) == 3
    except Exception:
        ap.error(f"--bg must be 'r,g,b', got {args.bg!r}")

    out = build_sheet(panels, args.out, width=args.width, height=args.height,
                      cols=cols, rows=rows, pad=args.pad, margin=args.margin,
                      bg=bg, center_last_row=not args.no_center_last_row)
    im = Image.open(out)
    print(f"{out}  ({im.width}x{im.height}, {len(panels)} panels)")


if __name__ == "__main__":
    main()
