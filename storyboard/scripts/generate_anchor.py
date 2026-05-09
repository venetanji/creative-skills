#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []   # stdlib only — calls comfy_graph.py via subprocess
# ///
"""CLI — produce ONE shot-specific anchor image via flux2, with prompt
preprocessing (subject expansion, style tail, action emphasis) applied
consistently so anchors from drama-video / music-video / one-offs all
go through the same pipeline.

This script shells out to `comfy_graph.py` (the comfyui skill) for the
actual flux2 call, so the same flux2 workflow knowledge is preserved in
one place and only prompt-layer conventions live here.

Exit codes:
    0  success (or skipped because --out already exists)
    1  bad args
    2  flux2 call failed

Usage:
    generate_anchor.py --out <path> --type <t2i|i2i|i2i2> --prompt "..."
                        [--reference a.png]
                        [--references a.png,b.png]
                        [--subject name=desc] ...
                        [--style editorial|indie|storybook|risograph|graphic_novel]
                        [--style-tail "<custom tail>"]
                        [--action-emphasis]
                        [--width 1024] [--height 576] [--steps 8]
                        [--comfy-graph /path/to/comfy_graph.py]
                        [--force]   # re-render even if --out exists

Subject/style flags are applied in this order:
    1. Subject tokens — `{name}` → value in the prompt (and in each
       reference's comment if you set one).
    2. Action emphasis — appends the DYNAMIC-ACTION phrase.
    3. Style tail — append style suffix (named preset or custom string).

Either --style OR --style-tail accepted (not both). --style-tail wins.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── make the lib importable no matter where we're invoked from ──────
SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR.parent / "lib"
sys.path.insert(0, str(LIB_DIR))
from prompts import expand_subjects, apply_style_tail, inject_action_emphasis  # noqa: E402


def _find_comfy_graph(cli_override: str | None) -> Path:
    if cli_override:
        p = Path(cli_override)
        if p.exists():
            return p
        sys.exit(f"--comfy-graph path not found: {cli_override}")
    candidates: list[Path] = [
        Path("/home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py"),  # OpenClaw sandbox
        Path.home() / ".openclaw/skills/comfyui/scripts/comfy_graph.py",        # host install
        SCRIPT_DIR.parent.parent / "comfyui/scripts/comfy_graph.py",           # repo checkout sibling
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    sys.exit("could not locate comfy_graph.py — pass --comfy-graph <path>")


def _parse_subjects(pairs: list[str]) -> dict[str, str]:
    subjects: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"--subject must be 'name=description', got {p!r}")
        name, desc = p.split("=", 1)
        subjects[name.strip()] = desc.strip()
    return subjects


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--type", required=True,
                    choices=("t2i", "i2i", "i2i2", "multiprompt"))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--reference", default=None,
                    help="single reference image (i2i)")
    ap.add_argument("--references", default=None,
                    help="comma-separated references (i2i2 / i2iN)")
    ap.add_argument("--subject", action="append", default=[],
                    help="name=description, repeatable")
    ap.add_argument("--style", default=None,
                    help="named style tail (see lib/prompts.py)")
    ap.add_argument("--style-tail", default=None,
                    help="custom style tail string (overrides --style)")
    ap.add_argument("--action-emphasis", action="store_true",
                    help="inject DYNAMIC-ACTION-POSE anti-lineup phrase")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--comfy-graph", default=None,
                    help="override comfy_graph.py path (auto-detected by default)")
    ap.add_argument("--force", action="store_true",
                    help="re-render even if --out already exists")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(str(args.out))
        return

    subjects = _parse_subjects(args.subject)
    prompt = expand_subjects(args.prompt, subjects)
    if args.action_emphasis:
        prompt = inject_action_emphasis(prompt)
    prompt = apply_style_tail(prompt, args.style_tail or args.style)

    # Build the comfy_graph.py subcommand.
    comfy = _find_comfy_graph(args.comfy_graph)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # comfy writes to <output-dir>/<prefix>_NNNNN_.png then we rename.
    prefix = args.out.stem
    common = ["--prompt", prompt,
              "--width", str(args.width),
              "--height", str(args.height),
              "--steps", str(args.steps),
              "--prefix", prefix,
              "--output-dir", str(args.out.parent),
              "--timeout", "600"]

    if args.type == "t2i":
        cmd = ["python3", str(comfy), "t2i"] + common
    elif args.type == "i2i":
        if not args.reference:
            sys.exit("--reference required for i2i")
        cmd = ["python3", str(comfy), "i2i",
               "--image", args.reference] + common
    elif args.type == "i2i2":
        if not args.references:
            sys.exit("--references required for i2i2 (comma-separated a,b)")
        refs = [r.strip() for r in args.references.split(",") if r.strip()]
        if len(refs) != 2:
            sys.exit(f"i2i2 needs exactly 2 references, got {len(refs)}")
        cmd = ["python3", str(comfy), "i2i2",
               "--image1", refs[0], "--image2", refs[1]] + common
    elif args.type == "multiprompt":
        if not args.reference:
            sys.exit("--reference required for multiprompt")
        # Multiprompt treats --prompt as a single prompt line; for multi-line
        # pose batches, the caller should build the pipeline inline instead.
        cmd = ["python3", str(comfy), "multiprompt",
               "--image", args.reference,
               "--prompts", prompt] + common
    else:
        sys.exit(f"unknown --type {args.type}")

    print(f"[storyboard] rendering {args.out.name} via flux2 {args.type}…",
          file=sys.stderr)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(2)

    # comfy_graph saves to <prefix>_NNNNN_.png — find the newest and
    # rename to the caller's exact --out.
    matches = sorted(args.out.parent.glob(f"{prefix}_*.png"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        sys.exit(f"comfy produced no image matching {prefix}_*.png in "
                 f"{args.out.parent}")
    matches[0].rename(args.out)
    print(str(args.out))


if __name__ == "__main__":
    main()
