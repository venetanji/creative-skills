#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0", "imageio-ffmpeg>=0.6"]
# ///
"""Music-video orchestrator — song (suno) + per-scene video (comfyui ia2v) + assemble.

Pipeline:
  1. `plan`     validate YAML spec, show scene breakdown
  2. `song`     generate full song via suno-mcp, save song.mp3 + meta
  3. `scene N`  generate one scene's MP4 via comfyui ia2v (using the song slice)
  4. `assemble` concat scene MP4s into final.mp4, overlay original song audio
  5. `all`      run 2→3*→4 in sequence
  6. `status`   show what's been generated already

YAML spec (minimum):

    title: Glass Harbour
    style: "folk noir, 74 BPM, fingerpicked guitar, bowed upright bass, ..."
    lyrics: |
      [Verse]
      Salt in the air and rust on the crane...
      [Chorus]
      Glass harbour, glass harbour...
    video:
      fps: 24
      resolution: [1024, 576]
      negative: "pc game, cartoon, modern tech, text, ugly"
    anchor_image: anchor.png         # optional; first scene uses t2i/t2v if missing
    scenes:
      - label: establishing
        start_sec: 0
        duration_sec: 8
        prompt: "wide fog-shrouded harbour at dawn..."
        image: "@anchor"             # @anchor | @last | path/to/img
      - label: fisherman
        start_sec: 8
        duration_sec: 10
        prompt: "weathered fisherman hauling nets..."
        image: "@last"               # chains from prev scene's last frame

Folder layout the orchestrator creates:

    <project>/
      song.yaml
      song.mp3
      song_meta.json
      song_slices/
        001-establishing.mp3
        002-fisherman.mp3
      scenes/
        001-establishing.mp4
        001-establishing-last.png
        002-fisherman.mp4
        ...
      final.mp4
      run.log

CLI:
  music_video.py plan     <spec.yaml>
  music_video.py song     <spec.yaml>
  music_video.py scene N  <spec.yaml>
  music_video.py assemble <spec.yaml>
  music_video.py all      <spec.yaml>
  music_video.py status   <spec.yaml>

Notes:
  • Scene `image: "@last"` uses the prev scene's last frame → continuity between scenes.
  • Scene `image: "@anchor"` uses the top-level `anchor_image`.
  • If the first scene is `"@last"` with no anchor, the orchestrator runs t2v for that scene.
  • Each scene is generated in isolation (1-3 min each). Restart-safe — `all` skips
    scenes whose MP4 already exists.
  • Final assembly re-encodes to overlay the clean suno song over the silent video
    track (discarding ia2v's generated audio for each scene).
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

# Force UTF-8 on stdout/stderr so the script can print the unicode glyphs
# it uses for status (→ ✓ ⚠ — …) on Windows consoles whose default code
# page is cp1252. Without this, any unicode print() crashes with
# UnicodeEncodeError. Safe no-op on POSIX where stdout is already utf-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Lazy imports for pyyaml + imageio_ffmpeg. These are heavy deps that
# only some subcommands need — `init` just writes a text skeleton and
# should work on any minimal sandbox with stock python3 (no uv, no
# pip-installed packages). Call sites use `_require_yaml()` /
# `_require_ffmpeg()` to get the module and fail with a helpful error
# message if it's missing.
def _require_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        sys.exit("pyyaml not installed. Install with: pip install pyyaml, "
                 "or run this script via `uv run --script`.")

def _require_ffmpeg():
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe
    except ImportError:
        sys.exit("imageio_ffmpeg not installed. Install with: "
                 "pip install imageio-ffmpeg, or run via `uv run --script`.")


# ---------- paths & helpers ----------

# Skill root is the parent of scripts/
SKILL_ROOT = Path(__file__).resolve().parent.parent
# The comfyui and suno-mcp skills are sibling dirs under ~/.openclaw/skills/
SKILLS_ROOT = SKILL_ROOT.parent
COMFY = SKILLS_ROOT / "comfyui" / "scripts" / "comfy_graph.py"
SUNO  = SKILLS_ROOT / "suno-mcp" / "scripts" / "generate_song.py"
VIDEO_JOIN = SKILLS_ROOT / "comfyui" / "scripts" / "video_join.py"

def FFMPEG():
    return _require_ffmpeg()()


def _log(project: Path, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(project / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(cmd: list[str], cwd: Path | None = None, log_path: Path | None = None,
         env_override: dict[str, str] | None = None) -> int:
    """Run subprocess; stream stderr to log, return exit code.

    env_override — inject extra env vars (merged with os.environ). Used to
    route subprocess calls to different ComfyUI instances per tool:
      COMFY_URL_FLUX  → flux2 image gen (anchors) on the 12GB GPU
      COMFY_URL_VIDEO → LTX video gen (t2v/ia2v) on the 24GB GPU"""
    env = None
    if env_override:
        env = {**os.environ, **env_override}
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=env) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        p.wait()
        return p.returncode


def _comfy_env_for(kind: str) -> dict[str, str]:
    """Pick the COMFY_URL for a tool kind.
      kind=='flux'  → COMFY_URL_FLUX  (falls back to COMFY_URL)
      kind=='video' → COMFY_URL_VIDEO (falls back to COMFY_URL)
    Empty dict if neither is set (subprocess inherits parent's COMFY_URL)."""
    var = {"flux": "COMFY_URL_FLUX", "video": "COMFY_URL_VIDEO"}.get(kind)
    if not var:
        return {}
    val = os.environ.get(var)
    if not val:
        return {}
    return {"COMFY_URL": val}


def _load_spec(yaml_path: str) -> tuple[dict, Path]:
    p = Path(yaml_path).resolve()
    if not p.exists():
        sys.exit(f"spec not found: {p}")
    # Force utf-8 — Path.read_text() picks the OS default (cp1252 on
    # Windows), which mangles em-dashes and other non-ASCII glyphs that
    # legitimately appear in style/lyrics/scene prompts.
    spec = _require_yaml().safe_load(p.read_text(encoding="utf-8"))
    _fill_scene_start_sec(spec)
    return spec, p.parent  # project dir = dir containing the yaml


def _fill_scene_start_sec(spec: dict) -> None:
    """Auto-derive `start_sec` for scenes that omit it — cumulative sum of
    prior durations. Lets the agent write scenes as a flat sequential list
    with only `duration_sec`."""
    scenes = spec.get("scenes") or []
    cursor = 0.0
    for s in scenes:
        if "start_sec" not in s:
            s["start_sec"] = cursor
        cursor = float(s["start_sec"]) + float(s.get("duration_sec", 0))


# Default workspace root used by `init`. Resolution order:
#   1. MV_WORKSPACE_ROOT env var (explicit override for odd setups).
#   2. /workspace if present and writable (the OpenClaw sandbox convention —
#      bind-mounted to the agent's workspace dir on the host).
#   3. ~/.openclaw/workspace (host install fallback).
def _default_workspace_root() -> Path:
    override = os.environ.get("MV_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    if os.path.isdir("/workspace") and os.access("/workspace", os.W_OK):
        return Path("/workspace")
    return Path.home() / ".openclaw" / "workspace"

DEFAULT_WORKSPACE_ROOT = _default_workspace_root()


SKELETON_YAML_TEMPLATE = """# Music-video project: {slug}
# Edit this file: fill title / style / lyrics, then run:
#   music_video.py song {slug_yaml}
# After the song is picked, add scenes and run the rest of the pipeline.
{theme_block}
# TODO: fill title
title: "{default_title}"

# TODO: producer-style brief. See suno-mcp style-guide:
#   ~/.openclaw/skills/suno-mcp/references/style-guide.md
# Single prose sentence covering genre, BPM, instruments, production, vocal, mood.{theme_style_hint}
style: "TODO"

# TODO: lyrics using [Verse]/[Chorus]/[Bridge] tags, one line per phrase,
# no mid-line punctuation. See lyrics-guide:
#   ~/.openclaw/skills/suno-mcp/references/lyrics-guide.md{theme_lyrics_hint}
lyrics: |
  [Verse]
  TODO

suno:
  runs: 3  # how many suno calls; each returns 2 variants -> runs*2 total mp3s

video:
  fps: 24
  resolution: [1024, 576]
  negative: "cartoon, text, watermark, distorted face, still frame, facial hair on woman, beard on woman"
  tail_buffer_sec: 0.5
  transitions:
    enabled: false

anchor_image: anchor.png

# Quality-check gates — the `all` pipeline stops after song/anchors so a
# human (or agent) can review before the next expensive phase. Flip to
# false to fully automate (or pass --no-gate on the CLI).
gate_confirm_song: true
gate_confirm_anchors: true

# TODO: add scenes AFTER the song is confirmed. Each scene:
#   - label: short_id
#     start_sec: 0
#     duration_sec: 8
#     prompt: "what is on screen — continuous shot"
#     image: "@anchor"  # @anchor | @last | path/to/img
scenes: []
"""


def cmd_init(slug: str, theme: str | None, force: bool) -> None:
    """Create a new project skeleton under DEFAULT_WORKSPACE_ROOT/<slug>/."""
    if not slug or "/" in slug or slug.startswith("."):
        sys.exit(f"invalid slug: {slug!r} (no slashes, no leading dot)")
    project = DEFAULT_WORKSPACE_ROOT / slug
    yaml_path = project / "song.yaml"

    if project.exists() and not force:
        sys.exit(f"project already exists: {project} (pass --force to overwrite)")

    project.mkdir(parents=True, exist_ok=True)

    default_title = slug.replace("-", " ").replace("_", " ").title()
    theme_block = ""
    theme_style_hint = ""
    theme_lyrics_hint = ""
    if theme:
        theme_clean = theme.replace('"', "'")
        theme_block = f'#\n# Theme: {theme_clean}\n'
        theme_style_hint = f"\n# Theme context: {theme_clean}"
        theme_lyrics_hint = f"\n# Theme context: {theme_clean}"

    content = SKELETON_YAML_TEMPLATE.format(
        slug=slug,
        slug_yaml=str(yaml_path),
        default_title=default_title,
        theme_block=theme_block,
        theme_style_hint=theme_style_hint,
        theme_lyrics_hint=theme_lyrics_hint,
    )
    yaml_path.write_text(content, encoding="utf-8")

    print(f"created {project}/")
    print(f"  song.yaml (skeleton)")
    if theme:
        print(f"  theme: {theme}")
    print()
    print(f"next: edit {yaml_path} (title, style, lyrics),")
    print(f"      then run: music_video.py song {yaml_path}")


def _scene_stem(idx: int, label: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40] or "scene"
    return f"{idx:03d}-{safe}"


def _ensure_dirs(project: Path) -> None:
    (project / "song_slices").mkdir(exist_ok=True)
    (project / "scenes").mkdir(exist_ok=True)


def _video_spec(spec: dict) -> dict:
    """Return spec.video coerced into a renderer-friendly shape.

    Historically this returned ONLY fps/width/height/negative/tail_buffer_sec
    and silently dropped every other key from spec.video. That meant
    every video-level field added since (lipsync_audio, hdr_lora,
    base_guide_strength, refine_guide_strength, fast, transitions, ...)
    looked correct in the spec but never actually reached the renderer
    when read via `vs.get(...)` — a long-standing silent bug.

    Now we pass through every key from spec.video and just OVERLAY the
    five canonical fields with coerced types on top, so existing
    `vs["fps"]` / `vs["width"]` etc. still get the int/float guarantees
    they relied on, and any new field is also reachable via
    `vs.get("lipsync_audio")` / `vs.get("hdr_lora")` etc. without
    refactoring callers.
    """
    v = dict(spec.get("video") or {})
    w, h = (v.get("resolution") or [1024, 576])
    v["fps"] = int(v.get("fps", 24))
    v["width"] = int(w)
    v["height"] = int(h)
    v["negative"] = v.get("negative")
    v["tail_buffer_sec"] = float(v.get("tail_buffer_sec", 0.0) or 0.0)
    return v


# ---------- subcommands ----------

# Hard ceiling on per-scene duration. Empirically, LTX-2.3 at portrait
# resolutions (e.g. 448×832) runs ~quadratic in length: a 12s clip
# completes in ~6-8 min, a 20s clip took ~37 min and OOM-killed the comfy
# server (which also drops the rest of the queue). Cap at 12s and split
# longer scenes into chained shots if you need more runtime.
MAX_SCENE_DURATION = 15.0


# Simple {name} → description substitution driven by a top-level `subjects:`
# block in the yaml. Lets prompts reference recurring characters by short
# token (e.g. `{snakebird}`) and keep their canonical body description in
# one place. Unknown tokens pass through untouched so literal curly braces
# in prompts don't break. Applied to every scene.prompt and anchor.prompt.
def _expand_subjects(text: str, spec: dict) -> str:
    subjects = spec.get("subjects") or {}
    if not isinstance(text, str):
        return text
    for name, desc in subjects.items():
        text = text.replace("{" + str(name) + "}", str(desc))
    # Defensive: warn on any leftover {token} that wasn't substituted. Common
    # cause: spec author wrote `{narrator}` in scene prompts but forgot the
    # top-level `subjects:` block, so the literal `{narrator}` string was
    # being shipped to flux2 / LTX. Models tolerate it (CLIP treats it as
    # noise) but the prompt is silently degraded.
    import re
    leftover = re.findall(r"\{([a-zA-Z][a-zA-Z0-9_-]*)\}", text)
    if leftover:
        unique = sorted(set(leftover))
        print(f"[WARN] unresolved subject token(s) in prompt: "
              f"{', '.join('{' + n + '}' for n in unique)} — "
              f"add a top-level `subjects:` block to your spec",
              file=sys.stderr)
    return text


def cmd_plan(spec: dict, project: Path) -> None:
    scenes = spec.get("scenes", [])
    if not scenes:
        sys.exit("spec has no `scenes`")
    total = max((s["start_sec"] + s["duration_sec"]) for s in scenes)
    vs = _video_spec(spec)
    # Final video = sum(scene duration_sec) — each scene's tail buffer is
    # trimmed off at concat (via GetImageRangeFromBatch), no crossfade.
    assembled = sum(s["duration_sec"] for s in scenes)

    tail_buf = float((spec.get("video") or {}).get("tail_buffer_sec", 0) or 0)
    print(f"title         : {spec.get('title','(untitled)')}")
    print(f"style         : {spec.get('style','')[:120]}")
    print(f"video         : {vs['width']}x{vs['height']} @ {vs['fps']}fps  tail_buffer={tail_buf}s")
    print(f"scenes total  : {total:.1f}s across {len(scenes)} scene(s)")
    print(f"assembled len : {assembled:.1f}s  (trim each scene to duration_sec, buffer dropped)")
    print(f"anchor        : {spec.get('anchor_image','(none)')}")

    # Compare against the generated song if it already exists.
    song = project / "song.mp3"
    if song.exists():
        sd = _probe_duration(song)
        print(f"song.mp3      : {sd:.1f}s")
        gap = sd - assembled
        if abs(gap) < 1:
            print(f"→ assembled length matches song — good")
        elif gap > 0:
            print(f"⚠  song is {gap:.1f}s LONGER than assembled scenes — final video's tail "
                  f"will freeze on the last frame for that duration. Add or extend scenes.")
        else:
            print(f"⚠  scenes exceed song by {-gap:.1f}s — the assembly will be trimmed to song length.")
        for alt in sorted(project.glob("song_v*.mp3")):
            print(f"  {alt.name:14}: {_probe_duration(alt):.1f}s")

    # Per-scene warnings
    too_long = [s for s in scenes if s["duration_sec"] > MAX_SCENE_DURATION]
    if too_long:
        labels = ", ".join(s.get("label", "?") for s in too_long)
        print(f"⚠  scene(s) exceed MAX_SCENE_DURATION ({MAX_SCENE_DURATION}s): {labels}")
        print(f"   LTX-2.3 can OOM past ~20s per scene at portrait resolutions. Split them.")

    print()
    print(f"{'#':>3}  {'label':18}  {'start':>6}  {'dur':>5}  {'src':10}  prompt")
    for i, s in enumerate(scenes, 1):
        flag = " !" if s["duration_sec"] > MAX_SCENE_DURATION else "  "
        # src column reflects what will ACTUALLY be used as the ia2v starting
        # frame: an `anchor:` block (with prompt) wins via flux2 pre-render
        # over scene.image, which itself defaults to @last.
        ac = s.get("anchor")
        if ac and ac.get("prompt"):
            t = ac.get("type")
            if t is None:
                refs = ac.get("references") or ([ac["reference"]] if ac.get("reference") else [])
                t = ("t2i", "i2i", "i2i2", "i2iN")[min(len(refs), 3)]
            src = f"flux2:{t}"
        else:
            src = s.get("image", "@last")
        print(f"{i:>3}  {s.get('label',''):18}  {s['start_sec']:>6.1f}  {s['duration_sec']:>5.1f}{flag}"
              f"{src:10}  {s['prompt'][:60]}")


def _next_variant_slot(project: Path) -> int:
    """Return the next free v-index for song_vN.mp3 (smallest N >= 2 not in use)."""
    used = set()
    for p in project.glob("song_v*.mp3"):
        stem = p.stem.replace("song_v", "")
        if stem.isdigit():
            used.add(int(stem))
    n = 2
    while n in used:
        n += 1
    return n


def _run_suno_once(title: str, style: str, lyrics: str, instrumental: bool) -> dict:
    """Invoke the suno-mcp generator once. Returns the parsed meta dict.
    Each call yields 2 variants in meta['local_files']."""
    cmd = ["python3", str(SUNO),
           "--title", title, "--tags", style,
           "--timeout", "400"]
    if instrumental:
        cmd.append("--instrumental")
    if lyrics:
        cmd += ["--lyrics", lyrics]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"suno failed:\n{proc.stderr}\n{proc.stdout}")

    out = proc.stdout.strip()
    try:
        brace = out.rfind("{")
        meta = json.loads(out[brace:]) if brace >= 0 else {}
    except json.JSONDecodeError:
        meta = {"raw": out}
    return meta


def cmd_song(spec: dict, project: Path) -> None:
    """Generate N suno runs (spec.suno.runs; default 1). Each run returns 2
    variants. The very first variant of the first run is saved as song.mp3;
    all other variants are saved as song_v2.mp3, song_v3.mp3, ... in the
    order they're produced. Restart-safe: if song.mp3 already exists, counts
    existing song_v*.mp3 and issues only the remaining runs."""
    out_mp3 = project / "song.mp3"
    meta_path = project / "song_meta.json"

    title = spec.get("title") or "Untitled"
    style = spec.get("style") or ""
    lyrics = spec.get("lyrics") or ""
    instrumental = bool((spec.get("suno") or {}).get("make_instrumental", False))
    runs_wanted = int((spec.get("suno") or {}).get("runs", 1))

    # suno.runs: 0 means "use hand-placed song.mp3, don't call suno at all"
    # — honored only when song.mp3 actually exists on disk (otherwise fall
    # through and report the missing-style error or proceed with defaults).
    if runs_wanted <= 0:
        if out_mp3.exists():
            _log(project, "suno.runs: 0 — using existing song.mp3, skipping generation")
            return
        _log(project, "suno.runs: 0 but no song.mp3 — forcing 1 run")
        runs_wanted = 1

    if not style:
        sys.exit("spec.style is required for suno generation")

    # Each run yields 2 variants → total variants = runs_wanted * 2.
    # Existing count: song.mp3 (1 if present) + number of song_v*.mp3.
    existing_variants = (1 if out_mp3.exists() else 0) + len(list(project.glob("song_v*.mp3")))
    target_total = runs_wanted * 2
    if existing_variants >= target_total:
        _log(project, f"song already has {existing_variants}/{target_total} variants, skipping")
        return

    runs_done = existing_variants // 2
    runs_remaining = runs_wanted - runs_done
    _log(project, f"suno: {runs_done}/{runs_wanted} runs done, "
                  f"{runs_remaining} remaining ({existing_variants} variants on disk)")

    all_meta: list[dict] = []
    # Load any prior meta so we can accumulate across restarts.
    if meta_path.exists():
        try:
            prior = json.loads(meta_path.read_text())
            if isinstance(prior, list):
                all_meta = prior
            elif isinstance(prior, dict):
                all_meta = [prior]
        except json.JSONDecodeError:
            pass

    saved_files: list[str] = [str(out_mp3)] if out_mp3.exists() else []
    for p in sorted(project.glob("song_v*.mp3"),
                    key=lambda q: int(q.stem.replace("song_v", "") or "0")):
        saved_files.append(str(p))

    for r in range(runs_remaining):
        run_idx = runs_done + r + 1
        _log(project, f"generating song '{title}' via suno-mcp — run {run_idx}/{runs_wanted}")
        meta = _run_suno_once(title, style, lyrics, instrumental)
        all_meta.append(meta)

        variants = meta.get("local_files") or (
            [meta["local_file"]] if meta.get("local_file") else [])
        if not variants:
            sys.exit(f"suno run {run_idx} produced no local_files; meta={meta}")

        for v in variants:
            if not v or not Path(v).exists():
                _log(project, f"warning: suno returned missing file {v!r}")
                continue
            if not out_mp3.exists():
                shutil.copy(v, out_mp3)
                saved_files.append(str(out_mp3))
                _log(project, f"song saved: {out_mp3.name} "
                              f"({out_mp3.stat().st_size//1024} KB)")
            else:
                slot = _next_variant_slot(project)
                dst = project / f"song_v{slot}.mp3"
                shutil.copy(v, dst)
                saved_files.append(str(dst))
                _log(project, f"variant {slot} saved: {dst.name} "
                              f"({dst.stat().st_size//1024} KB)")

        # Write accumulated meta after each run so a crash mid-batch doesn't
        # lose the provenance of completed runs.
        meta_path.write_text(json.dumps({
            "runs": all_meta,
            "saved_files": saved_files,
        }, indent=2))

    _log(project, f"suno done: {len(saved_files)} variant(s) on disk")


def _slice_song(project: Path, scene: dict, stem: str,
                tail_buffer_sec: float = 0.0,
                source_path: Path | None = None) -> Path:
    """Slice the song for LTX audio conditioning. tail_buffer_sec extends the
    slice past the scene's nominal end so LTX has audio lookhead to finish
    the phoneme / close the mouth — otherwise the mouth freezes mid-word at
    scene boundary. Extra tail is absorbed by crossfade=tail_buffer in
    assembly so the timeline doesn't drift.

    source_path: optional override for the audio source (defaults to
    project/song.mp3). Use this to feed LTX a vocal-forward remix
    (vocals + backing vocals boosted) for cleaner lipsync conditioning,
    while still keeping song.mp3 as the canonical full-mix track for
    assembly. See video.lipsync_audio in the spec."""
    slice_mp3 = project / "song_slices" / f"{stem}.mp3"
    if slice_mp3.exists():
        return slice_mp3
    ffmpeg = FFMPEG()
    src = source_path if source_path is not None else (project / "song.mp3")
    total = float(scene["duration_sec"]) + float(tail_buffer_sec)
    cmd = [ffmpeg, "-y", "-ss", str(scene["start_sec"]),
           "-t",  str(total),
           "-i",  str(src),
           "-c",  "copy", str(slice_mp3)]
    rc = subprocess.run(cmd, capture_output=True).returncode
    if rc != 0:
        rc = subprocess.run([ffmpeg, "-y", "-ss", str(scene["start_sec"]),
                             "-t",  str(total),
                             "-i",  str(src),
                             "-c:a", "libmp3lame", "-b:a", "192k", str(slice_mp3)],
                            capture_output=True).returncode
        if rc != 0:
            sys.exit(f"failed to slice audio for {stem}")
    return slice_mp3


def _generate_anchor(spec: dict, project: Path, idx: int, scene: dict, stem: str) -> str | None:
    """Pre-generate a scene-specific anchor via flux2 (t2i or i2i) for consistency.

    Triggered when scene.anchor.prompt is set. If scene.anchor.reference is also
    set, uses flux2 i2i (blends the reference image + prompt); otherwise t2i.
    The generated PNG is saved to scenes/NNN-label-anchor.png and becomes the
    ia2v input image for this scene — overrides scene.image when present.

    Restart-safe: skips generation if the anchor PNG already exists."""
    anchor_cfg = scene.get("anchor")
    if not anchor_cfg or not anchor_cfg.get("prompt"):
        return None

    anchor_path = project / "scenes" / f"{stem}-anchor.png"
    if anchor_path.exists():
        return str(anchor_path)

    vs = _video_spec(spec)
    # Anchor resolution defaults to `video.resolution × video.anchor_scale`.
    # Default scale is 2.0 to match LTX-2.3's 2-pass spatial upscaler
    # (LTXVLatentUpsampler 2×): the final rendered video is 2× the LTX
    # working resolution, so rendering anchors at 2× video means flux2 puts
    # real detail at the output resolution rather than at the LTX working
    # resolution. Set `video.anchor_scale: 1.0` to opt out (legacy behavior),
    # or override per-scene via `anchor.width` / `anchor.height`.
    anchor_scale = float(vs.get("anchor_scale", 2.0))
    w = int(anchor_cfg.get("width",  int(vs["width"]  * anchor_scale)))
    h = int(anchor_cfg.get("height", int(vs["height"] * anchor_scale)))

    # Support t2i / i2i / i2i2 / angles. Either `reference` (single path) or
    # `references` (list) — type is inferred from the count, or set explicitly.
    refs = anchor_cfg.get("references") or (
        [anchor_cfg["reference"]] if anchor_cfg.get("reference") else [])
    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (project / p).resolve()
    ref_paths = [_abs(r) for r in refs]

    anchor_type = anchor_cfg.get("type")
    if anchor_type is None:
        if len(ref_paths) == 0:
            anchor_type = "t2i"
        elif len(ref_paths) == 1:
            anchor_type = "i2i"
        elif len(ref_paths) == 2:
            anchor_type = "i2i2"
        else:
            anchor_type = "i2iN"

    # Identity-preservation guard for i2i/i2i2 anchors: when we hand flux2 a
    # reference image of a person, auto-append a concrete instruction to
    # keep the same face. Otherwise flux2 reads the prompt as a brief and
    # invents a new-looking person every scene → drift + gender swap + the
    # "my face is degrading" problem. Prompt authors can skip this by
    # adding `keep_identity: false` to the anchor block.
    anchor_prompt = _expand_subjects(anchor_cfg["prompt"], spec)
    if anchor_type in ("i2i", "i2i2") and anchor_cfg.get("keep_identity", True):
        id_guard = (" Keep the exact same person from the reference image: "
                    "same face shape, facial features, eye shape and color, "
                    "nose, lips, skin tone, hair color and style, and body "
                    "type. Only the pose, expression, clothing styling, "
                    "lighting, and environment should change according to "
                    "this prompt.")
        if id_guard.strip().lower() not in anchor_prompt.lower():
            anchor_prompt = anchor_prompt.rstrip(" .") + "." + id_guard

    common = ["--prompt", anchor_prompt,
              "--width",  str(w),
              "--height", str(h),
              "--prefix", f"{stem}-anchor",
              "--output-dir", str(project / "scenes"),
              "--timeout", "600"]       # flux2 at 1024x576 is ~15-20s; give it room
    if "steps" in anchor_cfg:
        common += ["--steps", str(int(anchor_cfg["steps"]))]

    # Friendly ref-count validation BEFORE indexing ref_paths blindly —
    # otherwise an explicit `type: i2i2` with one (or zero) references
    # fails with an opaque IndexError instead of a hint.
    _min_refs = {"i2i": 1, "i2i2": 2, "i2iN": 1, "angles": 1}.get(anchor_type)
    if _min_refs is not None and len(ref_paths) < _min_refs:
        sys.exit(f"scene {idx}: anchor.type '{anchor_type}' needs at least "
                 f"{_min_refs} reference image(s), got {len(ref_paths)}. "
                 f"Set `reference:` (one path) or `references: [a.png, b.png]` "
                 f"in the scene's anchor block.")
    if anchor_type == "i2i2" and len(ref_paths) > 2:
        sys.exit(f"scene {idx}: anchor.type 'i2i2' takes exactly 2 references, "
                 f"got {len(ref_paths)}. Use `type: i2iN` for 3+.")

    if anchor_type == "t2i":
        cmd = ["python3", str(COMFY), "t2i"] + common
    elif anchor_type == "i2i":
        cmd = ["python3", str(COMFY), "i2i", "--image", str(ref_paths[0])] + common
    elif anchor_type == "i2i2":
        cmd = ["python3", str(COMFY), "i2i2",
               "--image1", str(ref_paths[0]),
               "--image2", str(ref_paths[1])] + common
    elif anchor_type == "i2iN":
        # 3+ references: comfy_graph.py i2iN takes a comma-separated list.
        images_csv = ",".join(str(p) for p in ref_paths)
        cmd = ["python3", str(COMFY), "i2iN", "--images", images_csv] + common
    elif anchor_type == "angles":
        # Generates a batch — uses the first prompt as the primary anchor.
        angle_prompts = anchor_cfg.get("angle_prompts") or [anchor_cfg["prompt"]]
        cmd = ["python3", str(COMFY), "multiprompt",
               "--image", str(ref_paths[0]),
               "--prompts", "\n".join(angle_prompts)] + common
    else:
        sys.exit(f"scene {idx}: unknown anchor.type '{anchor_type}' "
                 "(expected t2i|i2i|i2i2|i2iN|angles)")

    _log(project, f"scene {idx}: generating anchor via flux2 "
                  f"({anchor_type}{' refs='+','.join(refs) if refs else ''})")
    rc = _run(cmd, log_path=project / "run.log",
              env_override=_comfy_env_for("flux"))
    if rc != 0:
        sys.exit(f"anchor gen failed for scene {idx}")

    # comfy_graph auto-numbers outputs (_00001_, _00002_, ...). For `angles`
    # multiple are produced; pick the first numerically (the first angle prompt)
    # so the anchor prompt controls which composition becomes the scene anchor.
    candidates = sorted((project / "scenes").glob(f"{stem}-anchor_*.png"),
                        key=lambda p: p.name)
    if not candidates:
        sys.exit(f"scene {idx}: no anchor PNG produced")
    candidates[0].rename(anchor_path)
    # Archive any additional outputs (from angles) so they don't clutter scenes/.
    for extra in candidates[1:]:
        extra.rename(project / "scenes" / f"{stem}-anchor-extra-{extra.name}")
    _log(project, f"scene {idx}: anchor → {anchor_path.name}")
    return str(anchor_path)


def cmd_anchors(spec: dict, project: Path) -> None:
    """Generate anchor images ahead of scene rendering.

    Two kinds of anchors:
      1. Top-level `anchor_image` (e.g. `anchor.png`) — generated via flux2
         t2i from `spec.anchor_prompt` (or, as a fallback, from title + style).
         Only created if the file is missing.
      2. Per-scene anchors — any scene with an `anchor:` block gets its
         flux2 t2i / i2i / i2i2 / angles rendered into
         scenes/NNN-label-anchor.png.

    Restart-safe and idempotent — skips anchors already on disk."""
    _ensure_dirs(project)
    generated = 0
    skipped = 0

    # --- top-level anchor ---
    anchor_name = spec.get("anchor_image")
    if anchor_name:
        anchor_path = (project / anchor_name).resolve()
        if anchor_path.exists():
            _log(project, f"anchor_image already exists: {anchor_path.name}")
            skipped += 1
        else:
            prompt = spec.get("anchor_prompt")
            if not prompt:
                title = spec.get("title", "")
                style = spec.get("style", "")
                prompt = (f"{title}. {style}" if title or style else
                          "a cinematic music-video key frame, atmospheric lighting")
            vs = _video_spec(spec)
            # Match `_generate_anchor`: render at video.resolution × anchor_scale
            # so the top-level anchor has detail at the final upscaled video
            # resolution (LTX's 2× spatial upscaler doubles output).
            anchor_scale = float(vs.get("anchor_scale", 2.0))
            w = int(vs["width"]  * anchor_scale)
            h = int(vs["height"] * anchor_scale)
            prefix = anchor_path.stem   # e.g. "anchor"
            cmd = ["python3", str(COMFY), "t2i",
                   "--prompt", prompt,
                   "--width",  str(w),
                   "--height", str(h),
                   "--prefix", prefix,
                   "--output-dir", str(project),
                   "--timeout", "600"]
            _log(project, f"generating top-level anchor → {anchor_path.name} "
                          f"via flux2 t2i")
            rc = _run(cmd, log_path=project / "run.log",
                      env_override=_comfy_env_for("flux"))
            if rc != 0:
                sys.exit("top-level anchor generation failed")
            candidates = sorted(project.glob(f"{prefix}_*.png"),
                                key=lambda p: p.name)
            # When anchor_image points at a subdir (e.g. `scene_anchors/foo.png`)
            # the destination parent must exist before rename.
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            if not candidates:
                # flux2 may also drop .png without the _NNNNN_ suffix depending
                # on comfy settings; check bare name too.
                if anchor_path.exists():
                    generated += 1
                else:
                    sys.exit(f"anchor t2i produced no PNG matching {prefix}_*.png")
            else:
                candidates[0].rename(anchor_path)
                for extra in candidates[1:]:
                    extra.unlink()
                generated += 1

    # --- per-scene anchors ---
    scenes = spec.get("scenes", []) or []
    for i, scene in enumerate(scenes, 1):
        if not scene.get("anchor") or not scene["anchor"].get("prompt"):
            continue
        stem = _scene_stem(i, scene.get("label", "scene"))
        anchor_path = project / "scenes" / f"{stem}-anchor.png"
        if anchor_path.exists():
            skipped += 1
            continue
        _generate_anchor(spec, project, i, scene, stem)
        if anchor_path.exists():
            generated += 1

    _log(project, f"anchors: generated {generated}, skipped {skipped} existing")


def _resolve_image(spec: dict, project: Path, idx: int, ref: str | None) -> str | None:
    """Resolve image-token references → absolute path string (or None for t2v).
    Token vocabulary:
      @none           → no image conditioning (forces t2v)
      @last           → previous scene's last frame (chained continuity)
      @anchor         → top-level spec.anchor_image
      @scene_anchor   → THIS scene's flux2-rendered anchor PNG. Use this in
                        guides[].image when you want a multi-guide shot to
                        bias toward the scene's own composition (smudged
                        eyeliner, jacket, lighting) rather than the raw
                        character sheet anchors/narrator.png. Without this,
                        the model sees only the character's STUDIO portrait
                        + the establishing corridor and falls back to a
                        clean studio aesthetic that ignores the scene's
                        per-shot styling.
      <path>          → literal relative-to-project or absolute path
    Returns None for @none, or for @last at idx=1 with no anchor_image."""
    if ref == "@none":
        return None  # explicit t2v — no image conditioning, no audio slice
    if not ref or ref == "@last":
        if idx == 1:
            anchor = spec.get("anchor_image")
            if anchor:
                return str((project / anchor).resolve())
            return None  # use t2v
        prev = spec["scenes"][idx - 2]  # 1-indexed idx
        prev_stem = _scene_stem(idx - 1, prev.get("label", "scene"))
        last = project / "scenes" / f"{prev_stem}-last.png"
        if not last.exists():
            sys.exit(f"scene {idx}: @last requested but {last} does not exist "
                     f"(run scene {idx-1} first)")
        return str(last)
    if ref == "@anchor":
        anchor = spec.get("anchor_image")
        if not anchor:
            sys.exit(f"scene {idx}: @anchor requested but spec.anchor_image missing")
        return str((project / anchor).resolve())
    if ref == "@scene_anchor":
        scene = spec["scenes"][idx - 1]
        stem = _scene_stem(idx, scene.get("label", "scene"))
        scene_anchor = project / "scenes" / f"{stem}-anchor.png"
        if not scene_anchor.exists():
            sys.exit(f"scene {idx}: @scene_anchor requested but {scene_anchor} "
                     f"does not exist (run anchors phase first)")
        return str(scene_anchor)
    # Literal path (relative to project dir if not absolute)
    p = Path(ref)
    return str(p if p.is_absolute() else (project / p).resolve())


def _extract_last_frame(scene_mp4: Path, out_png: Path) -> None:
    # video_join.py is a `uv run --script` script with inline deps. Executing
    # the .py path directly works via shebang on POSIX but Windows rejects it
    # with "WinError 193: not a valid Win32 application". Invoke via `uv run
    # --script` everywhere so the same call works on both platforms.
    rc = subprocess.run(["uv", "run", "--script", str(VIDEO_JOIN), "last-frame",
                         "--input", str(scene_mp4), "--output", str(out_png)],
                        capture_output=True).returncode
    if rc != 0:
        sys.exit(f"failed to extract last frame of {scene_mp4}")


def _normalize_loras(scene: dict, vs: dict) -> list[dict]:
    """Normalize per-scene LoRA spec to a unified list of dicts.

    Two input shapes are supported:

    1. NEW (preferred):
         scene.loras: [{file, kind, strength, reference_video?, reference?,
                        reference_strength?}, ...]
       — chainable list. Today only `kind: ic_lora` is wired through; any
       other kind raises sys.exit (deferred until comfy_graph grows
       LoraLoaderModelOnly chaining flags).

    2. OLD (back-compat with every v8/v9 yaml in /workspace/handoffs/):
         scene.hdr_lora: {file, strength, reference|reference_video,
                          reference_strength, scene_emb?, scene_emb_strength?}
         video.hdr_lora: same shape (project-wide default).
       Converted to the new shape in-memory: file → ic_lora entry, and
       scene_emb (if set) → second ic_lora entry.

    Scene-level `loras:` and `hdr_lora:` both win over the video-level
    default. Scene-level `hdr_lora: null` (explicit) disables the default;
    scene-level `loras: []` is treated as "no LoRAs".

    Returns the normalized list (possibly empty). Does NOT touch the CLI
    surface — caller walks the list to assemble --ic_loras et al.
    """
    # Scene-level loras: short-circuits everything else.
    if "loras" in scene:
        scene_loras_raw = scene.get("loras") or []
        out: list[dict] = []
        for entry in scene_loras_raw:
            kind = entry.get("kind")
            if kind != "ic_lora":
                sys.exit(
                    f"scene loras[*].kind={kind!r} not yet supported — "
                    f"only `ic_lora` is wired through comfy_graph today. "
                    f"Standard LoRA chaining (LoraLoaderModelOnly) needs "
                    f"a new --extra-lora CLI flag on comfy_graph.py."
                )
            out.append(dict(entry))
        return out

    # Old hdr_lora shape (scene override or video default). An explicit
    # `scene.hdr_lora: null` disables the video-level default.
    if "hdr_lora" in scene:
        h = scene.get("hdr_lora")
    else:
        h = vs.get("hdr_lora")
    if not h:
        return []
    scene_loras = [{
        "file": h["file"],
        "kind": "ic_lora",
        "strength": h.get("strength", 1.0),
        "reference_video": h.get("reference_video"),
        "reference_image": h.get("reference"),
        "reference_strength": h.get("reference_strength", 1.0),
    }]
    if h.get("scene_emb"):
        scene_loras.append({
            "file": h["scene_emb"],
            "kind": "ic_lora",
            "strength": h.get("scene_emb_strength", 0.8),
        })
    return scene_loras


def cmd_scene(spec: dict, project: Path, idx: int) -> None:
    scenes = spec.get("scenes", [])
    if idx < 1 or idx > len(scenes):
        sys.exit(f"scene index out of range: {idx} (have {len(scenes)})")
    scene = scenes[idx - 1]
    if scene["duration_sec"] > MAX_SCENE_DURATION:
        sys.exit(f"scene {idx} '{scene.get('label','?')}' duration "
                 f"{scene['duration_sec']}s exceeds MAX_SCENE_DURATION "
                 f"{MAX_SCENE_DURATION}s — LTX-2.3 OOMs past ~20s at portrait "
                 f"resolutions. Split this into shorter scenes.")
    stem = _scene_stem(idx, scene.get("label", "scene"))
    out_mp4 = project / "scenes" / f"{stem}.mp4"
    last_png = project / "scenes" / f"{stem}-last.png"

    if out_mp4.exists():
        if not last_png.exists():
            # mp4 was dropped in manually (or the -last.png was deleted).
            # Extract the last frame via ffmpeg so downstream @last chains
            # still work, then skip the render.
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
            rc = subprocess.run([
                ffmpeg, "-y", "-loglevel", "error",
                "-sseof", "-0.1", "-i", str(out_mp4),
                "-vframes", "1", str(last_png),
            ]).returncode
            if rc != 0:
                _log(project, f"scene {idx} '{stem}' mp4 present but "
                              f"last-frame extraction failed (rc={rc}); "
                              f"deleting mp4 to force re-render")
                out_mp4.unlink()
            else:
                _log(project, f"scene {idx} '{stem}' mp4 present, "
                              f"extracted last frame, skipping render")
                return
        else:
            _log(project, f"scene {idx} '{stem}' already exists, skipping")
            return

    if not (project / "song.mp3").exists():
        sys.exit("song.mp3 missing — run `song` first")

    vs = _video_spec(spec)
    # Lipsync scenes render with a tail buffer so LTX has audio lookhead to
    # close the phoneme; the buffer tail is trimmed off at assemble time
    # (GetImageRangeFromBatch) so the final timeline is exact. t2v scenes
    # don't need audio lookhead — render at duration_sec exactly.
    is_lipsync = bool(scene.get("anchor"))
    tail_buffer = float(vs.get("tail_buffer_sec", 0.0))
    # tail_buffer always applied — LTX's ×8 latent temporal compression
    # frequently rounds frame counts DOWN, so without a buffer the rendered
    # mp4 lands short of duration_sec (e.g. 4.21s → 4.04s). Padding here +
    # trimming in storyboard/assemble.py keeps audio in lockstep with video.
    # The lipsync flag historically gated this for vocal-margin reasons but
    # the trim step already drops the tail, so all scenes can share it.
    effective_duration = float(scene["duration_sec"]) + tail_buffer

    # Audio source resolution for ia2v conditioning. Priority:
    #   1. scene.lipsync_audio (per-scene override; pass null/None to force
    #      song.mp3 even when a global override is set — useful for
    #      instrumental-driven scenes where the vocal-forward remix would
    #      muddy the music dynamics).
    #   2. video.lipsync_audio (project-wide vocal-forward remix; LTX's
    #      audio head locks onto lyric content best when vocals + backing
    #      vocals are boosted ~6dB over the instrumental bed).
    #   3. song.mp3 (default; full mix).
    # song.mp3 always wins at assembly so the final track is the original.
    if "lipsync_audio" in scene:
        lipsync_audio_name = scene["lipsync_audio"]
    else:
        lipsync_audio_name = vs.get("lipsync_audio")
    lipsync_src = (project / lipsync_audio_name).resolve() if lipsync_audio_name else None
    if lipsync_src and not lipsync_src.exists():
        sys.exit(f"lipsync_audio for scene {idx} points at {lipsync_src} which does not exist")
    slice_mp3 = _slice_song(project, scene, stem,
                            tail_buffer_sec=tail_buffer if is_lipsync else 0.0,
                            source_path=lipsync_src)

    # Prefer a scene-specific pre-generated anchor (flux2) when scene.anchor is set.
    # Falls back to the @last/@anchor/literal image chain otherwise.
    pre_gen_anchor = _generate_anchor(spec, project, idx, scene, stem)
    if pre_gen_anchor:
        image_path = pre_gen_anchor
    else:
        image_path = _resolve_image(spec, project, idx, scene.get("image"))

    # comfy_graph takes <cmd> as the first positional, then --opts.
    prefix = stem
    common = ["--seconds", f"{effective_duration:.2f}",
              "--fps",     str(vs["fps"]),
              "--width",   str(vs["width"]),
              "--height",  str(vs["height"]),
              "--prefix",  prefix,
              "--output-dir", str(project / "scenes"),
              "--timeout", "1800",       # 30 min — LTX 1024x576 10s quality can need 8-12 min
              "--prompt",  _expand_subjects(scene["prompt"], spec)]
    if vs["negative"]:
        common += ["--negative", vs["negative"]]

    # Per-scene camera LoRA → stacked on top of the distilled LoRA. Accepts
    # a shortname (dolly-in/out/left/right, jib-up/down, static) or a full
    # .safetensors filename. Top-level spec.video.camera_lora gives a default.
    scene_cam = scene.get("camera_lora") or (spec.get("video") or {}).get("camera_lora")
    if scene_cam:
        common += ["--camera-lora", scene_cam]
    scene_cam_s = scene.get("camera_lora_strength") or (spec.get("video") or {}).get("camera_lora_strength")
    if scene_cam_s is not None:
        common += ["--camera-lora-strength", str(scene_cam_s)]

    if scene.get("fast") or (spec.get("video") or {}).get("fast"):
        common += ["--fast"]

    # IC-LoRA stack — unified via _normalize_loras which accepts both the
    # new `scene.loras: [...]` shape and the legacy `scene.hdr_lora:` /
    # `video.hdr_lora:` shape. The list may contain multiple ic_lora
    # entries (e.g. union-control + HDR + future scene-emb), all stacked
    # through the same --ic_loras pair list. Reference video/image is
    # taken from the FIRST entry that declares one; only one reference
    # per scene is supported today (the LTXAddVideoICLoRAGuide chain in
    # ltx2.py emits a single guide using the last loader's
    # latent_downscale_factor — see open question in the design doc).
    scene_loras = _normalize_loras(scene, vs)
    if scene_loras:
        ic_pairs: list[str] = [
            f"{e['file']}:{float(e.get('strength', 1.0)):.2f}" for e in scene_loras
        ]
        # First entry that declares a reference wins; any subsequent
        # entry trying to declare its own reference is an error today.
        ref_entry: dict | None = None
        for e in scene_loras:
            if e.get("reference_video") or e.get("reference_image"):
                if ref_entry is None:
                    ref_entry = e
                else:
                    sys.exit(
                        "multi-reference IC-LoRA chains not yet supported — "
                        "needs ltx2.py guide-per-loader patch. Today "
                        "LTXAddVideoICLoRAGuide is emitted once per scene "
                        "using the last loader's latent_downscale_factor; "
                        f"second reference declared on {e.get('file')!r} "
                        "cannot be honoured."
                    )
        if ref_entry is None:
            sys.exit(
                "scene IC-LoRA stack needs `reference` (image) OR "
                "`reference_video` (mp4) on at least one entry — the LoRA "
                "weights only act in concert with the LTXAddVideoICLoRAGuide "
                "chain that consumes the reference."
            )
        ref_video = ref_entry.get("reference_video")
        ref = ref_entry.get("reference_image")
        # Static-image reference (HDR-style frame-0 bias): --ic_lora_reference.
        # Video reference (depth/canny/motion-track per-frame conditioning):
        # --ic_lora_reference_video. The video form bypasses
        # ImagePrepForICLora server-side and uses ResizeImageMaskNode
        # 'scale to multiple' — what the official Lightricks workflow does
        # and what gives full-frame coverage on portrait outputs (vs the
        # ImagePrepForICLora left-bias bug on the square-prep path).
        common += ["--ic_loras", ",".join(ic_pairs),
                   "--ic_lora_reference_strength",
                   str(float(ref_entry.get("reference_strength", 1.0)))]
        if ref_video:
            rv_path = (project / ref_video).resolve() if not Path(ref_video).is_absolute() else Path(ref_video)
            if not rv_path.exists():
                sys.exit(f"scene IC-LoRA reference_video points at {rv_path} which does not exist")
            common += ["--ic_lora_reference_video", str(rv_path)]
        else:
            ref_path = (project / ref).resolve() if not Path(ref).is_absolute() else Path(ref)
            if not ref_path.exists():
                sys.exit(f"scene IC-LoRA reference points at {ref_path} which does not exist")
            common += ["--ic_lora_reference", str(ref_path)]

    # guides: optional yaml list of additional keyframe guides at specified
    # positions within the shot. When present, route to `multiguide`
    # (chained LTXVAddGuides) instead of plain ia2v — the first-frame
    # anchor (image_path) is prepended at frame 0 automatically via
    # ensure_frame_zero so scenes starting with a character-less
    # establishing frame (e.g. a candle) can still carry a strong
    # character anchor mid-shot.
    guides_yaml = scene.get("guides")

    if image_path is None and not guides_yaml:
        # First scene with no anchor → fall back to t2v (no audio slice).
        cmd = ["python3", str(COMFY), "t2v"] + common
    elif guides_yaml:
        import importlib.util
        # Resolve storyboard/lib/guides.py from a sandbox install, the host
        # venetanji install, or — when neither exists — from the repo
        # checkout sibling to this script (../../storyboard/lib/guides.py).
        sb_candidates = [
            Path("/home/sandbox/.openclaw/skills/storyboard/lib/guides.py"),
            Path.home() / ".openclaw/skills/storyboard/lib/guides.py",
            Path(__file__).resolve().parent.parent.parent / "storyboard/lib/guides.py",
        ]
        sb_lib = next((p for p in sb_candidates if p.exists()), sb_candidates[-1])
        spec_gm = importlib.util.spec_from_file_location("storyboard_guides", sb_lib)
        gm = importlib.util.module_from_spec(spec_gm); spec_gm.loader.exec_module(gm)

        # Resolve at_sec / at_relative / at_frame / at_song_sec for each
        # guide entry. scene_start_sec lets specs use song-absolute time
        # (e.g. drum hit at 73.14s) without subtracting the scene's start
        # by hand — handy when the guide list is authored from MIDI markers.
        resolved = gm.resolve_guides(
            guides_yaml,
            duration_sec=float(effective_duration),
            fps=int(vs["fps"]),
            project_dir=project,
            token_resolver=lambda tok: _resolve_image(spec, project, idx, tok),
            scene_start_sec=float(scene.get("start_sec", 0.0)),
        )
        # Prepend frame-0 anchor if caller didn't put one there — the
        # shot's `image:` (already resolved to image_path) fills that slot.
        if image_path:
            gm.ensure_frame_zero(resolved, image_path=image_path, strength=1.0)

        guides_paths = ",".join(str(p) for _, p, _ in resolved)
        frame_indices = ",".join(str(f) for f, _, _ in resolved)
        strengths = ",".join(f"{s:.3f}" for _, _, s in resolved)
        cmd = ["python3", str(COMFY), "multiguide",
               "--guides", guides_paths,
               "--frame_indices", frame_indices,
               "--strengths", strengths,
               "--audio", str(slice_mp3),
               "--no_transition_lora", "1",
               ] + common
    else:
        # For lipsync scenes, add previous scene's last frame as a second
        # IMAGE reference so LTX has identity + scene-to-scene continuity
        # without hard-pasting the anchor as frame 0.
        ia2v_extras: list[str] = []
        if is_lipsync and idx > 1:
            prev_label = spec["scenes"][idx - 2].get("label", "scene")
            prev_last = project / "scenes" / f"{_scene_stem(idx - 1, prev_label)}-last.png"
            if prev_last.exists():
                ia2v_extras += ["--image-refs", str(prev_last)]
        # ia2v guide strengths — control how strongly the anchor PNG locks
        # the rendered scene's opening frames. ltx2.py's hardcoded defaults
        # are base=0.5 / refine=0.3 (deliberately weak — model has motion
        # freedom). For music-video work where the anchor was authored to
        # define composition + lighting + character placement, those
        # defaults throw the composition away: the singer's face survives
        # but the corridor / stage / kitchen environment is reinvented
        # from prompt. Default here to 0.9 / 0.7 so anchor PNGs actually
        # drive the scene's first frame. Override via video.* (project)
        # or scene[].* (per-scene). Drop to ~0.5 / 0.3 on action shots
        # where motion freedom matters more than first-frame fidelity.
        base_gs = scene.get("base_guide_strength",
                            vs.get("base_guide_strength", 0.9))
        refine_gs = scene.get("refine_guide_strength",
                              vs.get("refine_guide_strength", 0.7))
        ia2v_extras += ["--base_guide_strength", f"{float(base_gs):.2f}",
                        "--refine_guide_strength", f"{float(refine_gs):.2f}"]
        cmd = ["python3", str(COMFY), "ia2v",
               "--image", image_path,
               "--audio", str(slice_mp3)] + ia2v_extras + common

    _log(project, f"scene {idx} '{stem}': {scene['prompt'][:80]}")
    rc = _run(cmd, log_path=project / "run.log",
              env_override=_comfy_env_for("video"))
    if rc != 0:
        sys.exit(f"scene {idx} generation failed (rc={rc})")

    # comfy_graph saves to <output-dir>/<prefix>_NNNNN_.mp4 (comfy auto-numbers).
    # Find the most recent one matching prefix.
    candidates = sorted((project / "scenes").glob(f"{prefix}_*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(f"scene {idx}: no MP4 matched '{prefix}_*.mp4' in scenes/")
    produced = candidates[0]
    if produced != out_mp4:
        produced.rename(out_mp4)
    _extract_last_frame(out_mp4, last_png)
    _log(project, f"scene {idx} done → {out_mp4.name}, last frame → {last_png.name}")


def _probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffmpeg -i stderr parse."""
    r = subprocess.run([FFMPEG(), "-i", str(path)], capture_output=True, text=True)
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        return 0.0
    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mn * 60 + s


def _transition_duration(spec, idx) -> float:
    """Effective transition duration for the boundary between scenes
    idx→idx+1 (1-indexed). Returns 0 if transitions are disabled."""
    vs = spec.get("video") or {}
    tv = vs.get("transitions") or {}
    if not tv.get("enabled"):
        return 0.0
    scenes = spec.get("scenes", [])
    if idx < 1 or idx >= len(scenes):
        return 0.0
    # Per-boundary override lives on the INCOMING scene (scenes[idx])
    incoming = scenes[idx]
    override = (incoming.get("transition_from_prev") or {}).get("duration")
    return float(override if override is not None else tv.get("duration", 2.0))


def cmd_transitions(spec: dict, project: Path) -> None:
    """Render transition clips between each adjacent scene pair using real
    video segments as guides (not still images). Each transition is
    (guide_sec) real frames of scene A's tail → (empty_sec) empty-latent
    invention → (guide_sec) real frames of scene B's head. The LoRA drives
    the morph through the middle, and the real-video guides give it motion
    to blend *into* rather than a static endpoint to converge on.

    Song sync: transition occupies dur seconds of song time [a_end - dur/2,
    a_end + dur/2]. The guide blocks are cut from prev_video / next_video
    at the song-matched frame ranges so when the transition plays it looks
    continuous with scene A's native motion and ends on scene B's native
    motion. Controlled by `video.transitions.{duration, guide_sec}` in spec
    (default dur=4s, guide_sec=1s → 1/2/1 split).

    Transitions run on COMFY_URL_FLUX (the 12GB GPU) so they can render in
    parallel with the main scene batch. Restart-safe."""
    vs = spec.get("video") or {}
    tv = vs.get("transitions") or {}
    if not tv.get("enabled"):
        _log(project, "transitions disabled (video.transitions.enabled: false)")
        return
    default_prompt = tv.get("prompt",
        "a smooth cinematic morph between scenes, matching lighting and mood, "
        "continuous motion across the cut")
    default_fps = int(tv.get("fps", 24))
    guide_sec = float(tv.get("guide_sec", 1.0))
    tail_buffer = float(vs.get("tail_buffer_sec", 0.0))

    scenes = spec.get("scenes", [])
    if len(scenes) < 2:
        return
    tdir = project / "transitions"
    tdir.mkdir(exist_ok=True)
    ffmpeg = FFMPEG()

    for i in range(1, len(scenes)):
        dur = _transition_duration(spec, i)
        if dur <= 0:
            continue
        if dur < 2 * guide_sec + 0.25:
            _log(project, f"transition {i}→{i+1} skipped — dur={dur}s too short "
                          f"for guide_sec={guide_sec}s on each side")
            continue
        a_idx = i          # outgoing scene (1-indexed)
        b_idx = i + 1      # incoming scene (1-indexed)
        a = scenes[a_idx - 1]
        b = scenes[b_idx - 1]
        a_stem = _scene_stem(a_idx, a.get("label", "scene"))
        b_stem = _scene_stem(b_idx, b.get("label", "scene"))
        out_mp4 = tdir / f"{a_stem}__{b_stem}.mp4"
        if out_mp4.exists():
            _log(project, f"transition {a_idx}→{b_idx} already exists, skipping")
            continue

        a_mp4 = project / "scenes" / f"{a_stem}.mp4"
        b_mp4 = project / "scenes" / f"{b_stem}.mp4"
        if not a_mp4.exists() or not b_mp4.exists():
            _log(project, f"transition {a_idx}→{b_idx} skipped — "
                          f"prerequisites missing (a_mp4={a_mp4.exists()} "
                          f"b_mp4={b_mp4.exists()})")
            continue

        # Audio slice covering the boundary ± dur/2
        a_end_song = float(a["start_sec"]) + float(a["duration_sec"])
        slice_start = max(0.0, a_end_song - dur / 2)
        audio_slice = tdir / f"{a_stem}__{b_stem}-audio.mp3"
        if not audio_slice.exists():
            subprocess.run([ffmpeg, "-y", "-ss", f"{slice_start:.3f}",
                            "-t", f"{dur:.3f}",
                            "-i", str(project / "song.mp3"),
                            "-c:a", "libmp3lame", "-b:a", "192k",
                            str(audio_slice)],
                           capture_output=True)

        # Per-boundary overrides on scene B's transition_from_prev block.
        # Keys supported:
        #   prompt: string override for the transition prompt
        #   b_sparse: comma-sep latent positions or list of ints.
        #            Default "96" (= 1f, single B-anchor at the tail).
        #            Use "72,80,88,96" (= 4f) on boundaries INTO lipsync/
        #            singing scenes for smoother character establishment.
        bttp = (b.get("transition_from_prev") or {})
        prompt = bttp.get("prompt", default_prompt)
        b_sparse = bttp.get("b_sparse") or tv.get("default_b_sparse", "96")
        if isinstance(b_sparse, list):
            b_sparse_str = ",".join(str(int(x)) for x in b_sparse)
        else:
            b_sparse_str = str(b_sparse)

        # Guide-block frame counts must be multiples of 8 per LTX's latent
        # temporal quantization. 1s @ 24fps = 24 frames → ok.
        guide_frames = max(8, (int(guide_sec * default_fps) // 8) * 8)
        empty_start_sec = guide_sec
        empty_end_sec = dur - guide_sec

        prefix = f"trans_{a_idx}_{b_idx}"
        # Video-guide transition via LTXVAddGuide:
        #   A side:  24-frame batch at latent [0, 23], strength 1.0
        #            (prev_video's tail seen from slice_song_start)
        #   B side:  SPARSE — single frame at latent 96 (scene-B head),
        #            strength 1.0. Multi-frame contiguous B-blocks caused
        #            a mid-transition freeze at the snap-in (scene
        #            content hard-locking over 24 frames); a single anchor
        #            at the end lets the LoRA morph through the empty
        #            region and land cleanly on scene B's first in-stitch
        #            frame without a snap.
        #   Mask:    1.0-4.0s (audio drives invented motion from [24, 96])
        cmd = ["python3", str(COMFY), "transition",
               "--prev_video", str(a_mp4),
               "--next_video", str(b_mp4),
               "--audio", str(audio_slice),
               "--prompt", prompt,
               "--seconds", f"{dur:.2f}",
               "--fps", str(default_fps),
               "--width", str(vs["resolution"][0] if isinstance(vs.get("resolution"), list) else 448),
               "--height", str(vs["resolution"][1] if isinstance(vs.get("resolution"), list) else 832),
               "--prefix", prefix,
               "--output-dir", str(tdir),
               "--multiframe_guide", str(guide_frames),
               "--multiframe_guide_last", "1",
               "--first_guide_strength", "1.0",
               "--last_guide_strength", "1.0",
               "--use_addguide", "1",
               "--no_inplace", "1",
               "--b_sparse_latent_positions", b_sparse_str,
               "--slice_song_start_sec", f"{slice_start:.3f}",
               "--prev_video_song_start_sec", f"{float(a['start_sec']):.3f}",
               "--next_video_song_start_sec", f"{float(b['start_sec']):.3f}",
               "--prev_video_buffer", f"{tail_buffer:.3f}",
               "--mask_start_sec", "1.0",
               "--mask_end_sec", "4.0",
               "--timeout", "1800"]
        # Inherit fast/slow mode from the same knob the scene renders use.
        # Without this transitions silently render full 2-pass (LTXVLatentUpsampler
        # + 3-step refine) while scenes ran fast=true single-pass — they take
        # 2× the wall time of a fast scene and the quality mismatch shows up
        # as a visible sharpness pop at every transition.
        # Per-boundary override: video.transitions.fast (true|false) trumps
        # video.fast for transitions only — useful if you want fast scene
        # iteration but final-quality morph clips, or vice versa.
        trans_fast = tv.get("fast")
        if trans_fast is None:
            trans_fast = bool(vs.get("fast"))
        if trans_fast:
            cmd += ["--fast"]
        _log(project, f"transition {a_idx}→{b_idx}: {dur}s "
                      f"({guide_sec}s A-guide + {empty_end_sec - empty_start_sec}s empty + "
                      f"{guide_sec}s B-guide), audio from {slice_start:.2f}s")
        rc = _run(cmd, log_path=project / "run.log",
                  env_override=_comfy_env_for("flux"))  # flux comfy = parallel with main
        if rc != 0:
            _log(project, f"transition {a_idx}→{b_idx} failed (rc={rc})")
            continue
        cands = sorted(tdir.glob(f"{prefix}_*.mp4"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands and cands[0] != out_mp4:
            cands[0].rename(out_mp4)
        _log(project, f"transition {a_idx}→{b_idx} done → {out_mp4.name}")


def cmd_assemble(spec: dict, project: Path) -> None:
    """Concat + overlay-audio inside ComfyUI via post.concat_videos.

    Without transitions: each scene trimmed from frame 0 to duration_sec
    (drops lipsync tail buffer). Hard cuts. Scene N+1's frame 0 lands at
    sum(duration_1..N) = song timecode of its vocal phrase.

    With transitions (video.transitions.enabled): interleaves scene/
    transition/scene/... The scene flanking a transition gives up
    dur/2 on that side (skip-front and/or trim-back). Each transition
    adds `dur` seconds. Net change per boundary is zero, so the song
    timeline stays locked regardless of how many transitions you use."""
    scenes = spec.get("scenes", [])
    n = len(scenes)
    if n == 0:
        sys.exit("spec has no scenes")

    song = project / "song.mp3"
    if not song.exists():
        sys.exit("song.mp3 missing — run `song` first")

    vs = spec.get("video") or {}
    fps = int(vs.get("fps", 24))
    tv = vs.get("transitions") or {}
    trans_enabled = bool(tv.get("enabled"))

    clip_paths: list[Path] = []
    starts: list[float] = []
    durations: list[float] = []

    for i, s in enumerate(scenes, 1):
        stem = _scene_stem(i, s.get("label", "scene"))
        p = project / "scenes" / f"{stem}.mp4"
        if not p.exists():
            sys.exit(f"scene {i} MP4 missing at {p} — run it first")

        dur_in  = _transition_duration(spec, i - 1) if trans_enabled and i > 1 else 0.0
        dur_out = _transition_duration(spec, i)      if trans_enabled and i < n else 0.0
        skip_front = dur_in / 2.0
        keep = float(s["duration_sec"]) - dur_in / 2.0 - dur_out / 2.0
        # Guard: if a scene is shorter than its flanking transitions consume,
        # `keep` goes negative and ffmpeg refuses the trim. Rather than crash
        # on the whole assemble, shrink each transition proportionally so the
        # scene retains at least MIN_NATIVE_KEEP of native content. Whichever
        # transition gets clamped here will look slightly out-of-sync with
        # its rendered duration but the alternative is a crashed assemble.
        MIN_NATIVE_KEEP = 0.25
        if keep < MIN_NATIVE_KEEP:
            shortfall = (MIN_NATIVE_KEEP - keep)
            # Each transition gives up half the shortfall, capped at its own
            # current half-share so we don't go negative on dur_*.
            give_in = min(dur_in / 2.0, shortfall / 2.0)
            give_out = min(dur_out / 2.0, shortfall - give_in)
            dur_in = max(0.0, dur_in - 2.0 * give_in)
            dur_out = max(0.0, dur_out - 2.0 * give_out)
            skip_front = dur_in / 2.0
            keep = float(s["duration_sec"]) - dur_in / 2.0 - dur_out / 2.0
            keep = max(MIN_NATIVE_KEEP, keep)
            print(f"[WARN] scene {i} '{s.get('label')}' shorter than its transitions "
                  f"({s['duration_sec']:.2f}s vs {dur_in/2.0+dur_out/2.0:.2f}s consumed); "
                  f"shrunk to dur_in={dur_in:.2f}/dur_out={dur_out:.2f}, keep={keep:.2f}",
                  file=sys.stderr)

        clip_paths.append(p)
        starts.append(skip_front)
        durations.append(keep)

        if trans_enabled and i < n and dur_out > 0:
            nxt_stem = _scene_stem(i + 1, scenes[i].get("label", "scene"))
            tp = project / "transitions" / f"{stem}__{nxt_stem}.mp4"
            if not tp.exists():
                sys.exit(f"transition {i}→{i+1} MP4 missing at {tp} — "
                         f"run `transitions` pass first")
            clip_paths.append(tp)
            starts.append(0.0)
            durations.append(dur_out)

    song_variants = [("final", song)]
    for candidate in sorted(project.glob("song_v*.mp3")):
        sidx = candidate.stem.replace("song_v", "")
        song_variants.append((f"final_v{sidx}", candidate))

    # Delegate to the shared storyboard assemble helper (same pipeline
    # drama-video uses). Pre-trim → strip audio → concat → mux song.
    ffmpeg = FFMPEG()
    import importlib.util
    sb_candidates = [
        Path("/home/sandbox/.openclaw/skills/storyboard/lib/assemble.py"),
        Path.home() / ".openclaw/skills/storyboard/lib/assemble.py",
        Path(__file__).resolve().parent.parent.parent / "storyboard/lib/assemble.py",
    ]
    sb_lib = next((p for p in sb_candidates if p.exists()), sb_candidates[-1])
    m_spec = importlib.util.spec_from_file_location("storyboard_assemble", sb_lib)
    asm = importlib.util.module_from_spec(m_spec); m_spec.loader.exec_module(asm)

    clip_tuples = [(src, float(start), float(dur))
                   for src, start, dur in zip(clip_paths, starts, durations)]

    for prefix, song_path in song_variants:
        final = project / f"{prefix}.mp4"
        _log(project, f"assembling {prefix}.mp4 ({len(clip_paths)} clips"
                      f"{', with transitions' if trans_enabled else ''}, "
                      f"audio={song_path.name})")
        asm.assemble_clips(
            clips=clip_tuples, audio_file=song_path, output=final,
            work_dir=project / "scenes" / "_assemble_cache", ffmpeg=ffmpeg)
        _log(project, f"assembled → {final} "
                      f"({final.stat().st_size//1024 if final.exists() else '?'} KB)")


def cmd_all(spec: dict, project: Path, no_gate: bool = False) -> None:
    """End-to-end pipeline with two confirm-before-render gates:

      1. After `song` — STOP if spec.gate_confirm_song (default true) so a
         human can listen to the N*2 variants, pick the best, and rename it
         to song.mp3. Bypass with --no-gate or gate_confirm_song: false.
      2. After `anchors` — STOP if spec.gate_confirm_anchors (default true)
         so a human can inspect the generated anchor images before committing
         GPU time to the per-scene LTX renders. Bypass likewise.

    Without the gates the pipeline runs fully unattended."""
    _ensure_dirs(project)
    cmd_song(spec, project)

    # --- GATE 1: song confirmation ---
    gate_song = bool(spec.get("gate_confirm_song", True)) and not no_gate
    if gate_song:
        variants = []
        if (project / "song.mp3").exists():
            variants.append(project / "song.mp3")
        variants += sorted(project.glob("song_v*.mp3"))
        print()
        print("=" * 72)
        print(" QUALITY GATE 1/2 — song variants generated, human review required")
        print("=" * 72)
        for p in variants:
            try:
                d = _probe_duration(p)
            except Exception:
                d = 0.0
            print(f"  {p.name:20}  {d:6.1f}s  {p.stat().st_size//1024} KB")
        print()
        print(" Next steps:")
        print("   1. Listen to each variant.")
        print("   2. Pick the best. Rename it to `song.mp3` (back up the current one).")
        print("   3. Add scenes to song.yaml.")
        print("   4. Re-run `music_video.py all <spec>` (or `all --no-gate` to skip gates).")
        print()
        print(" Bypass: set `gate_confirm_song: false` in song.yaml, or pass --no-gate.")
        print("=" * 72)
        return

    # Hard check: scene total must cover the longest song variant, otherwise
    # the assembly will freeze-frame the tail for potentially minutes. Fail
    # loudly BEFORE burning GPU time on the scenes.
    scenes = spec.get("scenes") or []
    if not scenes:
        sys.exit("spec has no scenes — add scenes to song.yaml, then re-run `all`.")
    assembled = sum(s["duration_sec"] for s in scenes)
    longest_song = 0.0
    for p in [project / "song.mp3", *sorted(project.glob("song_v*.mp3"))]:
        if p.exists():
            longest_song = max(longest_song, _probe_duration(p))
    gap = longest_song - assembled
    if gap > 3:                          # >3s of freeze-frame is the problem threshold
        _log(project, f"⚠  song ({longest_song:.1f}s) is {gap:.1f}s LONGER "
                      f"than assembled scenes ({assembled:.1f}s) — the tail "
                      f"will freeze on the last frame for that duration.")
        _log(project, f"   add ~{int(gap/12)+1} more scenes (at 12s each) or "
                      f"extend existing durations up to 20s each, then rerun.")
        sys.exit("aborting before scene generation — scene total too short for song.")

    # --- anchors pass (top-level + per-scene flux2) ---
    cmd_anchors(spec, project)

    # --- GATE 2: anchors confirmation ---
    gate_anchors = bool(spec.get("gate_confirm_anchors", True)) and not no_gate
    if gate_anchors:
        print()
        print("=" * 72)
        print(" QUALITY GATE 2/2 — anchor images generated, human review required")
        print("=" * 72)
        anchor_imgs: list[Path] = []
        top = spec.get("anchor_image")
        if top and (project / top).exists():
            anchor_imgs.append((project / top).resolve())
        anchor_imgs += sorted((project / "scenes").glob("*-anchor.png"))
        for p in anchor_imgs:
            print(f"  {p.relative_to(project)}")
        print()
        print(" Next steps:")
        print("   1. Open these PNGs (filebrowser / image viewer).")
        print("   2. If an anchor looks wrong, DELETE it and re-run `anchors <spec>`")
        print("      (tweak the scene's anchor.prompt or top-level anchor_prompt).")
        print("   3. When anchors look right, re-run `all --no-gate <spec>` or set")
        print("      `gate_confirm_anchors: false` in song.yaml and re-run.")
        print("=" * 72)
        return

    for i in range(1, len(scenes) + 1):
        cmd_scene(spec, project, i)
    # Render transitions (no-op if video.transitions.enabled != true).
    # Runs on COMFY_URL_FLUX so they parallelize with... well, here they run
    # serially after scenes, but `transitions` can also be invoked standalone
    # during the main batch on a second terminal.
    cmd_transitions(spec, project)
    cmd_assemble(spec, project)


def cmd_scenes(spec: dict, project: Path) -> None:
    """Render every scene in sequence. Restart-safe (skips scenes already done).
    Refuses to submit if ANY scene exceeds MAX_SCENE_DURATION — a too-long
    clip can OOM-kill the comfy server and wipe the whole queue, forcing
    everyone to re-submit."""
    scenes = spec.get("scenes") or []
    if not scenes:
        sys.exit("spec has no scenes")
    if not (project / "song.mp3").exists():
        sys.exit("song.mp3 missing — run `song` first (and confirm the chosen variant)")
    too_long = [
        (i, s) for i, s in enumerate(scenes, 1)
        if float(s.get("duration_sec", 0)) > MAX_SCENE_DURATION
    ]
    if too_long:
        lines = [
            f"  scene {i} '{s.get('label','?')}': {s['duration_sec']}s"
            for i, s in too_long
        ]
        sys.exit("refusing to render — some scenes exceed MAX_SCENE_DURATION "
                 f"({MAX_SCENE_DURATION}s). Split these into shorter shots:\n"
                 + "\n".join(lines))
    for i in range(1, len(scenes) + 1):
        cmd_scene(spec, project, i)


def cmd_status(spec: dict, project: Path) -> None:
    song = project / "song.mp3"
    print(f"song.mp3          : {'✓' if song.exists() else '—'}")
    scenes = spec.get("scenes", [])
    for i, s in enumerate(scenes, 1):
        stem = _scene_stem(i, s.get("label", "scene"))
        mp4 = project / "scenes" / f"{stem}.mp4"
        png = project / "scenes" / f"{stem}-last.png"
        mark = "✓" if mp4.exists() and png.exists() else ("video" if mp4.exists() else "—")
        print(f"scene {i:>2} {stem:30} : {mark}")
    final = project / "final.mp4"
    print(f"final.mp4         : {'✓' if final.exists() else '—'}")


# ---------- entry ----------

def main() -> None:
    p = argparse.ArgumentParser(prog="music_video")
    sub = p.add_subparsers(dest="cmd", required=True)

    # `init` takes a slug, not a spec path.
    si = sub.add_parser("init", help="create a new project skeleton")
    si.add_argument("slug")
    si.add_argument("--theme", default=None,
                    help="optional theme/brief — injected as a comment hint")
    si.add_argument("--force", action="store_true",
                    help="overwrite an existing project dir")

    for c in ["plan", "song", "anchors", "scenes", "assemble", "transitions",
              "status"]:
        sp = sub.add_parser(c)
        sp.add_argument("spec")

    sa = sub.add_parser("all")
    sa.add_argument("spec")
    sa.add_argument("--no-gate", action="store_true",
                    help="bypass both confirm-before-render gates")

    ss = sub.add_parser("scene")
    ss.add_argument("idx", type=int)
    ss.add_argument("spec")

    args = p.parse_args()

    if args.cmd == "init":
        cmd_init(args.slug, args.theme, args.force)
        return

    spec, project = _load_spec(args.spec)
    _ensure_dirs(project)

    if args.cmd == "plan":      cmd_plan(spec, project)
    elif args.cmd == "song":    cmd_song(spec, project)
    elif args.cmd == "anchors": cmd_anchors(spec, project)
    elif args.cmd == "scene":   cmd_scene(spec, project, args.idx)
    elif args.cmd == "scenes":  cmd_scenes(spec, project)
    elif args.cmd == "assemble":cmd_assemble(spec, project)
    elif args.cmd == "transitions": cmd_transitions(spec, project)
    elif args.cmd == "all":     cmd_all(spec, project, no_gate=args.no_gate)
    elif args.cmd == "status":  cmd_status(spec, project)


if __name__ == "__main__":
    main()
