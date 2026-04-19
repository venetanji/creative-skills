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

import yaml
from imageio_ffmpeg import get_ffmpeg_exe


# ---------- paths & helpers ----------

# Skill root is the parent of scripts/
SKILL_ROOT = Path(__file__).resolve().parent.parent
# The comfyui and suno-mcp skills are sibling dirs under ~/.openclaw/skills/
SKILLS_ROOT = SKILL_ROOT.parent
COMFY = SKILLS_ROOT / "comfyui" / "scripts" / "comfy_graph.py"
SUNO  = SKILLS_ROOT / "suno-mcp" / "scripts" / "generate_song.py"
VIDEO_JOIN = SKILLS_ROOT / "comfyui" / "scripts" / "video_join.py"

FFMPEG = get_ffmpeg_exe


def _log(project: Path, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(project / "run.log", "a") as f:
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
                with open(log_path, "a") as f:
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
    spec = yaml.safe_load(p.read_text())
    return spec, p.parent  # project dir = dir containing the yaml


def _scene_stem(idx: int, label: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40] or "scene"
    return f"{idx:03d}-{safe}"


def _ensure_dirs(project: Path) -> None:
    (project / "song_slices").mkdir(exist_ok=True)
    (project / "scenes").mkdir(exist_ok=True)


def _video_spec(spec: dict) -> dict:
    v = spec.get("video", {})
    w, h = (v.get("resolution") or [1024, 576])
    return {
        "fps": int(v.get("fps", 24)),
        "width": int(w), "height": int(h),
        "negative": v.get("negative"),
    }


# ---------- subcommands ----------

MAX_SCENE_DURATION = 20.0   # LTX OOMs at portrait resolutions past ~20s per scene


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
    print(f"{'#':>3}  {'label':18}  {'start':>6}  {'dur':>5}  image  prompt")
    for i, s in enumerate(scenes, 1):
        flag = " !" if s["duration_sec"] > MAX_SCENE_DURATION else "  "
        print(f"{i:>3}  {s.get('label',''):18}  {s['start_sec']:>6.1f}  {s['duration_sec']:>5.1f}{flag}"
              f"{s.get('image','@last'):8}  {s['prompt'][:60]}")


def cmd_song(spec: dict, project: Path) -> None:
    out_mp3 = project / "song.mp3"
    meta_path = project / "song_meta.json"
    if out_mp3.exists() and meta_path.exists():
        _log(project, f"song already exists at {out_mp3}, skipping")
        return

    title = spec.get("title") or "Untitled"
    style = spec.get("style") or ""
    lyrics = spec.get("lyrics") or ""
    instrumental = bool((spec.get("suno") or {}).get("make_instrumental", False))

    if not style:
        sys.exit("spec.style is required for suno generation")

    _log(project, f"generating song '{title}' via suno-mcp (up to 400s)…")
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

    # Last JSON blob in stdout is the result (script prints progress then JSON)
    out = proc.stdout.strip()
    try:
        # Find first '{' that opens valid JSON at end
        brace = out.rfind("{")
        meta = json.loads(out[brace:]) if brace >= 0 else {}
    except json.JSONDecodeError:
        meta = {"raw": out}

    local = meta.get("local_file")
    if not local or not Path(local).exists():
        sys.exit(f"suno did not produce a local_file; meta={meta}")
    shutil.copy(local, out_mp3)
    # Suno always returns 2 variants — copy the extras as song_v2.mp3, song_v3.mp3, ...
    for i, extra in enumerate(meta.get("local_files", [])[1:], start=2):
        if extra and Path(extra).exists():
            dst = project / f"song_v{i}.mp3"
            shutil.copy(extra, dst)
            _log(project, f"variant {i} saved: {dst.name} ({dst.stat().st_size//1024} KB)")
    meta_path.write_text(json.dumps(meta, indent=2))
    _log(project, f"song saved: {out_mp3} ({out_mp3.stat().st_size//1024} KB)")


def _slice_song(project: Path, scene: dict, stem: str,
                tail_buffer_sec: float = 0.0) -> Path:
    """Slice the song for LTX audio conditioning. tail_buffer_sec extends the
    slice past the scene's nominal end so LTX has audio lookhead to finish
    the phoneme / close the mouth — otherwise the mouth freezes mid-word at
    scene boundary. Extra tail is absorbed by crossfade=tail_buffer in
    assembly so the timeline doesn't drift."""
    slice_mp3 = project / "song_slices" / f"{stem}.mp3"
    if slice_mp3.exists():
        return slice_mp3
    ffmpeg = FFMPEG()
    total = float(scene["duration_sec"]) + float(tail_buffer_sec)
    cmd = [ffmpeg, "-y", "-ss", str(scene["start_sec"]),
           "-t",  str(total),
           "-i",  str(project / "song.mp3"),
           "-c",  "copy", str(slice_mp3)]
    rc = subprocess.run(cmd, capture_output=True).returncode
    if rc != 0:
        rc = subprocess.run([ffmpeg, "-y", "-ss", str(scene["start_sec"]),
                             "-t",  str(total),
                             "-i",  str(project / "song.mp3"),
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
    # Anchor at the video's target resolution — LTX downsizes anyway via
    # ResizeImagesByLongerEdge(1536) + ResizeImageMaskNode. Higher anchor res
    # just wastes flux2 time. Override per-scene via anchor.width/height.
    w = int(anchor_cfg.get("width",  vs["width"]))
    h = int(anchor_cfg.get("height", vs["height"]))

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
            sys.exit(f"scene {idx}: anchor needs 0/1/2 references, got {len(ref_paths)}")

    # Identity-preservation guard for i2i/i2i2 anchors: when we hand flux2 a
    # reference image of a person, auto-append a concrete instruction to
    # keep the same face. Otherwise flux2 reads the prompt as a brief and
    # invents a new-looking person every scene → drift + gender swap + the
    # "my face is degrading" problem. Prompt authors can skip this by
    # adding `keep_identity: false` to the anchor block.
    anchor_prompt = anchor_cfg["prompt"]
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

    if anchor_type == "t2i":
        cmd = ["python3", str(COMFY), "t2i"] + common
    elif anchor_type == "i2i":
        cmd = ["python3", str(COMFY), "i2i", "--image", str(ref_paths[0])] + common
    elif anchor_type == "i2i2":
        cmd = ["python3", str(COMFY), "i2i2",
               "--image1", str(ref_paths[0]),
               "--image2", str(ref_paths[1])] + common
    elif anchor_type == "angles":
        # Generates a batch — uses the first prompt as the primary anchor.
        angle_prompts = anchor_cfg.get("angle_prompts") or [anchor_cfg["prompt"]]
        cmd = ["python3", str(COMFY), "multiprompt",
               "--image", str(ref_paths[0]),
               "--prompts", "\n".join(angle_prompts)] + common
    else:
        sys.exit(f"scene {idx}: unknown anchor.type '{anchor_type}' "
                 "(expected t2i|i2i|i2i2|angles)")

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


def _resolve_image(spec: dict, project: Path, idx: int, ref: str | None) -> str | None:
    """Resolve `image: @none | @anchor | @last | path`. Returns absolute path
    string, or None if @none (force t2v), or @last at idx=1 with no anchor → t2v."""
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
    # Literal path (relative to project dir if not absolute)
    p = Path(ref)
    return str(p if p.is_absolute() else (project / p).resolve())


def _extract_last_frame(scene_mp4: Path, out_png: Path) -> None:
    rc = subprocess.run([str(VIDEO_JOIN), "last-frame",
                         "--input", str(scene_mp4), "--output", str(out_png)],
                        capture_output=True).returncode
    if rc != 0:
        sys.exit(f"failed to extract last frame of {scene_mp4}")


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

    if out_mp4.exists() and last_png.exists():
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
    effective_duration = float(scene["duration_sec"]) + (tail_buffer if is_lipsync else 0.0)

    slice_mp3 = _slice_song(project, scene, stem,
                            tail_buffer_sec=tail_buffer if is_lipsync else 0.0)

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
              "--prompt",  scene["prompt"]]
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

    if image_path is None:
        # First scene with no anchor → fall back to t2v (no audio slice).
        cmd = ["python3", str(COMFY), "t2v"] + common
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
    """Render transition clips between each adjacent scene pair. Each
    transition is a morph from scene N's last frame → scene N+1's first
    frame, driven by the ltx2.3-transition LoRA with an audio slice of
    the song spanning the cut. Transitions run on COMFY_URL_FLUX (the
    12GB GPU) so they can render in parallel with the main scene batch
    on COMFY_URL_VIDEO. Restart-safe."""
    vs = spec.get("video") or {}
    tv = vs.get("transitions") or {}
    if not tv.get("enabled"):
        _log(project, "transitions disabled (video.transitions.enabled: false)")
        return
    default_prompt = tv.get("prompt",
        "a smooth cinematic morph between scenes, matching lighting and mood, "
        "continuous motion across the cut")
    default_fps = int(tv.get("fps", 25))

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

        a_last = project / "scenes" / f"{a_stem}-last.png"
        b_mp4 = project / "scenes" / f"{b_stem}.mp4"
        if not a_last.exists() or not b_mp4.exists():
            _log(project, f"transition {a_idx}→{b_idx} skipped — "
                          f"prerequisites missing (last={a_last.exists()} "
                          f"b_mp4={b_mp4.exists()})")
            continue

        # Extract scene B's first frame
        b_first = tdir / f"{b_stem}-first.png"
        if not b_first.exists():
            rc = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(b_mp4),
                 "-vf", "select=eq(n\\,0)", "-vframes", "1", "-q:v", "2",
                 str(b_first)], capture_output=True).returncode
            if rc != 0:
                _log(project, f"transition {a_idx}→{b_idx}: first-frame extract failed")
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

        # Per-boundary prompt override on scene B
        prompt = (b.get("transition_from_prev") or {}).get("prompt", default_prompt)

        prefix = f"trans_{a_idx}_{b_idx}"
        cmd = ["python3", str(COMFY), "transition",
               "--first", str(a_last),
               "--last", str(b_first),
               "--audio", str(audio_slice),
               "--prompt", prompt,
               "--seconds", f"{dur:.2f}",
               "--fps", str(default_fps),
               "--width", str(vs["resolution"][0] if isinstance(vs.get("resolution"), list) else 448),
               "--height", str(vs["resolution"][1] if isinstance(vs.get("resolution"), list) else 832),
               "--prefix", prefix,
               "--output-dir", str(tdir),
               "--timeout", "1200"]
        _log(project, f"transition {a_idx}→{b_idx}: {dur}s, audio from {slice_start:.2f}s")
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

    videos_csv = ",".join(str(p) for p in clip_paths)
    trims_csv  = ",".join(f"{d:.3f}" for d in durations)
    starts_csv = ",".join(f"{s:.3f}" for s in starts)

    for prefix, song_path in song_variants:
        cmd = ["python3", str(COMFY), "vconcat",
               "--videos", videos_csv,
               "--audio", str(song_path),
               "--fps", str(fps),
               "--trim-durations", trims_csv,
               "--trim-starts", starts_csv,
               "--prefix", prefix,
               "--output-dir", str(project),
               "--timeout", "1800"]
        _log(project, f"assembling {prefix}.mp4 ({len(clip_paths)} clips"
                      f"{', with transitions' if trans_enabled else ''}, "
                      f"audio={song_path.name})")
        rc = _run(cmd, log_path=project / "run.log",
                  env_override=_comfy_env_for("video"))
        if rc != 0:
            sys.exit(f"assembly failed for {prefix}")
        produced = sorted(project.glob(f"{prefix}_*.mp4"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        final = project / f"{prefix}.mp4"
        if produced and produced[0] != final:
            produced[0].rename(final)
        _log(project, f"assembled → {final} "
                      f"({final.stat().st_size//1024 if final.exists() else '?'} KB)")


def cmd_all(spec: dict, project: Path) -> None:
    _ensure_dirs(project)
    cmd_song(spec, project)

    # Hard check: scene total must cover the longest song variant, otherwise
    # the assembly will freeze-frame the tail for potentially minutes. Fail
    # loudly BEFORE burning GPU time on the scenes.
    # Final timeline is sum(duration_sec) — each scene's tail buffer is
    # trimmed off during concat via GetImageRangeFromBatch, no crossfade.
    assembled = sum(s["duration_sec"] for s in spec["scenes"])
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

    for i in range(1, len(spec["scenes"]) + 1):
        cmd_scene(spec, project, i)
    # Render transitions (no-op if video.transitions.enabled != true).
    # Runs on COMFY_URL_FLUX so they parallelize with... well, here they run
    # serially after scenes, but `transitions` can also be invoked standalone
    # during the main batch on a second terminal.
    cmd_transitions(spec, project)
    cmd_assemble(spec, project)


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
    for c in ["plan", "song", "assemble", "transitions", "all", "status"]:
        sp = sub.add_parser(c)
        sp.add_argument("spec")
    ss = sub.add_parser("scene")
    ss.add_argument("idx", type=int)
    ss.add_argument("spec")

    args = p.parse_args()
    spec, project = _load_spec(args.spec)
    _ensure_dirs(project)

    if args.cmd == "plan":      cmd_plan(spec, project)
    elif args.cmd == "song":    cmd_song(spec, project)
    elif args.cmd == "scene":   cmd_scene(spec, project, args.idx)
    elif args.cmd == "assemble":cmd_assemble(spec, project)
    elif args.cmd == "transitions": cmd_transitions(spec, project)
    elif args.cmd == "all":     cmd_all(spec, project)
    elif args.cmd == "status":  cmd_status(spec, project)


if __name__ == "__main__":
    main()
