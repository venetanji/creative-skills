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


def _run(cmd: list[str], cwd: Path | None = None, log_path: Path | None = None) -> int:
    """Run subprocess; stream stderr to log, return exit code."""
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            if log_path:
                with open(log_path, "a") as f:
                    f.write(line)
        p.wait()
        return p.returncode


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
    raw_video = spec.get("video") or {}
    cross = float(raw_video.get("crossfade", 0) or 0)
    assembled = sum(s["duration_sec"] for s in scenes) - cross * max(0, len(scenes) - 1)

    print(f"title         : {spec.get('title','(untitled)')}")
    print(f"style         : {spec.get('style','')[:120]}")
    print(f"video         : {vs['width']}x{vs['height']} @ {vs['fps']}fps  crossfade={cross}s")
    print(f"scenes total  : {total:.1f}s across {len(scenes)} scene(s)")
    print(f"assembled len : {assembled:.1f}s  (scene sum minus crossfade overlap)")
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


def _slice_song(project: Path, scene: dict, stem: str) -> Path:
    slice_mp3 = project / "song_slices" / f"{stem}.mp3"
    if slice_mp3.exists():
        return slice_mp3
    ffmpeg = FFMPEG()
    cmd = [ffmpeg, "-y", "-ss", str(scene["start_sec"]),
           "-t",  str(scene["duration_sec"]),
           "-i",  str(project / "song.mp3"),
           "-c",  "copy", str(slice_mp3)]
    rc = subprocess.run(cmd, capture_output=True).returncode
    if rc != 0:
        # fallback re-encode
        rc = subprocess.run([ffmpeg, "-y", "-ss", str(scene["start_sec"]),
                             "-t",  str(scene["duration_sec"]),
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

    common = ["--prompt", anchor_cfg["prompt"],
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
    rc = _run(cmd, log_path=project / "run.log")
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
    """Resolve `image: @anchor | @last | path`. Returns absolute path string, or None
    if @last at idx=1 and no anchor → t2v."""
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

    slice_mp3 = _slice_song(project, scene, stem)

    # Prefer a scene-specific pre-generated anchor (flux2) when scene.anchor is set.
    # Falls back to the @last/@anchor/literal image chain otherwise.
    pre_gen_anchor = _generate_anchor(spec, project, idx, scene, stem)
    if pre_gen_anchor:
        image_path = pre_gen_anchor
    else:
        image_path = _resolve_image(spec, project, idx, scene.get("image"))
    vs = _video_spec(spec)

    # comfy_graph takes <cmd> as the first positional, then --opts.
    prefix = stem
    common = ["--seconds", str(int(scene["duration_sec"])),
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
        cmd = ["python3", str(COMFY), "ia2v",
               "--image", image_path,
               "--audio", str(slice_mp3)] + common

    _log(project, f"scene {idx} '{stem}': {scene['prompt'][:80]}")
    rc = _run(cmd, log_path=project / "run.log")
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


def cmd_assemble(spec: dict, project: Path) -> None:
    scenes = spec.get("scenes", [])
    mp4s: list[Path] = []
    for i, s in enumerate(scenes, 1):
        stem = _scene_stem(i, s.get("label", "scene"))
        p = project / "scenes" / f"{stem}.mp4"
        if not p.exists():
            sys.exit(f"scene {i} MP4 missing at {p} — run it first")
        mp4s.append(p)

    song = project / "song.mp3"
    if not song.exists():
        sys.exit("song.mp3 missing — run `song` first")

    vs = spec.get("video") or {}
    crossfade = float(vs.get("crossfade", 0.0))  # seconds; 0 = hard cut
    ffmpeg = FFMPEG()
    video_only = project / "scenes" / "_concat_video.mp4"

    if crossfade > 0 and len(mp4s) > 1:
        # Chain ffmpeg xfade across all scenes. Each xfade needs the offset at
        # which the previous composite ends minus the fade duration.
        durs = [_probe_duration(p) for p in mp4s]
        inputs: list[str] = []
        for p in mp4s:
            inputs += ["-i", str(p)]
        filter_parts: list[str] = []
        last_label = "[0:v]"
        running = durs[0]
        for i in range(1, len(mp4s)):
            offset = max(running - crossfade, 0)
            nxt = f"[v{i}]"
            filter_parts.append(
                f"{last_label}[{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset:.3f}{nxt}"
            )
            last_label = nxt
            running = running + durs[i] - crossfade
        filter_complex = ";".join(filter_parts)
        cmd = [ffmpeg, "-y", *inputs,
               "-filter_complex", filter_complex,
               "-map", last_label, "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "18", str(video_only)]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0:
            _log(project, f"xfade failed: {rc.stderr[-800:]}")
            sys.exit("xfade video concat failed")
    else:
        concat_list = project / "scenes" / "_concat.txt"
        concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in mp4s) + "\n")
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
               "-crf", "18", str(video_only)]
        rc = subprocess.run(cmd, capture_output=True).returncode
        concat_list.unlink(missing_ok=True)
        if rc != 0:
            sys.exit("video concat failed")

    # Ensure the concatenated video is at least as long as the longest song
    # variant. With N scenes × D duration and (N-1) crossfades of C seconds,
    # the concat is (N*D - (N-1)*C) long — usually a bit shorter than the song.
    # If so, extend the video by freezing the last frame so audio doesn't get
    # cut off when we drop `-shortest` below.
    video_dur = _probe_duration(video_only)
    max_song_dur = max(_probe_duration(p) for (_, p) in
                       [("final.mp4", song), *[(f"final_v{c.stem.replace('song_v','')}.mp4", c)
                                               for c in sorted(project.glob("song_v*.mp3"))]])
    if max_song_dur > video_dur + 0.25:
        pad = max_song_dur - video_dur + 0.2  # small safety overshoot
        padded = project / "scenes" / "_padded_video.mp4"
        cmd = [ffmpeg, "-y", "-i", str(video_only),
               "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "18", str(padded)]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0:
            _log(project, f"tpad failed: {rc.stderr[-400:]}")
            sys.exit("video tpad failed")
        video_only.unlink(missing_ok=True)
        video_only = padded
        _log(project, f"video padded by {pad:.2f}s (clone last frame) "
                      f"to cover {max_song_dur:.2f}s song")

    # Overlay the clean song audio onto the (now long-enough) video — one final
    # mp4 per available song variant (final.mp4, final_v2.mp4, ...). We drop
    # `-shortest` so the full audio track plays; the video tail may freeze
    # briefly at the end which is preferable to cutting the song mid-outro.
    song_variants = [("final.mp4", song)]
    for candidate in sorted(project.glob("song_v*.mp3")):
        idx = candidate.stem.replace("song_v", "")
        song_variants.append((f"final_v{idx}.mp4", candidate))

    for out_name, song_path in song_variants:
        final = project / out_name
        cmd = [ffmpeg, "-y", "-i", str(video_only), "-i", str(song_path),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-map", "0:v:0", "-map", "1:a:0",
               "-t", f"{_probe_duration(song_path):.3f}",
               str(final)]
        rc = subprocess.run(cmd, capture_output=True).returncode
        if rc != 0:
            sys.exit(f"audio overlay failed for {out_name}")
        _log(project, f"assembled → {final} ({final.stat().st_size//1024} KB)"
                      + (f" with {crossfade}s crossfade" if crossfade > 0 else ""))
    video_only.unlink(missing_ok=True)


def cmd_all(spec: dict, project: Path) -> None:
    _ensure_dirs(project)
    cmd_song(spec, project)
    for i in range(1, len(spec["scenes"]) + 1):
        cmd_scene(spec, project, i)
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
    for c in ["plan", "song", "assemble", "all", "status"]:
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
    elif args.cmd == "all":     cmd_all(spec, project)
    elif args.cmd == "status":  cmd_status(spec, project)


if __name__ == "__main__":
    main()
