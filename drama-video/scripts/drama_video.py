#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0", "imageio-ffmpeg>=0.6"]
# ///
"""Drama-video orchestrator — dialogue-first narrative pipeline.

Stages:
  1. plan      validate spec + show shot breakdown
  2. audio     call the elevenlabs skill to render dialogue+bg (if audio.spec)
  3. anchors   pre-render per-shot flux2 anchors
  4. shots     generate each shot's mp4 via comfyui ia2v (@last chaining)
  5. assemble  vconcat shots, mux full dialogue audio
  6. all       runs 2→5, gates pause between stages when enabled in spec

Sibling to `music-video` (song-first, bar-aligned). This skill is
dialogue-first, cue-aligned — shot boundaries typically land on
pauses or line starts.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── lazy deps ────────────────────────────────────────────────────────

def _yaml():
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml not installed. Re-run via `uv run --script`.")
    return yaml


def _ffmpeg():
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        sys.exit("imageio-ffmpeg not installed. Re-run via `uv run --script`.")
    return get_ffmpeg_exe()


# ── constants ─────────────────────────────────────────────────────────

MAX_SHOT_DURATION = 15.0          # LTX-2.3 OOM threshold at portrait resolution

def _find_skill_script(rel: str) -> Path:
    """Resolve a sibling-skill script across install locations.
    Order: OpenClaw sandbox → host install (~/.openclaw) → repo checkout sibling.
    """
    candidates = [
        Path(f"/home/sandbox/.openclaw/skills/{rel}"),
        Path.home() / ".openclaw/skills" / rel,
        Path(__file__).resolve().parent.parent.parent / rel.split("/", 1)[0] / rel.split("/", 1)[1],
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last-resort default — caller will see ENOENT at first invocation.
    return candidates[0]

COMFY = _find_skill_script("comfyui/scripts/comfy_graph.py")
ELEVEN_SCENE = _find_skill_script("elevenlabs/scripts/eleven_scene_audio.py")


# ── helpers ───────────────────────────────────────────────────────────

def _log(project: Path, msg: str) -> None:
    ts = __import__("time").strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    try:
        with (project / "run.log").open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _run(cmd: list[str], cwd: Path, log_path: Path, env_override: dict | None = None) -> int:
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(str(a) for a in cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def _comfy_env_for(kind: str) -> dict:
    """Route comfy_graph.py at the right server: ia2v/flf2v on video, i2i on flux."""
    return {}  # comfy_graph.py auto-routes via VIDEO_COMMANDS; nothing needed.


def _probe_duration(path: Path) -> float:
    import re
    ffmpeg = _ffmpeg()
    r = subprocess.run([ffmpeg, "-i", str(path)],
                       capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def _video_spec(spec: dict) -> dict:
    v = spec.get("video", {})
    w, h = (v.get("resolution") or [1024, 576])
    return {
        "fps": int(v.get("fps", 24)),
        "width": int(w), "height": int(h),
        "negative": v.get("negative"),
        "tail_buffer_sec": float(v.get("tail_buffer_sec", 0.0) or 0.0),
    }


def _shot_stem(idx: int, label: str) -> str:
    """Consistent 001-label naming for on-disk artifacts."""
    return f"{idx:03d}-{label}"


def _character_tokens(spec: dict) -> dict:
    """Top-level character description is substituted wherever prompts use
    `{character}`. Same pattern as music-video `subjects:` block."""
    c = spec.get("character") or {}
    return {"character": c.get("description") or c.get("name") or ""}


def _expand_tokens(text: str, tokens: dict) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for k, v in tokens.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _load_spec(path: Path) -> tuple[dict, Path]:
    yaml = _yaml()
    spec = yaml.safe_load(path.read_text())
    project = path.parent.resolve()
    return spec, project


def _audio_paths(spec: dict, project: Path) -> tuple[Path, Path | None]:
    """Resolve audio.file + audio.cues to absolute paths. Either field may
    be relative to the spec's directory."""
    a = spec.get("audio") or {}
    if a.get("file"):
        f = project / a["file"] if not Path(a["file"]).is_absolute() else Path(a["file"])
    else:
        sys.exit("spec.audio.file is required (or set audio.spec to render it)")
    cues = None
    if a.get("cues"):
        cues = project / a["cues"] if not Path(a["cues"]).is_absolute() else Path(a["cues"])
    return f.resolve(), (cues.resolve() if cues else None)


# ── slicing the audio per shot ────────────────────────────────────────

def _slice_audio(project: Path, shot: dict, audio_file: Path,
                  tail_buffer_sec: float, out: Path) -> None:
    if out.exists():
        return
    total = float(shot["duration_sec"]) + float(tail_buffer_sec)
    ffmpeg = _ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode (not -c copy) so start_sec cuts at the sample, not the
    # nearest mp3 frame — keeps LTX audio conditioning aligned.
    rc = subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                         "-ss", f"{float(shot['start_sec']):.3f}",
                         "-t", f"{total:.3f}",
                         "-i", str(audio_file),
                         "-c:a", "libmp3lame", "-b:a", "192k",
                         str(out)], capture_output=True).returncode
    if rc != 0:
        sys.exit(f"failed to slice audio for {out.name}")


# ── anchor generation (flux2 via comfy_graph.py) ──────────────────────

def _generate_anchor(spec: dict, project: Path, idx: int, shot: dict,
                      stem: str) -> str | None:
    """Pre-generate a shot-specific anchor via flux2 (t2i / i2i / i2i2).
    Mirrors music-video's anchor flow but reads tokens from the character
    block. Returns absolute path to the generated PNG, or None if the shot
    has no `anchor:` block."""
    anchor_cfg = shot.get("anchor")
    if not anchor_cfg or not anchor_cfg.get("prompt"):
        return None
    out_path = project / "shots" / f"{stem}-anchor.png"
    if out_path.exists():
        return str(out_path)

    vs = _video_spec(spec)
    w = int(anchor_cfg.get("width", vs["width"]))
    h = int(anchor_cfg.get("height", vs["height"]))
    tokens = _character_tokens(spec)

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
            sys.exit(f"shot {idx}: anchor needs 0/1/2 references, got {len(ref_paths)}")

    prompt = _expand_tokens(anchor_cfg["prompt"], tokens)

    common = ["--prompt", prompt,
              "--width",  str(w),
              "--height", str(h),
              "--prefix", f"{stem}-anchor",
              "--output-dir", str(project / "shots"),
              "--timeout", "600"]
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
    else:
        sys.exit(f"shot {idx}: unknown anchor.type {anchor_type!r}")

    _log(project, f"shot {idx}: generating anchor via flux2 "
                  f"({anchor_type}{' refs='+','.join(str(r) for r in ref_paths) if ref_paths else ''})")
    rc = _run(cmd, cwd=project, log_path=project / "run.log")
    if rc != 0:
        sys.exit(f"shot {idx}: anchor render failed (rc={rc})")

    # comfy wrote <prefix>_NNNNN_.png — locate and rename.
    matches = sorted((project / "shots").glob(f"{stem}-anchor_*.png"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        sys.exit(f"shot {idx}: no anchor PNG produced")
    matches[0].rename(out_path)
    return str(out_path)


def _resolve_image(spec: dict, project: Path, idx: int, ref: str | None) -> str | None:
    """`@anchor` → character.anchor path.
    `@last` → previous shot's -last.png (idx-1).
    literal path → resolve relative to project."""
    if not ref or ref == "@last":
        if idx == 1:
            # First shot with @last and no @anchor fallback — force t2v.
            return None
        prev_label = spec["shots"][idx - 2].get("label", "shot")
        prev = project / "shots" / f"{_shot_stem(idx - 1, prev_label)}-last.png"
        if not prev.exists():
            sys.exit(f"shot {idx}: @last requested but {prev} does not exist "
                     f"— render shot {idx-1} first")
        return str(prev.resolve())
    if ref == "@anchor":
        c = spec.get("character") or {}
        if not c.get("anchor"):
            sys.exit(f"shot {idx}: @anchor requested but spec.character.anchor missing")
        a = c["anchor"]
        p = Path(a) if Path(a).is_absolute() else (project / a)
        return str(p.resolve())
    p = Path(ref) if Path(ref).is_absolute() else (project / ref)
    return str(p.resolve())


# ── per-shot ia2v render ──────────────────────────────────────────────

def _render_shot(spec: dict, project: Path, idx: int, shot: dict) -> None:
    if float(shot["duration_sec"]) > MAX_SHOT_DURATION:
        sys.exit(f"shot {idx} '{shot.get('label','?')}' duration "
                 f"{shot['duration_sec']}s exceeds {MAX_SHOT_DURATION}s cap "
                 f"(LTX OOM threshold at portrait resolutions). Split it.")
    stem = _shot_stem(idx, shot.get("label", "shot"))
    out_mp4 = project / "shots" / f"{stem}.mp4"
    last_png = project / "shots" / f"{stem}-last.png"
    if out_mp4.exists():
        if not last_png.exists():
            ffmpeg = _ffmpeg()
            subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                            "-sseof", "-0.1", "-i", str(out_mp4),
                            "-vframes", "1", str(last_png)],
                           capture_output=True).returncode
        _log(project, f"shot {idx} '{stem}' already rendered, skipping")
        return

    audio_file, _ = _audio_paths(spec, project)
    if not audio_file.exists():
        sys.exit(f"spec.audio.file missing at {audio_file} — run `audio` first")

    vs = _video_spec(spec)
    tail_buffer = float(vs["tail_buffer_sec"])
    effective_duration = float(shot["duration_sec"]) + tail_buffer

    slice_mp3 = project / "shots" / f"{stem}.mp3"
    _slice_audio(project, shot, audio_file, tail_buffer, slice_mp3)

    # Prefer a scene-specific pre-gen anchor when set.
    pre_gen = _generate_anchor(spec, project, idx, shot, stem)
    image_path = pre_gen or _resolve_image(spec, project, idx, shot.get("image"))

    tokens = _character_tokens(spec)
    prompt = _expand_tokens(shot["prompt"], tokens)
    common = ["--seconds", f"{effective_duration:.2f}",
              "--fps",     str(vs["fps"]),
              "--width",   str(vs["width"]),
              "--height",  str(vs["height"]),
              "--prefix",  stem,
              "--output-dir", str(project / "shots"),
              "--timeout", "1800",
              "--prompt",  prompt]
    if vs["negative"]:
        common += ["--negative", vs["negative"]]
    if shot.get("camera_lora"):
        common += ["--camera-lora", str(shot["camera_lora"])]
    if shot.get("camera_lora_strength") is not None:
        common += ["--camera-lora-strength", str(shot["camera_lora_strength"])]
    if shot.get("fast") or (spec.get("video") or {}).get("fast"):
        common += ["--fast"]

    # Identity anchor: pulls character face back toward a canonical
    # reference at frame_idx=-1 across the shot, preventing drift across
    # chained continuations or audio-driven lipsync. Resolution order:
    #   1. shot.identity_anchor (path or false)  — per-shot override
    #   2. character.identity_anchor              — top-level default
    #   3. character.anchor                       — fall back to the
    #                                               character's primary sheet
    # Set shot.identity_anchor: false to disable on a specific shot.
    char_cfg = spec.get("character") or {}
    id_anchor_val = shot.get("identity_anchor")
    if id_anchor_val is None:
        id_anchor_val = char_cfg.get("identity_anchor")
    if id_anchor_val is None:
        id_anchor_val = char_cfg.get("anchor")
    id_anchor_path: str | None = None
    id_strength: float = float(shot.get("identity_strength",
                               char_cfg.get("identity_strength", 0.3)))
    if id_anchor_val and id_anchor_val is not False:
        p = Path(id_anchor_val)
        p = p if p.is_absolute() else (project / id_anchor_val)
        if p.exists():
            id_anchor_path = str(p.resolve())
        else:
            _log(project, f"shot {idx}: identity_anchor {p} not found, skipping")

    id_extras: list[str] = []
    if id_anchor_path:
        id_extras = ["--identity-anchor", id_anchor_path,
                     "--identity-strength", f"{id_strength:.3f}"]

    # Three render paths in priority order:
    #   1. `continue_from_prev: true` on a non-first shot → LTX continuation
    #      (locks first N frames of this shot to the last N of prev.mp4).
    #      Seamless cut, no separate anchor needed.
    #   2. `image: "@last"` or no image + not first shot → classic ia2v
    #      with prev-last.png as a secondary image_ref for lipsync identity.
    #   3. Has flux2 anchor or fresh image path → ia2v with that image.
    use_continuation = bool(shot.get("continue_from_prev")) and idx > 1
    # `mode: ingredients` → LTX-2.3 reference-sheet IC-LoRA, conditioned on ONE
    # project sheet (project-level `reference_sheet:` or a per-shot override) for
    # consistent recurring characters / props / locations. Checked FIRST so it is
    # authoritative regardless of image_path / continuation.
    #
    # SILENT-INSERT: ingredients renders carry no useful audio and do NOT lipsync.
    # Drama is dialogue-first, so use this only for shots that don't need on-screen
    # speech (B-roll, establishing, inserts); identity comes from the sheet, not the
    # identity-anchor. The full audio track is muxed over the timeline at assemble.
    shot_mode = shot.get("mode") or (spec.get("video") or {}).get("mode")
    if shot_mode == "ingredients":
        sheet = shot.get("reference_sheet") or spec.get("reference_sheet")
        if not sheet:
            sys.exit(f"shot {idx}: mode=ingredients needs `reference_sheet:` "
                     f"(set it project-level or on the shot)")
        sheet_path = (Path(sheet) if Path(sheet).is_absolute()
                      else (project / sheet)).resolve()
        if not sheet_path.exists():
            sys.exit(f"shot {idx}: reference_sheet {sheet_path} does not exist")
        # Default to the ingredients LoRA's tuned negative — drop the project's
        # generic video.negative unless the shot sets its own.
        ing_common, _skip = [], False
        for tok in common:
            if _skip:
                _skip = False; continue
            if tok == "--negative":
                _skip = True; continue
            ing_common.append(tok)
        ing_extras: list[str] = []
        if shot.get("negative"):
            ing_extras += ["--negative", str(shot["negative"])]
        ls = shot.get("ingredients_lora_strength")
        if ls is not None:
            ing_extras += ["--lora_strength", str(float(ls))]
        rs = shot.get("ingredients_reference_strength")
        if rs is not None:
            ing_extras += ["--reference_strength", str(float(rs))]
        cmd = ["python3", str(COMFY), "ingredients",
               "--sheet", str(sheet_path)] + ing_extras + ing_common
    elif use_continuation:
        prev_label = spec["shots"][idx - 2].get("label", "shot")
        prev_mp4 = project / "shots" / f"{_shot_stem(idx - 1, prev_label)}.mp4"
        if not prev_mp4.exists():
            sys.exit(f"shot {idx}: continue_from_prev set but prev shot mp4 "
                     f"not found at {prev_mp4} — render shot {idx-1} first")
        overlap = float(shot.get("overlap_seconds", 1.0))
        cmd = ["python3", str(COMFY), "continuation",
               "--prev_video", str(prev_mp4),
               "--audio", str(slice_mp3),
               "--overlap-seconds", f"{overlap:.3f}",
               "--overlap-strength", str(shot.get("overlap_strength", 1.0))] + id_extras + common
    elif image_path is None:
        cmd = ["python3", str(COMFY), "t2v"] + common
    else:
        ia2v_extras: list[str] = []
        if idx > 1:
            prev_label = spec["shots"][idx - 2].get("label", "shot")
            prev_last = project / "shots" / f"{_shot_stem(idx - 1, prev_label)}-last.png"
            if prev_last.exists():
                ia2v_extras += ["--image-refs", str(prev_last)]
        cmd = ["python3", str(COMFY), "ia2v",
               "--image", image_path,
               "--audio", str(slice_mp3)] + ia2v_extras + id_extras + common

    _log(project, f"shot {idx} '{stem}': {prompt[:80]}")
    rc = _run(cmd, cwd=project, log_path=project / "run.log",
              env_override=_comfy_env_for("video"))
    if rc != 0:
        sys.exit(f"shot {idx} generation failed (rc={rc})")

    candidates = sorted((project / "shots").glob(f"{stem}_*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(f"shot {idx}: no mp4 matched '{stem}_*.mp4'")
    candidates[0].rename(out_mp4)

    # Extract last frame for @last chaining.
    ffmpeg = _ffmpeg()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                    "-sseof", "-0.1", "-i", str(out_mp4),
                    "-vframes", "1", str(last_png)],
                   capture_output=True)
    _log(project, f"shot {idx} done → {out_mp4.name}, last frame → {last_png.name}")


# ── stage commands ────────────────────────────────────────────────────

def cmd_plan(spec: dict, project: Path) -> None:
    shots = spec.get("shots") or []
    if not shots:
        sys.exit("spec has no `shots`")
    vs = _video_spec(spec)
    audio_file, cues_path = _audio_paths(spec, project)
    audio_dur = _probe_duration(audio_file) if audio_file.exists() else None

    total = sum(float(s["duration_sec"]) for s in shots)
    print(f"title         : {spec.get('title','(untitled)')}")
    print(f"character     : {(spec.get('character') or {}).get('name','?')}")
    print(f"video         : {vs['width']}x{vs['height']} @ {vs['fps']}fps "
          f"tail_buffer={vs['tail_buffer_sec']}s")
    print(f"audio file    : {audio_file.name}"
          f"{f' ({audio_dur:.2f}s)' if audio_dur else ' (MISSING)'}")
    if cues_path:
        print(f"cues          : {cues_path.name}"
              f"{' ✓' if cues_path.exists() else ' MISSING'}")
    print(f"shots         : {len(shots)} shot(s), "
          f"total duration {total:.2f}s"
          f"{f' (audio is {audio_dur:.2f}s)' if audio_dur else ''}")
    print()
    print(f"  {'#':>3}  {'label':<22}  {'start':>7}  {'dur':>6}  {'image':<9}  prompt")
    for i, s in enumerate(shots, 1):
        flag = " !" if float(s["duration_sec"]) > MAX_SHOT_DURATION else "  "
        print(f"  {i:>3}  {str(s.get('label','?')):<22}  "
              f"{float(s['start_sec']):>7.2f}  {float(s['duration_sec']):>6.2f}  "
              f"{str(s.get('image','@last')):<9}  "
              f"{s['prompt'][:60]}"
              f"{flag}")

    # Sanity: shot end must be <= audio duration
    if audio_dur is not None:
        end = max(float(s["start_sec"]) + float(s["duration_sec"]) for s in shots)
        if end > audio_dur + 0.05:
            print(f"\n⚠  last shot ends at {end:.2f}s but audio is only {audio_dur:.2f}s")
    # Warn if a shot starts mid-line
    if cues_path and cues_path.exists():
        try:
            cues = json.loads(cues_path.read_text()).get("cues", [])
            line_ranges = [(c["start"], c["end"]) for c in cues if c["kind"] == "line"]
            for i, s in enumerate(shots, 1):
                t = float(s["start_sec"])
                for ls, le in line_ranges:
                    if ls < t < le - 0.05:
                        print(f"⚠  shot {i} starts mid-line (cue {ls:.2f}-{le:.2f}s "
                              f"contains start {t:.2f})")
        except Exception:
            pass


def cmd_audio(spec: dict, project: Path) -> None:
    """If audio.spec is set, render via elevenlabs. Otherwise assert audio.file exists."""
    a = spec.get("audio") or {}
    audio_spec = a.get("spec")
    audio_file, _ = _audio_paths(spec, project)
    if audio_file.exists():
        _log(project, f"audio already at {audio_file.name}, skipping")
        return
    if not audio_spec:
        sys.exit(f"audio.file missing at {audio_file} and no audio.spec set "
                 f"— either drop the mp3 in place or author audio.spec to render it")
    spec_path = project / audio_spec if not Path(audio_spec).is_absolute() else Path(audio_spec)
    _log(project, f"rendering audio via elevenlabs: {spec_path.name}")
    rc = _run(["python3", str(ELEVEN_SCENE), str(spec_path)],
              cwd=project, log_path=project / "run.log")
    if rc != 0:
        sys.exit(f"audio render failed (rc={rc})")


def cmd_anchors(spec: dict, project: Path) -> None:
    shots = spec.get("shots") or []
    generated = 0
    skipped = 0
    for i, shot in enumerate(shots, 1):
        if not shot.get("anchor") or not shot["anchor"].get("prompt"):
            continue
        stem = _shot_stem(i, shot.get("label", "shot"))
        out = project / "shots" / f"{stem}-anchor.png"
        if out.exists():
            skipped += 1
            continue
        _generate_anchor(spec, project, i, shot, stem)
        generated += 1
    _log(project, f"anchors: generated {generated}, skipped {skipped} existing")


def cmd_shot(spec: dict, project: Path, idx: int) -> None:
    shots = spec.get("shots") or []
    if idx < 1 or idx > len(shots):
        sys.exit(f"shot index out of range: {idx} (have {len(shots)})")
    _render_shot(spec, project, idx, shots[idx - 1])


def cmd_shots(spec: dict, project: Path) -> None:
    shots = spec.get("shots") or []
    if not shots:
        sys.exit("spec has no `shots`")
    over_cap = [s["label"] for s in shots
                if float(s["duration_sec"]) > MAX_SHOT_DURATION]
    if over_cap:
        sys.exit(f"refusing to render — shot(s) exceed {MAX_SHOT_DURATION}s cap: "
                 f"{over_cap}. Split them into shorter shots.")
    for i in range(1, len(shots) + 1):
        _render_shot(spec, project, i, shots[i - 1])


def cmd_assemble(spec: dict, project: Path) -> None:
    shots = spec.get("shots") or []
    if not shots:
        sys.exit("spec has no `shots`")
    audio_file, _ = _audio_paths(spec, project)
    if not audio_file.exists():
        sys.exit(f"audio missing at {audio_file}")
    ffmpeg = _ffmpeg()

    clip_tuples = []
    for i, shot in enumerate(shots, 1):
        stem = _shot_stem(i, shot.get("label", "shot"))
        p = project / "shots" / f"{stem}.mp4"
        if not p.exists():
            sys.exit(f"shot {i} mp4 missing at {p} — run it first")
        clip_tuples.append((p, 0.0, float(shot["duration_sec"])))

    # Delegate to the shared storyboard assemble helper — same pipeline
    # used by music-video. Pre-trim → strip audio → concat → mux.
    import importlib.util
    sb_lib = _find_skill_script("storyboard/lib/assemble.py")
    m_spec = importlib.util.spec_from_file_location("storyboard_assemble", sb_lib)
    asm = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(asm)

    final = project / "final.mp4"
    _log(project, f"assembling final.mp4 ({len(clip_tuples)} shots, audio={audio_file.name})")
    asm.assemble_clips(
        clips=clip_tuples, audio_file=audio_file, output=final,
        work_dir=project / "shots" / "_assemble_cache", ffmpeg=ffmpeg)
    _log(project, f"assembled → {final} "
                  f"({final.stat().st_size // 1024 if final.exists() else '?'} KB)")


def cmd_all(spec: dict, project: Path, no_gate: bool = False) -> None:
    gates = {
        "audio": bool(spec.get("gate_confirm_audio", False)),
        "anchors": bool(spec.get("gate_confirm_anchors", True)),
    }
    cmd_audio(spec, project)
    if gates["audio"] and not no_gate:
        input(f"\n[GATE] audio rendered. Listen, then press ENTER to continue → anchors. ")

    cmd_anchors(spec, project)
    if gates["anchors"] and not no_gate:
        input(f"\n[GATE] anchors rendered. Review shots/*-anchor.png then press ENTER → shots. ")

    cmd_shots(spec, project)
    cmd_assemble(spec, project)


def cmd_status(spec: dict, project: Path) -> None:
    shots = spec.get("shots") or []
    audio_file, cues_path = _audio_paths(spec, project)
    print(f"audio      : {audio_file.name} "
          f"{'✓' if audio_file.exists() else 'MISSING'}")
    if cues_path:
        print(f"cues       : {cues_path.name} "
              f"{'✓' if cues_path.exists() else 'MISSING'}")
    print(f"final.mp4  : {'✓' if (project/'final.mp4').exists() else '-'}")
    for i, shot in enumerate(shots, 1):
        stem = _shot_stem(i, shot.get("label", "shot"))
        marks = []
        for suffix, label in (("-anchor.png", "anchor"),
                              (".mp3", "slice"),
                              (".mp4", "mp4"),
                              ("-last.png", "last")):
            ok = (project / "shots" / f"{stem}{suffix}").exists()
            marks.append(f"{label}={'✓' if ok else '-'}")
        print(f"  shot {i:>2} {stem:<22}  " + "  ".join(marks))


# ── entry ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(prog="drama_video")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("plan", "audio", "anchors", "shots", "assemble", "status", "all"):
        p = sub.add_parser(cmd)
        p.add_argument("spec")
        if cmd == "all":
            p.add_argument("--no-gate", action="store_true")
    shot_p = sub.add_parser("shot")
    shot_p.add_argument("idx", type=int)
    shot_p.add_argument("spec")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        sys.exit(f"spec not found: {spec_path}")
    spec, project = _load_spec(spec_path)

    if args.cmd == "plan":       cmd_plan(spec, project)
    elif args.cmd == "audio":    cmd_audio(spec, project)
    elif args.cmd == "anchors":  cmd_anchors(spec, project)
    elif args.cmd == "shot":     cmd_shot(spec, project, args.idx)
    elif args.cmd == "shots":    cmd_shots(spec, project)
    elif args.cmd == "assemble": cmd_assemble(spec, project)
    elif args.cmd == "status":   cmd_status(spec, project)
    elif args.cmd == "all":      cmd_all(spec, project, no_gate=args.no_gate)


if __name__ == "__main__":
    main()
