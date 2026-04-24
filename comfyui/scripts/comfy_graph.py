#!/usr/bin/env python3
"""CLI for ComfyUI workflow submission at https://comfyui.tail9683c.ts.net

Usage:
  python comfy_graph.py t2i   --prompt "a cat" [--notify-target discord:...]
  python comfy_graph.py i2i   --image photo.jpg --prompt "cat wearing hat"
  python comfy_graph.py t2v   --prompt "a forest" --seconds 5
  python comfy_graph.py i2v   --image photo.jpg --prompt "person walking"
  python comfy_graph.py ia2v  --image photo.jpg --audio track.mp3 --prompt "..."
  python comfy_graph.py flf2v --first a.png --last b.png --prompt "..." --seconds 5
  python comfy_graph.py transition --first a.png --last b.png --prompt "..." --seconds 4
  python comfy_graph.py multiguide --guides a.png,b.png,c.png --prompt "..." --seconds 8 --audio slice.mp3
  python comfy_graph.py continuation --prev_video prev.mp4 --prompt "..." --seconds 8 --audio slice.mp3 --overlap-seconds 1.0
  python comfy_graph.py stems  --audio song.mp3          # vocals + instrumental
  python comfy_graph.py stt    --audio vocals.flac       # whisper transcription
  python comfy_graph.py vconcat --videos a.mp4,b.mp4,c.mp4 --audio song.mp3 --fps 24
  python comfy_graph.py dump  t2i --prompt "a cat"   # print workflow JSON only

Environment:
  COMFY_URL          Base URL of ComfyUI server (default: https://comfyui.tail9683c.ts.net)
  OPENCLAW_NOTIFY_TARGET  Default notification target (e.g. discord:123...)
"""
from __future__ import annotations
import sys, os, json, time, urllib.request
from pathlib import Path
from core import _submit_and_wait, upload_if_local
import flux2, ltx2, tts, post
from flux2 import (flux2_text_to_image, flux2_single_image_edit, flux2_double_image_edit,
                    flux2_double_image_edit_multiprompt, flux2_multiple_angles,
                    flux2_multi_reference_edit, flux2_multi_reference_edit_multiprompt)
from ltx2 import (ltx2_text_to_video, ltx2_image_to_video, ltx2_image_audio_to_video,
                   ltx2_first_last_frame_to_video, ltx2_multi_guide_to_video,
                   ltx2_continuation_to_video,
                   extract_last_frame)
from tts import qwen_tts, qwen_voice_clone
from post import extract_stems, transcribe, concat_videos
import core

# Route video workflows to the video comfy server, image/audio workflows to
# the flux/default server. Bots submit via this CLI and shouldn't have to
# worry about which server to target.
VIDEO_COMMANDS = {"t2v", "i2v", "ia2v", "flf2v", "transition", "multiguide", "continuation"}
# vconcat uses ComfyUI-FFmpeg nodes (MergingVideoByTwo / AddAudio) — pure
# CPU stream-copy, no GPU needed, so it goes to the flux server to keep
# the video GPU free for actual LTX renders.

def _resolve_base_url(cmd: str) -> str:
    if cmd in VIDEO_COMMANDS:
        return (os.environ.get("COMFY_URL_VIDEO")
                or os.environ.get("COMFY_URL")
                or "https://comfyui-video.tail9683c.ts.net").rstrip("/")
    # default (images, tts, stems, stt): prefer COMFY_URL_FLUX, fall back to COMFY_URL
    return (os.environ.get("COMFY_URL_FLUX")
            or os.environ.get("COMFY_URL")
            or "https://comfyui.tail9683c.ts.net").rstrip("/")

BASE = os.environ.get("COMFY_URL", "https://comfyui.tail9683c.ts.net").rstrip("/")


def _parse_args(args):
    opts = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            key = a[2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                val = args[i + 1]
                if key in opts:
                    opts[key] = [opts[key], val] if isinstance(opts[key], list) else [opts[key], val]
                else:
                    opts[key] = val
                i += 2
            else:
                opts[key] = True; i += 1
        else:
            i += 1
    return opts


def _flux_extra(opts, include_steps=False):
    d = {"unet_name": opts.get("unet"), "vae_name": opts.get("vae"),
         "clip_name": opts.get("clip")}
    if include_steps:
        d["steps"] = int(opts.get("steps", 8))
    return {k: v for k, v in d.items() if v is not None}


def _upload_opt(opts, key):
    v = opts.get(key)
    return upload_if_local(v) if v else None


def _probe_frame_count(video_path: str) -> int:
    """Client-side ffprobe to get exact frame count of a video. Needed by
    the `continuation` workflow because GetImageRangeFromBatch doesn't
    support negative indexing — we must know total frames to compute
    start_index for the last-N slice."""
    if not video_path:
        raise ValueError("continuation requires --prev_video")
    import subprocess as _sp
    # -count_packets is fast and accurate for h264 mp4s
    r = _sp.run(["ffprobe", "-v", "error", "-count_packets",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except Exception:
        # Fallback: duration × framerate
        r = _sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=duration,r_frame_rate",
                     "-of", "default=nw=1:nk=1", video_path],
                    capture_output=True, text=True)
        lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        if len(lines) < 2:
            raise ValueError(f"could not probe frame count for {video_path}")
        rate_n, rate_d = lines[0].split("/")
        fps = float(rate_n) / float(rate_d)
        dur = float(lines[1])
        return int(round(dur * fps))


def _stt_free_models():
    # 'Apply Whisper' has no unload_models flag; older models pile up
    # across runs. Ask comfy to drop them before loading a new one.
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/free", method="POST",
            data=b'{"unload_models": true, "free_memory": true}',
            headers={"Content-Type": "application/json"}),
            timeout=5).read()
    except Exception:
        pass


def _h_stt(opts, seed, prompt):
    wf = post.transcribe(
        audio_filename=upload_if_local(opts.get("audio", "")),
        model_size=opts.get("model_size", "large-v3-turbo"),
        language=opts.get("language", "auto"),
        filename_prefix=opts.get("prefix", "transcript"))
    _stt_free_models()
    return wf


def _h_vconcat(opts, seed, prompt):
    video_list = [v.strip() for v in opts.get("videos", "").split(",") if v.strip()]
    if not video_list:
        print("vconcat requires --videos a.mp4,b.mp4,c.mp4"); sys.exit(1)
    trim_list = [float(t.strip()) for t in opts.get("trim_durations", "").split(",") if t.strip()]
    start_list = [float(t.strip()) for t in opts.get("trim_starts", "").split(",") if t.strip()]
    # Auto-route: use the stream-copy MergingVideoByTwo path when no trim is
    # needed. `--fast` forces it; `--no-fast` forces the BatchImagesNode path
    # (slow, high-mem, but supports per-clip trim). Without either flag, we
    # pick fast when trim params are absent or all identity (starts all 0
    # AND durations empty).
    trim_required = bool(trim_list and any(d > 0 for d in trim_list)) or \
                    bool(start_list and any(s > 0 for s in start_list))
    fast_requested = bool(opts.get("fast"))
    fast_forbidden = bool(opts.get("no_fast"))
    use_fast = (fast_requested or not trim_required) and not fast_forbidden
    if use_fast:
        # Stage all clips into a per-run subfolder so MergingVideoByPlenty
        # sees only our inputs (in numeric-prefixed order) and nothing else.
        import uuid as _uuid
        subfolder = f"concat_{_uuid.uuid4().hex[:8]}"
        post.stage_clips_for_concat(video_list, subfolder=subfolder)
        audio_name = None
        if opts.get("audio"):
            audio_name = upload_if_local(opts["audio"])
        return post.concat_videos_ffmpeg(
            staged_subfolder=subfolder,
            audio_filename=audio_name,
            filename_prefix=opts.get("prefix", "vconcat"))
    return post.concat_videos(
        video_filenames=[upload_if_local(v) for v in video_list],
        audio_filename=_upload_opt(opts, "audio"),
        fps=float(opts.get("fps", 24.0)),
        trim_durations=trim_list or None,
        trim_starts=start_list or None,
        filename_prefix=opts.get("prefix", "vconcat"),
        format=opts.get("format", "mp4"),
        codec=opts.get("codec", "h264"))


def _h_run(opts, seed, prompt):
    src = opts.get("file", opts.get("workflow", "-"))
    if src == "-":
        return json.load(sys.stdin)
    with open(src) as f:
        return json.load(f)


HANDLERS = {
    "t2i": lambda opts, seed, prompt: flux2.flux2_text_to_image(
        prompt=prompt,
        width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
        steps=int(opts.get("steps", 8)),
        filename_prefix=opts.get("prefix", "flux2_t2i"),
        seed=seed, **_flux_extra(opts)),
    "i2i": lambda opts, seed, prompt: flux2.flux2_single_image_edit(
        image_filename=upload_if_local(opts.get("image", "")), prompt=prompt,
        width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
        steps=int(opts.get("steps", 8)),
        filename_prefix=opts.get("prefix", "flux2_i2i"),
        seed=seed, **_flux_extra(opts)),
    "i2i2": lambda opts, seed, prompt: flux2.flux2_double_image_edit(
        image1_filename=upload_if_local(opts.get("image1", "")),
        image2_filename=upload_if_local(opts.get("image2", "")),
        prompt=prompt,
        width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
        steps=int(opts.get("steps", 8)),
        filename_prefix=opts.get("prefix", "flux2_i2i2"),
        seed=seed, **_flux_extra(opts)),
    "multiprompt": lambda opts, seed, prompt: flux2.flux2_multiple_angles(
        image_filename=upload_if_local(opts.get("image", "")),
        angle_prompts=[p.strip() for p in opts.get(
            "prompts", "front view\nside view\n3/4 view").splitlines() if p.strip()],
        prepend=opts.get("prepend", ""), append=opts.get("append", ""),
        filename_prefix=opts.get("prefix", "flux2_multiprompt"),
        **_flux_extra(opts, include_steps=True)),
    "i2i2multi": lambda opts, seed, prompt: flux2.flux2_double_image_edit_multiprompt(
        image1_filename=upload_if_local(opts.get("image1", "")),
        image2_filename=upload_if_local(opts.get("image2", "")),
        angle_prompts=[p.strip() for p in opts.get(
            "prompts", "").splitlines() if p.strip()],
        prepend=opts.get("prepend", ""), append=opts.get("append", ""),
        width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
        filename_prefix=opts.get("prefix", "flux2_i2i2_multi"),
        seed=seed, **_flux_extra(opts, include_steps=True)),
    "i2iN": lambda opts, seed, prompt: flux2.flux2_multi_reference_edit(
        image_filenames=[upload_if_local(p.strip())
                         for p in opts.get("images", "").split(",") if p.strip()],
        prompt=prompt,
        width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
        filename_prefix=opts.get("prefix", "flux2_i2iN"),
        seed=seed, **_flux_extra(opts, include_steps=True)),
    "i2iNmulti": lambda opts, seed, prompt: flux2.flux2_multi_reference_edit_multiprompt(
        image_filenames=[upload_if_local(p.strip())
                         for p in opts.get("images", "").split(",") if p.strip()],
        angle_prompts=[p.strip() for p in opts.get(
            "prompts", "").splitlines() if p.strip()],
        prepend=opts.get("prepend", ""), append=opts.get("append", ""),
        width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
        filename_prefix=opts.get("prefix", "flux2_i2iN_multi"),
        seed=seed, **_flux_extra(opts, include_steps=True)),
    "t2v": lambda opts, seed, prompt: ltx2.ltx2_text_to_video(
        prompt=prompt,
        seconds=float(opts.get("seconds", 5)), fps=int(opts.get("fps", 24)),
        width=int(opts.get("width", 768)), height=int(opts.get("height", 512)),
        filename_prefix=opts.get("prefix", "ltx2_t2v"),
        negative=opts.get("negative"), fast=bool(opts.get("fast")),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        seed=seed),
    "i2v": lambda opts, seed, prompt: ltx2.ltx2_image_to_video(
        image_filename=upload_if_local(opts.get("image", "")), prompt=prompt,
        seconds=float(opts.get("seconds", 5)), fps=int(opts.get("fps", 24)),
        width=int(opts.get("width", 768)), height=int(opts.get("height", 512)),
        filename_prefix=opts.get("prefix", "ltx2_i2v"),
        negative=opts.get("negative"), fast=bool(opts.get("fast")),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        seed=seed),
    "ia2v": lambda opts, seed, prompt: ltx2.ltx2_image_audio_to_video(
        image_filename=upload_if_local(opts.get("image", "")),
        audio_filename=upload_if_local(opts.get("audio", "")), prompt=prompt,
        seconds=float(opts.get("seconds", 5)), fps=int(opts.get("fps", 24)),
        width=int(opts.get("width", 768)), height=int(opts.get("height", 512)),
        filename_prefix=opts.get("prefix", "ltx2_ia2v"),
        negative=opts.get("negative"), fast=bool(opts.get("fast")),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        image_refs=[upload_if_local(r.strip()) for r in opts.get(
            "image_refs", "").split(",") if r.strip()] or None,
        base_guide_strength=float(opts.get("base_guide_strength", 0.5)),
        refine_guide_strength=float(opts.get("refine_guide_strength", 0.3)),
        identity_anchor_image=_upload_opt(opts, "identity_anchor"),
        identity_strength=float(opts.get("identity_strength", 0.3)),
        seed=seed),
    "continuation": lambda opts, seed, prompt: ltx2.ltx2_continuation_to_video(
        prev_video_filename=upload_if_local(opts.get("prev_video", "")),
        prev_total_frames=(int(opts["prev_frames"]) if opts.get("prev_frames")
                           else _probe_frame_count(opts.get("prev_video",""))),
        prompt=prompt,
        overlap_seconds=float(opts.get("overlap_seconds", 1.0)),
        overlap_strength=float(opts.get("overlap_strength", 1.0)),
        seconds=float(opts.get("seconds", 5)), fps=int(opts.get("fps", 24)),
        width=int(opts.get("width", 448)), height=int(opts.get("height", 832)),
        filename_prefix=opts.get("prefix", "ltx2_continuation"),
        negative=opts.get("negative"),
        audio_filename=_upload_opt(opts, "audio"),
        identity_anchor_image=_upload_opt(opts, "identity_anchor"),
        identity_strength=float(opts.get("identity_strength", 0.3)),
        fast=bool(opts.get("fast")),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        seed=seed),
    "multiguide": lambda opts, seed, prompt: ltx2.ltx2_multi_guide_to_video(
        guide_filenames=[upload_if_local(r.strip())
                         for r in opts.get("guides", "").split(",") if r.strip()],
        guide_frame_indices=(
            [int(x.strip()) for x in opts["frame_indices"].split(",") if x.strip()]
            if opts.get("frame_indices") else None),
        guide_strengths=(
            [float(x.strip()) for x in opts["strengths"].split(",") if x.strip()]
            if opts.get("strengths") else None),
        prompt=prompt,
        seconds=float(opts.get("seconds", 8)), fps=int(opts.get("fps", 24)),
        width=int(opts.get("width", 448)), height=int(opts.get("height", 832)),
        filename_prefix=opts.get("prefix", "ltx2_multiguide"),
        negative=opts.get("negative"),
        audio_filename=_upload_opt(opts, "audio"),
        use_transition_lora=(False if opts.get("no_transition_lora") else True),
        transition_lora_strength=float(opts.get("transition_lora_strength", 1.0)),
        fast=bool(opts.get("fast")),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        seed=seed),
    "flf2v": lambda opts, seed, prompt: ltx2.ltx2_first_last_frame_to_video(
        first_frame_filename=upload_if_local(opts.get("first", "")),
        last_frame_filename=upload_if_local(opts.get("last", "")),
        prompt=prompt,
        seconds=float(opts.get("seconds", 5)), fps=int(opts.get("fps", 25)),
        width=int(opts.get("width", 1280)), height=int(opts.get("height", 720)),
        filename_prefix=opts.get("prefix", "ltx2_flf2v"),
        negative=opts.get("negative"),
        guide_strength=float(opts.get("guide_strength", 0.7)),
        first_guide_strength=(float(opts["first_guide_strength"]) if opts.get("first_guide_strength") else None),
        last_guide_strength=(float(opts["last_guide_strength"]) if opts.get("last_guide_strength") else None),
        audio_filename=_upload_opt(opts, "audio"),
        use_transition_lora=bool(opts.get("use_transition_lora")),
        transition_lora_strength=float(opts.get("transition_lora_strength", 1.0)),
        fast=bool(opts.get("fast")),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        seed=seed),
    "transition": lambda opts, seed, prompt: ltx2.ltx2_transition(
        first_frame_filename=upload_if_local(opts.get("first", "")),
        last_frame_filename=upload_if_local(opts.get("last", "")),
        prompt=prompt,
        seconds=float(opts.get("seconds", 3.0)), fps=int(opts.get("fps", 24)),
        width=int(opts.get("width", 720)), height=int(opts.get("height", 1280)),
        filename_prefix=opts.get("prefix", "ltx2_transition"),
        negative=opts.get("negative"),
        first_guide_strength=float(opts.get("first_guide_strength", 1.0)),
        last_guide_strength=float(opts.get("last_guide_strength", 1.0)),
        audio_filename=_upload_opt(opts, "audio"),
        prev_video_filename=_upload_opt(opts, "prev_video"),
        next_video_filename=_upload_opt(opts, "next_video"),
        multiframe_guide=(int(opts["multiframe_guide"]) if opts.get("multiframe_guide") else None),
        multiframe_guide_last=(int(opts["multiframe_guide_last"]) if opts.get("multiframe_guide_last") else None),
        prev_video_start_frame=(int(opts["prev_video_start_frame"]) if opts.get("prev_video_start_frame") else None),
        next_video_start_frame=(int(opts["next_video_start_frame"]) if opts.get("next_video_start_frame") else None),
        prev_video_tail_buffer_sec=float(opts.get("prev_video_buffer", 0.0)),
        slice_song_start_sec=(float(opts["slice_song_start_sec"]) if opts.get("slice_song_start_sec") else None),
        prev_video_song_start_sec=(float(opts["prev_video_song_start_sec"]) if opts.get("prev_video_song_start_sec") else None),
        next_video_song_start_sec=(float(opts["next_video_song_start_sec"]) if opts.get("next_video_song_start_sec") else None),
        next_video_vocal_offset_sec=float(opts.get("next_video_vocal_offset_sec", 0.0)),
        mask_start_sec=(float(opts["mask_start_sec"]) if opts.get("mask_start_sec") else None),
        mask_end_sec=(float(opts["mask_end_sec"]) if opts.get("mask_end_sec") else None),
        mask_max_length=opts.get("mask_max_length", "pad"),
        use_mask=(False if opts.get("no_mask") else True),
        use_inplace=(False if opts.get("no_inplace") else True),
        use_addguide=bool(opts.get("use_addguide") or opts.get("addguide")),
        addguide_strength=float(opts.get("addguide_strength", 0.6)),
        last_guide_index=(int(opts["last_guide_index"]) if opts.get("last_guide_index") else None),
        fast=bool(opts.get("fast")),
        b_sparse_latent_positions=(
            [int(x.strip()) for x in opts["b_sparse_latent_positions"].split(",") if x.strip()]
            if opts.get("b_sparse_latent_positions") else None),
        camera_lora=opts.get("camera_lora"),
        camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
        debug_save_audio=bool(opts.get("debug_save_audio")),
        seed=seed),
    "tts": lambda opts, seed, prompt: tts.qwen_tts(
        text=opts.get("text", prompt), filename_prefix=opts.get("prefix", "tts")),
    "stems": lambda opts, seed, prompt: post.extract_stems(
        audio_filename=upload_if_local(opts.get("audio", "")),
        model_name=opts.get("model", "MelBandRoformer_fp16.safetensors"),
        filename_prefix=opts.get("prefix", "stems")),
    "stt": _h_stt,
    "vconcat": _h_vconcat,
    "run": _h_run,
    "last_frame": lambda opts, seed, prompt: ltx2.extract_last_frame(
        video_server_path=opts.get("video_path", ""),
        filename_prefix=opts.get("prefix", "last_frame")),
}

# Commands with non-default timeouts. Values are ints, used as fallback
# when --timeout isn't passed on the CLI (opts overrides this).
DEFAULT_TIMEOUTS = {"tts": 120, "stems": 600, "stt": 300, "vconcat": 1200}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)

    dump_only = args[0] == "dump"
    if dump_only:
        args = args[1:]
    if not args:
        print(__doc__); sys.exit(1)

    cmd  = args[0]
    opts = _parse_args(args[1:])

    # Route to the right comfy server based on command class. Lets bots
    # call t2v / i2v / ia2v without knowing the URL; lets t2i / i2i stay
    # on the flux server.
    global BASE
    BASE = _resolve_base_url(cmd)
    core.BASE = BASE
    # Some submodules also cache BASE at import time — sync them if present.
    for mod in (flux2, ltx2, tts, post):
        if hasattr(mod, "BASE"):
            setattr(mod, "BASE", BASE)

    output_dir  = Path(opts.get("output_dir", "outputs"))
    timeout     = int(opts.get("timeout", DEFAULT_TIMEOUTS.get(cmd, 600)))
    seed_raw    = opts.get("seed")
    seed        = int(seed_raw) if seed_raw else None
    notify      = opts.get("notify_target") or os.environ.get("OPENCLAW_NOTIFY_TARGET")
    caption_tpl = opts.get("caption_template")
    prompt      = opts.get("prompt", "")

    handler = HANDLERS.get(cmd)
    if handler is None:
        print(f"Unknown command: {cmd}\n{__doc__}"); sys.exit(1)

    wf = handler(opts, seed, prompt)

    if dump_only:
        print(json.dumps(wf, indent=2))
        return

    prompt_id = _submit_and_wait(wf, output_dir, timeout, notify=notify,
                                 caption_template=caption_tpl,
                                 user_prompt=prompt)

    # Save SRT writes the file server-side and returns its absolute path
    # as its STRING output. We capture it via ShowText and use the path's
    # basename + parent-folder to fetch the real SRT via /view. Order
    # matches post.py: [0]=plain transcript, [1]=segments SRT path,
    # [2]=words SRT path.
    if cmd == "stt" and prompt_id:
        import urllib.parse as up
        hist = json.loads(urllib.request.urlopen(
            f"{BASE}/history/{prompt_id}", timeout=15).read())
        entry = hist.get(prompt_id, {})
        captured = []
        for nid in sorted(entry.get("outputs", {}).keys(), key=int):
            nout = entry["outputs"][nid]
            if "text" in nout and isinstance(nout["text"], list):
                captured.extend([t for t in nout["text"] if isinstance(t, str)])
        pfx = opts.get("prefix", "transcript")
        output_dir.mkdir(parents=True, exist_ok=True)
        if captured:
            (output_dir / f"{pfx}.txt").write_text(captured[0])
            print(f"wrote: {output_dir / f'{pfx}.txt'} ({len(captured[0])} chars)")
        for srt_path in captured[1:]:
            fname = Path(srt_path.strip()).name
            subfolder = Path(srt_path.strip()).parent.name  # typically 'srt'
            url = f"{BASE}/view?filename={up.quote(fname)}&subfolder={up.quote(subfolder)}&type=output"
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    if r.status == 200:
                        dest = output_dir / fname
                        dest.write_bytes(r.read())
                        print(f"fetched: {dest} ({dest.stat().st_size} bytes)")
            except Exception as e:
                print(f"[WARN] could not fetch {srt_path}: {e}", file=sys.stderr)

    # vconcat (fast / ffmpeg path): MergingVideoByPlenty + AddAudio write the
    # final file directly into the server's output dir. The ShowText sink
    # captures its path — fetch it here so the caller sees the mp4 locally.
    if cmd == "vconcat" and prompt_id and not opts.get("no_fast"):
        import urllib.parse as up
        hist = json.loads(urllib.request.urlopen(
            f"{BASE}/history/{prompt_id}", timeout=15).read())
        entry = hist.get(prompt_id, {})
        captured = []
        for nid in sorted(entry.get("outputs", {}).keys(), key=int):
            nout = entry["outputs"][nid]
            if "text" in nout and isinstance(nout["text"], list):
                captured.extend([t for t in nout["text"] if isinstance(t, str)])
        # The final file path is the LAST captured string (AddAudio wins
        # over MergingVideoByPlenty if audio was provided).
        if captured:
            final_path = Path(captured[-1].strip())
            fname = final_path.name
            subfolder = final_path.parent.name if final_path.parent.name != "output" else ""
            url = f"{BASE}/view?filename={up.quote(fname)}&subfolder={up.quote(subfolder)}&type=output"
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    if r.status == 200:
                        dest = output_dir / fname
                        dest.write_bytes(r.read())
                        print(f"asset_url: {url}")
                        print(f"saved: {dest} ({dest.stat().st_size // 1024} KB)")
            except Exception as e:
                print(f"[WARN] could not fetch {final_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
