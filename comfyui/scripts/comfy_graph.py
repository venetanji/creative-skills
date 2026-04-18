#!/usr/bin/env python3
"""CLI for ComfyUI workflow submission at https://comfyui.tail9683c.ts.net

Usage:
  python comfy_graph.py t2i   --prompt "a cat" [--notify-target discord:...]
  python comfy_graph.py i2i   --image photo.jpg --prompt "cat wearing hat"
  python comfy_graph.py t2v   --prompt "a forest" --seconds 5
  python comfy_graph.py i2v   --image photo.jpg --prompt "person walking"
  python comfy_graph.py ia2v  --image photo.jpg --audio track.mp3 --prompt "..."
  python comfy_graph.py flf2v --first a.png --last b.png --prompt "..." --seconds 5
  python comfy_graph.py dump  t2i --prompt "a cat"   # print workflow JSON only

Environment:
  COMFY_URL          Base URL of ComfyUI server (default: https://comfyui.tail9683c.ts.net)
  OPENCLAW_NOTIFY_TARGET  Default notification target (e.g. discord:123...)
"""
from __future__ import annotations
import sys, os, json, time, urllib.request
from pathlib import Path
from core import _submit_and_wait, upload_if_local
import flux2, ltx2, tts
from flux2 import flux2_text_to_image, flux2_single_image_edit, flux2_double_image_edit, flux2_multiple_angles
from ltx2 import (ltx2_text_to_video, ltx2_image_to_video, ltx2_image_audio_to_video,
                   ltx2_first_last_frame_to_video, extract_last_frame)
from tts import qwen_tts, qwen_voice_clone

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

    output_dir  = Path(opts.get("output_dir", "outputs"))
    timeout     = int(opts.get("timeout", 600))
    seed_raw    = opts.get("seed")
    seed        = int(seed_raw) if seed_raw else None
    notify      = opts.get("notify_target") or os.environ.get("OPENCLAW_NOTIFY_TARGET")
    caption_tpl = opts.get("caption_template")
    prompt      = opts.get("prompt", "")

    generation_cmds = {"t2i", "i2i", "i2i2", "angles", "t2v", "i2v", "ia2v", "flf2v", "tts", "last_frame", "run"}

    if cmd == "t2i":
        extra = {k: v for k, v in {
            "unet_name": opts.get("unet"),
            "vae_name": opts.get("vae"),
            "clip_name": opts.get("clip"),
        }.items() if v is not None}
        wf = flux2.flux2_text_to_image(
            prompt=prompt,
            width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
            steps=int(opts.get("steps", 4)),
            filename_prefix=opts.get("prefix", "flux2_t2i"),
            seed=seed,
            **extra)

    elif cmd == "i2i":
        extra = {k: v for k, v in {
            "unet_name": opts.get("unet"),
            "vae_name": opts.get("vae"),
            "clip_name": opts.get("clip"),
        }.items() if v is not None}
        img = upload_if_local(opts.get("image", ""))
        wf = flux2.flux2_single_image_edit(
            image_filename=img, prompt=prompt,
            width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
            steps=int(opts.get("steps", 4)),
            filename_prefix=opts.get("prefix", "flux2_i2i"),
            seed=seed,
            **extra)

    elif cmd == "i2i2":
        extra = {k: v for k, v in {
            "unet_name": opts.get("unet"),
            "vae_name": opts.get("vae"),
            "clip_name": opts.get("clip"),
        }.items() if v is not None}
        img1 = upload_if_local(opts.get("image1", ""))
        img2 = upload_if_local(opts.get("image2", ""))
        wf = flux2.flux2_double_image_edit(
            image1_filename=img1, image2_filename=img2, prompt=prompt,
            width=int(opts.get("width", 1024)), height=int(opts.get("height", 576)),
            steps=int(opts.get("steps", 4)),
            filename_prefix=opts.get("prefix", "flux2_i2i2"),
            seed=seed,
            **extra)

    elif cmd == "multiprompt":
        extra = {k: v for k, v in {
            "unet_name": opts.get("unet"),
            "vae_name": opts.get("vae"),
            "clip_name": opts.get("clip"),
            "steps": int(opts.get("steps", 4)),
        }.items() if v is not None}
        prompts_raw = opts.get("prompts", "front view\nside view\n3/4 view")
        prompts_list = [p.strip() for p in prompts_raw.splitlines() if p.strip()]
        img = upload_if_local(opts.get("image", ""))
        wf = flux2.flux2_multiple_angles(
            image_filename=img, angle_prompts=prompts_list,
            prepend=opts.get("prepend", ""), append=opts.get("append", ""),
            filename_prefix=opts.get("prefix", "flux2_multiprompt"),
            **extra)

    elif cmd == "t2v":
        wf = ltx2.ltx2_text_to_video(
            prompt=prompt,
            seconds=int(opts.get("seconds", 5)),
            fps=int(opts.get("fps", 24)),
            width=int(opts.get("width", 768)),
            height=int(opts.get("height", 512)),
            filename_prefix=opts.get("prefix", "ltx2_t2v"),
            negative=opts.get("negative"),
            fast=bool(opts.get("fast")),
            camera_lora=opts.get("camera_lora"),
            camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
            seed=seed)

    elif cmd == "i2v":
        img = upload_if_local(opts.get("image", ""))
        wf = ltx2.ltx2_image_to_video(
            image_filename=img, prompt=prompt,
            seconds=int(opts.get("seconds", 5)),
            fps=int(opts.get("fps", 24)),
            width=int(opts.get("width", 768)),
            height=int(opts.get("height", 512)),
            filename_prefix=opts.get("prefix", "ltx2_i2v"),
            negative=opts.get("negative"),
            fast=bool(opts.get("fast")),
            camera_lora=opts.get("camera_lora"),
            camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
            seed=seed)

    elif cmd == "ia2v":
        img = upload_if_local(opts.get("image", ""))
        aud = upload_if_local(opts.get("audio", ""))
        wf = ltx2.ltx2_image_audio_to_video(
            image_filename=img, audio_filename=aud, prompt=prompt,
            seconds=int(opts.get("seconds", 5)),
            fps=int(opts.get("fps", 24)),
            width=int(opts.get("width", 768)),
            height=int(opts.get("height", 512)),
            filename_prefix=opts.get("prefix", "ltx2_ia2v"),
            negative=opts.get("negative"),
            fast=bool(opts.get("fast")),
            camera_lora=opts.get("camera_lora"),
            camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
            seed=seed)

    elif cmd == "flf2v":
        first = upload_if_local(opts.get("first", ""))
        last = upload_if_local(opts.get("last", ""))
        wf = ltx2.ltx2_first_last_frame_to_video(
            first_frame_filename=first, last_frame_filename=last, prompt=prompt,
            seconds=int(opts.get("seconds", 5)),
            fps=int(opts.get("fps", 25)),
            width=int(opts.get("width", 1280)),
            height=int(opts.get("height", 720)),
            filename_prefix=opts.get("prefix", "ltx2_flf2v"),
            negative=opts.get("negative"),
            guide_strength=float(opts.get("guide_strength", 0.7)),
            fast=bool(opts.get("fast")),
            camera_lora=opts.get("camera_lora"),
            camera_lora_strength=float(opts.get("camera_lora_strength", 0.8)),
            seed=seed)

    elif cmd == "tts":
        wf = tts.qwen_tts(
            text=opts.get("text", prompt),
            filename_prefix=opts.get("prefix", "tts"))
        timeout = int(opts.get("timeout", 120))

    elif cmd == "run":
        # Run an arbitrary workflow JSON from a file path or stdin ("-")
        src = opts.get("file", opts.get("workflow", "-"))
        if src == "-":
            wf = json.load(sys.stdin)
        else:
            with open(src) as f:
                wf = json.load(f)

    elif cmd == "last_frame":
        wf = ltx2.extract_last_frame(
            video_server_path=opts.get("video_path", ""),
            filename_prefix=opts.get("prefix", "last_frame"))

    else:
        print(f"Unknown command: {cmd}\n{__doc__}"); sys.exit(1)

    if dump_only:
        print(json.dumps(wf, indent=2))
        return

    _submit_and_wait(wf, output_dir, timeout, notify=notify,
                     caption_template=caption_tpl, user_prompt=prompt)


if __name__ == "__main__":
    main()
