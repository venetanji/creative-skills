"""Curated native MiniMax H3 FL8 image/text and REF20 generated-AV graphs.

Pure/offline only: this module neither imports ComfyUI nor checks asset/GPU
availability. Bounds are captured schema/training limits, not live certification.
"""
from __future__ import annotations

import re

from .graph import WorkflowGraph


UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _integer(name: str, value: int, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def build_t2v(
    prompt: str, *, width: int = 864, height: int = 480, frames: int = 124,
    seed: int = 0, filename_prefix: str = "video/minimax_h3",
) -> dict:
    """Build exactly one FL8step joint AV graph, at native 24 fps.

    No images, supplied audio, negative CFG, control, asset overrides or
    alternate sampling recipe are supported. Explicit frames must already
    lie on H3's 17k+5 grid; there is no duration rounding or silent fallback.
    """
    if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
        raise ValueError("prompt must be a nonempty string without NUL")
    for name, value in (("width", width), ("height", height)):
        _integer(name, value, 32, 16384)
        if value % 32:
            raise ValueError(f"{name} must be a multiple of 32")
    _integer("frames", frames, 124, 362)
    if (frames - 5) % 17:
        raise ValueError("frames must lie on the native 17k+5 grid (124..362)")
    _integer("seed", seed, 0, 2**64 - 1)
    # SaveVideo supports path/template expansion. Admit only literal relative
    # segments, never absolute/traversal paths, backslashes or format macros.
    if (not isinstance(filename_prefix, str) or len(filename_prefix) > 200
            or re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", filename_prefix) is None):
        raise ValueError("filename_prefix must be a safe relative literal prefix (max 200 characters)")

    graph = WorkflowGraph()
    unet = graph.node("UNETLoader", unet_name=UNET, weight_dtype="default")
    lora = graph.node("LoraLoaderModelOnly", model=unet, lora_name=LORA, strength_model=1.0)
    model = graph.node("MiniMaxH3SigmaShift", model=lora, shift_video=12.0, shift_audio=3.0)
    clip = graph.node("CLIPLoader", clip_name=CLIP, type="minimax", device="default")
    video_vae = graph.node("VAELoader", vae_name=VIDEO_VAE)
    audio_vae = graph.node("VAELoader", vae_name=AUDIO_VAE)
    # Native ImageToVideo without optional frames supplies text conditioning
    # and the joint video/stereo-audio latent; no LTX latent/audio substitutes.
    av = graph.node("MiniMaxH3ImageToVideo", clip=clip, vae=video_vae,
                    prompt=prompt, width=width, height=height, length=frames)
    guider = graph.node("BasicGuider", model=model, conditioning=av[0])
    noise = graph.node("RandomNoise", noise_seed=seed)
    sampler = graph.node("KSamplerSelect", sampler_name="euler")
    sigmas = graph.node("BasicScheduler", model=model, scheduler="simple", steps=8, denoise=1.0)
    samples = graph.node("SamplerCustomAdvanced", noise=noise, guider=guider,
                         sampler=sampler, sigmas=sigmas, latent_image=av[1])
    images = graph.node("VAEDecode", samples=samples[0], vae=video_vae)
    audio = graph.node("VAEDecodeAudio", samples=samples[0], vae=audio_vae)
    video = graph.node("CreateVideo", images=images, fps=24.0, audio=audio, bit_depth=8)
    graph.node("SaveVideo", video=video, filename_prefix=filename_prefix, format="mp4", codec="h264")
    return graph.to_dict()


REF_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def _image_name(value: str) -> str:
    # Names are managed input basenames, not local paths or annotated outputs.
    if (not isinstance(value, str) or len(value) > 200
            or re.fullmatch(r"[A-Za-z0-9_-]+\.(?:png|jpg|jpeg)", value) is None):
        raise ValueError("image filename must be a safe input basename ending in png/jpg/jpeg (max 200 characters)")
    return value


def build_i2v(
    prompt: str, *, image_filename: str, width: int = 864, height: int = 480,
    frames: int = 124, seed: int = 0, filename_prefix: str = "video/minimax_h3",
) -> dict:
    """FL8 first-frame anchor only; native stretch/keyframe 0, no strength."""
    name = _image_name(image_filename)
    graph = build_t2v(prompt, width=width, height=height, frames=frames,
                      seed=seed, filename_prefix=filename_prefix)
    graph["16"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    graph["6"]["inputs"]["first_frame"] = ["16", 0]
    return graph


def build_ia2v(
    prompt: str, *, image_filename: str, audio_filename: str,
    width: int = 864, height: int = 480, frames: int = 124,
    seed: int = 0, filename_prefix: str = "video/minimax_h3",
) -> dict:
    """Native FL8 first-frame/audio conditioning with the source soundtrack muxed."""
    if (not isinstance(audio_filename, str) or len(audio_filename) > 200 or
            re.fullmatch(r"[A-Za-z0-9_-]+\.(?:wav|mp3|flac|ogg|m4a)", audio_filename) is None):
        raise ValueError("audio filename must be a managed audio input basename")
    graph = build_i2v(prompt, image_filename=image_filename, width=width, height=height,
                      frames=frames, seed=seed, filename_prefix=filename_prefix)
    graph["17"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_filename}}
    graph["18"] = {"class_type": "TrimAudioDuration", "inputs": {
        "audio": ["17", 0], "start_index": 0.0, "duration": frames / 24,
    }}
    graph["19"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
        "positive": ["6", 0], "latent": ["6", 1], "audio_vae": ["5", 0],
        "audio": ["18", 0], "frame_idx": 0,
    }}
    graph["7"]["inputs"]["conditioning"] = ["19", 0]
    graph["14"]["inputs"]["audio"] = ["18", 0]
    del graph["13"]
    return graph


def build_r2v(
    prompt: str, *, reference_filenames: list[str] | tuple[str, ...],
    ref_image_size: str = "match", width: int = 864, height: int = 480,
    frames: int = 124, seed: int = 0, filename_prefix: str = "video/minimax_h3",
) -> dict:
    """REF20 semantic references in Picture order, with video-VAE latents.

    The active upstream false branches select base REF weights and 20 steps.
    No FL SigmaShift/LoRA or REF4 shortcut. V3 Autogrow API ports are dotted
    direct links, which the backend expands into the ordered ref_images dict.
    """
    if not isinstance(reference_filenames, (list, tuple)) or not 1 <= len(reference_filenames) <= 2:
        raise ValueError("reference_filenames must contain 1..2 ordered image basenames")
    names = [_image_name(name) for name in reference_filenames]
    if not isinstance(ref_image_size, str) or ref_image_size not in ("match", "max"):
        raise ValueError("ref_image_size must be match or max")
    graph = build_t2v(prompt, width=width, height=height, frames=frames,
                      seed=seed, filename_prefix=filename_prefix)
    # Reuse the shared AV decode spine, but remove both FL-only model patches.
    del graph["1"], graph["2"]
    graph["0"]["inputs"]["unet_name"] = REF_UNET
    graph["7"]["inputs"]["model"] = ["0", 0]
    graph["9"]["inputs"]["sampler_name"] = "res_multistep"
    graph["10"]["inputs"].update(model=["0", 0], steps=20)
    graph["6"]["class_type"] = "MiniMaxH3ReferenceToVideo"
    graph["6"]["inputs"].update(audio_vae=["5", 0], ref_image_size=ref_image_size)
    for index, name in enumerate(names):
        nid = str(16 + index)
        graph[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["6"]["inputs"][f"ref_images.ref_image_{index}"] = [nid, 0]
    return graph
