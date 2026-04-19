"""LTX-2.3 video generation — quality-first two-pass refine pattern.

All four variants default to two-pass with latent upsample between passes:

  ltx2_text_to_video(prompt, ..., fast=False)
  ltx2_image_to_video(image, prompt, ..., fast=False)
  ltx2_image_audio_to_video(image, audio, prompt, ..., fast=False)
  ltx2_first_last_frame_to_video(first, last, prompt, ..., fast=False)

Two-pass pattern (default, "quality"):
  pass-1 coarse (9 sigmas, euler_ancestral_cfg_pp)
  → LTXVSeparateAVLatent
  → LTXVLatentUpsampler (2x)
  → re-apply image guidance at strength 1.0 (ImgToVideoInplace for i2v/ia2v,
    AddGuide×2 for flf2v; skipped for t2v)
  → LTXVConcatAVLatent (with pass-1 audio)
  → LTXVCropGuides (cropped conditioning from coarse video latent)
  → pass-2 refine (4 sigmas, euler_cfg_pp)
  → decode + save

Fast pattern (`fast=True`): stop after pass-1, decode its output directly. About
half the wall time, lower resolution (no upsample), worse detail — use for
iteration, not final output.

Models required on the server:
  checkpoints/ltx-2.3-22b-dev-fp8.safetensors
  text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
  loras/ltx-2.3-22b-distilled-lora-384.safetensors
  latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
"""
from core import WorkflowGraph
import time


CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"
NEG_DEFAULT = "pc game, console game, video game, cartoon, childish, ugly, blurry, low quality, watermark, distorted, still frame, text, captions, subtitles, signs, logos, lettering, typography, words, letters"

# sigmas match the production corgi workflow — do not tweak without A/B testing
SIGMAS_PASS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"
SAMPLER_PASS1 = "euler_ancestral_cfg_pp"
SAMPLER_PASS2 = "euler_cfg_pp"


# -------- shared helpers --------

def _rand_seed():
    return int(time.time() * 1000) % (2**31)


def _loaders(g, ckpt=CKPT, text_encoder=TEXT_ENCODER):
    """Load checkpoint, text encoder, audio VAE, and latent upscaler. Kept
    uniform across variants so graph shape is predictable."""
    checkpoint = g.node("CheckpointLoaderSimple", ckpt_name=ckpt)
    clip = g.node("LTXAVTextEncoderLoader",
                  text_encoder=text_encoder, ckpt_name=ckpt, device="default")
    audio_vae = g.node("LTXVAudioVAELoader", ckpt_name=ckpt)
    upscaler = g.node("LatentUpscaleModelLoader", model_name=UPSCALER)
    return checkpoint, clip, audio_vae, upscaler


def _distilled_lora(g, model, strength=0.6):
    """Required whenever using ltx-2.3-22b-dev-fp8. Comfy-Org's reference templates
    use ltx-2.3-22b-distilled-fp8 which has the LoRA pre-merged; our server only
    has the dev ckpt, so we apply it explicitly. Without it the 8-step schedule
    under-denoises — output becomes badly blurred."""
    return g.node("LoraLoaderModelOnly", model=model,
                  lora_name=DISTILLED_LORA, strength_model=strength)


CAMERA_LORAS = {
    "dolly-in":    "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly-out":   "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly-left":  "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly-right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "jib-up":      "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib-down":    "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "static":      "ltx-2-19b-lora-camera-control-static.safetensors",
}


def _apply_extra_lora(g, model, lora_name, strength=0.8):
    """Stack an extra LoRA on top of the distilled one. Accepts either a camera-
    lora shortname (e.g. 'dolly-in') from CAMERA_LORAS, or a full .safetensors
    filename as it appears on the server."""
    if not lora_name:
        return model
    if lora_name in CAMERA_LORAS:
        lora_name = CAMERA_LORAS[lora_name]
    return g.node("LoraLoaderModelOnly", model=model[0],
                  lora_name=lora_name, strength_model=float(strength))


def _encode_prompts(g, clip, positive, negative, fps):
    """CLIP encode prompts and wrap in LTXVConditioning. Returns the
    LTXVConditioning NodeRef — [0]=positive, [1]=negative."""
    pos = g.node("CLIPTextEncode", text=positive, clip=clip)
    neg = g.node("CLIPTextEncode", text=negative or NEG_DEFAULT, clip=clip)
    return g.node("LTXVConditioning", positive=pos[0], negative=neg[0], frame_rate=float(fps))


def _round_length(seconds, fps):
    """LTX-V requires video length ≡ 1 (mod 8)."""
    raw = int(seconds * fps)
    return ((raw // 8) * 8) + 1


def _image_branch(g, image_filename, width, height, img_compression=18):
    """Preprocess user image for LTXV image-conditioning."""
    loaded = g.node("LoadImage", image=image_filename)
    resized = g.node("ResizeImageMaskNode", **{
        "input": loaded[0],
        "resize_type": "scale dimensions",
        "resize_type.width": int(width),
        "resize_type.height": int(height),
        "resize_type.crop": "center",
        "scale_method": "lanczos",
    })
    longer = g.node("ResizeImagesByLongerEdge", images=resized[0], longer_edge=1536)
    return g.node("LTXVPreprocess", image=longer[0], img_compression=img_compression)


def _image_branch_multi(g, image_filenames, width, height, img_compression=18):
    """Preprocess N reference images and concatenate into one IMAGE batch for
    LTXVImgToVideoInplace. First ref is the canonical identity (e.g. anchor.png);
    subsequent refs are variation (per-scene anchor, previous-scene last frame)."""
    branches = [_image_branch(g, f, width, height, img_compression) for f in image_filenames]
    if len(branches) == 1:
        return branches[0]
    out = branches[0]
    for b in branches[1:]:
        out = g.node("ImageBatch", image1=out[0], image2=b[0])
    return out


def _flf2v_preprocess_frame(g, image_filename, width, height):
    """FLF2V frame preprocess — no longer-edge resize (frames are already at
    target resolution); nearest-exact scaler + img_compression=25 per ref template."""
    loaded = g.node("LoadImage", image=image_filename)
    resized = g.node("ResizeImageMaskNode", **{
        "input": loaded[0],
        "resize_type": "scale dimensions",
        "resize_type.width": int(width),
        "resize_type.height": int(height),
        "resize_type.crop": "center",
        "scale_method": "nearest-exact",
    })
    return g.node("LTXVPreprocess", image=resized[0], img_compression=25)


def _audio_from_file(g, audio_filename, seconds, width, height, audio_vae,
                     length=None, fps=None):
    """Load and trim audio to the same number of seconds as the rendered
    video. `length`+`fps` are optional; when provided, trim to length/fps
    instead of the raw `seconds` input — that aligns audio to LTX's
    actually-rendered frame count (which `_round_length` snapped to
    `8k+1`). Otherwise audio can overhang the video by ~0.2s and LTX's
    audio-VAE truncation chops the TAIL — exactly where a transition's
    target vocal starts."""
    loaded = g.node("LoadAudio", audio=audio_filename)
    if length is not None and fps:
        effective = float(length) / float(fps)
    else:
        effective = float(seconds)
    trimmed = g.node("TrimAudioDuration", audio=loaded[0],
                     duration=effective, start_index=0)
    encoded = g.node("LTXVAudioVAEEncode", audio=trimmed[0], audio_vae=audio_vae[0])
    mask = g.node("SolidMask", value=0, width=int(width), height=int(height))
    return g.node("SetLatentNoiseMask", samples=encoded[0], mask=mask[0])


def _empty_audio_latent(g, length, fps, audio_vae):
    return g.node("LTXVEmptyLatentAudio",
                  frames_number=int(length), frame_rate=float(fps),
                  batch_size=1, audio_vae=audio_vae[0])


def _base_video_latent(g, width, height, length, vae, image_ref, strength=0.7):
    """Empty video latent, optionally conditioned on a single reference image
    (or a multi-frame IMAGE batch) via LTXVImgToVideoInplace. Lower strength
    means the refs act as a soft identity prior instead of a locked first
    frame — use ~0.4-0.6 for lipsync where the character should be free to
    move, dance, and act rather than statically singing into the camera."""
    empty = g.node("EmptyLTXVLatentVideo",
                   width=int(width), height=int(height),
                   length=int(length), batch_size=1)
    if image_ref is None:
        return empty
    bypass = g.node("PrimitiveBoolean", value=False)
    return g.node("LTXVImgToVideoInplace",
                  vae=vae, image=image_ref[0], latent=empty[0],
                  strength=strength, bypass=bypass[0])


def _pass_one(g, model, cond, av_latent, seed):
    """Coarse sample — 9 sigmas (8 steps), euler_ancestral_cfg_pp."""
    sampler = g.node("KSamplerSelect", sampler_name=SAMPLER_PASS1)
    sigmas = g.node("ManualSigmas", sigmas=SIGMAS_PASS1)
    noise = g.node("RandomNoise", noise_seed=seed)
    guider = g.node("CFGGuider", model=model[0], positive=cond[0], negative=cond[1], cfg=1.0)
    return g.node("SamplerCustomAdvanced",
                  noise=noise[0], guider=guider[0], sampler=sampler[0],
                  sigmas=sigmas[0], latent_image=av_latent[0])


def _pass_two(g, model, cond, av_latent, seed):
    """Refine sample — 4 sigmas (3 steps), euler_cfg_pp."""
    sampler = g.node("KSamplerSelect", sampler_name=SAMPLER_PASS2)
    sigmas = g.node("ManualSigmas", sigmas=SIGMAS_PASS2)
    noise = g.node("RandomNoise", noise_seed=seed)
    guider = g.node("CFGGuider", model=model[0], positive=cond[0], negative=cond[1], cfg=1.0)
    return g.node("SamplerCustomAdvanced",
                  noise=noise[0], guider=guider[0], sampler=sampler[0],
                  sigmas=sigmas[0], latent_image=av_latent[0])


def _upsample_between(g, av_pass1, cond, vae, upscaler, image_ref,
                       refine_guide_strength=0.3):
    """Between-pass: separate pass-1, LTXVCropGuides for cond, upsample,
    optionally re-apply the image ref via LTXVImgToVideoInplace at a
    configurable strength, concat with pass-1 audio.

    Historically this called LTXVImgToVideoInplace(s=1.0) which HARD-LOCKED
    the first frames to the anchor, producing static-looking openings. At
    strength 0 we skip the re-apply entirely (pass-2 refines freely over
    pass-1's already-conditioned latent); at 0.3-0.5 the anchor bleeds
    identity through the refine without pinning motion.

    Returns (av_latent_for_pass2, cropped_conditioning)."""
    sep = g.node("LTXVSeparateAVLatent", av_latent=av_pass1[0])
    cropped = g.node("LTXVCropGuides",
                     positive=cond[0], negative=cond[1], latent=sep[0])
    upsampled = g.node("LTXVLatentUpsampler",
                       samples=sep[0], upscale_model=upscaler[0], vae=vae)
    if image_ref is not None and refine_guide_strength > 0:
        bypass = g.node("PrimitiveBoolean", value=False)
        video_re = g.node("LTXVImgToVideoInplace",
                          vae=vae, image=image_ref[0], latent=upsampled[0],
                          strength=float(refine_guide_strength), bypass=bypass[0])
    else:
        video_re = upsampled
    av_for_pass2 = g.node("LTXVConcatAVLatent",
                          video_latent=video_re[0], audio_latent=sep[1])
    return av_for_pass2, cropped


def _upsample_between_flf2v(g, av_pass1, cond, vae, upscaler, first_img, last_img,
                             guide_strength=1.0):
    """FLF2V between-pass: separate pass-1, upsample, re-apply AddGuide×2 at full
    strength, concat with pass-1 audio. Returns (av_latent_for_pass2, cond_for_pass2)
    where cond_for_pass2 is the AddGuide-chained cond on the upsampled latent."""
    sep = g.node("LTXVSeparateAVLatent", av_latent=av_pass1[0])
    upsampled = g.node("LTXVLatentUpsampler",
                       samples=sep[0], upscale_model=upscaler[0], vae=vae)
    r1 = g.node("LTXVAddGuide",
                positive=cond[0], negative=cond[1], vae=vae,
                latent=upsampled[0], image=first_img[0],
                frame_idx=0, strength=float(guide_strength))
    r2 = g.node("LTXVAddGuide",
                positive=r1[0], negative=r1[1], vae=vae,
                latent=r1[2], image=last_img[0],
                frame_idx=-1, strength=float(guide_strength))
    av_for_pass2 = g.node("LTXVConcatAVLatent",
                          video_latent=r2[2], audio_latent=sep[1])
    return av_for_pass2, r2


def _decode_and_save(g, av_final, vae, audio_vae, fps, filename_prefix,
                      strip_guides_cond=None):
    """Decode AV latent and save with embedded audio.

    When `strip_guides_cond` is provided (flf2v), run LTXVCropGuides on the
    final video latent so the injected first/last-frame guide samples are
    stripped before decode — otherwise those frames appear as raw input
    images in the output without blend."""
    sep = g.node("LTXVSeparateAVLatent", av_latent=av_final[0])
    video_src = sep
    video_idx = 0
    if strip_guides_cond is not None:
        cropped = g.node("LTXVCropGuides",
                         positive=strip_guides_cond[0],
                         negative=strip_guides_cond[1],
                         latent=sep[0])
        video_src = cropped
        video_idx = 2  # CropGuides outputs [2] = latent
    video = g.node("VAEDecodeTiled",
                   vae=vae, samples=video_src[video_idx],
                   tile_size=768, overlap=64,
                   temporal_size=4096, temporal_overlap=4)
    audio = g.node("LTXVAudioVAEDecode", samples=sep[1], audio_vae=audio_vae[0])
    created = g.node("CreateVideo", images=video[0], audio=audio[0], fps=float(fps))
    g.node("SaveVideo", video=created[0],
           filename_prefix=filename_prefix, format="auto", codec="auto")


# -------- shared builder for t2v / i2v / ia2v --------

def _build(prompt, *, fps, width, height, length, seed, filename_prefix,
           image_ref_builder=None, audio_ref_builder=None, negative=None,
           fast=False, camera_lora=None, camera_lora_strength=0.8,
           base_guide_strength=0.7, refine_guide_strength=0.3,
           ckpt=CKPT, text_encoder=TEXT_ENCODER):
    g = WorkflowGraph()
    checkpoint, clip, audio_vae, upscaler = _loaders(g, ckpt, text_encoder)
    model = _distilled_lora(g, checkpoint[0])
    model = _apply_extra_lora(g, model, camera_lora, camera_lora_strength)
    cond = _encode_prompts(g, clip[0], prompt, negative, fps)

    image_ref = image_ref_builder(g) if image_ref_builder else None
    video_latent = _base_video_latent(g, width, height, length,
                                      vae=checkpoint[2], image_ref=image_ref,
                                      strength=base_guide_strength)

    if audio_ref_builder:
        audio_latent = audio_ref_builder(g, audio_vae, width, height)
    else:
        audio_latent = _empty_audio_latent(g, length, fps, audio_vae)

    av_latent = g.node("LTXVConcatAVLatent",
                       video_latent=video_latent[0], audio_latent=audio_latent[0])

    base_seed = seed or _rand_seed()
    av_pass1 = _pass_one(g, model, cond, av_latent, seed=base_seed)

    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix)
    else:
        av_for_pass2, cropped_cond = _upsample_between(
            g, av_pass1, cond, vae=checkpoint[2], upscaler=upscaler,
            image_ref=image_ref,
            refine_guide_strength=refine_guide_strength)
        av_final = _pass_two(g, model, cropped_cond, av_for_pass2,
                             seed=base_seed + 1)
        _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix)
    return g.to_dict()


# -------- top-level variants --------

def ltx2_text_to_video(prompt, seconds=5, fps=24,
                        width=768, height=512,
                        filename_prefix="ltx2_t2v",
                        seed=None, negative=None, fast=False,
                        camera_lora=None, camera_lora_strength=0.8,
                        checkpoint_name=None, text_encoder=None,
                        **_):
    length = _round_length(seconds, fps)
    return _build(prompt, fps=fps, width=width, height=height, length=length,
                  seed=seed, filename_prefix=filename_prefix,
                  negative=negative, fast=fast,
                  camera_lora=camera_lora, camera_lora_strength=camera_lora_strength,
                  ckpt=checkpoint_name or CKPT,
                  text_encoder=text_encoder or TEXT_ENCODER)


def ltx2_image_to_video(image_filename, prompt, seconds=5, fps=24,
                         width=768, height=512,
                         filename_prefix="ltx2_i2v",
                         seed=None, negative=None, fast=False,
                         camera_lora=None, camera_lora_strength=0.8,
                         checkpoint_name=None, text_encoder=None,
                         **_):
    length = _round_length(seconds, fps)
    return _build(prompt, fps=fps, width=width, height=height, length=length,
                  seed=seed, filename_prefix=filename_prefix,
                  image_ref_builder=lambda g: _image_branch(g, image_filename, width, height),
                  negative=negative, fast=fast,
                  camera_lora=camera_lora, camera_lora_strength=camera_lora_strength,
                  ckpt=checkpoint_name or CKPT,
                  text_encoder=text_encoder or TEXT_ENCODER)


def ltx2_image_audio_to_video(image_filename, audio_filename, prompt,
                               seconds=5, fps=24,
                               width=768, height=512,
                               filename_prefix="ltx2_ia2v",
                               seed=None, negative=None, fast=False,
                               camera_lora=None, camera_lora_strength=0.8,
                               image_refs=None,
                               base_guide_strength=0.5,
                               refine_guide_strength=0.3,
                               checkpoint_name=None, text_encoder=None,
                               **_):
    """Image+Audio → Video.

    image_filename: primary reference (required; backward-compat).
    image_refs: optional list of additional references to concatenate into
      an IMAGE batch — use for multi-frame identity (e.g. [anchor.png,
      previous_scene-last.png]) for lipsync continuity + less face drift.
    base_guide_strength: strength of the LTXVImgToVideoInplace on the
      initial latent. 0.5 gives identity without locking frame 0.
    refine_guide_strength: strength of the re-apply after upsampling. 0.3
      is a soft refine; 0 disables it entirely. Historical default 1.0
      hard-locked the opening frames — do not use for action shots."""
    length = _round_length(seconds, fps)
    all_refs = [image_filename] + list(image_refs or [])
    return _build(prompt, fps=fps, width=width, height=height, length=length,
                  seed=seed, filename_prefix=filename_prefix,
                  image_ref_builder=lambda g: _image_branch_multi(g, all_refs, width, height),
                  audio_ref_builder=lambda g, avae, w, h:
                      _audio_from_file(g, audio_filename, seconds, w, h, avae),
                  negative=negative, fast=fast,
                  camera_lora=camera_lora, camera_lora_strength=camera_lora_strength,
                  base_guide_strength=base_guide_strength,
                  refine_guide_strength=refine_guide_strength,
                  ckpt=checkpoint_name or CKPT,
                  text_encoder=text_encoder or TEXT_ENCODER)


TRANSITION_LORA = "ltx2.3-transition.safetensors"
TRANSITION_TRIGGER = "zhuanchang"


def _flf2v_graph_core(prompt, *, fps, width, height, length, seed, filename_prefix,
                      first_img_builder, last_img_builder, audio_ref_builder=None,
                      negative=None, fast=False, use_av_mask=False,
                      first_guide_strength=0.7, last_guide_strength=0.7,
                      refine_guide_strength=1.0, extra_loras=None,
                      camera_lora=None, camera_lora_strength=0.8,
                      ckpt=CKPT, text_encoder=TEXT_ENCODER):
    g = WorkflowGraph()
    checkpoint, clip, audio_vae, upscaler = _loaders(g, ckpt, text_encoder)
    model = _distilled_lora(g, checkpoint[0])
    for lora_name, lora_strength in (extra_loras or []):
        model = g.node("LoraLoaderModelOnly", model=model[0],
                       lora_name=lora_name, strength_model=float(lora_strength))
    model = _apply_extra_lora(g, model, camera_lora, camera_lora_strength)
    cond = _encode_prompts(g, clip[0], prompt, negative, fps)
    first_img = first_img_builder(g)
    last_img = last_img_builder(g)
    empty_video = g.node("EmptyLTXVLatentVideo",
                         width=int(width), height=int(height),
                         length=int(length), batch_size=1)
    g1 = g.node("LTXVAddGuide",
                positive=cond[0], negative=cond[1], vae=checkpoint[2],
                latent=empty_video[0], image=first_img[0],
                frame_idx=0, strength=float(first_guide_strength))
    g2 = g.node("LTXVAddGuide",
                positive=g1[0], negative=g1[1], vae=checkpoint[2],
                latent=g1[2], image=last_img[0],
                frame_idx=-1, strength=float(last_guide_strength))
    if audio_ref_builder:
        audio_latent = audio_ref_builder(g, audio_vae, width, height,
                                         length=length, fps=fps)
    else:
        audio_latent = _empty_audio_latent(g, length, fps, audio_vae)
    if use_av_mask:
        effective_seconds = float(length) / float(fps)
        aligned = g.node("LTXVAudioVideoMask",
                         video_latent=g2[2], audio_latent=audio_latent[0],
                         video_fps=float(fps),
                         video_start_time=0.0, video_end_time=effective_seconds,
                         audio_start_time=0.0, audio_end_time=effective_seconds,
                         max_length="pad")
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=aligned[0], audio_latent=aligned[1])
    else:
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=g2[2], audio_latent=audio_latent[0])
    base_seed = seed or _rand_seed()
    av_pass1 = _pass_one(g, model, g2, av_latent, seed=base_seed)
    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=g2)
    else:
        av_for_pass2, pass2_cond = _upsample_between_flf2v(
            g, av_pass1, cond, vae=checkpoint[2], upscaler=upscaler,
            first_img=first_img, last_img=last_img,
            guide_strength=float(refine_guide_strength))
        av_final = _pass_two(g, model, pass2_cond, av_for_pass2,
                             seed=base_seed + 1)
        _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=pass2_cond)
    return g.to_dict()


def ltx2_first_last_frame_to_video(first_frame_filename, last_frame_filename, prompt,
                                    seconds=5, fps=24, width=768, height=512,
                                    filename_prefix="ltx2_flf2v",
                                    seed=None, negative=None, fast=False,
                                    guide_strength=0.7,
                                    camera_lora=None, camera_lora_strength=0.8,
                                    checkpoint_name=None, text_encoder=None,
                                    **_):
    """First-last-frame to video. Pattern adapted from Comfy-Org flf2v template
    into the same two-pass structure as the other variants (coarse → upsample +
    re-inject AddGuides at strength=1.0 → refine). `fast=True` yields the
    original single-pass behaviour at lower resolution."""
    return _flf2v_graph_core(
        prompt, fps=fps, width=width, height=height,
        length=_round_length(seconds, fps),
        seed=seed, filename_prefix=filename_prefix,
        first_img_builder=lambda g: _flf2v_preprocess_frame(g, first_frame_filename, width, height),
        last_img_builder=lambda g: _flf2v_preprocess_frame(g, last_frame_filename, width, height),
        negative=negative, fast=fast,
        first_guide_strength=guide_strength, last_guide_strength=guide_strength,
        camera_lora=camera_lora, camera_lora_strength=camera_lora_strength,
        ckpt=checkpoint_name or CKPT, text_encoder=text_encoder or TEXT_ENCODER)


def _video_range_frames(g, video_filename, start_index, num_frames, width, height,
                         _pre_loaded_images=None):
    """Load a video and return `num_frames` frames starting at
    `start_index` as an LTXVPreprocess'd IMAGE batch. Only supports
    start_index >= 0 (GetImageRangeFromBatch min=-1 limit); tail slices
    use `_video_tail_frames` which wraps this via reverse→head→reverse.

    `_pre_loaded_images` lets the tail helper pass in an already
    reverse-batched IMAGE ref to slice from, avoiding a second LoadVideo."""
    if _pre_loaded_images is None:
        vid = g.node("LoadVideo", file=video_filename)
        comp = g.node("GetVideoComponents", video=vid[0])
        images = comp[0]
    else:
        images = _pre_loaded_images
    sliced = g.node("GetImageRangeFromBatch",
                    images=images, start_index=int(start_index),
                    num_frames=int(num_frames))
    resized = g.node("ResizeImageMaskNode", **{
        "input": sliced[0],
        "resize_type": "scale dimensions",
        "resize_type.width": int(width),
        "resize_type.height": int(height),
        "resize_type.crop": "center",
        "scale_method": "nearest-exact",
    })
    return g.node("LTXVPreprocess", image=resized[0], img_compression=25)


def _video_tail_frames(g, video_filename, num_frames, width, height):
    """Return the LAST N frames of a video as an LTXVPreprocess'd IMAGE
    batch in original order. Uses reverse→head→reverse because
    GetImageRangeFromBatch only supports start_index >= -1."""
    vid = g.node("LoadVideo", file=video_filename)
    comp = g.node("GetVideoComponents", video=vid[0])
    rev1 = g.node("ReverseImageBatch", images=comp[0])
    head_pp = _video_range_frames(g, video_filename, 0, num_frames,
                                   width, height,
                                   _pre_loaded_images=rev1[0])
    return g.node("ReverseImageBatch", images=head_pp[0])


def _video_head_frames(g, video_filename, num_frames, width, height):
    """Load a video and return the FIRST N frames as an LTXVPreprocess'd
    IMAGE batch."""
    return _video_range_frames(g, video_filename, 0, num_frames, width, height)


def ltx2_transition(first_frame_filename, last_frame_filename, prompt,
                    seconds=4, fps=25, width=720, height=1280,
                    filename_prefix="ltx2_transition",
                    seed=None, negative=None,
                    first_guide_strength=1.0, last_guide_strength=1.0,
                    audio_filename=None,
                    prev_video_filename=None, next_video_filename=None,
                    multiframe_guide=9,
                    checkpoint_name=None, text_encoder=None,
                    **_):
    """Transition clip scene N → scene N+1 via the ltx2.3-transition LoRA.

    Two-pass refine chain, AddGuide at both ends at strength 1.0 by
    default. With prev_video_filename / next_video_filename, the guides
    become MULTI-FRAME: the last `multiframe_guide` frames of the prev
    video seed frame 0 (LTX has N frames of motion before the morph),
    and the first `multiframe_guide` frames of the next video seed
    frame -1 (LTX has N frames to prepare scene-B's motion into). Both
    use built-in LoadVideo + GetVideoComponents + GetImageRangeFromBatch
    — no VHS dependency.

    Audio is trimmed to `length/fps` (= _round_length output over fps),
    matching the actually-rendered video duration so LTX doesn't
    truncate the audio tail. The boundary-spanning song slice that
    starts the next scene's vocal should fit entirely inside."""
    if TRANSITION_TRIGGER not in prompt.lower():
        prompt = f"{prompt.rstrip('. ')}. {TRANSITION_TRIGGER}"

    def _first_builder(g):
        if prev_video_filename:
            return _video_tail_frames(g, prev_video_filename, multiframe_guide, width, height)
        return _flf2v_preprocess_frame(g, first_frame_filename, width, height)

    def _last_builder(g):
        if next_video_filename:
            return _video_head_frames(g, next_video_filename, multiframe_guide, width, height)
        return _flf2v_preprocess_frame(g, last_frame_filename, width, height)

    audio_ref_builder = None
    if audio_filename:
        audio_ref_builder = lambda g, avae, w, h, length, fps: _audio_from_file(
            g, audio_filename, seconds, w, h, avae, length=length, fps=fps)

    return _flf2v_graph_core(
        prompt, fps=fps, width=width, height=height,
        length=_round_length(seconds, fps),
        seed=seed, filename_prefix=filename_prefix,
        first_img_builder=_first_builder, last_img_builder=_last_builder,
        audio_ref_builder=audio_ref_builder,
        negative=negative, fast=False, use_av_mask=True,
        first_guide_strength=first_guide_strength,
        last_guide_strength=last_guide_strength,
        refine_guide_strength=last_guide_strength,
        extra_loras=[(TRANSITION_LORA, 1.0)],
        ckpt=checkpoint_name or CKPT,
        text_encoder=text_encoder or TEXT_ENCODER)


# -------- unchanged utility --------

def extract_last_frame(video_server_path, filename_prefix="last_frame"):
    """Extract the last frame from a ComfyUI output video.
    video_server_path: absolute path on the ComfyUI server."""
    g = WorkflowGraph()
    frames = g.node("VHS_LoadVideoPath", video=video_server_path, force_rate=0,
                    custom_width=0, custom_height=0, frame_load_cap=0,
                    skip_first_frames=0, select_every_nth=1)
    last = g.node("GetImageRangeFromBatch", images=frames[0], start_index=-1, num_frames=1)
    g.node("SaveImage", images=last[0], filename_prefix=filename_prefix)
    return g.to_dict()
