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
import os
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
    uniform across variants so graph shape is predictable.

    Uses `CheckpointLoaderKJ` (from kijai/ComfyUI-KJNodes) when sage attention
    is enabled — this saves ~30-50% VRAM and lets longer videos fit on the
    3090. Falls back to the stock `CheckpointLoaderSimple` when disabled.

    Env override `LTX_SAGE_ATTENTION`:
      - "auto" or unset → kj loader with sageattn_qk_int8_pv_fp16_cuda
        (safe choice on Ampere/Ada; "auto" lets KJ pick)
      - "off"           → stock loader, no sage
      - any other value → used verbatim as the sage_attention enum
    """
    mode = os.environ.get("LTX_SAGE_ATTENTION", "auto").strip()
    if mode.lower() in ("off", "false", "0", "disabled"):
        checkpoint = g.node("CheckpointLoaderSimple", ckpt_name=ckpt)
    else:
        sage = "auto" if mode.lower() == "auto" else mode
        checkpoint = g.node("CheckpointLoaderKJ",
                            ckpt_name=ckpt,
                            weight_dtype="default",
                            compute_dtype="default",
                            patch_cublaslinear=False,
                            sage_attention=sage,
                            enable_fp16_accumulation=True)
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
    elif not lora_name.endswith(".safetensors"):
        # Looks like a shortname but not in the table. Fail fast here with
        # the valid set, rather than letting comfy reject the workflow with
        # a generic HTTP 400.
        raise ValueError(
            f"camera_lora shortname '{lora_name}' is not recognised. "
            f"Valid shortnames: {sorted(CAMERA_LORAS)}. "
            f"Pass a full .safetensors filename for anything else.")
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
    """FLF2V / multi-guide frame preprocess. Mirrors `_image_branch` for the
    input path — without the ResizeImagesByLongerEdge(1536) cap, multi-guide
    renders emit at 2x the target resolution (e.g. 1792x3328 for an 896x1664
    request), which mismatches i2v/ia2v scene clips and breaks concat.
    Nearest-exact scale + img_compression=25 per the FLF2V ref template."""
    loaded = g.node("LoadImage", image=image_filename)
    resized = g.node("ResizeImageMaskNode", **{
        "input": loaded[0],
        "resize_type": "scale dimensions",
        "resize_type.width": int(width),
        "resize_type.height": int(height),
        "resize_type.crop": "center",
        "scale_method": "nearest-exact",
    })
    longer = g.node("ResizeImagesByLongerEdge", images=resized[0], longer_edge=1536)
    return g.node("LTXVPreprocess", image=longer[0], img_compression=25)


def _audio_from_file(g, audio_filename, seconds, width, height, audio_vae,
                     length=None, fps=None, debug_save_prefix=None,
                     wrap_noise_mask=True):
    """Load and trim audio to the same number of seconds as the rendered
    video. `length`+`fps` are optional; when provided, trim to length/fps
    instead of the raw `seconds` input — that aligns audio to LTX's
    actually-rendered frame count (which `_round_length` snapped to
    `8k+1`). Otherwise audio can overhang the video by ~0.2s and LTX's
    audio-VAE truncation chops the TAIL — exactly where a transition's
    target vocal starts.

    debug_save_prefix: when set, also saves the trimmed audio (before VAE
    encode) AND the VAE-roundtripped audio (encoded then decoded) — useful
    to debug whether what LTX "hears" matches what we intended to pass,
    and what distortion the audio VAE introduces."""
    loaded = g.node("LoadAudio", audio=audio_filename)
    if length is not None and fps:
        effective = float(length) / float(fps)
    else:
        effective = float(seconds)
    trimmed = g.node("TrimAudioDuration", audio=loaded[0],
                     duration=effective, start_index=0)
    encoded = g.node("LTXVAudioVAEEncode", audio=trimmed[0], audio_vae=audio_vae[0])
    if debug_save_prefix:
        # What LTX's audio head literally sees (trimmed input):
        g.node("SaveAudio", audio=trimmed[0],
               filename_prefix=f"{debug_save_prefix}_trimmed_in")
        # Roundtrip — encoded→decoded — shows VAE degradation:
        decoded = g.node("LTXVAudioVAEDecode", samples=encoded[0],
                         audio_vae=audio_vae[0])
        g.node("SaveAudio", audio=decoded[0],
               filename_prefix=f"{debug_save_prefix}_vae_roundtrip")
    if not wrap_noise_mask:
        return encoded
    mask = g.node("SolidMask", value=0, width=int(width), height=int(height))
    return g.node("SetLatentNoiseMask", samples=encoded[0], mask=mask[0])


def _empty_audio_latent(g, length, fps, audio_vae):
    return g.node("LTXVEmptyLatentAudio",
                  frames_number=int(length), frame_rate=float(fps),
                  batch_size=1, audio_vae=audio_vae[0])


def _base_video_latent(g, width, height, length, vae, image_ref, strength=0.7,
                        condition_only=False):
    """Empty video latent, optionally conditioned on a single reference image
    (or a multi-frame IMAGE batch). Two anchor-conditioning modes:

      condition_only=False  (default for plain ia2v):
          LTXVImgToVideoInplace bakes the anchor INTO the initial latent state.
          The sampler then refines from this partially-filled latent. Strong
          first-frame fidelity but it COMPETES SPATIALLY with any downstream
          LTXAddVideoICLoRAGuide conditioning — both want to drive the latent
          and the IC-LoRA tends to win the left half / lose the right half,
          producing a visible left-biased output. Don't use this with IC-LoRA.

      condition_only=True   (use when LTXAddVideoICLoRAGuide is active):
          LTXVImgToVideoConditionOnly attaches the anchor as a side-channel
          condition WITHOUT modifying the latent. The latent stays empty
          (full noise), and the conditioning chain (anchor + IC-LoRA video)
          jointly drives generation through positive/negative pairs. This
          matches the official Lightricks LTX-2.3 IC-LoRA Union-Control
          workflow shape and produces full-frame coverage instead of the
          left-half output we got with Inplace + IC-LoRA combined.

    Lower strength means the refs act as a soft identity prior instead of a
    locked first frame — use ~0.4-0.6 for lipsync where the character should
    move freely.
    """
    empty = g.node("EmptyLTXVLatentVideo",
                   width=int(width), height=int(height),
                   length=int(length), batch_size=1)
    if image_ref is None:
        return empty
    if condition_only:
        return g.node("LTXVImgToVideoConditionOnly",
                       vae=vae, image=image_ref[0], latent=empty[0],
                       strength=float(strength))
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
    # Feed the CROPPED latent (cropped[2]) into the upsampler — not sep[0].
    # When IC-LoRA / id_branch added conditioning frames in pass-1, those
    # frames live in the latent and must be removed before upsampling,
    # otherwise pass-2 refines 2x as many latent frames as the user
    # requested → 2x-duration glitched output. The official lipdub
    # 2-stage workflow wires its upsampler from CropGuides.latent for
    # this reason. The basic ia2v 2-stage workflow (no AddGuide) can use
    # sep[0] because it has no cond frames to strip.
    upsampled = g.node("LTXVLatentUpsampler",
                       samples=cropped[2], upscale_model=upscaler[0], vae=vae)
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


def _upsample_between_multi_guide(g, av_pass1, cond, vae, upscaler,
                                   guide_imgs, guide_frame_indices,
                                   pass1_guide_chain_cond=None,
                                   guide_strength=1.0):
    """Multi-guide between-pass: separate pass-1, upsample, **strip the
    pass-1 guide frames** via LTXVCropGuides, then re-apply N LTXVAddGuide
    nodes at the same frame positions + full strength on the clean latent.

    CropGuides between passes is essential: every AddGuide inserts an extra
    image-frame into the latent. Without cropping between passes, pass 1's
    N guide-frames carry into pass 2, then pass 2 adds N more, and only the
    most recent chain gets stripped at decode — leaving N unattributed
    tail frames in the final mp4 (silent past the audio slice).

    Returns (av_latent_for_pass2, cond_for_pass2) where cond_for_pass2 is
    the chain of all N AddGuides re-applied on the cleaned upsampled
    latent."""
    sep = g.node("LTXVSeparateAVLatent", av_latent=av_pass1[0])
    upsampled = g.node("LTXVLatentUpsampler",
                       samples=sep[0], upscale_model=upscaler[0], vae=vae)
    # Strip pass-1 guide tokens from the upsampled latent. Requires the
    # pass-1 guide chain's conditioning — falls back to plain upsampled
    # (legacy behaviour) if not provided.
    if pass1_guide_chain_cond is not None:
        cleaned = g.node("LTXVCropGuides",
                         positive=pass1_guide_chain_cond[0],
                         negative=pass1_guide_chain_cond[1],
                         latent=upsampled[0])
        latent_ref = cleaned
        latent_idx = 2  # CropGuides output [2] = latent
    else:
        latent_ref = upsampled
        latent_idx = 0
    prev = cond
    for img, idx in zip(guide_imgs, guide_frame_indices):
        ag = g.node("LTXVAddGuide",
                    positive=prev[0], negative=prev[1], vae=vae,
                    latent=latent_ref[latent_idx], image=img[0],
                    frame_idx=int(idx), strength=float(guide_strength))
        prev = ag
        latent_ref = ag
        latent_idx = 2  # AddGuide output [2] = latent
    av_for_pass2 = g.node("LTXVConcatAVLatent",
                          video_latent=prev[2], audio_latent=sep[1])
    return av_for_pass2, prev


def _upsample_between_flf2v(g, av_pass1, cond, vae, upscaler, first_img, last_img,
                             pass1_guide_chain_cond=None,
                             guide_strength=1.0):
    """FLF2V between-pass: separate pass-1, upsample, **strip pass-1 guide
    frames** via LTXVCropGuides (if pass1_guide_chain_cond provided), then
    re-apply AddGuide×2 at full strength on the clean latent, concat with
    pass-1 audio. Returns (av_latent_for_pass2, cond_for_pass2).

    See _upsample_between_multi_guide for why the CropGuides step matters
    between passes — same bug, same fix."""
    sep = g.node("LTXVSeparateAVLatent", av_latent=av_pass1[0])
    upsampled = g.node("LTXVLatentUpsampler",
                       samples=sep[0], upscale_model=upscaler[0], vae=vae)
    if pass1_guide_chain_cond is not None:
        cleaned = g.node("LTXVCropGuides",
                         positive=pass1_guide_chain_cond[0],
                         negative=pass1_guide_chain_cond[1],
                         latent=upsampled[0])
        base_latent = cleaned
        base_idx = 2  # CropGuides output [2] = latent
    else:
        base_latent = upsampled
        base_idx = 0
    r1 = g.node("LTXVAddGuide",
                positive=cond[0], negative=cond[1], vae=vae,
                latent=base_latent[base_idx], image=first_img[0],
                frame_idx=0, strength=float(guide_strength))
    r2 = g.node("LTXVAddGuide",
                positive=r1[0], negative=r1[1], vae=vae,
                latent=r1[2], image=last_img[0],
                frame_idx=-1, strength=float(guide_strength))
    av_for_pass2 = g.node("LTXVConcatAVLatent",
                          video_latent=r2[2], audio_latent=sep[1])
    return av_for_pass2, r2


def _decode_and_save(g, av_final, vae, audio_vae, fps, filename_prefix,
                      strip_guides_cond=None,
                      source_audio_filename=None, source_audio_seconds=None):
    """Decode AV latent and save with embedded audio.

    When `strip_guides_cond` is provided (flf2v), run LTXVCropGuides on the
    final video latent so the injected first/last-frame guide samples are
    stripped before decode — otherwise those frames appear as raw input
    images in the output without blend.

    When `source_audio_filename` is provided, mux the trimmed source audio
    into the saved mp4 instead of LTX's audio-head VAE output. LTX's audio
    head is tuned for lipsync/SFX, not music reproduction — for transitions
    where the song should continue, the generated audio comes out silent or
    degraded. Using the source file preserves the song."""
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
    if source_audio_filename:
        loaded = g.node("LoadAudio", audio=source_audio_filename)
        audio_out = g.node("TrimAudioDuration", audio=loaded[0],
                           duration=float(source_audio_seconds), start_index=0)
    else:
        audio_out = g.node("LTXVAudioVAEDecode",
                           samples=sep[1], audio_vae=audio_vae[0])
    created = g.node("CreateVideo", images=video[0], audio=audio_out[0], fps=float(fps))
    g.node("SaveVideo", video=created[0],
           filename_prefix=filename_prefix, format="auto", codec="auto")


# -------- shared builder for t2v / i2v / ia2v --------

def _build(prompt, *, fps, width, height, length, seed, filename_prefix,
           image_ref_builder=None, audio_ref_builder=None, negative=None,
           fast=False, camera_lora=None, camera_lora_strength=0.8,
           base_guide_strength=0.7, refine_guide_strength=0.3,
           identity_anchor_image=None, identity_strength=0.3,
           source_audio_filename=None,
           ic_loras=None,
           ic_lora_reference_filename=None,
           ic_lora_reference_strength=1.0,
           ic_lora_reference_size=None,
           ckpt=CKPT, text_encoder=TEXT_ENCODER):
    """
    ic_loras: optional list of (lora_name, strength) tuples. Each is loaded
        via Lightricks' LTXICLoRALoaderModelOnly node which is required for
        IC-LoRA weights (e.g. ltx-2.3-22b-ic-lora-hdr-0.9.safetensors); the
        regular LoraLoaderModelOnly does not extract the latent_downscale_factor
        these LoRAs need. Stack multiple in order — typically one main IC-LoRA
        + one scene-emb companion.
    ic_lora_reference_filename: REQUIRED when ic_loras is set. The IC-LoRA
        only acts when paired with a reference image fed through the
        ImagePrepForICLora + LTXAddVideoICLoRAGuide chain. The IC-LoRA
        weights without the conditioning image have no visible effect.
    ic_lora_reference_strength: strength on the LTXAddVideoICLoRAGuide
        node — how strongly the IC-LoRA reference biases the latent.
    ic_lora_reference_size: ImagePrepForICLora target size (square).
    """
    g = WorkflowGraph()
    checkpoint, clip, audio_vae, upscaler = _loaders(g, ckpt, text_encoder)
    model = _distilled_lora(g, checkpoint[0])
    model = _apply_extra_lora(g, model, camera_lora, camera_lora_strength)
    # IC-LoRAs (e.g. HDR, Union-Control) — applied AFTER distilled+camera so
    # they sit closest to the sampler. Each IC-LoRA loader returns
    # (model, latent_downscale_factor); we keep both — the model wires
    # forward, and the LATEST latent_downscale_factor is required by
    # LTXAddVideoICLoRAGuide downstream.
    ic_lora_active = bool(ic_loras and ic_lora_reference_filename)
    ic_loaded = None    # last LTXICLoRALoaderModelOnly node ref
    if ic_loras:
        for lora_name, lora_strength in ic_loras:
            ic_load = g.node("LTXICLoRALoaderModelOnly",
                              model=model[0],
                              lora_name=lora_name,
                              strength_model=float(lora_strength))
            model = ic_load
            ic_loaded = ic_load
    cond = _encode_prompts(g, clip[0], prompt, negative, fps)

    image_ref = image_ref_builder(g) if image_ref_builder else None
    # When IC-LoRA is active, follow the official Lightricks workflow shape
    # and use ConditionOnly anchor mode so the anchor doesn't bake into the
    # latent state and fight the IC-LoRA conditioning for spatial coverage.
    video_latent = _base_video_latent(g, width, height, length,
                                      vae=checkpoint[2], image_ref=image_ref,
                                      strength=base_guide_strength,
                                      condition_only=ic_lora_active)

    # IC-LoRA reference image conditioning. Loads the reference, preps it
    # to a square tile via ImagePrepForICLora (the node centers + pads, no
    # crop), then injects it via LTXAddVideoICLoRAGuide which adds an
    # IC-LoRA-specific conditioning branch on top of the existing
    # positive/negative tuple. Frame_idx=0 makes it a first-frame style
    # bias (HDR envelope, lighting palette); use frame_idx=-1 or middle
    # for late-shot biases. Returns (positive, negative, latent).
    if ic_lora_active:
        # Reference can be a single image (LoadImage) or a video frame batch
        # (LoadVideo → GetVideoComponents). The video path is what makes
        # depth-control / canny-control / motion-track-control IC-LoRAs
        # actually conditioning-per-frame instead of a static bias —
        # use ic_lora_reference_filename ending in .mp4/.mov to trigger.
        is_video_ref = bool(ic_lora_reference_filename and
                            str(ic_lora_reference_filename).lower().rsplit(".", 1)[-1]
                            in ("mp4", "mov", "webm", "mkv"))
        if is_video_ref:
            # Match the official Lightricks LTX-2.3 IC-LoRA Union-Control
            # workflow: LoadVideo → GetVideoComponents[image] →
            # ResizeImageMaskNode 'scale to multiple' (32) → LTXAddVideoICLoRAGuide.image.
            # Skip ImagePrepForICLora entirely — that node distorts non-
            # square inputs and was the root cause of the left-bias bug
            # we chased through three iterations of size-tweaks.
            # Caller is expected to render the conditioning video at
            # exactly the output W×H dims (which are already ÷32-aligned),
            # so the resize step is effectively a no-op but kept for
            # safety against off-by-padding inputs.
            ic_load = g.node("LoadVideo", file=ic_lora_reference_filename)
            ic_components = g.node("GetVideoComponents", video=ic_load[0])
            ic_ref_prep = g.node("ResizeImageMaskNode", **{
                "input": ic_components[0],
                "resize_type": "scale to multiple",
                "resize_type.multiple": 32,
                "scale_method": "lanczos",
            })
        else:
            # Single-image reference (HDR-style static bias). Keep the
            # ImagePrepForICLora path here — square-padded 1024×1024 input
            # is what HDR was trained on, and the bias issue only manifests
            # for video-batch conditioning under Union-Control.
            ic_load = g.node("LoadImage", image=ic_lora_reference_filename)
            if ic_lora_reference_size is None:
                prep_w, prep_h = int(width), int(height)
            else:
                prep_w = prep_h = int(ic_lora_reference_size)
            ic_ref_prep = g.node("ImagePrepForICLora",
                                  reference_image=ic_load[0],
                                  output_width=prep_w,
                                  output_height=prep_h,
                                  border_width=0)
        # latent_downscale_factor: source it from LTXICLoRALoaderModelOnly's
        # slot 1 — the LoRA's metadata-embedded factor (2.0 for Union-Control,
        # 1.0 for HDR). The official workflow's wired pattern. Earlier we
        # tried hardcoding 1.0 to fix the left-bias bug — that didn't help;
        # the actual fix was switching ImagePrepForICLora → ResizeImageMaskNode
        # for video references (above).
        ic_guide = g.node("LTXAddVideoICLoRAGuide",
                           positive=cond[0], negative=cond[1],
                           vae=checkpoint[2], latent=video_latent[0],
                           image=ic_ref_prep[0],
                           frame_idx=0,
                           strength=float(ic_lora_reference_strength),
                           latent_downscale_factor=ic_loaded[1],
                           crop="disabled",
                           use_tiled_encode=False,
                           tile_size=256,
                           tile_overlap=64)
        # Re-bind cond + video_latent so downstream uses the IC-LoRA-conditioned
        # variants. ic_guide outputs are (positive, negative, latent).
        cond = (ic_guide[0], ic_guide[1])
        video_latent = (ic_guide[2],)

    # Optional identity anchor — injects the character reference at a
    # MIDDLE frame (not frame_idx=-1, which would make this a flf2v-style
    # last-frame TARGET). Middle placement at low strength acts as a
    # soft identity bias for the model without steering the ending. For
    # ia2v the middle of the latent is a natural "the character still
    # looks like this" signal; the model can drift toward it or away
    # on either side without being forced to converge.
    id_branch = None
    cond_for_sampling = cond
    latent_for_sampling = video_latent
    latent_idx = 0
    if identity_anchor_image:
        id_branch = _flf2v_preprocess_frame(
            g, identity_anchor_image, width, height)
        # Middle frame, snapped down to the nearest multiple of 8 (LTX
        # temporal-compression requirement for length >= 9).
        id_frame = max(8, ((int(length) // 2) // 8) * 8)
        id_guide = g.node("LTXVAddGuide",
                           positive=cond[0], negative=cond[1], vae=checkpoint[2],
                           latent=video_latent[0], image=id_branch[0],
                           frame_idx=id_frame, strength=float(identity_strength))
        cond_for_sampling = id_guide
        latent_for_sampling = id_guide
        latent_idx = 2  # AddGuide output [2] = latent

    if audio_ref_builder:
        audio_latent = audio_ref_builder(g, audio_vae, width, height)
    else:
        audio_latent = _empty_audio_latent(g, length, fps, audio_vae)

    av_latent = g.node("LTXVConcatAVLatent",
                       video_latent=latent_for_sampling[latent_idx],
                       audio_latent=audio_latent[0])

    base_seed = seed or _rand_seed()
    av_pass1 = _pass_one(g, model, cond_for_sampling, av_latent, seed=base_seed)
    source_audio_seconds = float(length) / float(fps) if source_audio_filename else None

    # `strip_guides_cond` triggers an LTXVCropGuides pass before decode. We
    # need it whenever any LTXVAddGuide / LTXAddVideoICLoRAGuide added
    # conditioning frames to the latent — those frames stay in the output
    # otherwise and the rendered video runs ~2× the requested seconds with
    # latent-noise tail (the "the video is 13s long with noise" bug).
    # Trigger on either id_branch (LTXVAddGuide) OR ic_lora_active
    # (LTXAddVideoICLoRAGuide).
    needs_crop = id_branch is not None or ic_lora_active
    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=(cond_for_sampling if needs_crop else None),
                         source_audio_filename=source_audio_filename,
                         source_audio_seconds=source_audio_seconds)
    else:
        # 2-pass refine path. _upsample_between strips IC-LoRA / id_branch
        # cond frames from the latent before upsampling (see comment in
        # that helper) so pass-2 refines only the requested-length latent.
        av_for_pass2, cropped_cond = _upsample_between(
            g, av_pass1, cond, vae=checkpoint[2], upscaler=upscaler,
            image_ref=image_ref,
            refine_guide_strength=refine_guide_strength)
        final_cond = cropped_cond
        if id_branch is not None:
            sep2 = g.node("LTXVSeparateAVLatent", av_latent=av_for_pass2[0])
            id_frame = max(8, ((int(length) // 2) // 8) * 8)
            id_guide2 = g.node("LTXVAddGuide",
                                positive=cropped_cond[0], negative=cropped_cond[1],
                                vae=checkpoint[2], latent=sep2[0],
                                image=id_branch[0], frame_idx=id_frame,
                                strength=float(identity_strength))
            av_for_pass2 = g.node("LTXVConcatAVLatent",
                                  video_latent=id_guide2[2],
                                  audio_latent=sep2[1])
            final_cond = id_guide2
        av_final = _pass_two(g, model, final_cond, av_for_pass2,
                             seed=base_seed + 1)
        _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=(final_cond if needs_crop else None),
                         source_audio_filename=source_audio_filename,
                         source_audio_seconds=source_audio_seconds)
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
                               identity_anchor_image=None,
                               identity_strength=0.3,
                               ic_loras=None,
                               ic_lora_reference_filename=None,
                               ic_lora_reference_strength=1.0,
                               ic_lora_reference_size=None,
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
                  audio_ref_builder=lambda g, avae, w, h, length=length, fps=fps:
                      _audio_from_file(g, audio_filename, seconds, w, h, avae,
                                       length=length, fps=fps),
                  negative=negative, fast=fast,
                  camera_lora=camera_lora, camera_lora_strength=camera_lora_strength,
                  base_guide_strength=base_guide_strength,
                  refine_guide_strength=refine_guide_strength,
                  identity_anchor_image=identity_anchor_image,
                  identity_strength=identity_strength,
                  source_audio_filename=audio_filename,
                  ic_loras=ic_loras,
                  ic_lora_reference_filename=ic_lora_reference_filename,
                  ic_lora_reference_strength=ic_lora_reference_strength,
                  ic_lora_reference_size=ic_lora_reference_size,
                  ckpt=checkpoint_name or CKPT,
                  text_encoder=text_encoder or TEXT_ENCODER)


TRANSITION_LORA = "ltx2.3-transition.safetensors"
TRANSITION_TRIGGER = "zhuanchang"


def _flf2v_graph_core(prompt, *, fps, width, height, length, seed, filename_prefix,
                      first_img_builder, last_img_builder, audio_ref_builder=None,
                      source_audio_filename=None,
                      first_guide_num_frames=1, last_guide_num_frames=1,
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
    # LTXVAddGuide places image-frames FORWARD from frame_idx — a multi-frame
    # image at frame_idx=-1 is ill-defined (there's only 1 frame position at
    # -1, the image would be truncated or wrap weirdly). For multi-frame end
    # guides we compute an explicit positive frame_idx = length - num_frames,
    # snapped down to the nearest multiple of 8 (LTX requirement for 9+ frame
    # videos). Single-frame guides keep the clean frame_idx=-1 convention.
    if last_guide_num_frames > 1:
        last_idx = (int(length) - int(last_guide_num_frames)) // 8 * 8
    else:
        last_idx = -1
    g1 = g.node("LTXVAddGuide",
                positive=cond[0], negative=cond[1], vae=checkpoint[2],
                latent=empty_video[0], image=first_img[0],
                frame_idx=0, strength=float(first_guide_strength))
    g2 = g.node("LTXVAddGuide",
                positive=g1[0], negative=g1[1], vae=checkpoint[2],
                latent=g1[2], image=last_img[0],
                frame_idx=last_idx, strength=float(last_guide_strength))
    if audio_ref_builder:
        audio_latent = audio_ref_builder(g, audio_vae, width, height,
                                         length=length, fps=fps)
    else:
        audio_latent = _empty_audio_latent(g, length, fps, audio_vae)
    if use_av_mask:
        # Noise-mask semantics per ComfyUI convention: the range between
        # start_time and end_time is the portion to be REGENERATED. We want
        # the video fully regenerated (full span masked), and the audio
        # preserved as lipsync reference (zero-length mask at the tail =
        # nothing masked). This matches the ltxv reference extend-video
        # workflow where audio_start == audio_end == latent_end.
        effective_seconds = float(length) / float(fps)
        aligned = g.node("LTXVAudioVideoMask",
                         video_latent=g2[2], audio_latent=audio_latent[0],
                         video_fps=float(fps),
                         video_start_time=0.0, video_end_time=effective_seconds,
                         audio_start_time=effective_seconds,
                         audio_end_time=effective_seconds,
                         max_length="pad")
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=aligned[0], audio_latent=aligned[1])
    else:
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=g2[2], audio_latent=audio_latent[0])
    base_seed = seed or _rand_seed()
    av_pass1 = _pass_one(g, model, g2, av_latent, seed=base_seed)
    source_audio_seconds = float(length) / float(fps) if source_audio_filename else None
    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=g2,
                         source_audio_filename=source_audio_filename,
                         source_audio_seconds=source_audio_seconds)
    else:
        av_for_pass2, pass2_cond = _upsample_between_flf2v(
            g, av_pass1, cond, vae=checkpoint[2], upscaler=upscaler,
            first_img=first_img, last_img=last_img,
            pass1_guide_chain_cond=g2,
            guide_strength=float(refine_guide_strength))
        av_final = _pass_two(g, model, pass2_cond, av_for_pass2,
                             seed=base_seed + 1)
        _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=pass2_cond,
                         source_audio_filename=source_audio_filename,
                         source_audio_seconds=source_audio_seconds)
    return g.to_dict()


def ltx2_first_last_frame_to_video(first_frame_filename, last_frame_filename, prompt,
                                    seconds=5, fps=24, width=768, height=512,
                                    filename_prefix="ltx2_flf2v",
                                    seed=None, negative=None, fast=False,
                                    first_guide_strength=None,
                                    last_guide_strength=None,
                                    guide_strength=0.7,
                                    audio_filename=None,
                                    use_transition_lora=False,
                                    transition_lora_strength=1.0,
                                    camera_lora=None, camera_lora_strength=0.8,
                                    checkpoint_name=None, text_encoder=None,
                                    **_):
    """First-last-frame (optionally + audio) to video — effectively flfa2v
    when `audio_filename` is provided. Pattern adapted from the Comfy-Org
    flf2v template into the same two-pass structure as the other variants
    (coarse → upsample + re-inject AddGuides at strength=1.0 → refine).
    `fast=True` yields the original single-pass behaviour at lower
    resolution.

    Guide strengths: `guide_strength` sets BOTH first and last. Override
    either independently with `first_guide_strength` / `last_guide_strength`.

    Transition LoRA: set `use_transition_lora=True` when this is a scene
    boundary (i.e. the two frames come from different scenes). This loads
    `ltx2.3-transition.safetensors` and appends the `zhuanchang` trigger
    to the prompt — gives the model a proper scene-transition prior and
    avoids the "first frame stuck at the end" failure mode you otherwise
    see without the LoRA."""
    fgs = float(first_guide_strength) if first_guide_strength is not None else float(guide_strength)
    lgs = float(last_guide_strength)  if last_guide_strength  is not None else float(guide_strength)
    extra_loras = []
    if use_transition_lora:
        extra_loras.append((TRANSITION_LORA, float(transition_lora_strength)))
        if TRANSITION_TRIGGER not in (prompt or "").lower():
            prompt = f"{(prompt or '').rstrip('. ')}. {TRANSITION_TRIGGER}"

    length = _round_length(seconds, fps)
    audio_ref_builder = None
    if audio_filename:
        # wrap_noise_mask=False: LTXVAudioVideoMask (via use_av_mask below)
        # owns the alignment/masking. Double-wrapping in SetLatentNoiseMask
        # would conflict and flatten the audio-driven motion cue.
        audio_ref_builder = lambda g, avae, w, h, length=length, fps=fps: _audio_from_file(
            g, audio_filename, seconds, w, h, avae, length=length, fps=fps,
            wrap_noise_mask=False)

    return _flf2v_graph_core(
        prompt, fps=fps, width=width, height=height,
        length=length,
        seed=seed, filename_prefix=filename_prefix,
        first_img_builder=lambda g: _flf2v_preprocess_frame(g, first_frame_filename, width, height),
        last_img_builder=lambda g: _flf2v_preprocess_frame(g, last_frame_filename, width, height),
        audio_ref_builder=audio_ref_builder,
        source_audio_filename=audio_filename,
        use_av_mask=bool(audio_filename),
        negative=negative, fast=fast,
        first_guide_strength=fgs, last_guide_strength=lgs,
        extra_loras=extra_loras,
        camera_lora=camera_lora, camera_lora_strength=camera_lora_strength,
        ckpt=checkpoint_name or CKPT, text_encoder=text_encoder or TEXT_ENCODER)


# ──────────── multi-guide (N keyframes in one latent) ────────────

def _multi_guide_graph_core(prompt, *, fps, width, height, length,
                             guide_filenames, guide_frame_indices,
                             guide_strengths, seed, filename_prefix,
                             audio_ref_builder=None,
                             source_audio_filename=None,
                             negative=None, fast=False, use_av_mask=False,
                             refine_guide_strength=1.0, extra_loras=None,
                             camera_lora=None, camera_lora_strength=0.8,
                             ckpt=CKPT, text_encoder=TEXT_ENCODER):
    """Build a graph that chains N LTXVAddGuide nodes at the specified
    frame positions through one continuous latent, then samples once (two-
    pass unless fast=True). Use for rapid character montages, beat-synced
    keyframe sequences, etc. — the transition LoRA handles the morphs.

    `guide_frame_indices` must be multiples of 8 per LTX's latent temporal
    quantization (8:1 compression). The caller is responsible for picking
    positions that fit inside [0, length-1]."""
    g = WorkflowGraph()
    checkpoint, clip, audio_vae, upscaler = _loaders(g, ckpt, text_encoder)
    model = _distilled_lora(g, checkpoint[0])
    for lora_name, lora_strength in (extra_loras or []):
        model = g.node("LoraLoaderModelOnly", model=model[0],
                       lora_name=lora_name, strength_model=float(lora_strength))
    model = _apply_extra_lora(g, model, camera_lora, camera_lora_strength)
    cond = _encode_prompts(g, clip[0], prompt, negative, fps)

    guide_imgs = [_flf2v_preprocess_frame(g, fn, width, height)
                  for fn in guide_filenames]

    empty_video = g.node("EmptyLTXVLatentVideo",
                         width=int(width), height=int(height),
                         length=int(length), batch_size=1)

    # Chain N AddGuides: each takes prev positive/negative/latent + its own image + frame_idx.
    prev = cond
    latent_ref = empty_video
    latent_idx = 0  # empty_video[0] is latent
    for img, idx, strength in zip(guide_imgs, guide_frame_indices, guide_strengths):
        ag = g.node("LTXVAddGuide",
                    positive=prev[0], negative=prev[1], vae=checkpoint[2],
                    latent=latent_ref[latent_idx], image=img[0],
                    frame_idx=int(idx), strength=float(strength))
        prev = ag
        latent_ref = ag
        latent_idx = 2  # AddGuide output [2] = latent
    guide_chain_cond = prev  # the last AddGuide output [positive,negative,latent]

    # Audio path (same as flf2v)
    if audio_ref_builder:
        audio_latent = audio_ref_builder(g, audio_vae, width, height,
                                         length=length, fps=fps)
    else:
        audio_latent = _empty_audio_latent(g, length, fps, audio_vae)
    if use_av_mask:
        effective_seconds = float(length) / float(fps)
        aligned = g.node("LTXVAudioVideoMask",
                         video_latent=guide_chain_cond[2],
                         audio_latent=audio_latent[0],
                         video_fps=float(fps),
                         video_start_time=0.0, video_end_time=effective_seconds,
                         audio_start_time=effective_seconds,
                         audio_end_time=effective_seconds,
                         max_length="pad")
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=aligned[0], audio_latent=aligned[1])
    else:
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=guide_chain_cond[2],
                           audio_latent=audio_latent[0])

    base_seed = seed or _rand_seed()
    av_pass1 = _pass_one(g, model, guide_chain_cond, av_latent, seed=base_seed)
    source_audio_seconds = float(length) / float(fps) if source_audio_filename else None
    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=guide_chain_cond,
                         source_audio_filename=source_audio_filename,
                         source_audio_seconds=source_audio_seconds)
    else:
        av_for_pass2, pass2_cond = _upsample_between_multi_guide(
            g, av_pass1, cond, vae=checkpoint[2], upscaler=upscaler,
            guide_imgs=guide_imgs, guide_frame_indices=guide_frame_indices,
            pass1_guide_chain_cond=guide_chain_cond,
            guide_strength=float(refine_guide_strength))
        av_final = _pass_two(g, model, pass2_cond, av_for_pass2,
                             seed=base_seed + 1)
        _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=pass2_cond,
                         source_audio_filename=source_audio_filename,
                         source_audio_seconds=source_audio_seconds)
    return g.to_dict()


def ltx2_multi_guide_to_video(guide_filenames, prompt,
                               guide_frame_indices=None,
                               guide_strengths=None,
                               seconds=8.0, fps=24, width=768, height=512,
                               filename_prefix="ltx2_multi_guide",
                               seed=None, negative=None, fast=False,
                               audio_filename=None,
                               use_transition_lora=True,
                               transition_lora_strength=1.0,
                               camera_lora=None, camera_lora_strength=0.8,
                               checkpoint_name=None, text_encoder=None, **_):
    """N-keyframe video in a single LTX pass: chains N LTXVAddGuide nodes
    through one latent so the model has to hit each image at its frame
    position. With the transition LoRA + audio latent this gives a coherent
    beat-synced keyframe montage (e.g. 12 character sheets across 8s).

    Args:
        guide_filenames: list of N image paths (server-relative or local —
            `comfy_graph.py` uploads locals before dispatch).
        prompt: text prompt describing the overall motion/style. The
            `zhuanchang` trigger is auto-appended when use_transition_lora.
        guide_frame_indices: list of N frame positions. If None, evenly
            spaced from 0 to the largest multiple-of-8 ≤ length-1. MUST be
            multiples of 8 (LTX 8:1 latent compression requirement).
        guide_strengths: list of N floats (per-guide AddGuide strength).
            Defaults to [1.0]*N. Lower (~0.6) = softer landing (more morph);
            higher (~1.0) = harder landing (closer to hard cut at that beat).
        seconds, fps, width, height: video shape (length = _round_length).
        audio_filename: optional song slice — drives the audio latent AND
            gets muxed into the output mp4 (same plumbing as flfa2v).
        use_transition_lora: defaults to True for this path — it's what
            makes 12-guide chains produce interpretable morphs.
    """
    length = _round_length(seconds, fps)

    # Default frame positions: evenly spaced multiples of 8.
    if guide_frame_indices is None:
        n = len(guide_filenames)
        max_idx = ((int(length) - 1) // 8) * 8
        if n == 1:
            guide_frame_indices = [0]
        else:
            step = max_idx // (n - 1)
            step = (step // 8) * 8 or 8  # snap to multiple of 8, min 8
            guide_frame_indices = [min(max_idx, i * step) for i in range(n)]
    else:
        guide_frame_indices = list(guide_frame_indices)
    if len(guide_frame_indices) != len(guide_filenames):
        raise ValueError(
            f"guide_frame_indices has {len(guide_frame_indices)} entries "
            f"but {len(guide_filenames)} guide images provided")
    for idx in guide_frame_indices:
        if int(idx) % 8 != 0:
            raise ValueError(f"guide_frame_indices must be multiples of 8, "
                             f"got {idx}")

    if guide_strengths is None:
        guide_strengths = [1.0] * len(guide_filenames)
    elif len(guide_strengths) != len(guide_filenames):
        raise ValueError(
            f"guide_strengths has {len(guide_strengths)} entries "
            f"but {len(guide_filenames)} guide images provided")

    extra_loras = []
    if use_transition_lora:
        extra_loras.append((TRANSITION_LORA, float(transition_lora_strength)))
        if TRANSITION_TRIGGER not in (prompt or "").lower():
            prompt = f"{(prompt or '').rstrip('. ')}. {TRANSITION_TRIGGER}"

    audio_ref_builder = None
    if audio_filename:
        audio_ref_builder = lambda g, avae, w, h, length=length, fps=fps: _audio_from_file(
            g, audio_filename, seconds, w, h, avae, length=length, fps=fps,
            wrap_noise_mask=False)

    return _multi_guide_graph_core(
        prompt, fps=fps, width=width, height=height, length=length,
        guide_filenames=guide_filenames,
        guide_frame_indices=guide_frame_indices,
        guide_strengths=guide_strengths,
        seed=seed, filename_prefix=filename_prefix,
        audio_ref_builder=audio_ref_builder,
        source_audio_filename=audio_filename,
        use_av_mask=bool(audio_filename),
        negative=negative, fast=fast,
        extra_loras=extra_loras,
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


def _video_tail_frames(g, video_filename, num_frames, width, height,
                        tail_skip_frames=0):
    """Return N frames ending at (total - tail_skip_frames). Set
    tail_skip_frames > 0 to skip a trailing buffer that isn't part of
    the scene's real content — e.g. the 0.5s × fps frames appended to
    lipsync renders as LTX audio-lookhead, which hold no meaningful
    continuation. Uses reverse→head→reverse because GetImageRangeFromBatch
    only supports start_index >= -1."""
    vid = g.node("LoadVideo", file=video_filename)
    comp = g.node("GetVideoComponents", video=vid[0])
    rev1 = g.node("ReverseImageBatch", images=comp[0])
    head_pp = _video_range_frames(g, video_filename,
                                   int(tail_skip_frames), num_frames,
                                   width, height,
                                   _pre_loaded_images=rev1[0])
    return g.node("ReverseImageBatch", images=head_pp[0])


def _video_head_frames(g, video_filename, num_frames, width, height):
    """Load a video and return the FIRST N frames as an LTXVPreprocess'd
    IMAGE batch."""
    return _video_range_frames(g, video_filename, 0, num_frames, width, height)


def ltx2_transition(first_frame_filename, last_frame_filename, prompt,
                    seconds=3.0, fps=24, width=720, height=1280,
                    filename_prefix="ltx2_transition",
                    seed=None, negative=None,
                    first_guide_strength=1.0, last_guide_strength=1.0,
                    audio_filename=None,
                    prev_video_filename=None, next_video_filename=None,
                    multiframe_guide=None,
                    multiframe_guide_last=None,
                    prev_video_start_frame=None,
                    next_video_start_frame=None,
                    prev_video_tail_buffer_sec=0.0,
                    slice_song_start_sec=None,
                    prev_video_song_start_sec=None,
                    next_video_song_start_sec=None,
                    next_video_vocal_offset_sec=0.0,
                    mask_start_sec=None,
                    mask_end_sec=None,
                    mask_max_length="pad",
                    use_mask=True,
                    use_inplace=True,
                    use_addguide=False,
                    addguide_strength=0.6,
                    fast=False,
                    # Sparse B-guide mode: pass a list of latent positions
                    # (multiples of 8). Each gets a SINGLE frame of next_video
                    # extracted at the song-time-matched frame, applied via
                    # LTXVAddGuide at strength=last_guide_strength. Overrides
                    # the default contiguous B-batch behaviour (which hard-
                    # locks a block of frames and causes a snap into B).
                    b_sparse_latent_positions=None,
                    last_guide_index=None,
                    camera_lora=None,
                    camera_lora_strength=0.8,
                    debug_save_audio=False,
                    checkpoint_name=None, text_encoder=None,
                    **_):
    """Transition clip scene N → scene N+1 via the ltx2.3-transition LoRA
    using a masked-latent architecture.

    Layout (defaults for 3s @ 24fps, length=73):
      - [0, mask_start_sec * fps)           → scene-A tail frames, locked via Inplace
      - [mask_start_sec*fps, mask_end_sec*fps) → REGENERATION region (noise-masked)
      - [mask_end_sec*fps, length)          → scene-B frames, locked via Inplace

    Defaults put the regeneration region in the middle third:
    mask_start_sec = seconds/3, mask_end_sec = 2*seconds/3. For a 3s clip
    this gives scene-A at [0,1s], morph at [1s,2s], scene-B at [2s,3s].

    The `LTXVAudioVideoMask` node marks the video sub-range as the region
    LTX should regenerate (noise-mask semantics). Audio is preserved in
    full via a zero-length mask at the audio tail. No LTXVAddGuide calls —
    guides are baked in via Inplace only, which has well-defined
    pixel-space indexing.

    Frame alignment (two ways to express):

      RAW: pass `prev_video_start_frame` / `next_video_start_frame`
      directly — the function uses scene-A frames [start, start+count) and
      scene-B frames [start, start+count) from the source mp4s.

      SONG-AWARE (preferred, auto-computes the raw frame indices):
        - `slice_song_start_sec`: song timecode where the audio slice begins
        - `prev_video_song_start_sec`: song timecode of prev_video.mp4 frame 0
        - `next_video_song_start_sec`: song timecode of next_video.mp4 frame 0
        - `next_video_vocal_offset_sec`: extra seconds to skip into
          next_video (if scene B has a silent / mouth-closed buffer before
          its vocal actually starts)

      Derived:
        prev_video_start_frame =
          round((slice_song_start_sec - prev_video_song_start_sec) * fps)
        next_video_start_frame =
          round((slice_song_start_sec + mask_end_sec
                 - next_video_song_start_sec
                 + next_video_vocal_offset_sec) * fps)

      Assembly must play scene-B from frame
      (next_video_start_frame + scene_b_count) to avoid rewind.
    """
    if TRANSITION_TRIGGER not in prompt.lower():
        prompt = f"{prompt.rstrip('. ')}. {TRANSITION_TRIGGER}"

    # === Convert all timing params to integer frames up-front. Internal
    # math uses frame indices exclusively to avoid cumulative
    # `round(sec * fps)` drift. Seconds-based inputs remain available as
    # user-facing convenience. ===
    fps_int = int(fps)
    length_frames = _round_length(seconds, fps)
    if mask_start_sec is None:
        mask_start_frame = length_frames // 3
    else:
        mask_start_frame = int(round(float(mask_start_sec) * fps_int))
    if mask_end_sec is None:
        mask_end_frame = (2 * length_frames) // 3
    else:
        mask_end_frame = int(round(float(mask_end_sec) * fps_int))

    # Derive block boundaries in latent frames, snapped to multiples of 8
    # (LTX requires multi-frame guides to start at 8-multiples).
    scene_a_count = (mask_start_frame // 8) * 8
    if multiframe_guide is not None:
        scene_a_count = int(multiframe_guide)
    if scene_a_count < 1:
        scene_a_count = 8

    scene_b_start = (mask_end_frame // 8) * 8
    if last_guide_index is not None:
        scene_b_start = int(last_guide_index)
    scene_b_count = int(multiframe_guide_last) if multiframe_guide_last is not None \
                                                else (length_frames - scene_b_start)

    # LTXVAudioVideoMask takes seconds; use the USER-specified frame
    # boundaries directly (NOT derived from scene block counts). If scene
    # A block size and mask_start don't match, latent frames between them
    # become empty zeros and decode as gray/held content. Caller must
    # align mask bounds and block sizes themselves.
    mask_start_time = float(mask_start_frame) / float(fps_int)
    mask_end_time = float(mask_end_frame) / float(fps_int)
    effective_seconds = float(length_frames) / float(fps_int)

    ckpt = checkpoint_name or CKPT
    g = WorkflowGraph()
    checkpoint, clip, audio_vae, upscaler = _loaders(g, ckpt, text_encoder or TEXT_ENCODER)
    model = _distilled_lora(g, checkpoint[0])
    model = _apply_extra_lora(g, model, camera_lora, camera_lora_strength)
    model = g.node("LoraLoaderModelOnly", model=model[0],
                   lora_name=TRANSITION_LORA, strength_model=1.0)
    cond = _encode_prompts(g, clip[0], prompt, negative, fps)

    # Base latent — empty (zeros), of the correct length. Then AddNoise
    # with sigma=1.0 fills it with pure noise. LTX reference workflows
    # VAE-encode a real video for this (non-zero latent throughout); we
    # synthesize equivalent via noise so the sampler's denoising step sees
    # a proper diffusion starting point in the masked region (rather than
    # zeros, which can cause mid-mask motion collapse).
    empty_video = g.node("EmptyLTXVLatentVideo",
                         width=int(width), height=int(height),
                         length=length_frames, batch_size=1)
    base_seed = seed or _rand_seed()
    noise_seed_val = base_seed + 10_000
    init_noise = g.node("RandomNoise", noise_seed=int(noise_seed_val))
    init_sigmas = g.node("ManualSigmas", sigmas="1.0, 0.0")
    noisy_video = g.node("AddNoise",
                         model=model[0],
                         noise=init_noise[0],
                         sigmas=init_sigmas[0],
                         latent_image=empty_video[0])

    # Resolve raw frame indices — either passed directly, or derived from
    # song-aware timing params. Song-time → frame conversion uses exact
    # fps_int multiplication with single rounding step.
    if prev_video_start_frame is None and slice_song_start_sec is not None \
            and prev_video_song_start_sec is not None:
        prev_video_start_frame = int(round(
            (float(slice_song_start_sec) - float(prev_video_song_start_sec)) * fps_int))
    if next_video_start_frame is None and slice_song_start_sec is not None \
            and next_video_song_start_sec is not None:
        # Scene-B arrives at transition latent `scene_b_start` which
        # corresponds to song time (slice_song_start_sec + scene_b_start/fps).
        next_video_start_frame = int(round(
            (float(slice_song_start_sec)
             + (float(scene_b_start) / float(fps_int))
             - float(next_video_song_start_sec)
             + float(next_video_vocal_offset_sec)) * fps_int))

    # Scene-A block: prev_video frames [prev_video_start_frame,
    # +scene_a_count). If the caller didn't specify a start, fall back to
    # the legacy "last N real frames" behavior (for backward compat).
    if prev_video_filename:
        if prev_video_start_frame is not None:
            first_img_ref = _video_range_frames(
                g, prev_video_filename,
                int(prev_video_start_frame), int(scene_a_count),
                width, height)
        else:
            tail_skip_frames = int(round(
                float(prev_video_tail_buffer_sec) * float(fps)))
            first_img_ref = _video_tail_frames(
                g, prev_video_filename, scene_a_count,
                width, height, tail_skip_frames=tail_skip_frames)
    else:
        first_img_ref = _flf2v_preprocess_frame(
            g, first_frame_filename, width, height)

    # Scene-B block: `scene_b_count` consecutive frames of next_video
    # starting at `next_video_start_frame` (or 0 if unspecified).
    # If sparse mode is active, we still build a contiguous batch (not used
    # by addguide_path below), mainly to keep inplace/flf2v fallbacks happy.
    if next_video_filename:
        last_img_ref = _video_range_frames(
            g, next_video_filename,
            int(next_video_start_frame or 0), int(scene_b_count),
            width, height)
    else:
        last_img_ref = _flf2v_preprocess_frame(
            g, last_frame_filename, width, height)

    # Build sparse single-frame B-guides (one per latent position in
    # b_sparse_latent_positions). Each is a distinct _video_range_frames
    # call with num_frames=1, anchored at the song-time-matched scene B
    # frame. Returns list of (latent_idx, image_ref) pairs.
    sparse_b_refs: list[tuple[int, object]] = []
    if b_sparse_latent_positions and next_video_filename:
        # Need slice_song_start_sec + next_video_song_start_sec to map
        # latent_idx → scene-B source-frame. If caller didn't pass those,
        # skip the mapping and place scene-B frame 0 at every position.
        have_song_mapping = (slice_song_start_sec is not None
                             and next_video_song_start_sec is not None)
        for lat_idx in b_sparse_latent_positions:
            lat_idx = int(lat_idx)
            if have_song_mapping:
                src_frame = int(round(
                    (float(slice_song_start_sec)
                     + float(lat_idx) / float(fps_int)
                     - float(next_video_song_start_sec)
                     + float(next_video_vocal_offset_sec)) * fps_int))
                src_frame = max(0, src_frame)
            else:
                src_frame = 0
            ref = _video_range_frames(
                g, next_video_filename, src_frame, 1, width, height)
            sparse_b_refs.append((lat_idx, ref))

    # Conditioning path used for sampling; may be modified by AddGuide
    cond_for_sampling = cond

    if use_addguide:
        # AddGuide handles multi-frame IMAGE batches natively — pass a 24-
        # frame batch with frame_idx=K and it occupies [K, K+23]. Unlike
        # LTXVImgToVideoInplaceKJ, which takes num_images="2" as TWO single
        # image insertions and only uses the first frame of each batch,
        # leaving the other 23 positions as noise → decoded output shows
        # one held frame at the start + empty middle + one held frame at
        # the end (the "frozen frame" artefact).
        #
        # Place the B-guide batch at `scene_b_start` (NOT frame_idx=-1 —
        # that's the flf2v single-frame END-target behaviour and it puts
        # the batch at [length-batch_size, length-1], fine for the
        # single-frame case but off-by-one against the caller-specified
        # scene_b_start in multi-frame mode).
        sa_guide = g.node("LTXVAddGuide",
                          positive=cond[0], negative=cond[1],
                          vae=checkpoint[2], latent=noisy_video[0],
                          image=first_img_ref[0],
                          frame_idx=0,
                          strength=float(first_guide_strength))
        if sparse_b_refs:
            # Chain N single-frame B-guides at their latent positions. Each
            # call conditions the next, so the sampler sees all of them.
            prev = sa_guide
            for lat_idx, img_ref in sparse_b_refs:
                prev = g.node("LTXVAddGuide",
                              positive=prev[0], negative=prev[1],
                              vae=checkpoint[2], latent=prev[2],
                              image=img_ref[0],
                              frame_idx=int(lat_idx),
                              strength=float(last_guide_strength))
            base_video_latent = prev[2]
            cond_for_sampling = prev
        else:
            sb_guide = g.node("LTXVAddGuide",
                              positive=sa_guide[0], negative=sa_guide[1],
                              vae=checkpoint[2], latent=sa_guide[2],
                              image=last_img_ref[0],
                              frame_idx=int(scene_b_start),
                              strength=float(last_guide_strength))
            base_video_latent = sb_guide[2]
            cond_for_sampling = sb_guide  # [0]=pos, [1]=neg
    elif use_inplace:
        # Bake scene A/B frames into the PRE-NOISED latent at idx 0 and
        # scene_b_start, via LTXVImgToVideoInplaceKJ. Inplace replaces the
        # noise at those specific latent positions with VAE-encoded scene
        # content (strength 1.0 = hard replace); the middle mask region
        # retains the noise, giving the sampler a proper diffusion
        # starting point to denoise from.
        inplace = g.node("LTXVImgToVideoInplaceKJ",
                         vae=checkpoint[2], latent=noisy_video[0],
                         num_images="2",
                         **{
                             "num_images.image_1":    first_img_ref[0],
                             "num_images.index_1":    0,
                             "num_images.strength_1": float(first_guide_strength),
                             "num_images.image_2":    last_img_ref[0],
                             "num_images.index_2":    int(scene_b_start),
                             "num_images.strength_2": float(last_guide_strength),
                         })
        base_video_latent = inplace[0]
    else:
        base_video_latent = noisy_video[0]

    # --- Audio: encode + optional SetLatentNoiseMask wrap. With
    # `use_mask=False`, the SetLatentNoiseMask wrap lets the sampler treat
    # the video as fully regeneratable noise (letting the LoRA fill it).
    if audio_filename:
        audio_latent = _audio_from_file(
            g, audio_filename, seconds, width, height, audio_vae,
            length=length_frames, fps=fps_int,
            debug_save_prefix=(filename_prefix if debug_save_audio else None),
            wrap_noise_mask=(not use_mask))
    else:
        audio_latent = _empty_audio_latent(g, length_frames, fps_int, audio_vae)

    if use_mask:
        masked_av = g.node("LTXVAudioVideoMask",
                            video_fps=float(fps_int),
                            video_start_time=mask_start_time,
                            video_end_time=mask_end_time,
                            audio_start_time=effective_seconds,
                            audio_end_time=effective_seconds,
                            max_length=str(mask_max_length),
                            video_latent=base_video_latent,
                            audio_latent=audio_latent[0])
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=masked_av[0],
                           audio_latent=masked_av[1])
    else:
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=base_video_latent,
                           audio_latent=audio_latent[0])

    # --- Sample. `cond_for_sampling` is either the raw LTXVConditioning
    # or the chained AddGuide output (pos, neg) — works for KSampler either
    # way since only positive/negative slots are used.
    # base_seed already assigned earlier for noise init.
    av_pass1 = _pass_one(g, model, cond_for_sampling, av_latent, seed=base_seed)

    # Fast path: decode pass-1 directly, no upsample/refine. Strip the pass-1
    # guide tokens before decode (same CropGuides pattern _build uses) so the
    # N_A + N_B extra latent frames don't tail-append to the decoded mp4.
    # Handy for rapid iteration — output is ~half the wall time and lower
    # detail but correct geometry and duration.
    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=(cond_for_sampling if use_addguide else None))
        return g.to_dict()

    # Between-pass refine: separate, upsample video. Re-apply anchoring on
    # the upsampled latent (AddGuide or Inplace, matching pass 1), then
    # re-apply mask (if use_mask), then concat audio.
    sep = g.node("LTXVSeparateAVLatent", av_latent=av_pass1[0])
    cropped_cond = cond_for_sampling
    if use_addguide:
        # Strip pass-1 AddGuides from latent before upsample (CropGuides
        # removes the guide frames from the latent so the upsample doesn't
        # stretch them).
        cropped = g.node("LTXVCropGuides",
                         positive=cond_for_sampling[0],
                         negative=cond_for_sampling[1],
                         latent=sep[0])
        upsampled = g.node("LTXVLatentUpsampler",
                           samples=cropped[2], upscale_model=upscaler[0],
                           vae=checkpoint[2])
        cropped_cond = cropped
    else:
        upsampled = g.node("LTXVLatentUpsampler",
                           samples=sep[0], upscale_model=upscaler[0],
                           vae=checkpoint[2])

    refine_cond = cropped_cond
    if use_addguide:
        sa_guide2 = g.node("LTXVAddGuide",
                           positive=cropped_cond[0], negative=cropped_cond[1],
                           vae=checkpoint[2], latent=upsampled[0],
                           image=first_img_ref[0],
                           frame_idx=0,
                           strength=float(first_guide_strength))
        if sparse_b_refs:
            prev2 = sa_guide2
            for lat_idx, img_ref in sparse_b_refs:
                prev2 = g.node("LTXVAddGuide",
                               positive=prev2[0], negative=prev2[1],
                               vae=checkpoint[2], latent=prev2[2],
                               image=img_ref[0],
                               frame_idx=int(lat_idx),
                               strength=float(last_guide_strength))
            refine_video_latent = prev2[2]
            refine_cond = prev2
        else:
            sb_guide2 = g.node("LTXVAddGuide",
                               positive=sa_guide2[0], negative=sa_guide2[1],
                               vae=checkpoint[2], latent=sa_guide2[2],
                               image=last_img_ref[0],
                               frame_idx=int(scene_b_start),
                               strength=float(last_guide_strength))
            refine_video_latent = sb_guide2[2]
            refine_cond = sb_guide2
    elif use_inplace:
        inplace2 = g.node("LTXVImgToVideoInplaceKJ",
                          vae=checkpoint[2], latent=upsampled[0],
                          num_images="2",
                          **{
                              "num_images.image_1":    first_img_ref[0],
                              "num_images.index_1":    0,
                              "num_images.strength_1": float(first_guide_strength),
                              "num_images.image_2":    last_img_ref[0],
                              "num_images.index_2":    int(scene_b_start),
                              "num_images.strength_2": float(last_guide_strength),
                          })
        refine_video_latent = inplace2[0]
    else:
        refine_video_latent = upsampled[0]

    if use_mask:
        masked_av2 = g.node("LTXVAudioVideoMask",
                             video_fps=float(fps_int),
                             video_start_time=mask_start_time,
                             video_end_time=mask_end_time,
                             audio_start_time=effective_seconds,
                             audio_end_time=effective_seconds,
                             max_length=str(mask_max_length),
                             video_latent=refine_video_latent,
                             audio_latent=sep[1])
        av_for_pass2 = g.node("LTXVConcatAVLatent",
                              video_latent=masked_av2[0],
                              audio_latent=masked_av2[1])
    else:
        av_for_pass2 = g.node("LTXVConcatAVLatent",
                              video_latent=refine_video_latent,
                              audio_latent=sep[1])
    av_final = _pass_two(g, model, refine_cond, av_for_pass2, seed=base_seed + 1)
    # Strip the pass-2 AddGuide tokens before decode — each AddGuide inserts
    # its guide frames into the latent (pass-1 guides were already stripped
    # via CropGuides before the upsample; pass-2 re-adds them at the same
    # positions). Without this, the final mp4 contains the N_A + N_B guide
    # frames appended at the tail and the video duration overshoots by
    # (N_A + N_B) / fps seconds.
    _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                     fps=fps, filename_prefix=filename_prefix,
                     strip_guides_cond=(refine_cond if use_addguide else None))
    return g.to_dict()


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


# ──────────── continuation (last-frames-to-video) ────────────

def _previous_frames_branch(g, prev_video_filename, prev_total_frames,
                              overlap_frames, width, height):
    """Build an IMAGE-batch branch that carries the LAST `overlap_frames`
    frames of `prev_video_filename` (server-side filename), resized to
    (width, height) and preprocessed for LTXVAddGuide consumption.

    Uses LoadVideo → GetVideoComponents → GetImageRangeFromBatch(
    start_index=prev_total_frames - overlap_frames, num_frames=overlap_frames)
    per the pattern already used in post.concat_videos. The caller MUST
    probe `prev_total_frames` (via ffprobe or similar) because the
    GetImageRangeFromBatch node doesn't support negative indexing on
    this server.
    """
    loaded = g.node("LoadVideo", file=prev_video_filename)
    components = g.node("GetVideoComponents", video=loaded[0])
    # components[0] is the IMAGE batch (all frames of the video)
    start_idx = max(0, int(prev_total_frames) - int(overlap_frames))
    tail = g.node("GetImageRangeFromBatch",
                  images=components[0],
                  start_index=start_idx,
                  num_frames=int(overlap_frames))
    # Resize + preprocess the same way single-frame guides do.
    resized = g.node("ResizeImageMaskNode", **{
        "input": tail[0],
        "resize_type": "scale dimensions",
        "resize_type.width": int(width),
        "resize_type.height": int(height),
        "resize_type.crop": "center",
        "scale_method": "nearest-exact",
    })
    return g.node("LTXVPreprocess", image=resized[0], img_compression=25)


def ltx2_continuation_to_video(prev_video_filename, prev_total_frames,
                                prompt,
                                overlap_seconds=1.0,
                                seconds=5, fps=24, width=768, height=512,
                                filename_prefix="ltx2_continuation",
                                seed=None, negative=None, fast=False,
                                overlap_strength=1.0,
                                audio_filename=None,
                                identity_anchor_image=None,
                                identity_strength=0.3,
                                camera_lora=None, camera_lora_strength=0.8,
                                checkpoint_name=None, text_encoder=None, **_):
    """Seamless video continuation.

    Lock the first `overlap_seconds` of the new clip's latent to the
    LAST `overlap_seconds` of `prev_video_filename` via a multi-frame
    LTXVAddGuide(frame_idx=0, strength=1.0). The model then generates
    the rest of the clip from the prompt (+ optional audio), giving
    frame-exact continuity at the seam instead of a hard cut.

    Args:
        prev_video_filename: server-side basename of the previous clip
            (already uploaded via comfy_graph's upload_if_local). Must
            match the target fps — force_rate handles minor mismatches.
        prev_total_frames: int, total frames in the previous video.
            Probe via ffprobe on the caller side:
                ffprobe -count_packets -show_entries stream=nb_read_packets ...
            We can't derive this server-side without a node that does
            negative indexing.
        overlap_seconds: how many seconds of the previous clip to lock.
            Default 1.0 s. Too small (<0.5s) and LTX bootstrap drifts;
            too large (>2s) and the new shot has to "break away" from
            the lock which costs intent. 0.5-1.5s is the sweet spot.
        overlap_strength: LTXVAddGuide strength, default 1.0 (hard lock).
            Lower values allow the model to blend the tail rather than
            commit to it — try 0.85 if the seam looks "frozen".
        All other args match ltx2_image_audio_to_video.
    """
    length = _round_length(seconds, fps)
    overlap_frames = max(1, int(round(float(overlap_seconds) * float(fps))))
    # LTXVAddGuide with num_frames > 1 needs the guide to start at a
    # multiple of 8. frame_idx=0 is already aligned; no snap needed.

    audio_ref_builder = None
    if audio_filename:
        audio_ref_builder = lambda g, avae, w, h, length=length, fps=fps: _audio_from_file(
            g, audio_filename, seconds, w, h, avae, length=length, fps=fps,
            wrap_noise_mask=not bool(audio_filename))

    # Build the graph inline (simpler than adding another *_graph_core).
    g = WorkflowGraph()
    ckpt = checkpoint_name or CKPT
    te = text_encoder or TEXT_ENCODER
    checkpoint, clip, audio_vae, upscaler = _loaders(g, ckpt, te)
    model = _distilled_lora(g, checkpoint[0])
    model = _apply_extra_lora(g, model, camera_lora, camera_lora_strength)
    cond = _encode_prompts(g, clip[0], prompt, negative, fps)

    # Last N frames of the previous clip as the anchor image batch.
    tail_branch = _previous_frames_branch(g, prev_video_filename,
                                          prev_total_frames, overlap_frames,
                                          width, height)

    empty_video = g.node("EmptyLTXVLatentVideo",
                         width=int(width), height=int(height),
                         length=int(length), batch_size=1)

    # Lock the first `overlap_frames` frames via a multi-frame AddGuide
    # at frame_idx=0 strength=overlap_strength. LTXVCropGuides at the
    # end will strip the inserted guide tokens.
    front_guide = g.node("LTXVAddGuide",
                          positive=cond[0], negative=cond[1], vae=checkpoint[2],
                          latent=empty_video[0], image=tail_branch[0],
                          frame_idx=0, strength=float(overlap_strength))

    # Optional identity anchor — placed at a MIDDLE frame (past the
    # overlap lock window so it doesn't fight the front anchor). Low
    # strength 0.3 default acts as a soft identity bias for mid-shot
    # frames; it's NOT a last-frame target (which is the flf2v behavior
    # and forces generation to converge on that image — breaks lipsync
    # and motion). Position = overlap_end + (remaining // 2), snapped
    # to multiple of 8.
    id_branch = None
    if identity_anchor_image:
        id_branch = _flf2v_preprocess_frame(
            g, identity_anchor_image, width, height)
        overlap_end = overlap_frames
        remaining = int(length) - overlap_end
        raw_frame = overlap_end + remaining // 2
        id_frame = max(overlap_end + 8, (raw_frame // 8) * 8)
        # Clamp below length-8 so we don't accidentally target last frame
        id_frame = min(id_frame, ((int(length) - 8) // 8) * 8)
        guide = g.node("LTXVAddGuide",
                        positive=front_guide[0], negative=front_guide[1],
                        vae=checkpoint[2], latent=front_guide[2],
                        image=id_branch[0], frame_idx=id_frame,
                        strength=float(identity_strength))
    else:
        guide = front_guide

    # Audio: if caller passed audio_filename, we both condition on it AND
    # (via use_av_mask=True) feed the real song slice as the final mp4's
    # audio track — same treatment as flfa2v.
    use_av_mask = bool(audio_filename)
    if audio_ref_builder:
        audio_latent = audio_ref_builder(g, audio_vae, width, height)
    else:
        audio_latent = _empty_audio_latent(g, length, fps, audio_vae)

    if use_av_mask:
        eff_s = float(length) / float(fps)
        aligned = g.node("LTXVAudioVideoMask",
                         video_latent=guide[2], audio_latent=audio_latent[0],
                         video_fps=float(fps),
                         video_start_time=0.0, video_end_time=eff_s,
                         audio_start_time=eff_s, audio_end_time=eff_s,
                         max_length="pad")
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=aligned[0], audio_latent=aligned[1])
    else:
        av_latent = g.node("LTXVConcatAVLatent",
                           video_latent=guide[2], audio_latent=audio_latent[0])

    base_seed = seed or _rand_seed()
    av_pass1 = _pass_one(g, model, guide, av_latent, seed=base_seed)
    source_audio_seconds = float(length) / float(fps) if audio_filename else None

    if fast:
        _decode_and_save(g, av_pass1, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=guide,
                         source_audio_filename=audio_filename,
                         source_audio_seconds=source_audio_seconds)
    else:
        # Two-pass: separate → upsample → crop pass-1 guides → re-apply
        # BOTH guides (front + identity) on the upsampled latent → pass 2.
        sep = g.node("LTXVSeparateAVLatent", av_latent=av_pass1[0])
        upsampled = g.node("LTXVLatentUpsampler",
                           samples=sep[0], upscale_model=upscaler[0],
                           vae=checkpoint[2])
        cleaned = g.node("LTXVCropGuides",
                         positive=guide[0], negative=guide[1],
                         latent=upsampled[0])
        front2 = g.node("LTXVAddGuide",
                         positive=cond[0], negative=cond[1], vae=checkpoint[2],
                         latent=cleaned[2], image=tail_branch[0],
                         frame_idx=0, strength=float(overlap_strength))
        if id_branch is not None:
            overlap_end = overlap_frames
            remaining = int(length) - overlap_end
            raw_frame = overlap_end + remaining // 2
            id_frame = max(overlap_end + 8, (raw_frame // 8) * 8)
            id_frame = min(id_frame, ((int(length) - 8) // 8) * 8)
            g2 = g.node("LTXVAddGuide",
                         positive=front2[0], negative=front2[1],
                         vae=checkpoint[2], latent=front2[2],
                         image=id_branch[0], frame_idx=id_frame,
                         strength=float(identity_strength))
        else:
            g2 = front2
        av_for_pass2 = g.node("LTXVConcatAVLatent",
                              video_latent=g2[2], audio_latent=sep[1])
        av_final = _pass_two(g, model, g2, av_for_pass2, seed=base_seed + 1)
        _decode_and_save(g, av_final, vae=checkpoint[2], audio_vae=audio_vae,
                         fps=fps, filename_prefix=filename_prefix,
                         strip_guides_cond=g2,
                         source_audio_filename=audio_filename,
                         source_audio_seconds=source_audio_seconds)
    return g.to_dict()
