"""Curated two-pass LTX profiles and explicit continuation source spans.

The legacy 2.3 builders remain available in ltx2. This narrow profile entry point
uses final (not coarse) dimensions and rejects unverified optional features.
"""
from __future__ import annotations

import math
from .graph import WorkflowGraph
from . import ltx2

PROFILES = ('ltx-2.3', 'ltx-2.5')
ASSETS_25 = {
    'transformer': 'ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors',
    'text_encoder': 'gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors',
    'video_vae': 'ltx-2.5-video-vae-bf16.safetensors',
    'audio_vae': 'ltx-2.5-audio-vae-bf16.safetensors',
    'upscaler': 'ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors',
}


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def continuation_timeline(*, previous_frames, frames, fps=24,
                          overlap_seconds=1.0, overlap_frames=None,
                          previous_retained_end=None, previous_source_start=0):
    """Return half-open local guide/retain and global source spans.

    Round seconds UP to 8k+1 before selecting the previous tail. An explicit
    count must already satisfy the cadence; never silently crop it downstream.
    """
    _integer('previous_frames', previous_frames, 1)
    _integer('frames', frames, 1)
    _integer('fps', fps, 1)
    _integer('previous_source_start', previous_source_start)
    end = previous_frames if previous_retained_end is None else previous_retained_end
    _integer('previous_retained_end', end, 1)
    if overlap_frames is None:
        if not math.isfinite(overlap_seconds) or overlap_seconds <= 0:
            raise ValueError('overlap_seconds must be positive and finite')
        requested = math.ceil(overlap_seconds * fps)
        overlap_frames = max(1, math.ceil((requested - 1) / 8) * 8 + 1)
    _integer('overlap_frames', overlap_frames, 1)
    if frames % 8 != 1 or overlap_frames % 8 != 1:
        raise ValueError('frames and overlap_frames must be 8k+1')
    if end > previous_frames or overlap_frames > end or overlap_frames >= frames:
        raise ValueError('overlap must fit the retained tail and leave new frames')
    start = previous_source_start + end - overlap_frames
    return dict(overlap_frames=overlap_frames, guide_span=[end-overlap_frames, end],
                source_span=[start, start+frames], retain_span=[overlap_frames, frames],
                audio_start_sample_48k=start*48000//fps)


def build_video(prompt, *, model_profile='ltx-2.3', workflow=None,
                frames=49, fps=24, width=512, height=320, seed=None,
                filename_prefix='ltx_profile', negative=None,
                image_filename=None, prev_video_filename=None, previous_frames=None,
                previous_retained_end=None, previous_source_start=None,
                overlap_seconds=None, overlap_frames=None, overlap_strength=None,
                audio_filename=None, audio_start_frame=None,
                base_guide_strength=None, refine_guide_strength=None,
                first_frame_filename=None, last_frame_filename=None,
                first_guide_strength=None, last_guide_strength=None, **unsupported):
    """Pure builder; uploaded audio is frozen conditioning AND source mux.

    Continuation is a guide, not a guarantee of perceptually seamless motion.
    Its audio offset defaults to the global source span, not the retained span.
    """
    if model_profile not in PROFILES:
        raise ValueError(f'unknown model_profile: {model_profile}')
    if unsupported:
        raise ValueError(f'unverified options: {sorted(unsupported)}')
    if workflow not in (None, 't2v', 'i2v', 'ia2v', 'continuation', 'flf2v'):
        raise ValueError(f'unverified workflow: {workflow}')
    for name, value in (('prompt', prompt), ('filename_prefix', filename_prefix),
                        ('negative', negative), ('image_filename', image_filename),
                        ('prev_video_filename', prev_video_filename), ('audio_filename', audio_filename),
                        ('first_frame_filename', first_frame_filename), ('last_frame_filename', last_frame_filename)):
        if (value is not None or name in ('prompt', 'filename_prefix')) and not isinstance(value, str):
            raise ValueError(f'{name} must be a string')
        if name.endswith('_filename') and value is not None and not value.strip():
            raise ValueError(f'{name} must not be empty')
    endpoints = workflow == 'flf2v'
    if endpoints:
        if not first_frame_filename or not last_frame_filename:
            raise ValueError('flf2v requires first_frame_filename and last_frame_filename')
        if any(v is not None for v in (image_filename, prev_video_filename, audio_start_frame,
                                      base_guide_strength, refine_guide_strength)):
            raise ValueError('flf2v does not accept image/continuation/audio offset fields')
    elif any(v is not None for v in (first_frame_filename, last_frame_filename,
                                     first_guide_strength, last_guide_strength)):
        raise ValueError('first/last fields require flf2v')
    first_guide_strength = .7 if first_guide_strength is None else first_guide_strength
    last_guide_strength = .7 if last_guide_strength is None else last_guide_strength
    for name, value in [('frames', frames), ('fps', fps), ('width', width), ('height', height)]:
        _integer(name, value, 1)
    if frames % 8 != 1 or frames < 9 or width % 64 or height % 64:
        raise ValueError('frames must be 8k+1 >=9; final dimensions must be multiples of 64')
    # Bounds captured from the local backend on 2026-09-30. Account for
    # half-size coarse latents and final-size guide resize nodes separately.
    maximum_dimension = 16384 if image_filename or prev_video_filename or endpoints else 32768
    if not 128 <= width <= maximum_dimension or not 128 <= height <= maximum_dimension:
        raise ValueError(f'final dimensions must be in [128,{maximum_dimension}]')
    if frames > (16384 if audio_filename else 1000):
        raise ValueError('frames exceed video/generated-audio backend bounds')
    if endpoints and frames - 1 > 9999:
        raise ValueError('last frame index exceeds LTXVAddGuide bounds')
    if fps > 120:
        raise ValueError('fps must be <= 120 (CreateVideo)')
    if seed is not None:
        _integer('seed', seed)
        if seed > 2**64 - 2:
            raise ValueError('seed and seed+1 must fit unsigned 64-bit RandomNoise')
    if not prev_video_filename and any(value is not None for value in (
            previous_frames, previous_retained_end, previous_source_start,
            overlap_seconds, overlap_frames, overlap_strength)):
        raise ValueError('continuation timing/strength requires prev_video_filename')
    if not image_filename and (base_guide_strength is not None or refine_guide_strength is not None):
        raise ValueError('image guide strengths require image_filename')
    previous_source_start = 0 if previous_source_start is None else previous_source_start
    overlap_seconds = 1.0 if overlap_seconds is None else overlap_seconds
    overlap_strength = 1.0 if overlap_strength is None else overlap_strength
    base_guide_strength = 0.7 if base_guide_strength is None else base_guide_strength
    refine_guide_strength = 1.0 if refine_guide_strength is None else refine_guide_strength
    if image_filename and prev_video_filename:
        raise ValueError('image and continuation guide cannot be combined')
    if workflow in ('i2v', 'ia2v') and not image_filename:
        raise ValueError(f'{workflow} requires image_filename')
    if workflow == 'ia2v' and not audio_filename:
        raise ValueError('ia2v requires audio_filename')
    if workflow == 'continuation' and not prev_video_filename:
        raise ValueError('continuation requires prev_video_filename')
    if workflow in ('t2v', 'continuation') and image_filename:
        raise ValueError('workflow does not accept image_filename')
    if workflow in ('t2v', 'i2v', 'ia2v') and prev_video_filename:
        raise ValueError('workflow does not accept prev_video_filename')
    if not audio_filename and audio_start_frame is not None:
        raise ValueError('audio_start_frame requires master audio')
    for value in (overlap_strength, base_guide_strength, refine_guide_strength,
                  first_guide_strength, last_guide_strength):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('guide strengths must be finite in [0,1]')
    timeline = None
    if prev_video_filename:
        timeline = continuation_timeline(previous_frames=previous_frames, frames=frames, fps=fps,
            overlap_seconds=overlap_seconds, overlap_frames=overlap_frames,
            previous_retained_end=previous_retained_end, previous_source_start=previous_source_start)
    if timeline and (timeline['guide_span'][0] > 4096 or timeline['overlap_frames'] > 4096):
        raise ValueError('continuation guide range exceeds GetImageRangeFromBatch bounds')
    if audio_start_frame is None:
        audio_start_frame = timeline['source_span'][0] if timeline else 0
    _integer('audio_start_frame', audio_start_frame)
    if timeline and audio_filename and audio_start_frame != timeline['source_span'][0]:
        raise ValueError('master audio offset must equal continuation source start')

    g = WorkflowGraph()
    if model_profile == 'ltx-2.5':
        model = g.node('UNETLoader', unet_name=ASSETS_25['transformer'], weight_dtype='default')
        clip = g.node('CLIPLoader', clip_name=ASSETS_25['text_encoder'], type='ltxv', device='default')
        vae = g.node('VAELoader', vae_name=ASSETS_25['video_vae'])[0]
        avae = g.node('VAELoader', vae_name=ASSETS_25['audio_vae'])
        up = g.node('LatentUpscaleModelLoader', model_name=ASSETS_25['upscaler'])
    else:
        checkpoint, clip, avae, up = ltx2._loaders(g)
        vae = checkpoint[2]
        model = ltx2._distilled_lora(g, checkpoint[0])
    cond = ltx2._encode_prompts(g, clip[0], prompt, negative, fps)
    latent = g.node('EmptyLTXVLatentVideo', width=width//2, height=height//2, length=frames, batch_size=1)[0]
    guide_image = None
    if timeline:
        guide_image = ltx2._previous_frames_branch(g, prev_video_filename,
            timeline['guide_span'][1], timeline['overlap_frames'], width, height)
    elif image_filename:
        guide_image = ltx2._flf2v_preprocess_frame(g, image_filename, width, height)

    endpoint_images = None
    if endpoints:
        endpoint_images = [ltx2._flf2v_preprocess_frame(g, fn, width, height, single_frame=True)
                           for fn in (first_frame_filename, last_frame_filename)]

    source_audio = None
    if audio_filename:
        loaded = g.node('LoadAudio', audio=audio_filename)
        source_audio = g.node('TrimAudioDuration', audio=loaded[0],
                              start_index=audio_start_frame/fps, duration=frames/fps)
        encoded = g.node('LTXVAudioVAEEncode', audio=source_audio[0], audio_vae=avae[0])
        mask = g.node('SolidMask', value=0, width=width//2, height=height//2)
        audio = g.node('SetLatentNoiseMask', samples=encoded[0], mask=mask[0])[0]
    else:
        audio = ltx2._empty_audio_latent(g, frames, fps, avae)[0]
    base_seed = ltx2._rand_seed() if seed is None else seed

    for stage in range(2):
        stage_cond = cond
        if endpoint_images:
            # Positive last index targets the requested last frame, not the
            # growing guide-token latent (and never a multi-frame -1 guide).
            for img, idx, strength in zip(endpoint_images, (0, frames - 1),
                                         (first_guide_strength, last_guide_strength)):
                guide = g.node('LTXVAddGuide', positive=stage_cond[0], negative=stage_cond[1],
                               vae=vae, latent=latent, image=img[0], frame_idx=idx, strength=strength)
                stage_cond, latent = guide, guide[2]
        elif timeline:
            guide = g.node('LTXVAddGuide', positive=cond[0], negative=cond[1],
                           vae=vae, latent=latent, image=guide_image[0], frame_idx=0,
                           strength=overlap_strength)
            stage_cond, latent = guide, guide[2]
        elif guide_image:
            guided = g.node('LTXVImgToVideoInplace', vae=vae, image=guide_image[0],
                            latent=latent, strength=base_guide_strength if stage == 0 else refine_guide_strength,
                            bypass=False)
            latent = guided[0]
        av = g.node('LTXVConcatAVLatent', video_latent=latent, audio_latent=audio)
        if model_profile == 'ltx-2.5':
            guider = g.node('LTXVDualCFGGuider', model=model[0], positive=stage_cond[0],
                           negative=stage_cond[1], video_cfg=1.0, audio_cfg=1.0)
            sampled = g.node('SamplerCustomAdvanced',
                noise=g.node('RandomNoise', noise_seed=base_seed+stage)[0], guider=guider[0],
                sampler=g.node('KSamplerSelect', sampler_name='euler_ancestral')[0],
                sigmas=g.node('ManualSigmas', sigmas=ltx2.SIGMAS_PASS1 if stage == 0 else ltx2.SIGMAS_PASS2)[0],
                latent_image=av[0])
        else:
            sample = ltx2._pass_one if stage == 0 else ltx2._pass_two
            sampled = sample(g, model, stage_cond, av, base_seed+stage)
        split = g.node('LTXVSeparateAVLatent', av_latent=sampled[0])
        latent, audio = split[0], split[1]
        if timeline or endpoint_images:
            cropped = g.node('LTXVCropGuides', positive=stage_cond[0], negative=stage_cond[1], latent=latent)
            latent = cropped[2]
        if stage == 0:
            latent = g.node('LTXVLatentUpsampler', samples=latent, upscale_model=up[0], vae=vae)[0]
            if source_audio:
                # Reuse the original encoded source, not a potentially altered sampler output.
                audio = g.node('SetLatentNoiseMask', samples=encoded[0], mask=mask[0])[0]
    images = g.node('VAEDecodeTiled', samples=latent, vae=vae,
                    tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16)
    audio_out = source_audio or g.node('LTXVAudioVAEDecode', samples=audio, audio_vae=avae[0])
    video = g.node('CreateVideo', images=images[0], audio=audio_out[0], fps=float(fps))
    g.node('SaveVideo', video=video[0], filename_prefix=filename_prefix, format='auto', codec='auto')
    return g.to_dict()


def build_flf2v(prompt, *, model_profile='ltx-2.3', first_frame_filename,
                 last_frame_filename, frames=49, fps=24, width=512, height=320,
                 seed=None, filename_prefix='ltx_flf2v_profile', negative=None,
                 audio_filename=None, first_guide_strength=None, last_guide_strength=None):
    """First loaded frame of each endpoint guides both passes; final dimensions.

    Supplied audio starts at zero, is frozen in both passes, and is muxed from
    the original source. This path is offline-validated, not live-certified.
    """
    return build_video(prompt, model_profile=model_profile, workflow='flf2v',
        first_frame_filename=first_frame_filename, last_frame_filename=last_frame_filename,
        frames=frames, fps=fps, width=width, height=height, seed=seed,
        filename_prefix=filename_prefix, negative=negative, audio_filename=audio_filename,
        first_guide_strength=first_guide_strength, last_guide_strength=last_guide_strength)
