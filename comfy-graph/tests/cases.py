"""Finite legacy characterization matrix; no local media is ever required.

Parameter-to-case map: every builder has /default (required arguments only)
and /custom (every explicit parameter, including required inputs, changed).
**kw wrappers forward the complete target parameter set in /custom; **_ LTX
builders receive ignored_keyword there. The coverage test verifies this map
against signatures, so a newly declared parameter cannot silently escape.
Named variants below cover interacting branches without a Cartesian product.
Seeds/negatives/fast and attention are intentionally ambient legacy behavior.
"""

P = 'A dancer crosses a sunlit room.'
BUILDERS = {
    'flux2_text_to_image': ('flux2', {'prompt': P}),
    'flux2_multi_reference_edit': ('flux2', {'image_filenames': ['ref-a.png'], 'prompt': P}),
    'flux2_single_image_edit': ('flux2', {'image_filename': 'ref-a.png', 'prompt': P}),
    'flux2_double_image_edit': ('flux2', {'image1_filename': 'ref-a.png', 'image2_filename': 'ref-b.png', 'prompt': P}),
    'flux2_multi_reference_edit_multiprompt': ('flux2', {'image_filenames': ['ref-a.png'], 'angle_prompts': ['front', 'side']}),
    'flux2_double_image_edit_multiprompt': ('flux2', {'image1_filename': 'ref-a.png', 'image2_filename': 'ref-b.png', 'angle_prompts': ['front', 'side']}),
    'flux2_multiple_angles': ('flux2', {'image_filename': 'ref-a.png', 'angle_prompts': ['front', 'side']}),
    'qwen_tts': ('tts', {'text': 'Hello.'}),
    'qwen_voice_clone': ('tts', {'text': 'Hello.'}),
    'extract_stems': ('post', {'audio_filename': 'song.wav'}),
    'transcribe': ('post', {'audio_filename': 'song.wav'}),
    'concat_videos_ffmpeg': ('post', {'staged_subfolder': 'staged-clips'}),
    'concat_videos': ('post', {'video_filenames': ['scene-a.mp4']}),
    'ltx2_text_to_video': ('ltx2', {'prompt': P}),
    'ltx2_ingredients_to_video': ('ltx2', {'sheet_image': 'sheet.png', 'prompt': P}),
    'ltx2_image_to_video': ('ltx2', {'image_filename': 'ref-a.png', 'prompt': P}),
    'ltx2_image_audio_to_video': ('ltx2', {'image_filename': 'ref-a.png', 'audio_filename': 'song.wav', 'prompt': P}),
    'ltx2_first_last_frame_to_video': ('ltx2', {'first_frame_filename': 'ref-a.png', 'last_frame_filename': 'ref-b.png', 'prompt': P}),
    'ltx2_multi_guide_to_video': ('ltx2', {'guide_filenames': ['ref-a.png', 'ref-b.png'], 'prompt': P}),
    'ltx2_transition': ('ltx2', {'first_frame_filename': 'ref-a.png', 'last_frame_filename': 'ref-b.png', 'prompt': P}),
    'extract_last_frame': ('ltx2', {'video_server_path': '/app/ComfyUI/output/scene-a.mp4'}),
    'ltx2_continuation_to_video': ('ltx2', {'prev_video_filename': 'scene-a.mp4', 'prev_total_frames': 121, 'prompt': P}),
}
FLUX_CUSTOM = dict(width=640, height=384, steps=5, filename_prefix='custom-flux',
                   unet_name='custom-unet.safetensors', vae_name='custom-vae.safetensors',
                   clip_name='custom-clip.safetensors', seed=42,
                   lora='custom-flux-lora.safetensors', lora_strength=0.65)
LTX_CUSTOM = dict(seconds=1.01, fps=23.976, width=640, height=384,
                  filename_prefix='custom-ltx', seed=42, negative='No lettering.', fast=True,
                  checkpoint_name='custom-ckpt.safetensors', text_encoder='custom-te.safetensors',
                  ignored_keyword='intentionally swallowed')
CAMERA = dict(camera_lora='dolly-in', camera_lora_strength=0.45)
IC = dict(ic_loras=[('first-ic.safetensors', 0.7), ('second-ic.safetensors', 0.4)],
          ic_lora_reference_filename='control.png', ic_lora_reference_strength=0.6)
CUSTOM = {}
for name, (family, required) in BUILDERS.items():
    if family == 'flux2':
        custom = dict(FLUX_CUSTOM)
        if name == 'flux2_multiple_angles':
            del custom['width'], custom['height']
        if 'multiprompt' in name or name == 'flux2_multiple_angles':
            custom.update(prepend='Start: ', append=' :end')
    elif family == 'ltx2' and name != 'extract_last_frame':
        custom = dict(LTX_CUSTOM)
        if name != 'ltx2_ingredients_to_video':
            custom.update(CAMERA)
    else:
        custom = {'filename_prefix': 'custom-output'}
    # All required inputs have a named nondefault counterpart, too.
    custom.update({key: (['other-' + v for v in value]
                         if isinstance(value, list) else
                         value + ' Changed.' if key in ('prompt', 'text') else
                         129 if key == 'prev_total_frames' else 'other-' + value)
                   for key, value in required.items()})
    CUSTOM[name] = custom
CUSTOM['qwen_tts'].update(voice_instruct='Quiet, measured delivery.', language='Italian')
CUSTOM['qwen_voice_clone'].update(voice_name='synthetic-voice', language='Italian')
CUSTOM['extract_stems'].update(model_name='custom-stems.safetensors')
CUSTOM['transcribe'].update(model_size='small', language='it')
CUSTOM['concat_videos_ffmpeg'].update(audio_filename='overlay.wav')
CUSTOM['concat_videos'].update(video_filenames=['a.mp4', 'b.mp4', 'c.mp4'], audio_filename='overlay.wav',
                             fps=23.976, trim_durations=[1.01], trim_starts=[0.5, 0.25], format='webm', codec='vp9')
CUSTOM['ltx2_ingredients_to_video'].update(lora_strength=0.9, reference_strength=0.55)
CUSTOM['ltx2_image_audio_to_video'].update(IC, image_refs=['ref-b.png', 'ref-c.png'],
    base_guide_strength=0.65, refine_guide_strength=0.2, identity_anchor_image='identity.png',
    identity_strength=0.25, ic_lora_reference_size=512)
CUSTOM['ltx2_first_last_frame_to_video'].update(IC, first_guide_strength=0.4, last_guide_strength=0.6,
    guide_strength=0.5, audio_filename='overlay.wav', use_transition_lora=True, transition_lora_strength=0.75)
CUSTOM['ltx2_multi_guide_to_video'].update(guide_frame_indices=[0, 16], guide_strengths=[0.4, 0.6],
    audio_filename='overlay.wav', use_transition_lora=False, transition_lora_strength=0.75)
CUSTOM['ltx2_transition'].update(IC, first_guide_strength=0.4, last_guide_strength=0.6,
    audio_filename='overlay.wav', prev_video_filename='scene-a.mp4', next_video_filename='scene-b.mp4',
    multiframe_guide=8, multiframe_guide_last=9, prev_video_start_frame=8, next_video_start_frame=16,
    prev_video_tail_buffer_sec=0.5, slice_song_start_sec=10.0, prev_video_song_start_sec=8.0,
    next_video_song_start_sec=10.0, next_video_vocal_offset_sec=0.25, mask_start_sec=0.25,
    mask_end_sec=0.75, mask_max_length='truncate', use_mask=False, use_inplace=False,
    use_addguide=True, addguide_strength=0.35, b_sparse_latent_positions=[8, 16],
    last_guide_index=16, debug_save_audio=True, ic_lora_frame_idx=8)
CUSTOM['ltx2_continuation_to_video'].update(overlap_seconds=0.75, overlap_strength=0.85,
    audio_filename='overlay.wav', identity_anchor_image='identity.png', identity_strength=0.2)

CASES = []


def add(name, label, *, attention='auto', **overrides):
    family, required = BUILDERS[name]
    CASES.append({'id': name + '/' + label, 'builder': name, 'family': family,
                  'kwargs': {**required, **overrides}, 'attention': attention})


for name in BUILDERS:
    add(name, 'default')
    add(name, 'custom', **CUSTOM[name])
    family = BUILDERS[name][0]
    if family == 'flux2' or (family == 'ltx2' and name != 'extract_last_frame'):
        for seed in (None, 0, 42):
            add(name, 'seed-' + str(seed), seed=seed)
    if family == 'ltx2' and name != 'extract_last_frame':
        add(name, 'fast', fast=True)
        add(name, 'negative-empty', negative='')
        add(name, 'negative-custom-refined', negative='No captions.')
        add(name, 'custom-refined', **{**CUSTOM[name], 'fast': False})
        if name != 'ltx2_ingredients_to_video':
            add(name, 'camera-file', camera_lora='camera-custom.safetensors')
            add(name, 'camera-error', camera_lora='not-a-camera')

for name in ('flux2_multi_reference_edit', 'flux2_multi_reference_edit_multiprompt'):
    for count in (0, 1, 2, 3):
        add(name, f'refs-{count}', image_filenames=['a.png', 'b.png', 'c.png'][:count])
for name in ('flux2_multiple_angles', 'flux2_multi_reference_edit_multiprompt', 'flux2_double_image_edit_multiprompt'):
    add(name, 'blank-prompts', angle_prompts=['', 'front', '', ' ', 'side', ''])
    add(name, 'empty-prompts', angle_prompts=[])
for count in (0, 1, 2, 3):
    add('concat_videos', f'clips-{count}', video_filenames=['a.mp4', 'b.mp4', 'c.mp4'][:count])
add('concat_videos', 'start-only', video_filenames=['a.mp4', 'b.mp4'], trim_starts=[0.75])
add('concat_videos', 'short-trims', video_filenames=['a.mp4', 'b.mp4', 'c.mp4'], trim_durations=[1.01], trim_starts=[0.5], fps=23.976)
add('concat_videos', 'nonpositive-trims', trim_durations=[0], trim_starts=[-1])
for attention in (None, 'auto', 'off', 'false', '0', 'disabled', 'sageattn_qk_int8_pv_fp16_cuda'):
    add('ltx2_text_to_video', 'attention-' + str(attention), attention=attention)

ia = 'ltx2_image_audio_to_video'
add(ia, 'guidance-disabled', base_guide_strength=0, refine_guide_strength=0)
add(ia, 'ic-image-default-size', **IC)
add(ia, 'ic-video', **{**IC, 'ic_lora_reference_filename': 'control.mp4'})
add(ia, 'ic-stack-without-reference', ic_loras=IC['ic_loras'])
add(ia, 'reference-without-stack', ic_lora_reference_filename='control.png')
flf = 'ltx2_first_last_frame_to_video'
add(flf, 'audio-transition-refined', audio_filename='song.wav', use_transition_lora=True)
add(flf, 'ic-video', **{**IC, 'ic_lora_reference_filename': 'control.mov'})
add(flf, 'shared-guide-strength', guide_strength=0.45)
add(flf, 'trigger-present', prompt='Already zhuanchang.', use_transition_lora=True)
mg = 'ltx2_multi_guide_to_video'
add(mg, 'one-guide', guide_filenames=['a.png'])
add(mg, 'three-guides', guide_filenames=['a.png', 'b.png', 'c.png'])
add(mg, 'empty-guides', guide_filenames=[])
add(mg, 'audio-transition', audio_filename='song.wav', transition_lora_strength=0.4)
add(mg, 'duplicate-indices', guide_frame_indices=[8, 8])
add(mg, 'negative-out-of-range', guide_frame_indices=[-8, 800])
add(mg, 'index-count-error', guide_frame_indices=[0])
add(mg, 'index-alignment-error', guide_frame_indices=[0, 7])
add(mg, 'strength-count-error', guide_strengths=[0.4])
tr = 'ltx2_transition'
video = dict(prev_video_filename='scene-a.mp4', next_video_filename='scene-b.mp4')
add(tr, 'video-tail', **video)
add(tr, 'tail-buffer', **video, prev_video_tail_buffer_sec=0.75)
add(tr, 'raw-frames', **video, prev_video_start_frame=16, next_video_start_frame=8)
add(tr, 'song-time', **video, slice_song_start_sec=10.0, prev_video_song_start_sec=8.0,
    next_video_song_start_sec=11.0, next_video_vocal_offset_sec=0.25)
# Keep masks enabled and guide sizes unshadowed: each bound and max_length
# must independently change the graph on both sampling paths.
for fast in (True, False):
    add(tr, 'mask-controls-fast' if fast else 'mask-controls-refined',
        fast=fast, use_mask=True, mask_start_sec=0.5, mask_end_sec=2.5,
        mask_max_length='truncate')
add(tr, 'no-mask', use_mask=False)
add(tr, 'no-inplace', use_inplace=False)
add(tr, 'addguide', use_addguide=True, addguide_strength=0.4)
add(tr, 'sparse-unmapped', **video, use_addguide=True, b_sparse_latent_positions=[24, 48, 64])
add(tr, 'sparse-negative-warning', **video, use_addguide=True, b_sparse_latent_positions=[8, 24, 48, 64],
    slice_song_start_sec=10.0, next_video_song_start_sec=11.0)
add(tr, 'debug-audio', audio_filename='song.wav', debug_save_audio=True)
add(tr, 'ic-video-frame-index', **{**IC, 'ic_lora_reference_filename': 'control.webm'}, ic_lora_frame_idx=24, use_addguide=True)
add('ltx2_continuation_to_video', 'overlap-075-121', overlap_seconds=0.75)
add('ltx2_continuation_to_video', 'audio-only', audio_filename='song.wav', overlap_seconds=0.75)
add('ltx2_continuation_to_video', 'identity-only', identity_anchor_image='identity.png', overlap_seconds=0.75)
assert len({case['id'] for case in CASES}) == len(CASES)
