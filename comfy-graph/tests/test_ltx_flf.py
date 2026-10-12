"""Pure endpoint topology: no backend or generated media required."""
import pytest
from creative_comfy_graph import ltx_profiles as p


def build(**kw):
    return p.build_flf2v('ball', first_frame_filename='first.png',
                         last_frame_filename='last.png', seed=0, **kw)


def nodes(g, kind):
    return [n['inputs'] for n in g.values() if n['class_type'] == kind]


@pytest.mark.parametrize('profile', p.PROFILES)
@pytest.mark.parametrize('frames,width,height',[(49,512,320),(97,768,512)])
def test_flf_endpoint_chains_crop_and_geometry(profile,frames,width,height):
    g=build(model_profile=profile,frames=frames,width=width,height=height,
            first_guide_strength=.4,last_guide_strength=.9)
    guides=nodes(g,'LTXVAddGuide')
    assert [n['frame_idx'] for n in guides]==[0,frames-1,0,frames-1]
    assert [n['strength'] for n in guides]==[.4,.9,.4,.9]
    assert len(nodes(g,'LoadImage'))==2 and len(nodes(g,'LTXVCropGuides'))==2
    for kind in ('LTXVLatentUpsampler','VAEDecodeTiled'):
        assert g[nodes(g,kind)[0]['samples'][0]]['class_type']=='LTXVCropGuides'
    assert nodes(g,'EmptyLTXVLatentVideo')[0]==dict(width=width//2,height=height//2,length=frames,batch_size=1)
    assert [n['noise_seed'] for n in nodes(g,'RandomNoise')]==[0,1]
    for stage in (0,2):
        first,last=guides[stage:stage+2]
        assert last['positive'][0]==last['negative'][0]==last['latent'][0]
        assert g[last['latent'][0]]['inputs']==first
        assert g[first['image'][0]]['class_type']=='LTXVPreprocess'
    assert bool(nodes(g,'LTXVDualCFGGuider'))==(profile=='ltx-2.5')
    assert bool(nodes(g,'LoraLoaderModelOnly'))==(profile=='ltx-2.3')


def test_defaults_and_strength_changes_affect_both_passes():
    g=build()
    assert [n['strength'] for n in nodes(g,'LTXVAddGuide')]==[.7,.7,.7,.7]
    assert build(first_guide_strength=.2)!=g and build(last_guide_strength=.2)!=g


def test_source_audio_frozen_twice_and_original_mux_offset_zero():
    g=build(model_profile='ltx-2.5',audio_filename='master.wav')
    assert nodes(g,'TrimAudioDuration')[0]['start_index']==0
    assert nodes(g,'TrimAudioDuration')[0]['duration']==49/24
    assert nodes(g,'SolidMask')[0]['value']==0
    masks=nodes(g,'SetLatentNoiseMask')
    assert len(masks)==2 and masks[0]['samples']==masks[1]['samples']
    assert g[nodes(g,'CreateVideo')[0]['audio'][0]]['class_type']=='TrimAudioDuration'
    assert not nodes(g,'LTXVAudioVAEDecode')


@pytest.mark.parametrize('kw',[dict(first_frame_filename=''),dict(last_frame_filename=None),
    dict(first_frame_filename=[]),dict(last_frame_filename=True),dict(audio_filename=[]),
    dict(prompt=1),dict(negative=1),dict(filename_prefix=False),dict(model_profile=[]),
    dict(frames=48),dict(frames=True),dict(seed=True),dict(width=64),
    dict(first_guide_strength=True),dict(last_guide_strength=float('nan')),
    dict(first_guide_strength=-.1),dict(last_guide_strength=1.1),dict(fast=True),
    dict(image_filename='other'),dict(prev_video_filename='other'),dict(audio_start_frame=0),
    dict(base_guide_strength=.7),dict(previous_source_start=0),dict(camera_lora='x'),
    dict(ckpt='x'),dict(refine_guide_strength=1)])
def test_invalid_conflicting_inputs(kw):
    args=dict(prompt='ball',first_frame_filename='f.png',last_frame_filename='l.png',model_profile='ltx-2.5');args.update(kw)
    with pytest.raises((ValueError,TypeError)):p.build_flf2v(**args)


def test_endpoint_index_with_long_audio_is_backend_bounded():
    with pytest.raises(ValueError):build(audio_filename='master.wav',frames=10009)


@pytest.mark.parametrize('kw',[dict(image_filename=''),dict(prev_video_filename=''),
    dict(audio_start_frame=0),dict(base_guide_strength=0),dict(refine_guide_strength=0),
    dict(previous_frames=49),dict(overlap_seconds=1)])
def test_shared_flf_preflight_cannot_bypass_public_contract(kw):
    with pytest.raises(ValueError):
        p.build_video('ball',workflow='flf2v',first_frame_filename='f',last_frame_filename='l',**kw)


@pytest.mark.parametrize('kw',[dict(first_frame_filename=''),dict(last_frame_filename=''),
    dict(first_frame_filename=None),dict(last_frame_filename=None)])
def test_shared_requires_actual_two_endpoints(kw):
    args=dict(workflow='flf2v',first_frame_filename='f',last_frame_filename='l');args.update(kw)
    with pytest.raises(ValueError):p.build_video('ball',**args)


def test_empty_single_image_does_not_hide_inapplicable_strength():
    with pytest.raises(ValueError):
        p.build_video('ball',image_filename='',base_guide_strength=0)


def test_profile_file_inputs_cannot_silently_be_empty():
    with pytest.raises(ValueError):p.build_video('ball',audio_filename='')
