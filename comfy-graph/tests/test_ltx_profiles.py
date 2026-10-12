import pytest
from creative_comfy_graph.ltx_profiles import build_video, continuation_timeline


def nodes(graph, kind):
    return [n['inputs'] for n in graph.values() if n['class_type'] == kind]


def test_real_profile_and_final_geometry():
    g = build_video('ball', model_profile='ltx-2.5', frames=49, width=512, height=320, seed=0)
    assert nodes(g, 'UNETLoader')[0]['unet_name'].startswith('ltx-2.5-22b-distilled-transformer')
    assert nodes(g, 'CLIPLoader')[0]['clip_name'].startswith('gemma4-12b')
    assert len(nodes(g, 'VAELoader')) == 2
    assert not nodes(g, 'LoraLoaderModelOnly')
    assert len(nodes(g, 'LTXVDualCFGGuider')) == 2
    assert {n['sampler_name'] for n in nodes(g, 'KSamplerSelect')} == {'euler_ancestral'}
    assert nodes(g, 'EmptyLTXVLatentVideo')[0]['width'] == 256
    assert nodes(g, 'EmptyLTXVLatentVideo')[0]['height'] == 160
    assert nodes(g, 'RandomNoise')[0]['noise_seed'] == 0


def test_exact_overlap_and_source_spans():
    t = continuation_timeline(previous_frames=49, frames=49, fps=24, overlap_seconds=1)
    assert t['overlap_frames'] == 25
    assert t['guide_span'] == [24, 49]
    assert t['source_span'] == [24, 73]
    assert t['retain_span'] == [25, 49]
    assert t['audio_start_sample_48k'] == 48000
    assert continuation_timeline(previous_frames=57, previous_retained_end=49, frames=49)['guide_span'] == [24,49]


@pytest.mark.parametrize('kw', [{'overlap_frames':24}, {'previous_frames':17}, {'frames':25}, {'previous_retained_end':60}])
def test_invalid_tail(kw):
    args = dict(previous_frames=49, frames=49, fps=24)
    args.update(kw)
    with pytest.raises(ValueError):
        continuation_timeline(**args)


def test_crop_before_upscale_and_audio_offset():
    g = build_video('ball', model_profile='ltx-2.5', frames=49, width=512, height=320,
                    prev_video_filename='A.mp4', previous_frames=49, audio_filename='master.wav', audio_start_frame=24)
    assert nodes(g, 'GetImageRangeFromBatch')[0]['start_index'] == 24
    assert nodes(g, 'GetImageRangeFromBatch')[0]['num_frames'] == 25
    assert len(nodes(g, 'LTXVCropGuides')) == 2
    up = nodes(g, 'LTXVLatentUpsampler')[0]
    assert g[up['samples'][0]]['class_type'] == 'LTXVCropGuides'
    assert nodes(g, 'TrimAudioDuration')[0]['start_index'] == 1.0
    assert nodes(g, 'TrimAudioDuration')[0]['duration'] == 49/24
    assert nodes(g, 'SolidMask')[0]['value'] == 0
    assert len(nodes(g, 'SetLatentNoiseMask')) == 2
    assert not nodes(g, 'LTXVAudioVAEDecode')


@pytest.mark.parametrize('kw', [{'camera_lora':'static'}, {'ic_loras':['x']}, {'fast':True}, {'model_profile':'bogus'}, {'width':500}, {'frames':48}, {'workflow':'transition'}])
def test_reject_unsupported(kw):
    args=dict(model_profile='ltx-2.5', frames=49, width=512, height=320)
    args.update(kw)
    with pytest.raises(ValueError):
        build_video('ball', **args)


def test_explicit_old_profile_and_image_topology():
    old = build_video('ball', model_profile='ltx-2.3')
    assert nodes(old, 'CheckpointLoaderKJ') or nodes(old, 'CheckpointLoaderSimple')
    assert nodes(old, 'LoraLoaderModelOnly')
    image = build_video('ball', model_profile='ltx-2.5', image_filename='anchor.png')
    assert len(nodes(image, 'LTXVImgToVideoInplace')) == 2
    assert nodes(image, 'LoadImage')[0]['image'] == 'anchor.png'


def test_audio_offset_matches_source_span():
    with pytest.raises(ValueError, match='source start'):
        build_video('ball', model_profile='ltx-2.5', prev_video_filename='A.mp4',
                    previous_frames=49, audio_filename='master.wav', audio_start_frame=49)


@pytest.mark.parametrize('kw', [{'workflow':'i2v'}, {'workflow':'continuation'},
                               {'workflow':'t2v','image_filename':'x'}, {'audio_start_frame':24}])
def test_reject_conflicting_workflow(kw):
    with pytest.raises(ValueError):
        build_video('ball',model_profile='ltx-2.5',**kw)
