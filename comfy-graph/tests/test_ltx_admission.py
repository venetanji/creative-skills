"""Captured-schema admission regressions; no backend discovery or submissions."""
import pytest
from creative_comfy_graph.ltx_profiles import build_video


@pytest.mark.parametrize('kw', [
    {'width':64}, {'height':64}, {'width':32832}, {'height':32832},
    {'width':16448,'image_filename':'x'}, {'height':16448,'prev_video_filename':'x','previous_frames':49},
    {'frames':1001}, {'frames':16385,'audio_filename':'x'}, {'fps':121},
    {'seed':-1}, {'seed':2**64-1}, {'seed':True}, {'seed':1.5},
    {'prev_video_filename':'x','previous_frames':4130},
    {'prev_video_filename':'x','previous_frames':4105,'frames':4113,'overlap_frames':4105,'audio_filename':'x'},
])
def test_known_backend_bounds(kw):
    with pytest.raises(ValueError):
        build_video('ball', model_profile='ltx-2.5', **kw)


@pytest.mark.parametrize('kw', [
    {'width':128,'height':128}, {'width':32768,'height':32768},
    {'width':16384,'image_filename':'x'}, {'frames':993},
    {'frames':16377,'audio_filename':'x'}, {'fps':120}, {'seed':2**64-2},
    {'prev_video_filename':'x','previous_frames':4121},
])
def test_static_boundary_values_not_live_certification(kw):
    build_video('ball', model_profile='ltx-2.5', **kw)


@pytest.mark.parametrize('kw', [
    {'base_guide_strength':.7}, {'refine_guide_strength':1},
    {'overlap_strength':1,'image_filename':'x'},
    {'base_guide_strength':.1,'prev_video_filename':'x','previous_frames':49},
    {'refine_guide_strength':.2,'prev_video_filename':'x','previous_frames':49},
])
def test_reject_inapplicable_guide_controls(kw):
    with pytest.raises(ValueError, match='guide|continuation'):
        build_video('ball', **kw)


@pytest.mark.parametrize('kw', [
    {'previous_frames':49}, {'previous_retained_end':49},
    {'previous_source_start':0}, {'overlap_seconds':1.0},
    {'overlap_frames':25}, {'overlap_strength':1.0},
])
def test_reject_orphan_timing_even_explicit_defaults(kw):
    with pytest.raises(ValueError, match='continuation'):
        build_video('ball', audio_filename='master.wav', **kw)


def test_every_accepted_strength_changes_graph():
    for key in ('base_guide_strength','refine_guide_strength'):
        assert build_video('ball',seed=0,image_filename='x',**{key:.1}) != build_video('ball',seed=0,image_filename='x',**{key:.9})
    assert build_video('ball',seed=0,prev_video_filename='x',previous_frames=49,overlap_strength=.1) != build_video('ball',seed=0,prev_video_filename='x',previous_frames=49,overlap_strength=.9)
