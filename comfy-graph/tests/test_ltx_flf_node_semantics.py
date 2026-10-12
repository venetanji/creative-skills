"""Execute captured actual node methods without importing ComfyUI/GPU modules."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
from creative_comfy_graph.ltx_profiles import build_flf2v


def node_methods():
    captured = json.loads((Path(__file__).parent / 'fixtures/flf_node_semantics.json').read_text())
    namespace = {'get_keyframe_idxs': lambda cond: (None, cond)}
    for record in captured.values():
        source = record['source']
        assert hashlib.sha256(source.encode()).hexdigest() == record['source_sha256']
        # These methods need only Python slicing and an injected keyframe count.
        exec(compile(ast.parse(source), record['path'], 'exec'), namespace)
    return namespace['imagesfrombatch'], namespace['get_latent_index']


@pytest.mark.parametrize('profile', ['ltx-2.3', 'ltx-2.5'])
@pytest.mark.parametrize('audio', [None, 'sounds/master.wav'])
@pytest.mark.parametrize('batch_size', [1, 9, 17])
def test_actual_range_semantics_select_first_before_resize_and_both_guides(profile, audio, batch_size):
    select, index = node_methods()
    graph = build_flf2v('ball', model_profile=profile, first_frame_filename='animated-first.gif',
        last_frame_filename='animated-last.gif', frames=97, width=768, height=512,
        seed=0, audio_filename=audio)
    guides = [n['inputs'] for n in graph.values() if n['class_type'] == 'LTXVAddGuide']
    assert [n['frame_idx'] for n in guides] == [0, 96, 0, 96]
    range_ids = set()
    batches = []
    for guide in guides:
        preprocess = graph[guide['image'][0]]
        assert preprocess['class_type'] == 'LTXVPreprocess'
        longer = graph[preprocess['inputs']['image'][0]]
        assert longer['class_type'] == 'ResizeImagesByLongerEdge'
        resize = graph[longer['inputs']['images'][0]]
        assert resize['class_type'] == 'ResizeImageMaskNode'
        range_id = resize['inputs']['input'][0]
        span = graph[range_id]
        assert span['class_type'] == 'GetImageRangeFromBatch'
        assert span['inputs']['start_index'] == 0 and span['inputs']['num_frames'] == 1
        loaded = graph[span['inputs']['images'][0]]
        assert loaded['class_type'] == 'LoadImage'
        range_ids.add(range_id)
        batch, masks = select(None, 0, 1, images=list(range(batch_size)))
        assert batch == [0] and masks is None
        batches.append(batch)
    assert len(range_ids) == 2  # Each selected endpoint is reused in pass two.
    for stage in (0, 2):
        for keyframes, guide in enumerate(guides[stage:stage+2]):
            actual = index(None, keyframes, 13 + keyframes, len(batches[stage+keyframes]), guide['frame_idx'], (8, 32, 32))
            assert actual == ((0, 0) if keyframes == 0 else (96, 12))
        assert 13 + sum((len(batches[i]) - 1) // 8 + 1 for i in (stage, stage+1)) == 15


def test_available_addguide_multiframe_counterexample_is_not_single_frame_policy():
    _, index = node_methods()
    assert index(None, 1, 14, 9, 96, (8, 32, 32)) == (89, 12)
    assert index(None, 1, 14, 1, 96, (8, 32, 32)) == (96, 12)
