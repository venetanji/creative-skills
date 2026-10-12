"""LTX-2.3 extraction preserves ordered graphs and existing branch quirks."""
import builtins
import importlib
import inspect
import json
import os
import sys
import time

import pytest

from test_characterization import capture_module, fixtures


def nodes(graph, kind):
    return [(key, node['inputs']) for key, node in graph.items() if node['class_type'] == kind]


def linked(graph, link, kind, output=0):
    assert link[1] == output
    assert graph[link[0]]['class_type'] == kind
    return graph[link[0]]['inputs']


@pytest.mark.parametrize('path', ['legacy', 'package'])
def test_ltx_package_parity(legacy_dir, path):
    tool = capture_module()
    expected_api, expected_graphs, expected_errors = fixtures()
    api, graphs, errors = tool.capture(legacy_dir, package_family='ltx2' if path == 'package' else None)
    assert api['builders'] == expected_api['builders']
    assert api['exports'] == expected_api['exports']
    assert api['constants'] == expected_api['constants']
    assert json.dumps(graphs) == json.dumps(expected_graphs)
    assert errors == expected_errors


def test_ltx_signature_constants(legacy_dir):
    tool = capture_module()
    api, _, _ = fixtures()
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        package = importlib.import_module('creative_comfy_graph.ltx2')
        legacy = modules['ltx2']
        assert [name for name in vars(legacy) if not name.startswith('_')] == api['exports']['ltx2']
        assert [name for name in vars(package) if not name.startswith('_')] == api['exports']['ltx2']
        for name in api['exports']['ltx2']:
            assert getattr(legacy, name) is getattr(package, name), name
        for name, (family, _) in tool.BUILDERS.items():
            if family == 'ltx2':
                assert tool.api_record(getattr(package, name)) == api['builders'][name]
        assert {name: getattr(package, name) for name in tool.CONSTANTS['ltx2']} == api['constants']['ltx2']
        assert package.WorkflowGraph is modules['core'].WorkflowGraph
        assert package.os is os and package.sys is sys and package.time is time


@pytest.mark.parametrize('path', ['legacy', 'package'])
def test_ltx_ambient_and_kwargs(legacy_dir, monkeypatch, path):
    tool = capture_module()
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        module = modules['ltx2'] if path == 'legacy' else importlib.import_module('creative_comfy_graph.ltx2')
        for name, (family, required) in tool.BUILDERS.items():
            if family != 'ltx2' or name == 'extract_last_frame':
                continue
            builder = getattr(module, name)
            assert inspect.signature(builder).parameters['_'].kind is inspect.Parameter.VAR_KEYWORD
            for seed_kwargs, seed in (({}, 1340434557), ({'seed': None}, 1340434557),
                                      ({'seed': 0}, 1340434557), ({'seed': 42}, 42)):
                graph = builder(**required, **seed_kwargs)
                assert json.dumps(builder(**required, **seed_kwargs, unknown_option=object())) == json.dumps(graph)
                samplers = nodes(graph, 'SamplerCustomAdvanced')
                assert [linked(graph, inputs['noise'], 'RandomNoise')['noise_seed']
                        for _, inputs in samplers] == [seed, seed + 1]
        for mode, loader, value in ((None, 'CheckpointLoaderKJ', 'auto'),
                                    (' AUTO ', 'CheckpointLoaderKJ', 'auto'),
                                    (' OFF ', 'CheckpointLoaderSimple', None),
                                    ('false', 'CheckpointLoaderSimple', None),
                                    ('0', 'CheckpointLoaderSimple', None),
                                    ('disabled', 'CheckpointLoaderSimple', None),
                                    ('sageattn_qk_int8_pv_fp16_cuda', 'CheckpointLoaderKJ', 'sageattn_qk_int8_pv_fp16_cuda')):
            if mode is None:
                monkeypatch.delenv('LTX_SAGE_ATTENTION', raising=False)
            else:
                monkeypatch.setenv('LTX_SAGE_ATTENTION', mode)
            graph = module.ltx2_text_to_video('offline', fast=True)
            assert graph['0']['class_type'] == loader
            assert graph['0']['inputs'].get('sage_attention') == value


@pytest.mark.parametrize('path', ['legacy', 'package'])
@pytest.mark.parametrize('fast', [True, False], ids=['fast', 'refined'])
def test_ltx_audio_guides_transition(legacy_dir, path, fast):
    tool = capture_module()
    _, golden, _ = fixtures()
    with tool.boundaries(legacy_dir) as attempts, tool.legacy_modules(legacy_dir) as modules:
        module = modules['ltx2'] if path == 'legacy' else importlib.import_module('creative_comfy_graph.ltx2')
        for name, (family, _) in tool.BUILDERS.items():
            if family != 'ltx2' or name == 'extract_last_frame':
                continue
            label = name + ('/custom' if fast else '/custom-refined')
            case = next(case for case in tool.CASES if case['id'] == label)
            graph = getattr(module, name)(**case['kwargs'])
            assert json.dumps(graph) == json.dumps(golden[label]['graph'])
            _, empty = nodes(graph, 'EmptyLTXVLatentVideo')[0]
            assert empty == {'width': 640, 'height': 384, 'length': 25, 'batch_size': 1}
            assert empty['length'] % 8 == 1
            samplers = nodes(graph, 'SamplerCustomAdvanced')
            assert [linked(graph, inputs['noise'], 'RandomNoise')['noise_seed']
                    for _, inputs in samplers] == ([42] if fast else [42, 43])
            upscalers = nodes(graph, 'LTXVLatentUpsampler')
            assert len(upscalers) == (0 if fast else 1)
            if not fast:
                up_id, up = upscalers[0]
                if name in ('ltx2_text_to_video', 'ltx2_ingredients_to_video',
                            'ltx2_image_to_video', 'ltx2_image_audio_to_video', 'ltx2_transition'):
                    crop = linked(graph, up['samples'], 'LTXVCropGuides', 2)
                    sep = linked(graph, crop['latent'], 'LTXVSeparateAVLatent')
                    assert sep['av_latent'] == [samplers[0][0], 0]
                    assert list(graph).index(up['samples'][0]) < list(graph).index(up_id)
                else:
                    # These legacy branches crop AFTER upscaling, before re-guiding.
                    sep = linked(graph, up['samples'], 'LTXVSeparateAVLatent')
                    assert sep['av_latent'] == [samplers[0][0], 0]
                    crops = [(key, inputs) for key, inputs in nodes(graph, 'LTXVCropGuides')
                             if inputs['latent'] == [up_id, 0]]
                    assert len(crops) == 1
                    assert any(inputs['latent'] == [crops[0][0], 2]
                               for _, inputs in nodes(graph, 'LTXVAddGuide'))
            _, created = nodes(graph, 'CreateVideo')[0]
            assert created['fps'] == 23.976
            decode = linked(graph, created['images'], 'VAEDecodeTiled')
            assert decode['tile_size'] == 768 and decode['temporal_size'] == 4096
            if name in ('ltx2_image_audio_to_video', 'ltx2_first_last_frame_to_video',
                        'ltx2_multi_guide_to_video', 'ltx2_continuation_to_video'):
                audio = case['kwargs']['audio_filename']
                trim = linked(graph, created['audio'], 'TrimAudioDuration')
                assert trim['duration'] == 25 / 23.976 and trim['start_index'] == 0
                assert linked(graph, trim['audio'], 'LoadAudio')['audio'] == audio
                _, encoded = nodes(graph, 'LTXVAudioVAEEncode')[0]
                conditioning_trim = linked(graph, encoded['audio'], 'TrimAudioDuration')
                assert conditioning_trim['duration'] == trim['duration']
                assert linked(graph, conditioning_trim['audio'], 'LoadAudio')['audio'] == audio
                assert encoded['audio'] != created['audio']
                assert not nodes(graph, 'LTXVAudioVAEDecode')
            elif name == 'ltx2_transition':
                # Transition currently muxes VAE output, not the supplied source.
                linked(graph, created['audio'], 'LTXVAudioVAEDecode')
                _, encoded = nodes(graph, 'LTXVAudioVAEEncode')[0]
                trim = linked(graph, encoded['audio'], 'TrimAudioDuration')
                assert trim['duration'] == 25 / 23 and trim['start_index'] == 0
                assert linked(graph, trim['audio'], 'LoadAudio')['audio'] == 'overlay.wav'
                noise_id, noise = nodes(graph, 'AddNoise')[0]
                linked(graph, noise['latent_image'], 'EmptyLTXVLatentVideo')
                assert linked(graph, noise['noise'], 'RandomNoise')['noise_seed'] == 10042
                ic_id, ic = nodes(graph, 'LTXAddVideoICLoRAGuide')[0]
                assert ic['latent'] == [noise_id, 0] and ic['frame_idx'] == 8
                assert list(graph).index(noise_id) < list(graph).index(ic_id)
                assert any(inputs['latent'] == [ic_id, 2] and inputs['positive'] == [ic_id, 0]
                           for _, inputs in nodes(graph, 'LTXVAddGuide'))
            else:
                linked(graph, created['audio'], 'LTXVAudioVAEDecode')
            save_id, save = nodes(graph, 'SaveVideo')[0]
            assert save == {'video': [nodes(graph, 'CreateVideo')[0][0], 0],
                            'filename_prefix': 'custom-ltx', 'format': 'auto', 'codec': 'auto'}
            assert list(graph)[-1] == save_id
        assert not attempts


def test_ltx_package_no_transport_import(legacy_dir, monkeypatch):
    tool = capture_module()
    name = 'creative_comfy_graph.ltx2'
    saved = sys.modules.pop(name, None)
    parent = importlib.import_module('creative_comfy_graph')
    previous = vars(parent).pop('ltx2', None)
    original = builtins.__import__
    forbidden = {'core', 'flux2', 'tts', 'post', 'ltx2', 'comfy_graph', 'urllib', 'socket', 'subprocess', 'pathlib'}

    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in forbidden, 'package imported transport/adapter: ' + name
        return original(name, *args, **kwargs)

    try:
        with tool.boundaries(legacy_dir) as attempts, monkeypatch.context() as patcher:
            patcher.setattr(builtins, '__import__', guarded)
            module = importlib.import_module(name)
            for name, (family, required) in tool.BUILDERS.items():
                if family == 'ltx2':
                    assert getattr(module, name)(**required)
            assert not attempts
    finally:
        sys.modules.pop('creative_comfy_graph.ltx2', None)
        if saved is not None:
            sys.modules['creative_comfy_graph.ltx2'] = saved
        vars(parent).pop('ltx2', None)
        if previous is not None:
            setattr(parent, 'ltx2', previous)
