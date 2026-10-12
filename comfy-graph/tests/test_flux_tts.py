"""Flux/TTS extraction contracts: graph bytes, ambient seeds and pure imports."""
import builtins
import importlib
import json
import sys

import pytest

from test_characterization import capture_module, fixtures


@pytest.mark.parametrize('family', ['flux2', 'tts'])
def test_flux_tts_package_parity(legacy_dir, family):
    tool = capture_module()
    expected_api, expected_graphs, expected_errors = fixtures()
    actual_api, actual_graphs, actual_errors = tool.capture(legacy_dir, package_family=family)
    assert actual_api['builders'] == expected_api['builders']
    assert actual_api['exports'] == expected_api['exports']
    assert json.dumps(actual_graphs) == json.dumps(expected_graphs)
    assert actual_errors == expected_errors
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        package = importlib.import_module('creative_comfy_graph.' + family)
        assert modules[family].WorkflowGraph is package.WorkflowGraph
        for name, (builder_family, _) in tool.BUILDERS.items():
            if builder_family == family:
                assert getattr(modules[family], name) is getattr(package, name)
        if family == 'flux2':
            assert modules[family].upload_if_local is modules['core'].upload_if_local
            assert modules[family].time is package.time
            assert not hasattr(package, 'upload_if_local')


@pytest.mark.parametrize(('name', 'args', 'zero'), [
    ('flux2_text_to_image', ('fixture',), 3487918205),
    ('flux2_multi_reference_edit', (['input.png'], 'fixture'), 3487918205),
    ('flux2_multi_reference_edit_multiprompt', (['input.png'], ['front']), 3487918205),
    ('flux2_multiple_angles', ('input.png', ['front']), 0),
])
def test_flux_seed_zero_quirk(legacy_dir, name, args, zero):
    tool = capture_module()
    with tool.boundaries(legacy_dir):
        builder = getattr(importlib.import_module('creative_comfy_graph.flux2'), name)
        for kwargs, expected in (({}, 3487918205), ({'seed': None}, 3487918205),
                                 ({'seed': 0}, zero), ({'seed': 17}, 17)):
            graph = builder(*args, **kwargs)
            seeds = [n['inputs']['noise_seed'] for n in graph.values() if n['class_type'] == 'RandomNoise']
            assert seeds == [expected]


@pytest.mark.parametrize(('name', 'args', 'prefix', 'refs'), [
    ('flux2_single_image_edit', ('one.png', 'fixture'), 'flux2_i2i', ['one.png']),
    ('flux2_double_image_edit', ('one.png', 'two.png', 'fixture'), 'flux2_i2i2', ['one.png', 'two.png']),
    ('flux2_double_image_edit_multiprompt', ('one.png', 'two.png', ['front', '', 'side']),
     'flux2_i2i2_multi', ['one.png', 'two.png']),
])
def test_flux_wrapper_forwarding(legacy_dir, name, args, prefix, refs):
    tool = capture_module()
    with tool.boundaries(legacy_dir):
        builder = getattr(importlib.import_module('creative_comfy_graph.flux2'), name)
        for extra, expected_prefix in (({}, prefix), ({'filename_prefix': 'custom'}, 'custom')):
            graph = builder(*args, width=640, height=384, steps=11, seed=17,
                            unet_name='custom-unet', vae_name='custom-vae', clip_name='custom-clip',
                            lora='custom-lora', lora_strength=0.6, **extra)
            inputs = lambda kind: [n['inputs'] for n in graph.values() if n['class_type'] == kind]
            assert [n['image'] for n in inputs('LoadImage')] == refs
            assert inputs('EmptyFlux2LatentImage') == [{'width': 640, 'height': 384, 'batch_size': 1}]
            assert inputs('Flux2Scheduler') == [{'steps': 11, 'width': 640, 'height': 384}]
            assert inputs('RandomNoise') == [{'noise_seed': 17}]
            assert inputs('SaveImage')[0]['filename_prefix'] == expected_prefix
            assert inputs('UNETLoader')[0]['unet_name'] == 'custom-unet'
            assert inputs('VAELoader')[0]['vae_name'] == 'custom-vae'
            assert inputs('CLIPLoader')[0]['clip_name'] == 'custom-clip'
            assert inputs('LoraLoader')[0]['lora_name'] == 'custom-lora'
            assert inputs('LoraLoader')[0]['strength_model'] == 0.6
            assert inputs('LoraLoader')[0]['strength_clip'] == 0.6
            if name.endswith('multiprompt'):
                assert inputs('SimplePromptBatcher') == [{'prepend': '', 'prompts': 'front\nside\n', 'append': ''}]
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            builder(*args, unsupported=True)


@pytest.mark.parametrize('voice', ['gio', 'uploads/reference.wav', '/synthetic/unopened.wav'])
def test_tts_voice_filename(legacy_dir, voice):
    tool = capture_module()
    with tool.boundaries(legacy_dir) as attempts:
        module = importlib.import_module('creative_comfy_graph.tts')
        graph = module.qwen_voice_clone('fixture', voice_name=voice, language='English', filename_prefix='custom')
        assert graph['0'] == {'class_type': 'VHS_LoadAudioUpload', 'inputs': {'audio': voice}}
        assert graph['1']['inputs']['voice_name'] == voice
        assert graph['1']['inputs']['reference_audio'] == ['0', 0]
        assert graph['2']['inputs']['target_text'] == 'fixture'
        assert graph['2']['inputs']['language'] == 'English'
        assert graph['3']['inputs']['filename_prefix'] == 'custom'
        assert not attempts


@pytest.mark.parametrize('family', ['flux2', 'tts'])
def test_flux_tts_package_no_transport_import(legacy_dir, monkeypatch, family):
    tool = capture_module()
    name = 'creative_comfy_graph.' + family
    saved = sys.modules.pop(name, None)
    parent = importlib.import_module('creative_comfy_graph')
    previous = vars(parent).pop(family, None)
    original = builtins.__import__
    forbidden = {'core', 'flux2', 'tts', 'post', 'ltx2', 'comfy_graph', 'urllib', 'socket', 'subprocess'}

    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in forbidden, 'package imported transport/adapter: ' + name
        return original(name, *args, **kwargs)

    try:
        with tool.boundaries(legacy_dir) as attempts, monkeypatch.context() as patcher:
            patcher.setattr(builtins, '__import__', guarded)
            module = importlib.import_module(name)
            if family == 'flux2':
                assert module.flux2_text_to_image('offline', seed=17)
            else:
                assert module.qwen_voice_clone('offline', voice_name='server.wav')
            assert not attempts
    finally:
        sys.modules.pop(name, None)
        if saved is not None:
            sys.modules[name] = saved
        vars(parent).pop(family, None)
        if previous is not None:
            setattr(parent, family, previous)
