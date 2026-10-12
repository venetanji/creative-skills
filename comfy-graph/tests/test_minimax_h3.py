"""Native FL eight-step admission: independent retained graph/schema fixtures."""
import importlib
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / 'fixtures'


def build(*args, **kwargs):
    module = importlib.util.find_spec('creative_comfy_graph.minimax_h3')
    assert module is not None, 'pure native H3 builder is not implemented'
    return importlib.import_module('creative_comfy_graph.minimax_h3').build_t2v(*args, **kwargs)


def test_matches_retained_native_prototype_exactly():
    prototype = json.loads((FIXTURES / 'h3_t2v_prototype.json').read_text())
    graph = build(prototype['6']['inputs']['prompt'], seed=20261002,
                  filename_prefix='video/minimax_h3_bell_20261002')
    assert graph == prototype


def test_default_seed_zero_and_joint_decode_no_reference_inputs():
    graph = build('One bell strike. Audio: clear bell resonance.', seed=0)
    assert graph['8']['inputs']['noise_seed'] == 0
    assert build('One bell strike. Audio: clear bell resonance.') == graph
    assert graph['6']['inputs'] == dict(clip=['3', 0], vae=['4', 0],
        prompt='One bell strike. Audio: clear bell resonance.', width=864, height=480, length=124)
    assert graph['12']['inputs']['samples'] == graph['13']['inputs']['samples'] == ['11', 0]
    assert graph['14']['inputs'] == dict(images=['12', 0], audio=['13', 0], fps=24.0, bit_depth=8)
    assert not any(n['class_type'].startswith(('LTX', 'LoadImage', 'LoadAudio')) for n in graph.values())


@pytest.mark.parametrize('kwargs', [
    {'width': True}, {'height': False}, {'width':864.0}, {'width':'864'},
    {'width':float('inf')}, {'height':float('nan')}, {'width':0}, {'height':-32},
    {'width':865}, {'height':481}, {'width':16416}, {'height':16416},
    {'frames':True}, {'frames':124.0}, {'frames':'124'}, {'frames':123},
    {'frames':125}, {'frames':107}, {'frames':379}, {'frames':float('nan')},
    {'seed':True}, {'seed':0.0}, {'seed':'0'}, {'seed':-1}, {'seed':2**64},
    {'filename_prefix':''}, {'filename_prefix':'../escape'}, {'filename_prefix':'/absolute'},
    {'filename_prefix':'a/../b'}, {'filename_prefix':'a\\b'}, {'filename_prefix':'%date:yyyy%'},
    {'filename_prefix':'a//b'}, {'filename_prefix':'a\x00b'}, {'filename_prefix':True}, {'filename_prefix':'x'*201},
])
def test_rejects_invalid_admission(kwargs):
    with pytest.raises(ValueError):
        build('bell', **kwargs)


@pytest.mark.parametrize('prompt', ['', '  ', None, True, [], 'a\x00b'])
def test_rejects_invalid_prompt(prompt):
    with pytest.raises(ValueError):
        build(prompt)


@pytest.mark.parametrize('kwargs', [
    {'audio_filename':'local.wav'}, {'image_filename':'local.png'}, {'first_frame':'x'},
    {'last_frame':'x'}, {'references':['x']}, {'control_video':'x'}, {'negative':'bad'},
    {'cfg':1}, {'steps':4}, {'sampler':'euler_ancestral'}, {'model_profile':'ltx-2.5'},
    {'unet_name':'/unsafe/model'}, {'lora_name':'other'}, {'fps':25}, {'seconds':5},
])
def test_rejects_unimplemented_combinations(kwargs):
    with pytest.raises((TypeError, ValueError)):
        build('bell', **kwargs)


@pytest.mark.parametrize('frames', [124,141,158,175,192,209,226,243,260,277,294,311,328,345,362])
def test_explicit_native_grid_is_not_silently_snapped(frames):
    graph = build('bell', frames=frames)
    assert graph['6']['inputs']['length'] == frames


@pytest.mark.parametrize('kwargs', [
    {'width':32,'height':32}, {'width':16384,'height':16384}, {'seed':2**64-1},
    {'filename_prefix':'x'*200},
    {'width':896,'height':512,'frames':141,'seed':1,'filename_prefix':'video/fixture_1'},
])
def test_static_bounds_not_claim_of_gpu_feasibility(kwargs):
    graph = build('bell', **kwargs)
    for key in ('width', 'height'):
        if key in kwargs:
            assert graph['6']['inputs'][key] == kwargs[key]
    if 'seed' in kwargs:
        assert graph['8']['inputs']['noise_seed'] == kwargs['seed']


def test_all_nodes_admit_to_captured_native_schema_and_reach_output():
    graph = build('bell')
    schemas = json.loads((FIXTURES / 'h3_native_schema.json').read_text())
    visited = set()
    def visit(nid, ancestors):
        assert nid not in ancestors, 'cycle'
        node = graph[nid]
        schema = schemas[node['class_type']]
        assert not schema.get('api_node', False)
        fields = schema['input']
        allowed = dict(fields.get('required', {}), **fields.get('optional', {}))
        assert set(fields.get('required', {})) <= node['inputs'].keys()
        assert node['inputs'].keys() <= allowed.keys()
        for name, value in node['inputs'].items():
            spec = allowed[name]
            kind, options = spec[0], spec[1] if len(spec) > 1 else {}
            if isinstance(value, list):
                source, output = value
                assert schemas[graph[source]['class_type']]['output'][output] == kind
                visit(source, ancestors | {nid})
            elif isinstance(kind, list):
                assert value in kind
            elif kind in {'COMBO', 'COMFY_DYNAMICCOMBO_V3'}:
                choices = [v['key'] if isinstance(v, dict) else v for v in options['options']]
                assert value in choices
            elif kind in {'INT', 'FLOAT'}:
                if kind == 'INT':
                    assert type(value) is int
                assert options.get('min', value) <= value <= options.get('max', value)
            elif kind == 'STRING':
                assert isinstance(value, str)
        visited.add(nid)
    visit('15', set())
    assert visited == set(graph)
