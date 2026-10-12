"""Pinned contracts detect builder/API drift, ordering drift and unsafe I/O."""
import importlib.util
import json
from pathlib import Path

import pytest

TESTS = Path(__file__).parent


def capture_module():
    path = TESTS / 'tools' / 'capture_baseline.py'
    assert path.is_file(), 'offline capture implementation is missing'
    spec = importlib.util.spec_from_file_location('capture_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures():
    files = [TESTS / 'fixtures' / (name + '.json') for name in ('api', 'graphs', 'errors')]
    assert all(p.is_file() for p in files), 'immutable baseline fixtures are missing'
    return tuple(json.loads(p.read_text()) for p in files)


def test_baseline_manifest_complete():
    api, graphs, errors = fixtures()
    from cases import BUILDERS, CASES
    assert len(BUILDERS) == 22
    assert set(api['builders']) == set(BUILDERS)
    assert set(graphs) | set(errors) == {case['id'] for case in CASES}
    assert set(graphs).isdisjoint(errors)
    for name in BUILDERS:
        assert name + '/default' in graphs
        assert name + '/custom' in graphs


def test_graphs_match_baseline(legacy_dir, expected_families):
    tool = capture_module()
    _, expected, errors = fixtures()
    _, actual, actual_errors = tool.capture(legacy_dir)
    assert actual == expected
    assert json.dumps(actual) == json.dumps(expected), 'graph/input insertion order changed'
    assert actual_errors == errors
    for family in sorted(expected_families - {'primitives'}):
        _, actual, actual_errors = tool.capture(legacy_dir, package_family=family)
        assert actual == expected
        assert json.dumps(actual) == json.dumps(expected)
        assert actual_errors == errors


def test_signatures_match_baseline(legacy_dir, expected_families):
    tool = capture_module()
    expected, _, _ = fixtures()
    actual, _, _ = tool.capture(legacy_dir)
    assert actual['builders'] == expected['builders']
    assert actual['primitives'] == expected['primitives']
    assert actual['exports'] == expected['exports']
    assert actual['constants'] == expected['constants']
    for family in sorted(expected_families - {'primitives'}):
        actual, _, _ = tool.capture(legacy_dir, package_family=family)
        assert actual['builders'] == expected['builders']
        assert actual['constants'] == expected['constants']


def test_capture_boundaries(legacy_dir, tmp_path):
    tool = capture_module()
    evidence = tool.exercise_boundaries(legacy_dir, tmp_path)
    assert len(evidence) >= 20
    assert all(value == 'blocked' for value in evidence.values())
    assert not list(tmp_path.iterdir())


def test_parameter_coverage(legacy_dir):
    tool = capture_module()
    assert tool.uncovered_parameters(legacy_dir) == {}


def test_capture_stable_recapture(legacy_dir):
    tool = capture_module()
    first = tool.capture(legacy_dir)
    second = tool.capture(legacy_dir)
    assert tool.serialize(first) == tool.serialize(second)


def test_capture_restores_modules_and_environment(legacy_dir):
    import os
    import sys
    import time
    tool = capture_module()
    before = {name: sys.modules.get(name) for name in ('core', 'flux2', 'tts', 'post', 'ltx2')}
    env = dict(os.environ)
    clock, bytecode = time.time, sys.dont_write_bytecode
    tool.capture(legacy_dir)
    assert before == {name: sys.modules.get(name) for name in before}
    assert dict(os.environ) == env
    assert time.time is clock
    assert sys.dont_write_bytecode == bytecode


def test_baseline_provenance_and_expected_errors(legacy_dir):
    tool = capture_module()
    api, graphs, errors = fixtures()
    assert api['provenance']['revision'] == tool.PIN
    assert api['provenance']['source_sha256'] == tool.SOURCE_HASHES
    assert len(errors) == 13
    assert all(error['type'] == 'ValueError' for error in errors.values())
    warning = graphs['ltx2_transition/sparse-negative-warning']['stderr']
    assert 'src_frame=-16' in warning and 'skipping this anchor' in warning


def test_capture_import_boundaries_and_allowed_io(legacy_dir, tmp_path):
    tool = capture_module()
    probe = tmp_path / 'legacy'
    probe.mkdir()
    for name in ('core', 'flux2', 'tts', 'post', 'ltx2'):
        (probe / (name + '.py')).write_text((legacy_dir / (name + '.py')).read_text())
    original = (probe / 'core.py').read_text()
    attacks = ["open('/synthetic/media.png', 'rb')",
               "Path('/synthetic/.openclaw/config.json').exists()",
               "urllib.request.urlopen('http://invalid.test')",
               "subprocess.run(['must-not-launch'])"]
    for attack in attacks:
        (probe / 'core.py').write_text(original + '\n' + attack + '\n')
        with pytest.raises(tool.BoundaryViolation):
            tool.capture(probe)
    output = tmp_path / 'allowed.json'
    with tool.boundaries(legacy_dir, write_paths=[output]):
        output.write_text('{}\n')
        assert (legacy_dir / 'core.py').read_text().startswith('"""Core')
        with pytest.raises(tool.BoundaryViolation):
            (tmp_path / 'not-allowed.json').write_text('{}')
    assert output.read_text() == '{}\n'


def test_capture_boundary_low_level_import_reads(legacy_dir, tmp_path):
    import _io
    import os
    tool = capture_module()
    media = tmp_path / 'media.png'
    with tool.boundaries(legacy_dir):
        with pytest.raises(tool.BoundaryViolation):
            _io.open_code(str(media))
        with pytest.raises(tool.BoundaryViolation):
            os.readlink(media)


def test_parameter_coverage_includes_required_arguments(legacy_dir):
    tool = capture_module()
    for name, (_, required) in tool.BUILDERS.items():
        custom = next(case['kwargs'] for case in tool.CASES if case['id'] == name + '/custom')
        assert all(custom[key] != value for key, value in required.items()), name


def test_baseline_primitive_api_inventory():
    api, _, _ = fixtures()
    assert set(api['primitives']) == {
        'NodeRef', 'NodeRef.__getitem__', 'NodeRef.as_link', 'NodeRef.__repr__',
        'WorkflowGraph', 'WorkflowGraph.node', 'WorkflowGraph.to_dict', 'WorkflowGraph.to_json',
    }


@pytest.mark.parametrize('fast', [True, False], ids=['fast', 'refined'])
def test_transition_nondefault_mask_controls(legacy_dir, fast):
    tool = capture_module()
    label = 'mask-controls-fast' if fast else 'mask-controls-refined'
    case = next((case for case in tool.CASES
                 if case['id'] == 'ltx2_transition/' + label), None)
    assert case is not None, 'enabled nondefault mask characterization is missing'
    _, graphs, _ = fixtures()
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        builder = modules['ltx2'].ltx2_transition
        kwargs = case['kwargs']
        assert kwargs['use_mask'] is True and kwargs['fast'] is fast
        actual = builder(**kwargs)
        assert json.dumps(actual) == json.dumps(graphs[case['id']]['graph'])
        for control in ('mask_start_sec', 'mask_end_sec', 'mask_max_length'):
            omitted = {key: value for key, value in kwargs.items() if key != control}
            assert json.dumps(builder(**omitted)) != json.dumps(actual), control
        masks = [node['inputs'] for node in actual.values()
                 if node['class_type'] == 'LTXVAudioVideoMask']
        assert len(masks) == (1 if fast else 2)
        assert all(mask['max_length'] == 'truncate' for mask in masks)


@pytest.mark.parametrize('root_name', ['prefix', 'base_prefix'])
@pytest.mark.parametrize('operation', ['exists', 'stat', 'lstat', 'readlink', 'listdir', 'scandir'])
def test_capture_media_metadata_under_interpreter_roots(legacy_dir, root_name, operation):
    import os
    import sys
    tool = capture_module()
    media = Path(getattr(sys, root_name)) / 'synthetic-client-media.png'
    action = media.exists if operation == 'exists' else lambda: getattr(os, operation)(media)
    with tool.boundaries(legacy_dir):
        with pytest.raises(tool.BoundaryViolation):
            action()


def test_capture_fresh_ordinary_import(legacy_dir):
    import importlib
    import sys
    tool = capture_module()
    names = ('xml', 'xml.etree', 'xml.etree.ElementPath', 'xml.etree.ElementTree', '_elementtree')
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        with tool.boundaries(legacy_dir) as attempts:
            module = importlib.import_module('xml.etree.ElementTree')
            assert module.fromstring('<root><child /></root>').find('child') is not None
            assert not attempts
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
