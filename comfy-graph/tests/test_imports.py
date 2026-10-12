"""Package boundary and legacy import identity."""
import builtins
import importlib
import sys

import pytest

from test_characterization import capture_module

MESSAGE = 'creative-comfy-graph is required; install it into the Python environment running this script.'


def test_imports_public_exports_and_identity(legacy_dir):
    package = importlib.import_module('creative_comfy_graph')
    assert package.__version__ == '0.1.0'
    assert package.__all__ == ['NodeRef', 'WorkflowGraph', '__version__']
    tool = capture_module()
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        core = modules['core']
        assert core.NodeRef is package.NodeRef
        assert core.WorkflowGraph is package.WorkflowGraph
        graph = package.WorkflowGraph()
        graph.node('Sink', source=core.NodeRef('legacy', 4))
        assert graph.to_dict()['0']['inputs']['source'] == ['legacy', 4]


def test_imports_missing_dependency_fails_before_transport(legacy_dir, monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'creative_comfy_graph':
            raise ModuleNotFoundError("No module named 'creative_comfy_graph'", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing)
    tool = capture_module()
    with tool.boundaries(legacy_dir) as attempts:
        with pytest.raises(ImportError, match='^' + MESSAGE.replace('.', r'\.') + '$'):
            with tool.legacy_modules(legacy_dir):
                pass
        assert not attempts


@pytest.mark.parametrize('failure', [ModuleNotFoundError('broken nested dependency', name='nested_dependency'),
                                    ImportError('broken export'), RuntimeError('broken module')])
def test_imports_unrelated_failures_propagate(legacy_dir, monkeypatch, failure):
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == 'creative_comfy_graph':
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', broken)
    tool = capture_module()
    with tool.boundaries(legacy_dir):
        with pytest.raises(type(failure)) as error:
            with tool.legacy_modules(legacy_dir):
                pass
        assert error.value is failure


def test_imports_package_is_offline(legacy_dir):
    tool = capture_module()
    names = [name for name in sys.modules if name == 'creative_comfy_graph' or name.startswith('creative_comfy_graph.')]
    saved = {name: sys.modules.pop(name) for name in names}
    try:
        with tool.boundaries(legacy_dir) as attempts:
            package = importlib.import_module('creative_comfy_graph')
            package.WorkflowGraph().node('Offline')
            assert not attempts
    finally:
        for name in list(sys.modules):
            if name == 'creative_comfy_graph' or name.startswith('creative_comfy_graph.'):
                sys.modules.pop(name)
        sys.modules.update(saved)
