"""Exact legacy primitive semantics, via normal installed imports."""
import importlib
import json

import pytest


@pytest.fixture
def primitives():
    return importlib.import_module('creative_comfy_graph')


def test_primitives_reference_attributes_and_indexing(primitives):
    ref = primitives.NodeRef('12', 3)
    assert (ref.node_id, ref.output_idx) == ('12', 3)
    assert ref.as_link() == ['12', 3]
    assert repr(ref) == "NodeRef('12', 3)"
    for index in (0, 5, -1):
        other = ref[index]
        assert other is not ref
        assert isinstance(other, primitives.NodeRef)
        assert other.as_link() == ['12', index]
    assert ref.output_idx == 3


def test_primitives_ids_order_and_nonrecursive_links(primitives):
    graph = primitives.WorkflowGraph()
    assert graph.to_dict() == {}
    refs = [graph.node('Source', value=i) for i in range(12)]
    assert [ref.node_id for ref in refs] == [str(i) for i in range(12)]
    nested = [refs[0], {'ref': refs[1]}]
    graph.node('Sink', direct=refs[2][-1], nested=nested)
    result = graph.to_dict()
    assert list(result) == [str(i) for i in range(13)]
    assert result['12'] == {'class_type': 'Sink', 'inputs': {'direct': ['2', -1], 'nested': nested}}
    assert result['12']['inputs']['nested'] is nested
    assert nested[0] is refs[0]
    with pytest.raises(TypeError):
        graph.to_json()


def test_primitives_shallow_copy_semantics(primitives):
    graph = primitives.WorkflowGraph()
    values = [1]
    graph.node('Source', values=values)
    first, second = graph.to_dict(), graph.to_dict()
    assert first is not second
    first['extra'] = {}
    del first['0']
    assert list(graph.to_dict()) == ['0']
    second['0']['inputs']['added'] = True
    values.append(2)
    assert graph.to_dict()['0']['inputs'] == {'values': [1, 2], 'added': True}
    second['0']['class_type'] = 'Changed'
    assert graph.to_dict()['0']['class_type'] == 'Changed'


@pytest.mark.parametrize('indent', [2, None, 4])
def test_primitives_json(primitives, indent):
    graph = primitives.WorkflowGraph()
    graph.node('First', z=2, a=1)
    assert graph.to_json(indent=indent) == json.dumps(graph.to_dict(), indent=indent)
    assert graph.to_json() == json.dumps(graph.to_dict(), indent=2)
