"""Graph primitives shared by the library and legacy adapters."""
from __future__ import annotations

import json


class NodeRef:
    def __init__(self, node_id: str, output_idx: int = 0):
        self.node_id = node_id
        self.output_idx = output_idx

    def __getitem__(self, idx: int) -> "NodeRef":
        return NodeRef(self.node_id, idx)

    def as_link(self) -> list:
        return [self.node_id, self.output_idx]

    def __repr__(self):
        return f"NodeRef({self.node_id!r}, {self.output_idx})"


class WorkflowGraph:
    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._counter = 0

    def node(self, class_type: str, **inputs) -> NodeRef:
        node_id = str(self._counter)
        self._counter += 1
        processed = {}
        for k, v in inputs.items():
            if isinstance(v, NodeRef):
                processed[k] = v.as_link()
            else:
                processed[k] = v
        self._nodes[node_id] = {"class_type": class_type, "inputs": processed}
        return NodeRef(node_id)

    def to_dict(self) -> dict:
        return dict(self._nodes)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
