"""Legacy Flux2 exports; transport remains in the skill adapter."""
try:
    import creative_comfy_graph as _graph_package
except ModuleNotFoundError as _exc:
    if _exc.name != "creative_comfy_graph":
        raise
    # Reuse core's actionable missing-distribution diagnostic.
    from core import WorkflowGraph
    raise

from creative_comfy_graph.graph import WorkflowGraph
from core import upload_if_local
from creative_comfy_graph.flux2 import (
    time,
    flux2_text_to_image,
    flux2_multi_reference_edit,
    flux2_single_image_edit,
    flux2_double_image_edit,
    flux2_multi_reference_edit_multiprompt,
    flux2_double_image_edit_multiprompt,
    flux2_multiple_angles,
)
