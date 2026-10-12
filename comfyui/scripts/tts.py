"""Legacy TTS graph exports."""
try:
    import creative_comfy_graph as _graph_package
except ModuleNotFoundError as _exc:
    if _exc.name != "creative_comfy_graph":
        raise
    # Reuse core's actionable missing-distribution diagnostic.
    from core import WorkflowGraph
    raise

from creative_comfy_graph.tts import WorkflowGraph, qwen_tts, qwen_voice_clone
