"""Legacy post graph exports and local clip transport adapter."""
try:
    import creative_comfy_graph as _graph_package
except ModuleNotFoundError as _exc:
    if _exc.name != "creative_comfy_graph":
        raise
    # Reuse core's actionable missing-distribution diagnostic.
    from core import WorkflowGraph
    raise

from pathlib import Path
from creative_comfy_graph.graph import WorkflowGraph
from creative_comfy_graph.post import (
    extract_stems,
    transcribe,
    COMFY_INPUT_DIR,
    COMFY_OUTPUT_DIR,
    concat_videos_ffmpeg,
)


def stage_clips_for_concat(local_paths, subfolder):
    """Upload N local video files into a single comfy-input subfolder with
    zero-padded numeric filename prefixes so alphabetical order matches the
    caller's sequence. Returns the remote names actually used.

    E.g. caller passes [a.mp4, b.mp4, c.mp4] and gets back
    ['001_a.mp4', '002_b.mp4', '003_c.mp4'] stored under
    `/app/ComfyUI/input/<subfolder>/`."""
    from core import upload_image  # lazy: core imports post in some places
    out_names = []
    for i, lp in enumerate(local_paths, 1):
        name = f"{i:03d}_{Path(lp).name}"
        remote = upload_image(str(lp), subfolder=subfolder, upload_as=name)
        out_names.append(remote)
    return out_names


from creative_comfy_graph.post import concat_videos
