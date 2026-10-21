"""Native MOSS-Image HF graphs based on OpenMOSS's ComfyUI example workflow."""
from __future__ import annotations

import math
import secrets

from .graph import WorkflowGraph

MODEL_PATH = "/app/ComfyUI/models/moss-image/MOSS-Image-Base"
WEIGHT_DTYPES = ("default", "bf16", "fp32", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")


def moss_image_text_to_image(
    prompt: str, *, model_path: str = MODEL_PATH,
    width: int = 1024, height: int = 1024, steps: int = 50,
    cfg: float = 3.0, shift: float = 5.0, negative_prompt: str = "",
    skip_layers: str = "22", weight_dtype: str = "fp8_e4m3fn",
    text_device: str = "cpu", batch_size: int = 1, seed: int | None = None,
    filename_prefix: str = "moss-image",
) -> dict:
    """Build a text-to-image graph; paths refer to the ComfyUI host, not the client.

    Sampling follows the upstream example. FP8 DiT weights and CPU text encoding
    are the low-VRAM defaults; use weight_dtype="default", text_device="default"
    for the upstream loader defaults. An empty skip_layers disables SLG.
    """
    if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
        raise ValueError("prompt and negative_prompt must be strings")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a local HF model directory on the ComfyUI host")
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or value < 16 or value % 16:
            raise ValueError(f"{name} must be a positive multiple of 16")
    for name, value in (("steps", steps), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"weight_dtype must be one of {WEIGHT_DTYPES}")
    if text_device not in ("default", "cpu"):
        raise ValueError("text_device must be 'default' or 'cpu'")
    if not math.isfinite(cfg) or cfg < 0 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("cfg must be finite and nonnegative; shift must be finite and positive")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 64):
        raise ValueError("seed must be an unsigned 64-bit integer")

    graph = WorkflowGraph()
    loaded = graph.node("MOSSImageHFLoader", model_path=model_path,
                        weight_dtype=weight_dtype, text_device=text_device)
    positive = graph.node("CLIPTextEncode", text=prompt, clip=loaded[1])
    negative = graph.node("CLIPTextEncode", text=negative_prompt, clip=loaded[1])
    latent = graph.node("EmptyFlux2LatentImage", width=width, height=height,
                        batch_size=batch_size)
    shifted = graph.node("ModelSamplingSD3", model=loaded[0], shift=shift)
    guided = graph.node("SkipLayerGuidanceDiTSimple", model=shifted[0],
                        double_layers="", single_layers=skip_layers,
                        start_percent=0.0, end_percent=1.0)
    sampled = graph.node(
        "KSampler", model=guided[0], positive=positive[0], negative=negative[0],
        latent_image=latent[0], seed=secrets.randbits(64) if seed is None else seed,
        steps=steps, cfg=cfg, sampler_name="euler", scheduler="sgm_uniform", denoise=1.0,
    )
    decoded = graph.node("VAEDecode", samples=sampled[0], vae=loaded[2])
    graph.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return graph.to_dict()
