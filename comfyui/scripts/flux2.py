"""Flux2 image generation workflows for ComfyUI at https://comfyui.tail9683c.ts.net

IMPORTANT: This server uses qwen_3_8b_fp8mixed as the CLIP model (NOT gemma).
The UNET must be flux-2-klein-9b-fp8.safetensors for full quality.

Models available on this server:
- UNET: flux-2-klein-4b-fp8.safetensors, flux-2-klein-9b-fp8.safetensors
- VAE: flux2-vae.safetensors, ae.safetensors
- CLIP: qwen_3_8b_fp8mixed.safetensors (use type="flux2"), gemma_3_12B_it_fp8_e4m3fn.safetensors

NOTE: LatentAddNoise is NOT installed on this server. Always pass the latent
directly to SamplerCustomAdvanced without an intermediate LatentAddNoise node.
"""
from core import WorkflowGraph, upload_if_local
import time


# ---- Text-to-Image ----

def _load_models(g, unet_name, vae_name, clip_name, lora=None, lora_strength=1.0):
    """Load UNET, VAE, CLIP and optionally apply a LoRA. Returns (model, vae, clip)."""
    unet = g.node("UNETLoader", unet_name=unet_name, weight_dtype="default")
    vae  = g.node("VAELoader", vae_name=vae_name)
    clip = g.node("CLIPLoader", clip_name=clip_name, type="flux2", device="default")
    model, clip_out = unet[0], clip[0]
    if lora:
        lora_node = g.node("LoraLoader", model=model, clip=clip_out,
                           lora_name=lora, strength_model=lora_strength,
                           strength_clip=lora_strength)
        model, clip_out = lora_node[0], lora_node[1]
    return model, vae[0], clip_out


def flux2_text_to_image(prompt, width=1024, height=576, steps=4,
                         filename_prefix="flux2_t2i",
                         unet_name="flux-2-klein-9b-fp8.safetensors",
                         vae_name="flux2-vae.safetensors",
                         clip_name="qwen_3_8b_fp8mixed.safetensors",
                         seed=None, lora=None, lora_strength=1.0):
    g = WorkflowGraph()
    model, vae, clip = _load_models(g, unet_name, vae_name, clip_name, lora, lora_strength)
    pos    = g.node("CLIPTextEncode", text=prompt, clip=clip)
    neg    = g.node("CLIPTextEncode", text="", clip=clip)
    latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    sample = g.node("KSamplerSelect", sampler_name="euler")
    sched  = g.node("Flux2Scheduler", steps=steps, width=width, height=height)
    noise  = g.node("RandomNoise", noise_seed=seed or (int(time.time() * 1000) % (2**32)))
    guider = g.node("CFGGuider", model=model, positive=pos[0], negative=neg[0], cfg=1.0)
    sampled= g.node("SamplerCustomAdvanced",
                    noise=noise[0], guider=guider[0], sampler=sample[0],
                    sigmas=sched[0], latent_image=latent[0])
    decoded= g.node("VAEDecode", samples=sampled[0], vae=vae)
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return g.to_dict()


# ---- Single image edit (reference image + prompt) ----

def flux2_single_image_edit(image_filename, prompt,
                              width=1024, height=576, steps=4,
                              filename_prefix="flux2_i2i",
                              unet_name="flux-2-klein-9b-fp8.safetensors",
                              vae_name="flux2-vae.safetensors",
                              clip_name="qwen_3_8b_fp8mixed.safetensors",
                              seed=None, lora=None, lora_strength=1.0):
    g = WorkflowGraph()
    model, vae, clip = _load_models(g, unet_name, vae_name, clip_name, lora, lora_strength)
    pos    = g.node("CLIPTextEncode", text=prompt, clip=clip)
    neg    = g.node("CLIPTextEncode", text="", clip=clip)
    ref    = g.node("LoadImage", image=image_filename)
    scaled = g.node("ImageScaleToTotalPixels", image=ref[0],
                    upscale_method="nearest-exact", megapixels=1, resolution_steps=1)
    enc    = g.node("VAEEncode", pixels=scaled[0], vae=vae)
    pos_ref= g.node("ReferenceLatent", conditioning=pos[0], latent=enc[0])
    neg_ref= g.node("ReferenceLatent", conditioning=neg[0], latent=enc[0])
    latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    sample = g.node("KSamplerSelect", sampler_name="euler")
    sched  = g.node("Flux2Scheduler", steps=steps, width=width, height=height)
    noise  = g.node("RandomNoise", noise_seed=seed or (int(time.time() * 1000) % (2**32)))
    guider = g.node("CFGGuider", model=model, positive=pos_ref[0], negative=neg_ref[0], cfg=1.0)
    sampled= g.node("SamplerCustomAdvanced",
                    noise=noise[0], guider=guider[0], sampler=sample[0],
                    sigmas=sched[0], latent_image=latent[0])
    decoded= g.node("VAEDecode", samples=sampled[0], vae=vae)
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return g.to_dict()


# ---- Double image edit (two reference images merged) ----

def flux2_double_image_edit(image1_filename, image2_filename, prompt,
                              width=1024, height=576, steps=4,
                              filename_prefix="flux2_i2i2",
                              unet_name="flux-2-klein-9b-fp8.safetensors",
                              vae_name="flux2-vae.safetensors",
                              clip_name="qwen_3_8b_fp8mixed.safetensors",
                              seed=None, lora=None, lora_strength=1.0):
    g = WorkflowGraph()
    model, vae, clip = _load_models(g, unet_name, vae_name, clip_name, lora, lora_strength)
    pos    = g.node("CLIPTextEncode", text=prompt, clip=clip)
    neg    = g.node("CLIPTextEncode", text="", clip=clip)

    def _encode(fname):
        ref   = g.node("LoadImage", image=fname)
        scaled= g.node("ImageScaleToTotalPixels", image=ref[0],
                       upscale_method="nearest-exact", megapixels=1, resolution_steps=1)
        return g.node("VAEEncode", pixels=scaled[0], vae=vae)

    merged = g.node("LatentBatch", samples1=_encode(image1_filename)[0],
                    samples2=_encode(image2_filename)[0])
    pos_ref= g.node("ReferenceLatent", conditioning=pos[0], latent=merged[0])
    neg_ref= g.node("ReferenceLatent", conditioning=neg[0], latent=merged[0])
    latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    sample = g.node("KSamplerSelect", sampler_name="euler")
    sched  = g.node("Flux2Scheduler", steps=steps, width=width, height=height)
    noise  = g.node("RandomNoise", noise_seed=seed or (int(time.time() * 1000) % (2**32)))
    guider = g.node("CFGGuider", model=model, positive=pos_ref[0], negative=neg_ref[0], cfg=1.0)
    sampled= g.node("SamplerCustomAdvanced",
                    noise=noise[0], guider=guider[0], sampler=sample[0],
                    sigmas=sched[0], latent_image=latent[0])
    decoded= g.node("VAEDecode", samples=sampled[0], vae=vae)
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return g.to_dict()


# ---- Multiple angles from single reference ----

def flux2_multiple_angles(image_filename, angle_prompts, prepend="", append="",
                           filename_prefix="flux2_angles",
                           unet_name="flux-2-klein-9b-fp8.safetensors",
                           vae_name="flux2-vae.safetensors",
                           clip_name="qwen_3_8b_fp8mixed.safetensors",
                           steps=4, lora=None, lora_strength=1.0):
    g = WorkflowGraph()
    model, vae, clip = _load_models(g, unet_name, vae_name, clip_name, lora, lora_strength)
    batcher= g.node("SimplePromptBatcher", prepend=prepend,
                     prompts="\n".join([p for p in angle_prompts if p]) + "\n", append=append)
    pos    = g.node("CLIPTextEncode", text=batcher[0], clip=clip)
    neg    = g.node("ConditioningZeroOut", conditioning=pos[0])
    ref    = g.node("LoadImage", image=image_filename)
    scaled = g.node("ImageScaleToTotalPixels", image=ref[0],
                    upscale_method="lanczos", megapixels=1, resolution_steps=1)
    size   = g.node("GetImageSize", image=scaled[0])
    enc    = g.node("VAEEncode", pixels=scaled[0], vae=vae)
    pos_ref= g.node("ReferenceLatent", conditioning=pos[0], latent=enc[0])
    neg_ref= g.node("ReferenceLatent", conditioning=neg[0], latent=enc[0])
    latent = g.node("EmptyFlux2LatentImage", width=size[0], height=size[1], batch_size=1)
    sched  = g.node("Flux2Scheduler", steps=steps, width=size[0], height=size[1])
    sample = g.node("KSamplerSelect", sampler_name="euler")
    noise  = g.node("RandomNoise", noise_seed=int(time.time() * 1000) % (2**32))
    guider = g.node("CFGGuider", model=model, positive=pos_ref[0], negative=neg_ref[0], cfg=1.0)
    sampled= g.node("SamplerCustomAdvanced",
                    noise=noise[0], guider=guider[0], sampler=sample[0],
                    sigmas=sched[0], latent_image=latent[0])
    decoded= g.node("VAEDecode", samples=sampled[0], vae=vae)
    g.node("SaveImage", images=decoded[0], filename_prefix=filename_prefix)
    return g.to_dict()
