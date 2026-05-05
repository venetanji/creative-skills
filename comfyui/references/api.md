# ComfyUI API Reference

The deployment has **two ComfyUI servers** behind the skill's CLI:

| Workload | Base URL | GPU |
|---|---|---|
| Flux/images, TTS, audio post | `https://comfyui.tail9683c.ts.net` | RTX 3080 Ti 12GB |
| LTX-2.3 video | `https://comfyui-video.tail9683c.ts.net` | RTX 3090 24GB |

`comfy_graph.py` routes by command class; override per-class with
`COMFY_URL_FLUX` / `COMFY_URL_VIDEO`, or force a single endpoint with
`COMFY_URL`. Examples below assume the flux server.

**Version:** 0.17.0 | PyTorch 2.10+cu130

---

## Core Endpoints

### POST /prompt — Submit workflow
```json
{ "prompt": { "<node_id>": { "class_type": "NodeName", "inputs": {...} } } }
```
Response: `{ "prompt_id": "uuid", "number": N, "node_errors": {} }`

### GET /history/{prompt_id} — Poll results
```json
{
  "uuid": {
    "status": { "status_str": "success", "completed": true },
    "outputs": { "<node_id>": { "images": [{ "filename": "out.png", "subfolder": "", "type": "output" }] } }
  }
}
```
Poll every 1-2s. `status_str`: `"success"` | `"error"` | `"executing"`.

### GET /view?filename=F&subfolder=S&type=output — Download asset
Add `&_=<timestamp>` to bypass caching.

### POST /upload/image — Upload reference image
```bash
curl -X POST -F "image=@photo.jpg" -F "type=input" -F "subfolder=" \
  https://comfyui.tail9683c.ts.net/upload/image
```

---

## Key Node Types

| Node | Required inputs | Output |
|------|----------------|--------|
| `UNETLoader` | `unet_name` | MODEL |
| `CLIPLoader` | `clip_name`, `type` ("flux2") | CLIP |
| `VAELoader` | `vae_name` | VAE |
| `CLIPTextEncode` | `text`, `clip` | CONDITIONING |
| `CFGGuider` | `model`, `positive`, `negative`, `cfg` | GUIDER |
| `Flux2Scheduler` | `steps`, `width`, `height` | SIGMAS |
| `KSamplerSelect` | `sampler_name` | SAMPLER |
| `RandomNoise` | `noise_seed` | NOISE |
| `EmptyFlux2LatentImage` | `width`, `height`, `batch_size` | LATENT |
| `SamplerCustomAdvanced` | `noise`, `guider`, `sampler`, `sigmas`, `latent_image` | LATENT |
| `VAEDecode` | `samples`, `vae` | IMAGE |
| `SaveImage` | `images`, `filename_prefix` | — |
| `LoadImage` | `image` (filename) | IMAGE, MASK |
| `ReferenceLatent` | `conditioning`, `latent` | CONDITIONING |

**Video (LTX2):**
| Node | Required inputs | Output |
|------|----------------|--------|
| `EmptyLTXVLatentVideo` | `width`, `height`, `length`, `batch_size` | LATENT |
| `LTXVImgToVideoInplace` | `vae`, `image`, `latent` | LATENT |
| `LTXVSpatioTemporalTiledVAEDecode` | `vae`, `latents`, `spatial_tiles`, ... | IMAGE |
| `CreateVideo` | `images`, `fps` | VIDEO |
| `SaveVideo` | `video`, `filename_prefix` | — |
| `LTXVScheduler` | `steps`, `max_shift`, `base_shift`, `terminal`, `latent` | SIGMAS |

---

## Workflow Structure Pattern

```
UNETLoader → CFGGuider.model
CLIPLoader → CLIPTextEncode → CFGGuider.positive
             CLIPTextEncode → CFGGuider.negative
RandomNoise → SamplerCustomAdvanced.noise
KSamplerSelect → SamplerCustomAdvanced.sampler
Flux2Scheduler → SamplerCustomAdvanced.sigmas
EmptyFlux2LatentImage → SamplerCustomAdvanced.latent_image
                               ↓
                        VAEDecode → SaveImage
VAELoader  → VAEDecode.vae
```

---

## Available Models

**UNET / diffusion_models:**
- `flux-2-klein-4b-fp8.safetensors` ⚡ fast (default for image)
- `flux-2-klein-9b-fp8.safetensors` 🎬 higher quality (use for video)
- `Wan2_2-Animate-14B_fp8_scaled_e5m2_KJ_v2.safetensors`

**VAE:**
- `flux2-vae.safetensors` — Flux image VAE
- `LTX2_video_vae_bf16.safetensors` — LTX video VAE
- `ae.safetensors` — autoencoder

**CLIP / text_encoders (⚠️ use qwen_3_8b_fp8mixed for Flux2 on this server):**
- `qwen_3_8b_fp8mixed.safetensors` ← USE THIS with `type="flux2"` ✅
- `gemma_3_12B_it_fp8_e4m3fn.safetensors` (causes "3D tensor" error with Klein models!)
- `qwen_0.6b_ace15.safetensors`, `qwen_1.7b_ace15.safetensors`, `qwen_3_4b.safetensors`
- `qwen_3_4b_fp4_flux2.safetensors`, `qwen_3_8b_fp8mixed.safetensors`

**LoRAs (in `models/loras/`):**
- `ltx-2-19b-distilled-lora-384.safetensors` — 4-step distilled
- `ltx-2-19b-lora-camera-control-{static,dolly-in,dolly-out,...}.safetensors` — camera
