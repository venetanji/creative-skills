---
name: comfyui
description: >
  Generate images and videos using ComfyUI workflows via direct REST API at
  https://comfyui.tail9683c.ts.net. Also manages character reference image downloads
  from HuggingFace. Use when asked to create images, edit photos, generate videos,
  or run Flux/LTX2/Wan workflows. Triggers on: generate an image, create a video,
  Flux2, ComfyUI, text-to-image, image-to-image, image-to-video, character image,
  Athena, scene generation, TTS, voice clone, workflow.
---

# ComfyUI Skill

Direct ComfyUI REST API access for image and video generation.

**ComfyUI:** https://comfyui.tail9683c.ts.net | **RTX 3090 24GB** | v0.16.1

## Script paths (use absolute; tilde expansion is unreliable under `exec`)

| Where you're running | Path to use |
|---|---|
| Inside the sandbox (most agents) | `/home/sandbox/.openclaw/skills/comfyui/scripts/…` |
| On the host (main, zeus) | `/home/venetanji/.openclaw/skills/comfyui/scripts/…` |

`~/.openclaw/skills/…` works in an interactive shell but NOT always in the `exec` tool — use the absolute form so the first call succeeds.

## LTX-2.3 Video Models (Required)

To use LTX-2.3 video generation, download these models to your ComfyUI installation:

```bash
# From your ComfyUI models directory, run:
mkdir -p checkpoints text_encoders loras latent_upscale_models

wget -P checkpoints https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors
wget -P text_encoders https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
wget -P loras https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors
```

**One-liner for copy-paste:**
```bash
cd /path/to/ComfyUI/models && mkdir -p checkpoints text_encoders loras && wget -P checkpoints https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors && wget -P text_encoders https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors && wget -P loras https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors
```

**Alternative - distilled checkpoint (smaller, faster):**
```bash
wget -P checkpoints https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors
```

**Alternative text encoder (fp8 - faster):**
```bash
wget -P text_encoders https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors
```

### LTX-2.3 Model Files Summary

| Model | Location | Size | Purpose |
|-------|----------|------|---------|
| `ltx-2.3-22b-dev-fp8.safetensors` | `checkpoints/` | ~22B params | Main video diffusion model |
| `ltx-2.3-22b-distilled-fp8.safetensors` | `checkpoints/` | ~22B params | Distilled 4-step version |
| `gemma_3_12B_it_fp4_mixed.safetensors` | `text_encoders/` | ~12B params | Text encoder (CLIP) - fp4 |
| `gemma_3_12B_it_fp8_e4m3fn.safetensors` | `text_encoders/` | ~12B params | Text encoder (CLIP) - fp8 **(recommended)** |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | `loras/` | Small | 4-step distilled LoRA |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `latent_upscale_models/` | Small | Latent upscaler (optional) |

### Text Encoder Path

⚠️ **Important**: When using `LTXAVTextEncoderLoader`, the text encoder path must include the subdirectory:
- Correct: `split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`
- Or use `gemma_3_12B_it_fp8_e4m3fn.safetensors` if downloaded directly to `text_encoders/`

The skill defaults to `gemma_3_12B_it_fp8_e4m3fn.safetensors` (fp8 variant) for faster inference.

## Character Reference Images (HuggingFace)

All 84 character reference images are on HuggingFace:
**Dataset:** `venetanji/polyu-storyworld-characters`

### Download
```bash
# Download all characters (~420 images)
HF_TOKEN=<YOUR_HF_TOKEN> \
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/download_characters.py

# Download specific characters only
HF_TOKEN=<YOUR_HF_TOKEN> \
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/download_characters.py 6166r 1822g
```

**Storage:** `~/.openclaw/assets/characters/<code>/` (e.g. `~/.openclaw/assets/characters/6166r/`)
- Multiple reference images per character (e.g. `1.png`, `2.jpg`, `img3.jpeg`)
- `.txt` files alongside each image contain tag-based captions

### Generate with Character References
```bash
# 1. Upload Athena's reference image to ComfyUI
python3 -c "
import sys; sys.path.insert(0,'~/.openclaw/skills/comfyui/scripts')
from core import upload_if_local
name = upload_if_local('~/.openclaw/assets/characters/6166r/1.png')
print(name)
"

# 2. Use in i2i — reference image guides the generation
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py i2i \
  --image /path/to/uploaded/6166r_1.png \
  --prompt "Athena walking through a moonlit forest, dramatic portrait" \
  --steps 12 --seed 42
```

### Workflow for "Generate image of character 6166r"
1. Read YAML: `repos/polyu-storyworld/characters/6166r.yaml` → character description
2. Read captions: `~/.openclaw/assets/characters/6166r/*.txt`
3. Pick best reference image (e.g. `1.png` — usually the primary front view)
4. Upload to ComfyUI via `upload_if_local()`
5. Build i2i prompt: combine YAML description + reference caption
6. Run `flux2_single_image_edit` via comfy_graph.py i2i
7. Download → crop → send to Discord

**No MCP needed.** All assets are local. The MCP is only needed for MCP-aware tools (not used here).

```bash
# Image generation (text-to-image)
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py t2i \
  --prompt "Athena in a forest"

# Image-to-image with reference
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py i2i \
  --image athena_ref.png \
  --prompt "Athena riding a white horse"

# Text-to-video
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py t2v \
  --prompt "Athena walking through ancient ruins" \
  --seconds 5

# Query server state
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_query.py stats

# Run workflow from JSON
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_run.py workflow.json --output-dir /tmp/imgs
```

## Script Inventory

### comfy_graph.py — CLI entry point (most common)
Builds workflows and submits them. All commands support `--output-dir`, `--timeout`, `--seed`.
```
t2i      Text-to-image (Flux2)
i2i      Single image edit (Flux2 + reference image)
i2i2     Double image edit (two references blended)
angles   Multi-angle batch from one reference
t2v      Text-to-video (LTX-2.3)
i2v      Image-to-video (LTX-2.3, first-frame)
tts      Text-to-speech (Qwen3)
dump     Print workflow JSON only, no execution
last_frame  Extract last frame from server video file
```

### comfy_query.py — Server diagnostics
```
stats              GPU/RAM usage, comfy version
queue              Running and pending jobs
loras              Available LoRAs
models [type]      Models in category (loras, diffusion_models, checkpoints…)
node <ClassName>   Input/output schema for any node
history [id]       Inspect a specific prompt_id (or last 5)
```

### comfy_run.py — Workflow runner
Submit a workflow JSON and wait for results. Accepts stdin (`-`) or file path.
```bash
# Pipe from dump command
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py dump t2i --prompt "a cat" | \
  python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_run.py - --output-dir /tmp/imgs

# Or from a saved JSON
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_run.py workflow.json --output-dir /tmp/imgs --timeout 300
```

### video_join.py — ffmpeg helper (concat / trim / extract frames)

Standalone script with a `#!/usr/bin/env -S uv run --script` shebang and PEP 723 inline metadata; `uv` auto-installs `imageio-ffmpeg` (static ffmpeg binary) on first run. Works identically on host and inside the sandbox — no apt install needed.

```bash
# Concatenate clips (stream-copy when codecs/sizes match → fast; otherwise re-encode)
/home/sandbox/.openclaw/skills/comfyui/scripts/video_join.py concat \
  --inputs scene_a.mp4,scene_b.mp4,scene_c.mp4 \
  --output full_scene.mp4
# or from a list file: --inputs @clips.txt  (one path per line)

# Trim a clip
/home/sandbox/.openclaw/skills/comfyui/scripts/video_join.py trim \
  --input v.mp4 --start 0.5 --duration 2.0 --output v_trim.mp4

# Extract first or last frame (for FLF2V chaining)
/home/sandbox/.openclaw/skills/comfyui/scripts/video_join.py last-frame \
  --input clip1.mp4 --output last.png
```

### FLF2V (first-last-frame-to-video) / scene continuation pattern

1. Generate clip A with i2v from an anchor image.
2. Extract the last frame of clip A via `video_join.py last-frame`.
3. Generate clip B with i2v from that last frame (same prompt or next-scene prompt).
4. Concatenate A + B via `video_join.py concat` → continuous scene.

Works the same for longer chains (A → B → C → …). For audio-reactive continuations, use `ia2v` for each segment and slice the audio to match.

### test_all_workflows.py — Regression test suite
Run all workflow builders end-to-end. Results logged to `test_results.json`.
```bash
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/test_all_workflows.py
```
Covers: Flux2 t2i, t2i+LoRA, i2i, angles, TTS, LTX-2.3 t2v, LTX-2.3 i2v.

## CLI Reference

| Command | Description | Key options |
|---------|-------------|-------------|
| `t2i` | Text-to-image (Flux2) | `--prompt`, `--width`, `--height`, `--steps`, `--prefix`, `--lora` |
| `i2i` | Single image edit (Flux2 + reference) | `--image`, `--prompt`, `--width`, `--steps` |
| `i2i2` | Double image edit (two references) | `--image1`, `--image2`, `--prompt` |
| `angles` | Multi-angle batch from one reference | `--image`, `--prompts`, `--prepend`, `--append` |
| `t2v` | Text-to-video (LTX-2.3, two-pass) | `--prompt`, `--seconds`, `--fps`, `--width`, `--height` |
| `i2v` | Image-to-video (LTX-2.3, first-frame + refine) | `--image`, `--prompt`, `--seconds`, `--fps` |
| `ia2v` | Image + audio to audio-reactive video | `--image`, `--audio`, `--prompt`, `--seconds`, `--fps` |
| `flf2v` | First+last frame to video (LTX-2.3, single-pass) | `--first`, `--last`, `--prompt`, `--seconds`, `--fps`, `--guide-strength` |
| `tts` | Text-to-speech (Qwen3 TTS) | `--text`, `--prefix` |
| `run` | Submit any workflow JSON (file or stdin) | `--file workflow.json` or `cat wf.json \| ... run` |
| `dump` | Print workflow JSON, no execution | workflow command + options |
| `last_frame` | Extract last frame from video | `--video-path` |

### Global options
- `--notify-target <target>` — OpenClaw notification target (e.g. `discord:1486985676066000957`)
- `--output-dir <path>` — local output directory (default: `outputs/`)
- `--timeout <seconds>` — generation timeout
- `--seed <int>` — random seed for reproducibility
- `--fast` *(video only: t2v/i2v/ia2v/flf2v)* — skip the refine pass. About half the wall time, half the output resolution (no 2× upsample), rougher detail. Use for prompt iteration; leave off for final output.

### Environment variables
- `COMFY_URL` — ComfyUI server URL (default: `https://comfyui.tail9683c.ts.net`)
- `OPENCLAW_NOTIFY_TARGET` — default notification target

## Architecture

### Flux2 (images)
- **UNET:** `flux-2-klein-9b-fp8.safetensors` (9B full quality) or `flux-2-klein-4b-fp8.safetensors`
- **CLIP:** `qwen_3_8b_fp8mixed.safetensors` with `type="flux2"` ⚠️ **NOT gemma!**
  - gemma causes `ValueError: Input img and txt tensors must have 3 dimensions`
- **VAE:** `flux2-vae.safetensors`
- **Workflow:** `RandomNoise → CFGGuider → SamplerCustomAdvanced → Flux2Scheduler → VAEDecode → SaveImage`
  - ⚠️ **No LatentAddNoise** — that node is NOT installed on this server

### LTX-2.3 (video) — two-pass refine pattern

- **Checkpoint:** `ltx-2.3-22b-dev-fp8.safetensors`
- **Text Encoder:** `gemma_3_12B_it_fp4_mixed.safetensors` via `LTXAVTextEncoderLoader`
- **Audio VAE:** via `LTXVAudioVAELoader` (uses same ckpt_name)
- **Upscaler (between passes):** `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` via `LatentUpscaleModelLoader`
- **LoRA:** `ltx-2.3-22b-distilled-lora-384.safetensors` at **strength 0.6** — applied on ALL flows (t2v/i2v/ia2v/flf2v). This reproduces the distilled checkpoint behaviour on top of the dev-fp8 checkpoint we have installed; without it the 8-step schedule under-denoises and output is badly blurred.
- **Pass 1 (coarse, 9 steps):** `euler_ancestral_cfg_pp` + ManualSigmas `"1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"`, raw `LTXVConditioning`
- **Between passes:** Separate AV → `LTXVLatentUpsampler` → re-apply image via `LTXVImgToVideoInplace(strength=1.0)` (i2v/ia2v only) → re-concat with audio → `LTXVCropGuides` on coarse video latent for conditioning
- **Pass 2 (refine, 4 steps):** `euler_cfg_pp` + ManualSigmas `"0.85, 0.7250, 0.4219, 0.0"`, cropped conditioning
- **Always AV-concat:** All three variants (t2v/i2v/ia2v) use `LTXVConcatAVLatent`/`LTXVSeparateAVLatent`. For non-audio variants, `LTXVEmptyLatentAudio` provides a blank audio side. This keeps one graph shape.
- **Length:** `(seconds × fps)` rounded to `≡ 1 (mod 8)`
- **Audio for ia2v:** `LoadAudio` → `TrimAudioDuration` → `LTXVAudioVAEEncode` → `SetLatentNoiseMask` (with zero-valued `SolidMask`)
- **flf2v (first-last frame):** adapted from the `Comfy-Org/workflow_templates` flf2v template into the same two-pass shape as the other flows. Pass-1 uses chained `LTXVAddGuide` (frame_idx=0 + frame_idx=-1 at strength 0.7) on the base latent; pass-2 re-injects both guides at strength 1.0 on the upsampled latent. Post-decode `LTXVCropGuides` strips the injected guide frames so the output doesn't show the raw input images as first/last frames (use `--fast` to keep the single-pass shape the template originally had).

### Available LoRAs (19)
- Camera: `ltx-2-19b-lora-camera-control-dolly-in/out/left/right/jib-up/jib-down/static`
- Quality: `ltx-2-19b-lora-detailer`, `ltx-2-19b-distilled-lora-384`
- Style: `pixel_art_style_z_image_turbo.safetensors`
- Relight: `WanAnimate_relight_lora_fp16.safetensors`
- I2V: `wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise`, `lightx2v_I2V_14B_480p...`
- Lightning: `Qwen-Image-Lightning-{4,8}steps*`, `Qwen-Image-Edit-2511-Lightning*`

## Delivery to Discord (you send it, not the script)

`comfy_graph.py` **blocks** until the workflow completes. There is no async / webhook / notification path — the sandbox does not have the `openclaw` CLI, so the script cannot deliver on your behalf. The final stdout always includes a `saved: /workspace/outputs/<file>.ext` line with the sandbox path.

Your job after the exec returns:

1. Parse the `saved: …` line from the output.
2. Translate sandbox → host path: `/workspace/…` → `/home/venetanji/.openclaw/workspace-<agent>/…` (the `message` tool resolves paths on the host).
3. Send with the `message` tool's `media` arg.

```bash
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py t2i \
  --prompt "…" \
  --output-dir /workspace/outputs/
# stdout: … saved: /workspace/outputs/flux2_t2i_XXXXX_.png …
```

Then (pseudo):
```
message({
  action: "send",
  channel: "discord",
  to: "<channel_id>",
  media: "/home/venetanji/.openclaw/workspace-<agent>/outputs/flux2_t2i_XXXXX_.png",
  message: "Here's your image."
})
```

### `--notify-target` (advisory only)

`--notify-target discord:<channel_id>` does **not** send anything. It only makes the script print an extra line `[NOTIFY_URL 0] target=discord:<channel> media=<comfy_view_url> caption=<...>` so you have a ready-made URL + caption to feed to `message`. It's convenience, not automation. You still call `message` yourself.

### When to offload to a subagent

Blocking exec is fine for **images (< 30s)** and **short TTS**. For **video** (LTX-2.3 is 2-5 min), spawn a subagent so the main Discord reply doesn't hang:

```
sessions_spawn({
  task: "Generate t2v of '<prompt>' (seconds=5) via /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py, then send the resulting video to discord channel <id> with caption '<caption>'. Use exec + message tools. Host path for message is /home/venetanji/.openclaw/workspace-<agent>/outputs/<file>.",
  runtime: "subagent",
  sandbox: "inherit",
  mode: "run"
})
```

The subagent runs inside the same shared sandbox, does the long gen + delivery, and auto-announces completion to the main session when done. You (the main agent) are free to continue talking.

## Troubleshooting

### HTTP 400 "Node 'LatentAddNoise' not found"
→ `flux2.py` was updated to remove `LatentAddNoise` from the graph.
If you see this error, the `flux2.py` workflow builder is stale — re-import it.

### HTTP 400 "unet_name: 'None' not in list"
→ This was a bug where `opts.get("unet")` passed `None` as an explicit argument,
overriding the function default. Fixed by filtering None values with `**extra` pattern.
If you see this, the `comfy_graph.py` is stale — re-import it.
**Symptom:** any workflow command fails with 400 even though the graph looks correct.

### "Input img and txt tensors must have 3 dimensions" error
→ Wrong CLIP model! Use `qwen_3_8b_fp8mixed.safetensors` (NOT gemma) for Flux2.
For LTX-2.3 video, use `gemma_3_12B_it_fp8_e4m3fn.safetensors` via `LTXAVTextEncoderLoader`.

### "No output" after submit
→ Poll `/history/{prompt_id}` manually — check `node_errors` in the response.
Use: `python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_query.py history <prompt_id>`

### Slow generation
→ Use distilled models for 4-step generation. Full quality needs 20+ steps.

### Memory errors (OOM)
→ Reduce resolution or use smaller models. RTX 3090 has 24GB VRAM.

### LTX-2.3 "mat1 and mat2 shapes cannot be multiplied"
→ CLIP model mismatch. Make sure you're using `gemma_3_12B_it_fp8_e4m3fn.safetensors`
with the LTX-2.3 checkpoint, not UMT5 or other CLIP models.

### LTX-2.3 "Value not in list" for text_encoder
→ The text encoder path must include the subdirectory if using the fp4 variant:
- Correct: `split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`
- Or use fp8 variant directly: `gemma_3_12B_it_fp8_e4m3fn.safetensors`

For API details → `references/api.md`
For prompt tips → `references/prompt-patterns.md`
**For LTX-2.3 prompt writing → `references/ltx-prompt-guide.md`** (camera vocabulary, what to avoid, worked rewrites)
