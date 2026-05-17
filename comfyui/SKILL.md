---
name: comfyui
description: >
  Generate images and videos using ComfyUI workflows via direct REST API. Use
  when asked to create images, edit photos, generate videos, or run
  Flux/LTX2/Wan workflows. Triggers on: generate an image, create a video,
  Flux2, ComfyUI, text-to-image, image-to-image, image-to-video, scene
  generation, TTS, voice clone, workflow.
---

# ComfyUI Skill

Direct ComfyUI REST API access for image and video generation.

## Server URL configuration

The scripts read three environment variables, in priority order:

| Var | Used by | Falls back to |
|---|---|---|
| `COMFY_URL_FLUX`  | image / TTS / audio commands (`t2i`, `i2i`, `tts`, `stems`, `stt`, `vconcat`, `last_frame`) | `COMFY_URL` |
| `COMFY_URL_VIDEO` | LTX video commands (`t2v`, `i2v`, `ia2v`, `flf2v`, `multiguide`, `transition`, `continuation`) | `COMFY_URL` |
| `COMFY_URL`       | single-server fallback for everything | `http://localhost:8188` |

**Defaults**: a fresh clone with no env vars set assumes a ComfyUI server at `http://localhost:8188` — the standard port when you start ComfyUI manually (`python main.py`) or via Docker.

**ComfyUI Desktop**: the desktop app binds to `http://localhost:8000` instead. If you're using Desktop, set `COMFY_URL=http://localhost:8000` (or persist it in your shell rc).

**Two-server topology** (e.g. separate Flux + LTX boxes for VRAM headroom): set both `COMFY_URL_FLUX` and `COMFY_URL_VIDEO`. The CLI auto-routes per command — agents don't have to track which server.

**OpenClaw sandbox**: agents in an agentic-media sandbox have these vars injected at boot via the per-sandbox `credentials.env` propagation; agent prompts and skill code don't need to mention them.

## Output directory configuration

| Var | Used by | Falls back to |
|---|---|---|
| `OPENCLAW_MEDIA_DIR` | all comfy_graph commands — where final assets land AND where the post-run outbound copy goes | `/workspace/media/outbound` (when in an OpenClaw sandbox) → `~/.openclaw/workspace/media/outbound` (host install) |

Precedence: `OPENCLAW_MEDIA_DIR` > `/workspace/media/outbound` default in sandbox > `--output-dir` CLI arg (i.e. the env var, when set, **overrides** the CLI arg for sandbox writes — set the env var per-sandbox, leave `--output-dir` alone). Mirrors suno-mcp's `SUNO_OUTPUT_DIR` pattern.

**Why it exists**: commons-tier OpenClaw sandboxes bind-mount `/workspace` **read-only**, so the previous hardcoded `/workspace/media/outbound` raised `PermissionError` on `mkdir`. Set `OPENCLAW_MEDIA_DIR` to a writable path (e.g. `/home/sandbox/outbound`) for those sandboxes.

## Script paths (use absolute; tilde expansion is unreliable under `exec`)

The canonical install location is `~/.openclaw/skills/comfyui/scripts/…`. In OpenClaw sandboxes that resolves to `/home/sandbox/.openclaw/skills/comfyui/scripts/…`; on a host install it depends on the user account that ran the install. `~/.openclaw/…` works in an interactive shell but NOT always under `exec` — pass an absolute path when invoking from another agent or process.

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

## Quickstart examples

```bash
# Image generation (text-to-image)
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py t2i \
  --prompt "<your-character> in a forest"

# Image-to-image with reference
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py i2i \
  --image <reference.png> \
  --prompt "<your-character> riding a white horse"

# Text-to-video
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py t2v \
  --prompt "<your-character> walking through ancient ruins" \
  --seconds 5

# Query server state
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_query.py stats

# Run workflow from JSON
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_run.py workflow.json --output-dir /tmp/imgs
```

## Working with named character references

For storyworld-style workflows where you have a named character with
multiple reference images (cataloged on HuggingFace, with structured
yaml metadata), see the companion skill:

  https://github.com/venetanji/polyu-storyworld/tree/main/skills/storyworld-references

That skill handles HuggingFace dataset downloads, character-yaml lookup,
and reference-image selection; it calls into this comfyui skill for
the actual t2i / i2i / video generation. They are intentionally
separate so this skill stays generic.

## Script Inventory

### comfy_graph.py — CLI entry point (most common)
Builds workflows and submits them. All commands support `--output-dir`, `--timeout`, `--seed`.

#### Sandboxed-agent pattern: `--input-json` (no wrapper scripts)

Sandboxed agents (commons, per-role lordships) cannot reliably pass
long prompts via `--prompt "..."` on the shell command line — the
exec preflight rejects "complex interpreter invocation" when prompts
contain quotes, newlines, or shell-special characters. Do NOT write
a per-invocation python wrapper file; instead use the two-step
write+exec pattern:

1. Write a JSON spec via the `write` tool (no shell parsing):

   ```jsonc
   // /workspace/.comfy/job.json
   {
     "prompt": "an elephant playing chess in a smoky 1920s salon, ...",
     "seconds": 8,
     "seed": 42,
     "notify_target": "discord:channel:1234"
   }
   ```

2. Run comfy_graph.py with a single argument — no shell quoting of
   user content:

   ```bash
   python3 /agentic-media/creative-skills/comfyui/scripts/comfy_graph.py \
     t2v --input-json /workspace/.comfy/job.json
   ```

JSON keys map 1:1 to the long-form CLI flag names with dashes
converted to underscores (`notify_target` ↔ `--notify-target`). Any
subset is allowed. CLI flags after `--input-json` still override
individual values, so an ad-hoc re-run with a tweaked seed is one
flag:

```bash
python3 comfy_graph.py t2v --input-json job.json --seed 99
```

The same pattern works for every subcommand (t2i, i2i, t2v, i2v,
ia2v, flf2v, transition, multiguide, multiprompt, i2i2multi,
i2iNmulti, tts, …).

```
t2i          Text-to-image (Flux2)
i2i          Single-reference image edit (Flux2, 1 ref + prompt)
i2i2         Two-reference image edit (Flux2)
i2iN         N-reference image edit (Flux2) — pass --images a.png,b.png,c.png
multiprompt  Many prompts × one reference  — one submission, N outputs
i2i2multi    Many prompts × two references — one submission, N outputs
i2iNmulti    Many prompts × N references   — one submission, N outputs
t2v          Text-to-video (LTX-2.3)
i2v          Image-to-video (LTX-2.3, first-frame)
ia2v         Image + audio → video (LTX-2.3)
flf2v        First + last frame → video (LTX-2.3, converges to the last frame)
transition   Song-aligned cross-scene morph (prev_video + next_video guides + masked middle)
multiguide   Chained LTXVAddGuide — N anchors at N latent positions within one clip
tts          Text-to-speech (Qwen3)
dump         Print workflow JSON only, no execution
last_frame   Extract last frame from a server-side video file
```

### ⚡ When the user asks for "variants" / "multiple versions" / "N angles" / "variations"

**USE `multiprompt` or `i2iNmulti` — NEVER a shell for-loop.** Each of these batches
all prompts into ONE comfy submission (one model load, one queue slot, same reference
reused) and outputs N PNGs with the conventional `_00001_`..`_NNNNN_` suffixes. This
is dramatically faster and the sandbox preflight blocks for-loops / `&& chained &&
commands` anyway.

```bash
# 10 variants of the SAME subject from one reference image — ONE submission:
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py multiprompt \
    --image /workspace/outputs/char.png \
    --prompts "$(printf 'variant 1 description\nvariant 2 description\n…\nvariant 10 description')" \
    --append ". Consistent style: <global style brief>. Preserve same character." \
    --width 896 --height 1664 \
    --prefix char_variants \
    --output-dir /workspace/outputs/variants/
# → char_variants_00001_.png, char_variants_00002_.png, … char_variants_00010_.png

# 8 composite scenes of the SAME two characters — ONE submission:
python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py i2i2multi \
    --image1 /workspace/outputs/snakebird.png \
    --image2 /workspace/outputs/owl.png \
    --prompts "$(printf 'scene A\nscene B\nscene C\n…')" \
    --append ". Preserve exact appearance of both characters. <style brief>"
```

**Prompting each line (edit-style)**: every prompt line should be an edit instruction —
"Take the character from image 1 … place them in … preserve same features …" — not a
standalone scene description. Flux-2 Klein is an image-editing model when given refs;
verbose edit instructions outperform terse scene descriptions.

### comfy_query.py — Server diagnostics

The first thing to reach for when a render fails, the queue is wedged, or you need to know what's installed on the server. All commands honour `COMFY_URL`.

```
stats                          # GPU/RAM usage, comfy version, python/torch versions
queue                          # currently-running prompt + pending list
loras                          # available LoRAs
models [<type>]                # files in a category (loras|diffusion_models|checkpoints|vae|text_encoders|upscale_models|…)
node <ClassName>               # the node's input schema + dropdown enum values
history [<prompt_id>]          # status, executed nodes, errors for a specific run
history --limit N              # the last N runs
```

**Canonical workflow when a render fails:**

```bash
# 1. Pull the failed prompt_id from your render output. It's printed at the start:
#    prompt_id: 7690667a-f37f-497c-b57e-827b1c1d6f49

# 2. Inspect the failure
COMFY_URL=https://media-relay.tail74c072.ts.net:8189 \
  python3 comfy_query.py history 7690667a-f37f-497c-b57e-827b1c1d6f49

# Output shows status (running / error / success), executed nodes, and on error
# the failing node + the upstream node_type + the exception_message.

# 3. If `status: error`, the exception_message is the actual problem.
#    Common patterns:
#      "Conditioning frames exceed the length of the latent sequence"
#         → cond+pass-1-output-length math is off. Common when re-applying
#           IC-LoRA on a latent that wasn't cropped first; or when the
#           cond video is shorter than the requested duration.
#      "Adding guide to a combined AV latent is not supported"
#         → an LTXVAddGuide / LTXAddVideoICLoRAGuide ran AFTER ConcatAV.
#           Always Add{Video}Guide BEFORE ConcatAV in the pass-1 chain.
#      "HTTP 413"
#         → ComfyUI server proxy body-size limit; re-encode the audio
#           (only when running STT, not when ia2v needs full quality).

# 4. If `status: running` for too long, check queue / server stats:
COMFY_URL=… python3 comfy_query.py queue
COMFY_URL=… python3 comfy_query.py stats     # VRAM exhaustion shows up here

# 5. If a node-type validation error suggests an enum/dropdown mismatch,
#    look up the node's allowed values:
COMFY_URL=… python3 comfy_query.py node LTXICLoRALoaderModelOnly
# returns the input schema + the list of available .safetensors filenames
```

**Recovering an output after a client-side timeout:**

`comfy_graph.py` polls `/history/<prompt_id>` every 2s with a 15s per-request timeout. If the bridge hiccups or the server restarts mid-render, the client raises `TimeoutError` even though the file may have been saved. **Don't immediately re-render** — use comfy_query to check first:

```bash
# Did the prompt finish? (history is keyed by prompt_id printed at submit time)
COMFY_URL=… python3 comfy_query.py history <prompt_id>

# Anything else running / queued?
COMFY_URL=… python3 comfy_query.py queue

# Pull the file directly if the server still has it (filename in history outputs):
curl -O "https://media-relay.tail74c072.ts.net:8189/view?filename=<name>&type=output"
```

If history is empty AND queue is empty AND the file is not at `/view`, the server was restarted and the render genuinely needs to be re-run.

### `dump` — print the workflow JSON without submitting

`comfy_graph.py dump <op> <args>` builds the exact workflow that `<op>` would submit and prints it to stdout. Use it to:

- See what the script will send to ComfyUI before running it (sanity-check params).
- Pipe into `comfy_run.py` for fine-grained control (e.g. custom output dir).
- Diff against the official Lightricks workflows at `https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.3` when investigating an output that looks wrong — that's how the 2026-05-16 "glitchy squares" bug was found (our `_upsample_between` was wiring `LTXVLatentUpsampler` from the pre-crop latent; the official lipdub-2-stage wires from `cropped[2]`).

```bash
# Sanity check what would be submitted
COMFY_URL=… python3 comfy_graph.py dump ia2v \
  --image anchor.png --audio slice.mp3 --prompt "..." --seconds 5 \
  --ic_loras "..." --ic_lora_reference_video cond.mp4 \
  > /tmp/wf.json

# Count IC-LoRA nodes, samplers, etc.
python3 -c "import json; d=json.load(open('/tmp/wf.json'));
from collections import Counter;
print(Counter(n['class_type'] for n in d.values()))"
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
| `i2i` | Single-reference edit (1 ref + prompt) | `--image`, `--prompt`, `--width`, `--steps` |
| `i2i2` | Two-reference edit | `--image1`, `--image2`, `--prompt` |
| `i2iN` | N-reference edit | `--images a.png,b.png,c.png`, `--prompt` |
| `multiprompt` | **Batch N prompts × 1 ref** → N outputs / 1 submission | `--image`, `--prompts` (newline sep), `--prepend`, `--append` |
| `i2i2multi` | **Batch N prompts × 2 refs** → N outputs / 1 submission | `--image1`, `--image2`, `--prompts`, `--prepend`, `--append` |
| `i2iNmulti` | **Batch N prompts × N refs** → N outputs / 1 submission | `--images a,b,c`, `--prompts`, `--prepend`, `--append` |
| `t2v` | Text-to-video (LTX-2.3, two-pass) | `--prompt`, `--seconds`, `--fps`, `--width`, `--height` |
| `i2v` | Image-to-video (LTX-2.3, first-frame + refine) | `--image`, `--prompt`, `--seconds`, `--fps` |
| `ia2v` | Image + audio to audio-reactive video | `--image`, `--audio`, `--prompt`, `--seconds`, `--fps`, `--image_refs a,b,c`, `--base_guide_strength 0.5`, `--refine_guide_strength 0.3`, `--identity_anchor`, `--identity_strength 0.3` |
| `flf2v` | First+last frame to video (LTX-2.3, two-pass; default fps=25) | `--first`, `--last`, `--prompt`, `--seconds`, `--fps`, `--guide_strength`, `--use_transition_lora` |
| `continuation` | Extend an existing video (LTX-2.3, two-pass) | `--prev_video`, `--prompt`, `--seconds`, `--audio`, `--overlap_seconds 1.0`, `--overlap_strength 1.0`, `--prev_frames N` (bypass ffprobe) |
| `multiguide` | N image guides at N latent positions (LTX-2.3) | `--guides a.png,b.png,...`, `--frame_indices 0,96,168`, `--strengths 1.0,1.0,1.0`, `--audio`, `--no_transition_lora 1` (transition LoRA is **on by default** here) |
| `transition` | Song-aligned morph between two clips (LTX-2.3) | `--prev_video`, `--next_video`, `--audio`, `--seconds`, `--mask_start_sec`, `--mask_end_sec`, `--use_addguide 1`, `--multiframe_guide 24`, `--multiframe_guide_last 1`, `--b_sparse_latent_positions "96"` or `"72,80,88,96"`, `--ic_loras` + `--ic_lora_reference_video` + `--ic_lora_reference_strength` + `--ic_lora_frame_idx 24` for multiguide stitching (see music-video SKILL.md "Multiguide stitching") |
| `tts` | Text-to-speech (Qwen3 TTS) | `--text`, `--prefix` |
| `stems` | Vocals + instrumental split (MelBandRoFormer) | `--audio`, `--model`, `--prefix` |
| `stt` | Whisper transcription → txt + word/segment SRTs | `--audio`, `--model_size large-v3-turbo`, `--language auto`, `--prefix` |
| `vconcat` | Concatenate clips, optionally with audio | `--videos a.mp4,b.mp4,...`, `--audio`, `--fps 24`, `--trim_durations`, `--trim_starts`, `--fast` / `--no-fast` |
| `last_frame` | Extract last frame from a **server-side** video path | `--video_path` (must be an absolute path on the ComfyUI server, not a local file) |
| `run` | Submit any workflow JSON (file or stdin) | `--file workflow.json` or `cat wf.json \| ... run` |
| `dump` | Print workflow JSON, no execution | prefix any workflow command (e.g. `dump t2i …`) |

### Global options
- `--notify-target <target>` — OpenClaw notification target (e.g. `discord:1486985676066000957`)
- `--output-dir <path>` — local output directory (default: `outputs/`)
- `--timeout <seconds>` — generation timeout
- `--seed <int>` — random seed for reproducibility
- `--fast` *(any video command: t2v/i2v/ia2v/flf2v/continuation/multiguide/transition)* — skip the refine pass. About half the wall time, half the output resolution (no 2× upsample), rougher detail. Use for prompt iteration; leave off for final output.

### Environment variables
- `COMFY_URL_FLUX` / `COMFY_URL_VIDEO` — separate endpoints for the flux (image/TTS/audio) and video servers when you run them on different hosts. In an OpenClaw sandbox these are pre-set via per-sandbox `credentials.env`, so agents can just run `comfy_graph.py t2i …` and the right server is picked automatically.
- `COMFY_URL` — single-server fallback (used as default when the per-class env var is unset, and by `comfy_run.py` / `comfy_query.py` which talk to one server at a time). Default `http://localhost:8188`. Set `http://localhost:8000` for ComfyUI Desktop.
- `OPENCLAW_MEDIA_DIR` — override the sandbox output directory (final asset writes + outbound copy). Full path, used as-is. Falls back to `/workspace/media/outbound` in a sandbox, or `~/.openclaw/workspace/media/outbound` on host. Required on commons-tier sandboxes where `/workspace` is read-only.
- `OPENCLAW_NOTIFY_TARGET` — default notification target

### Discord attachments — `MEDIA:` directive in your reply

Every successful comfy_graph call automatically copies the outputs to
`/workspace/media/outbound/<filename>` (sandboxed agents) or
`~/.openclaw/workspace/media/outbound/<filename>` (host install). In an
OpenClaw sandbox those are the same file via the workspace bind mount.

To attach in a Discord reply, add a line **starting with `MEDIA:`** (only leading
whitespace allowed before it) pointing at the **host-side** path — the reply parser
runs host-side and resolves literal paths only:

```
Here's your image!
MEDIA:$HOME/.openclaw/workspace/media/outbound/flux2_t2i_00012_.png
```

Inline forms (`"... image: MEDIA:/path ..."`) are NOT parsed. One `MEDIA:` line per
file. Do NOT `cp` or `mv` files into random workspace subdirs — core.py already
places them in the right spot.

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
- **Between passes:** Separate AV → `LTXVCropGuides` on the post-pass-1 video latent → `LTXVLatentUpsampler` **on the cropped latent** (NOT on `sep[0]` — see the gotcha below) → re-apply image via `LTXVImgToVideoInplace(strength=refine_guide_strength, default 0.3)` (i2v/ia2v only) → re-concat with audio. The cropped conditioning (LTXVCropGuides output) is what feeds pass-2's CFGGuider.
  - ⚠️ **Gotcha — upsample from CROPPED latent, not separated.** When pass-1 added conditioning frames via `LTXVAddGuide` / `LTXAddVideoICLoRAGuide` (IC-LoRA / id_branch / multiguide / flf2v), the post-pass-1 latent's temporal dim is `requested_length + cond_frames` (e.g. 32 frames for a 16-frame request with 16 cond frames). `LTXVCropGuides` strips the cond frames from BOTH the cond's pos/neg metadata AND the latent (it slices `latent_image[:, :, :-num_keyframes]`). If `LTXVLatentUpsampler` is wired from the pre-crop `sep[0]` instead of `cropped[2]`, pass-2 refines a 2x-temporal latent → 2x-duration glitched output. The basic ia2v 2-stage workflow (no AddGuide) gets away with `sep[0]` because it has no cond frames to strip; the official lipdub-2-stage IC-LoRA workflow correctly wires from `cropped[2]`. Fixed in `ltx2._upsample_between` 2026-05-16 after the stable-altitude v8 "glitchy squares" arc.
- **Pass 2 (refine, 4 steps):** `euler_cfg_pp` + ManualSigmas `"0.85, 0.7250, 0.4219, 0.0"`, cropped conditioning
- **Always AV-concat:** All three variants (t2v/i2v/ia2v) use `LTXVConcatAVLatent`/`LTXVSeparateAVLatent`. For non-audio variants, `LTXVEmptyLatentAudio` provides a blank audio side. This keeps one graph shape.
- **Length:** `(seconds × fps)` rounded to `≡ 1 (mod 8)`
- **Audio for ia2v:** `LoadAudio` → `TrimAudioDuration` → `LTXVAudioVAEEncode` → `SetLatentNoiseMask` (with zero-valued `SolidMask`)
- **flf2v (first-last frame):** adapted from the `Comfy-Org/workflow_templates` flf2v template into the same two-pass shape as the other flows. Pass-1 uses chained `LTXVAddGuide` (frame_idx=0 + frame_idx=-1 at strength 0.7) on the base latent; pass-2 re-injects both guides at strength 1.0 on the upsampled latent. Post-decode `LTXVCropGuides` strips the injected guide frames so the output doesn't show the raw input images as first/last frames (use `--fast` to keep the single-pass shape the template originally had). Default `fps=25` here (every other video command defaults to 24); pass `--fps 24` for concat-friendly output. Transition LoRA is **off by default** (opt-in with `--use_transition_lora`) — the inverse convention from `multiguide`, where it is on by default and disabled via `--no_transition_lora 1`.
- **multiguide (N-anchor):** chained `LTXVAddGuide` calls, one per guide, each
  at its specified latent `frame_idx` (snapped to 8 upstream). Used by
  music-video's `guides:` scene field to anchor identity mid-shot when the
  first frame doesn't show the character, and for scenes ≥13s where
  single-anchor ia2v drifts by the tail. Pass `--no_transition_lora 1` when
  the output is a scene (not a transition) — the transition LoRA pulls hard
  toward endpoint convergence, which is wrong for a normal ia2v-style shot.
- **transition (song-aligned morph between two clips):** pass-1 injects real
  video frames from `prev_video` (A-guide, latent positions 0..N-1,
  strength 1.0) and `next_video` (B-guide, single frame at position 96 by
  default — see `b_sparse_latent_positions`). The masked middle is driven by
  the audio slice via `LTXVAudioVideoMask`. `use_addguide=1` is the mode
  used by music-video — `LTXVAddGuide` natively handles multi-frame IMAGE
  batches; the older `LTXVImgToVideoInplaceKJ` with `num_images=2` only
  picks the first frame of each batch and produces a frozen middle. Final
  decode MUST pass `strip_guides_cond` to remove the injected guide tokens
  from the latent tail, or the clip will run `guide_frames` longer than
  requested. Keep B-side guides sparse (every 8 latent frames); contiguous
  multi-frame B-blocks cause a snap-in freeze where the scene content
  hard-locks. `"96"` (1f) is the default and works for most boundaries;
  `"72,80,88,96"` (4f) gives stronger character establishment going into
  singing / lipsync scenes.

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
2. Translate sandbox → host path: `/workspace/…` → `$HOME/.openclaw/workspace-<agent>/…` on the host (the `message` tool resolves paths on the host, so the host's literal absolute path is what `media:` wants).
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
  media: "$HOME/.openclaw/workspace-<agent>/outputs/flux2_t2i_XXXXX_.png",
  message: "Here's your image."
})
```

### `--notify-target` (advisory only)

`--notify-target discord:<channel_id>` does **not** send anything. It only makes the script print an extra line `[NOTIFY_URL 0] target=discord:<channel> media=<comfy_view_url> caption=<...>` so you have a ready-made URL + caption to feed to `message`. It's convenience, not automation. You still call `message` yourself.

### When to offload to a subagent

Blocking exec is fine for **images (< 30s)** and **short TTS**. For **video** (LTX-2.3 is 2-5 min), spawn a subagent so the main Discord reply doesn't hang:

```
sessions_spawn({
  task: "Generate t2v of '<prompt>' (seconds=5) via /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py, then send the resulting video to discord channel <id> with caption '<caption>'. Use exec + message tools. Host path for message is $HOME/.openclaw/workspace-<agent>/outputs/<file>.",
  runtime: "subagent",
  sandbox: "inherit",
  mode: "run"
})
```

The subagent runs inside the same shared sandbox, does the long gen + delivery, and auto-announces completion to the main session when done. You (the main agent) are free to continue talking.

## Debugging an output that looks wrong

Server-side validation (empty outputs, AssertionErrors, missing nodes) is covered by the `comfy_query.py history <id>` workflow in the comfy_query section above. This section is for the harder case: the workflow ran to completion, the file was saved, but the video looks **wrong** (glitchy noise, doubled duration, wrong identity, conditioning ignored, etc.).

### Step 1 — extract a middle frame and look at it

Before re-rendering or reading code, check what was actually produced. The `imageio_ffmpeg` Python wheel ships a static ffmpeg binary so this works on any host with `uv` available:

```bash
uv run --with imageio-ffmpeg python3 -c "
import subprocess, imageio_ffmpeg as iio, re
ff = iio.get_ffmpeg_exe()
r = subprocess.run([ff, '-i', 'out.mp4'], capture_output=True, text=True)
m = re.search(r'Duration: (\S+)', r.stderr); d = re.search(r'(\d{2,4})x(\d{2,4})', r.stderr)
print(f'duration={m.group(0)}, dims={d.group(0)}')
subprocess.run([ff, '-ss', '2.5', '-i', 'out.mp4', '-frames:v', '1', '-y', 'mid.png'], capture_output=True)
"
```

**What the frame tells you:**

| Symptom | Likely cause |
|---|---|
| Solid noise / raw latent grid pattern | Pass-2 sampling on a latent with conditioning frames not stripped — see the upsample-from-cropped gotcha in the LTX-2.3 architecture section |
| Duration is exactly 2× requested | Same as above: cond frames in the upsampled latent get sampled and decoded as if they were content |
| First frame OK, rest drifts to noise | Pass-1 worked but the refine pass lost the conditioning (model + cond mismatch — e.g. IC-LoRA model with cropped-clean cond) |
| Wrong identity / character | `image_ref` was passed but `condition_only=False` baked it into the latent and IC-LoRA fought for spatial coverage — use `condition_only=True` for IC-LoRA flows |
| Left-half is right, right-half is noise | `ImagePrepForICLora` left-bias bug — for video references, use `ResizeImageMaskNode "scale to multiple" 32` instead (already fixed) |
| Output is `guide_frames` longer than requested | `strip_guides_cond` not passed to `_decode_and_save`, OR passed but the cond's `keyframe_idxs` was cleared upstream of the strip (e.g. crop already ran). Check that the cond going into the final crop still has `keyframe_idxs` |

### Step 2 — compare your workflow against the official Lightricks examples

The single most useful debugging tool: **diff your generated workflow JSON against the official example workflow for the same flow shape**. Official examples are at `https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.3`:

| Flow | Use this example as reference |
|---|---|
| t2v / i2v / ia2v single-stage | `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json` |
| t2v / i2v / ia2v 2-stage refine | `LTX-2.3_T2V_I2V_Two_Stage_Distilled.json` |
| IC-LoRA HDR (single-stage) | `LTX-2.3_ICLoRA_HDR_Distilled.json` |
| IC-LoRA Union-Control (single-stage) | `LTX-2.3_ICLoRA_Union_Control_Distilled.json` |
| IC-LoRA Motion-Track (single-stage) | `LTX-2.3_ICLoRA_Motion_Track_Distilled.json` |
| **IC-LoRA 2-stage** | `LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json` ← the only 2-stage IC-LoRA example |

Workflow:

```bash
# 1. Dump what your script produces (API format)
python3 comfy_graph.py dump <op> <args> > /tmp/ours.json

# 2. Fetch the official UI-format example
gh api repos/Lightricks/ComfyUI-LTXVideo/contents/example_workflows/2.3/LTX-2.3_ICLoRA_Lipdub_Two_Stage_Distilled.json \
  --jq '.content' | base64 -d > /tmp/official.json

# 3. Diff the topology. The two formats are different (API vs UI), so trace
#    node-by-node which output feeds which input. Focus on the SAMPLER's
#    inputs (latent_image, guider) and the upstream chain feeding them.
```

The 2026-05-16 "glitchy squares" fix was found exactly this way: our `_upsample_between` wired `LTXVLatentUpsampler.samples` from the post-pass-1 separated latent (`sep[0]`); the official lipdub-2-stage workflow wires it from `LTXVCropGuides.latent` output (`cropped[2]`). That single mis-wire caused the pass-2 sampler to refine 2× as many latent frames as the user requested, producing the doubled-duration glitched output.

### Step 3 — re-run with the smallest possible reproduction

Don't re-render the full project. Pull one short scene out of the project YAML, render it standalone via `comfy_graph.py ia2v` directly with all the same flags, and iterate on that. A 4-second scene takes 1-3 minutes vs an hour for the full music video.

### Step 4 — bisect against working configurations

If a parameter change broke the output, render with the LAST KNOWN GOOD params alongside the broken ones (same prompt, same image, same audio). When both sit side-by-side in the output directory, the structural difference is usually obvious.

The matrix that works on the LTX-2.3 server today (as of 2026-05-16):

| `fast` | IC-LoRA | Works |
|---|---|---|
| true  | no  | ✓ single-pass |
| true  | yes | ✓ single-pass at requested dims (use 2× dims if you want high-res from IC-LoRA + Union-Control) |
| false | no  | ✓ standard 2-stage refine |
| false | yes | ✓ 2-stage refine (after the cropped-latent fix landed 2026-05-16) |

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
