# Easel generation through API, CLI and agents

Use this route when the user names Easel or works inside Easel Client. Keep
ComfyUI graph construction on Easel's Python server. Internal agents, external
MCP agents and CLI users should consume the same validated API instead of
copying old graph builders into JavaScript or bypassing managed assets.

## Discover before choosing advanced controls

- API: authenticated `GET /v1/videos/capabilities` and `GET /v1/videos/loras`.
- CLI: `easel video capabilities` and `easel video loras`.
- MCP: `discover_video_capabilities({model})` and `list_video_loras({model})`.
  Use the exact enabled ID from `list_models`; an in-app ID may contain an
  endpoint prefix. The selected model supplies its own endpoint credentials.

Capabilities are `object:"video.capabilities"`, `schema_version:1`. They state
model, FPS, durations, sizes, exact seed bounds, upload limit and guide-node
availability. Unsupported/older endpoints fail discovery explicitly; never
invent support from a model name. LoRA `supported`, live `installed`, required
inputs and `validation` are separate facts. Check all of them. An installed
weight or available node does not establish quality, compatibility or execution.

The generation-controls changes may still be draft/unreleased when this guide
is read. Tool schemas and the live endpoint are authoritative. If the required
field or discovery tool is absent, stop that path and report the missing
capability; do not silently drop it or use direct ComfyUI as a fallback.

## Typed option map

| Meaning | HTTP multipart | MCP input | CLI |
| --- | --- | --- | --- |
| Camera | `camera_lora`, `camera_lora_strength` | `cameraLora`, `cameraLoraStrength` | `--camera-lora`, `--camera-lora-strength` |
| Curated adapters | `loras` JSON array | `loras:[{id,strength?}]` | repeated `--lora ID[=STRENGTH]` |
| Exact seed | `seed` decimal text | `seed` decimal **string** | `--seed` exact integer |
| First image | `input_reference` file | app `inputReferenceAssetId`; standalone `inputReference` | `--input-reference` |
| Ingredients sheet | `lora_reference` file | app `loraReferenceAssetId`; standalone `loraReference` | `--lora-reference` |
| Sheet strength | `lora_reference_strength` | `loraReferenceStrength` | `--lora-reference-strength` |
| Slow motion | `motion_speed` | `motionSpeed` | `--motion-speed` |
| Timed images | `guiding_frames` metadata + repeated `guiding_images` files | `guidingFrames` | repeated `--guide-frame FRAME IMAGE STRENGTH` |

MCP standalone image uploads have `{data:<base64>,mimeType,name?}`. In-app
reference arguments carry only real saved asset IDs, resolved by the host from
the chosen project or shared library. Use `list_media_assets` and inspect the
chosen images. Never invent IDs, put arbitrary paths/URLs in tool fields or
paste credentials into agent arguments. All uploaded references share 32 MiB.
CLI paths identify local regular files which are uploaded; they never become
ComfyUI server paths. Legacy CLI `--loras` is validated JSON and is mutually
exclusive with repeated `--lora`.

At most four distinct curated LoRAs, counting camera shorthand; strengths are
finite 0–2. Camera defaults to 0.8, other LoRA weights to 1. No arbitrary filename
or URL is accepted. Keep model weights distinct from guide/reference strength.

- `cinemagraph` needs the first-image reference and rejects moving-camera LoRAs.
- `slow-motion` needs first image and `motionSpeed` 0.025–1; this conditions new
  generated motion, not deterministic interpolation of an existing clip.
- `ingredients` needs its own reference sheet, at least five seconds, no other
  LoRA stack, and `loraReferenceStrength` 0–1 (default 1).

## Timed still-image anchors

Easel LTX-2.5 keeps 24 FPS and `seconds*24+1` output frames. Whole durations are
1–12 seconds. A two-second clip therefore has positions 0..48 inclusive.
`frameIndex` is an integer **pixel-frame position**, not seconds or a latent
frame index. Single-image anchors do not require a multiple of eight.

One through eight anchors are allowed. Positions must be distinct, within the
original clip, strengths finite 0–1 (default 1). The same strength is used in
both sampling passes. Images must be valid still PNG/JPEG/WebP; APNG/animated
WebP are rejected because multi-frame guidance has different semantics.
Guides are limited to 32 megapixels each and 32 MiB total upload bytes.

In-app request for two anchors at pixel frames 1 and 48:

```json
{
  "model": "<exact-enabled-video-model-id>",
  "prompt": "One continuous locked-camera shot; the subject turns toward the window.",
  "seconds": 2,
  "size": "512x320",
  "seed": "18446744073709551614",
  "loras": [{"id":"camera-static","strength":0.8}],
  "guidingFrames": [
    {"assetId":"<real-first-saved-image-id>","frameIndex":1,"strength":0.7},
    {"assetId":"<real-last-saved-image-id>","frameIndex":48,"strength":0.5}
  ]
}
```

Placeholders above must be replaced with discovered model/asset IDs. Standalone
MCP replaces each `assetId` with `image:<ImageUpload>`. HTTP metadata uses
`[{"image_index":0,"frame_index":1,"strength":0.7},...]`, one-to-one with the
ordered repeated `guiding_images` uploads. Do not include a local path in JSON.

Equivalent CLI:

```bash
easel video create "One continuous locked-camera shot; the subject turns toward the window." \
  --seconds 2 --size 512x320 --seed 18446744073709551614 \
  --lora camera-static=0.8 \
  --guide-frame 1 first.png 0.7 --guide-frame 48 last.png 0.5 \
  --receipt guided-shot.json
```

Timed-guide mode cannot also use first-image `inputReference`, Ingredients or
its reference sheet. Cinemagraph/slow-motion need first-image mode, so cannot
be mixed into timed-guide mode. Unknown keys and invalid combinations fail
rather than being ignored. A reference/preflight validation failure means
correct the request; an ambiguous submission error means do not resubmit.

Guide strength 1 is still soft conditioning, not an exact-pixel lock, seamless
stitch, deterministic camera path or perfect loop. The source-backed two-pass
graph is `graph_contract_tested`, pending explicit GPU/visual validation. Inspect
first/middle/last frames, requested anchor positions, timing, generated audio
and decoded geometry in an authorized bounded pilot before scaling.

## Submit once and keep the receipt

Save the accepted job ID immediately. Easel Client persists and monitors accepted
jobs across restarts and brings the output back to the original conversation.
Let the host poll; do not repeatedly spend agent tool calls or generate a
replacement. Standalone MCP retrieves with `get_video` using the original ID
and same model. CLI uses `video status`, `video queue`, `video wait`, and
`video download` with the original ID. Resume is GET-only.

Stop/timeout does not cancel an accepted server job. There is no idempotency or
receipt-loss recovery endpoint, so a lost POST response is ambiguous. Never
blindly retry or switch endpoints. Video retention is 24 hours and still relies
on ComfyUI queue/history/output. Queue estimates are approximate, not measured
sampler progress or a promise of completion time.

## Source and graph boundary

Easel's [API documentation](https://github.com/venetanji/easel/blob/main/VIDEO_API.md)
and [adapter registry](https://github.com/venetanji/easel/blob/main/easel/video_loras.json)
are the current HTTP authority. This repository's published `ltx2.py` remains
legacy LTX-2.3; the separately described canonical LTX-2.5 runtime was not
published with PR #11. Do not claim a canonical import/migration is complete.

Official ComfyUI [LTX-2.5 first/last blueprint](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/blueprints/First%20%26%20Last%20Frame%20to%20Video%20(LTX-2.5).json)
and [AddGuide node source](https://github.com/Comfy-Org/ComfyUI/blob/e9027f2b30f37bb3052714eb08fcf479542f4fc0/comfy_extras/nodes_lt.py#L251-L519)
justify ordinary still-image guides with Easel's model. They are added to
video-only latents before AV concatenation; all guide slots/conditioning are
cropped before upscale, then reapplied and cropped again before decode. This
preserves requested length and generated audio. The official blueprint is
single-stage; Easel's two-pass composition needs its own execution validation.

Union/depth/Canny/pose **control videos**, source-audio conditioning, continuation,
transitions, custom checkpoints/samplers and full arbitrary ComfyUI graphs are
separate capabilities and remain outside this contract. IC reference pass
policy varies by model/workflow; do not copy timed-still-guide reinsertion into
an unsupported full-video IC recipe.
