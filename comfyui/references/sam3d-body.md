# SAM 3D Body

Based on the [official ComfyUI tutorial](https://docs.comfy.org/tutorials/utility/sam3d-body)
and [native workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/utility_sam3d_body.json).
This is human mesh extraction from video, not SAM 3D Objects reconstruction.

## Setup

The server needs the native `SAM3_VideoTrack`, `SAM3DBody_Loader`,
`SAM3DBody_Predict`, `SAM3DBody_FaceExpression`, `SAM3DBody_Smooth`,
`SAM3DBody_Render`, `BuildPoseFile`, `Preview3D`, and MoGe / RT-DETR nodes.
Check `/object_info`; update ComfyUI only if these are missing. No additional
custom-node pack is needed for this workflow.

Download all four models into the **server's** model directory:

```bash
python3 comfyui/scripts/sam3d_models.py --models-dir /path/to/ComfyUI/models
python3 comfyui/scripts/sam3d_models.py --models-dir /path/to/ComfyUI/models --check
```

| Directory | Model | Approximate Size |
|---|---|---|
| `checkpoints/` | `sam3.1_multiplex_fp16.safetensors` | 1.75 GB |
| `detection/` | `sam_3d_body_dinov3_bf16.safetensors` | 2.83 GB |
| `geometry_estimation/` | `moge_2_vitl_normal_fp16.safetensors` | 0.66 GB |
| `diffusion_models/` | `rt_detr_v4-x-hgnet_fp32.safetensors` | 0.25 GB |

The downloader uses the official Comfy-Org Hugging Face repositories, skips
complete existing files, and validates safetensors data offsets before renaming
a `.part` download. Reserve at least 6 GB free disk space. Face-expression
weights and the MHR rig are included in the body checkpoint; no separate
MediaPipe download is required by the tested native implementation.

## Run

```bash
export COMFY_URL_VIDEO=http://localhost:8188
python3 comfyui/scripts/comfy_graph.py sam3d --video person.mp4 \
  --seconds 5 --batch-size 4 --output-dir outputs/sam3d

python3 comfyui/scripts/comfy_graph.py dump sam3d --video person.mp4 \
  --seconds 0.25 --batch-size 1
```

Default pipeline: video slice -> SAM3 tracking + RT-DETR person boxes ->
MoGe vertical FOV -> SAM 3D Body prediction -> face expressions -> temporal
smoothing -> animated GLB + mesh-overlay MP4. RT-DETR boxes are supplied as
the native predictor's fallback; tracking data takes precedence when present.

Outputs download locally and print `saved:` paths. `Preview3D` saves the GLB
on the server with a `preview3d_<uuid>.glb` name; the video uses
`video/SAM3D_body` unless `--prefix` overrides it. Native 3D preview results
use string paths, so the client handles these separately from image/video
asset dictionaries. Source FPS and sliced audio are retained.

| Option | Default / Meaning |
|---|---|
| `--seconds` | `5`; `0` processes the entire remaining video |
| `--start-time` | `0`; start offset in seconds |
| `--batch-size` | `4`; body prediction crops per chunk (use `1` for smoke tests) |
| `--moge-batch-size` | `1`; geometry-estimation images per batch |
| `--no-tracking` | Skip SAM3; single-person RT-DETR fallback only |
| `--no-moge` | Skip automatic camera FOV estimation |
| `--no-face` / `--no-hand-refinement` | Skip those refinements |
| `--no-overlay` | Render mesh on black instead of source video |
| `--no-audio` | Omit source audio from the MP4 |
| `--export-style` | `body_mesh` (full rigged mesh), or `scail` (template-style capsules) |
| `--prompt` | `person`; tracking text prompt |
| `--detection-threshold` / `--max-people` | `0.5` / `4` |
| `--body-model`, `--track-model`, `--moge-model`, `--detector-model` | Override model filenames |
| `--timeout` | `1800` seconds; increase for longer clips |

## Verify

Offline tests do not require ComfyUI, models, or a GPU:

```bash
python3 comfyui/scripts/test_sam3d_body.py
```

Run the entire pipeline using the official tutorial input:

```bash
curl -fL https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/woman_holding_water_glass.mp4 \
  -o /tmp/woman_holding_water_glass.mp4
python3 comfyui/scripts/test_sam3d_body.py --live \
  --video /tmp/woman_holding_water_glass.mp4 --seconds 0.25 --batch-size 1 \
  --url "$COMFY_URL_VIDEO" --output-dir test_outputs/sam3d
```

The live test requires successful server history, downloaded MP4 and GLB,
and a valid GLB header containing meshes and animations. When `ffprobe` is
available, it also verifies decodable video frames and records media metadata. It writes
`workflow.json`, `history.json`, and `result.json` alongside the media.
An execution error, including errors after a partial GLB preview, fails the
command rather than reporting success. For the single-person fallback test,
add `--low-memory` (disables tracking, MoGe, face and hand refinement).

Batching reduces inference VRAM but not the RAM needed to decode the video.
Start with a short slice and increase gradually; multi-person tracking and
hand refinement cost additional memory. Do not interrupt other queued jobs
or restart the server just to clear VRAM.

## Verified Installation (2026-10-09)

Tested on the existing Docker installation, ComfyUI `0.37.0`, Python `3.12.3`,
PyTorch `2.14.0+cu130`, using the official five-second tutorial video.
All four models were downloaded and validated; no ComfyUI update, restart,
custom-node installation, or extra MediaPipe model was needed.

| GPU | Test | Result |
|---|---|---|
| RTX 3090, 24 GB (`localhost:8190`) | Full pipeline, 5 s, body batch 4 / MoGe batch 1 | 150-frame 1280x720 H.264 overlay at 30 FPS, AAC audio, animated body GLB |
| RTX 3080 Ti, 12 GB (`localhost:8189`) | Full pipeline, 0.25 s, body batch 1 | 8-frame overlay and animated body GLB |
| RTX 3080 Ti, 12 GB | Single-person `--low-memory` smoke test | Animated body GLB and overlay with audio |
| RTX 3090, 24 GB | CLI `--export-style scail`, 1 s starting at 1 s | Capsule GLB and overlay video |

The five-second body GLB contains one skinned mesh, a 127-joint skeleton,
and one animation with 150 keyframes. Server execution took approximately
117 seconds. The overlay was decoded with `ffprobe` and visually checked.
These tests do not establish limits for longer or multi-person clips.
