# WebGL space -> depth/edges -> LTX Union video

## Reading map

- [Source authoring](#1-build-an-inhabited-space-not-a-floating-object-gallery),
  [depth/Canny preparation](#2-export-calibrated-registered-depth-or-choose-real-canny)
  and [lossless timing/export](#3-preserve-the-artifacttiming-contract).
- [Union two-pass graph](#4-select-the-union-graph-branch-not-the-rgb-anchor-branch)
  and [terminal/media gates](#5-verify-and-show-the-accepted-example-honestly).
- [Ordinary I2V handoffs](#stitching-lesson-first-frame-conditioning-must-be-10),
  [periodic-source limits](#optional-periodic-source-does-not-guarantee-a-generated-loop)
  and [non-causal drift diagnostics](#diagnose-drift-without-inventing-a-cause).
- [Canny props](#worked-canny-branch-shapes---carried-props),
  [projected-depth sculpture](#worked-depth-branch-projected-tesseract---kinetic-sculpture)
  and [capability/proof matrix](#capability-and-proof-matrix).
- [Audio/bounded execution](#source-audio-timing-and-bounded-execution),
  [embedded provenance](#embedded-workflows-and-portable-provenance)
  and [typed Easel harness boundary](#next-harness-easel-client-storyboards).

## Scope and branch choice

Author a small 3D scene to specify spatial layout, occlusion and camera movement;
let LTX supply the final visual treatment. This is ControlNet-style guidance via
LTX **Union IC-LoRA**, an actual control adapter plus per-frame conditioning,
not a generic grayscale reference or an RGB first-frame appearance anchor.

- **Verified worked example:** camera inside an ascending staircase, real
  WebGL camera depth -> Union -> LTX-2.5. Gio accepted it as "good enough" to
  illustrate the technique on 2026-09-30; rails/arches still drift.
- **Verified spatial pilots:** approved faster cube/sphere/cylinder RGB -> actual
  Canny plus declared dilation -> three adults carrying metal box/red ball/wood
  cylinder; approved projected tesseract camera depth -> glass/gold kinetic
  sculpture. Source geometry/kinematics are proven; generated exact 2x speed,
  16-vertex/32-edge topology and perfect looping are not.
- **Controlled strength comparison:** video-guide 0.6 -> 1.0, all other model,
  seed, prompt, audio and topology settings fixed. Changes were modest for
  Canny; depth became a cleaner/open cube-like cage with fewer bars. Higher
  strength is not a universal fidelity improvement.
- **Optional periodic source:** a looping authored world is possible, but source
  periodicity is not proof of a seamless generated video or audio loop.

Do not run GPU work merely because this reference exists. Produce and share the
source RGB/control previews first; obtain approval of the actual guide before
upload/submission. Keep media and pilot scripts outside the repository.

## 1. Build an inhabited space, not a floating-object gallery

Use Three.js (or equivalent WebGL) with a pinned local library and recorded
integrity. For the staircase, use 15 static modules: each advances 5.8 m and rises
1.76 m; eight 0.55 m treads / 0.22 m risers, followed by a 1.4 m landing. Include
solid step/riser meshes, enclosing side walls, barrel vault, foreground posts,
rails and repeated distant arches. The 4.8 m corridor has a 4.4 m roof peak.
Near/mid/far surfaces and parallax make camera motion readable; an isolated
floating knot was not accepted as evidence of motion *through* space.

Scene-authoring pseudocode (helpers construct solid meshes, not edge overlays):

```js
for (let module = 0; module < 15; module++) {
  const z0 = module * 5.8, y0 = module * 1.76;
  for (let step = 0; step < 8; step++) {
    addSolidTreadAndRiser(z0 + step * 0.55, y0 + step * 0.22);
  }
  addLanding(z0 + 4.4, y0 + 1.76, 1.4);
  addSideWallsAndBarrelVault(z0, y0, 5.8);
  addArchRibsPostsAndHandrails(z0, y0);
}
// Fixed geometry; advancing first-person camera stays above the floor.
for (let f = 0; f < 97; f++) {
  const t = f / 24;
  updateCameraFromRecordedPath(t);
  camera.updateMatrixWorld();
  renderRegisteredRgbAndDepth();
}
```

Baseline camera: perspective FOV 68 degrees, near/far 0.1/160 m, initial
(x,y,z)=(0,1.75,0.25), eye 1.65 m above a smoothed stair-floor profile. Look
10 m forward and 2.65 m up; mild lateral motion x=0.12*sin(0.5*t). Actual forward
travel was 6.307 m, rise 1.963 m; eye above actual steps 1.64..1.86 m and minimum
wall clearance 2.28 m. Log poses, view/projection matrices and clearances;
check floor/wall intersections at intermediate times, not just endpoints.

Optional audio-driven source motion: measure causal RMS/FFT features from the
actual exported PCM at each frame (48 kHz -> sample 2000*f at 24 FPS). The
baseline used measured kick RMS: speed=1.55+0.16*kick m/s, integrated with the
trapezoid rule; geometry remained static. It reused a diagnostic owned beat,
not arbitrary music. Retained feature metadata from earlier experiments is not
proof those mappings were used: record the scene's actual feature-to-motion
mapping. Measured feature response lagged impulses by one frame. This source
mapping does not establish that the generated camera obeys every beat.

Done when RGB previews visibly show traversal inside the environment and pose /
clearance checks pass on the full path.

## 2. Export calibrated, registered depth (or choose real Canny)

Evaluate animation/camera **once** per frame. Render RGB into a WebGL render
target with a real `DepthTexture`; sample that exact depth attachment in a
fullscreen post-pass. Do not render a second independently animated scene.
For the usual perspective depth-buffer sample d in [0,1], reconstruct positive
axial camera view-Z, not Euclidean camera-to-surface distance:

```glsl
float z = near * far / (far - d * (far - near));
float g = clamp(1.0 - log(1.0 + max(0.0, z - 0.5) / 1.5)
                      / log(1.0 + (45.0 - 0.5) / 1.5), 0.0, 1.0);
// Background/no surface is black. Round 255*g when storing RGB8.
```

This is a fixed-global, near-bright **logarithmic axial-depth visualization**:
not metric depth storage, not reciprocal 1/z, not learned VDA/DepthAnything.
Keep the numerical guide out of RGB tone mapping/gamma; output three identical
8-bit gray channels. RGB uses its intended display color space separately.
Explicitly flip WebGL bottom-left readback to top-left frame order for *both*
exports. Do not normalize each frame independently, change contrast over time,
or lose aspect by square-padding an HDR-style reference sheet.

Calibrate against known near/far geometry, occlusion and a displaced vertex;
check top/bottom orientation. Raycast representative actual floor/riser/wall /
rail/vault pixels and compare axial-Z to the encoded value. Baseline tolerance
was +/-2 gray levels, >99.6% occupied environment, 24,612 rendered triangles.
If using authored/stylized distance bands instead, label them honestly and
validate separately; they are not camera-depth calibration evidence.

Preprocessing choice:

| Input | Control preparation | Status in this example |
|---|---|---|
| Shared-scene WebGL depth visualization | Feed directly; no learned estimator or Canny | Executed baseline |
| Ordinary RGB where depth is needed | Use an appropriate depth preprocessor, record model/range | Separate unvalidated branch |
| WebGL RGB where silhouette edges are needed | Actual Canny white edges on black, record implementation/thresholds | Faster-props local pilot verified; exact motion unproven |

Union consumes the prepared IMAGE batch through the same IC-LoRA guide branch;
do not label wireframe/depth as Canny or run Canny on an already prepared depth
guide. Inspect the pinned official workflow's chosen preprocessing branch and
live node schema rather than guessing a control-type CLI switch.

Done when paired RGB/guide contact sheets, calibration and gray-channel checks
pass, and the user has seen the actual conditioning guide.

## 3. Preserve the artifact/timing contract

Export 97 frames, indices 0..96 at 512x384 / 24 FPS (`8k+1`). Sample t=f/24:
last sample t=4 s, then hold it for 1/24 s; video duration is **97/24 s**, not
4 s. Save `rgb-frames/%03d.png`, `guide-frames/%03d.png`, poses/matrices,
feature mapping and hashes in a media job directory. Optional source audio is
mono PCM16 48 kHz WAV, exactly 194,000 samples for this span.

Working export commands, after generating those PNG sequences (not new CLI
capabilities; no GPU or network):

```sh
# Run in your media source directory; these commands create master files.
ffmpeg -hide_banner -loglevel error -framerate 24 -i guide-frames/%03d.png \
  -frames:v 97 -c:v ffv1 -level 3 -pix_fmt bgr0 guide-master.mkv
ffmpeg -hide_banner -loglevel error -framerate 24 -i rgb-frames/%03d.png \
  -frames:v 97 -c:v ffv1 -level 3 -pix_fmt bgr0 rgb-master.mkv
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,pix_fmt,r_frame_rate,nb_read_frames \
  -of json guide-master.mkv
ffmpeg -hide_banner -loglevel error -xerror -i guide-master.mkv -f null -
```

FFV1/Matroska `bgr0` is lossless RGB-family storage, **not native RGB24**.
Decode every master to RGB24 and compare all 97 frame bytes against the PNGs
(57,212,928 bytes per sequence). Check three equal guide channels and unchanged
frame ordering/FPS/aspect. Make H.264/AAC MP4s only for human review; never feed
lossy review exports as conditioning masters.

Run WebGL in a sandboxed, isolated browser/profile with network denied and a
private CDP pipe; do not expose home/keys or use `--no-sandbox` / unsafe software
renderer opt-ins. The baseline used software-backed browser WebGL2/SwiftShader,
11.069 s for 97 frames, not a custom CPU rasterizer. Record renderer identity;
fail on unavailable/lost context rather than silently changing render methods.

Done when metadata, all-frame decode/byte comparison, source-audio span and
representative frames agree with the recorded contract.

## 4. Select the Union graph branch, not the RGB-anchor branch

**Portable caller gap:** at source `394142a243104d32baaf163346163bded5447210`,
`comfyui/scripts/ltx_profile.py` admits t2v/i2v/ia2v/continuation/flf2v but no
Union control-video field; `creative_comfy_graph.ltx_profiles.build_video`
rejects unverified options. The legacy LTX-2.3 builder has video IC-LoRA
references, but that is not this LTX-2.5 composition. There is no working
`comfy_graph.py --control-video` command to advertise. Use a reviewed explicit
Comfy API graph in authorised prototype scope; typed caller integration remains
a separate code task/blocker, not a reason to invent flags or expand this task.

Baseline authority: `Lightricks/ComfyUI-LTXVideo` commit
`5722b53a08daec57f5442347b235cb84df0afbe7`,
`example_workflows/2.5/LTX-2.5_ICLoRA_Union_Control_Distilled.json` plus
`LTX-2.5_A2V_Two_Stage_Distilled.json`. The pilot combined these topologies;
it was not an unmodified stock example. Keep explicit `ltx-2.3` and `ltx-2.5`
profile identities; this recipe validates only the listed local 2.5 combination,
not silent substitution between profiles. Use no Partner/API nodes or alternate
backend for this local workflow. Verify current local node schemas and
asset compatibility before a new run. The executed assets were:

| Role / loader | Asset |
|---|---|
| `UNETLoader` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` |
| `CLIPLoader` type `ltxv` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` |
| `VAELoader` video | `ltx-2.5-video-vae-bf16.safetensors` |
| `VAELoader` audio | `ltx-2.5-audio-vae-bf16.safetensors` |
| `LatentUpscaleModelLoader` | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| `LTXICLoRALoaderModelOnly` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` |

The 2.3-named Union asset was used with the 2.5 model after compatibility checks:
480 patch pairs matched and **480 patches actually attached in each executed
pass**, without unloaded-key/shape/factor fallback warnings. Do not infer that
any 2.3 LoRA works on 2.5 from this single asset's result.

Executed node recipe (arrows below are wiring, not runnable CLI commands):

```text
UNETLoader -> LTXICLoRALoaderModelOnly(strength_model=1.0)
  output[0] -> BOTH CFGGuider.model
  output[1] -> LTXAddVideoICLoRAGuide.latent_downscale_factor (actual factor 2)
CLIPTextEncode positive + negative -> LTXVConditioning(frame_rate=24)
LoadVideo(lossless guide) -> GetVideoComponents.image
  -> ImageScale(bilinear, 256x192, crop=disabled)
EmptyLTXVLatentVideo(256x192, length=97)
  + scaled guide + video VAE + positive/negative
  -> LTXAddVideoICLoRAGuide(frame_idx=0, strength=0.6,
       crop=disabled, use_tiled_encode=false, tile_size=256, tile_overlap=64)
  -> guided positive[0], negative[1], video latent[2]
LoadAudio(source WAV) -> TrimAudioDuration(start_index=0, duration=97/24)
  -> LTXVAudioVAEEncode -> SetLatentNoiseMask(SolidMask(value=0))
guided video + frozen audio -> LTXVConcatAVLatent
  -> SamplerCustomAdvanced(seed=42030, CFG=1, euler_ancestral, D8)
  -> LTXVSeparateAVLatent.video
  -> LTXVCropGuides(ALL guide positive, negative, video latent)
  -> LTXVLatentUpsampler(cropped latent, x2 model, video VAE)
upsampled video + RESET original encoded audio with zero mask
  -> LTXVConcatAVLatent
  -> SamplerCustomAdvanced(seed=42030, CFG=1, euler, R3)
  -> LTXVSeparateAVLatent.video
  -> VAEDecodeTiled(tile_size=512, overlap=64,
       temporal_size=128, temporal_overlap=32)
  -> CreateVideo(fps=24, audio=ORIGINAL trimmed WAV, bit_depth=8,
       color_space=sRGB, codec=none) -> SaveVideo(mp4/h264)
D8 = 1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0
R3 = 0.909375,0.725,0.421875,0.0
```

Source 512x384 -> coarse 256x192 -> factor-2 internal reference 128x96;
real latent x2 -> final 512x384. Crop **latent AND positive/negative guide
conditioning before upsampling**; do not upsample appended guide slots or
reinsert the full video in refinement. Refine uses the cropped conditioning and
the Union-loaded model. Frozen audio conditioning and final direct-source mux
are separate requirements; an audio stream alone proves neither.

Prompt skeleton: "Continuous first-person shot climbing a deep arched limestone
stairwell, teal walls, bronze rails, repeated landings and arches. Preserve the
structural depth guide, foreground steps, side-wall layout and vanishing point.
Advance smoothly forward/up with near-post parallax. Empty environment, no
cuts, sudden turns, floating objects or text." Keep an explicit negative prompt
for cuts, flicker/jitter, broken silhouettes and distorted/unrelated geometry.
The staircase baseline used video guide 0.6 / adapter 1 / seed 42030. The
separate spatial pilots tested video strengths 0.6 and 1.0 with the same
adapter/seed. Neither establishes optimality, a no-guide ablation or robustness.

Done before submission when the explicit graph matches schemas/DAG, uploaded
source binding/hashes, audio duration and crop topology; share guide approval
and authorised single-run scope. Submission itself is not completion.

## 5. Verify and show the accepted example honestly

Require terminal successful history, expected output selection, 97 decoded
frames at 512x384 / 24 FPS, full A/V decoder passes, all-frame comparison/contact
sheets and native-resolution representative frames. Check direction, parallax,
steps/walls, occlusion and drift rather than judging only a pretty first frame.
Compare final decoded audio content/alignment with the supplied WAV.

Local evidence root (not bundled assets or a portable dependency):
`/home/venetanji/.openclaw/workspace-creative-skills/media/ltx25-staircase-poc-20260930/`.
Use `source/scene.js`, `manifest.json`, `validation.json`, `depth-samples.json`
and `ltx/prepare_union.py`, `submitted-graph.json`, `terminal-history.json`,
`runtime-lora-qa.json`, `actual-guide-runtime-decode.json`, `audio-qa.json` and
`qa-report.md`. Preparation manifests retain historical placeholder/acceptance
states: actual submitted graph + source binding + terminal history + runtime
QA, followed by Gio's acceptance, establish this example's outcome.

- Prompt `6d035e11-752d-4dc7-9967-ad75e03edf4e`: one successful run, 26.86 s.
- `ltx/result.mp4`: 305,452 bytes, H.264/AAC, 4.041667 s; SHA256
  `0b41dd6dbc7de8779b745fd75335148b9d36cc53a97395e6cbec4f70d1215f66`.
- `ltx/review-guide-vs-ltx.mp4`: 592,009 bytes, labeled depth left / generated
  video right, supplied beat; SHA256
  `df6f25267b178cbbdf6844c80d77407b8e7f39f295fe5af22b2aa954a9585c0c`.
- Source/AAC correlation 0.982469, best lag 0 samples. Decoded AAC has 193,968
  samples: final 32 omitted samples are source silence from MP4 duration
  rounding, explicitly accounted, not hidden time stretching.
- Readable steps, teal vault, rails and continuous travel; rail/arch detail
  drifts. Strict physical camera match and production robustness remain unproven.

This is sufficient to showcase authored spatial guidance, not a promise of
exact scene reconstruction. Finish by showing paired source/guide/output media
and recording human review separately from machine success.

## Stitching lesson: first-frame conditioning must be 1.0

Gio closed the staircase/material experiment on 2026-09-30. For a stitched
successor, use the actual preceding native last frame with **first-frame
conditioning 1.0 in both applicable passes**. Do not use 0.7 soft appearance
conditioning as a shared-boundary constraint. The ordinary I2V recipe has **no end-frame constraint**; do not add one as a
second variable. A separately requested terminal constraint is a different
workflow requiring its own validation. These are image-reference strengths,
not the depth/Canny video-control value or LoRA strength; do not indiscriminately
change every guide to 1.0.

The ten-material test correctly uploaded the actual preceding native frame 96
at all nine joins, verified against backend PNG pixels and executed history.
Nevertheless, strength-0.7 conditioning regenerated a different first output
frame. Assembly then skipped that generated frame 0 as if it were an identical
shared boundary and cut to frame 1, generally increasing the visible difference.
Correct extraction/upload is not correct stitching. Materials were mixed and
stairs/arches drifted cumulatively; no seamless result was certified.

Verify input provenance, actual native output frame 0, the first retained
playback frame, geometry and motion across the join. Use a genuinely shared
boundary frame once and record the complete frame/overlap/audio mapping. Never
discard a regenerated start on an assumed duplicate-pixel basis or hide a bad
join with undisclosed fades/holds. Strength 1.0 is the required stitching policy,
not automatic proof of pixel-identical VAE/codec reconstruction or continuous
velocity. Single-frame I2V chaining is not prior-video motion-history context.
No corrected strength-1 render preceded closure of the ten-stage experiment.
A separately authorised one-render Ice comparison subsequently tested strength
1.0 in both image passes, leaving the looping depth-video guide at 0.6, without
end frame or prior-video history. Its actual graph had 43 nodes / 66 links; only
two image-strength fields and the output prefix changed. Prompt
`08be6411-b174-42f7-94b0-73262a3f2328` completed and was independently verified.
Previous Snow native f96 -> Ice f0/f1 RGB MAE changed from 5.2559/7.1223
(strength 0.7) to 4.6112/6.3074 (strength 1). Better here, but neither new frame
is pixel-identical or certified seamless. MAE is a diagnostic, not seam proof.
The review using Snow f0..96 + Ice f1..96 has 193 frames: a declared conventional
cut, not evidence that discarded Ice f0 was a duplicate. Preserve its local /
global frame and source-audio map explicitly; do not infer a shorter audio span
from the retained frame count or change the native output.

Keep the evidence; do not restart this archived experiment merely because its
old graphs or pending source candidates exist. Future work must recheck current
contracts and have its own scope and media acceptance criteria.

## Next harness: Easel Client storyboards

Use this knowledge through the existing Easel Client creative harness:
storyboard -> WebGL canvas -> exported RGB/depth/Canny assets -> typed Easel
video job -> Comfy Graph/local ComfyUI -> returned storyboard video asset ->
playback, review, tweak and export. Gio reports client video ingestion/display,
WebGL canvas video export and Easel API video support now exist; recheck their
actual revisions and supported conditioning fields, rather than rebuilding
from earlier missing-video assumptions.

Keep model/loader/guide/crop knowledge in Comfy Graph, admission/auth/job
contracts in Easel, credentials in the client privileged process and canvases
offline. Use managed asset IDs and validated typed capabilities, not arbitrary
client-supplied graphs, model paths or filesystem paths. Start with one real
storyboard shot, then exported-guide conditioning and a verified two-shot
strength-1 handoff before expanding. Source export, model execution and visual
acceptance are separate evidence gates. This direction does not install,
publish or deploy the staged source recipe or grant new render authority.

## Optional: periodic source does not guarantee a generated loop

A separate source test advanced one world period (5.8 m / 1.76 m) in 4 s at
1.45 m/s, with periodic materials/lighting and declared render-coordinate wrap
plus monotonically advancing unwrapped poses. Independently rendered frame 96
was byte-equal to frame 0 in RGB and depth; 97 conditioning frames include the
closure duplicate, whereas playback uses 96 unique frames / 4 s. Check the
95->0 seam against ordinary adjacent differences, not only endpoint equality.
Loop-source RGB lighting changed; depth frame 0 remained equal to baseline.

The accepted baseline is **not looped** (endpoint MAE 22.34 versus adjacent
median 6.02). One separate mixed Union + first/last-frame + frozen-source A2V
experiment then reached terminal success: prompt
`9b565315-da86-43b9-8bed-1febc883ba67`, 30.76 s, native 97 frames / 512x384 /
24 FPS, full A/V decode. It used the original generated baseline frame 0 at
both endpoints 0/96 with `LTXVAddGuide` strength 0.7 in both passes. Coarse
ALL-guide cropping reduced temporal slots 28->13 before x2 upsampling; only
the two FLF references were reinserted in refinement, then cropped 15->13
before decode. These slot counts are CPU bookkeeping evidence; terminal/model
runtime establishes execution, not perfect attention or camera compliance.

**Partial closure, not a perfect loop:** generated endpoint MAE improved to
7.71, but normal adjacent median was 4.81; the 96-unique-frame playback join
95->0 was 8.71. Terminal rails/arches softened/ghosted and snapped to sharper
first-frame details at wrap. Source periodicity therefore does not imply model
seam continuity. `LTXVAddGuide` is soft reference conditioning, not hard pixel
copy; no endpoint clone, crossfade, reversal or warp was used. Audio loop
boundaries and human acceptance remain separate checks. Evidence is
`/home/venetanji/.openclaw/workspace-creative-skills/media/ltx25-staircase-loop-20260930/ltx/`
(`submitted-graph.json`, `terminal-history.json`, `cpu-guide-bookkeeping.json`,
`qa-summary.json`, `seam-qa.json`, decoder logs and seam frames). This single
technical experiment is not a production seamless-loop capability claim.

## Worked Canny branch: shapes -> carried props

The original source-only candidate is retained as preprocessing evidence. The
subsequently approved **faster-sphere variant**, not that original candidate or
a labeled composite, was used for the verified carried-props pilots.
It contains only a cube, sphere and horizontal cylinder, left/middle/right,
with unlit face colors, neutral background and no authored bodies/faces,
shadows, floor edges or clutter. Fixed perspective camera: FOV 34 degrees,
position [0,0.95,4.5], target [0,0.95,0]; a physical 1.8 m person's head/feet
would fit the frame. Cube size 0.4 m at y=1.03 m; sphere radius 0.22 m at
y=1.16 m; cylinder length 0.65 m / radius 0.10 m at y=1.03 m. Centers are
1.2 m apart. All share a gentle +0.32 m horizontal shift over 4 s, <=0.012 m
measured-kick RMS bob and tiny stable rotations. All 97 frames retain three
separated props without clipping/background edges; at 128x96, minimum nominal
prop-guide extents are 16 x 7.75 pixels.

Actual CPU preprocessing uses existing OpenCV 4.13.0 in
`/home/venetanji/dev/ComfyUI/.venv/bin/python`, no dependency installation,
Torch import, CUDA or Comfy API/node execution. OpenCL is disabled. For a new
media frame, with `cv2`, `numpy as np` and `PIL.Image` imported and `rgb_png`
set to your source PNG path, this executable six-line recipe matches the
verified preprocessing (disable OpenCL/set two CPU threads before the loop):

```python
rgb = np.asarray(Image.open(rgb_png).convert("RGB"))
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # BT.601
blur = cv2.GaussianBlur(gray, (3, 3), 0.7, borderType=cv2.BORDER_REPLICATE)
raw = cv2.Canny(blur, 60, 120, apertureSize=3, L2gradient=True)
guide = cv2.dilate(raw, np.ones((3, 3), np.uint8), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
guide_rgb = np.repeat(guide[:, :, None], 3, axis=2)
```

Save `raw` separately as the **one-pixel Canny** diagnostic and `guide_rgb` as
conditioning PNG. The latter is **Canny plus one declared 3x3 dilation**, an
approximately three-pixel edge band for coarse 128x96 visibility, not raw
three-pixel Canny. Thresholds 60/120 are fixed Sobel-gradient thresholds in
8-bit input-intensity units; no automatic per-frame thresholds. The local
Comfy `Canny` node's normalized 0.01..0.99 range/defaults 0.4/0.8 are a different
implementation contract: do not transfer these numeric thresholds unchanged.
This is actual RGB-derived hysteresis/nonmaximum-suppression Canny, not depth
or manual wireframe.

Export binary 0/255, three identical gray channels, 97 frames / 512x384 / 24 FPS
in FFV1 `bgr0` Matroska. All 97 decoded RGB24 frames match the guide PNGs exactly.
Feed this **already prepared** edge batch into the Union branch, without a
learned depth estimator or a second Canny. Depth success does not validate Canny automatically; the faster-Canny branch
has its own executed graph, source binding and media evidence.

Local source evidence:
`/home/venetanji/.openclaw/workspace-creative-skills/media/ltx25-canny-carry-poc-20260930/source/`
contains `manifest.json`, `validation.json`, `canny-metadata.json`, `canny.py`,
`scene.js`, `world.json` and `hashes.json`. Actual source-only artifacts:

- `canny-guide-master.mkv`: 161,027 bytes; SHA256
  `f3665e8502c2fd2d280e94be8ba66c4ce0a0767dc1858f558b94a4b78ce2a7fd`.
- `canny-choice-preview.mp4`: 169,889 bytes, labeled RGB/Canny approval view;
  SHA256 `f7fb17e17fe38029911291b025b61892b5e3fb2a4a540852516632551e7b79c5`.
- Aggregate decoded guide RGB24 SHA256:
  `28e9a33eab4451321435a1b560d0418ad1b5a478a75be9a65bd4af3479650686`.

The original source manifest correctly records zero model submissions at its
creation time; it is not the status of subsequent separately authorised pilots.
For new work, show the actual single-panel guide and obtain approval before
submission. Prompted people are added in unguided regions: geometry alone does
not guarantee bodies, people count, hand contact or material conversion.

### Faster-sphere source and model comparison

Keep the same camera, scale, geometry, audio bob and other prop transforms.
The authored sphere travels +0.16 m/s, versus +0.08 m/s for cube/cylinder:
0.64 m versus 0.32 m in four seconds. All-frame RGB/Canny checks isolate the
changed sphere region; screen displacement ratios are 1.9994088 / 1.9994804.
This is measured **source** motion, not measured generated-person speed.
Original single-panel `canny-master.mkv` matches all 97 PNGs exactly;
`guide-choice-preview.mp4` is the labeled RGB/Canny human comparison, never
conditioning input. Source evidence is `ltx25-guide-variations-20260930/fast-props/`
under the local media root: `scene.js`, `world.json`, `manifest.json`,
`source-proof.json` and `validation.json`.

Executed prompt specifies exactly three walking adults left/center/right,
carrying a metal cube-shaped box, a bright red ball, and a horizontal wooden
cylinder with plausible hand contact; center person gains on the others.
At video control 0.6, the model produces three carriers and qualitatively a
middle carrier gaining; exact 2x speed is not measured. At 1.0, all three remain
and changes are modest. Inspect hand contact, silhouettes, trajectory, count,
background invention and temporal drift in native frames, not only a preview.

## Worked depth branch: projected tesseract -> kinetic sculpture

Author the actual 16 vertices {(-1,+1)^4}; join pairs differing in exactly one
coordinate, producing 32 edges and degree 4 at every vertex. Rotate XW at
+pi/2 rad/s and YZ at -pi/2 rad/s. Project rotated points with
`xyz3D = 3.2 * xyzRotated / (4.2 - wRotated)`, then render **solid** edge beams
(radius 0.045 m) and nodes (radius 0.06 m), not a flat wireframe drawing.
Camera: perspective FOV 43 degrees, [0,0,6.5] looking at origin, near/far
0.1/160 m. Render registered RGB and real DepthTexture from this projected 3D
geometry. Depth is projected-3D axial camera view-Z, **not 4D distance**; apply
the same calibrated fixed-global log transfer above.

Validate the combinatorics, projection denominator, frustum/occlusion and all
97 frame transforms. Independently rendered RGB/depth f0 and f96 are equal;
all frames stay inside the frustum. This periodic source does not certify a
generated loop. Exact master/PNG agreement also does not prove the model kept
16 vertices / 32 edges. Source evidence is
`ltx25-guide-variations-20260930/hypercube/`: `scene.js`, `world.json`,
`source-proof.json`, `manifest.json`, `validation.json`; feed only the original
single-panel `depth-master.mkv`, never its three-panel comparison.

Prompt intent is a solid rotating blue-glass/gold kinetic sculpture following
the depth structure, no people or additional room. At video guide 0.6 the model
produces that material/motion idea with simplified topology. At 1.0 it becomes
a cleaner/open cube-like cage, not a restored exact tesseract; fewer bars can
be visually cleaner while losing authored topology. Judge the output itself.

## Capability and proof matrix

Record profile x workflow x optional features for the **specific asset set**,
not a blanket label such as "LTX supports guidance". Recheck live loaders,
node schemas, LoRA patches, encoder, both VAEs and latent upscaler before a new
submission. Retain negative prompts and reject unsupported combinations before
uploads. Never substitute another model version/backend on failure.

| Combination | Evidence | Limitation / admission |
|---|---|---|
| Local 2.5 INT8 distilled + listed Union + authored depth + frozen WAV | Staircase terminal/runtime/media + accepted illustrative output | Not exact camera/geometry compliance |
| Same assets, faster Canny, video 0.6 / 1.0, D8/R3, CFG 1, seed 42030 | Two verified outputs, actual input hashes/audio, controlled field diff | No measured generated 2x speed, universal improvement or loop claim |
| Same assets, projected depth, video 0.6 / 1.0 | Two verified sculpture outputs | Not generated 16/32 topology or certified looping |
| Same guide + previous actual native f96, image 1 both passes | Verified separate Ice probe, no end constraint | Improved boundary here; no hard pixel copy, motion history or seamless certification |
| Curated 2.5 CLI Union control-video | Absent at 394142a; profile unknown fields rejected before upload | Explicit reviewed prototype graph only; typed integration still separate |
| Alternate LoRA/model/sampler, mixed edge+depth stack, fresh R3 full-video guide, image/end variants | Not validated by these pilots | Reject or request separate scoped compatibility/visual test |
| Easel API/client exported-guide fields | Private graph evidence does not prove typed consumer support | Reinspect current contract/revisions; do not advertise missing fields |

Guide-only pilot graphs have 39 nodes / 53 links, no image or end constraint.
For the selected 2.5 VAE, native 97 frames encode to T=13; video guide adds
T=13, then ALL-guide crop reduces 26->13 **before** x2 upsampling. R3 stays at
T=13 with no fresh full-video guide. These counts are **derived** from selected
VAE config and installed append/crop code, not runtime tensor instrumentation
(`ltx25-two-spatial-pilots-20260930/derived-latent-crop-counts.json`).
Image+video composition adds one coarse image slot (27->13), then one fresh
**image**, not fresh video, in R3 (14->13 before decode). The actual installed
CPU node probe used a fake zero-shape VAE, not real model latents
(`ltx25-staircase-material-chain-20260930/ltx/cpu-guide-bookkeeping.json`);
distinguish this shape/mask/RoPE proof from successful actual graph/history/model
evidence and from pixel/velocity continuity.
Appended guide temporal slots are conditioning context, not frames to concatenate
onto the 97-frame final native video.

Keep three independent controls in manifests and graph diffs:
`LTXICLoRALoaderModelOnly.strength_model=1`,
`LTXAddVideoICLoRAGuide.strength=0.6 or 1`, and ordinary stitching
`LTXVAddGuide.strength=1` in both image passes. A change to one is not a change
to the others. The strength-1 spatial reruns changed node 13 video strength and
SaveVideo prefix **only** (two POSTs, zero uploads total, no backend retry).
Completed prompt IDs are `b4c1c123-b874-4b2d-a91b-10115288cf00` and
`08574822-fb2d-4d17-871e-b0db93074b1a`; check the actual retained terminal history
if reconciling a lost worker, not a timeout callback. Local evidence roots
`ltx25-two-spatial-pilots-20260930/` and
`ltx25-spatial-guide-strength1-20260930/` each retain per-scene
`submitted-graph.json`, `terminal-history.json`, `source-binding.json`,
`runtime-lora-qa.json`, `audio-qa.json` and independent `parent-verification/`.
Human comparisons were delivered; technical verification is not human fidelity
acceptance. Existing outputs stay immutable; this reference grants no rerun.

## Diagnose drift without inventing a cause

The ten-material chain used the actual preceding native f96 at all nine
joins, byte-proven against backend input PNGs/executed history, but soft 0.7
image conditioning did not preserve seamless foreground, velocity or detail.
Snow/ice/lava/marble were clear; other materials only partial. Correct chaining
provenance is necessary, not sufficient. Late ghosting at f88..96 occurred even
with periodic source f0=f96. A CPU source/code-flow probe inspected guide
append/crop, frame-96 latent index 12 and one-image RoPE range 96..97 versus
final block 89..97. Those observations **do not establish a causal explanation**
for drift or justify a new end frame, fresh refinement video context, fade,
frame deletion or native-history claim. Separate measured behavior, implementation
bookkeeping and hypotheses. Preserve the closed experiment and require new
scope for any additional test.

## Source audio, timing and bounded execution

Freeze the original encoded audio latent with a zero noise mask in **both**
passes; reset to that original before R3, not the sampled coarse audio output.
Final CreateVideo muxes the original trimmed WAV directly. Conditioning with
supplied audio and final source-audio mux are separate claims. Check both graph
connections, plus actual decoded content, best lag, sample count, priming and
tail. For 97/24 s at 48 kHz the source has 194,000 samples. The spatial native
AAC outputs decode to 193,968: correlation about 0.982469 at zero lag, with a
32-sample silent source tail omitted. Ice decodes to the same count, correlation
about 0.983459, but its omitted 32 samples are **not silence** (peak about
0.11923). Do not extrapolate silent-tail results to Ice/material segments, pad
zeros to manufacture success, retime, or call lossy AAC bit-identical WAV.
Review/composite encodes can have different sample accounting; verify each.

Guides, audio and shot assembly need explicit half-open source/local/retained
frame spans, FPS, cuts and overlap mapping. A 97-frame native output is not a
4-second clip. A 96-frame periodic playback export is a distinct derivative;
endpoint equality must be checked before omitting a closure frame. Do not
silently drop regenerated I2V f0 as an identical duplicate. Never conceal a bad
boundary with undisclosed cosmetic fades/holds.

Before any authorised render: read current source/assets/node schemas, check
queue/history, approve the actual source, hash/verify it, admit every option,
and bound the batch/seed/settings. Save a write-ahead attempt record, unique
prefix, exact graph and returned prompt ID. Stop on HTTP 400 rather than trying
another profile or mutation. For ambiguous submission/worker failure, reconcile
queue and retained history/output bindings before any retry; running is not
completion and a stale timeout is not permission to resubmit. Keep models warm;
no broad `test_all_workflows.py` sweep for documentation validation.

At source 394142a, curated profile `dump` bypasses uploads, but legacy image
handlers still call `upload_if_local`; `i2v --help` is parsed as options and can
reach submit. Recheck source and use mocked/no-network discovery. Do not treat
help or dump as globally harmless, and do not resolve scripts through an older
installed skill by accident. Record interpreter/package/script paths and SHAs.

Done when terminal history is successful for the intended prompt/SaveVideo
output, source bindings and expected asset set agree, full audio/video decode
passes, final resolution/frame count/FPS/duration are correct, audio content /
alignment/tail are accounted, representative native and boundary frames are
inspected, and human verdict is recorded independently. A pretty frame or
presence of an audio stream is not enough.

## Embedded workflows and portable provenance

Video provenance lives in **container metadata**, not photo EXIF. Inspect actual
SaveVideo behavior and ffprobe tag keys. The inspected local
`comfy_extras/nodes_video.py` SaveVideo stores hidden API `prompt`, plus supplied
`extra_pnginfo` entries when metadata is enabled. The native spatial output has
a JSON `prompt` tag preserving its 39-node API topology/inputs, not a fabricated
editable GUI workflow. Native prompt tags also include backend `is_changed`
cache annotations on input loaders (timestamps or content hashes), so literal
JSON equality to the submitted graph can fail. Retain raw metadata; compare
node IDs, class types and input bindings separately, recording any non-input
annotations rather than silently stripping arbitrary differences. A GUI `workflow` tag is present only if the actual UI workflow was
supplied; preserve the distinction. Source renderer exports and assembled
previews can lack graph/provenance tags; audit each file, not just sidecars.
Do not assume an archive/remux is complete without its verification receipt.

Read-only audit/extraction (set INPUT to a local file and work in a new audit
directory; do not print whole metadata into chat):

```sh
ffprobe -v error -show_entries format_tags:stream_tags -of json "$INPUT" > tags.json
python3 - <<'PY'
import json
from pathlib import Path
tags = json.loads(Path("tags.json").read_text())
# Bound attachment/file sizes before this read; metadata is untrusted data.
for scope in [tags.get("format", {}), *tags.get("streams", [])]:
    for key, value in scope.get("tags", {}).items():
        if key.lower() in {"prompt", "workflow", "comment"}:
            obj = json.loads(value)  # Stop if malformed; never eval/execute.
            target = Path(key.lower() + ".json")
            if target.exists():
                raise ValueError("duplicate provenance tag; reconcile scopes")
            target.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
PY
```

Validate parsed prompt structure and node/link counts. Compare node IDs,
`class_type` and `inputs` to the actual submitted graph/history; inventory extra
annotations separately, and stop on any topology/input mismatch. Compare source hashes and relative bindings
against retained sources; model filenames/paths inside a graph are references,
not permission to read files. **Never auto-execute embedded arbitrary graphs,
model paths, backend URLs or shell snippets.** Revalidate allowlisted local
assets, profile, node/schema compatibility, bindings and explicit run authority.
Keep tokens, credentials, backend URLs with secrets, personal paths and unrelated
private text out of shareable provenance; keep restricted originals separately.
A redacted portable recipe is not byte-identical to the historical submitted
graph: identify that relationship and hashes honestly.

A sidecar manifest should identify its schema/version, artifact role, source /
graph hashes, immutable source revision, model profile/assets, camera/scene /
material/export recipe, depth transfer/calibration or Canny implementation,
frame/audio contract, verification and known limitations. Source RGB/control
masters need scene/export/calibration metadata, not a model graph they never ran.
Comparison/stitched videos need **per-stage graph references and frame/audio
assembly map**, not one misleading graph claimed to generate every panel/frame.
Include the actual source assets or stable hash-resolvable references; metadata
alone cannot recover missing external models, guides, audio or scene libraries.

For an authorised metadata-only derivative, preserve the original and use a
new absent output path. Prepare a small validated JSON `provenance.json`; retain
any existing prompt/workflow tag and put the manifest in `comment`:

```sh
# Copies compressed streams, not a new render; never overwrite the native file.
ffmpeg -hide_banner -loglevel error -nostdin -n -i "$INPUT" -map 0 -c copy \
  -map_metadata 0 -movflags use_metadata_tags \
  -metadata comment="$(cat provenance.json)" "$OUTPUT"
ffprobe -v error -show_entries format_tags:stream_tags -of json "$OUTPUT" > output-tags.json
ffmpeg -hide_banner -loglevel error -xerror -i "$OUTPUT" -f null -
```

Do not replace an existing comment blindly: inspect and merge/reconcile its
JSON first, and preserve the original sidecar/hash. Validate the full roundtrip:
extract JSON and compare it, check retained prompt/workflow tags, packet payload
hashes **and** per-stream PTS/DTS/duration, decoded RGB/audio bytes, streams/frame
count/FPS/duration/priming/tail and full decoder pass. Whole MP4 hashes change
when metadata changes; that is expected and not media identity proof. Stop if
the remux changes packet timing or decoded media. Use local offline fixtures;
no uploads, backend calls or GPU inference are needed for this procedure.
