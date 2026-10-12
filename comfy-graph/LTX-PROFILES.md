# Curated LTX profile path

`creative_comfy_graph.ltx_profiles.build_video` is a pure graph builder with
explicit `ltx-2.3` and `ltx-2.5`. The default stays 2.3. Existing legacy builders
remain unchanged; profile routing is opt-in and has a narrower checked contract.
The 2.5 profile uses UNETLoader, Gemma4 CLIPLoader, separate video/audio VAEs,
2.5 upscaler, dual CFG 1/1 and distilled Euler ancestral 8+3 steps, without LoRA.

Final dimensions are multiples of 64, not coarse-pass dimensions. Minimum final
width/height is 128 (coarse latent minimum 64). Maximum is 32768 for text, 16384
with image/continuation guides (final-size resize node limit). `frames` is 8k+1,
at least 9 and at most 993 with generated audio, 16377 with supplied audio.
Integer fps is 1..120; seed is 0..2**64-2 so both sampling seeds fit uint64.
Continuation guide start/count must each fit GetImageRangeFromBatch's 4096 max.
These are captured local backend schema bounds, NOT GPU memory or live-support
claims; upper-bound graphs may be infeasible on available hardware. The first latent is half width/height; refinement restores the
requested final geometry. Fast mode, arbitrary assets, camera/transition/IC
LoRAs, extra identity references and arbitrary multikeyframe/transition workflows are rejected.
The narrow two-single-endpoint FLF path is described separately below.
No silent model fallback or profile promotion.

## Capability evidence (2026-09-30)

| Path | Offline | Live |
|---|---|---|
| 2.5 text + supplied master audio, 49f 512x320/24 | passing | successful A |
| 2.5 continuation + supplied master audio, 25-frame tail | passing | successful B |
| 2.5 image / image+audio | topology supported | not yet smoke-validated |
| 2.5 generated audio | topology supported | not yet smoke-validated |
| 2.3 explicit curated profile | topology supported | no new smoke |
| 2.5 transitions/multikeyframes/IC/camera/fast/identity | rejected | unverified |

Live success is not a claim of prompt-faithful or universally seamless motion.
The pilot keeps object identity and a close join but fails steady rightward
motion/static-camera intent. Do not expose this as a validated seamless API.

## CLI

Use the interpreter with the package installed normally; no PYTHONPATH needed:

```
python comfyui/scripts/comfy_graph.py dump t2v --model-profile ltx-2.5 --frames 49 --width 512 --height 320 --seed 4201 --prompt 'A rolling red ball'
python comfyui/scripts/comfy_graph.py dump continuation --model-profile ltx-2.5 --prev-video A.mp4 --prev-frames 49 --frames 49 --overlap-frames 25 --audio master.wav --timeline-json B-timeline.json --seed 4202 --prompt 'The same ball continues rolling'
```

Profile admission builds once before uploads. Integer options accept integer CLI
strings or JSON integers, never booleans/floats. `--seed 0` stays zero. Malformed
selected-route input and unknown options fail locally. Base/refine guide strengths
are image-only; overlap strength/timing is continuation-only. Explicit irrelevant
values (including defaults) are rejected by the pure builder as well as the CLI.
Omitted/None guide strengths retain image defaults 0.7/1.0 and overlap default 1.0.
The pure timeline helper remains model-independent; graph-specific bounds live
in build_video, not in callers. `dump` performs no uploads;
provide `--prev-frames` to avoid local ffprobe. With no `--frames`, seconds round
down to the previous multiple of eight, then add one (legacy duration rule).
`--overlap-seconds 1 --fps 24` rounds UP to 25 before tail extraction; explicit
24 is rejected. `--previous-retained-end` excludes discarded look-ahead.
`--previous-source-start` locates a previous clip on the global source timeline.

`continuation_timeline` exposes half-open guide/source/retain spans. A49 followed
by B49 with overlap25 yields A[0:49]+B[25:49]=73 frames. B source starts global24.
With master audio, B's conditioning offset is therefore 24/24 seconds, not 49/24.
The original audio is trimmed to the source span, encoded with a frozen mask in
both passes, and directly muxed into individual outputs instead of VAE decoding.
Audio offset mismatches are rejected. Passing a pre-sliced file requires a
separate future API; this entry point explicitly expects the global master.

## Assembly and boundaries

`storyboard/lib/assemble.py:assemble_frame_spans` accepts `(path,start,end)` integer
spans, rejects mismatched CFR/dimensions/short audio, normalizes timestamps,
encodes once, muxes the original master once, and verifies decoded frame counts
and full decode plus EXACT decoded 48k audio sample count before replacing the
destination. Coverage and mux both select the first audio stream, `a:0`; other
tracks cannot satisfy coverage. AAC builds that produce 32 fewer samples fail
without replacing an existing output, rather than reporting intended samples
as observed. System ffmpeg 9 passed the retained proof; no silent padding. No crossfade or `-shortest`.
Legacy seconds-based assembly and graph concat are unchanged. Explicit-profile
drama/music callers use this path; legacy implicit routes do not.

The shared transport submits only once: rejected HTTP responses retain the body
and node diagnostics; ambiguous submission failures require queue/history
inspection, never automatic resubmission. History GET retries share a bounded
monotonic deadline. Terminal successful history is required; execution errors
win over preview outputs. Only assets typed `output` are downloaded. The selected
profile route also requires a successful output download (input/temp previews
are not generated results).

## First/Last Single-Frame Profile (Offline)

`ltx_profiles.build_flf2v(prompt, *, first_frame_filename, last_frame_filename,
model_profile='ltx-2.3', frames=49, fps=24, width=512, height=320, seed=None,
filename_prefix='ltx_flf2v_profile', negative=None, audio_filename=None,
first_guide_strength=None, last_guide_strength=None)` shares the checked profile
builder; it does not clone the legacy 2.3 family or substitute model filenames.
Both endpoints explicitly select the FIRST loaded image frame via
`GetImageRangeFromBatch(start_index=0, num_frames=1)` BEFORE resize/preprocess.
This also applies to animated GIF/APNG and backend-relative assets: animation
frames after the first are ignored, not appended as extra guide tokens. Both
selected endpoints are reused as independent single-image AddGuide chains in
BOTH passes at indices `0` and `frames-1`. Existing image/continuation and legacy
multi-frame handling is unchanged. Positive end indexing avoids interpreting the end of
a guide-token-expanded latent as the requested final frame. Each pass crops its
guide tokens; the first crop precedes upsampling, the second precedes decode.
The endpoint index must fit the captured AddGuide maximum 9999 (largest admitted
8k+1 count here is 9993 with supplied audio). The generated-audio limit remains
993. These are offline schema bounds, not hardware feasibility promises.

Endpoint strengths default to 0.7 independently and persist in both passes.
Explicit finite [0,1] values change the graph; image base/refine and continuation
controls are NOT accepted. No fast/custom models/camera/transition/IC LoRAs.
2.3 remains the default, not silently promoted. Optional full-source audio starts
at zero, is frozen conditioning twice, and the original zero-offset trim is
muxed; no sliced-master or audio-offset convention is supported for FLF.

| Additional Path | Offline | Live |
|---|---|---|
| 2.5 two single-image endpoints, generated audio | topology/tests passing | NOT validated |
| 2.5 two single-image endpoints + original supplied audio | topology/tests passing | NOT validated |
| 2.3 explicit first/last profile | topology/tests passing | no new smoke |

Example (use the final-wheel Python environment, not a source PYTHONPATH):

```
python comfyui/scripts/comfy_graph.py dump flf2v --model-profile ltx-2.5 --first paper-registered.png --last circuit-registered.png --frames 97 --fps 24 --width 768 --height 512 --seed 0 --first-guide-strength 0.7 --last-guide-strength 0.7 --prompt 'Locked front camera, central ignition, heat out through creases, charring and curling, copper and cyan fibers; preserve framing and logo footprint; no drift, panels, plume, extra marks, letters, cuts or static dissolve; upper-right dot empty'
```

Remove `dump` only for an authorized live run. JSON `first`, `last`, strengths and
integer seed 0 use the same path. Dump retains filename placeholders without
uploads/probes; continuation additionally needs known `prev_frames` for this.
Graph dimensions are FINAL: 97 frames at 768x512 use coarse 384x256, then x2
refinement. This is not a render/end-quality certification.

For the SD2112 placement, anchors at global250 and345 have a95-frame distance;
97 generated frames span local0..96. The parent/assembler must explicitly map
both endpoints to a96-frame interval [250,346): e.g. select local input indices
`round(i*96/95)` (nearest, half-up) for output i=0..95, preserving input0 AND96.
This omits one interior generated frame, not the last anchor. Blindly retaining
[0,96) loses the last anchor; placing the last at346 changes the assignment.
The package generates97; it does not perform this placement/resampling. The
606-frame final timeline at24fps is25.25s. Oasis is the final master soundtrack;
pilot generated audio may be ignored at final assembly. Supplying audio to the
graph is optional, not a pilot requirement. No orchestrator/timeline change is
included here.

## Selected-Profile Upload Preflight

Every selected profile validates all graph options and all known local paths
before its first upload. Present empty or whitespace-only image/audio/video or
endpoint filename options reject, including in dump; they never mean omission.
Missing optional audio still requests generated audio. Existing files are local; missing absolute, `./`, `../`
and `~` paths reject. Nonexistent bare names and ordinary subfolder names remain
legitimate Comfy input-relative remote assets, not evidence of a local file.
Use an absolute/explicit-relative path when missing-file rejection matters.
Dump deliberately does not stat/probe placeholders. Local supplied audio is
preflighted using ffprobe's first audio stream duration and the requested full
source span; short/unprobeable sources reject before uploads. Remote audio
coverage cannot be verified offline. This duration preflight is not decoded
sample/alignment proof; callers/assembly retain their stronger decoded checks.

Selected upload uses exactly one POST and accepts a nonempty input asset name
plus its returned subfolder. HTTP errors preserve response diagnostics; lost,
malformed or missing-name/wrong-type responses raise without fallback to the
old host path and without prompt submission. Earlier successful uploads may
remain server-side if a later upload fails, but no graph is submitted. Generic
legacy upload behavior remains unchanged. No API arbitrary filesystem/model
access is introduced.
