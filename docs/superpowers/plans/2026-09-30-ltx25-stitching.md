# LTX-2.5 stitching implementation (approved scope, revised post-M1)

Base: 7c5edd9e3ba6bd8e3d53d538d1dfc096841d78b5. Isolated branch only.
Approved Sep29 design is adapted to the standalone comfy-graph package;
Sep30 prioritizes actual same-scene continuation over compatibility. No new gate.

## 1. Real profile and continuation/timing
- Test first: real UNET/Gemma4/split VAEs/2.5 upscaler, dual CFG 1/1,
  distilled euler_ancestral 8+3 sampling, no extra distilled LoRA.
- Keep explicit ltx-2.3/default; admit 2.5 text/image and continuation only.
  Reject unknown profiles, fast mode, camera/transition/IC LoRAs, keyframes,
  identity anchors and unsupported combinations before any upload/submission.
- Final 512x320 means coarse 256x160, two-pass, 24fps, 49 frames.
- Normalize one-second/24-frame overlap to explicit 25 BEFORE extraction.
  Expose source spans and retained endpoint, reject invalid/short tails.
  Crop guide tokens before upsample and final decode; guide != pixel lock.
- Uploaded master audio uses explicit source frame offsets, frozen encoded
  audio conditioning, and original source audio for mux (never VAE output).

## 2. Frame-exact assembly and actual render proof
- Production local assembly helper: integer frame trim, common geometry/fps/
  timebase, master audio once, no -shortest, decoded frame validation.
- Single serial GPU owner. Local http://10.99.0.9:8188 only; query queue before
  each job; validate graph against current nodes (dynamic SaveVideo enum).
  No cloud, model downloads, cancellation, cache flush, OOM sweep or blind retry.
- Two-pass A49 seed4201/B49 seed4202, red ball rolls right on marked wood,
  static camera. B guide A[24:49], source starts global24; no tail buffer.
- Correct A[0:49]+B[25:49]=73 frames versus naive 98 from SAME renders.
  48k PCM master146000 samples, ticks frames12/36/60; A[0:98000],
  B[48000:146000]. Prefer real uploaded conditioning on first attempt.
- Media/live JSON/logs outside repo in workspace media/ltx25-stitch-20260930.
  Require successful terminal histories, IDs, MP4 selection, full decode,
  frame counts/512x320/24fps/uniform timestamps, A44-48/B23-29/join sheets,
  motion/identity/texture/camera assessment, audio correlation/tick alignment.
  If continuity fails, measure it; no crossfade disguise or seamless claim.
  Targeted rerender only after observed failure/fix; stop real loader/resource blocker.

## 3. Focused review and caller integration handoff
- Install package normally via uv in isolated environment; no PYTHONPATH or
  installed-skill fallback. Targeted RED/GREEN and existing offline suite.
- Preserve all unrelated sources. Commit code/docs/tests, no push/PR/merge.
- Report exact capability evidence and remaining defects. Graph concat audio
  trimming/overlay and drama buffered-tail/audio-offset/overlap-replay plus
  music installed-helper resolution remain deferred unless selected path needs them.
- Parent independent review and caller follow-up required; this slice is not
  whole ecosystem completion. No default promotion without advertised-path proof.

## Execution record

Stage 1 implemented and tested (real loaders, masks, guide crops, exact spans,
pre-upload profile adapter). Stage 2 completed two real audio-conditioned jobs;
A49/B49 and corrected73/naive98 have successful decoder/timing checks. Master
mux correlation 0.999939, tick errors <1 sample. No extra GPU retry required.
Stage 3 focused review/tests in progress. Existing graph goldens unchanged;
wheel allowlist intentionally adds ltx_profiles.py. Python3.14 AST dump omits
empty fields versus 3.11/3.12 baselines; use documented3.12 rather than regenerate.
Perceptual limitation: red-ball identity and close local join persist, but the
pilot does not honor steady rightward motion or fully fixed background. Keep
continuation experimental, not a seamless-motion certification.

Stage 3 local focused review complete: reject mismatched workflow/media and
orphan timing options before upload. Final offline run:147 passed,2 artifact-only
skips. Clean wheel and sdist-rebuilt wheel each pass109 tests on Python3.11 and
3.12 (four independent environments), with no source PYTHONPATH. No old golden
changed. Parent independent review/caller work remains mandatory. The small
join does not justify an extra GPU retry: visible long-range camera/motion drift
already begins in A; report that limit rather than tuning away the evidence.
