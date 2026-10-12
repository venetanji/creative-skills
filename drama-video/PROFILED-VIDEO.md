# Explicit frame-timed LTX profiles

Opt in with `video.model_profile: ltx-2.5`; `ltx-2.3` is also selectable.
Omitting the field keeps the historical route/default unchanged. This is NOT a
promotion of every LTX-2.5 path: only text+master-audio and same-scene
continuation+master-audio have independent real GPU evidence. Image and generated
audio paths have offline command/graph tests, not independent live validation.

Run callers with the Python environment in which `creative-comfy-graph` is
normally installed, plus PyYAML and imageio-ffmpeg. Python children use
`sys.executable`, not a global `python3`. Keep `drama-video`, `music-video`,
`comfyui`, and `storyboard` as sibling directories from the same revision;
music loads the narrow `drama-video/scripts/profiled_video.py` helper. Checkout
siblings take precedence over separately installed skills. The system `ffmpeg`
and `ffprobe` must be on PATH; profile assembly deliberately uses them rather
than imageio's bundled ffmpeg 7, whose AAC container duration truncated 32
samples in the 73-frame proof. System ffmpeg 9 passes exact sample validation.
No package-source `PYTHONPATH` is required or supported as an installation fix.

```yaml
video:
  model_profile: ltx-2.5
  resolution: [512, 320]  # Final geometry, multiples of 64, each dimension >=128
  fps: 24
  tail_buffer_sec: 0
  negative: cuts, text
  seed: 4201
audio:
  file: master.wav       # FULL global master, not new-shot slices
shots:
  - label: A
    duration_sec: 2.0416666666666665  # 49 visible frames at 24 fps
    mode: t2v
    prompt: A red ball on a table.
  - label: B
    duration_sec: 1      # 24 NEW visible frames, not total generated duration
    continue_from_prev: true
    overlap_seconds: 1 # Normalizes UP to 25 valid guide frames
    prompt: The same red ball continues moving.
```

For music use `scenes:` instead of `shots:` and the existing `song.mp3` full
master. No `lipsync_audio` override is accepted in this narrow profile route.
Run `plan SPEC`, `shot N SPEC` / `scene N SPEC`, then `assemble SPEC` through the
normal drama/music entrypoint. `shots` / `scenes` and `all` use the same route.
Existing audio-generation stages are unchanged. Profile `anchors` validates and
returns: supply an existing `image` or `anchor_image`, not an anchor-generation
configuration. Explicit `mode: i2v` omits conditioning audio (final assembly
still uses the master); `mode: ia2v` requires an image and conditions on master
audio. Default mode is ia2v with an image or t2v without one. Both callers honor a
top-level existing `anchor_image` for the first item and `image: '@anchor'`.
Explicit i2v/ia2v without an image is rejected during whole-spec planning. `mode: t2v` or
`image: '@none'` explicitly bypasses the implicit previous-frame image.

## Timing protocol

- Requested durations describe visible dialogue/shot time. Absolute boundaries
  round to nearest frame (half up), with exact rational accumulation for omitted
  starts. Explicit starts must describe a contiguous timeline starting at zero.
  No millisecond formatting enters generation or frame assembly.
- Full render length rounds UP to `8k+1 >= 9` after adding overlap and tail
  buffer. Planning reports required decoded 48 kHz master samples, including
  rounding/look-ahead. Insufficient master audio is rejected before submission;
  it is never padded, looped, frozen, or silently truncated.
- `NNN-label.timeline.json` persists full render count, local retained `[begin,
  end)`, global full source span, visible source span, previous source start,
  previous retained endpoint, guide span and master audio frame/sample offsets.
  Metadata is required when resuming or assembling existing clips, and must
  agree with the current spec. `generation` records the actual expanded prompt,
  canonical mode, seed, negative and applicable strengths. A
  `generation_fingerprint` additionally binds master/image file content and the
  preceding clip/request for continuation and implicit/explicit `@last`.
  Changed inputs or an upstream rerender invalidate reuse with an actionable
  rerender error. Independent images do not depend on preceding clips. Old
  timeline-only metadata is rejected; it is never silently upgraded or inferred
  from a clip's full tail. Reviewed fixture imports must explicitly record their
  verified timeline and matching request fingerprint, plus import provenance.
- Direct `shot N` / `scene N` with `@last` verifies preceding metadata and clip
  geometry/frame count, then freshly extracts its retained endpoint, never a
  pre-existing full-tail PNG. Whole-spec planning recognizes future `@last`
  dependencies without requiring files that earlier items have not rendered.
  It checks known independent image files and validates the selected workflow
  with inert media-name placeholders before any GPU/paid-audio batch work.
- A49 plus B49 with overlap25 retains A[0,49)+B[25,49)=73 frames. B's source
  span is [24,73). C starts at global48, not24 or73; D starts at72. A 73-frame
  buffered A retaining49 still guides B from A[24,49), not A[48,73).
- Assembly calls production `storyboard/lib/assemble.py:assemble_frame_spans`,
  muxes the full master once, checks full decoded video count/geometry/CFR and
  final decoded audio sample count. Master preflight and muxing both use the first audio stream (`0:a:0`).
  The shared assembler validates candidate video/audio before atomically
  replacing the destination. Reassembly invalidates old final timeline
  certificates before planning/clip checks; failures cannot leave an old success
  certificate. A preserved previous MP4 is not evidence of fresh success.
  Inspect the exception and fresh verification metadata, not file existence.

## Strict limits

Profile `video` and shot/scene dictionaries are allowlisted. Unsupported fields
are errors, including LoRAs/camera/IC/HDR, fast, arbitrary models, ingredients,
keyframes/multiguide/transitions, extra image/identity references, anchor
creation, separate lipsync audio and unknown options. Use an explicit image
instead of a character anchor with implicit identity semantics. Even harmless
legacy optional-field placeholders must be removed from a profiled spec; the
caller does not silently drop them. Image guide strengths are forwarded only on
image workflows; specifying them for continuation/text is an error. Use either
`overlap_frames` (already 8k+1) or `overlap_seconds`, not both.

Full generated duration, including prefix/tail/rounding, must fit the existing
caller cap (drama 15 seconds, music 12 seconds). Retained spans need not be
8k+1. Frame boundaries must map to integer 48 kHz samples. There is no automatic
migration of old MP4s without provenance/timeline metadata; imported proof
fixtures were explicitly annotated with their already-verified source spans.

Selected-profile transport now requires terminal successful history and a real
output; generic legacy transport behavior is not certified by these callers. No caller GPU render was performed for this integration.
Real proof is actual drama AND music CLI assembly of the previously rendered
A/B pair: 73 frames, 146000 decoded samples, full decode and aligned master ticks.
This establishes timing/audio correctness, not constant-velocity or perceptually
seamless continuation; the original ball-motion/camera limitations remain.
