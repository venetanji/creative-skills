---
name: drama-video
description: Build a dialogue-first narrative video. Use when the user wants a "scene", "drama", "monologue", "dramatic clip", or a character acting through lines (not a music video). Spec authors the character, an ElevenLabs audio bed (dialogue + ambience + pauses) and a shot list whose start/duration aligns to the dialogue cues, then this skill renders flux anchors → LTX ia2v shots → vconcat + overlay full audio. Sibling to `music-video` (which is song-first and bar-aligned).
---

# drama-video

A narrative-first sibling to `music-video`. The primary axis is
**dialogue** — you author lines with pauses, generate them through
ElevenLabs (along with an ambience bed), and cut shots against the
per-line timestamps the API returns. Each shot is one LTX ia2v render
(image + audio slice → video); shots chain via `@last` so the character's
identity carries across cuts.

## When to use this vs `music-video`

| Use | If the primary driver is |
|---|---|
| `music-video` | a song (scene lengths snap to bars/BPM, transitions land on beat) |
| `drama-video` | dialogue + ambience (shots align to cue boundaries, cuts happen at pauses or line starts) |

If the piece has both (song *and* dialogue over it), use `music-video`
and add the dialogue as a per-scene `audio:` override.

## Spec shape

```yaml
title: "Fennie's Moment"
description: "Fennie practising alone late at night…"        # optional, for logs

character:
  name: Fennie
  anchor: ./path/to/character_anchor.png                     # primary identity image
  description: "Fennie — 17-year-old Korean ballet trainee…" # optional, available as {character} in prompts

video:
  fps: 24
  resolution: [1024, 576]
  negative: "cartoon, text, watermark, distorted face, still frame, low quality, extra fingers"
  tail_buffer_sec: 0.5                                       # same semantics as music-video

audio:
  # Either point at an already-generated scene audio mp3 + sidecar:
  file: ./outputs/scene_complete.mp3
  cues: ./outputs/scene_complete.cues.json                   # optional — if present, `drama_video.py plan` validates shot starts/ends against line cues
  # …or author inline and let this skill call the elevenlabs scripts:
  # spec: ./scene_audio.yaml                                 # an eleven_scene_audio.py spec; this skill renders it in stage 1

gate_confirm_audio: false
gate_confirm_anchors: true

shots:
  - label: opening_focus
    start_sec: 0.00
    duration_sec: 13.43                                      # ≤ 15s (LTX cap)
    image: "@anchor"                                         # @anchor | @last | path
    camera_lora: static                                      # optional
    camera_lora_strength: 0.8
    prompt: "Close-mid shot of {character} in a late-night dance studio…"
    anchor:                                                  # optional flux2 pre-gen anchor (same block as music-video)
      type: i2i
      reference: ./path/to/character_anchor.png
      prompt: "Wide shot of the studio, overhead spot, wooden floor…"

  - label: frustration
    start_sec: 13.43
    duration_sec: 10.85
    image: "@last"
    camera_lora: dolly-in
    prompt: "{character} sinks onto the floor…"
```

### Shot durations

Hard cap **15 s** per shot (same as music-video, for the same LTX-2.3
OOM reasons). If a cue range is longer, split it into two shots with
the same prompt trend — you pick the visual cut point.

### Continuation between shots (recommended for most drama)

A regular `image: "@last"` shot starts from the previous shot's *last
single frame* — LTX re-interprets that one frame and often produces a
visible cut even with the transition LoRA. For truly seamless
continuation, use the `continue_from_prev` flag:

```yaml
shots:
  - label: opening_focus          # shot 1 — uses anchor (scene setup)
    start_sec: 0.0
    duration_sec: 4.68
    image: "@anchor"
    anchor:
      type: i2i
      reference: ./media/character.png
      prompt: "The character in the studio, overhead light, …"
    prompt: "Close-mid shot, she exhales, rises onto her toes …"

  - label: first_attempt          # shot 2 — continues from shot 1
    start_sec: 4.68
    duration_sec: 7.96
    continue_from_prev: true      # ← locks first 1s to shot 1's last 1s
    overlap_seconds: 1.0          # ← default 1.0 (0.5–1.5s is the sweet spot)
    overlap_strength: 1.0         # ← default 1.0 (hard lock); 0.85 for softer
    prompt: "She attempts the spin, mid-motion, arms extended …"
    # no `image:` or `anchor:` needed — LTX picks up from shot 1's tail
```

How it works: the pipeline invokes the `continuation` comfyui workflow,
which loads the previous shot's mp4, extracts the last N frames via
`LoadVideo → GetVideoComponents → GetImageRangeFromBatch`, and injects
them as a multi-frame `LTXVAddGuide(frame_idx=0, strength=overlap_strength)`
on an empty latent. The rest of the shot is generated from the new
prompt + audio slice.

**When to use `continue_from_prev`:** same physical space, time flows
forward, character continues to be present. Most drama shots fit this.

**When to fall back to `anchor` + fresh ia2v:** hard scene change
(different room, different character, flashback). Skip
`continue_from_prev` on that shot; give it a fresh `anchor:` block.

## CLI

```
drama_video.py plan      <spec.yaml>           # show shots + audio alignment
drama_video.py audio     <spec.yaml>           # render audio via elevenlabs (if audio.spec set)
drama_video.py anchors   <spec.yaml>           # pre-render all shot anchors via flux2
drama_video.py shot N    <spec.yaml>           # render one shot (1-indexed)
drama_video.py shots     <spec.yaml>           # render all shots serially (skips existing)
drama_video.py assemble  <spec.yaml>           # vconcat + overlay the full audio
drama_video.py all       <spec.yaml>           # 1→5 end-to-end, with quality gates
drama_video.py status    <spec.yaml>           # show what's on disk
```

Each stage is idempotent — mp4s and anchors are skipped if already
present. To force a re-render, `rm` the artifact and re-run the stage.

## Output layout

```
<project>/
  spec.yaml
  audio/                       # if audio.spec was used
    scene_complete.mp3
    scene_complete.cues.json
  shots/
    001-opening_focus-anchor.png
    001-opening_focus.mp3      # audio slice for this shot
    001-opening_focus.mp4
    001-opening_focus-last.png
    …
  final.mp4
  run.log
```

## Prompt conventions

- The top-level `character.description` is available as `{character}`
  inside any shot prompt (and anchor prompt). Recurring-character
  substitution just like `music-video`'s `subjects:` block — useful so
  you don't copy/paste the identity description into every shot.
- Inside anchor prompts, refer to the character by name or via
  `{character}` so flux2 i2i preserves identity when blending with a
  new scene environment.
- Keep shot prompts tight: one sentence for setting, one for action,
  one for camera motion. Long shot prompts compete with the character
  anchor and drift identity.

## Gotchas

- **Shot 1's `@anchor`** resolves to `character.anchor`; later shots
  with `image: "@last"` chain from the prev shot's `-last.png`. First
  shot *must* have either `@anchor` or a literal `image:` path.
- The `audio` block must exist before `shots` stage runs. If you set
  `audio.spec:`, the `audio` stage renders it; otherwise `audio.file:`
  must already exist on disk.
- Shot `start_sec + duration_sec` must stay within the audio file's
  length. `plan` checks this.
- If your audio has `cues.json`, `plan` warns when a shot's
  `start_sec` lands mid-line (cuts across a word).
