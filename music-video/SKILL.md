---
name: music-video
description: >
  Generate a complete music video from a YAML spec — song via suno-mcp, per-scene
  audio-reactive video via comfyui LTX-2.3 ia2v, final concatenation with the clean
  song audio overlaid. Use when the user asks for a music video, long-form video
  synced to music, multi-scene video with a soundtrack, or anything that chains
  multiple ia2v clips into one output. Triggers on: "music video", "video to a song",
  "make a song and a video", "clip to this song", "visualizer", "lyric video".
metadata:
  {
    "openclaw":
      {
        "emoji": "🎬",
        "requires": { "skills": ["comfyui", "suno-mcp"] }
      }
  }
---

# Music Video Skill

Author a YAML spec → run one command → get a full music video.

## The loop

Nine-step workflow with two **quality-check gates** where the pipeline stops
and waits for a human (or reviewing agent) to confirm before the next
expensive phase. A bad song variant or a broken base anchor will ruin every
downstream scene — don't skip the gates unless you're sure.

1. **`init <slug>`** — creates `~/.openclaw/workspace/<slug>/` with a
   skeleton `song.yaml`. Use `--theme "<text>"` to inject context for the
   lyrics/style fields. Example:
   `music_video.py init new-beginnings --theme "fresh start"`
2. **Fill `song.yaml`** — `title`, `style` (producer-style brief — see
   `~/.openclaw/skills/suno-mcp/references/style-guide.md`), and `lyrics`
   (`[Verse]/[Chorus]/[Bridge]` tags, one line per phrase, no mid-line
   punctuation — see
   `~/.openclaw/skills/suno-mcp/references/lyrics-guide.md`). Leave `scenes`
   empty for now.
3. **`song <spec>`** — generates `spec.suno.runs * 2` suno variants
   (`song.mp3`, `song_v2.mp3`, `song_v3.mp3`, ...). Restart-safe: re-runs
   only top up missing variants.
4. **QUALITY GATE 1 — pick the best variant.** Listen to every `song*.mp3`,
   pick the one you want, back up or delete `song.mp3`, then rename your
   chosen variant to `song.mp3`. Keep the others as `song_vN.mp3` — they
   become parallel `final_vN.mp4` renders during assembly.
5. **MANDATORY — transcribe the chosen song for word-level timestamps.**
   You cannot eyeball scene `start_sec`/`duration_sec`. Whisper STT against
   the actual audio is the source of truth. Run:

   ```bash
   # comfy_graph.py uploads the audio to ComfyUI's input dir, runs
   # 'Apply Whisper' + 'Save SRT', and downloads three artefacts to the
   # spec dir: <prefix>.txt (plain transcript), <prefix>_segments.srt
   # (line-level cues), <prefix>_words.srt (per-word cues — use this for
   # lipsync alignment).
   comfy_graph.py stt --audio song.mp3 --prefix <slug> \
     --output-dir <project-dir> --language English
   ```

   Use the segment SRT cues to set `start_sec`/`duration_sec` so each
   scene starts on a vocal phrase boundary and ends just before the
   next one. The orchestrator will slice the song audio per scene at
   exactly these boundaries, so misalignment shows up as words cut in
   half across scene cuts.

   **If you have separated stems (e.g. from Suno's stem-export tool, or
   `comfy_graph.py stems` on the full mix), transcribe the relevant stems
   individually and merge.** Lead vocals, backing vocals, and the full
   mix can each surface lines the others miss:
   - Lead-vocal stem alone may drop hooks/answer-phrases that live only
     in the backing-vocals layer (e.g. a refrain "Stable altitude"
     answered only by the backing vocal).
   - Whisper on the full mix can hallucinate text where backing-vocal
     bleed sounds like a vocal phrase.
   Run STT on each stem with a different `--prefix`, diff the segment
   SRTs, and reconcile timing conflicts in favour of the cleanest stem
   for each line. Don't trust a single pass.

   **Defensive note:** if `stt` fails with HTTP 413, your ComfyUI server
   (or the proxy in front of it) has a body-size limit on `/upload/image`
   below your audio file's size. Either fix the proxy or re-encode the
   audio for the STT pass only:
   `ffmpeg -i song.mp3 -c:a libmp3lame -b:a 24k -ar 16000 -ac 1 song_stt.mp3`
   — that re-encoded copy is fine for transcription but do NOT use it as
   the ia2v audio source (LTX's audio VAE wants the full-quality input).

6. **Add the scene list to `song.yaml`** — each scene: `label`, `start_sec`,
   `duration_sec`, `prompt`, `image` (`@anchor | @last | path`). Optionally
   give a scene an `anchor:` block to pre-render a flux2 key frame for it.
   Lift `start_sec`/`duration_sec` straight from `<prefix>_segments.srt`
   (step 5); each scene's span should land on a vocal-phrase boundary,
   and total duration must equal the song length (probe with
   `ffmpeg -i song.mp3` if unknown).
7. **`anchors <spec>`** — flux2 renders the top-level `anchor_image` (from
   `anchor_prompt`, or fallback `title + style`) plus any per-scene anchors.
   Idempotent — skips files already on disk.
8. **QUALITY GATE 2 — review the anchors.** Open every PNG under
   `<project>/` and `<project>/scenes/`. If anything is wrong (wrong face,
   wrong mood, bad composition), DELETE that PNG and re-run `anchors <spec>`
   — tweak the scene's `anchor.prompt` or top-level `anchor_prompt` first.
   A bad base anchor propagates into every LTX scene that chains off it.
9. **`scenes <spec>`** — renders every scene (LTX ia2v with the song slice +
   resolved image). Restart-safe. Iterate one at a time with
   `scene N <spec>`.
10. **`assemble <spec>`** — concat all scene MP4s, overlay the clean suno
    audio → `final.mp4` (plus `final_vN.mp4` for each extra song variant).

### Running the whole thing

`music_video.py all <spec>` runs steps 3→10 in order. It STOPS automatically
at gates 1 and 2 (controlled by `gate_confirm_song` / `gate_confirm_anchors`
in the YAML, default both true). To run fully unattended, either flip both
flags to `false` in `song.yaml` or pass `--no-gate`:

```bash
music_video.py all <spec>              # stops at gate 1, then gate 2
music_video.py all --no-gate <spec>    # skip both gates, full autopilot
```

Restart-safe throughout: anything already on disk is skipped. Iterate
individual scenes with `scene N song.yaml`.

## Using with openclaw agents

Any agent with `music-video` in its `agents.list[<id>].skills` array can run
this skill end-to-end. The gates make it safe to delegate: the agent runs
`all`, waits at gate 1, and posts the variants back for a human review.
After renaming, the agent is prompted again and runs `all` a second time to
pass gate 2, and so on.

Example — asking Athena (`char6166r`) to start a project:

```
openclaw agents run char6166r --message "Start a music video project: create a song about <theme>, init it in your workspace, and launch the suno generation."
```

Athena will `init`, fill the YAML, run `song`, then report the variants back
(gate 1). After a human renames the chosen variant, Athena resumes with
scene authoring → `anchors` → gate 2 → `scenes` → `assemble`.

## Script paths (absolute; no `~`)

The canonical install path is `~/.openclaw/skills/music-video/scripts/music_video.py`.
`~/.openclaw/…` works in an interactive shell but NOT always under `exec`, so
when invoking from another agent or process pass an absolute path:

- OpenClaw sandbox: `/home/sandbox/.openclaw/skills/music-video/scripts/music_video.py`
- Host install: `$HOME/.openclaw/skills/music-video/scripts/music_video.py`
- Fresh clone: `<repo>/music-video/scripts/music_video.py`

The script uses a `uv run --script` shebang with inline deps (`pyyaml`,
`imageio-ffmpeg`). **Invoke it via `uv run --script`** so deps auto-install:

```bash
uv run --script /path/to/music-video/scripts/music_video.py <cmd> ...
```

Plain `python3 music_video.py …` will fail with `ModuleNotFoundError: yaml`
unless you've already installed `pyyaml` in your env. Use `uv` instead — it's
the designed path.

## Commands

```bash
# Prefix every command below with the uv invocation above.
music_video.py init <slug> [--theme=<text>] [--force]   # create project skeleton under ~/.openclaw/workspace/<slug>/ (sandboxed agents get /workspace/<slug>/)
music_video.py plan     <spec.yaml>       # validate + show scene breakdown
music_video.py song     <spec.yaml>       # suno only — N*2 variants (spec.suno.runs)
music_video.py anchors  <spec.yaml>       # flux2 top-level + per-scene anchors (idempotent)
music_video.py scene N  <spec.yaml>       # one scene's ia2v (1-3 min)
music_video.py scenes   <spec.yaml>       # every scene in sequence
music_video.py transitions <spec.yaml>    # LTX transition clips per boundary (runs if video.transitions.enabled)
music_video.py assemble <spec.yaml>       # concat + overlay song audio (splices in transitions)
music_video.py all      <spec.yaml> [--no-gate]   # end-to-end, stops at the 2 gates by default
music_video.py status   <spec.yaml>       # what's already generated
```

Typical for a 3-scene 60-second video: ~6 minutes wall time (song ~4 min, scenes ~45s each sequential, assembly a few sec).

## YAML spec

Minimum viable:

```yaml
title: Glass Harbour
style: "folk noir, coastal Americana, 74 BPM, fingerpicked acoustic guitar, bowed upright bass, sparse brushed snare, foghorn field recording, salt wind ambience, weathered male vocal, close-mic room reverb, mournful tone"
lyrics: |
  [Verse]
  Salt in the air and rust on the crane,
  the fishermen leave before the morning train.
  [Chorus]
  Glass harbour, glass harbour,
  where the cold light bends.

video:
  fps: 24
  resolution: [1024, 576]
  negative: "pc game, cartoon, modern tech, text, ugly"
  crossfade: 0.5                      # optional; ffmpeg xfade between scenes
  camera_lora_strength: 0.8           # default applied per-scene if camera_lora is set
  fast: true                          # optional global default for scene[].fast (iteration shortcut; flip off for final)
  # lipsync_audio: song_lipsync.mp3    # optional — vocal-forward remix used for ia2v
  #                                      conditioning only. song.mp3 stays canonical
  #                                      for assembly. Hard-fails if set but missing.
  #                                      Per-scene override: scene[].lipsync_audio
  #                                      (use null to force song.mp3 on instrumental
  #                                      / guitar-riff scenes where the vocal-forward
  #                                      mix would mute the music dynamics).

anchor_image: anchor.png                # first scene uses this; omit → t2v
# anchor_prompt: "..."                  # optional — drives `anchors <spec>` for the
#                                         top-level PNG. Falls back to title + style.

scenes:
  - label: establishing
    start_sec: 0
    duration_sec: 8
    prompt: "wide fog-shrouded harbour at dawn, lonely crane silhouette, salt spray, muted palette"
    image: "@anchor"
  - label: fisherman
    start_sec: 8
    duration_sec: 10
    prompt: "weathered fisherman hauling nets, breath visible, early light"
    image: "@last"                      # chains from prev scene's last frame
  - label: lighthouse
    start_sec: 18
    duration_sec: 8
    camera_lora: static                 # optional: dolly-in/out/left/right, jib-up/down, static
    prompt: "lighthouse blinks through fog, beam sweeps across waves"
    image: "@last"
```

### Camera LoRAs

Seven cinematic motion LoRAs are installed on the comfy server and can be
applied per-scene via `camera_lora: <name>`:

| name | effect |
|---|---|
| `static` | locks camera still (good for contemplative shots) |
| `dolly-in` / `dolly-out` | push in / pull back |
| `dolly-left` / `dolly-right` | lateral tracking |
| `jib-up` / `jib-down` | vertical crane |

Defaults to no LoRA if unset. Strength via `camera_lora_strength` (top-level
default 0.8; overridable per-scene). Pair the LoRA name with a matching prose
description of the same move in the prompt — the LoRA reinforces the intent.

### Scene transitions (LTX, not ffmpeg)

Boundaries between `@last`-chained scenes usually land with a visible seam —
ia2v conditioning drifts enough that adjacent scenes don't quite meet. The
`transitions` stage renders a short LTX clip per boundary that morphs from
scene A's tail into scene B's head, using real video guides from both
sides (not still frames) so the motion continues across the cut.

Enable + tune in `video.transitions`:

```yaml
video:
  transitions:
    enabled: true
    duration: 4.0            # total transition length (seconds)
    guide_sec: 1.0           # real-video guide on each side; middle = dur - 2*guide_sec
    fps: 24
    default_b_sparse: "96"   # "96" = 1 single B-anchor frame at the tail (default)
                             # "72,80,88,96" = 4 sparse B-anchors (smoother into singing)
```

The resulting shape for a 4.0s / 1.0s boundary is **1s A-guide → 2s masked
morph (audio drives the invention) → 1s that ends on scene B's head**.
Assemble overlaps the transition with scene A's trailing `duration/2`s and
scene B's leading `duration/2`s so the final timeline stays bit-exact with
the song.

**Per-boundary override** goes on the INCOMING scene (not scene A) via
`transition_from_prev`:

```yaml
scenes:
  - label: chorus_in           # boundary is prev_scene → this_scene
    transition_from_prev:
      duration: 4.0
      b_sparse: "72,80,88,96"  # 4f — use INTO lipsync / singing scenes
      prompt: "…optional override for this boundary's morph…"
```

**Hard cut**: set `duration: 0` to skip the LTX transition for one
boundary. The assemble step butt-cuts straight from prev scene's tail
to this scene's head — useful on drum-hit / chorus-crash entries
where you want the impact, not a smooth morph. Pair with a punchy
visual change (palette flip, location jump) for maximum bite.

```yaml
scenes:
  - label: chorus_crash
    transition_from_prev:
      duration: 0              # ← hard cut, no LTX transition rendered
```

**Defaults that work:** 4s duration, 1s guide, `default_b_sparse: "96"` (1f).
4f (`"72,80,88,96"`) was empirically the sweet spot for boundaries into
singing shots — more anchors prevent identity drift when the character
immediately has to sing. Contiguous multi-frame B-blocks (e.g. 16 frames at
positions 80-96) caused a mid-transition freeze at snap-in; always keep
the B-side sparse (every 8 latent frames).

### Multi-guide scenes (`guides:` field)

A scene whose first frame can't/shouldn't show the character — e.g. opens
on an empty landscape or a prop before the character enters — needs a
mid-scene character anchor or LTX will invent a random identity. Add a
`guides:` list with pre-rendered flux anchors at specific times:

```yaml
scenes:
  - label: meets_hare
    start_sec: 98.110
    duration_sec: 13.630
    image: "sheets/mid_anchors/hare_candle_first_00001_.png"   # candle only
    prompt: "Wide dusk-meadow — a candle on a stone; {hare} cartwheels around it; {snakebird} shields the flame…"
    guides:
      - image: "sheets/mid_anchors/hare_character_00001_.png"
        at_relative: 0.5         # halfway through the shot
        strength: 1.0
```

Each guide entry: `image` (literal path, `@anchor`, or `@last`) + **one of**
`at_sec` / `at_relative` (0..1) / `at_frame` + optional `strength` (default 1.0).
Frame indices snap to multiples of 8 (LTX latent quantization). For long
shots (13s+), add two guides (e.g. at 0.5 and 0.9) to hold identity through
the whole shot — a single mid-anchor drifts by the tail.

When `guides:` is present, the scene renders via LTX multiguide (chained
`LTXVAddGuide`). The shared implementation lives in the `storyboard` skill
(`lib/guides.py`) so drama-video and one-off renders use the same resolver.

A fuller reference with all optional fields is at `references/example.yaml`. Format details in `references/song-format.md`.

## Folder layout (auto-created)

```
<project>/
  song.yaml
  song.mp3
  song_meta.json
  song_slices/
    001-establishing.mp3
    002-fisherman.mp3
    003-lighthouse.mp3
  scenes/
    001-establishing.mp4
    001-establishing-last.png
    002-fisherman.mp4
    002-fisherman-last.png
    003-lighthouse.mp4
    003-lighthouse-last.png
  final.mp4
  run.log
```

`<project>` = the directory containing `song.yaml`. The script writes everything there.

## Prompting recipe (for agents filling the YAML)

1. **Style brief** is a producer-style text-to-music prompt. See `~/.openclaw/skills/suno-mcp/references/style-guide.md` for the full pattern (genre, BPM, instruments, production, vocal, mood, narrative sentence). Avoid keyword lists.
2. **Lyrics** use `[Verse]/[Chorus]/[Bridge]/[Instrumental]` tags. One line per phrase; no mid-line punctuation. See `~/.openclaw/skills/suno-mcp/references/lyrics-guide.md`.
3. **Scenes** should map to lyric structure. Typical layout for a 2-minute song at ~90 BPM:
   - verse A → 1 scene, 15-20s
   - chorus 1 → 1 scene, 10-15s
   - verse B → 1 scene
   - chorus 2 → 1 scene
   - bridge/outro → 1 scene
   Align `start_sec`/`duration_sec` with the actual song structure. Duration accuracy matters for audio-reactive feel.
4. **Scene prompts** should describe what's on screen — not tell a story. LTX-2.3 handles short camera moves well (dolly, pan, push-in) and struggles with cuts. Keep each scene as a continuous shot.
5. **Image chain** with `@last` for continuity between adjacent scenes; use `@anchor` or a literal path to reset to a different look (verse/chorus transitions).
6. **Resolution**: 1024×576 is a good default for music videos. 768×512 is faster. Higher eats GPU.
7. **Negative prompt**: use the `video.negative` field to blacklist "ugly, pc game, cartoon, text" etc. Keeps LTX-2.3 from drifting.

## Troubleshooting

**"song.mp3 missing"** when running `scene` → run `song` first, or use `all`.

**"scene N: @last requested but ...-last.png does not exist"** → previous scene hasn't been generated yet; run them in order or use `all`.

**Final video's audio feels off** → each scene is conditioned on its audio slice, but the final pass overlays the clean suno track. If a scene's video is mis-synced, regenerate that scene (`scene N spec.yaml`) — suno audio is not re-generated, just the video.

**Suno login required** → `mcporter call suno.suno_login --config ~/.openclaw/config/mcporter.json` first.

**Scenes visually inconsistent** → pin `@anchor` on the first scene and chain `@last` through the rest. Add specific character/place descriptors in every scene prompt, not just the first.

## Advanced (not in v1)

- Beat detection & waveform-linked light effects → not implemented. If needed, add a separate post-process pass with ffmpeg's `showwaves` / `asetnsamples` + blend modes, fed by a beat track from `librosa.onset.onset_strength`.
- Lyric subtitles → not generated. Add an SRT track separately.
