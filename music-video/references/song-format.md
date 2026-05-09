# song.yaml format reference

Complete field list. All fields except `title`, `style`, and `scenes` are optional.

## Top-level

| Field | Type | Notes |
|---|---|---|
| `title` | string | Song title. Passed to suno. |
| `style` | string | Producer-style text-to-music prompt (genre, BPM, instruments, production, vocal, mood, narrative sentence). Not a keyword list. See `~/.openclaw/skills/suno-mcp/references/style-guide.md` for the pattern. |
| `lyrics` | string (multi-line) | With `[Verse]/[Chorus]/[Bridge]/[Instrumental]` tags. Omit or leave empty if `suno.make_instrumental: true`. |
| `anchor_image` | path | First scene starts from this image. Relative paths resolve against the YAML's directory. If omitted and the first scene uses `@anchor`, the script falls back to t2v for scene 1. |
| `anchor_prompt` | string | Used by `anchors <spec>` when rendering the top-level anchor PNG (the file at `anchor_image`). Lets you author the anchor in-spec instead of pre-rendering separately. Falls back to `title + style` when omitted. |
| `video` | object | Per-project video defaults (see below). |
| `suno` | object | Per-project suno flags (see below). |
| `scenes` | list | Ordered scene list (see below). Minimum 1. |

## `video` (all optional)

| Field | Default | Notes |
|---|---|---|
| `fps` | 24 | Frame rate for every scene. |
| `resolution` | `[1024, 576]` | `[width, height]`. 768×512 is faster; 1280×720 is slower but sharper. |
| `negative` | (none) | Negative prompt, applied to every scene. Good default: `"pc game, cartoon, modern tech, text, watermark, ugly, blurry"`. |
| `tail_buffer_sec` | 0.0 | Extra seconds appended to lipsync scene renders; trimmed off at assemble. Gives LTX audio lookhead to close the phoneme. Only applied when the scene has an `anchor:` block. |
| `lipsync_audio` | (none) | Path (relative to the project dir) to an alternate audio source used **only for ia2v conditioning** — typically a vocal-forward remix where vocals + backing vocals are boosted ~6dB over the instrumental stems. LTX-2.3's audio head locks onto vocal content cleanly when the lyrics sit above the bed; the full `song.mp3` stays canonical for assembly so the final track is the original mix. The orchestrator hard-fails if the path is set but missing, rather than silently falling back. |
| `fast` | `false` | Global default for `scene[].fast`. When `true`, every scene skips the refine pass (half wall, half resolution) unless that scene sets `fast: false` explicitly. Use during scene authoring; flip back to `false` (or remove) for the final render. |
| `transitions` | (off) | Sub-object — see below. Opt-in per-boundary LTX morph clips between adjacent scenes. |

### `video.transitions` (all optional; whole block optional)

| Field | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch. When true, `transitions` stage renders a clip per boundary and `assemble` splices it in. |
| `duration` | 4.0 | Seconds per transition clip. Shape: `guide_sec` A-guide + middle-morph + `guide_sec` B-guide. |
| `guide_sec` | 1.0 | Length of the real-video guide taken from each side. Middle = `duration - 2*guide_sec` is masked-latent morph. |
| `fps` | 24 | Frame rate for transition clips. Usually match the scene fps. |
| `default_b_sparse` | `"96"` | B-side guide shape. `"96"` = 1 single anchor frame at the tail (smooth, default). `"72,80,88,96"` = 4 sparse anchors (use into singing / lipsync scenes for stronger identity carryover). Never use contiguous multi-frame blocks — they freeze at the snap-in. |
| `prompt` | generic morph | Default prompt for the transition itself. Can be overridden per boundary. |

## `suno`

| Field | Default | Notes |
|---|---|---|
| `make_instrumental` | false | Skip vocals. Requires `lyrics` be empty. |

## `scenes[]`

| Field | Required | Notes |
|---|---|---|
| `label` | yes | Short identifier, used in filenames. Safe chars only (`a-zA-Z0-9_-`). |
| `start_sec` | yes | Seconds into the song where this scene's audio slice begins. |
| `duration_sec` | yes | Scene length. `start_sec + duration_sec` determines where the audio slice ends. Max ~15s per scene (LTX-2.3 OOMs past ~20s at portrait resolutions). |
| `prompt` | yes | Video prompt for ia2v. Describes what's visible — not story. Favor single-take shots. |
| `image` | no (defaults to `@last`) | `@anchor` = use top-level anchor_image. `@last` = use previous scene's last frame. Literal path = use that specific file. |
| `camera_lora` | no | LoRA name (`dolly-in`/`dolly-out`/`dolly-left`/`dolly-right`/`jib-up`/`jib-down`/`static`) or full `.safetensors` filename. Pair with a matching prose description of the move in `prompt`. |
| `camera_lora_strength` | no | Default 0.8. |
| `fast` | no | Skip the refine pass. Half wall time, half resolution. Use for iteration. |
| `anchor` | no | Sub-block for flux2 pre-render of this scene's anchor PNG. Keys: `type: t2i \| i2i \| i2i2 \| i2iN`, `prompt`, `reference`/`references`, `width`, `height`, `steps`. Without an `anchor` block, `@anchor` uses the top-level `anchor_image`. See `storyboard` skill for the anchor design pattern. |
| `guides` | no | List of mid-scene keyframe guides. Each entry: `image` (literal path / `@anchor` / `@last`) + **one of** `at_sec` (float seconds) / `at_relative` (0..1 of shot duration) / `at_frame` (LTX latent frame, snapped to 8) + optional `strength` (default 1.0). When present, routes to LTX multiguide instead of plain ia2v. Use when the scene's first frame doesn't show the character, or to anchor identity through long shots (≥13s). |
| `transition_from_prev` | no | Per-boundary override for the incoming edge. Keys: `duration` (float), `b_sparse` (string `"96"` / `"72,80,88,96"` or list of ints), `prompt` (string). Only applied when `video.transitions.enabled: true`. |

## Timing

The sum of all scene durations should match the song's length. Suno songs are typically 2–4 minutes. Check the output song length with ffprobe if you're slicing a suno track and the generated length is unknown; or just generate the song first and then build the scenes:

```bash
music_video.py song spec.yaml
ffprobe -v error -show_entries format=duration -of csv=p=0 project/song.mp3
# adjust scene start_sec/duration_sec to fit, then:
music_video.py all spec.yaml
```

## Image chaining

`@last` uses the previous scene's **last frame** (auto-extracted by the orchestrator). This produces visual continuity between scenes.

For chorus/verse cuts where continuity should break, use `@anchor` to reset to the initial anchor image — or swap in a dedicated chorus anchor via a literal path:

```yaml
anchor_image: verse-anchor.png

scenes:
  - label: verse1
    image: "@anchor"      # verse-anchor.png
    ...
  - label: chorus1
    image: chorus-anchor.png   # literal path → different anchor
    ...
  - label: verse2
    image: "@last"        # chains from chorus1's last frame
    ...
```

## Prompting guide

**Scene prompts are not story beats.** They're visual descriptions of a single continuous shot. LTX-2.3 handles slow camera moves well (dolly, pan, push-in) and hard cuts badly. Break story beats into separate scenes.

Good:
> "weathered fisherman hauling nets over the gunwale of a small boat, breath visible, early blue-grey light, slow camera pan left"

Bad (has a cut):
> "fisherman pulls in the net, then smiles at camera, then looks out at horizon"

## Lyrics format

See `~/.openclaw/skills/suno-mcp/references/lyrics-guide.md`. Summary:

- `[Verse]`, `[Chorus]`, `[Bridge]`, `[Instrumental]`, `[Outro]`, `[Intro]` tags on their own line
- Delivery hints inline: `(soft)`, `(whispered)`, `(harmonies)`, `(male vocal)`
- One line per phrase, no mid-line punctuation

## Style brief format

See `~/.openclaw/skills/suno-mcp/references/style-guide.md`. Summary:

1. Genre + sub-genre (be specific)
2. BPM
3. Key instruments
4. Production texture
5. Vocal style
6. Atmosphere / mood
7. Short narrative sentence describing how the track unfolds
