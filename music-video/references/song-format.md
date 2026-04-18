# song.yaml format reference

Complete field list. All fields except `title`, `style`, and `scenes` are optional.

## Top-level

| Field | Type | Notes |
|---|---|---|
| `title` | string | Song title. Passed to suno. |
| `style` | string | Producer-style text-to-music prompt (genre, BPM, instruments, production, vocal, mood, narrative sentence). Not a keyword list. See `~/.openclaw/skills/suno-mcp/references/style-guide.md` for the pattern. |
| `lyrics` | string (multi-line) | With `[Verse]/[Chorus]/[Bridge]/[Instrumental]` tags. Omit or leave empty if `suno.make_instrumental: true`. |
| `anchor_image` | path | First scene starts from this image. Relative paths resolve against the YAML's directory. If omitted and the first scene uses `@anchor`, the script falls back to t2v for scene 1. |
| `video` | object | Per-project video defaults (see below). |
| `suno` | object | Per-project suno flags (see below). |
| `scenes` | list | Ordered scene list (see below). Minimum 1. |

## `video` (all optional)

| Field | Default | Notes |
|---|---|---|
| `fps` | 24 | Frame rate for every scene. |
| `resolution` | `[1024, 576]` | `[width, height]`. 768×512 is faster; 1280×720 is slower but sharper. |
| `negative` | (none) | Negative prompt, applied to every scene. Good default: `"pc game, cartoon, modern tech, text, watermark, ugly, blurry"`. |

## `suno`

| Field | Default | Notes |
|---|---|---|
| `make_instrumental` | false | Skip vocals. Requires `lyrics` be empty. |

## `scenes[]`

| Field | Required | Notes |
|---|---|---|
| `label` | yes | Short identifier, used in filenames. Safe chars only (`a-zA-Z0-9_-`). |
| `start_sec` | yes | Seconds into the song where this scene's audio slice begins. |
| `duration_sec` | yes | Scene length. `start_sec + duration_sec` determines where the audio slice ends. |
| `prompt` | yes | Video prompt for ia2v. Describes what's visible — not story. Favor single-take shots. |
| `image` | no (defaults to `@last`) | `@anchor` = use top-level anchor_image. `@last` = use previous scene's last frame. Literal path = use that specific file. |

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
