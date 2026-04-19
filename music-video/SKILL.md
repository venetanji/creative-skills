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

1. Author `song.yaml` — title, style brief, lyrics, scene list (each: label, start_sec, duration_sec, prompt, image).
2. `music_video.py all song.yaml` runs the whole pipeline:
   - suno-mcp generates the song (`song.mp3` + `song_meta.json`)
   - ffmpeg slices the song per-scene (`song_slices/`)
   - For each scene: comfyui `ia2v` with that slice as audio + an image (anchor or previous scene's last frame)
   - Last frame of each scene is extracted → becomes next scene's anchor (visual continuity)
   - All scenes concat + clean song audio overlaid → `final.mp4`

Restart-safe: rerunning `all` skips anything already on disk. Iterate individual scenes with `scene N song.yaml`.

## Script paths (absolute; no `~`)

- Sandbox (default): `/home/sandbox/.openclaw/skills/music-video/scripts/music_video.py`
- Host (main/zeus): `~/.openclaw/skills/music-video/scripts/music_video.py`

The script is a `uv run --script` shebang — no manual deps needed; uv pulls `pyyaml` + `imageio-ffmpeg` on first run.

## Commands

```bash
music_video.py plan     <spec.yaml>       # validate + show scene breakdown
music_video.py song     <spec.yaml>       # suno only (3-5 min)
music_video.py scene N  <spec.yaml>       # one scene's ia2v (1-3 min)
music_video.py assemble <spec.yaml>       # concat + overlay song audio
music_video.py all      <spec.yaml>       # end-to-end
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

anchor_image: anchor.png                # first scene uses this; omit → t2v

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

### Crossfades

Set `video.crossfade: 0.5` (seconds) to apply an ffmpeg `xfade=transition=fade`
at every scene boundary during `assemble`. Hides the stitching artifacts that
can appear when passing one scene's last frame as the next scene's anchor
(ia2v conditioning has some drift, so cuts aren't perfectly seamless). Costs
a re-encode on assembly. Leave at 0 for hard cuts.

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
- Cross-fades at scene boundaries → currently hard cuts. Add xfade via `video_join.py` extension if needed.
- Lyric subtitles → not generated. Add an SRT track separately.
