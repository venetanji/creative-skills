# creative-skills

Five composable [AgentSkills](https://agentskills.ai/) that build on each other so an agent can take a text description and produce a finished music video or dialogue-driven drama clip. Designed for agents running under [OpenClaw](https://openclaw.ai/), but the scripts are standalone and work on any Linux host with a reachable ComfyUI server.

```
 prompt ─┐                                                           ┌─► suno song (.mp3, both variants)
         ├─ suno-mcp ────────────────────────────────────────────────┘
 YAML ───┼─ music-video ──┐   (song-first, bar-aligned)
         │                │
         │                ├─► per-scene flux2 anchor (.png)  ─┐
         ├─ storyboard ───┤   (shared: anchors + multi-guide) │
         │                │                                   ├─► LTX ia2v / multiguide → scene.mp4
         ├─ comfyui ──────┤                                   │    + cross-scene LTX transition clips
         │                └─► per-scene audio slice (.mp3)    ┘
         │
         └─ drama-video ──► dialogue-first shots (ElevenLabs audio + LTX ia2v)
         │
         └──────────────► ffmpeg assemble → final.mp4(+v2,v3,…)
```

## The skills

### [`comfyui/`](comfyui/) — direct ComfyUI REST access

A thin Python wrapper that submits ComfyUI workflows by name and downloads the outputs. Everything goes through one CLI:

```bash
python3 scripts/comfy_graph.py t2i   --prompt "..."                     # Flux2 text-to-image
python3 scripts/comfy_graph.py i2i   --image ref.png --prompt "..."     # single-ref edit
python3 scripts/comfy_graph.py i2i2  --image1 a.png --image2 b.png ...  # double-ref blend
python3 scripts/comfy_graph.py multiprompt --image a.png --prompts "angle1\nangle2\n..."  # multi-angle batch
python3 scripts/comfy_graph.py t2v   --prompt "..."  --seconds 10        # LTX-2.3 text-to-video
python3 scripts/comfy_graph.py i2v   --image ref.png --prompt "..."      # LTX-2.3 image-to-video
python3 scripts/comfy_graph.py ia2v  --image ref.png --audio a.mp3 ...   # LTX-2.3 audio-reactive video
python3 scripts/comfy_graph.py flf2v --first a.png --last b.png ...      # LTX-2.3 first-last-frame (default fps=25)
python3 scripts/comfy_graph.py continuation  --prev_video prev.mp4 \
        --prompt "..."  --seconds 8  --audio slice.mp3                   # LTX-2.3 extend an existing clip
python3 scripts/comfy_graph.py multiguide  --guides a.png,b.png,c.png \
        --frame_indices 0,96,168  --strengths 1.0,1.0,1.0 ...            # LTX-2.3 N-anchor chain
python3 scripts/comfy_graph.py transition  --prev_video a.mp4 \
        --next_video b.mp4  --audio slice.mp3 ...                        # LTX-2.3 song-aligned morph
python3 scripts/comfy_graph.py tts   --text "..."                        # Qwen3 TTS
python3 scripts/comfy_graph.py stems --audio song.mp3                    # vocals + instrumental split
python3 scripts/comfy_graph.py stt   --audio vocals.flac                 # whisper transcript + SRTs
python3 scripts/comfy_graph.py vconcat --videos a.mp4,b.mp4 --audio s.mp3  # multi-clip concat
python3 scripts/comfy_graph.py last_frame --video_path /server/path/to/clip.mp4  # SERVER-SIDE path
python3 scripts/comfy_graph.py dump  t2i --prompt "..."                  # print workflow JSON, no execute
```

Add `dump` as a prefix to any workflow command for a side-effect-free preview of the workflow JSON. Useful for diffing against `object_info` to detect node-class drift after a ComfyUI upgrade.

Flags shared across video commands: `--fps`, `--width`, `--height`, `--seconds`, `--seed`, `--negative`, `--camera-lora {static|dolly-{in,out,left,right}|jib-{up,down}}`, `--camera-lora-strength`, `--fast` (skip 2-pass refine).

Plus `video_join.py` (ffmpeg via uv) for post-assembly: `concat`, `trim`, `first-frame`, `last-frame`. PEP 723 inline metadata so `imageio-ffmpeg` pulls itself.

The LTX-2.3 builder uses a **two-pass refine pattern**: coarse 8-step sampler → `LTXVLatentUpsampler` 2× → re-apply image conditioning at strength 1.0 → refine 3-step sampler → decode. Required on the dev-fp8 checkpoint because the distilled LoRA has to be applied explicitly at 0.6 strength — without it the 8-step schedule under-denoises and output is badly blurred. See [`comfyui/references/ltx-prompt-guide.md`](comfyui/references/ltx-prompt-guide.md) for prompt-writing tips.

### [`suno-mcp/`](suno-mcp/) — suno song generation via MCP

Generates songs through a `suno.generate_song` MCP tool (typically hosted at `suno-mcp.<tailnet>/mcp`). Handles:
- Retry-on-transient (HTTP 5xx, SSE errors, suno-side "no songs produced" responses) with 5s → 60s backoff.
- Suno's plain-text response format (extracts IDs and direct CDN download URLs).
- **Both variants** — suno always returns two. `local_files` in the JSON result is the list of all downloaded MP3s.

```bash
python3 scripts/generate_song.py \
  --title "Song Title" \
  --tags "<producer-style prompt with instruments, BPM, mood, narrative sentence>" \
  --lyrics "[Verse] ..."
```

`references/style-guide.md` and `references/lyrics-guide.md` are the source of truth for how to write suno prompts. The short version: **describe sound qualities, don't name artists** (suno filters those), and write lyrics with `[Verse]/[Chorus]/[Bridge]/[Instrumental]` section tags on their own lines.

### [`storyboard/`](storyboard/) — shared anchors + multi-guide resolver

Shared toolkit used by both `music-video` and `drama-video` so prompt conventions and multi-guide scene design are identical across the two. The trap it fixes: passing a raw character *sheet* PNG as the ia2v first frame bakes the sheet's neutral-backdrop lighting into every scene. `storyboard` owns the `i2i` / `i2i2` / `i2iN` / `angles` patterns that produce shot-specific anchors — character *in* the scene's setting, not laid on top of it.

`lib/guides.py` is the resolver that turns a yaml `guides:` list into frame-indexed `LTXVAddGuide` chains. Music-video and drama-video both delegate their multi-guide scenes to it, so the yaml format is the same:

```yaml
guides:
  - image: "@anchor"            # @anchor / @last / literal path
    at_relative: 0.5            # or at_sec / at_frame
    strength: 1.0
```

Use when the scene's first frame can't show the character (opens on a prop or landscape), or for long shots (≥13s) where single-anchor ia2v drifts by the tail.

### [`music-video/`](music-video/) — song-first pipeline orchestrator

A YAML-driven orchestrator that composes the other skills. One YAML describes a full music video; one command runs it end-to-end.

```bash
scripts/music_video.py plan        <spec.yaml>   # validate + show scene breakdown + warn on song/scene length mismatches
scripts/music_video.py song        <spec.yaml>   # suno only (saves all variants as song.mp3 + song_v2.mp3 + ...)
scripts/music_video.py anchors     <spec.yaml>   # flux2 top-level + per-scene anchors (idempotent)
scripts/music_video.py scene N     <spec.yaml>   # one scene: flux2 anchor → LTX ia2v (or multiguide if `guides:` set)
scripts/music_video.py scenes      <spec.yaml>   # every scene in sequence
scripts/music_video.py transitions <spec.yaml>   # per-boundary LTX morph clips (opt-in via video.transitions.enabled)
scripts/music_video.py assemble    <spec.yaml>   # ffmpeg concat + clean song audio overlay (splices in transitions)
scripts/music_video.py all         <spec.yaml>   # end-to-end with two quality gates, restart-safe
scripts/music_video.py status      <spec.yaml>   # what's already on disk
```

Per-scene anchor types (bypasses the `@last` drift problem when chaining scenes):
- `t2i` — pure generation, no reference (setting shots)
- `i2i` — single reference image (character portraits, continuity)
- `i2i2` — **two references blended** (character + setting for placed-in-world shots)
- `angles` — multi-prompt batch (multiple compositions from one reference; first is used as the anchor)

Scene-boundary options:
- **Hard cut** (default) — just concat; fast; fine if the scene prompts change hard.
- **LTX transitions** (`video.transitions.enabled: true`) — per-boundary morph clip using real video frames from both sides as `LTXVAddGuide` conditioning + a masked middle driven by the song audio. Defaults to a 4s clip shape `1s A-guide / 2s masked morph / 1s B-head`. Per-boundary `transition_from_prev` overrides `duration`, `prompt`, and `b_sparse` (B-side anchor shape: `"96"` = 1f, `"72,80,88,96"` = 4f for singing scenes).

Plus `run_overnight.sh` which wraps `all` with a VRAM monitor + per-boundary stitch-similarity report (`frame_check.py`). See [`music-video/SKILL.md`](music-video/SKILL.md) for the full YAML schema and prompting recipe; [`references/example.yaml`](music-video/references/example.yaml) is the full annotated example.

### [`drama-video/`](drama-video/) — dialogue-first sibling

Same LTX ia2v engine as `music-video`, but the primary axis is dialogue cues (from ElevenLabs) instead of song bars. Shots align to per-line timestamps from the audio; cuts happen at pauses and line starts. `continue_from_prev: true` on a shot uses the previous shot's tail frames as a multi-frame `LTXVAddGuide` so same-space shots flow seamlessly without a cut.

Use `drama-video` when the piece is dialogue + ambience; use `music-video` when it's song-driven (even if there's dialogue over it — add dialogue as a per-scene `audio:` override in the music-video spec).

## How they compose

An agent asks the user for a song concept → hand-writes a YAML (style brief, lyrics, scene list with per-scene flux2 anchor prompts + camera LoRAs) → runs `music_video.py all` → delivers the resulting `final.mp4`(s) back over whatever channel.

The orchestrator invokes `suno-mcp/scripts/generate_song.py` for the song, `comfyui/scripts/comfy_graph.py {t2i,i2i,i2i2,multiprompt}` for per-scene anchor generation, and `comfyui/scripts/comfy_graph.py ia2v` for each scene's animated clip. Final assembly is ffmpeg (via `imageio-ffmpeg`) in the orchestrator itself. When one song generation produces multiple variants (suno always does), `assemble` emits one `final_vN.mp4` per variant against the same scene visuals.

## Example projects

Ready-made YAML inputs (one project per song) live in the companion repo
[`venetanji/creative-scripts`](https://github.com/venetanji/creative-scripts).
Pick one, copy it as `song.yaml` into a fresh project dir, supply the
`anchor.png` the spec expects, and run `music_video.py all`. The current
catalog includes folk noir, shoegaze, dreampop, and a 70s-disco lipsync
pipeline (`glitter-down.yaml`) — see that repo's `README.md` for the
full table.

## Install

These are AgentSkill directories. Drop them wherever your runtime expects skills:

```bash
# OpenClaw global (read-only, shared across all agents):
cp -r comfyui suno-mcp storyboard music-video drama-video ~/.openclaw/skills/

# OpenClaw per-agent (workspace-scoped):
cp -r comfyui suno-mcp storyboard music-video drama-video /path/to/agent/workspace/skills/
```

Or use the scripts directly — they all have shebangs (`#!/usr/bin/env python3` or `#!/usr/bin/env -S uv run --script` where a dependency is needed).

## Host requirements

- **ComfyUI** reachable at `COMFY_URL` (default `http://localhost:8188`), with at minimum: Flux2, LTX-2.3 dev-fp8 + distilled LoRA + spatial upscaler, Qwen3 TTS. The LTX camera-control LoRAs are optional but recommended.
- **suno-mcp** reachable as an MCP server, invoked via `mcporter` configured in `~/.openclaw/config/mcporter.json`.
- **uv** for the PEP 723 scripts (`video_join.py`, `music_video.py`, `frame_check.py`, `vram_monitor.py`). `apt install uv` / `brew install uv` / `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Standard Python 3.10+. No other host-level deps — uv pulls `pyyaml`, `imageio-ffmpeg`, `Pillow`, `numpy` into ephemeral envs on first run.

### ComfyUI server URL

The scripts read three env vars, in priority order:

| Var | Used by | Falls back to |
|---|---|---|
| `COMFY_URL_FLUX`  | image / TTS / audio commands | `COMFY_URL` |
| `COMFY_URL_VIDEO` | LTX video commands             | `COMFY_URL` |
| `COMFY_URL`       | single-server fallback         | `http://localhost:8188` |

A fresh clone with no env vars set assumes a ComfyUI server on `http://localhost:8188` — the default port when ComfyUI is started via `python main.py` or Docker.

**ComfyUI Desktop** binds to `http://localhost:8000` instead of `8188`. If you're running the desktop app, set `COMFY_URL=http://localhost:8000` (or persist it in your shell rc).

**Two-server topology** (separate Flux + LTX boxes for VRAM headroom): set both `COMFY_URL_FLUX` and `COMFY_URL_VIDEO`. `comfy_graph.py` routes by command class so a caller doesn't have to track which server.

**OpenClaw sandbox**: agents in an OpenClaw sandbox have these env vars injected at boot via the per-sandbox `credentials.env` propagation, so agent prompts and skill code never need to mention them — `comfy_graph.py t2i …` just works.

## License

MIT.
