# creative-skills

Three composable [AgentSkills](https://agentskills.ai/) that build on each other so an agent can take a text description and produce a finished music video. Designed for agents running under [OpenClaw](https://openclaw.ai/), but the scripts are standalone and work on any Linux host with a reachable ComfyUI server.

```
 prompt ─┐                                                           ┌─► suno song (.mp3, both variants)
         ├─ suno-mcp ────────────────────────────────────────────────┘
 YAML ───┼─ music-video ──┐
         │                ├─► per-scene flux2 anchor (.png)  ─┐
         ├─ comfyui ──────┤                                   ├─► ia2v → scene.mp4
         │                └─► per-scene audio slice (.mp3)    ┘
         │
         └──────────────► ffmpeg assemble → final.mp4(+v2,v3,…)
```

## The three skills

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
python3 scripts/comfy_graph.py flf2v --first a.png --last b.png ...      # LTX-2.3 first-last-frame
python3 scripts/comfy_graph.py tts   --text "..."                        # Qwen3 TTS
```

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

### [`music-video/`](music-video/) — full pipeline orchestrator

A YAML-driven orchestrator that composes the other two. One YAML describes a full music video; one command runs it end-to-end.

```bash
scripts/music_video.py plan     <spec.yaml>     # validate + show scene breakdown + warn on song/scene length mismatches
scripts/music_video.py song     <spec.yaml>     # suno only (saves all variants as song.mp3 + song_v2.mp3 + ...)
scripts/music_video.py scene N  <spec.yaml>     # one scene: flux2 anchor (t2i/i2i/i2i2/angles) → LTX ia2v
scripts/music_video.py assemble <spec.yaml>     # ffmpeg concat (optional xfade) + clean song audio overlay
scripts/music_video.py all      <spec.yaml>     # end-to-end, restart-safe
scripts/music_video.py status   <spec.yaml>     # what's already on disk
```

Per-scene anchor types (bypasses the `@last` drift problem when chaining scenes):
- `t2i` — pure generation, no reference (setting shots)
- `i2i` — single reference image (character portraits, continuity)
- `i2i2` — **two references blended** (character + setting for placed-in-world shots)
- `angles` — multi-prompt batch (multiple compositions from one reference; first is used as the anchor)

Plus `run_overnight.sh` which wraps `all` with a VRAM monitor + per-boundary stitch-similarity report (`frame_check.py`). See [`music-video/SKILL.md`](music-video/SKILL.md) for the full YAML schema and prompting recipe; [`references/example.yaml`](music-video/references/example.yaml) is the full annotated example.

## How they compose

An agent asks the user for a song concept → hand-writes a YAML (style brief, lyrics, scene list with per-scene flux2 anchor prompts + camera LoRAs) → runs `music_video.py all` → delivers the resulting `final.mp4`(s) back over whatever channel.

The orchestrator invokes `suno-mcp/scripts/generate_song.py` for the song, `comfyui/scripts/comfy_graph.py {t2i,i2i,i2i2,multiprompt}` for per-scene anchor generation, and `comfyui/scripts/comfy_graph.py ia2v` for each scene's animated clip. Final assembly is ffmpeg (via `imageio-ffmpeg`) in the orchestrator itself. When one song generation produces multiple variants (suno always does), `assemble` emits one `final_vN.mp4` per variant against the same scene visuals.

## Install

These are AgentSkill directories. Drop them wherever your runtime expects skills:

```bash
# OpenClaw global (read-only, shared across all agents):
cp -r comfyui suno-mcp music-video ~/.openclaw/skills/

# OpenClaw per-agent (workspace-scoped):
cp -r comfyui suno-mcp music-video /path/to/agent/workspace/skills/
```

Or use the scripts directly — they all have shebangs (`#!/usr/bin/env python3` or `#!/usr/bin/env -S uv run --script` where a dependency is needed).

## Host requirements

- **ComfyUI** reachable at `COMFY_URL` (default `https://comfyui.tail9683c.ts.net`), with at minimum: Flux2, LTX-2.3 dev-fp8 + distilled LoRA + spatial upscaler, Qwen3 TTS. The LTX camera-control LoRAs are optional but recommended.
- **suno-mcp** reachable as an MCP server, invoked via `mcporter` configured in `~/.openclaw/config/mcporter.json`.
- **uv** for the PEP 723 scripts (`video_join.py`, `music_video.py`, `frame_check.py`, `vram_monitor.py`). `apt install uv` / `brew install uv` / `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Standard Python 3.10+. No other host-level deps — uv pulls `pyyaml`, `imageio-ffmpeg`, `Pillow`, `numpy` into ephemeral envs on first run.

## License

MIT.
