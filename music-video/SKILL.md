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

## Day-1 decisions (lock these before step 7)

A few choices propagate into every downstream prompt and every render. Flipping
them mid-pipeline forces you to throw away anchors and redo them. Settle these
up front, when only `song.yaml` exists and no GPU time has been spent:

| Decision | Where it lives | Why it has to be early |
|---|---|---|
| **Aspect & resolution** | `video.resolution: [w, h]` | Hard-coded into every anchor prompt ("vertical 9:16" vs "16:9 widescreen") and every render. This is LTX's **working** resolution; the LTX-2.3 spatial upscaler doubles it for the final output. Common picks: `[832, 448]` (16:9 broadcast → 1664×896 final), `[448, 832]` (9:16 vertical → 896×1664 final), `[576, 1024]` (9:16 a touch larger → 1152×2048 final). Both dims must be multiples of 32. Switching aspect after `anchors` invalidates every PNG. |
| **Anchor scale** | `video.anchor_scale` | Default `2.0` — flux2 anchors render at `resolution × scale` so they carry detail at the final upscaled video resolution. Larger anchors take longer to render but produce noticeably crisper output. Set `1.0` if you'd rather flux2 work at the working resolution (faster, looser final detail). |
| **Cast** | `subjects:` + reference photos in the project dir | Each named subject (`operator`, `dancers`, `interviewer`, …) needs (a) a one-line description token and (b) at least one reference image so per-scene `i2i`/`i2i2` anchors can lock identity. Adding a character later means re-rolling every scene they appear in. |
| **Canonical settings** | `sheets/<setting>.png` (project convention) + `scenes[].anchor.references` | Pre-render one PNG per recurring location (e.g. main stage, b-stage, exterior). Scene anchors `i2i2` against [character, setting] so the world stays visually constant. Without canonical setting refs, each scene anchor invents its own version of the place. |
| **Lipsync mode** | `video.lipsync_audio` | If you want LTX to lipsync against a vocal-forward remix instead of the mastered mix, supply it now. Switching mid-project means re-rendering every singing scene. |
| **Number of suno variants** | `suno.runs` (× 2 variants per run) | More variants = more parallel `final_vN.mp4` outputs at assembly time, but every variant runs against the same scene visuals. Setting this *up* later is cheap; setting it *down* means you waste suno credits. |
| **Quality gates** | `gate_confirm_song`, `gate_confirm_anchors` | Default `true` (recommended). Flip to `false` only when you're fully confident; bad gate-skips waste hours of LTX time on bad anchors. |

Anything else (per-scene prompts, camera LoRAs, transitions on/off, individual
anchor tweaks) is cheap to iterate on later — those only re-render one scene
at a time and don't propagate.

## Command index

Every step of the pipeline maps to one CLI subcommand in
[`scripts/music_video.py`](scripts/music_video.py). The table below
cross-references each subcommand to its handler in code so authors can read
the exact behaviour, and lists every spec field the step consumes.

| Step | CLI | Handler | Reads from spec | Writes |
|---|---|---|---|---|
|  1  | `init <slug> [--theme=<txt>] [--force]` | [`cmd_init`](scripts/music_video.py#L259) | *(no spec; takes `slug` + `--theme` + `--force`)* | `<workspace>/<slug>/song.yaml` skeleton |
|  —  | `plan <spec>` | [`cmd_plan`](scripts/music_video.py#L374) | full spec (validation + breakdown) | stdout |
|  3  | `song <spec>` | [`cmd_song`](scripts/music_video.py#L472) | `title`, `style`, `lyrics`, `suno.runs`, `suno.make_instrumental` | `song.mp3`, `song_v2.mp3`, …, `song_meta.json` |
|  7  | `anchors <spec>` | [`cmd_anchors`](scripts/music_video.py#L731) → [`_generate_anchor`](scripts/music_video.py#L604) per scene | `anchor_image`, `anchor_prompt`, `subjects`, `scenes[].anchor.*` | `anchor.png` (top-level if missing) + `scenes/NNN-<label>-anchor.png` × N |
|  9  | `scene N <spec>` | [`cmd_scene`](scripts/music_video.py#L871) | full scene block, `video.*`, `subjects` | `scenes/NNN-<label>.mp4`, `scenes/NNN-<label>-last.png`, `song_slices/NNN-<label>.mp3` |
|  9  | `scenes <spec>` | [`cmd_scenes`](scripts/music_video.py#L1519) (loops `cmd_scene`) | same | same × N |
|  9b | `transitions <spec>` | [`cmd_transitions`](scripts/music_video.py#L1149) | `video.transitions.*`, `scenes[].transition_from_prev.*` | `scenes/NNN-<label>-transition.mp4` × (N−1) (only if `video.transitions.enabled`) |
| 10  | `assemble <spec>` | [`cmd_assemble`](scripts/music_video.py#L1310) | `scenes[].duration_sec`, `video.transitions.*`, `tail_buffer_sec` | `final.mp4`, `final_v2.mp4`, …  (one per song variant) |
|  *  | `all <spec> [--no-gate]` | [`cmd_all`](scripts/music_video.py#L1420) | `gate_confirm_song`, `gate_confirm_anchors` (+ everything above) | runs `song → anchors → scenes → transitions → assemble`, stops at the two gates by default |
|  —  | `status <spec>` | [`cmd_status`](scripts/music_video.py#L1545) | scenes list (to enumerate expected outputs) | stdout report (✓/—) per file |

**Quality gates** — `cmd_all` stops after `song` (gate 1) and after `anchors` (gate 2) by default. Override per-spec via `gate_confirm_song: false` / `gate_confirm_anchors: false`, or per-invocation via `--no-gate`.

**Workflow-step column (1, 3, 7, 9, 10)** — corresponds to the numbered phases in [**The loop**](#the-loop) above. Steps 2/4/5/6/8 are human review/decisions (variant pick, STT, scene authoring, anchor review), not CLI commands.

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

### Long renders MUST go through `sessions_spawn` — never block the agent

`scenes`, `transitions`, `all` for any non-trivial spec take **30 minutes
to several hours**. The agent invoking those commands must NOT exec them
directly and tail stdout — that locks the agent's turn for hours, floods
the chat with `[18:21:43] transition 2→3 done` lines, and prevents any
further user message from being handled.

The right pattern is OpenClaw's [sub-agent](/tools/subagents) mechanism:
`sessions_spawn` (or `/subagents spawn`) creates a background child run
with **push-based completion**. The parent returns to the user within
seconds; the child runs the long subprocess; when the child finishes,
OpenClaw wakes the parent with the result. **Do not poll** `sessions_list`
or `/subagents list` — the announce is automatic.

Skeleton the agent should follow (pseudocode for the parent's tool call):

```
sessions_spawn({
  agentId: "<self>",          // same agent, isolated child session
  context: "isolated",        // child gets the brief only — no transcript fork
  task: """
    Run the music-video scenes phase for /workspace/<slug>:
      uv run --script /home/sandbox/.openclaw/skills/music-video/scripts/music_video.py \\
        scenes /workspace/<slug>/song.yaml
    Stream nothing back during the run. When complete, report:
      - exit code
      - the count of files in <slug>/scenes/*.mp4
      - the path to <slug>/run.log
  """,
  runTimeoutSeconds: 21600,   // 6 h ceiling — adjust per spec size
})
```

The parent immediately tells the user "Render started in the background,
I'll let you know when it's done" and yields its turn. The next visible
parent turn is the announce when the child finishes (success or fail).

Same pattern for `transitions`, `all`, and `assemble` when the spec is
large. The short-running commands (`init`, `plan`, `song`, `anchors`,
single `scene N`) stay inline — they finish within an agent turn budget.

For raw subprocess detachment without spawning a child agent (you don't
need it here, but it's the same shape), see OpenClaw's [background
tasks](/automation/tasks) — `nohup`, `setsid`, or `disown` on a wrapped
command, with the agent later inspecting the run.log to surface status.

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

### Complete schema reference

Every key the orchestrator reads from a spec, cross-referenced to the
function in [`scripts/music_video.py`](scripts/music_video.py) that consumes
it. Anything not listed is silently ignored.

#### Top-level

| key | type | default | code | meaning |
|---|---|---|---|---|
| `title` | str | `"Untitled"` | [`cmd_song`](scripts/music_video.py#L472) | suno generation title |
| `style` | str | `""` | [`cmd_song`](scripts/music_video.py#L472) | suno producer-style brief |
| `lyrics` | str | `""` | [`cmd_song`](scripts/music_video.py#L472) | suno lyrics with `[Verse]`/`[Chorus]` tags |
| `suno.runs` | int | `1` | [`cmd_song`](scripts/music_video.py#L472) | each run yields 2 variants → `runs * 2` mp3s |
| `suno.make_instrumental` | bool | `false` | [`cmd_song`](scripts/music_video.py#L472) | request instrumental-only variants |
| `subjects` | `dict[str, str]` | `{}` | [`_expand_subjects`](scripts/music_video.py#L352) | `{name}` tokens substituted in every prompt (scene + anchor) |
| `anchor_image` | path | none | [`cmd_anchors`](scripts/music_video.py#L731) | top-level PNG; scenes referencing `image: "@anchor"` use this |
| `anchor_prompt` | str | falls back to `title + style` | [`cmd_anchors`](scripts/music_video.py#L731) | flux2 prompt for the top-level anchor when `anchor_image` is missing |
| `gate_confirm_song` | bool | `true` | [`cmd_all`](scripts/music_video.py#L1420) | `all` halts after `song` for human review |
| `gate_confirm_anchors` | bool | `true` | [`cmd_all`](scripts/music_video.py#L1420) | `all` halts after `anchors` for human review |
| `video` | dict | `{}` | [`_video_spec`](scripts/music_video.py#L310) | renderer config (see below) |
| `scenes` | list | `[]` | [`cmd_plan`](scripts/music_video.py#L374) | scene list (rendered in order) |

#### `video.*`

| key | type | default | code | meaning |
|---|---|---|---|---|
| `video.fps` | int | `24` | [`_video_spec`](scripts/music_video.py#L310) | LTX/output frame rate |
| `video.resolution` | `[w, h]` | `[1024, 576]` | [`_video_spec`](scripts/music_video.py#L310) | LTX **working** resolution (final video is `2×` this — see `anchor_scale`); both dims must be multiples of 32 |
| `video.anchor_scale` | float | `2.0` | [`_generate_anchor`](scripts/music_video.py#L604) | flux2 anchor renders at `resolution × anchor_scale`. Default `2.0` matches LTX-2.3's spatial upscaler so anchors carry real detail at the **final** output resolution. Set to `1.0` for legacy behaviour (anchor at working res). |
| `video.negative` | str | `None` | [`_video_spec`](scripts/music_video.py#L310) | negative prompt applied to every LTX scene |
| `video.tail_buffer_sec` | float | `0.0` | [`_video_spec`](scripts/music_video.py#L310) | extra seconds rendered past each lipsync scene's `duration_sec`, trimmed at assembly (gives LTX phoneme look-ahead) |
| `video.lipsync_audio` | path | none | [`cmd_scene`](scripts/music_video.py#L871) | vocal-forward remix used for ia2v conditioning only; `song.mp3` stays canonical for assembly |
| `video.camera_lora` | str | none | [`cmd_scene`](scripts/music_video.py#L871) | default camera LoRA for scenes that don't set their own |
| `video.camera_lora_strength` | float | `0.8` | [`cmd_scene`](scripts/music_video.py#L871) | default LoRA strength |
| `video.fast` | bool | `false` | [`cmd_scene`](scripts/music_video.py#L871) | default for `scene[].fast` (skip 2-pass refine; iteration shortcut) |
| `video.hdr_lora` | str | none | [`cmd_scene`](scripts/music_video.py#L871) | optional HDR LoRA name; per-scene override available |
| `video.base_guide_strength` | float | `0.9` | [`cmd_scene`](scripts/music_video.py#L871) | LTX base-pass guide strength for multiguide scenes |
| `video.refine_guide_strength` | float | `0.7` | [`cmd_scene`](scripts/music_video.py#L871) | LTX refine-pass guide strength for multiguide scenes |
| `video.transitions` | dict | none | [`cmd_transitions`](scripts/music_video.py#L1149) | per-boundary LTX morph clips (see below) |

#### `video.transitions.*`

| key | type | default | code | meaning |
|---|---|---|---|---|
| `enabled` | bool | `false` | [`cmd_transitions`](scripts/music_video.py#L1149) | turn the `transitions` stage on |
| `duration` | float | `2.0` | [`cmd_transitions`](scripts/music_video.py#L1149) | total per-boundary clip length (seconds); `0` = hard cut |
| `guide_sec` | float | `1.0` | [`cmd_transitions`](scripts/music_video.py#L1149) | real-video guide on each side; middle = `duration − 2·guide_sec` |
| `fps` | int | inherits `video.fps` | [`cmd_transitions`](scripts/music_video.py#L1149) | transition fps |
| `prompt` | str | "smooth morph" | [`cmd_transitions`](scripts/music_video.py#L1149) | default morph prompt; overridable per boundary |
| `default_b_sparse` | str | `"96"` | [`cmd_transitions`](scripts/music_video.py#L1149) | comma list of latent positions for B-side anchor (e.g. `"72,80,88,96"` for 4f into singing) |
| `fast` | bool | inherits `video.fast` | [`cmd_transitions`](scripts/music_video.py#L1149) | skip 2-pass refine on transition renders |

#### `scenes[]` items

| key | type | default | code | meaning |
|---|---|---|---|---|
| `label` | str | `"scene"` | [`_scene_stem`](scripts/music_video.py#L300) | short id; used in output filenames (`NNN-<label>.mp4`) |
| `start_sec` | float | required | [`cmd_scene`](scripts/music_video.py#L871) | scene start in song; drives audio-slice start |
| `duration_sec` | float | required, ≤15s | [`cmd_scene`](scripts/music_video.py#L871) | scene length; hard cap is `MAX_SCENE_DURATION` (15.0s) |
| `prompt` | str | required | [`cmd_scene`](scripts/music_video.py#L871) | LTX ia2v prompt (continuous shot; `{subjects}` tokens expand) |
| `image` | str | `"@anchor"` | [`_resolve_image`](scripts/music_video.py#L813) | first-frame source: `"@anchor"`, `"@last"`, `"@none"` (forces t2v), or a literal path |
| `anchor` | dict | none | [`_generate_anchor`](scripts/music_video.py#L604) | per-scene flux2 pre-render config (see below); when present, this PNG becomes the scene's first frame |
| `guides` | list | none | [`cmd_scene`](scripts/music_video.py#L871) (resolved via `storyboard.lib.guides.resolve_guides`) | mid-scene `LTXVAddGuide` entries (see "Multi-guide scenes") |
| `camera_lora` | str | from `video.camera_lora` | [`cmd_scene`](scripts/music_video.py#L871) | one of `static`, `dolly-in`, `dolly-out`, `dolly-left`, `dolly-right`, `jib-up`, `jib-down` |
| `camera_lora_strength` | float | from `video.camera_lora_strength` | [`cmd_scene`](scripts/music_video.py#L871) | LoRA strength override |
| `fast` | bool | inherits `video.fast` | [`cmd_scene`](scripts/music_video.py#L871) | skip 2-pass refine for this scene |
| `hdr_lora` | str/null | inherits `video.hdr_lora` | [`cmd_scene`](scripts/music_video.py#L871) | per-scene HDR LoRA override; `null` disables |
| `lipsync_audio` | path/null | inherits `video.lipsync_audio` | [`cmd_scene`](scripts/music_video.py#L871) | per-scene ia2v audio override; `null` forces `song.mp3` |
| `base_guide_strength` | float | inherits `video.base_guide_strength` | [`cmd_scene`](scripts/music_video.py#L871) | multiguide base-pass guide strength override |
| `refine_guide_strength` | float | inherits `video.refine_guide_strength` | [`cmd_scene`](scripts/music_video.py#L871) | multiguide refine-pass guide strength override |
| `transition_from_prev` | dict | none | [`cmd_transitions`](scripts/music_video.py#L1149) | override the LTX morph clip on the INCOMING boundary (see below) |

#### `scenes[].anchor.*`

When `anchor.prompt` is set, [`_generate_anchor`](scripts/music_video.py#L604) flux2-pre-renders a PNG per scene and uses it as that scene's first frame, overriding `image`.

| key | type | default | code | meaning |
|---|---|---|---|---|
| `type` | `t2i` \| `i2i` \| `i2i2` \| `i2iN` \| `angles` | inferred from `len(references)` | [`_generate_anchor`](scripts/music_video.py#L604) | flux2 mode |
| `prompt` | str | required | [`_generate_anchor`](scripts/music_video.py#L604) | flux2 prompt; `{subjects}` tokens expand |
| `reference` | path | none | [`_generate_anchor`](scripts/music_video.py#L604) | single reference image (for `i2i`) |
| `references` | list[path] | none | [`_generate_anchor`](scripts/music_video.py#L604) | 2+ references (for `i2i2`, `i2iN`, `angles`); first is the primary |
| `width` | int | from `video.resolution[0]` | [`_generate_anchor`](scripts/music_video.py#L604) | anchor render width override |
| `height` | int | from `video.resolution[1]` | [`_generate_anchor`](scripts/music_video.py#L604) | anchor render height override |
| `steps` | int | flux2 default (~8) | [`_generate_anchor`](scripts/music_video.py#L604) | flux2 sampler steps |
| `keep_identity` | bool | `true` | [`_generate_anchor`](scripts/music_video.py#L604) | for `i2i`/`i2i2`: auto-append "keep face/features from reference" guard. Set `false` to suppress |
| `angle_prompts` | list[str] | `[prompt]` | [`_generate_anchor`](scripts/music_video.py#L604) | for `type: angles` — multi-pose batch from one reference |

**Anchor-type cheatsheet** (full discussion in the [`storyboard`](../storyboard/SKILL.md) skill):

| type | refs | use when |
|---|---|---|
| `t2i` | 0 | pure environment / no character (setting shots) |
| `i2i` | 1 | character into a specific scene (most common) |
| `i2i2` | 2 | character + setting blend, or character A + character B meet |
| `i2iN` | 3+ | small group scenes (cap at 3–4 — identity drifts past that) |
| `angles` | 1 | character-sheet building (multi-pose batch from one reference) |

#### `scenes[].guides[]` items (multi-guide)

Resolved via `storyboard.lib.guides.resolve_guides`.

| key | type | default | meaning |
|---|---|---|---|
| `image` | str | required | `@anchor`, `@last`, or a path; relative paths resolve against project dir |
| `at_sec` | float | — | absolute seconds from scene start (takes precedence) |
| `at_relative` | float (0..1) | — | fraction of `duration_sec` |
| `at_frame` | int | — | explicit LTX latent frame (snaps to multiples of 8) |
| `strength` | float | `1.0` | `LTXVAddGuide` weight |
| `label` | str | — | human-readable, ignored by resolver |

#### `scenes[].transition_from_prev.*`

Goes on the INCOMING scene (B-side), not scene A. Read by [`cmd_transitions`](scripts/music_video.py#L1149).

| key | type | default | meaning |
|---|---|---|---|
| `duration` | float | from `video.transitions.duration` | per-boundary length; `0` = hard cut (LTX morph skipped) |
| `b_sparse` | str | from `video.transitions.default_b_sparse` | B-side latent positions for this boundary |
| `prompt` | str | from `video.transitions.prompt` | morph prompt for this boundary only |

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

**`UnicodeEncodeError` on Windows** → the script prints unicode glyphs (→ ✓ ⚠) and writes them to `run.log`. As of the latest version, `music_video.py` reconfigures stdout/stderr/log file to utf-8 at startup, so this should "just work" on PowerShell / cmd.exe. If you're on an older copy and still see `'charmap' codec can't encode character '→'`, prefix the invocation with `PYTHONIOENCODING=utf-8` or upgrade the skill.

## Advanced (not in v1)

- Beat detection & waveform-linked light effects → not implemented. If needed, add a separate post-process pass with ffmpeg's `showwaves` / `asetnsamples` + blend modes, fed by a beat track from `librosa.onset.onset_strength`.
- Lyric subtitles → not generated. Add an SRT track separately.
