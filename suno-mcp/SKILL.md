---
name: suno-mcp
description: Generate AI music with Suno via MCP. Use when creating songs with custom lyrics, instrumental tracks, or specific genres/moods. Triggers on "generate a song", "create music", "make a track", "Suno", "AI music", or any music generation request.
metadata:
  {
    "openclaw":
      {
        "emoji": "🎵",
        "requires": { "skills": [] },
      },
  }
---

# Suno MCP Skill

Direct Suno MCP server access for AI music generation. The scripts talk to the
MCP server over plain JSON-RPC / HTTP (no `mcporter`, no `npx`) and download
both song variants to a local directory.

## Server URL configuration

The scripts read two environment variables, with CLI overrides:

| Var | Used by | Falls back to |
|---|---|---|
| `SUNO_MCP_URL`    | every command; `<base>/mcp` for JSON-RPC, `<base>/audio/<file>.mp3` for downloads | `http://localhost:8190` |
| `SUNO_OUTPUT_DIR` | where generated MP3s land | `./outputs/suno` (relative to CWD) |

CLI flags override env vars on every invocation: `--url URL` and `--output-dir DIR`.

**Defaults**: a fresh clone with no env vars set assumes a Suno MCP server
listening on `http://localhost:8190` — the conventional port for the
`bootstrap/media-relay` Worker bridge that fronts Suno. The same bridge serves
`/mcp` (control plane) and `/audio/<file>.mp3` (downloads), so one base URL
covers both.

**Per-tailnet examples** — what the operator actually uses:

| agent location | `SUNO_MCP_URL` |
|---|---|
| `tail9683c` (operator's personal tailnet) | `https://suno-mcp.tail9683c.ts.net` |
| `tail74c072` (`tag:overlord` sandboxes)   | `https://media-relay.tail74c072.ts.net:8190` |

Future media-relays plug in cleanly by exposing the same `/mcp` + `/audio`
surface on their own port; just point `SUNO_MCP_URL` at the new host.

**OpenClaw sandbox**: agents in an agentic-media sandbox have `SUNO_MCP_URL`
(and `SUNO_OUTPUT_DIR`) injected at boot via the per-sandbox `credentials.env`
propagation; agent prompts and skill code don't need to mention them.

## Script paths (use absolute; tilde expansion is unreliable under `exec`)

Resolve the script once at the top of the agent run with `find`, then call it
by absolute path. The skill works equally well from a host install, an
OpenClaw sandbox checkout, or a fresh `git clone`.

**Sandboxed agents (commons / per-role lordships)**: the skill bundle is
RO-bind-mounted at `/agentic-media/creative-skills/`, so the script is
always at `/agentic-media/creative-skills/suno-mcp/scripts/generate_song.py`.
Use the absolute path directly — no `find` needed.

**Host or fresh clone**: include `/agentic-media` in the search roots
*before* the legacy paths, since both can be present simultaneously
when working from an overlord workspace:

```bash
SCRIPT=$(find /agentic-media "$HOME" /home /workspace -maxdepth 7 \
        -name generate_song.py -path '*/suno-mcp/scripts/*' 2>/dev/null | head -1)
python3 "$SCRIPT" --lyrics='...' --tags='...' --title='...'
```

The legacy form `find "$HOME" /home /workspace ...` silently fails in
agentic-media sandboxes because creative-skills are not mirrored under
`/workspace/skills/` — they ship via the `/agentic-media` canonical bind.

⚠️ `--tags` = full producer brief (NOT keywords). Read first: `cat <skill-dir>/references/style-guide.md`.

**Lyrics:** `[Verse]`/`[Chorus]`/`[Bridge]`/`[Instrumental]` tags — guide: `cat <skill-dir>/references/lyrics-guide.md`. Read it before drafting — it's the difference between AI-sounding lyrics and lyrics worth listening to.

**🚨 Two variants per generation.** Suno *always* returns 2 takes from a single request. The helper script downloads BOTH (`local_files` is a list). Present both to the user and let them choose — agents that send only the first systematically lose half the value of every generation. The full output JSON has `local_files` (all variants) and `local_file` (variant 1, kept for backward compat).

## Quick Start

Use the Python helper script — it handles shell quoting safely for long lyrics
and detailed style prompts.

```python
# 1. Locate the script.
#    Sandboxed agents: hard-code /agentic-media/creative-skills/suno-mcp/scripts/generate_song.py.
#    Host / fresh clone: search /agentic-media first, then $HOME/home/workspace.
SCRIPT=$(find /agentic-media "$HOME" /home /workspace -maxdepth 7 \
        -name generate_song.py -path '*/suno-mcp/scripts/*' 2>/dev/null | head -1)

# 2. Generate + auto-download BOTH variants (3–5 minutes, use 400s timeout)
result = exec({
    "command": f"python3 {SCRIPT} --lyrics='{lyrics}' --tags='{tags}' --title='{title}'",
    "timeout": 400
})
# result is a JSON dict with at least:
#   local_files: ["<path>/suno_<id1>.mp3", "<path>/suno_<id2>.mp3"]
#   all_ids:     ["<id1>", "<id2>"]
#   local_file:  <local_files[0]>   (legacy alias)

# 3. Send BOTH to the user, not just the first.
for i, path in enumerate(result["local_files"], start=1):
    message({
        "action": "send",
        "channel": "discord",
        "filePath": path,
        "caption": f"🎵 Variant {i}/{len(result['local_files'])} — https://suno.com/song/{result['all_ids'][i-1]}",
    })
```

For instrumental tracks, pass `--instrumental` instead of `--lyrics`.

## Importing an existing Suno song

If the user gives you a Suno share URL (`https://suno.com/s/<share_id>`)
instead of asking you to generate a new song, you don't need the MCP
generate path at all. Resolve to the canonical share URL in a browser
to read the song UUID, then pull the mp3 directly from Suno's CDN:

```bash
# Share URL → canonical URL → uuid (the canonical URL contains it):
#   https://suno.com/s/34nlQuwqbiaJXzo1
#   → https://suno.com/song/<uuid>?sh=<share_id>
# The same `og:image` meta on the share page contains the uuid; you can
# also use playwright-cli to grab `<meta property="og:image">` directly.

UUID=5717194f-048e-4e92-8f4c-505a44ca3138
curl -sS -o song.mp3 "https://cdn1.suno.ai/${UUID}.mp3"

# Cover image (1200×630 og:image) is also public:
curl -sS -o cover.jpeg "https://cdn2.suno.ai/image_large_${UUID}.jpeg"
```

After the pull, hand `song.mp3` to the music-video skill at step 5
(STT → align scene timings) and skip steps 1-4 of that skill.

---

## Writing Lyrics

Suno uses **section tags** in square brackets to structure songs. Place them on their own line before each section.

### Section tags

| Tag | Purpose |
|-----|---------|
| `[Verse]` | Main storytelling section |
| `[Chorus]` | Repeated hook, usually the emotional peak |
| `[Pre-Chorus]` | Build-up before the chorus |
| `[Bridge]` | Contrasting section, usually once near the end |
| `[Outro]` | Closing section |
| `[Intro]` | Opening section (lyrical or descriptive) |
| `[Hook]` | Short repeated phrase |
| `[Refrain]` | Shorter repeated line within a verse |
| `[Instrumental]` | Tells Suno to play an instrumental break here — no vocals |
| `[Solo]` | Instrumental solo (guitar, sax, etc.) |
| `[Break]` | Rhythmic or percussive break |
| `[Interlude]` | Transitional passage |
| `[Spoken]` | Spoken word delivery (not sung) |
| `[Whispered]` | Whispered vocal delivery |
| `[Ad lib]` | Free improvised vocal fills |
| `[Fade out]` | Signals the song to fade at the end |

### Vocal direction tags

You can add delivery hints inside the lyrics themselves:

- `(soft)`, `(loud)`, `(whispering)`, `(screaming)` — volume/intensity
- `(harmonies)`, `(echo)`, `(choir)` — texture
- `(spoken)`, `(rapped)`, `(chanted)` — delivery style
- `(male vocal)`, `(female vocal)`, `(duet)` — voice

### Example lyrics

```
[Intro]
(soft guitar, ambient)

[Verse]
Salt in the air and rust on the crane,
the fishermen leave before the morning train.
Old radios hum through the fog and the grey,
singing the names of the ones who stayed.

[Chorus]
Glass harbour, glass harbour,
where the cold light bends,
where the sea keeps all the secrets
that the shore pretends.

[Instrumental]

[Verse 2]
A lighthouse blinks its one-eyed prayer,
a child draws boats on the kitchen stair.

[Bridge]
(whispered)
Everything returns to the water.
Everything returns.

[Outro]
(fade out)
```

### Tips for better vocals

- Keep lines to a natural spoken length — Suno matches syllables to melody
- Avoid punctuation mid-line; use line breaks instead
- Rhyme schemes don't have to be strict — half-rhymes work well
- `[Chorus]` repeating the exact same text across the song makes it stick
- Use `[Instrumental]` for breathing room between sections

---

## Writing Style Prompts (tags)

The `tags` field is a **text-to-music prompt**, not a keyword list. Write it like a producer brief: genres, instruments, tempo, production style, atmosphere, and a short narrative description of the track.

### Structure

1. **Genre and sub-genre** — be specific: "trip-hop" not just "hip hop"; "folk noir" not just "folk"
2. **BPM** — e.g. `92 BPM`, `slow 68 BPM`, `uptempo 128 BPM`
3. **Instruments** — list the key ones: `detuned Rhodes chords`, `bowed upright bass`, `slap bass`
4. **Production texture** — `vinyl crackle intro`, `tape delay throws`, `sidechain pumping`, `analog tape saturation`
5. **Vocal style** — `breathy lead vocal`, `half-spoken delivery`, `harmonized chorus`, `world-weary baritone`
6. **Atmosphere / mood** — `nocturnal cityscape`, `minor key melancholy`, `rain-soaked streets`
7. **Narrative sentence** — a short description of how the track unfolds

### Example style prompts

**Folk noir:**
> trip-hop, folk noir, coastal Americana, 74 BPM, fingerpicked acoustic guitar, bowed upright bass, sparse brushed snare, foghorn field recording, salt wind ambience, weathered male vocal, world-weary baritone, close-mic room reverb, minor pentatonic melody, mournful tone, open tuning resonance, subtle string swell on chorus, fading tape hiss outro. A sparse track built around fingerpicked guitar and bowed bass. The vocal sits close and unadorned over brushed percussion and field recordings of harbour wind. A restrained string swell lifts the chorus before the arrangement dissolves back into silence and tape hiss.

**Synthwave:**
> cinematic synthwave, 80s retrowave, 110 BPM, analog polysynth arpeggios, driving gated snare, pulsing Moog bass, detuned pad layer, gated reverb on snare, vocoder harmonies, neon-lit melancholy, nocturnal highway atmosphere. A driving retrowave track that opens on a slow synth pad swell before the gated snare kicks in and the arpeggiator climbs. Vocoder harmonies glide over the chorus as the bass locks into a four-on-the-floor pulse.

**Trip-hop / downtempo:**
> trip-hop, deep house, 92 BPM, sampled ferris wheel ambience, vinyl crackle intro, detuned Rhodes chords, wobbling sub-bass lead, broken kick pattern, shuffled rim clicks, breathy lead vocal, half-spoken delivery, harmonized chorus, tape delay throws, stereo spring reverb, sidechain pumping, analog tape saturation, minor key melancholy. Opens with crumbling drum machine grooves and detuned Rhodes over a pulsing sub-bass. Shuffled rim clicks and vinyl texture build into the chorus with harmonized vocals and tape delay. The bridge dissolves into a half-spoken reverie before returning to the hypnotic groove.

**Tip:** The narrative sentence at the end helps Suno understand the *shape* of the track — how energy builds and falls — not just the sound palette.

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `lyrics` | Full song text with section tags (see above) |
| `tags` | Style prompt — genres, instruments, production, atmosphere, narrative sentence |
| `title` | Song title |
| `make_instrumental` | `true` to skip vocals entirely |
| `negative_prompt` | Styles to avoid, e.g. `"heavy metal, screaming"` |

---

## Output

Suno always generates **two song variants** from a single request. Both are equally valid. The helper script downloads both — `local_files` is the canonical list, `local_file` is a legacy alias for the first variant. Default behaviour for any agent: send both with their Suno preview URLs so the user can pick (see Quick Start step 3).

## CLI Reference

### `generate_song.py` — primary entry point

| Flag | Description |
|------|-------------|
| `--lyrics`, `-l`             | Full song text with section tags |
| `--tags`, `-t` *(required)*  | Style prompt (producer brief; see above) |
| `--title`, `-T` *(required)* | Song title |
| `--instrumental`, `-i`       | Instrumental track (no vocals) |
| `--negative-prompt`, `-n`    | Styles to avoid, e.g. `"heavy metal, screaming"` |
| `--url`                      | Override `SUNO_MCP_URL` env var |
| `--output-dir`               | Override `SUNO_OUTPUT_DIR` env var |
| `--timeout`                  | Total timeout in seconds (default 400) |
| `--dry-run`                  | Print constructed MCP request body, no API call |

### `download_from_mcp.py` — re-download an existing song by ID

| Flag | Description |
|------|-------------|
| `song_id` *(positional)*     | Suno song UUID |
| `--url`                      | Override `SUNO_MCP_URL` env var |
| `--output-dir`, `-o`         | Override `SUNO_OUTPUT_DIR` env var |
| `--max-wait`, `-w`           | Max seconds to wait for file availability (default 120) |

## Dry-run (no API call, no Suno credits)

Pass `--dry-run` to print the constructed MCP request body and target endpoint
without contacting the server. Use during development to verify quoting /
argument shape:

```bash
python3 generate_song.py --dry-run \
  --title 'Test Song' \
  --tags 'trip-hop, 92 BPM' \
  --lyrics '[Verse]\nHello world'
```

## Troubleshooting

- **Connection refused / DNS error**: `SUNO_MCP_URL` is wrong or unreachable. Check the per-tailnet table above; from a sandbox, confirm `credentials.env` exported `SUNO_MCP_URL`.
- **`HTTP 400 Bad Request: Missing session ID`**: the script's MCP client forgot to send `mcp-session-id` after `initialize`. This shouldn't happen in the bundled client — if you see it, the script is stale, re-import.
- **Login required**: the Suno MCP server needs a one-time Google login via the `suno_login` MCP tool. Connect to the server's noVNC URL and complete the OAuth dance; the session persists in the saved Chrome profile.
- **Wrong parameter name**: Use `lyrics`, not `prompt` (deprecated).
- **Timeout**: The full pipeline takes 3–5 minutes; always set exec timeout ≥ 400 seconds.
- **"No song IDs in response"**: the server returned a non-standard payload. Re-run; if it persists, call the `debug_browser` MCP tool to inspect page state.
- **Only one variant in `local_files`**: bug. Both should always be present. Re-run.
- **403 on direct CDN URL**: harmless — the script falls back to the MCP-proxied `<base>/audio/<file>.mp3` URL (and vice-versa).

## Audio URL pattern

The audio URL is always `${SUNO_MCP_URL}/audio/<short>.mp3` — the same Worker
bridge that fronts `/mcp` also serves the downloaded files, so there is no
separate hostname or port to track. The script rebuilds this URL from
`SUNO_MCP_URL` after every `download_song` call, ignoring any host embedded
in the server's response (which is often the server's internal `0.0.0.0`
binding and unreachable off-host).
