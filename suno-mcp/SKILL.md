---
name: suno-mcp
description: Generate AI music with Suno via MCP. Use when creating songs with custom lyrics, instrumental tracks, or specific genres/moods. Triggers on "generate a song", "create music", "make a track", "Suno", "AI music", or any music generation request.
metadata:
  {
    "openclaw":
      {
        "emoji": "🎵",
        "requires": { "skills": ["mcporter"] },
      },
  }
---

# Suno MCP Skill

Run: `python3 ~/.openclaw/skills/suno-mcp/scripts/generate_song.py --lyrics='...' --tags='...' --title='...'` (timeout 400s)

⚠️ `--tags` = full producer brief (NOT keywords). Read first: `cat ~/.openclaw/skills/suno-mcp/references/style-guide.md`

**Lyrics:** `[Verse]`/`[Chorus]`/`[Bridge]`/`[Instrumental]` tags — guide: `cat ~/.openclaw/skills/suno-mcp/references/lyrics-guide.md`

Generate AI music using Suno's API via the MCP server.

## Quick Start

Use the Python helper script — it handles shell quoting safely for long lyrics and detailed style prompts. Direct `mcporter call` on the command line will fail with anything beyond short, simple values.

```python
# 1. Locate the script (works on host and inside sandbox)
SCRIPT=$(find /home -maxdepth 6 -name "generate_song.py" -path "*/suno-mcp/scripts/*" 2>/dev/null | head -1)

# 2. Generate + auto-download (3–5 minutes, use 400s timeout)
exec({
    "command": f"python3 {SCRIPT} --lyrics='{lyrics}' --tags='{tags}' --title='{title}'",
    "timeout": 400
})

# 3. Send the file (local_file path is in the JSON output)
message({
    "action": "send",
    "channel": "discord",
    "filePath": "<local_file from output>",
    "caption": "🎵 Here's your track!"
})
```

For instrumental tracks, pass `--instrumental` instead of `--lyrics`.

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

Suno always generates **two song variants** from a single request. Both are equally valid. Present both to the user with their Suno links so they can listen and choose their favourite. The response from `generate_song` contains IDs for both; download and share whichever the user prefers, or share both.

## Troubleshooting

- **Shell quoting errors**: Always use the Python script, not `mcporter call`, for anything with real lyrics or long style prompts
- **Login required**: Run `mcporter call suno.suno_login --config ~/.openclaw/config/mcporter.json` first
- **Wrong parameter name**: Use `lyrics`, not `prompt` (deprecated)
- **Timeout**: The full pipeline takes 3–5 minutes; always set exec timeout ≥ 400 seconds

## MCP Config

`~/.openclaw/config/mcporter.json` — Server: `https://suno-mcp.tail9683c.ts.net/mcp`
