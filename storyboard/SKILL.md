---
name: storyboard
description: Shared toolkit for building per-shot reference images (anchors) and keyframe sequences via flux2 i2i/i2i2/multi-reference. Use when any pipeline — music-video, drama-video, a one-off render — needs a scene-specific anchor, a character pose sheet, or first/last keyframes for a transition. Exposes a `generate_anchor` CLI + a prompting guide the operator should skim before writing prompts. All production-quality anchors on this deployment go through this skill.
---

# storyboard

Shared foundation for turning a recurring character + a scene description
into a **scene-specific anchor image** that LTX ia2v / flf2v / multiguide
can consume. Music-video and drama-video both delegate their anchor
stage to this skill; one-off renders can call the CLI directly.

## Why this exists — the trap without it

Pipelines that use `image: "@anchor"` on their first shot often pass a
raw character *sheet* (neutral backdrop, canonical pose) as the LTX
ia2v starting frame. The shot prompt then asks for a completely
different scene ("dimly lit studio, sweat, mirror, overhead light").
LTX compromises — you get the character-sheet pose wearing the sheet's
neutral-backdrop lighting, with the new scene vaguely visible behind.
Looks wrong. Always.

The fix: **always flux2 i2i (or i2iN) the character sheet against the
scene prompt first** to produce a shot-specific anchor — character
*in* the setting, new lighting, appropriate pose — and feed *that*
into LTX. This skill owns the pattern.

## Anchor types

| Type | Inputs | Use when |
|---|---|---|
| `t2i` | prompt only | Environments, objects, no character required. Bad default for character scenes. |
| `i2i` | 1 reference + prompt | Most common. Place the character into a specific scene. |
| `i2i2` | 2 refs + prompt | Character meets another character. Composite scene. |
| `i2iN` | 3+ refs + prompt | Group scenes. ≥4 refs rapidly loses identity; cap at 3-4 in practice. |
| `angles` | 1 ref + N prompts | Character sheet — N poses of the same subject in one submission. Use for building the sheet, not per-shot. |

## CLI — `generate_anchor.py`

```
python3 /home/sandbox/.openclaw/skills/storyboard/scripts/generate_anchor.py \
    --out     <path>.png \
    --type    t2i|i2i|i2i2 \
    --prompt  "..." \
    [--reference  a.png]                   # i2i
    [--references a.png,b.png]             # i2i2 / i2iN
    [--subject    name=description] ...    # repeatable — {name} substituted in prompt
    [--style      editorial|indie|storybook|risograph]   # preset tail, or --style-tail "<custom>"
    [--width 1024] [--height 576] [--steps 8]
```

Idempotent — if `--out` already exists, skips. Subject tokens are
expanded before the prompt goes to flux2. Valid camera-LoRA names are
validated only when the caller also runs an LTX pass; this CLI just
makes images.

Examples:

```
# Single-character scene anchor
generate_anchor.py --type i2i \
    --reference character.png \
    --prompt  "{fennie} in a late-night dance studio, overhead spot, wooden floor, mirror behind her, sweat on her brow, paused mid-balance" \
    --subject "fennie=Fennie, 17yo Korean ballet trainee, black hair in a tight bun, black leotard, pale pink tights" \
    --style   indie \
    --out     shots/001-opening-anchor.png

# Two-character meeting
generate_anchor.py --type i2i2 \
    --references fennie.png,trainer.png \
    --prompt  "{fennie} kneeling on the studio floor; {trainer} kneeling beside her, hand on her shoulder, warm key light" \
    --subject "fennie=…" --subject "trainer=older man, trimmed grey beard, charcoal wrap cardigan" \
    --style   indie \
    --out     shots/003-coaching-anchor.png
```

## Python API — `lib/prompts.py`

Orchestrators can also import the prompt-preprocessing helpers directly:

```python
import sys
sys.path.insert(0, "/home/sandbox/.openclaw/skills/storyboard/lib")
from prompts import expand_subjects, apply_style_tail, STYLE_TAILS
```

- `expand_subjects(text, subjects: dict[str,str]) -> str` — `{name}` → value.
- `apply_style_tail(text, style_id_or_custom) -> str` — append a known-good style suffix.
- `STYLE_TAILS: dict[str,str]` — named tails (see `references/style-tails.md`).

## Multi-guide resolver — `lib/guides.py`

Shared resolver that turns a yaml-friendly `guides:` list into
`(frame_idx, absolute_path, strength)` tuples ready for
`comfy_graph.py multiguide --guides --frame_indices --strengths`. Both
music-video and drama-video use this so a yaml authored for one skill
reads the same in the other.

```python
import sys
sys.path.insert(0, "/home/sandbox/.openclaw/skills/storyboard/lib")
from guides import resolve_guides, ensure_frame_zero

resolved = resolve_guides(
    scene.get("guides"),              # the yaml list
    duration_sec=scene["duration_sec"],
    fps=24,
    total_frames=None,                # defaults to int(duration_sec * fps)
    project_dir=Path("./project"),    # relative image paths resolve against this
    token_resolver=lambda tok: ...,   # optional: resolve @anchor/@last → abs path
)
ensure_frame_zero(resolved, image_path=first_frame_path, strength=1.0)
# → [(0, "/abs/first.png", 1.0), (96, "/abs/mid.png", 1.0), ...]
```

Supported per-entry keys:

| key | type | notes |
|---|---|---|
| `image` | str | Absolute/relative path, `@anchor`, or `@last`. Relative paths resolve against `project_dir`. `@`-tokens are passed through `token_resolver` if provided. |
| `at_sec` | float | Absolute seconds from shot start. Takes precedence if multiple time keys are set. |
| `at_relative` | float (0..1) | Fraction of shot duration. |
| `at_frame` | int | Explicit LTX latent frame (auto-snapped to multiples of 8). |
| `strength` | float | Default 1.0. |
| `label` | str | Human-readable, ignored by the resolver. |

Frame indices are snapped to multiples of 8 (LTX's temporal latent
quantization) and clamped to `[0, ((total_frames-1)//8)*8]`. The output
list is sorted by `frame_idx` ascending. `ensure_frame_zero` prepends an
entry at `frame_idx=0` if none of the resolved guides targets frame 0 —
useful when the yaml-level list only describes mid-scene anchors and the
caller wants the first-frame anchor (e.g. a resolved `@anchor` PNG) to
be included automatically.

### When to add guides to a shot

- **First frame can't show the character** — e.g. scene opens on a prop,
  a landscape, an empty room. LTX will invent a random identity halfway
  through unless you anchor it with a mid-scene guide at strength 1.0.
- **Long shots (≥13s)** — single-anchor ia2v drifts by the tail of a
  long shot. Add a second anchor near the end (e.g. `at_relative: 0.9`)
  to keep the character consistent through the whole take.
- **Specific visual beats synced to lyrics/audio** — place an anchor at
  the exact frame where a line lands to time a character appearance to
  the word.

## When to use what

| Scenario | Pattern |
|---|---|
| Every shot of a character drama-video | `i2i` per shot against the character sheet + shot-specific prompt |
| Snakebird meeting another animal | `i2i2` with snakebird + other-animal sheets |
| Building the character sheet itself | `angles` with N pose prompts from one reference photo |
| First/last frame of a flf2v transition | Two `i2i` calls, one per scene, share the character ref |
| 12-character montage keyframes | 12 `i2i` calls (one per character), use each output as a multiguide frame |
| Empty establishing shot / title card | `t2i` |

## See also

- `references/prompting-guide.md` — the hard-learned rules: subject
  tokens, action emphasis, pose-neutral descriptions, common failure
  modes (lineups, Ghibli-ification, identity drift).
- `references/style-tails.md` — preset style suffixes by aesthetic.
- `references/camera-loras.md` — valid LTX camera LoRA names
  (relevant for downstream video, not for anchor generation).
- `comfyui` skill — the flux2 `t2i`/`i2i`/`i2i2`/`multiprompt` subcommands this CLI wraps.
