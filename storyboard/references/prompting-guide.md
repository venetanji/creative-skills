# Prompting Guide — Anchors & Keyframes

This is the hard-learned playbook for producing consistent, on-model
anchor frames and keyframe sequences. Every rule in here came from a
specific regression; ignoring a rule tends to reproduce the original
bug. Skim before writing prompts; come back before debugging output.

---

## 1. Subjects, not inline descriptions

**Don't** repeat a character's description in every shot. Flux2 at
8 steps has limited prompt attention; every repeat of "17yo ballet
trainee with tight bun…" is stealing attention from the *action* and
*setting* of the current shot.

**Do** define the character once as a `subject token` and reference it
by short name in each shot prompt. Let the preprocessor expand.

```yaml
subjects:
  fennie:  "Fennie — 17-year-old Korean ballet trainee with straight black hair pulled into a tight bun, lean build, wearing a simple black leotard and pale pink tights, barefoot"
  trainer: "an older man in his 60s, trim grey beard, charcoal wrap cardigan, grey sweatpants, kind eyes, retired dancer"

shots:
  - prompt: "{fennie} stands in the centre of the studio, exhaling slowly, preparing for another attempt"
  - prompt: "{trainer} enters from the left, kneels next to {fennie}, places one hand on her shoulder"
```

### Recursion footgun (learned the hard way)

Do NOT include `{fennie}` inside the description of `fennie` itself.
The expander is one-pass — self-reference becomes literal braces
showing up in the prompt → flux sees `{fennie}` verbatim and hallucinates
a brace pattern. Keep subject descriptions plain strings with no tokens.

---

## 2. Pose-neutral subject descriptions

The biggest anchor-drift trap: subject descriptions that prescribe a
*pose* lock flux to that pose regardless of the shot prompt.

**Bad:**
```
fennie: "… stands tall, feet in first position, arms at her sides …"
```
Every anchor will show her standing in first position — even if the
shot prompt asks for "kneeling", "mid-leap", "collapsed on the floor".

**Good:**
```
fennie: "… lean build, wearing black leotard and pale pink tights, barefoot …"
```
Appearance only. The shot prompt owns the pose.

Specifically strip trailing clauses like "stands on TWO legs like a
gentleman", "walks upright", "never on all fours" — these read to flux
as pose instructions, not anatomy facts.

---

## 3. Action emphasis for dynamic shots

When the character reference is a static sheet pose (usual case), flux
i2i tends to reproduce that pose and call it done. If you want visible
action, append this after the scene description:

> `DYNAMIC ACTION POSE captured mid-motion, NOT a character lineup, NOT posing for camera, NOT standing side-by-side.`

The `generate_anchor.py` CLI injects this automatically when
`--action-emphasis` is passed. Drama-video and music-video have a
shot-level `action_emphasis: true` flag that does the same. Use it on:

- Dancing / running / fighting shots
- Multi-character shots (without it, flux nearly always produces a lineup)
- Shots where the character is doing a verb, not standing

Don't use it on:

- Establishing / landscape / title shots (no character)
- Intentional portrait / still-frame shots

---

## 4. Reference count vs flexibility

| Refs | Pattern | What you get | Downside |
|---|---|---|---|
| 0 | `t2i` | Flux invents the whole scene from the prompt | Character drift — never use for recurring characters |
| 1 | `i2i` | Character + new scene. Most reliable. | Identity anchor loses some definition if scene prompt is long |
| 2 | `i2i2` | Two distinct characters composited. | Harder for flux to interpret; needs clearer prompt direction about who is where |
| 3-4 | `i2iN` | Group scenes. Works. | Identity dilutes: each additional ref halves attention budget |
| 5+ | `i2iN` | Technically possible | Quality falls off a cliff; just composite in post |

For a group scene of 5+ characters, don't try to jam 5 refs into one
i2iN call. Pick the hero + 1-2 key supporting characters, let the rest
be described in the prompt ("in the background a loose cluster of
other friends").

---

## 5. Style tails — pick one, stay consistent

A *style tail* is a short suffix appended to every prompt in a project
to lock the visual aesthetic. Named presets live in
`lib/prompts.py → STYLE_TAILS`. Current presets:

- `editorial` — Moebius / Wolfwalkers / Wes Anderson, inked, muted
  dusty rose/teal/ochre palette, risograph texture.
- `storybook` — Gouache children's book, rounded, warm palette.
- `indie` — Cinematic indie-film, natural light, 35mm grain, realistic
  skin tones.
- `risograph` — Limited 2-3 colour spot palette, halftone dots.
- `graphic_novel` — Bold inked, high-contrast, paneled framing.

**Rules:**
- Every anchor in a project uses the same style tail. Mixing tails
  across shots in the same edit is jarring.
- Don't stack style terms in the prompt itself — the tail carries the
  look; the prompt carries the scene.
- If none of the presets fit, pass `--style-tail "<custom string>"`
  and keep it identical across all shots. Or add a new named preset
  to `STYLE_TAILS` so the project spec just references it by id.

**Failure mode to avoid:** writing "whimsical storybook Ghibli" into
*every* prompt, then also adding a different style preset. Flux
compromises into a flat Disney-ish look that's neither. Pick one.

---

## 6. Negative prompts that actually matter

Flux2 uses `ConditioningZeroOut` on negative — so negative strings
have **limited** effect. Don't rely on them to fix problems.

Still worth including for common failures:
```
photorealistic, text, watermark, logo, caption, ugly, distorted face,
harsh lighting, cgi plastic, extra fingers, duplicate limbs,
children's book, rounded cute cartoon, chibi,
static pose, standing still, posing for camera, character-sheet pose,
lineup of characters, characters facing forward side by side
```

But the real fix for static poses / lineups is action emphasis (§3),
not longer negatives.

---

## 7. Writing good scene prompts

**Structure, in order:**

1. **Framing sentence** — wide / medium / close, from what angle, in
   what setting. One sentence.
2. **Action sentence** — what is happening, what the character is doing
   *right now*. Verbs. One sentence.
3. **Light/mood sentence** — optional, helps flux set the lighting budget.
   One sentence.
4. **Camera motion hint** — for downstream LTX video. One clause.

Skip bullet lists in the prompt itself — flux reads them as literal
bullets and draws… bullets. Comma-or-period-separated prose only.

**Good example** (drama-video shot):
> "Medium close shot of {fennie} kneeling on the wooden floor, head down,
> shoulders rising and falling from recovery breaths, sweat on her brow.
> A single overhead spot, rest of the studio in deep shadow.
> Slow dolly-in from medium to tight."

**Bad example** (what not to do):
> "A shot of Fennie. She is 17 years old and wears a black leotard.
> She is in a dance studio. It is late at night. She is tired. There
> is a single light overhead. The floor is wooden. She is kneeling.
> The camera is close. The mood is sad. Style: cinematic, realistic,
> natural light, grain, moody, beautiful, award-winning, 8k, HDR,
> trending on artstation."

The good example is shorter, specifies the *verb*, and trusts the
style tail + subject token to handle identity and aesthetic.

---

## 8. Keyframe sequences for transitions

Use-case: you want a smooth LTX transition from scene A to scene B
(flf2v). The sequence needs a pair of keyframes that belong to the
same visual world.

- Generate both keyframes via **the same anchor CLI invocation style**
  (same character ref, same style tail) — different scene prompts, same
  character subject tokens. That gives you two images that read as "the
  same character in two moments", ready for flf2v.
- Don't mix styles across a keyframe pair.
- If the transition is meant to be magical (scene morphs into scene),
  keep the character central and similar-pose; flux will interpret an
  aggressively different pose as a hard cut.
- If the transition is a hard cut, let both keyframes be whatever
  composition the new scene wants — flf2v + the transition LoRA will
  still blend, but the `zhuanchang` cut-feel takes over.

For multi-keyframe sequences (LTX `multiguide`), generate N anchors
with the same character ref + N shot prompts, then feed them into
LTX at evenly-spaced frame indices. See the snakebird fun-montage
precedent: 12 sheet portraits × 8 frames apart = a rapid-fire beat-synced
character reel.

---

## 9. Common failure modes — diagnostic checklist

If the anchor looks wrong, walk through this list:

| Symptom | Likely cause |
|---|---|
| Character looks like a different person | Reference image is low quality, subject description contradicts it, or prompt is too long (identity drowned out) |
| Character is in the right setting but wrong pose | Subject description prescribes a pose — strip the pose clause |
| All characters stand in a line | Missing action emphasis (§3). Add `--action-emphasis`. |
| Everything comes out "Ghibli-ified" / too cute | Style tail or prompt contains "whimsical storybook Ghibli rounded cute" — remove and pick a single tail |
| Background takes over, character vanishes | Too many scene details in the prompt, too few in the character ref — shorten the scene description |
| Character wearing a character-sheet neutral backdrop | Forgot to do i2i — pipeline is passing the raw sheet as the anchor. This is exactly the footgun storyboard exists to prevent. |
| Extra limbs / mutated hands | Standard flux issue — add those to negative; also try lower `--steps` (4 instead of 8 can be more stable) |
| Character "copies" the reference pose exactly | Action emphasis off, OR scene prompt lacks verbs, OR reference is too dominant a pose |

---

## 10. One last rule

If the output isn't right after three iterations of the same prompt,
**change the reference**, not the prompt. A well-lit three-quarter
reference usually fixes more than any prompt tweak. Regenerate the
character sheet (via `flux2 multiprompt` / `angles`) with better pose
prompts before spending more time on scene anchors.
