# Native MiniMax H3 prompting guide

Maintained 2026-10-02. Native local H3-Base FL2VA / Ref2VA, not Hailuo API,
Hunyuan3D or LTX. See [routing and maintenance](model-prompting-index.md).

## Main correction: author the official structured representation

MiniMax publishes separate base and full-reference guides. Follow their field
names, order, reference labels and timing notation; write natural English inside
the fields. Its hosted H3-Context-IR preprocessing is not in the open-source
release. Our local text encoder/Comfy execution does not automatically perform
that rewrite. Plain short prose may execute but is not the official final format.

## Official base family: T2VA and I2VA

T2VA starts directly with three fields in this exact order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

I2VA puts this exact instruction on the first line, followed by one blank line
before those same fields:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

I2VA establishes the supplied composition, identity, clothing, scene and spatial
relationships, then describes action onset, development and outcome. A raw selfie
wired to `first_frame` is the scene's actual start, not an identity-only ingredient.
Compose the scene anchor to the target aspect first; the native first-frame input
is stretched to the canvas. Do not use a portrait photo directly as a landscape
first frame and expect prompt text to correct its geometry/background.

### Authored base landscape example (untested)

```text
integrated_multimodal_description: [Shot 1] Cinematic, a low wide shot opens between pink salt pillars above a cobalt tidal basin. A shallow tide advances through wet mineral grooves while silver foam gathers at their edges. The camera trucks right with small amplitude at slow speed, revealing a pale arch through suspended mist. The pillars remain still as the water moves beneath them, and warm sidelight picks out their rough surfaces.

overall_soundscape: Wind whistles across the salt ridges while water laps against the basin edges. Small bubbles pop softly near the foreground.

non_diegetic_music: N/A
```

The official base guide separately specifies FL2VA and L2VA first/last alignment
sentences, including exact punctuation and a two-decimal effective duration.
Read the pinned original when authoring those modes; do not substitute I2VA's
line. The current local wrapper does not admit FLF or last-only generation.

## Official Ref2VA: semantic ingredients, six fields

Use REF weights and these sections in order:

```text
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

`<Picture 1>` / `<Picture 2>` correspond to actual image connection order,
with exact case and spaces. `<Subject N>` defines reusable visible content,
not an asset index; multiple assets can define one subject and vice versa.
A picture used only for identity, setting, clothing or style is cited inside
that subject's definition, not given a standalone composition/keyframe role.

### Identity plus landscape scaffold (not a complete final prompt)

```text
subject_definitions:
<Subject 1> is the man's facial identity in <Picture 1>, including his bald head, beard and facial structure; this source supplies identity only.
<Subject 2> is the landscape in <Picture 2>, supplying its terrain, horizon and light.

summary:
[reference generation] The target video places <Subject 1> in <Subject 2> in a continuous eye-level medium close-up.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - his defined facial identity is retained, with natural head-to-shoulder proportions in the new scene.
<Subject 2> (appears in [Shot 1]): partially_preserved - its terrain and light are retained behind the foreground subject, with framing adapted to the portrait.

detailed_description:
Cinematic natural textures and coherent landscape light establish the finished look. [Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
N/A
```

Expand `detailed_description` before submitting: the official generation guide
normally requests 350-500 English words, including for one shot. Dialogue-dense
or source-editing cases have documented exceptions. Establish style in one or
two sentences BEFORE `[Shot 1]`, then describe visible composition, subject scale,
placement, features, lighting, action, camera and sound in playback order. Do not
replace that detail with only a reference-relationship list. Retention markers
express desired relationships, not proof that a generated result preserved them.

Visual retention markers: `fully_preserved`, `partially_preserved`,
`attribute_transfer`, `weak_reference`. Upstream audio relationships use
`fully_copy`, `partially_copy`, `reference`, `weak_reference`; these do not create
unsupported local inputs or guarantee direct waveform copying.

## Shared camera, timing, dialogue and sound rules

- `[Shot 1]` has no timestamp. Later shots use `[Shot N] At MM:SS.mmm, ...`
  with increasing cut times inside the effective duration. Use ordinary prose
  to name cuts; prompted time is not a verified frame-accurate edit contract.
- Camera motion is an English action with movement type, meaningful amplitude
  and speed, not a pile of tags: a slow small truck-right differs from a pan.
- Only vocalizing sources receive stable `(S1)`, `(S2)` IDs. Dialogue/lyrics use
  `<d>[Language] original words</d>`; keep user language/content. Off-screen
  voiceover uses `says in an off-screen voiceover` and states lips remain closed.
- Visible text belongs in double quotation marks. Exact lettering must still
  be inspected across frames; do not claim guaranteed spelling.
- Action-linked sounds accompany their action in the main description.
  `overall_soundscape` uses 1-4 sentences of ambience/physical/nonverbal sound;
  do not repeat dialogue or singing there. `N/A` means explicitly requested
  complete silence, not merely no music.
- `non_diegetic_music` uses 1-3 concrete sentences of instrumentation, tempo and
  dynamics; use `N/A` for absent audience-only score. Diegetic music remains in
  the shot narrative. Audition and inspect synchronization separately.

Current local graphs use BasicGuider with no negative branch; the CLI rejects
negative/CFG. Prefer affirmative composition and precise source roles over
SD-style negative tags. None of these rules guarantees one-event physics,
identity, anatomy, background suppression or seamless chaining.

## Reference images, temporal guides, control and LoRAs are distinct

REF images are semantic inputs, not automatically output-frame anchors.
`MiniMaxH3AddGuide` anchors temporal still/clip/audio content; it is not ordinary
V2V motion imitation. AddGuide-only images do not acquire `<Picture N>` labels
unless additionally connected to REF image inputs. Native REF video/audio and
Fun ControlNet have separate contracts; plain WebGL RGB is not depth/Canny.
The original Alibaba Union card is now marked legacy and points to Union-2.0;
local 2.0 asset/graph compatibility is unverified. Never silently substitute it.

Turbo LoRAs are speed/sampling recipes, not a general artistic ingredient pack.
The selected local non-768p FL8 adapter uses shifts 12/3 and eight steps; the
maker's 768p FL4/FL8 variants use 6/3. Current REF20 is res_multistep/simple/20,
without LoRA/SigmaShift. Choose LoRA, schedule, shifts and family together.
H3 community style embeddings are not official MiniMax creative LoRAs. Do not
transfer LTX adapters to H3 or infer support from an asset name.

## Local scope and empirical lessons

[Local profile](../../comfy-graph/H3-PROFILE.md): source-only h3-t2v, FL8 h3-i2v,
and REF20 h3-r2v with 1-2 ordered images. Fixed 24 FPS; 124..362 frames on 17k+5,
not LTX's grid. The minimum 124 frames is 5.166667 seconds. Supplied video/audio,
FLF/continuation/control, arbitrary LoRAs and an image-strength knob are not
admitted by that wrapper. Upstream node capability is not shipped Easel support.

Oct2's twelve-clip study decoded successfully but used simpler prompts rather
than the newly retrieved official structured representation. REF imported selfie
background and missed one wide portrait; FL bell turns violated grounding. Gio
judged head proportions too small. Evidence: retained
`media/minimax-h3-review-batch-20261002/review/`. These are limitations of those
specific inputs/recipes/prompts, not proof that the new syntax fixes them.

Proposed craft: review identity crop, perspective and head-to-shoulder proportion;
keep an identity-only `<Subject>` separate from a landscape `<Subject>`. For
FL I2V, approve a proportion-correct scene-specific anchor before animation.
A crop reduces unwanted reference context but does not guarantee background
suppression. Single-image chaining uses the actual generated last frame and
is not multi-frame continuation or proof of a seamless boundary.

## Primary sources and precedence

Accessed 2026-10-02; exact archived hashes in [source records](model-prompting-sources.json).

- [MiniMax base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
- [MiniMax full-reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
- [MiniMax prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/SKILL.md): packaged guides match the pinned originals; not installed by this documentation work.
- [Comfy native integration](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native) and [Comfy prompt guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-prompt-guide).
- [Turbo maker](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/COMFYUI_SETUP_AND_INFERENCE.md).

Prefer MiniMax's guides for prompt grammar. The Turbo setup example differs on
I2VA instruction placement and dialogue in soundscape; use that maker for adapter
settings, not to overwrite model-maker syntax. Check actual admitted inputs and
recipe before adopting any rolling documentation's broader capabilities.
