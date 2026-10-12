# LTX-2.5 prompting guide

Maintained 2026-10-02 from official Lightricks guidance. Applies to LTX-2.5,
not a relabeled LTX-2.3 cheatsheet. See [routing and maintenance](model-prompting-index.md).

## Official guidance: text-to-video

For a simple continuous shot, use a focused flowing paragraph in present tense,
usually 4-8 descriptive sentences. Match length to complexity, not an arbitrary
word count or duration alone. Establish shot scale, coherent scene/light/materials,
action, identifying character details, camera and audio. Give each sentence
something concrete to show or hear; use physical expressions rather than only
emotional labels. Separate camera movement from subject movement, and describe
what becomes visible after the move. Keep one coherent lighting logic per shot.

A landscape needs action too: water advances, reeds bow, fog lifts, shadows move.
Describe one legible transformation and camera trajectory rather than stacking
unrelated spectacles. Avoid crowded subjects, contradictory motion and keyword
soup. Quote dialogue, specify voice/language/accent, and put pauses into the action.

### Authored landscape example (not a tested result)

A low wide shot opens between pink salt pillars above a cobalt tidal basin.
Wet mineral ridges catch the warm sidelight while distant arches remain pale
through suspended mist. A shallow tide advances along the grooves, carrying a
thin line of silver foam toward the foreground. The camera tracks slowly forward
between the pillars, revealing a larger arch beyond them without changing height.
Wind whistles softly across the ridges and water laps against the salt. The shot
remains one continuous take with no musical score.

## Official guidance: I2V, endpoints and audio

The image supplies appearance and composition; prompt what should happen next:
action, camera behavior, sound. Prefer one continuous take unless intentionally
cutting away. Match image aspect to the output. Repair anatomy/proportions in the
scene-specific anchor rather than asking video text to contradict the still.
First/last-frame prose describes the connecting motion, not two inventories.
Official FLF and I2V templates differ from one another and from our curated
builder; do not copy their stages/strengths without checking the actual graph.

With supplied audio, describe its visual interpretation. Official A2V freezes
conditioning in both stages and saves the original trimmed audio; post-generation
soundtrack muxing is not proof of conditioning. Exact source offsets, retained
frame spans and final sync belong to the workflow/assembly contract. Generated
audio and quoted dialogue still need audition and visual synchronization review.

LTX-2.5 supports prompting multiple shots: prefer 2-4, in chronological prose.
Name every cut, re-establish framing/identity and state audio continuity. Do not
assume prose cut times are frame-accurate or a shipped multishot timeline feature.
The official optional duration predictor is not in our curated route. That route
uses explicit frames/FPS and LTX's 8k+1 grid. Its optional Gemma4 E2B prompt enhancer
is a separate rewrite pass, not the Gemma4 text encoder; our curated route has no
enhancer. Do not assume deliberate text was automatically rewritten.

## Negatives and adapter-specific exceptions

Distilled local sampling uses CFG1/1. An encoded negative field is not evidence
of useful negative steering; do not promise it fixes anatomy or raise CFG to make
it work. Prefer a correct anchor and affirmative framing/action. Photoreal
negative presets that exclude cartoons are not universal aesthetic requirements.

| Workflow | Prompt/reference rule | Evidence boundary |
| --- | --- | --- |
| Union depth/Canny | Prompt finished appearance; actual registered video guides provide structure/motion. Depth suits terrain/spatial layers; Canny suits crisp architecture. | The named 2.3 Union asset has specific successful local 2.5 prototype evidence, not blanket cross-version compatibility. |
| Ingredients | Use the selected card's reference-sheet instructions, not extra ordinary I2V images. Older official card uses `Reference sheet:` / `Generated video:`. Important panels should be prominent and uncrowded. | The retrieved 2.3 card is inherited advice; a separately installed 2.5 Ingredients adapter has local execution evidence. Do not transplant 2.3 dev CFG4/30 steps into 2.5 distilled 8+3. |
| Layout-to-Render | Official 2.5 card asks for one or two sentences describing the finished shot, matching art-directed first/keyframes; avoid describing the output as clay/3D/Unreal. | Grey-clay/blocky playblast is structural input, not the finished look anchor. Local asset/workflow/fidelity not validated in this research. |
| Cinemagraph | Initial image plus the selected adapter's `CINEMAGRAPH_MOTION` trigger and defined moving region. | Local 2.5 adapter execution exists; selective-motion fidelity still requires review. |
| Camera / Slow Motion | Follow the exact compatible card and conditioning/playback FPS recipe. | File names or loader presence alone are insufficient. No generic camera/slow-motion flag is promised for the curated profile. |

Standard LoRAs modify model behavior; IC-LoRAs also require actual conditioning.
Check base version, loader, reference factor, trigger, both-pass model/guide wiring,
guide cropping, sampler, timing and quantization. Test compatible adapters singly
before stacking. Do not treat adapter strength as interchangeable with image
strength or control-video strength.

## Local scope and empirical observations

The curated split-loader profile admits t2v, i2v, ia2v, continuation and a narrow
single-image-endpoint flf2v route. It rejects arbitrary LoRA/IC/Ingredients/control-
video extras. These have retained explicit prototype graphs or separate consumer
execution evidence, not a newly working `--control-video` flag. Read
[profile contracts](../../comfy-graph/LTX-PROFILES.md) and
[WebGL guide evidence](webgl-video-guidance.md) for exact admission and limitations.
The profile evidence table is dated; later receipts must be checked separately.

Our local stitching policy is actual prior native last frame at strength 1.0 in
both applicable passes, not a general override of official I2V appearance defaults.
Correct source pixels do not certify a seamless generated boundary. Spatial
control trials transferred useful geometry but simplified topology; stronger
control was not universally better. Evidence: retained Sept30 video-guide archive.

Gio judged the Oct2 portrait study's head too small. Distinguish a face small in
frame from an anatomically undersized head: closer framing fixes the former,
anchor proportion repair fixes the latter. Review normal perspective, head-to-
shoulder scale and modest headroom before animating. A face crop/medium close-up
with a landscape behind it is a proposed craft treatment, not official numeric
proportion control or a fidelity guarantee. Keep identity proof separate from
a tiny-person landscape establishing shot when their framing conflicts.

## Local adapter provenance and unresolved card access

The local facts above are traceable through the `local_evidence` entries in
[source records](model-prompting-sources.json): the pinned adapter catalog,
download verification, and Oct2 readonly readiness receipt. Cinemagraph's
trigger is a retained catalog/consumer contract, not inferred from its name.
The readiness receipt binds installed 2.5 Ingredients to its maker repository,
revision, verified download record, current size/header and target-shape audit.
Its pinned maker card returned HTTP 401 during research; the exact current
2.5 prompt instructions were not retrieved. The older official 2.3 sheet syntax
is inherited guidance, NOT newly verified 2.5 maker instructions. Before a new
2.5 Ingredients experiment, inspect its exact retained working graph/contract
and available pinned instructions; do not transplant the older sampler recipe.

## Official sources

Accessed 2026-10-02; these are rolling pages unless separately revision-pinned.
Exact snapshot hashes and additional model cards are in [source records](model-prompting-sources.json).

- [Lightricks prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide.md): shot/action/camera/audio, single-shot and 2.5 multishot guidance.
- [2.5-specific blog](https://ltx.io/blog/ltx-2-5-prompt-guide) and [model card](https://huggingface.co/Lightricks/LTX-2.5).
- [I2V](https://docs.ltx.io/open-source-model/usage-guides/image-to-video.md) and [A2V](https://docs.ltx.io/open-source-model/usage-guides/audio-to-video.md).
- [LoRAs](https://docs.ltx.io/open-source-model/usage-guides/lo-ra.md), [adapter index](https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters.md), and [Union control](https://docs.ltx.io/open-source-model/feature-guides/structural-control/union-control.md).
- [Inherited 2.3 Ingredients card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients): not a 2.5 sampling recipe.
- [2.5 Layout-to-Render](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render): task-specific short-prompt exception.
