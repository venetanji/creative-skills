# FLUX.2 [klein] Prompting Guide

> Master narrative prompting for FLUX.2 [klein] — scene-first prose, lighting mastery, and multi-reference composition.

**⚠️ No prompt upsampling:** [klein] does NOT auto-enhance prompts. What you write is what you get — so be descriptive.

---

## Core Principle: Write Like a Novelist

Describe your scene as flowing prose — subject first, then setting, details, and lighting. Write sentences, not keyword lists.

| ✅ Do this | ❌ Not this |
|---|---|
| *"A weathered fisherman in his late sixties stands at the bow of a small wooden boat, wearing a salt-stained wool sweater, hands gripping frayed rope. Golden hour sunlight filters through morning mist."* | *"fisherman, old, boat, sweater, rope, golden hour, mist, portrait"* |

---

## Prompt Structure Framework

> **Subject → Setting → Details → Lighting → Atmosphere**

| Element | Purpose | Example |
|---|---|---|
| **Subject** | What the image is about | "A weathered fisherman in his late sixties" |
| **Setting** | Where the scene takes place | "stands at the bow of a small wooden boat" |
| **Details** | Specific visual elements | "wearing a salt-stained wool sweater, hands gripping frayed rope" |
| **Lighting** | How light shapes the scene | "golden hour sunlight filters through morning mist" |
| **Atmosphere** | Mood and emotional tone | "creating a sense of quiet determination and solitude" |

### Word Order Matters

[klein] pays more attention to what comes first. Front-load the most important elements.

**Strong:** Subject and action lead.
> *"An elderly woman with silver hair carefully arranges wildflowers in a ceramic vase. Soft afternoon light streams through lace curtains, casting delicate shadows across her focused expression."*

**Weak:** Subject buried in description.
> *"In a warm, nostalgic room with antique furniture, soft afternoon light streams through lace curtains. An elderly woman with silver hair is there arranging wildflowers."*

**Priority order:** Main subject → Key action → Style → Context → Secondary details

---

## Lighting: The Most Important Element

Lighting has the single greatest impact on [klein] output quality. Describe it like a photographer.

Describe all of these:

* **Source:** natural, artificial, ambient, window light
* **Quality:** soft, harsh, diffused, direct, overcast
* **Direction:** side, back, overhead, fill, camera-left, camera-right
* **Temperature:** warm, cool, golden, blue, amber
* **Interaction:** catches, filters through, reflects on surfaces, creates shadows

### Example Lighting Phrases

```
soft, diffused natural light filtering through sheer curtains
dramatic side lighting creating deep shadows and highlights
golden hour backlighting with lens flare
overcast light creating even, shadow-free illumination
soft sidelight from a large window camera-left, creating gentle shadows that define the subject's features
cold blue moonlight streaming through a frosted glass window
warm candlelight flickering across weathered stone walls
```

---

## Prompt Length

| Length | Words | Best For |
|---|---|---|
| **Short** | 10–30 | Quick concepts, style exploration |
| **Medium** | 30–80 | Most production work |
| **Long** | 80–300+ | Complex editorial, detailed product shots |

> ⚠️ Longer prompts work well **when every detail serves the image**. Avoid filler — each phrase should add visual information.

---

## Style and Mood Annotations

Add explicit style and mood descriptors at the end for consistent aesthetics:

```
[Scene description]. Style: Country chic meets luxury lifestyle editorial.
Mood: Serene, romantic, grounded.
```

```
[Scene description]. Shot on 35mm film (Kodak Portra 400) with shallow
depth of field — subject razor-sharp, background softly blurred.
```

### Style Examples

| Style | Effect |
|---|---|
| `Shot on 35mm film (Kodak Portra 400)` | Warm, nostalgic, film grain |
| `hyper-detailed fantasy illustration, artstation quality` | Rich, detailed, painterly |
| `cinematic volumetric lighting, atmospheric perspective` | Dramatic, immersive |
| `anime style, Studio Ghibli-inspired` | Whimsical, hand-painted feel |
| `fashion editorial, Vogue-style` | Polished, high-fashion |
| `oil painting, classical realism` | Rich textures, brushstroke feel |

---

## Image Editing (Flux2 Single/Double Image Edit)

For image editing, prompts describe the transformation you want. Reference images provide the foundation.

### Edit Patterns

| Edit Type | Pattern | Example |
|---|---|---|
| **Style transfer** | "Turn into [style]" | "Reskin this into a realistic mountain vista" |
| **Object swap** | "Replace [element] with [new element]" | "Replace the bike with a rearing black horse" |
| **Element replacement** | "Replace [element] with [new element]" | "Replace all the feathers with rose petals" |
| **Add elements** | "Add [element] to [location]" | "Add small goblins climbing the right wall" |
| **Environmental** | "Change [aspect] to [new state]" | "Change the season to winter" |
| **Age/modify** | "Age this portrait by 30 years" | "Change her dress from blue to deep burgundy" |

### Multi-Reference Editing

When using multiple reference images, specify the role of each:
> *"Change image 1 to match the style of image 2. Make the woman's hair just as fluffy."*
> *"Image of the Black Forest. Use the style from the reference images."*

---

## Effective vs. Ineffective Prompts

| ✅ Good prompts | ❌ Avoid |
|---|---|
| "Add dramatic storm clouds to the sky" | "Make it better" |
| "Change her dress from blue to deep burgundy" | "Improve the lighting" |
| "Age this portrait by 30 years" | "Make it more professional" |
| "Replace the background with a misty mountain range" | "Fix the image" |
| "Change image 1 to match the style of image 2" | "Add more detail" |

**Be specific about what changes and clear about the target state.**

---

## Flux2 [klein] Model Variants

| Variant | Speed | Best For |
|---|---|---|
| **klein 4B** | Sub-second | High-volume, local (~13GB VRAM) |
| **klein 9B** | Sub-second | Production, best prompt understanding |

---

## Quick Reference: Lighting Descriptors

```
Soft:     soft light, diffused light, gentle light, overcast
Hard:     harsh light, direct sunlight, spotlights, stark shadows
Direction: side lighting, backlight, front light, Rembrandt lighting
           three-quarter light, underlighting, rim light
Temperature: warm golden light, cool blue light, amber, warm afternoon
Weather: golden hour, blue hour, overcast flat light, storm light
Special:  volumetric light, god rays, lens flare, light leaks,
           bokeh, caustics, subsurface scattering
```

---

## Quick Reference: Atmosphere/Mood Descriptors

```
Emotional: serene, moody, melancholic, triumphant, mysterious, whimsical
Temporal:  nostalgic, timeless, futuristic, ancient
Weather:   misty, foggy, stormy, clear, hazy, dewy
Quality:   ethereal, grounded, raw, polished, gritty, dreamlike
Energy:    energetic, quiet, tense, peaceful, chaotic
```
