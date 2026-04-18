# LTX-2.3 Prompt Cheatsheet (ComfyUI)

Condensed prompt-writing reference for the LTX-2.3 video model. LTX-2.3's redesigned text connector is more sensitive to prompt detail than prior versions — specific, cinematic, long-form prompts significantly outperform short ones.

## Quick reference

1. **Length must match duration.** Short prompts + long videos = model rushes or repeats. For an 8–10s clip, write a dense paragraph covering subject, action, environment, lighting, camera, and audio.
2. **Write one flowing paragraph in present tense.** Not a bullet list, not a JSON blob, not numerical specs. Natural language describing the shot like a cinematographer's brief.
3. **Show emotion physically, never as labels.** Replace "sad" with "his eyes lower, he exhales slowly, his jaw tightens." The model interprets physical cues; it ignores or flattens internal states.
4. **Break dialogue into short phrases with acting directions between them.** Put speech in quotes. Insert pauses, micro-expressions, and voice qualities between lines: `"I remember..." He pauses, looks to the side, then continues with a cracking voice, "...something I never understood."`
5. **Describe audio explicitly.** Ambient sound, room tone, voice texture, music — 2.3's audio is high quality and responds to direction. Include accent/language if relevant.

## Shot and camera vocabulary

**Shot scale / framing**
- Wide establishing shot, medium shot, close-up, extreme close-up, over-the-shoulder, overhead / top-down, low angle, high angle, macro lens, shallow depth of field.

**Camera movement** (describe *how* the camera moves *relative to the subject*, and *what becomes visible* after the move):
- Slow dolly in / pull back
- Tracking shot / follows / handheld tracking
- Pan left / pan right / tilt up / tilt down
- Circles around (orbit)
- Push in, zoom in slowly
- Static frame
- Crane up, whip pan

**Pacing / temporal**
- Slow motion, time-lapse, lingering shot, continuous shot, freeze-frame, seamless transition, rapid cuts, sudden stop, fade-in / fade-out.

**Tip:** Say when the camera moves and what the move reveals: "the camera pans right, slowly revealing a construction site surrounded by workers in hard hats."

## Style and mood vocabulary

**Genre categories**
- *Animation:* stop-motion, 2D / 3D animation, claymation, hand-drawn.
- *Stylized:* comic book, cyberpunk, 8-bit pixel, surreal, minimalist, painterly, illustrated.
- *Cinematic:* period drama, film noir, fantasy, epic space opera, thriller, modern romance, arthouse, documentary, experimental.

**Lighting**
- Golden hour, backlight, rim light, neon glow, flickering candles, flickering lamps, natural sunlight, dramatic shadows, soft diffused light.

**Atmosphere**
- Fog, mist, rain, dust, smoke, particles, reflections on wet pavement, lens flares, film grain.

**Color palette**
- Vibrant, muted, monochromatic, high contrast, warm / cool, teal-and-orange.

**Textures**
- Rough stone, smooth metal, worn fabric, glossy surfaces.

**Scale feel**
- Expansive, epic, intimate, claustrophobic.

**Audio / voice**
- Ambient: coffeeshop noise, wind and rain, forest ambience with birds, faint room tone.
- Voice style: energetic announcer, resonant voice with gravitas, distorted radio-style, robotic monotone, childlike curiosity.
- Volume: whisper, mutter, shout, scream.

## What to avoid

| Don't | Why |
|---|---|
| Emotional labels ("sad", "confused", "happy") | The model renders physical cues, not inner states. Describe the face and body. |
| Readable text or logos in frame | Text rendering is not reliable. |
| Overloaded scenes (many characters, many actions) | Reduces clarity; the model loses the throughline. |
| Complex chaotic physics (e.g. shattering, fluid splashes) | Introduces artifacts. Dancing and organic motion are OK. |
| Conflicting lighting ("harsh midday sun in a candlelit cave") | Mixed light logic confuses the scene. |
| Numerical over-specification ("3 birds at 45°, pan at 2°/sec") | Natural language outperforms numbers. |
| Contradictions ("still peaceful lake with crashing waves") | Be internally consistent. |
| Vague one-liners ("a nice nature video") | Model picks arbitrarily; be specific about frame contents. |
| Mismatched prompt/duration | 10-word prompt for a 10s shot leaves the model under-directed. |
| Ornamental flourishes early | Start simple, layer complexity. Don't stack every technique at once. |

## Worked examples — bad prompt rewrites

### Example A — vague one-liner

**Before:**
> A person walking in the city at night.

**After:**
> A young woman in a red wool coat walks briskly along a rain-soaked Tokyo street at night, her breath faintly visible. Neon signs in magenta and cyan reflect on the wet pavement beneath her boots. The handheld camera follows from slightly behind and to her left, bobbing gently with each step, shallow depth of field keeping her sharp while distant pedestrians blur into colored light. The ambient audio carries the hiss of tires on wet asphalt, a distant train horn, and the muffled hum of a crowded izakaya as she passes its sliding door.

### Example B — emotion labels instead of physical cues

**Before:**
> An old man feels sad and says he misses his wife.

**After:**
> A medium close-up of a man in his late sixties seated by a rain-streaked window, soft grey daylight across his face. He speaks slowly, voice low and a little hoarse, "I still set two cups out in the morning." He pauses, his gaze drifting to the empty chair opposite him, and his jaw tightens briefly. He continues, almost a whisper, "Habit, I suppose." The camera pushes in very slowly as he exhales. Audio is intimate and dry — faint rain on glass, the tick of a wall clock, no music.

### Example C — over-constrained numerical prompt

**Before:**
> Exactly 3 birds flying left to right at 45 degrees while the camera pans right at 2 degrees per second, golden hour, f/2.8.

**After:**
> A small flock of birds glides across a golden-hour sky, silhouetted against warm amber clouds. The camera pans smoothly to the right, keeping pace with them as they cross the frame. Shallow depth of field softens the distant ridgeline below. The only sound is wind and faint, distant birdcalls.

### Example D — short prompt for a long video

**Before (for a 10-second generation):**
> A chef cooks pasta.

**After:**
> Interior, a small trattoria kitchen lit by warm tungsten lamps and a single window of late-afternoon sun. A bearded chef in a flour-dusted apron tosses a pan of garlic and olive oil over a gas flame; the flame flares briefly blue-orange as the oil catches. He adds a handful of spaghetti to a pot of salted boiling water, steam rising past his face. The camera tracks slowly around the counter from his right side to a frontal medium shot as he cracks black pepper into the pan with a practiced flick of the wrist. He tastes the sauce from a wooden spoon, nods once to himself, then plates the pasta with a quick twirl of tongs and a drizzle of oil. The audio is close and textured — the sizzle of the pan, the knock of the spoon on ceramic, faint background chatter from the dining room, no music.

### Example E — dialogue without acting direction

**Before:**
> A woman says "I can't believe you're leaving. Please don't go. I'll miss you so much."

**After:**
> A close-up of a woman in her thirties in a dimly lit hallway, warm practical light spilling from an open doorway behind her. She speaks haltingly, voice unsteady, "I can't believe you're leaving." She looks down at her hands, then back up, blinking quickly. "Please..." A small beat, her lips press together. "...don't go." Her voice cracks on the last line, almost a whisper, "I'll miss you so much." The camera holds static on her face. Audio is intimate with faint room tone and the distant hum of a refrigerator.

## Prompt structure checklist

A strong LTX-2.3 prompt covers, in a single paragraph:

1. **Shot** — scale, lens, angle ("medium close-up, shallow depth of field, low angle")
2. **Scene** — location, lighting, color palette, atmosphere
3. **Character(s)** — age, hair, clothing, distinguishing features
4. **Action** — a natural sequence from beginning to end, present tense
5. **Camera movement** — how it moves, what it reveals
6. **Audio** — ambient sound, music, voice texture, dialogue in quotes

## Mode-specific notes

- **Text-to-video (t2v):** Include everything — you're generating from scratch.
- **Image-to-video (i2v):** Don't re-describe what's visible in the input image. Focus on motion, transition from stillness, camera behavior, and sound.
- **Image+Audio-to-video (ia2v):** Audio anchors the timing. Use the prompt to describe the visual interpretation that should accompany the soundtrack.
- **First-Last-Frame-to-video (flf2v):** Describe the motion that connects the two frames (e.g. "the camera moves from a high position to a low position, keeping the subject centered"). Don't re-describe the anchor frames themselves.

---

Source: https://ltx.io/model/model-blog/ltx-2-3-prompt-guide (LTX Blog, April 2026)
