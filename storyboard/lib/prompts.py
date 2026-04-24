"""Shared prompt-preprocessing helpers used by `generate_anchor.py` and
imported by music-video / drama-video orchestrators that want to keep
their call sites out of the inline prompt logic.

Tiny, no runtime deps. Orchestrators inject into sys.path and import."""
from __future__ import annotations
import re

# ── named style tails ───────────────────────────────────────────────
# Each tail is appended to the end of a prompt. Keep them dense — flux's
# attention at 8 steps gets a few hundred chars; more is wasted.

STYLE_TAILS: dict[str, str] = {
    # Moebius / Wolfwalkers / Wes Anderson — inked editorial comic look.
    # Used on the snakebird project.
    "editorial":
        " Adult editorial illustration × Moebius × Joann Sfar × Cartoon Saloon Wolfwalkers × Wes Anderson, "
        "inked line work over flat gouache washes, muted dusty rose teal ochre plum graphite palette, "
        "risograph texture, film-grain haze, not children's book, not rounded cute cartoon, no text, no watermark.",

    # Soft storybook / children's-book gouache. Wrong for gritty scenes;
    # right for warm, whimsical ones.
    "storybook":
        " Gouache children's book illustration, soft rounded shapes, warm limited palette, "
        "painterly brush texture, whimsical storybook mood, no text, no watermark.",

    # Cinematic live-action indie-film aesthetic. Used for drama-video
    # projects like fennie-shine.
    "indie":
        " Cinematic indie-film aesthetic, natural light, 35mm grain, shallow depth of field, "
        "realistic skin tones, muted colour grade, no text, no watermark.",

    # Risograph-heavy poster/editorial.
    "risograph":
        " Risograph print aesthetic, limited 2-3 colour spot palette, paper-grain texture, "
        "halftone dots, inked linework, mid-century editorial composition, no text, no watermark.",

    # High-contrast graphic novel.
    "graphic_novel":
        " Graphic novel aesthetic, bold high-contrast inked lines, flat limited palette, "
        "dramatic lighting, paneled framing, no text, no watermark.",
}


# ── subject token expansion ─────────────────────────────────────────
# Subjects let callers keep one canonical description per recurring
# character and reference them by short token in prompts. Pattern:
#     expand_subjects("The {fennie} stretches", {"fennie": "17yo ballet trainee..."})

def expand_subjects(text: str, subjects: dict[str, str]) -> str:
    """Replace every `{name}` in `text` with `subjects[name]`. Unknown
    tokens pass through untouched so literal curly braces don't break.
    Non-recursive — a subject description that itself contains `{foo}`
    is NOT expanded further (prevents accidental recursion like the
    snakebird yaml regression we hit)."""
    if not isinstance(text, str) or not subjects:
        return text
    out = text
    for name, desc in subjects.items():
        out = out.replace("{" + str(name) + "}", str(desc))
    return out


# ── style tail ──────────────────────────────────────────────────────

def apply_style_tail(text: str, style: str | None) -> str:
    """Append a named style tail from STYLE_TAILS, or an arbitrary string
    as-is. `None` or empty → no-op."""
    if not style:
        return text
    tail = STYLE_TAILS.get(style, style)
    # Strip trailing whitespace / period on the prompt so the tail reads
    # cleanly. Leave the user's punctuation alone otherwise.
    base = text.rstrip().rstrip(".")
    return base + "." + tail if not tail.startswith(".") else base + tail


# ── action emphasis ─────────────────────────────────────────────────
# When the character reference is a static sheet pose, flux2 i2i tends
# to lock onto the sheet's pose and ignore the shot prompt's verbs.
# Injecting an explicit anti-lineup, anti-static phrase after the main
# prompt (but before the style tail) pushes flux to produce dynamic
# composition. Empirically measured on the snakebird project.

ACTION_EMPHASIS = (
    " DYNAMIC ACTION POSE captured mid-motion, "
    "NOT a character lineup, NOT posing for camera, NOT standing side-by-side."
)


def inject_action_emphasis(text: str) -> str:
    """Append the DYNAMIC-ACTION phrase if it's not already present."""
    if not text:
        return text
    if "DYNAMIC ACTION POSE" in text:
        return text
    return text.rstrip().rstrip(".") + "." + ACTION_EMPHASIS


# ── camera LoRA names (LTX, not flux — informational here) ──────────

VALID_CAMERA_LORAS = {
    "dolly-in", "dolly-out", "dolly-left", "dolly-right",
    "jib-up", "jib-down",
    "static",
}


def validate_camera_lora(name: str | None) -> str | None:
    """Raise ValueError if `name` isn't a recognised LTX camera LoRA
    shortname and doesn't look like a full .safetensors filename.
    `None` / empty pass through (no camera LoRA)."""
    if not name:
        return None
    if name in VALID_CAMERA_LORAS:
        return name
    if name.endswith(".safetensors"):
        return name
    raise ValueError(
        f"camera_lora {name!r} is not a valid LTX shortname. "
        f"Known shortnames: {sorted(VALID_CAMERA_LORAS)}. "
        f"Pass a full .safetensors filename for anything else."
    )


# ── tiny helper: flatten a prompt through the full pipeline ─────────

def render_prompt(text: str,
                   subjects: dict[str, str] | None = None,
                   style: str | None = None,
                   action_emphasis: bool = False) -> str:
    """Convenience: expand subjects → (optionally inject action) → style tail."""
    out = expand_subjects(text, subjects or {})
    if action_emphasis:
        out = inject_action_emphasis(out)
    out = apply_style_tail(out, style)
    return out
