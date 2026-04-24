"""Shared guide-resolver for drama-video, music-video, and any skill that
drives LTX-V scene rendering with multiple keyframe guides.

Takes a list of per-shot guide specs (yaml-friendly dicts) and returns a
normalized list of `(frame_idx, absolute_image_path, strength)` tuples
that plug straight into `ltx2_multi_guide_to_video`'s CLI
(`--guides`, `--frame_indices`, `--strengths`).

Yaml schema:
    guides:
      - image: "@anchor"                     # or "@last", or a literal path
        at_sec: 0.0                          # pick ONE of at_sec / at_relative / at_frame
        strength: 1.0                        # optional, default 1.0
      - image: sheets/mid_anchors/hare.png
        at_relative: 0.5                     # 0.0-1.0 within shot duration
        strength: 1.0
      - image: sheets/mid_anchors/hare2.png
        at_frame: 96                         # explicit LTX latent frame (snapped to 8)

The resolver:
- Converts at_sec / at_relative into integer latent frame indices.
- Snaps every index to a multiple of 8 (LTX latent temporal quantization).
- Clamps to the valid range [0, ((total_frames - 1) // 8) * 8].
- Resolves `@anchor` / `@last` tokens when the caller passes matching
  paths (kept generic — resolver doesn't know about per-skill conventions).
- Sorts by frame_idx ascending.

The caller is responsible for prepending a frame-0 entry if none of
the specs targets frame 0 and the caller expects one.
"""
from __future__ import annotations
from pathlib import Path


def resolve_guides(guides_spec, *,
                    duration_sec: float,
                    fps: int,
                    total_frames: int | None = None,
                    project_dir: Path | str | None = None,
                    token_resolver=None):
    """Resolve a list of guide dicts into normalized tuples.

    Args:
        guides_spec: list of dicts. Each dict must have:
            - `image`: path string. May be "@anchor" / "@last" / literal.
            - Exactly ONE of `at_sec` (float), `at_relative` (0..1 float),
              or `at_frame` (int). Absolute latent-frame wins if multiple.
            - Optional `strength` (float, default 1.0).
            - Optional `label` (str, ignored; useful for human-reading yaml).
        duration_sec: shot duration (seconds). Sets the default
            total_frames when unspecified.
        fps: integer frames per second.
        total_frames: total latent frames in the render. If None, derived
            as int(duration_sec * fps).
        project_dir: resolved-against for relative image paths. If None,
            relative paths are returned as-is.
        token_resolver: optional callable `(token:str) -> str` used to
            resolve `@anchor` / `@last` tokens. If None, tokens are
            returned verbatim (caller must substitute downstream).

    Returns:
        list of (frame_idx: int, image_path: str, strength: float),
        sorted by frame_idx ascending.
    """
    if total_frames is None:
        total_frames = int(float(duration_sec) * float(fps))
    if total_frames < 1:
        raise ValueError(f"total_frames must be ≥1, got {total_frames}")
    max_valid_frame = ((int(total_frames) - 1) // 8) * 8

    pd = Path(project_dir).resolve() if project_dir else None
    out: list[tuple[int, str, float]] = []
    for g in guides_spec or []:
        img = g.get("image")
        if not img:
            raise ValueError(f"guide spec missing 'image': {g}")
        if img.startswith("@"):
            img = token_resolver(img) if token_resolver else img
        else:
            p = Path(img)
            if pd and not p.is_absolute():
                img = str((pd / p).resolve())
            else:
                img = str(p)

        if "at_frame" in g:
            raw_frame = int(g["at_frame"])
        elif "at_sec" in g:
            raw_frame = int(round(float(g["at_sec"]) * float(fps)))
        elif "at_relative" in g:
            raw_frame = int(round(float(g["at_relative"]) * float(total_frames)))
        else:
            raise ValueError(
                f"guide spec needs one of at_sec / at_relative / at_frame: {g}")

        frame_idx = (raw_frame // 8) * 8
        frame_idx = max(0, min(frame_idx, max_valid_frame))
        strength = float(g.get("strength", 1.0))
        out.append((frame_idx, img, strength))

    out.sort(key=lambda row: row[0])
    return out


def ensure_frame_zero(resolved, *, image_path: str, strength: float = 1.0):
    """Prepend a (0, image_path, strength) entry if none of `resolved`
    targets frame 0. Mutates + returns the list.

    Useful when a scene/shot has `image: "@anchor"` (which drama-video
    + music-video already resolve to an absolute PNG) and the caller
    wants that anchor as the first-frame guide — and the yaml-level
    `guides:` list focuses on mid-scene additions only."""
    if not any(f == 0 for f, _, _ in resolved):
        resolved.insert(0, (0, image_path, float(strength)))
    return resolved
