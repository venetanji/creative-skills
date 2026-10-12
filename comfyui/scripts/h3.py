"""Strict native H3 CLI adapter; graph knowledge lives in the pure package."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import warnings
from string import Formatter

import core
from creative_comfy_graph.minimax_h3 import build_t2v, build_i2v, build_r2v, build_ia2v


def _cli_integer(value: str) -> int:
    if re.fullmatch(r"[+-]?[0-9]+", value) is None:
        raise argparse.ArgumentTypeError("must be an integer, not a float or boolean")
    return int(value)


def _validate_caption(template: str, prompt: str) -> None:
    # Check syntax and allocation bounds before any shared transport can run.
    if len(template) > 4096:
        raise ValueError("caption_template must be at most 4096 characters")
    formatter = Formatter()
    values = dict.fromkeys(("filename", "workflow", "id", "ts"), "")
    values["prompt"] = prompt[:100].replace("\n", " ")
    fields = expansion = 0
    try:
        for _, field, spec, conversion in formatter.parse(template):
            if field is None:
                continue
            if field not in values:
                raise ValueError("only filename, workflow, id, ts and prompt fields are allowed")
            if conversion not in (None, "s", "r", "a"):
                raise ValueError("only !s, !r and !a conversions are allowed")
            match = re.fullmatch(
                r"(?:[^{}]?[<^>])?(?P<width>[0-9]{1,4})?(?:\.(?P<precision>[0-9]{1,4}))?s?", spec)
            if match is None:
                raise ValueError("format must be a literal string alignment/width/precision spec")
            width, precision = (int(match[name] or 0) for name in ("width", "precision"))
            fields += 1
            expansion += max(width, precision)
            if fields > 32 or max(width, precision) > 1024 or expansion > 16384:
                raise ValueError("at most 32 fields, width/precision <=1024 and total format budget <=16384")
            # Transport supplies strings (including empty workflow/id), not numbers.
            value = formatter.convert_field(values[field], conversion) if conversion else values[field]
            formatter.format_field(value, spec)
    except ValueError as exc:
        raise ValueError(f"caption_template: {exc}") from exc


def _parse(args: list[str], mode: str):
    parser = argparse.ArgumentParser(
        prog="comfy_graph.py " + mode, allow_abbrev=False,
        description="Experimental native MiniMax H3 " + ("REF twenty-step semantic references" if mode == "h3-r2v" else "FL eight-step text/first-frame") + (" + supplied-audio conditioning" if mode == "h3-ia2v" else " + generated stereo AV") + " (offline-validated only).",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--input-json", "-j", help="JSON object; CLI flags override its values")
    parser.add_argument("--prompt", help="visual action and intended sound in one text prompt")
    for name, help_text in (
        ("width", "multiple of 32, 32..16384 (default 864; not GPU certification)"),
        ("height", "multiple of 32, 32..16384 (default 480; not GPU certification)"),
        ("frames", "explicit 17k+5 grid, 124..362 (default 124); native fps is fixed 24"),
        ("seed", "uint64 integer, including zero (default 0)"),
        ("timeout", "positive submission/history deadline in seconds (default 600)"),
    ):
        parser.add_argument("--" + name, type=_cli_integer, help=help_text)
    for name in ("prefix", "output-dir", "notify-target", "caption-template"):
        parser.add_argument("--" + name)
    if mode in ("h3-i2v", "h3-ia2v"):
        parser.add_argument("--image", help="one local single-frame PNG/JPEG first-frame anchor")
        if mode == "h3-ia2v":
            parser.add_argument("--audio", help="local WAV/MP3/FLAC/Ogg/M4A soundtrack covering frames/24 seconds")
    elif mode == "h3-r2v":
        parser.add_argument("--references", nargs="+", help="1..2 local single-frame PNG/JPEG images in <Picture 1>, <Picture 2> order")
        parser.add_argument("--ref-image-size", help="match (default) or max; max can be much slower")
    cli = vars(parser.parse_args(args))
    path = cli.pop("input_json", None)
    opts = {}
    if path is not None:
        try:
            spec = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise ValueError("input JSON must be an object")
            allowed = {action.dest for action in parser._actions} - {"help", "input_json"}
            for key, value in spec.items():
                normalized = key.replace("-", "_")
                if normalized not in allowed or normalized in opts:
                    raise ValueError(f"unknown or duplicate input JSON option: {key}")
                opts[normalized] = value
        except (OSError, ValueError, UnicodeError) as exc:
            parser.error(str(exc))
    opts.update(cli)
    return parser, opts


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_IMAGE_EDGE = 8192


def _admit_images(paths: list[str]) -> list[tuple[str, bytes]]:
    # Import only for image-bearing admission; text/help and pure graphs stay
    # dependency-free. Decode every file before allowing ANY upload.
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Pillow is required for H3 image admission; install it in the executing environment") from exc
    admitted = []
    for index, value in enumerate(paths):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("image paths must be nonempty local strings without NUL")
        path = Path(value)
        ext = path.suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg") or not path.is_file():
            raise ValueError("each image must be an existing local PNG/JPEG file")
        try:
            # Bounded read, even if the file grows after stat. These admitted
            # bytes are staged later so uploads cannot reread changed sources.
            with path.open("rb") as stream:
                data = stream.read(MAX_IMAGE_BYTES + 1)
            if not data or len(data) > MAX_IMAGE_BYTES:
                raise ValueError("image must be nonempty and at most 32 MiB")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    expected = "PNG" if ext == ".png" else "JPEG"
                    if image.format != expected or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("image must be single-frame decoded PNG/JPEG matching its extension")
                    w, h = image.size
                    if not (1 <= w <= MAX_IMAGE_EDGE and 1 <= h <= MAX_IMAGE_EDGE) or w * h > MAX_IMAGE_PIXELS:
                        raise ValueError("image dimensions must be <=8192 per edge and <=16 Mi pixels")
                    image.load()
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError(f"invalid local image {path}: {exc}") from exc
        name = f"h3_{index}_{hashlib.sha256(data).hexdigest()}" + (".png" if ext == ".png" else ".jpg")
        admitted.append((name, data))
    return admitted


def main(args: list[str], *, base_url: str, dump: bool = False, mode: str = "h3-t2v") -> None:
    parser, opts = _parse(args, mode)
    try:
        timeout = opts.get("timeout", 600)
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        for name in ("output_dir", "notify_target", "caption_template"):
            if name in opts and (not isinstance(opts[name], str) or not opts[name] or "\x00" in opts[name]):
                raise ValueError(f"{name} must be a nonempty string without NUL")
        common = dict(width=opts.get("width", 864), height=opts.get("height", 480),
                      frames=opts.get("frames", 124), seed=opts.get("seed", 0),
                      filename_prefix=opts.get("prefix", "video/minimax_h3"))
        prompt = opts.get("prompt", "")
        # Shared options and caption must pass before even decoding local media.
        build_t2v(prompt, **common)
        if "caption_template" in opts:
            _validate_caption(opts["caption_template"], prompt)
        admitted = []
        if mode in ("h3-i2v", "h3-ia2v"):
            admitted = _admit_images([opts.get("image")])
            if mode == "h3-ia2v":
                from audio_input import admit_audio
                audio = admit_audio(opts.get("audio"), frames=common['frames'])
                graph = build_ia2v(prompt, image_filename=admitted[0][0], audio_filename=audio[0], **common)
            else:
                graph = build_i2v(prompt, image_filename=admitted[0][0], **common)
        elif mode == "h3-r2v":
            paths = opts.get("references")
            if not isinstance(paths, list) or not 1 <= len(paths) <= 2:
                raise ValueError("references must be a list of 1..2 ordered local image paths")
            size = opts.get("ref_image_size", "match")
            # Check size/options before decoding; build again with stable names.
            build_r2v(prompt, reference_filenames=["admission.png"], ref_image_size=size, **common)
            admitted = _admit_images(paths)
            graph = build_r2v(prompt, reference_filenames=[name for name, _ in admitted],
                              ref_image_size=size, **common)
        else:
            graph = build_t2v(prompt, **common)
    except ValueError as exc:
        parser.error(str(exc))
    if dump:
        print(json.dumps(graph, indent=2))
        return
    core.BASE = base_url
    if admitted:
        try:
            with tempfile.TemporaryDirectory(prefix="h3-admitted-") as staging:
                for index, (name, data) in enumerate(admitted):
                    local = Path(staging) / name
                    local.write_bytes(data)
                    remote = core.upload_image(str(local), upload_as=name, strict_input=True)
                    # Server may suffix a collision, never provide arbitrary
                    # paths/annotated output names to LoadImage or the caller.
                    stem, ext = name.rsplit(".", 1)
                    if not isinstance(remote, str) or len(remote) > 200 or re.fullmatch(re.escape(stem) + r"(?:_[0-9]+)?\." + ext, remote) is None:
                        raise ValueError("backend returned an unexpected managed image name")
                    graph[str(16 + index)]["inputs"]["image"] = remote
                if mode == "h3-ia2v":
                    local = Path(staging) / audio[0]
                    local.write_bytes(audio[1])
                    remote = core.upload_image(str(local), upload_as=audio[0], strict_input=True)
                    if not isinstance(remote, str) or re.fullmatch(r'h3_audio_[a-f0-9]{32}(?:_[0-9]+)?\.wav', remote) is None:
                        raise ValueError('backend returned an unexpected managed audio name')
                    graph['17']['inputs']['audio'] = remote
        except Exception as exc:
            raise RuntimeError("H3 upload/admission failed; NO_POST; uploaded inputs may remain on server: " + str(exc)) from exc
    core._submit_and_wait(
        graph, Path(opts.get("output_dir", "outputs")), timeout,
        notify=opts.get("notify_target") or os.environ.get("OPENCLAW_NOTIFY_TARGET"),
        caption_template=opts.get("caption_template"), user_prompt=opts["prompt"],
        require_output=True,
    )
