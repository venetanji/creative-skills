#!/usr/bin/env python3
"""Download and validate the four official SAM 3D Body workflow models."""
import argparse
import json
import shutil
import struct
import urllib.request
from pathlib import Path

from sam3d_body import BODY_MODEL, TRACK_MODEL, MOGE_MODEL, DETECTOR_MODEL


MODEL_FILES = (
    ("checkpoints", TRACK_MODEL, "sam3.1"),
    ("detection", BODY_MODEL, "sam-3d-body"),
    ("geometry_estimation", MOGE_MODEL, "MoGe"),
    ("diffusion_models", DETECTOR_MODEL, "SDPose"),
)


def validate_safetensors(path):
    """Reject incomplete downloads by checking tensor offsets against file size."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(header_size))
    offsets = [tensor["data_offsets"] for name, tensor in header.items()
               if name != "__metadata__"]
    if not offsets or max(end for start, end in offsets) + header_size + 8 != path.stat().st_size:
        raise ValueError(f"Incomplete safetensors data: {path}")
    return path.stat().st_size


def download_models(models_dir, check_only=False):
    for subfolder, filename, repository in MODEL_FILES:
        target = models_dir / subfolder / filename
        if target.exists():
            size = validate_safetensors(target)
            print(f"OK: {target} ({size / 1e9:.2f} GB)", flush=True)
            continue
        if check_only:
            raise FileNotFoundError(f"Missing model: {target}")
        url = f"https://huggingface.co/Comfy-Org/{repository}/resolve/main/{subfolder}/{filename}"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        print(f"Downloading: {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            size = int(response.headers.get("Content-Length", 0))
            if size and shutil.disk_usage(target.parent).free < size + 512 * 1024**2:
                raise OSError(f"Insufficient disk space for {target}")
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        validate_safetensors(partial)
        partial.replace(target)
        print(f"Saved: {target} ({target.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="The models/ directory of the ComfyUI server, not the skill")
    parser.add_argument("--check", action="store_true", help="Validate without downloading")
    args = parser.parse_args()
    download_models(args.models_dir, check_only=args.check)


if __name__ == "__main__":
    main()
