#!/usr/bin/env python3
"""
comfy_query.py — Query ComfyUI server state.

Usage:
 python comfy_query.py loras
 python comfy_query.py models [<type>] # type: loras|diffusion_models|checkpoints|vae|text_encoders|...
 python comfy_query.py node <NodeClass> # input schema + available values
 python comfy_query.py queue
 python comfy_query.py history [<prompt_id>] [--limit N]
 python comfy_query.py stats

Env: COMFY_URL (default: http://localhost:8188; set http://localhost:8000 for ComfyUI Desktop)
"""

import json
import os
import sys
import urllib.request
from urllib.parse import quote

BASE = os.environ.get("COMFY_URL", "http://localhost:8188").rstrip("/")


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as r:
        return json.load(r)


def cmd_loras():
    loras = get("/models/loras")
    print(f"{len(loras)} LoRAs:")
    for l in sorted(loras):
        print(f"  {l}")


def cmd_models(model_type=None):
    if model_type is None:
        types = get("/models")
        print("Model types:", types)
        return
    items = get(f"/models/{model_type}")
    print(f"{len(items)} {model_type}:")
    for i in sorted(items):
        print(f"  {i}")


def cmd_node(class_name):
    data = get(f"/object_info/{quote(class_name)}")
    info = data.get(class_name, {})
    inputs = info.get("input", {})
    print(f"Node: {class_name}")
    print(f"Category: {info.get('category', 'unknown')}")
    print(f"Output: {info.get('output_name', info.get('output', []))}")
    print()
    for section in ("required", "optional"):
        params = inputs.get(section, {})
        if not params:
            continue
        print(f"{section.upper()}:")
        for name, spec in params.items():
            type_info = spec[0] if spec else "?"
            options = spec[1] if len(spec) > 1 else {}
            if isinstance(type_info, list):
                print(f"  {name}: enum ({len(type_info)} options)")
                for v in type_info[:10]:
                    print(f"    - {v}")
                if len(type_info) > 10:
                    print(f"    ... +{len(type_info)-10} more")
            else:
                extra = ""
                if isinstance(options, dict):
                    if "default" in options:
                        extra += f" default={options['default']}"
                    if "min" in options or "max" in options:
                        extra += f" [{options.get('min','')}..{options.get('max','')}]"
                print(f"  {name}: {type_info}{extra}")


def cmd_queue():
    q = get("/queue")
    running = q.get("queue_running", [])
    pending = q.get("queue_pending", [])
    print(f"Running: {len(running)}, Pending: {len(pending)}")
    for item in running:
        print(f"  [running] #{item[0]} {item[1]}")
    for item in pending[:5]:
        print(f"  [pending] #{item[0]} {item[1]}")
    if len(pending) > 5:
        print(f"  ... +{len(pending)-5} more pending")


def cmd_history(prompt_id=None, limit=5):
    if prompt_id:
        data = get(f"/history/{prompt_id}")
        entry = data.get(prompt_id, {})
        if not entry:
            print(f"No history for {prompt_id}")
            return
        _print_entry(prompt_id, entry, verbose=True)
    else:
        data = get(f"/history?max_items={limit}")
        for pid, entry in list(data.items())[:limit]:
            _print_entry(pid, entry, verbose=False)


def _print_entry(pid, entry, verbose=False):
    status = entry.get("status", {})
    outputs = entry.get("outputs", {})
    node_errors = entry.get("node_errors", {})
    print(f"prompt_id: {pid}")
    print(f"  status: {status.get('status_str') or status.get('status', '?')}")
    if verbose and node_errors:
        print(f"  node_errors: {json.dumps(node_errors, indent=2)}")
    for nid, nout in outputs.items():
        for k, v in nout.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "filename" in v[0]:
                for asset in v:
                    fname = asset["filename"]
                    sf = asset.get("subfolder", "")
                    tp = asset.get("type", "output")
                    url = f"{BASE}/view?filename={fname}&subfolder={sf}&type={tp}"
                    print(f"  asset: {fname}")
                    if verbose:
                        print(f"    url: {url}")


def cmd_stats():
    data = get("/system_stats")
    sys_info = data.get("system", {})
    print(f"ComfyUI {sys_info.get('comfyui_version')}, Python {sys_info.get('python_version','?').split()[0]}")
    print(f"RAM: {sys_info.get('ram_free',0)//1024//1024//1024}GB free / {sys_info.get('ram_total',0)//1024//1024//1024}GB total")
    for dev in data.get("devices", []):
        vf = dev.get('vram_free', 0) // 1024 // 1024
        vt = dev.get('vram_total', 0) // 1024 // 1024
        print(f"GPU: {dev['name']} — {vf}MB free / {vt}MB total")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    rest = args[1:]

    if cmd == "loras":
        cmd_loras()
    elif cmd == "models":
        cmd_models(rest[0] if rest else None)
    elif cmd == "node":
        if not rest:
            print("Usage: node <NodeClass>")
            sys.exit(1)
        cmd_node(rest[0])
    elif cmd == "queue":
        cmd_queue()
    elif cmd == "history":
        pid = None
        limit = 5
        for i, a in enumerate(rest):
            if a == "--limit" and i + 1 < len(rest):
                limit = int(rest[i + 1])
            elif not a.startswith("--"):
                pid = a
        cmd_history(pid, limit)
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
