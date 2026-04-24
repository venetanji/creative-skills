#!/usr/bin/env python3
"""
comfy_run.py — Submit a ComfyUI workflow and wait for output.

Usage:
 python comfy_run.py <workflow.json>        # submit file
 python comfy_run.py -                      # read workflow JSON from stdin
 cat wf.json | python comfy_run.py -

Options:
 --output-dir DIR   Save generated files to DIR (default: ./outputs)
 --no-wait          Queue only, don't poll for result (prints prompt_id)
 --timeout N       Max seconds to wait (default: 600)
 --url URL          Override COMFY_URL env var

Output: prints asset URLs and saves files to --output-dir.

Env: COMFY_URL (default: https://comfyui.tail9683c.ts.net)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE = os.environ.get("COMFY_URL", "https://comfyui.tail9683c.ts.net").rstrip("/")


def post_json(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as r:
        return json.load(r)


def queue_workflow(workflow):
    result = post_json("/prompt", {"prompt": workflow})
    if result.get("node_errors"):
        print(f"[WARN] Node errors: {result['node_errors']}", file=sys.stderr)
    return result["prompt_id"]


def poll_result(prompt_id, timeout=600):
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        hist = get(f"/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            status = entry.get("status", {})
            status_str = status.get("status_str") or status.get("status")
            if status_str and status_str != last_status:
                print(f"[{status_str}]", file=sys.stderr)
                last_status = status_str
            if outputs:
                return entry
            if status.get("completed") is True or status_str in ("error", "success"):
                return entry
        time.sleep(2)
    raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout}s")


def extract_assets(entry):
    outputs = entry.get("outputs", {})
    assets = []
    seen = set()
    for nout in outputs.values():
        for v in nout.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "filename" in item:
                        key = (item["filename"], item.get("subfolder"), item.get("type"))
                        if key not in seen:
                            seen.add(key)
                            assets.append(item)
    return assets


def download(asset, output_dir):
    fname = asset["filename"]
    sf = asset.get("subfolder", "")
    tp = asset.get("type", "output")
    url = f"{BASE}/view?filename={urllib.parse.quote(fname)}&subfolder={sf}&type={tp}&_={int(time.time()*1000)}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / fname
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def run_workflow(workflow, output_dir="outputs", no_wait=False, timeout=600):
    """Submit workflow, optionally wait, download assets. Returns (prompt_id, assets, saved_paths)."""
    print(f"Submitting workflow...", file=sys.stderr)
    prompt_id = queue_workflow(workflow)
    print(f"prompt_id: {prompt_id}", file=sys.stderr)
    if no_wait:
        return prompt_id, [], []

    print(f"Waiting for result (timeout={timeout}s)...", file=sys.stderr)
    entry = poll_result(prompt_id, timeout)
    assets = extract_assets(entry)

    if not assets:
        status = entry.get("status", {})
        print(f"No assets produced. Status: {status}", file=sys.stderr)
        return prompt_id, [], []

    output_dir = Path(output_dir)
    saved = []
    for asset in assets:
        fname = asset["filename"]
        sf = asset.get("subfolder", "")
        tp = asset.get("type", "output")
        url = f"{BASE}/view?filename={urllib.parse.quote(fname)}&subfolder={sf}&type={tp}"
        print(f"asset_url: {url}")
        saved_path = download(asset, output_dir)
        print(f"saved: {saved_path}", file=sys.stderr)
        saved.append(saved_path)

    return prompt_id, assets, saved


def main():
    args = sys.argv[1:]
    workflow_src = None
    output_dir = Path("outputs")
    no_wait = False
    timeout = 600

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--output-dir":
            output_dir = Path(args[i + 1]); i += 2
        elif a == "--no-wait":
            no_wait = True; i += 1
        elif a == "--timeout":
            timeout = int(args[i + 1]); i += 2
        elif a == "--url":
            global BASE
            BASE = args[i + 1].rstrip("/"); i += 2
        elif a.startswith("--"):
            print(f"Unknown option: {a}", file=sys.stderr); sys.exit(1)
        else:
            workflow_src = a; i += 1

    if workflow_src is None:
        print(__doc__)
        sys.exit(1)

    if workflow_src == "-":
        workflow = json.load(sys.stdin)
    else:
        with open(workflow_src) as f:
            workflow = json.load(f)

    prompt_id, assets, saved = run_workflow(workflow, output_dir, no_wait, timeout)

    if no_wait:
        print(prompt_id)


if __name__ == "__main__":
    main()
