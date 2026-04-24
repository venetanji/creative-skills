"""Core helpers for ComfyUI workflow creation and submission."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.parse
import shutil
import subprocess
import shlex
from pathlib import Path

BASE = os.environ.get("COMFY_URL", "https://comfyui.tail9683c.ts.net").rstrip("/")

# Default notification target: Discord DM channel for gio's DMs
NOTIFY_TARGET = os.environ.get(
    "OPENCLAW_NOTIFY_TARGET",
    os.environ.get("OPENCLAW_DISCORD_TARGET", "1486985676066000957")
)

VOICE_LIBRARY = {
    "gio": {"file": "voice.mp3", "reference_text": "", "x_vector_only": True},
}


class NodeRef:
    def __init__(self, node_id: str, output_idx: int = 0):
        self.node_id = node_id
        self.output_idx = output_idx

    def __getitem__(self, idx: int) -> "NodeRef":
        return NodeRef(self.node_id, idx)

    def as_link(self) -> list:
        return [self.node_id, self.output_idx]

    def __repr__(self):
        return f"NodeRef({self.node_id!r}, {self.output_idx})"


class WorkflowGraph:
    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._counter = 0

    def node(self, class_type: str, **inputs) -> NodeRef:
        node_id = str(self._counter)
        self._counter += 1
        processed = {}
        for k, v in inputs.items():
            if isinstance(v, NodeRef):
                processed[k] = v.as_link()
            else:
                processed[k] = v
        self._nodes[node_id] = {"class_type": class_type, "inputs": processed}
        return NodeRef(node_id)

    def to_dict(self) -> dict:
        return dict(self._nodes)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def upload_image(local_path: str, subfolder: str = "",
                  upload_as: str | None = None) -> str:
    """Upload a local file into ComfyUI's input/ dir, optionally under a
    subfolder and/or with a different remote name. Returns the server-side
    name (relative to input/).

    subfolder: if set, file lands at `/app/ComfyUI/input/<subfolder>/<name>`.
    upload_as: override the remote filename. Useful for enforcing
    alphabetical ordering (e.g. prefix with `001_` for concat staging)."""
    path = Path(local_path)
    data = path.read_bytes()
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "application/octet-stream"
    remote_name = upload_as or path.name
    boundary = "----ComfyUploadBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{remote_name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="subfolder"\r\n\r\n{subfolder}\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    return resp["name"]


def upload_if_local(path: str, upload_flag: bool = True) -> str:
    if not path:
        return path
    try:
        p = Path(path)
        if p.exists() and upload_flag:
            try:
                server = upload_image(str(p))
                print(f"Uploaded local image {p} -> {server}")
                return server
            except Exception as e:
                print(f"[WARN] Failed to upload image {p}: {e}", file=sys.stderr)
                return path
    except Exception:
        pass
    return path


def _submit_and_wait(workflow: dict, output_dir: Path, timeout: int = 600, notify: str | None = None, caption_template: str | None = None, user_prompt: str | None = None):
    # Inside sandbox (home=/home/sandbox), /workspace is mounted rw.
    # Redirect output_dir to /workspace/media/outbound so downloads succeed.
    if Path.home() == Path("/home/sandbox"):
        output_dir = Path("/workspace/media/outbound")
    def _retrying(url, *, data=None, method="GET", max_transient=30):
        """GET/POST with retry on transient HTTP 5xx / connection errors.
        Returns parsed JSON on success. Raises on persistent failure."""
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        transient = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                # 5xx: server error, retry generously. 400 on /prompt: usually
                # an upload-race (file not yet visible server-side when the
                # workflow references it) — retry a small number of times
                # with short backoff; real malformed workflows still fail.
                retry_400 = (e.code == 400 and url.endswith("/prompt")
                             and transient < 3)
                if (500 <= e.code < 600 and transient < max_transient) or retry_400:
                    transient += 1
                    wait = min(20, 2 * transient)
                    print(f"[transient HTTP {e.code}] {url}; retry {transient}/{max_transient} in {wait}s",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if transient < max_transient:
                    transient += 1
                    wait = min(20, 2 * transient)
                    print(f"[transient {type(e).__name__}] {url}; retry {transient}/{max_transient} in {wait}s",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise

    payload = json.dumps({"prompt": workflow}).encode()
    result = _retrying(f"{BASE}/prompt", data=payload, method="POST")
    if result.get("node_errors"):
        print(f"[WARN] Node errors: {result['node_errors']}", file=sys.stderr)
    prompt_id = result["prompt_id"]
    print(f"prompt_id: {prompt_id}", file=sys.stderr)

    deadline = time.time() + timeout
    last_st = None
    while time.time() < deadline:
        hist = _retrying(f"{BASE}/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            st = (entry.get("status") or {}).get("status_str") or (entry.get("status") or {}).get("status")
            if st and st != last_st:
                print(f"[{st}]", file=sys.stderr)
                last_st = st
            if outputs:
                _save_assets(entry, output_dir, notify=notify, caption_template=caption_template, user_prompt=user_prompt)
                return prompt_id
            if (entry.get("status") or {}).get("completed") is True or st == "error":
                print("Completed with no outputs.", file=sys.stderr)
                return prompt_id
        time.sleep(2)
    raise TimeoutError(f"Timed out after {timeout}s")


def _save_assets(entry: dict, output_dir: Path, notify: str | None = None, caption_template: str | None = None, user_prompt: str | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    saved_files = []
    for nout in entry.get("outputs", {}).values():
        for v in nout.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "filename" in item:
                        fname = item["filename"]
                        if fname in seen:
                            continue
                        seen.add(fname)
                        sf = item.get("subfolder", "") or ""
                        tp = item.get("type", "output")
                        norm_fname = fname.replace("\\", "/")
                        q_fname = urllib.parse.quote(norm_fname, safe='/')
                        q_sub = urllib.parse.quote(sf)
                        url = f"{BASE}/view?filename={q_fname}&subfolder={q_sub}&type={tp}&_={int(time.time()*1000)}"
                        print(f"asset_url: {url}")
                        dest = output_dir / Path(fname).name
                        try:
                            with urllib.request.urlopen(url, timeout=120) as r:
                                dest.write_bytes(r.read())
                        except Exception as e:
                            print(f"[WARN] Failed to download asset {fname}: {e}", file=sys.stderr)
                            continue
                        print(f"saved: {dest}")
                        saved_files.append(dest)

    # Copy outputs to the media/outbound dir inside the agent's workspace
    # so they are reachable via a `MEDIA:<host-path>` directive. Path:
    #   - Sandboxed agents: /workspace/media/outbound/  (the workspace
    #     bind mount surfaces this on the host at the agent's workspace dir,
    #     e.g. /home/venetanji/.openclaw/workspace/media/outbound/)
    #   - Host (main, zeus): /home/venetanji/.openclaw/workspace/media/outbound/
    #   - Override anything via the OPENCLAW_MEDIA_DIR env var (expects the
    #     `outbound/` subdir to exist or be creatable under it).
    try:
        env_override = os.environ.get("OPENCLAW_MEDIA_DIR")
        if env_override:
            outbound_dir = Path(env_override) / "outbound"
        elif Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
            outbound_dir = Path("/workspace/media/outbound")
        else:
            outbound_dir = Path("/home/venetanji/.openclaw/workspace/media/outbound")
        outbound_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        outbound_dir = Path("/home/venetanji/.openclaw/workspace/media/outbound")
        outbound_dir.mkdir(parents=True, exist_ok=True)

    for p in saved_files:
        try:
            dst = outbound_dir / p.name
            try:
                need_copy = (not dst.exists()) or (dst.stat().st_size != p.stat().st_size)
            except Exception:
                need_copy = True
            if need_copy:
                try:
                    dst.write_bytes(p.read_bytes())
                except Exception as e:
                    print(f"[WARN] Failed to copy {p} -> {dst}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed preparing outbound copy for {p}: {e}", file=sys.stderr)

    if notify:
        # Collect asset URLs and captions for agent to send via message tool (media param accepts URLs)
        # This avoids the openclaw CLI subprocess which fails inside sandboxed environments.
        for i, p in enumerate(saved_files):
            dst = outbound_dir / p.name
            url = f"{BASE}/view?filename={urllib.parse.quote(p.name)}&subfolder=&type=output"
            workflow_name = entry.get('workflow_name') or ''
            prompt_id = entry.get('prompt_id') or ''
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            prompt_snippet = (user_prompt or '')[:100].replace('\n', ' ') if user_prompt else ''
            if caption_template:
                caption = caption_template.format(filename=p.name, workflow=workflow_name, id=prompt_id, ts=timestamp, prompt=prompt_snippet)
            else:
                parts = [p.name]
                if workflow_name:
                    parts.append(f"workflow={workflow_name}")
                if prompt_snippet:
                    parts.append(f"prompt={prompt_snippet}")
                parts.append(timestamp)
                caption = " | ".join(parts)
            print(f"[NOTIFY_URL {i}] target={notify} media={url} caption={caption}")
