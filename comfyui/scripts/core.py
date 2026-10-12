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

BASE = os.environ.get("COMFY_URL", "http://localhost:8188").rstrip("/")

# Default notification target: Discord DM channel for gio's DMs
NOTIFY_TARGET = os.environ.get(
    "OPENCLAW_NOTIFY_TARGET",
    os.environ.get("OPENCLAW_DISCORD_TARGET", "1486985676066000957")
)

VOICE_LIBRARY = {
    "gio": {"file": "voice.mp3", "reference_text": "", "x_vector_only": True},
}


try:
    from creative_comfy_graph import NodeRef, WorkflowGraph
except ModuleNotFoundError as exc:
    if exc.name != "creative_comfy_graph":
        raise
    raise ImportError(
        "creative-comfy-graph is required; install it into the Python environment running this script."
    ) from exc


def upload_image(local_path: str, subfolder: str = "",
                  upload_as: str | None = None, *, strict_input: bool = False) -> str:
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
    if strict_input and (not isinstance(resp, dict) or
                         resp.get("type") != "input" or
                         resp.get("subfolder") != subfolder):
        raise ValueError("upload response must identify the requested input storage location")
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


def _submit_and_wait(workflow: dict, output_dir: Path, timeout: int = 600, notify: str | None = None, caption_template: str | None = None, user_prompt: str | None = None, *, require_output: bool = False):
    # Inside an OpenClaw sandbox, the default --output-dir (./outputs/) lands
    # in /home/sandbox which is ephemeral and not visible host-side. Redirect
    # to a workspace-mounted dir so downloads survive the session and the
    # MEDIA: directive can resolve them. Precedence:
    #   1. OPENCLAW_MEDIA_DIR  — explicit override (full path, no suffix).
    #   2. /workspace/media/outbound — sandbox bind-mount default.
    # On commons-tier sandboxes /workspace is mounted READ-ONLY, so the env
    # var is required there; without it the subsequent mkdir raises
    # PermissionError. Mirrors suno-mcp's SUNO_OUTPUT_DIR pattern.
    sandbox_media_dir = os.environ.get("OPENCLAW_MEDIA_DIR")
    if sandbox_media_dir:
        output_dir = Path(sandbox_media_dir)
    elif Path.home() == Path("/home/sandbox"):
        output_dir = Path("/workspace/media/outbound")
    # Graph-only imports/help/dump do not need the transport-only sibling.
    from queue_admission import wait_for_empty_queue

    deadline = time.monotonic() + timeout
    prompt_id = None

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            if prompt_id is None:
                raise TimeoutError(f"Timed out before submission after {timeout}s; NO_POST; prompt_id=None")
            raise TimeoutError(f"Timed out after {timeout}s; prompt_id={prompt_id}; do not resubmit")
        return seconds

    def request_json(url, *, data=None):
        # Only read-only history requests may retry. A failed POST response can
        # hide an accepted job; retrying would create an untracked duplicate.
        method = "POST" if data is not None else "GET"
        req = urllib.request.Request(url, data=data, method=method,
            headers={"Content-Type": "application/json"} if data is not None else {})
        transient = 0
        if method == "POST":
            # The sole POST here is /prompt; finish request preparation first.
            wait_for_empty_queue(BASE, deadline)
        while True:
            request_timeout = min(15, remaining())
            try:
                with urllib.request.urlopen(req, timeout=request_timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode('utf-8', errors='replace')
                if method == "POST" or not 500 <= exc.code < 600:
                    raise RuntimeError(f"{method} {url}: HTTP {exc.code}: {body}; no resubmission") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if method == "POST":
                    raise RuntimeError(f"Ambiguous prompt submission: {exc}; do not resubmit; inspect backend queue/history") from exc
                remaining()
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError(f"Invalid {method} response from {url}; submission may be ambiguous; do not resubmit") from exc
            transient += 1
            time.sleep(min(20, 2 * transient, remaining()))

    payload = json.dumps({"prompt": workflow}).encode()
    result = request_json(f"{BASE}/prompt", data=payload)
    if not isinstance(result, dict) or result.get("node_errors") or not result.get("prompt_id"):
        raise RuntimeError(f"Prompt rejected or ambiguous response: {json.dumps(result)}; no resubmission")
    prompt_id = result["prompt_id"]
    print(f"prompt_id: {prompt_id}", file=sys.stderr)
    last_st = None
    while True:
        remaining()
        hist = request_json(f"{BASE}/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            st = status.get("status_str") or status.get("status")
            if st and st != last_st:
                print(f"[{st}]", file=sys.stderr)
                last_st = st
            events = [m[0] for m in status.get("messages", []) if isinstance(m, (list, tuple)) and m]
            if st in {"error", "interrupted"} or any(e in {"execution_error", "execution_interrupted"} for e in events):
                raise RuntimeError(f"Prompt {prompt_id} failed: {json.dumps(status)}")
            if status.get("completed") is True:
                if st != "success":
                    raise RuntimeError(f"Prompt {prompt_id} completed without success: {json.dumps(status)}")
                saved = _save_assets(entry, output_dir, notify=notify, caption_template=caption_template,
                                     user_prompt=user_prompt, strict=require_output)
                if require_output and not saved:
                    raise RuntimeError(f"Prompt {prompt_id} succeeded without a downloadable output asset")
                return prompt_id
        time.sleep(min(2, remaining()))


def _save_assets(entry: dict, output_dir: Path, notify: str | None = None, caption_template: str | None = None, user_prompt: str | None = None, *, strict: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    saved_files = []
    for nout in entry.get("outputs", {}).values():
        asset_values = list(nout.values())
        preview = nout.get("result")
        if (isinstance(preview, list) and preview and isinstance(preview[0], str)
                and preview[0].endswith(".glb")):
            asset_values.append([{"filename": preview[0], "type": "output"}])
        for v in asset_values:
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "filename" in item and item.get("type") == "output":
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
                            if strict:
                                raise RuntimeError(f"Failed to download output {fname}: {e}") from e
                            print(f"[WARN] Failed to download asset {fname}: {e}", file=sys.stderr)
                            continue
                        print(f"saved: {dest}")
                        saved_files.append(dest)

    # Copy outputs to the media/outbound dir inside the agent's workspace so
    # they are reachable via a `MEDIA:<host-path>` directive. Resolution order:
    #   1. OPENCLAW_MEDIA_DIR  — explicit override (full path, used as-is —
    #      mirrors suno-mcp's SUNO_OUTPUT_DIR semantics).
    #   2. /workspace/media/outbound — the OpenClaw sandbox convention
    #      (bind-mounted to the agent's workspace dir on the host), when
    #      /workspace is writable.
    #   3. ~/.openclaw/workspace/media/outbound — host install fallback.
    try:
        env_override = os.environ.get("OPENCLAW_MEDIA_DIR")
        if env_override:
            outbound_dir = Path(env_override)
        elif Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
            outbound_dir = Path("/workspace/media/outbound")
        else:
            outbound_dir = Path.home() / ".openclaw" / "workspace" / "media" / "outbound"
        outbound_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        outbound_dir = Path.home() / ".openclaw" / "workspace" / "media" / "outbound"
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

    return saved_files
