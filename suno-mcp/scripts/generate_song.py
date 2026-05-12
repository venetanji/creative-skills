#!/usr/bin/env python3
"""
generate_song.py — Generate a song with the Suno MCP server and download
the audio file(s) locally.

Talks to the Suno MCP server directly over the MCP Streamable HTTP
transport (JSON-RPC 2.0 framed inside an SSE-style ``event: message``
envelope). No mcporter / npx dependency — just the standard library.

Usage:
  generate_song.py --lyrics='...' --tags='...' --title='...'
  generate_song.py --instrumental --tags='...' --title='...'
  generate_song.py --tags='...' --title='...' --negative-prompt='heavy metal'
  generate_song.py --tags='...' --title='...' --dry-run   # show request, no API call

Env:
  SUNO_MCP_URL      Base URL of the Suno MCP server. Default
                    ``http://localhost:8190``. The MCP endpoint is
                    ``<SUNO_MCP_URL>/mcp`` and downloaded songs are
                    served at ``<SUNO_MCP_URL>/audio/<file>.mp3``.
  SUNO_OUTPUT_DIR   Where to save downloaded MP3s. Default
                    ``./outputs/suno`` (relative to CWD).

Both env vars can be overridden per-call with ``--url`` and
``--output-dir`` CLI flags (CLI wins over env).
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE = "http://localhost:8190"
DEFAULT_OUTPUT_DIR = "./outputs/suno"


# ----------------------------------------------------------------------
# Minimal MCP-over-HTTP client (Streamable HTTP transport).
# One session per process: initialize → notifications/initialized → calls.
# ----------------------------------------------------------------------

class MCPClient:
    """Tiny synchronous MCP Streamable-HTTP client.

    Handles the SSE-framed ``event: message\\ndata: <json>`` response
    envelope FastMCP uses by default. Holds the ``mcp-session-id``
    header returned by ``initialize`` and reuses it for subsequent calls.
    """

    def __init__(self, base_url: str, client_name: str = "suno-mcp-skill",
                 client_version: str = "0.2.0"):
        self.base = base_url.rstrip("/")
        self.endpoint = f"{self.base}/mcp"
        self.session_id: str | None = None
        self._next_id = 0
        self.client_name = client_name
        self.client_version = client_version

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _post(self, body: dict[str, Any], *, timeout: int) -> tuple[dict[str, Any] | None, dict[str, str]]:
        """POST a JSON-RPC frame, parse the SSE envelope, return (result, headers).

        Returns (None, headers) for notifications (server returns no body).
        """
        data = json.dumps(body).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
                resp_headers = {k.lower(): v for k, v in r.headers.items()}
        except urllib.error.HTTPError as e:
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = ""
            raise RuntimeError(f"HTTP {e.code} from {self.endpoint}: {body_text}") from e

        # Notifications return 202 Accepted with no body.
        if not raw.strip():
            return None, resp_headers

        # FastMCP wraps single JSON-RPC frames in an SSE envelope:
        #   event: message
        #   data: <json>
        # …possibly followed by other events. Pull out the data: lines.
        data_chunks: list[str] = []
        for line in raw.splitlines():
            if line.startswith("data: "):
                data_chunks.append(line[6:])
            elif line.startswith("data:"):
                data_chunks.append(line[5:].lstrip())
        if data_chunks:
            payload = json.loads(data_chunks[-1])
        else:
            # Plain JSON body (some MCP servers don't use SSE framing).
            payload = json.loads(raw)
        if "error" in payload:
            err = payload["error"]
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
        return payload.get("result"), resp_headers

    def initialize(self, *, timeout: int = 30) -> dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": self._id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            },
        }
        result, headers = self._post(body, timeout=timeout)
        sid = headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        # Server expects the initialized notification before any tools/* call.
        note = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        self._post(note, timeout=timeout)
        return result or {}

    def call_tool(self, name: str, arguments: dict[str, Any], *, timeout: int = 600) -> dict[str, Any]:
        body = {
            "jsonrpc": "2.0",
            "id": self._id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        result, _ = self._post(body, timeout=timeout)
        return result or {}


# ----------------------------------------------------------------------
# Suno-specific glue
# ----------------------------------------------------------------------

# The current Suno MCP responses are plain-text strings (FastMCP
# generic-wrap). We extract IDs / download URLs / audio filenames by
# regex. If/when the server emits structured content we can switch to
# reading `result["structuredContent"]` instead.

_ID_RE = re.compile(r"\bID:\s*([a-f0-9-]{36})", re.IGNORECASE)
_DL_RE = re.compile(r"\bDownload:\s*(https?://\S+)")
_FILENAME_RE = re.compile(r"Downloaded to:\s*\S*?([0-9a-f]{8}\.mp3)", re.IGNORECASE)
_STREAM_RE = re.compile(r"(?:Stream|Local)\s*URL:\s*(\S+/audio/[0-9a-f]{8}\.mp3)", re.IGNORECASE)
_RETRYABLE_PATTERNS = (
    "Song generation timed out",
    "no new songs appeared",
    "generation failed",
    "temporarily unavailable",
)


def _text_from_result(result: dict[str, Any]) -> str:
    """Flatten the tools/call return value to a single string.

    FastMCP tools that wrap a string return value emit
    ``{"content":[{"type":"text","text":"..."}], "structuredContent":{"result":"..."}}``.
    """
    if not result:
        return ""
    sc = result.get("structuredContent") or {}
    if isinstance(sc, dict) and isinstance(sc.get("result"), str):
        return sc["result"]
    parts: list[str] = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts)


def _build_generate_args(lyrics: str, tags: str, title: str, *,
                         instrumental: bool, negative_prompt: str) -> dict[str, Any]:
    args: dict[str, Any] = {"tags": tags, "title": title}
    if instrumental:
        args["make_instrumental"] = True
    else:
        args["lyrics"] = lyrics
    if negative_prompt:
        args["negative_prompt"] = negative_prompt
    return args


def _fetch(url: str, dest: Path, *, tries: int = 40, per_try_timeout: int = 30) -> bool:
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; suno-mcp-skill/0.2)"
            })
            with urllib.request.urlopen(req, timeout=per_try_timeout) as r:
                if r.status == 200:
                    dest.write_bytes(r.read())
                    return True
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                print(f"  HTTP {e.code} on {url}", file=sys.stderr)
        except Exception as e:
            print(f"  fetch err: {e}", file=sys.stderr)
        time.sleep(3)
    return False


def _audio_url_from_download_text(base: str, body: str) -> str | None:
    """Derive the audio URL from a download_song response, using ``base``.

    The server reports either ``Downloaded to: /downloads/<short>.mp3`` or
    ``Stream URL: <SUNO_PUBLIC_URL>/audio/<short>.mp3`` — we ignore the
    embedded host (cross-tailnet agents can't reach it) and rebuild the
    URL against the agent's configured ``SUNO_MCP_URL`` base, mirroring
    how the same Worker bridge serves both ``/mcp`` and ``/audio/*``.
    """
    m = _FILENAME_RE.search(body)
    if m:
        return f"{base}/audio/{m.group(1)}"
    m2 = _STREAM_RE.search(body)
    if m2:
        # Replace the host part but keep the filename.
        fname = m2.group(1).rsplit("/", 1)[-1]
        return f"{base}/audio/{fname}"
    return None


def generate_song(lyrics: str, tags: str, title: str, *,
                  instrumental: bool = False,
                  negative_prompt: str = "",
                  base_url: str | None = None,
                  output_dir: str | os.PathLike | None = None,
                  timeout: int = 400) -> dict[str, Any]:
    """Generate a song via the Suno MCP server and download both variants.

    Returns a dict with at least ``all_ids``, ``local_files``,
    ``local_file`` (alias of ``local_files[0]``), and ``result`` (the
    raw text response from ``generate_song``).
    """
    base = (base_url or os.environ.get("SUNO_MCP_URL") or DEFAULT_BASE).rstrip("/")
    out_path = Path(output_dir or os.environ.get("SUNO_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    client = MCPClient(base)
    client.initialize()

    args = _build_generate_args(lyrics, tags, title,
                                instrumental=instrumental,
                                negative_prompt=negative_prompt)

    print(f"Generating song: {title} (base={base})...", file=sys.stderr)

    # Retry on transient upstream / suno-side failures. Both HTTP errors
    # and zero-status responses that contain a known retryable marker.
    last_err = ""
    text = ""
    for attempt in range(1, 6):
        try:
            result = client.call_tool("generate_song", args, timeout=timeout)
            text = _text_from_result(result)
        except RuntimeError as e:
            last_err = str(e)
            transient = any(s in last_err for s in (
                "HTTP 50", "HTTP 429", "timed out", "timeout",
                "ECONNRESET", "ETIMEDOUT", "EAI_AGAIN",
            ))
            if not transient:
                raise
            text = ""
        else:
            retryable = any(s in text for s in _RETRYABLE_PATTERNS)
            has_id = bool(_ID_RE.search(text))
            if has_id and not retryable:
                break
            if not retryable:
                # No IDs and no known-retryable marker — bail.
                break
            last_err = text

        backoff = min(60, 5 * (2 ** (attempt - 1)))
        print(f"Suno transient issue (attempt {attempt}/5); retrying in {backoff}s",
              file=sys.stderr)
        time.sleep(backoff)

    all_ids = _ID_RE.findall(text)
    all_urls = _DL_RE.findall(text)
    if not all_ids:
        raise RuntimeError(f"No song IDs in response: {text or last_err}")

    response: dict[str, Any] = {
        "result": text,
        "id": all_ids[0],
        "all_ids": all_ids,
        "download_url": all_urls[0] if all_urls else None,
        "all_download_urls": all_urls,
    }

    # Download each variant. Preference order:
    #   1. The MCP `download_song` proxy URL (`<base>/audio/<short>.mp3`)
    #      — stable, served by the same Worker that fronts /mcp.
    #   2. The direct CDN URL from the `generate_song` text response
    #      (cdn1.suno.ai/<uuid>.mp3) — works when reachable.
    local_files: list[str] = []
    for i, vid in enumerate(all_ids):
        out = out_path / f"suno_{vid}.mp3"
        direct = all_urls[i] if i < len(all_urls) else None
        print(f"Variant {i+1}/{len(all_ids)}: id={vid}", file=sys.stderr)

        ok = False
        try:
            dl_result = client.call_tool("download_song", {"song_id": vid}, timeout=300)
            dl_text = _text_from_result(dl_result)
            audio_url = _audio_url_from_download_text(base, dl_text)
            if audio_url:
                print(f"  MCP URL: {audio_url}", file=sys.stderr)
                ok = _fetch(audio_url, out, tries=40)
                if not ok:
                    print(f"  MCP URL didn't yield file; falling back to direct CDN", file=sys.stderr)
            else:
                print(f"  download_song returned no Stream URL", file=sys.stderr)
        except RuntimeError as e:
            print(f"  download_song MCP call failed: {e}", file=sys.stderr)

        if not ok and direct:
            print(f"  Direct: {direct}", file=sys.stderr)
            ok = _fetch(direct, out, tries=40)
        if not ok:
            raise RuntimeError(f"Failed to download variant {i+1} (id={vid})")
        print(f"  Downloaded: {out.stat().st_size} bytes to {out}", file=sys.stderr)
        local_files.append(str(out))

    response["local_file"] = local_files[0]
    response["local_files"] = local_files
    return response


def _print_dry_run(args, base: str) -> None:
    """Print the JSON-RPC body we'd send, without contacting the server."""
    gen_args = _build_generate_args(
        args.lyrics or "", args.tags, args.title,
        instrumental=args.instrumental,
        negative_prompt=args.negative_prompt or "",
    )
    body = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "generate_song", "arguments": gen_args},
    }
    print(json.dumps({
        "endpoint": f"{base}/mcp",
        "headers": {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        },
        "body": body,
        "output_dir": str(Path(args.output_dir or os.environ.get("SUNO_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a song with Suno MCP and download both variants.")
    # `--input-json` is the recommended path for sandboxed agents:
    # the agent writes a JSON file with the song spec via the `write`
    # tool, then runs this script with a single argument. No shell
    # quoting of multiline lyrics, no need to wrap in a per-invocation
    # python helper. CLI flags (--lyrics, --tags, …) still work and
    # override JSON values when both are supplied — handy for ad-hoc
    # tweaks (e.g. `--input-json spec.json --title "Take 2"`).
    parser.add_argument("--input-json", "-j",
                        help="Path to a JSON file with keys "
                             "{lyrics, tags, title, instrumental, "
                             "negative_prompt, url, output_dir, "
                             "timeout}. Any subset is allowed; CLI "
                             "flags override JSON values.")
    parser.add_argument("--lyrics", "-l", help="Song lyrics")
    parser.add_argument("--tags", "-t", help="Genre/style prompt (producer brief)")
    parser.add_argument("--title", "-T", help="Song title")
    parser.add_argument("--instrumental", "-i", action="store_true",
                        help="Instrumental track (no vocals)")
    parser.add_argument("--negative-prompt", "-n", default=None,
                        help="Styles to avoid, e.g. 'heavy metal, screaming'")
    parser.add_argument("--url", help="Override SUNO_MCP_URL env var")
    parser.add_argument("--output-dir", help="Override SUNO_OUTPUT_DIR env var")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Total timeout in seconds (default 400 — pipeline is 3-5 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the MCP request body and exit without calling the server")

    args = parser.parse_args()

    # Merge: JSON file (if any) provides defaults; CLI flags override.
    spec: dict = {}
    if args.input_json:
        try:
            spec = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading --input-json {args.input_json}: {e}",
                  file=sys.stderr)
            sys.exit(1)
        if not isinstance(spec, dict):
            print(f"Error: --input-json {args.input_json} must contain a JSON object, "
                  f"got {type(spec).__name__}",
                  file=sys.stderr)
            sys.exit(1)

    def _pick(cli_value, spec_key: str, default=None):
        if cli_value is not None:
            return cli_value
        if spec_key in spec:
            return spec[spec_key]
        return default

    lyrics          = _pick(args.lyrics,          "lyrics",          "")
    tags            = _pick(args.tags,            "tags",            None)
    title           = _pick(args.title,           "title",           None)
    instrumental    = args.instrumental or bool(spec.get("instrumental", False))
    negative_prompt = _pick(args.negative_prompt, "negative_prompt", "")
    url             = _pick(args.url,             "url",             None)
    output_dir      = _pick(args.output_dir,      "output_dir",      None)
    timeout         = _pick(args.timeout,         "timeout",         400)

    if not tags:
        print("Error: --tags (or 'tags' key in --input-json) is required.",
              file=sys.stderr)
        sys.exit(2)
    if not title:
        print("Error: --title (or 'title' key in --input-json) is required.",
              file=sys.stderr)
        sys.exit(2)

    base = (url or os.environ.get("SUNO_MCP_URL") or DEFAULT_BASE).rstrip("/")

    if args.dry_run:
        # _print_dry_run reads from args; fill the resolved fields back
        # in so the dry-run reflects the merged values.
        args.lyrics = lyrics
        args.tags = tags
        args.title = title
        args.instrumental = instrumental
        args.negative_prompt = negative_prompt
        args.output_dir = output_dir
        _print_dry_run(args, base)
        return

    try:
        result = generate_song(
            lyrics=lyrics or "",
            tags=tags,
            title=title,
            instrumental=instrumental,
            negative_prompt=negative_prompt or "",
            base_url=base,
            output_dir=output_dir,
            timeout=timeout,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
