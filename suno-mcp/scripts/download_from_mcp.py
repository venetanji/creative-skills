#!/usr/bin/env python3
"""
download_from_mcp.py — Re-download a previously generated song by ID.

Triggers ``download_song`` on the Suno MCP server, then pulls the audio
file from the ``/audio/<short>.mp3`` endpoint on the same host. No
mcporter / npx dependency.

Usage:
  download_from_mcp.py <song_id>
  download_from_mcp.py <song_id> --output-dir /tmp/songs
  download_from_mcp.py <song_id> --url http://localhost:8190

Env (CLI flags override):
  SUNO_MCP_URL      Base URL of the Suno MCP server (default
                    ``http://localhost:8190``). The MCP endpoint is at
                    ``<base>/mcp`` and audio at ``<base>/audio/<file>.mp3``.
  SUNO_OUTPUT_DIR   Where to save the MP3 (default ``./outputs/suno``).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Re-use the MCP client + helpers from generate_song.py so we don't drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_song import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_OUTPUT_DIR,
    MCPClient,
    _audio_url_from_download_text,
    _text_from_result,
)


def download_song(song_id: str, *,
                  base_url: str | None = None,
                  output_dir: str | os.PathLike | None = None,
                  max_wait: int = 120) -> str:
    """Trigger ``download_song`` and pull the MP3 from ``<base>/audio/<file>``.

    Returns the local path on disk.
    """
    base = (base_url or os.environ.get("SUNO_MCP_URL") or DEFAULT_BASE).rstrip("/")
    out_path = Path(output_dir or os.environ.get("SUNO_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / f"suno_{song_id}.mp3"

    client = MCPClient(base)
    client.initialize()

    print(f"Triggering server download for {song_id}...", file=sys.stderr)
    result = client.call_tool("download_song", {"song_id": song_id}, timeout=300)
    body = _text_from_result(result)
    audio_url = _audio_url_from_download_text(base, body)
    if not audio_url:
        # Fall back to the conventional 8-hex-char audio path; some
        # server versions only emit the short id implicitly. Last
        # resort — most callers will provide the full uuid here.
        audio_url = f"{base}/audio/{song_id}.mp3"

    print(f"Waiting for file at {audio_url}...", file=sys.stderr)
    waited = 0
    poll = 3
    while waited < max_wait:
        try:
            req = urllib.request.Request(audio_url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    print("File ready. Downloading...", file=sys.stderr)
                    break
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"HTTP {e.code}; retrying...", file=sys.stderr)
        except Exception as e:
            print(f"Connection error: {e}; retrying...", file=sys.stderr)
        time.sleep(poll)
        waited += poll
    else:
        raise RuntimeError(f"Timeout waiting for file after {max_wait}s")

    req = urllib.request.Request(audio_url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; suno-mcp-skill/0.2)"
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        output_file.write_bytes(r.read())

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError("Download failed — file empty or missing")

    print(f"Downloaded: {output_file.stat().st_size} bytes to {output_file}", file=sys.stderr)
    return str(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a song from Suno MCP")
    parser.add_argument("song_id", help="Suno song ID")
    parser.add_argument("--url", help="Override SUNO_MCP_URL env var")
    parser.add_argument("--output-dir", "-o", help="Override SUNO_OUTPUT_DIR env var")
    parser.add_argument("--max-wait", "-w", type=int, default=120,
                        help="Max seconds to wait for file availability")

    args = parser.parse_args()

    try:
        path = download_song(
            args.song_id,
            base_url=args.url,
            output_dir=args.output_dir,
            max_wait=args.max_wait,
        )
        print(json.dumps({"local_file": path, "song_id": args.song_id}))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
