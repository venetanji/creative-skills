#!/usr/bin/env python3
"""
Generate a song with Suno MCP and download the file.
Returns the local file path for attachment.
"""

import subprocess
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def _suno_mcp_base_url(config_path: str) -> str | None:
    """Read the mcporter config and return the agent's suno base URL.

    Strips the trailing ``/mcp`` so the returned base can be reused for
    both ``/mcp`` (control plane) and ``/audio/<file>.mp3`` (downloads).
    Returns None if the config doesn't have a suno entry.
    """
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    entry = (cfg.get("mcpServers") or {}).get("suno") or {}
    url = entry.get("url")
    if not url:
        return None
    return url[:-4] if url.endswith("/mcp") else url.rstrip("/")


def generate_song(lyrics, tags, title, instrumental=False, timeout=120):
    """
    Generate a song with Suno and download it.
    
    Args:
        lyrics: Song lyrics (can be empty if instrumental=True)
        tags: Genre/style tags (e.g., "pop, upbeat, electronic")
        title: Song title
        instrumental: True for instrumental track
        timeout: Max seconds to wait for generation
    
    Returns:
        dict with song info including 'local_file' path
    """
    config = os.path.expanduser("~/.openclaw/config/mcporter.json")
    output_dir = Path(os.path.expanduser("~/.openclaw/workspace/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build mcporter command (use npx to run mcporter). Pass --timeout explicitly
    # because mcporter's own default is 60s and suno's generate endpoint can take
    # longer on busy queues — we want it to survive up to the script-level timeout.
    mcporter_timeout_ms = max(int(timeout) * 1000, 120_000)
    cmd = ["npx", "mcporter", "call", "suno.generate_song",
           "--config", config, "--timeout", str(mcporter_timeout_ms)]

    if instrumental:
        cmd.extend(["make_instrumental=true"])
    else:
        cmd.extend([f"lyrics={lyrics}"])

    cmd.extend([f"tags={tags}", f"title={title}"])
    
    print(f"Generating song: {title}...", file=sys.stderr)
    
    # Retry on transient upstream failures. Includes both non-zero exits
    # (HTTP 5xx, SSE errors, network hiccups) AND zero-exit responses that
    # report suno-side timeouts / no-songs-produced.
    last_err = ""
    for attempt in range(1, 6):
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.expanduser("~"))
        combined = (result.stderr or "") + (result.stdout or "")

        if result.returncode == 0:
            # Success at mcporter layer — check the response body for suno-side
            # transient errors (timeout / no songs).
            suno_retryable_body = any(s in combined for s in (
                "Song generation timed out",
                "no new songs appeared",
                "generation failed",
                "temporarily unavailable",
            ))
            # If we can parse at least one ID we consider it a real success
            # regardless of other error-adjacent text.
            if re.search(r"ID:\s*[a-f0-9-]+", combined) and not suno_retryable_body:
                break
            if not suno_retryable_body:
                # No IDs and no known-retryable marker — bail without retrying.
                break
            last_err = combined
        else:
            last_err = combined
            transient = any(s in combined for s in (
                "HTTP 50", "HTTP 429", "SSE error",
                "timed out", "ECONNRESET", "ETIMEDOUT", "EAI_AGAIN",
            ))
            if not transient:
                break

        backoff = min(60, 5 * (2 ** (attempt - 1)))  # 5, 10, 20, 40, 60s
        print(f"Suno transient issue (attempt {attempt}/5); retrying in {backoff}s",
              file=sys.stderr)
        time.sleep(backoff)

    if result.returncode != 0:
        raise RuntimeError(f"Generation failed: {last_err}")
    
    # Parse response. Suno MCP now returns plain text like:
    #   Song generation started!
    #   ID: <uuid>
    #   Preview: https://suno.com/song/<uuid>
    #   Download: https://cdn1.suno.ai/<uuid>.mp3
    # Handle both the legacy JSON form and the new plain-text form.
    response: dict
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Try line-by-line JSON first (legacy)
        response = None
        for line in result.stdout.strip().split('\n'):
            try:
                response = json.loads(line); break
            except json.JSONDecodeError:
                continue
        if response is None:
            # Plain-text fallback: extract IDs + direct CDN download URLs.
            id_matches = re.findall(r'ID:\s*([a-f0-9-]+)', result.stdout)
            dl_matches = re.findall(r'Download:\s*(https?://\S+)', result.stdout)
            if not id_matches:
                raise RuntimeError(f"Could not parse response: {result.stdout}")
            response = {
                'result': result.stdout,
                'id': id_matches[0],
                'all_ids': id_matches,
                'download_url': dl_matches[0] if dl_matches else None,
                'all_download_urls': dl_matches,
            }

    # Extract song_id(s) from response. Suno always returns 2 variants; if the
    # upstream JSON branch didn't populate all_ids/all_download_urls, scrape them
    # out of the 'result' text.
    song_id = response.get('id') or response.get('song_id')
    result_text = response.get('result', '') if isinstance(response.get('result'), str) else ''
    if 'all_ids' not in response and result_text:
        ids = re.findall(r'ID:\s*([a-f0-9-]+)', result_text)
        if ids:
            response['all_ids'] = ids
            if not song_id:
                song_id = ids[0]
    if 'all_download_urls' not in response and result_text:
        urls = re.findall(r'Download:\s*(https?://\S+)', result_text)
        if urls:
            response['all_download_urls'] = urls

    if not song_id:
        raise RuntimeError(f"No song_id in response: {response}")

    # Direct CDN URL is kept only as a fallback — we prefer the tailscale-proxied
    # MCP download_song URL below, which is stable and doesn't 403 flakily.
    direct_url = response.get('download_url')
    if not direct_url and 'result' in response:
        dm = re.search(r'Download:\s*(https?://\S+)', response['result'])
        if dm:
            direct_url = dm.group(1)

    # Suno always returns 2 variants. Download both if we got both IDs; the
    # primary ('local_file') is still the first one for backward compatibility,
    # and a 'local_files' list carries all downloaded variants.
    all_urls = response.get('all_download_urls') or ([direct_url] if direct_url else [])
    all_ids  = response.get('all_ids') or [song_id]
    local_files: list[str] = []

    def _fetch(url: str, dest: Path, tries: int, per_try_timeout: int = 30) -> bool:
        for _ in range(tries):
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; OpenClaw/1.0)'
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

    def _mcp_tailscale_url(vid: str) -> str | None:
        """Trigger MCP download_song and return the tailscale-reachable URL.

        The server returns a `Stream URL: <SUNO_PUBLIC_URL>/audio/<short>.mp3`
        line. We can't trust that URL verbatim (cross-tailnet agents can't
        reach it); instead we extract just the filename and reattach it to
        the agent's own MCP base URL — that way the same agent code works
        whether mcporter points at https://suno-mcp.tail9683c.ts.net/mcp
        or at a bridge URL like https://comfyui-bridge.tail74c072.ts.net:8190/mcp.
        """
        print(f"  Triggering MCP download for {vid}...", file=sys.stderr)
        dr = subprocess.run(
            ["npx", "mcporter", "call", "suno.download_song",
             f"song_id={vid}", "--config", config, "--timeout", "300000"],
            capture_output=True, text=True, cwd=os.path.expanduser("~"), timeout=300,
        )
        body = (dr.stdout or "") + (dr.stderr or "")
        # Parse `Downloaded to: /downloads/<filename>` to get the canonical
        # filename (e.g. "ef0722ad.mp3"), then build the audio URL from the
        # mcporter-configured suno base URL.
        m = re.search(r'Downloaded to:\s*\S*?([0-9a-f]{8}\.mp3)', body)
        if not m:
            # Fall back to the legacy `Stream URL:` / `Local URL:` shapes for
            # older suno-mcp builds. If neither matches, give up and let the
            # outer code fall back to the direct CDN URL.
            m2 = re.search(r'(?:Stream|Local) URL:\s*(\S+/audio/[0-9a-f]{8}\.mp3)', body)
            if not m2:
                print(f"  MCP call returned no Stream URL (rc={dr.returncode})", file=sys.stderr)
                return None
            print(f"  MCP URL: {m2.group(1)}", file=sys.stderr)
            return m2.group(1)
        filename = m.group(1)
        base = _suno_mcp_base_url(config)
        url = f"{base}/audio/{filename}" if base else None
        if url:
            print(f"  MCP URL: {url}", file=sys.stderr)
        else:
            print(f"  Could not derive base URL from mcporter config", file=sys.stderr)
        return url

    for i, vid in enumerate(all_ids):
        out = output_dir / f"suno_{vid}.mp3"
        direct = all_urls[i] if i < len(all_urls) else None
        print(f"Variant {i+1}/{len(all_ids)}: id={vid}", file=sys.stderr)

        # Prefer tailscale-proxied MCP URL (stable, no 403 flakiness).
        mcp_url = _mcp_tailscale_url(vid)
        ok = False
        if mcp_url:
            ok = _fetch(mcp_url, out, tries=40)
            if not ok:
                print(f"  MCP URL didn't yield file; falling back to direct CDN", file=sys.stderr)
        if not ok and direct:
            print(f"  Direct: {direct}", file=sys.stderr)
            ok = _fetch(direct, out, tries=40)
        if not ok:
            raise RuntimeError(f"Failed to download variant {i+1} (id={vid})")
        print(f"  Downloaded: {out.stat().st_size} bytes to {out}", file=sys.stderr)
        local_files.append(str(out))

    response['local_file'] = local_files[0]   # primary (backward compat)
    response['local_files'] = local_files     # all variants in order
    return response


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a song with Suno MCP")
    parser.add_argument("--lyrics", "-l", help="Song lyrics")
    parser.add_argument("--tags", "-t", required=True, help="Genre/style tags")
    parser.add_argument("--title", "-T", required=True, help="Song title")
    parser.add_argument("--instrumental", "-i", action="store_true", help="Instrumental track")
    parser.add_argument("--timeout", type=int, default=400,
                        help="Total timeout in seconds (default 400 — full pipeline is 3-5 min)")
    
    args = parser.parse_args()
    
    try:
        result = generate_song(
            lyrics=args.lyrics or "",
            tags=args.tags,
            title=args.title,
            instrumental=args.instrumental,
            timeout=args.timeout
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
