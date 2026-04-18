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

    # Extract song_id from response
    song_id = response.get('id') or response.get('song_id')
    if not song_id and 'result' in response:
        id_match = re.search(r'ID:\s*([a-f0-9-]+)', response['result'])
        if id_match:
            song_id = id_match.group(1)

    if not song_id:
        raise RuntimeError(f"No song_id in response: {response}")

    # Fast path: if the response gave us a direct CDN download URL, just fetch it
    # (skips the slower MCP download_song round-trip).
    direct_url = response.get('download_url')
    if not direct_url and 'result' in response:
        dm = re.search(r'Download:\s*(https?://\S+)', response['result'])
        if dm:
            direct_url = dm.group(1)

    if direct_url:
        # Suno always returns 2 variants. Download both if we got both URLs; the
        # primary ('local_file') is still the first one for backward compatibility,
        # and a 'local_files' list carries all downloaded variants.
        all_urls = response.get('all_download_urls') or [direct_url]
        all_ids  = response.get('all_ids') or [song_id]
        local_files: list[str] = []

        def _fetch(url: str, dest: Path) -> bool:
            for attempt in range(40):
                try:
                    req = urllib.request.Request(url, headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; OpenClaw/1.0)'
                    })
                    with urllib.request.urlopen(req, timeout=30) as r:
                        if r.status == 200:
                            dest.write_bytes(r.read())
                            return True
                except urllib.error.HTTPError as e:
                    if e.code not in (403, 404):
                        raise
                except Exception:
                    pass
                time.sleep(5)
            return False

        for i, (vid, vurl) in enumerate(zip(all_ids, all_urls)):
            out = output_dir / f"suno_{vid}.mp3"
            print(f"Variant {i+1}/{len(all_urls)}: {vurl}", file=sys.stderr)
            if not _fetch(vurl, out):
                raise RuntimeError(f"Timed out fetching variant {i+1}: {vurl}")
            print(f"  Downloaded: {out.stat().st_size} bytes to {out}", file=sys.stderr)
            local_files.append(str(out))

        response['local_file'] = local_files[0]   # primary (backward compat)
        response['local_files'] = local_files     # all variants in order
        return response
    
    print(f"Song generated: {song_id}", file=sys.stderr)
    
    # Call MCP download_song to trigger server-side download
    # This can take 2-3 minutes as it downloads from Suno
    print(f"Triggering server download (this may take 2-3 minutes)...", file=sys.stderr)
    download_cmd = [
        "npx", "mcporter", "call", "suno.download_song",
        f"song_id={song_id}",
        "--config", config,
        "--timeout", "300000"  # 5 minutes in ms
    ]
    
    download_result = subprocess.run(
        download_cmd,
        capture_output=True,
        text=True,
        cwd=os.path.expanduser("~"),
        timeout=300  # 5 minutes
    )
    
    # Parse download response to get the local URL
    try:
        download_response = json.loads(download_result.stdout)
        # Extract Local URL from response like: "Local URL: http://0.0.0.0:8085/audio/36e9e99c.mp3"
        result_text = download_response.get('result', '')
        url_match = re.search(r'Local URL:\s*(http://[^\s]+)', result_text)
        if url_match:
            mcp_audio_url = url_match.group(1).replace('0.0.0.0', 'suno-mcp.tail9683c.ts.net')
            print(f"Found MCP URL: {mcp_audio_url}", file=sys.stderr)
        else:
            # Fallback: construct URL from song_id (may use short ID)
            mcp_audio_url = f"http://suno-mcp.tail9683c.ts.net:8085/audio/{song_id}.mp3"
    except Exception as e:
        print(f"Could not parse download response: {e}", file=sys.stderr)
        mcp_audio_url = f"http://suno-mcp.tail9683c.ts.net:8085/audio/{song_id}.mp3"
    
    if download_result.returncode != 0:
        print(f"Download trigger warning: {download_result.stderr}", file=sys.stderr)
        # Continue anyway - file might still be available
    output_file = output_dir / f"suno_{song_id}.mp3"
    
    print(f"Waiting for file at {mcp_audio_url}...", file=sys.stderr)
    
    max_wait = 120  # seconds
    poll_interval = 3  # seconds
    waited = 0
    
    while waited < max_wait:
        try:
            req = urllib.request.Request(mcp_audio_url, method='HEAD')
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    print(f"File ready! Downloading...", file=sys.stderr)
                    break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"File not ready yet... ({waited}s)", file=sys.stderr)
            else:
                print(f"HTTP error {e.code}, retrying...", file=sys.stderr)
        except Exception as e:
            print(f"Connection error: {e}, retrying...", file=sys.stderr)
        
        time.sleep(poll_interval)
        waited += poll_interval
    else:
        raise RuntimeError(f"Timeout waiting for file after {max_wait}s")
    
    # Download the file
    print(f"Downloading from {mcp_audio_url}...", file=sys.stderr)
    req = urllib.request.Request(mcp_audio_url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; OpenClaw/1.0)'
    })
    
    with urllib.request.urlopen(req, timeout=60) as response_obj:
        with open(output_file, 'wb') as f:
            f.write(response_obj.read())
    
    if not output_file.exists():
        raise RuntimeError("Download failed - file not created")
    
    file_size = output_file.stat().st_size
    print(f"Downloaded: {file_size} bytes to {output_file}", file=sys.stderr)
    
    # Add local file to response
    if isinstance(response, dict):
        response['local_file'] = str(output_file)
        return response
    else:
        # If response is not a dict, create a new response
        return {
            'result': response if isinstance(response, str) else str(response),
            'local_file': str(output_file),
            'song_id': song_id
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a song with Suno MCP")
    parser.add_argument("--lyrics", "-l", help="Song lyrics")
    parser.add_argument("--tags", "-t", required=True, help="Genre/style tags")
    parser.add_argument("--title", "-T", required=True, help="Song title")
    parser.add_argument("--instrumental", "-i", action="store_true", help="Instrumental track")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    
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
