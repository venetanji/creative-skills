#!/usr/bin/env python3
"""
Download a song from the Suno MCP server HTTP endpoint.
Polls until the file is ready, then downloads it.
"""

import subprocess
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

def download_song(song_id, output_dir=None, max_wait=120):
    """
    Download a song from the MCP audio endpoint.
    
    Args:
        song_id: The Suno song ID
        output_dir: Where to save the file (defaults to workspace/outputs)
        max_wait: Max seconds to wait for file availability
    
    Returns:
        Path to the downloaded file
    """
    if output_dir is None:
        output_dir = Path(os.path.expanduser("~/.openclaw/workspace/outputs"))
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"suno_{song_id}.mp3"
    
    config = os.path.expanduser("~/.openclaw/config/mcporter.json")
    
    # First trigger the server-side download via MCP
    print(f"Triggering server download for {song_id}...", file=sys.stderr)
    
    cmd = [
        "mcporter", "call", "suno.download_song",
        f"song_id={song_id}",
        "--config", config
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.expanduser("~/.openclaw"),
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"Download trigger warning: {result.stderr}", file=sys.stderr)
        # Continue anyway - file might already be available
    
    # Poll for file availability at MCP server
    mcp_audio_url = f"http://suno-mcp.tail9683c.ts.net:8085/audio/{song_id}.mp3"
    
    print(f"Waiting for file at {mcp_audio_url}...", file=sys.stderr)
    
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
    
    return str(output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download a song from Suno MCP")
    parser.add_argument("song_id", help="Suno song ID")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--max-wait", "-w", type=int, default=120, help="Max seconds to wait")
    
    args = parser.parse_args()
    
    try:
        result = download_song(args.song_id, args.output, args.max_wait)
        print(json.dumps({"local_file": result, "song_id": args.song_id}))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
