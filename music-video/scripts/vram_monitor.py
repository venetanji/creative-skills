#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Periodically poll ComfyUI's /system_stats and append one JSONL line per sample.

Usage:
  vram_monitor.py <log.jsonl> [--interval 10] [--until-file <path>]

  --until-file: stop when this file appears (e.g. project/final.mp4).
  --max-minutes N: stop after N minutes regardless.

Exits silently on stop. Safe to run alongside a long-running gen job.
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path


def sample(base: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base}/system_stats", timeout=5) as r:
            d = json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}
    devs = d.get("devices", [])
    if not devs:
        return {"raw": d}
    dev = devs[0]
    return {
        "ts": time.time(),
        "vram_total_mb": dev.get("vram_total", 0) // 1024 // 1024,
        "vram_free_mb": dev.get("vram_free", 0) // 1024 // 1024,
        "vram_used_mb": (dev.get("vram_total", 0) - dev.get("vram_free", 0)) // 1024 // 1024,
        "ram_free_mb": d.get("system", {}).get("ram_free", 0) // 1024 // 1024,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--base", default="https://comfyui.tail9683c.ts.net")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--until-file", default=None)
    p.add_argument("--max-minutes", type=float, default=None)
    args = p.parse_args()

    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    until_path = Path(args.until_file) if args.until_file else None
    while True:
        s = sample(args.base)
        if s is not None:
            with open(log, "a") as f:
                f.write(json.dumps(s) + "\n")
        if until_path and until_path.exists():
            break
        if args.max_minutes and (time.time() - started) > args.max_minutes * 60:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
