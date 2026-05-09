#!/bin/bash
# Overnight wrapper: VRAM monitor + full music-video pipeline + frame-stitch check.
# Usage: run_overnight.sh <project_dir>
set -e
proj="${1:?usage: run_overnight.sh <project_dir>}"
proj="$(realpath "$proj")"
cd "$proj"

# Resolve script bundle relative to this wrapper so the same script works
# in any install location (host, OpenClaw sandbox, or a fresh clone).
# Override SKILL to point elsewhere if you've installed scripts separately.
SKILL="${SKILL:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LOG="$proj/run_overnight.log"
VRAM="$proj/vram.jsonl"

echo "[$(date -Is)] starting overnight job for $proj" | tee -a "$LOG"
echo "[$(date -Is)] yaml: $proj/song.yaml"           | tee -a "$LOG"

# Start VRAM monitor in background; stops when final.mp4 exists or after 8h.
# Scripts are PEP 723 / uv-managed — invoke directly via their shebangs, not python3.
"$SKILL/vram_monitor.py" "$VRAM" --interval 15 \
  --until-file "$proj/final.mp4" --max-minutes 480 &
MON_PID=$!
echo "[$(date -Is)] vram monitor pid=$MON_PID → $VRAM" | tee -a "$LOG"

# Run the pipeline. Restart-safe — rerun to resume on partial completions.
if "$SKILL/music_video.py" all "$proj/song.yaml" 2>&1 | tee -a "$LOG"; then
    echo "[$(date -Is)] pipeline completed" | tee -a "$LOG"
else
    echo "[$(date -Is)] pipeline FAILED — check log" | tee -a "$LOG"
fi

# Stop monitor (files appeared or error).
kill $MON_PID 2>/dev/null || true

# Run stitch-analysis report.
if [ -f "$proj/final.mp4" ]; then
    echo "[$(date -Is)] frame-stitch check:" | tee -a "$LOG"
    "$SKILL/frame_check.py" "$proj" --save 2>&1 | tee -a "$LOG" || true
fi

# Quick summary
echo "[$(date -Is)] summary:" | tee -a "$LOG"
{
    echo "  final.mp4: $([ -f "$proj/final.mp4" ] && stat -c '%s bytes' "$proj/final.mp4" || echo 'missing')"
    echo "  scenes:    $(ls "$proj/scenes"/*.mp4 2>/dev/null | wc -l)"
    echo "  song.mp3:  $([ -f "$proj/song.mp3" ] && stat -c '%s bytes' "$proj/song.mp3" || echo 'missing')"
    if [ -f "$VRAM" ]; then
        peak=$(awk -F'"vram_used_mb":' '{print $2}' "$VRAM" | awk -F',' '{print $1}' | sort -n | tail -1)
        echo "  VRAM peak: ${peak} MB"
    fi
} | tee -a "$LOG"
