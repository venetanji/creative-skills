#!/bin/bash
# Generate a song with Suno MCP and download from HTTP endpoint
# Usage: ./generate_and_download.sh "lyrics" "tags" "title" [instrumental]

LYRICS="$1"
TAGS="$2"
TITLE="$3"
INSTRUMENTAL="${4:-false}"
CONFIG="${CONFIG:-$HOME/.openclaw/config/mcporter.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.openclaw/workspace/outputs}"
MCP_AUDIO_URL="http://suno-mcp.tail9683c.ts.net:8085/audio"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Generate the song
echo "Generating song: $TITLE..." >&2

if [ "$INSTRUMENTAL" = "true" ]; then
    RESULT=$(cd ~/.openclaw && npx mcporter call suno.generate_song \
        tags="$TAGS" \
        title="$TITLE" \
        make_instrumental=true \
        --config "$CONFIG" 2>&1)
else
    RESULT=$(cd ~/.openclaw && npx mcporter call suno.generate_song \
        lyrics="$LYRICS" \
        tags="$TAGS" \
        title="$TITLE" \
        --config "$CONFIG" 2>&1)
fi

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error generating song: $RESULT"
    exit 1
fi

# Extract song_id from result (handles various JSON formats)
SONG_ID=$(echo "$RESULT" | grep -oP '"id"\s*:\s*"\K[^"]+' | head -1)
if [ -z "$SONG_ID" ]; then
    SONG_ID=$(echo "$RESULT" | grep -oP '"song_id"\s*:\s*"\K[^"]+' | head -1)
fi

if [ -z "$SONG_ID" ]; then
    echo "Error: Could not extract song_id from response"
    echo "Response: $RESULT"
    exit 1
fi

echo "Song generated: $SONG_ID" >&2

# Trigger server-side download via MCP (can take 2-3 minutes)
echo "Triggering server download (this may take 2-3 minutes)..." >&2
DOWNLOAD_RESULT=$(cd ~/.openclaw && npx mcporter call suno.download_song \
    song_id="$SONG_ID" \
    --config "$CONFIG" \
    --timeout 300000 2>&1)

# Extract Local URL from download response
# Format: "Local URL: http://0.0.0.0:8085/audio/XXXX.mp3"
LOCAL_URL=$(echo "$DOWNLOAD_RESULT" | grep -oP 'Local URL:\s*\Khttp://[^[:space:]]+' | head -1)

if [ -n "$LOCAL_URL" ]; then
    # Replace 0.0.0.0 with actual hostname
    AUDIO_URL=$(echo "$LOCAL_URL" | sed 's/0\.0\.0\.0/suno-mcp.tail9683c.ts.net/')
    echo "Found MCP URL: $AUDIO_URL" >&2
else
    # Fallback: construct URL from song_id
    AUDIO_URL="$MCP_AUDIO_URL/${SONG_ID}.mp3"
fi
OUTPUT_FILE="$OUTPUT_DIR/suno_${SONG_ID}.mp3"

echo "Waiting for file at $AUDIO_URL..." >&2

MAX_WAIT=120
WAITED=0
POLL_INTERVAL=3

while [ $WAITED -lt $MAX_WAIT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$AUDIO_URL" 2>/dev/null)
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "File ready! Downloading..." >&2
        break
    fi
    
    echo "File not ready yet... (${WAITED}s)" >&2
    sleep $POLL_INTERVAL
    WAITED=$((WAITED + POLL_INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "Error: Timeout waiting for file after ${MAX_WAIT}s"
    exit 1
fi

# Download the file from MCP HTTP endpoint
echo "Downloading to $OUTPUT_FILE..." >&2
curl -sL "$AUDIO_URL" -o "$OUTPUT_FILE"

if [ $? -ne 0 ] || [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: Failed to download song from $AUDIO_URL"
    exit 1
fi

# Verify file is valid MP3 (check magic bytes)
FILE_HEADER=$(xxd -l 2 "$OUTPUT_FILE" 2>/dev/null | head -1)
if [[ ! "$FILE_HEADER" =~ "fffb" ]] && [[ ! "$FILE_HEADER" =~ "4944" ]]; then
    echo "Warning: Downloaded file may not be a valid MP3" >&2
fi

FILE_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
echo "Downloaded: $FILE_SIZE bytes" >&2

# Return the result with file path
if command -v jq &> /dev/null; then
    # If jq is available, merge file path into result
    echo "$RESULT" | jq --arg path "$OUTPUT_FILE" '. + {local_file: $path}'
else
    # Simple concatenation if jq not available
    echo "$RESULT"
    echo ""
    echo "LOCAL_FILE: $OUTPUT_FILE"
fi
