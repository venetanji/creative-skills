#!/bin/bash
# Generate a song with Suno MCP
# Usage: ./generate_song.sh "lyrics" "tags" "title" [instrumental]

LYRICS="$1"
TAGS="$2"
TITLE="$3"
INSTRUMENTAL="${4:-false}"
CONFIG="${CONFIG:-$HOME/.openclaw/config/mcporter.json}"

if [ -z "$LYRICS" ] && [ "$INSTRUMENTAL" != "true" ]; then
    echo "Error: Lyrics required (or set instrumental=true)"
    exit 1
fi

if [ "$INSTRUMENTAL" = "true" ]; then
    cd ~/.openclaw && npx mcporter call suno.generate_song \
        tags="$TAGS" \
        title="$TITLE" \
        make_instrumental=true \
        --config "$CONFIG"
else
    cd ~/.openclaw && npx mcporter call suno.generate_song \
        lyrics="$LYRICS" \
        tags="$TAGS" \
        title="$TITLE" \
        --config "$CONFIG"
fi
