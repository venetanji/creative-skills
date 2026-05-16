#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mido>=1.3"]
# ///
"""Analyze a Suno stem-pack midi set. Reports tempo map, bar grid, and
phrase boundaries per stem. Output is either a human table or JSON for
`music_video.py assemble` to snap `start_sec` values to musical bars.

Usage:
    analyze_midi.py <project_dir> [--bar-length 4] [--json]
        --bar-length   beats-per-bar (default 4)
        --json         emit machine-readable JSON instead of a table

Conventions:
    Suno's pack structure: <project>/midi/{vocals,drums,bass,guitar,keyboard,
    fx,percussion,backing_vocals}.mid

The script:
 1. Reads the tempo map from the first stem that has `set_tempo` events
    (usually drums).
 2. Builds a bar grid at `bar_length` beats/bar → list of bar-start
    times in seconds.
 3. For each stem, converts note-on events to seconds (relative to the
    piece start) and detects **phrase boundaries** — gaps longer than
    0.5s between consecutive notes.
 4. The vocals stem's phrase boundaries are the highest-signal transition
    anchors; other stems' boundaries are supplementary.

Use output to:
 - Snap scene `start_sec` to nearest bar → keeps cuts on the downbeat.
 - Place transitions at vocal-phrase gaps → morphs land during breath /
   line breaks rather than mid-word.
 - Decide camera motion per musical section (the `section` field is a
   heuristic: 0-25% = intro, 25-60% = verse, 60-85% = chorus, ... adjust
   manually for your song).
"""
import argparse
import json
import sys
from pathlib import Path

# Windows cp1252 default stdout can't print → ✓ ⚠ glyphs. Reconfigure to
# utf-8 at module load so this script's status output is portable.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    import mido
except ImportError:
    sys.exit("mido missing. Re-run via `uv run --script` (auto-installs) "
             "or `pip install --user mido`.")

# Suno's midi exports occasionally emit key-signature meta events with
# values outside the standard MIDI spec (e.g. 16 sharps, mode 0), which
# raises `mido.KeySignatureError` and prevents the file from loading.
# Pre-populate mido's decode table with safe fallbacks (default to "C")
# so loading is robust; the bar/tempo data we actually use is unaffected.
try:
    from mido.midifiles.meta import _key_signature_decode  # type: ignore
    for _n in range(-128, 128):
        for _mode in (0, 1):
            if (_n, _mode) not in _key_signature_decode:
                _key_signature_decode[(_n, _mode)] = "C"
except Exception:
    pass

PHRASE_GAP_SEC = 0.5


def analyze_tempo(mid: mido.MidiFile) -> list[tuple[float, int]]:
    """Return [(time_sec, tempo_us_per_qn), ...] in order. If no tempo events,
    returns the midi default (500000 = 120 bpm)."""
    events = []
    # mido exposes tempo changes pre-merged across tracks; iterate the merged
    # stream and track cumulative time in seconds via mid.ticks_per_beat and
    # the most-recent tempo.
    ticks_per_beat = mid.ticks_per_beat
    cur_tempo = 500_000
    cur_time = 0.0
    for msg in mido.merge_tracks(mid.tracks):
        # delta ticks → seconds using current tempo
        dt = mido.tick2second(msg.time, ticks_per_beat, cur_tempo)
        cur_time += dt
        if msg.type == "set_tempo":
            events.append((cur_time, msg.tempo))
            cur_tempo = msg.tempo
    if not events:
        events = [(0.0, 500_000)]
    return events


def note_on_times(mid: mido.MidiFile) -> list[float]:
    """Absolute times (s) of note_on events (velocity > 0)."""
    ticks_per_beat = mid.ticks_per_beat
    cur_tempo = 500_000
    cur_time = 0.0
    times = []
    for msg in mido.merge_tracks(mid.tracks):
        cur_time += mido.tick2second(msg.time, ticks_per_beat, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
            continue
        if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
            times.append(cur_time)
    return times


def build_bar_grid(tempo_events: list[tuple[float, int]], total_sec: float,
                   bar_length: int = 4) -> list[float]:
    """bar_length is beats-per-bar; returns bar-start timestamps in seconds."""
    bars = [0.0]
    t = 0.0
    i = 0
    # tempo_events is [(t_start, us_per_qn)]; walk forward
    while t < total_sec:
        # find applicable tempo
        while i + 1 < len(tempo_events) and tempo_events[i + 1][0] <= t:
            i += 1
        us_per_qn = tempo_events[i][1]
        sec_per_bar = (us_per_qn / 1_000_000.0) * bar_length
        t += sec_per_bar
        bars.append(round(t, 3))
    return bars


def phrase_boundaries(note_times: list[float], gap_sec: float = PHRASE_GAP_SEC
                      ) -> list[float]:
    """Timestamps where a gap ≥ gap_sec precedes the next note."""
    if not note_times:
        return []
    out = [note_times[0]]
    for prev, cur in zip(note_times, note_times[1:]):
        if cur - prev >= gap_sec:
            out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--bar-length", type=int, default=4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    midi_dir = args.project / "midi"
    if not midi_dir.is_dir():
        sys.exit(f"no midi/ dir in {args.project}")

    stems = {}
    total_sec = 0.0
    tempo_events = None
    for path in sorted(midi_dir.glob("*.mid")):
        mid = mido.MidiFile(path)
        length = mid.length
        total_sec = max(total_sec, length)
        notes = note_on_times(mid)
        if tempo_events is None or path.stem == "drums":
            # Prefer drums for the authoritative tempo map, fall back to first
            tempo_events = analyze_tempo(mid)
        stems[path.stem] = {
            "length_sec": round(length, 3),
            "n_notes": len(notes),
            "phrases_sec": [round(t, 3) for t in phrase_boundaries(notes)],
        }

    bars = build_bar_grid(tempo_events or [(0.0, 500_000)], total_sec,
                          args.bar_length)
    # Compute BPM from the first tempo event
    us_per_qn = (tempo_events or [(0, 500_000)])[0][1]
    bpm = 60_000_000 / us_per_qn

    result = {
        "song_length_sec": round(total_sec, 3),
        "bpm": round(bpm, 2),
        "bar_length_beats": args.bar_length,
        "sec_per_bar": round((us_per_qn / 1e6) * args.bar_length, 4),
        "bar_starts_sec": bars,
        "n_bars": len(bars),
        "stems": stems,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"song length:  {result['song_length_sec']:.2f}s")
    print(f"bpm:          {result['bpm']:.1f}  (→ {result['sec_per_bar']:.3f}s/bar at {args.bar_length}/4)")
    print(f"bars:         {result['n_bars']} (first 8 downbeats: {bars[:8]})")
    print()
    print("=== phrase boundaries per stem (first 8) ===")
    for name, info in stems.items():
        phrases = info["phrases_sec"]
        marker = "★" if name == "vocals" else " "
        print(f"  {marker} {name:16}  notes={info['n_notes']:>4}  phrases={len(phrases):>3}  first={phrases[:8]}")

    print()
    print("Transition suggestion: snap scene starts to `bar_starts_sec`; "
          "place transitions at vocals phrase boundaries where possible.")


if __name__ == "__main__":
    main()
