#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["mido>=1.3"]
# ///
"""Identify scene-boundary times for a music video by combining three
independent signals:

1. **MIDI bar grid** — bar-aligned downbeats (from `midi/drums.mid` tempo
   map). Scene cuts that land on a bar feel musical.
2. **MIDI vocal-phrase boundaries** — gaps between vocal notes in
   `midi/vocals.mid`. High-signal for "end of a line / pause" moments.
3. **Whisper STT line starts** — spoken-line start times from a Whisper
   transcription of `stems/vocals.wav` (SRT or JSON). Catches lyric
   boundaries that the midi quantizer may miss.

Outputs a ranked list of candidate scene start times and, optionally,
emits a ready-to-drop `scenes:` block for `song.yaml`.

Usage:
    analyze_song.py <project_dir> \
        [--whisper-srt <path>] [--bar-length 4] [--scenes 14]
        [--min-spacing 8] [--json]

Inputs expected under <project_dir>:
    midi/*.mid              from `unpack_suno_pack.py`
    stems/vocals.wav        (for reference; this script does NOT run
                             Whisper itself — pass the pre-run SRT via
                             --whisper-srt, or use
                             `comfy_graph.py stt --audio stems/vocals.wav`
                             then point at the resulting .srt)

Scoring heuristic:
    A candidate time T scores based on how closely it aligns with each
    of the 3 signals. Weights (default): bar=1.0, vocal-phrase=2.0,
    whisper-line=2.5. A time within 0.25s of a signal gets its full
    weight; the score decays linearly to 0 at 1.5s away.

The top-N highest-scoring times that respect `--min-spacing` are returned
as scene boundaries.
"""
import argparse
import json
import re
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
    sys.exit("mido missing. Re-run via `uv run --script` (auto-installs).")

# Suno's midi exports occasionally emit key-signature meta events outside
# the standard MIDI spec (e.g. 16 sharps, mode 0), raising
# `mido.KeySignatureError` on load. Pre-populate the decode table with safe
# fallbacks so loading is robust; the bar/tempo data we use is unaffected.
try:
    from mido.midifiles.meta import _key_signature_decode  # type: ignore
    for _n in range(-128, 128):
        for _mode in (0, 1):
            if (_n, _mode) not in _key_signature_decode:
                _key_signature_decode[(_n, _mode)] = "C"
except Exception:
    pass

# --- reused helpers from analyze_midi.py (kept local to avoid circular import) ---

PHRASE_GAP_SEC = 0.5


def analyze_tempo(mid):
    events = []
    ticks_per_beat = mid.ticks_per_beat
    cur_tempo = 500_000
    cur_time = 0.0
    for msg in mido.merge_tracks(mid.tracks):
        cur_time += mido.tick2second(msg.time, ticks_per_beat, cur_tempo)
        if msg.type == "set_tempo":
            events.append((cur_time, msg.tempo))
            cur_tempo = msg.tempo
    if not events:
        events = [(0.0, 500_000)]
    return events


def note_on_times(mid):
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


def build_bar_grid(tempo_events, total_sec, bar_length):
    bars = [0.0]
    t, i = 0.0, 0
    while t < total_sec:
        while i + 1 < len(tempo_events) and tempo_events[i + 1][0] <= t:
            i += 1
        sec_per_bar = (tempo_events[i][1] / 1_000_000.0) * bar_length
        t += sec_per_bar
        bars.append(round(t, 3))
    return bars


def phrase_boundaries(note_times, gap_sec=PHRASE_GAP_SEC):
    if not note_times:
        return []
    out = [note_times[0]]
    for prev, cur in zip(note_times, note_times[1:]):
        if cur - prev >= gap_sec:
            out.append(cur)
    return out


# --- whisper SRT parser ---

SRT_TIME = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def parse_srt(path: Path) -> list[dict]:
    """Returns [{start, end, text}, ...] in seconds. Accepts .srt or .vtt."""
    raw = path.read_text()
    blocks = []
    for m in SRT_TIME.finditer(raw):
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        # text lines between this timestamp and the next
        after = raw[m.end():]
        next_m = SRT_TIME.search(after)
        body = (after[:next_m.start()] if next_m else after).strip()
        body = re.sub(r"^\s*\d+\s*\n", "", body)  # strip index line if any
        blocks.append({"start": round(start, 3), "end": round(end, 3),
                       "text": body.splitlines()[0][:80] if body else ""})
    return blocks


# --- scoring ---

def proximity_score(t: float, targets: list[float],
                    full_window: float = 0.25,
                    zero_window: float = 1.5) -> float:
    """1.0 if any target within full_window of t; linearly decays to 0 at
    zero_window. Outside zero_window = 0."""
    if not targets:
        return 0.0
    delta = min(abs(t - x) for x in targets)
    if delta <= full_window:
        return 1.0
    if delta >= zero_window:
        return 0.0
    return 1.0 - (delta - full_window) / (zero_window - full_window)


def pick_scenes(candidates: list[tuple[float, float]], n: int,
                min_spacing: float) -> list[tuple[float, float]]:
    """Greedy: sort by score desc, pick top while respecting min_spacing."""
    ordered = sorted(candidates, key=lambda x: -x[1])
    picked: list[tuple[float, float]] = []
    for t, score in ordered:
        if all(abs(t - p[0]) >= min_spacing for p in picked):
            picked.append((t, score))
        if len(picked) == n:
            break
    picked.sort(key=lambda x: x[0])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--whisper-srt", type=Path, default=None,
                    help="Whisper .srt/.vtt for stems/vocals.wav (optional)")
    ap.add_argument("--bar-length", type=int, default=4)
    ap.add_argument("--scenes", type=int, default=14,
                    help="Number of scene boundaries to return (incl. start at 0)")
    ap.add_argument("--min-spacing", type=float, default=8.0,
                    help="Minimum seconds between consecutive scene starts")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    midi_dir = args.project / "midi"
    if not midi_dir.is_dir():
        sys.exit(f"no midi/ dir in {args.project}")

    # --- signal 1: bar grid ---
    drums_path = midi_dir / "drums.mid"
    tempo_source = drums_path if drums_path.exists() else next(midi_dir.glob("*.mid"))
    tempo_events = analyze_tempo(mido.MidiFile(tempo_source))
    bpm = round(60_000_000 / tempo_events[0][1], 2)

    # total length = max stem length
    total_sec = 0.0
    for p in midi_dir.glob("*.mid"):
        total_sec = max(total_sec, mido.MidiFile(p).length)

    bars = build_bar_grid(tempo_events, total_sec, args.bar_length)

    # --- signal 2: vocal phrase boundaries ---
    vocals_path = midi_dir / "vocals.mid"
    vocal_phrases = []
    if vocals_path.exists():
        vocal_phrases = phrase_boundaries(note_on_times(mido.MidiFile(vocals_path)))

    # --- signal 3: whisper line starts ---
    whisper_starts = []
    whisper_blocks = []
    if args.whisper_srt and args.whisper_srt.exists():
        whisper_blocks = parse_srt(args.whisper_srt)
        whisper_starts = [b["start"] for b in whisper_blocks]

    # --- score every bar as a candidate ---
    W_BAR = 1.0
    W_VOCAL = 2.0
    W_WHISPER = 2.5
    candidates = []
    for t in bars:
        if t >= total_sec:
            break
        score = (
            W_BAR * proximity_score(t, bars, full_window=0.1, zero_window=0.5) +
            W_VOCAL * proximity_score(t, vocal_phrases) +
            (W_WHISPER * proximity_score(t, whisper_starts) if whisper_starts else 0)
        )
        candidates.append((round(t, 3), round(score, 3)))

    # always include t=0 as a mandatory start
    picks = pick_scenes(candidates, args.scenes, args.min_spacing)
    if not any(abs(p[0]) < 1.0 for p in picks):
        picks = [(0.0, candidates[0][1] if candidates else 0.0)] + picks[:-1]
        picks.sort(key=lambda x: x[0])

    # durations between picks + tail
    durations = [round(b - a, 3) for a, b in zip([p[0] for p in picks], [p[0] for p in picks[1:]])]
    durations.append(round(total_sec - picks[-1][0], 3))

    result = {
        "song_length_sec": round(total_sec, 3),
        "bpm": bpm,
        "n_bars": len(bars),
        "whisper_lines_found": len(whisper_starts),
        "vocal_phrases_found": len(vocal_phrases),
        "scenes": [
            {
                "index": i + 1,
                "start_sec": p[0],
                "duration_sec": durations[i],
                "score": p[1],
                "lyric_hint": next(
                    (b["text"] for b in whisper_blocks if abs(b["start"] - p[0]) < 1.0),
                    None,
                ),
            }
            for i, p in enumerate(picks)
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"song: {result['song_length_sec']:.2f}s @ {bpm} bpm, {len(bars)} bars")
    print(f"signals: bars=✓  vocal_phrases={len(vocal_phrases)}  "
          f"whisper_lines={len(whisper_starts)}")
    if not whisper_starts:
        print("  ⚠  no whisper transcript — run:")
        print(f"     python3 /home/sandbox/.openclaw/skills/comfyui/scripts/comfy_graph.py stt "
              f"--audio {args.project}/stems/vocals.wav --output-dir {args.project}/stt/")
        print(f"     then rerun with --whisper-srt {args.project}/stt/transcript.srt")
    print()
    print("suggested scene timeline:")
    print(f"  {'#':>3}  {'start':>7}  {'dur':>6}  {'score':>5}  lyric")
    for s in result["scenes"]:
        hint = (s["lyric_hint"] or "").strip()[:50]
        print(f"  {s['index']:>3}  {s['start_sec']:>7.2f}  {s['duration_sec']:>6.2f}  "
              f"{s['score']:>5.2f}  {hint}")


if __name__ == "__main__":
    main()
