#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "librosa>=0.10",
#   "mido>=1.3",
#   "numpy",
# ]
# ///
"""Extract pitch+onset notes from a (vocal) audio stem and emit a MIDI
file that's TIME-ALIGNED to the audio.

Why: Suno's stem-pack MIDI is a hint — its tempo curve doesn't perfectly
match the audio. midi2canny.py / tunnel_3d.py drive cond videos from MIDI
timestamps. When MIDI is offset/stretched from audio, the cond's
pitch-driven effects (spin, pulse) land at the wrong moment in the
rendered video. This script replaces the Suno MIDI with one extracted
directly from the audio so timestamps lock to the song.

Pipeline:
  1. librosa.load(audio, sr=22050, mono)
  2. librosa.pyin(...) → f0 contour + voiced flag (no spurious unvoiced
     pitch values)
  3. Group consecutive voiced frames where Δf0 < pitch_step_threshold
     and gap < gap_threshold → "note" objects with start/duration/pitch
  4. Velocity = local RMS energy of the audio in the note's window
  5. Write a MIDI file with ticks_per_beat=480, tempo=500000us/beat
     (120 BPM) so downstream tick-to-second math is trivial.

Usage:
  audio_to_midi.py --input <vocal-stem.wav> --output <vocals_audio.mid>
                   [--fmin 80] [--fmax 1000]
                   [--frame-hop 512] [--min-note-dur 0.08]
                   [--note-gap 0.10]

For non-vocal stems (drums kit, etc.) onset-only detection (no pitch)
is more useful — that's a different script. This one focuses on
pitched vocals.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import librosa
import mido
import numpy as np


def f0_to_midi(f0_hz: float) -> int:
    """Hz → nearest MIDI note number. 440 Hz = MIDI 69 (A4)."""
    if f0_hz <= 0 or not np.isfinite(f0_hz):
        return 0
    return int(round(69 + 12 * np.log2(f0_hz / 440.0)))


def extract_notes(y: np.ndarray, sr: int, fmin: float, fmax: float,
                  frame_hop: int, min_note_dur: float, note_gap: float,
                  pitch_step_threshold: float = 1.0) -> list[dict]:
    """Run pyin on `y` and group voiced frames into note objects.

    pitch_step_threshold: max ΔMIDI between consecutive frames within
    the same note. 1.0 = same note (within a half-step). Larger value
    merges glides into one note.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048,
        hop_length=frame_hop, fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=frame_hop)
    midi_per_frame = np.array([f0_to_midi(f) if v and np.isfinite(f) else 0
                                for f, v in zip(f0, voiced_flag)])
    # RMS for velocity
    rms = librosa.feature.rms(y=y, frame_length=2048,
                               hop_length=frame_hop)[0]
    # Align rms length with f0
    rms = rms[:len(midi_per_frame)]
    if len(rms) < len(midi_per_frame):
        rms = np.pad(rms, (0, len(midi_per_frame) - len(rms)))
    rms_max = float(np.percentile(rms[midi_per_frame > 0], 95)) if (midi_per_frame > 0).any() else 1.0
    if rms_max <= 0:
        rms_max = 1.0

    notes: list[dict] = []
    cur_start = None
    cur_midi = None
    cur_rms_sum = 0.0
    cur_count = 0
    last_voiced_t = -1e9

    for i, (t, midi, r) in enumerate(zip(times, midi_per_frame, rms)):
        if midi == 0:
            # unvoiced frame — close out current note if gap exceeded
            if cur_start is not None and (t - last_voiced_t) > note_gap:
                _emit(notes, cur_start, last_voiced_t, cur_midi,
                      cur_rms_sum, cur_count, rms_max, min_note_dur)
                cur_start = cur_midi = None
                cur_rms_sum = 0.0
                cur_count = 0
            continue
        if cur_start is None:
            cur_start = t
            cur_midi = midi
            cur_rms_sum = float(r)
            cur_count = 1
            last_voiced_t = t
            continue
        # Pitch step too large → new note
        if abs(midi - cur_midi) > pitch_step_threshold:
            _emit(notes, cur_start, last_voiced_t, cur_midi,
                  cur_rms_sum, cur_count, rms_max, min_note_dur)
            cur_start = t
            cur_midi = midi
            cur_rms_sum = float(r)
            cur_count = 1
        else:
            # Keep extending the current note. Use median midi to be
            # robust to single-frame pitch wobble.
            cur_midi = int(np.median([cur_midi, midi]))
            cur_rms_sum += float(r)
            cur_count += 1
        last_voiced_t = t

    if cur_start is not None:
        _emit(notes, cur_start, last_voiced_t, cur_midi,
              cur_rms_sum, cur_count, rms_max, min_note_dur)

    return notes


def _emit(notes, start, end, midi, rms_sum, count, rms_max, min_dur):
    if midi <= 0 or count <= 0:
        return
    dur = end - start
    if dur < min_dur:
        return
    mean_rms = rms_sum / count
    velocity_norm = min(1.0, max(0.05, mean_rms / rms_max))
    notes.append({
        "start": float(start),
        "duration": float(dur),
        "midi": int(midi),
        "velocity_norm": float(velocity_norm),
    })


def write_midi(notes: list[dict], output: Path, ticks_per_beat: int = 480,
               tempo_us_per_beat: int = 500000):
    """Write notes to a single-track MIDI file. tempo fixed at 120 BPM
    (500000 us/beat) so downstream readers don't get confused by
    tempo-map drift; tick-to-second math is then a simple constant."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us_per_beat,
                                   time=0))

    seconds_per_tick = (tempo_us_per_beat / 1_000_000) / ticks_per_beat

    # Build event stream (note_on, note_off) sorted by absolute seconds.
    events = []
    for n in notes:
        events.append((n["start"], "on", n))
        events.append((n["start"] + n["duration"], "off", n))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    last_tick = 0
    for t, kind, n in events:
        abs_tick = int(round(t / seconds_per_tick))
        delta = max(0, abs_tick - last_tick)
        last_tick = abs_tick
        vel = int(round(n["velocity_norm"] * 127))
        vel = max(1, min(127, vel))
        track.append(mido.Message(
            "note_on" if kind == "on" else "note_off",
            note=int(n["midi"]),
            velocity=vel if kind == "on" else 0,
            time=delta,
        ))
    mid.save(str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Audio stem path (wav/mp3/flac). Use the isolated "
                        "vocal stem for best pitch detection.")
    p.add_argument("--output", required=True, help="Output .mid path.")
    p.add_argument("--sr", type=int, default=22050,
                   help="Resample rate (default 22050).")
    p.add_argument("--fmin", type=float, default=80.0,
                   help="Min pitch in Hz (default 80, ~E2).")
    p.add_argument("--fmax", type=float, default=1000.0,
                   help="Max pitch in Hz (default 1000, ~B5).")
    p.add_argument("--frame-hop", type=int, default=512,
                   help="pyin hop length in samples (default 512 = "
                        "~23ms at 22050 sr — good for vocal phrasing).")
    p.add_argument("--min-note-dur", type=float, default=0.08,
                   help="Drop notes shorter than this (s). Default 0.08.")
    p.add_argument("--note-gap", type=float, default=0.10,
                   help="Unvoiced gap > this ends a note (s). Default 0.10.")
    p.add_argument("--pitch-step", type=float, default=1.0,
                   help="Pitch step (semitones) within a note. Default 1.0 "
                        "= within a half-step is the same note.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    if not args.quiet:
        print(f"loading {in_path} ...", file=sys.stderr)
    y, sr = librosa.load(str(in_path), sr=args.sr, mono=True)
    if not args.quiet:
        print(f"loaded {len(y)/sr:.2f}s @ {sr}Hz, "
              f"running pyin (fmin={args.fmin}, fmax={args.fmax}) ...",
              file=sys.stderr)
    notes = extract_notes(
        y, sr=sr, fmin=args.fmin, fmax=args.fmax,
        frame_hop=args.frame_hop, min_note_dur=args.min_note_dur,
        note_gap=args.note_gap, pitch_step_threshold=args.pitch_step,
    )
    if not args.quiet:
        print(f"extracted {len(notes)} notes — first 5: "
              f"{[(round(n['start'],2), n['midi']) for n in notes[:5]]}",
              file=sys.stderr)
        if notes:
            print(f"  last 3: {[(round(n['start'],2), n['midi']) for n in notes[-3:]]}",
                  file=sys.stderr)
    write_midi(notes, out_path)
    if not args.quiet:
        print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
