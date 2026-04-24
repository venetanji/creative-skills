"""Shared ffmpeg assemble helper used by both drama-video and music-video.

Pre-trims each clip to its exact (start, duration), strips per-clip audio
(LTX-generated, typically garbage or double-mix), concats silent video
streams via the ffmpeg concat demuxer, then muxes a clean source audio
track on top. Frame-accurate, visually lossless (libx264 veryfast crf 18),
and doesn't suffer from the ComfyUI-FFmpeg AddAudio double-mux bug that
bit both skills' previous comfy-vconcat path.

Usage from the orchestrator:

    import sys
    sys.path.insert(0, "/home/sandbox/.openclaw/skills/storyboard/lib")
    from assemble import assemble_clips

    assemble_clips(
        clips=[(shot1_mp4, start_sec_0, duration_sec_0),
               (shot2_mp4, start_sec_1, duration_sec_1), ...],
        audio_file=song_mp3,
        output=final_mp4,
        work_dir=project_dir / "_assemble_cache",
        ffmpeg=ffmpeg_binary_path,
    )

`start_sec` defaults 0 (trim from start); `duration_sec` is how much of
the source clip to keep. Caching in `work_dir` makes re-runs cheap —
trimmed clips are reused when the source mp4 hasn't changed."""
from __future__ import annotations
import subprocess
from pathlib import Path


def _trim_clip(ffmpeg: str, src: Path, start_sec: float, duration_sec: float,
                dst: Path) -> None:
    """Frame-accurate re-encode trim. libx264 veryfast crf 18 is
    visually lossless at the ~10 s clip scale this is used for, and
    completes in ~1 s CPU. `-c copy` rounds to keyframes (± 0.3 s
    drift), which compounds over a 10-scene video."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if start_sec and start_sec > 0:
        cmd += ["-ss", f"{float(start_sec):.3f}"]
    cmd += ["-i", str(src),
            "-t", f"{float(duration_sec):.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            str(dst)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed for {src.name}")


def _strip_audio(ffmpeg: str, src: Path, dst: Path) -> None:
    """`-an` + stream-copy video (trim step already re-encoded). Makes
    the concat demuxer happy (all inputs have same stream count + params)
    and removes per-clip LTX audio tracks so the overlay mux is clean."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    if subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                       "-i", str(src), "-an", "-c:v", "copy", str(dst)],
                      capture_output=True).returncode != 0:
        raise RuntimeError(f"ffmpeg audio-strip failed for {src.name}")


def assemble_clips(clips, audio_file, output, work_dir, ffmpeg):
    """Stitch N clips with their overlay audio. Returns nothing; `output`
    exists on success.

    Args:
        clips: list of (src_mp4: Path, start_sec: float, duration_sec: float).
        audio_file: Path to the audio to mux over the concat. If None,
            the final mp4 has no audio.
        output: Path — final mp4.
        work_dir: Path — where trimmed + silent intermediates are cached.
            Safe to keep across runs; reused if source mp4 is unchanged.
        ffmpeg: Path/str to the ffmpeg binary.
    """
    output = Path(output)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trim each source to its exact (start, duration).
    trimmed: list[Path] = []
    for i, (src, start, dur) in enumerate(clips):
        src = Path(src)
        key = f"{i:03d}_{src.stem}_s{float(start):.3f}_d{float(dur):.3f}.mp4"
        dst = work_dir / key
        _trim_clip(ffmpeg, src, float(start), float(dur), dst)
        trimmed.append(dst)

    # 2. Strip audio from each trimmed clip.
    silent: list[Path] = []
    for t in trimmed:
        dst = t.with_suffix(".silent.mp4")
        _strip_audio(ffmpeg, t, dst)
        silent.append(dst)

    # 3. Concat silent clips via demuxer (stream-copy).
    list_file = work_dir / "_concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in silent))
    silent_concat = work_dir / "_silent_concat.mp4"
    if subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                       "-f", "concat", "-safe", "0", "-i", str(list_file),
                       "-c", "copy", str(silent_concat)],
                      capture_output=True).returncode != 0:
        raise RuntimeError("ffmpeg concat failed")

    # 4. Mux audio overlay.
    if audio_file is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                        "-i", str(silent_concat), "-c", "copy", str(output)],
                       capture_output=True, check=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                       "-i", str(silent_concat), "-i", str(audio_file),
                       "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                       "-shortest", str(output)],
                      capture_output=True).returncode != 0:
        raise RuntimeError("ffmpeg audio mux failed")
