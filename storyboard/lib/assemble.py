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


def assemble_frame_spans(clips, audio_file, output, *, fps, width, height,
                         ffmpeg='ffmpeg', ffprobe='ffprobe'):
    """Assemble half-open integer frame spans with one continuous master audio.

    Inputs must already share the requested CFR and geometry. Reject rather
    than silently duplicate/drop/scale frames. Audio is decoded to check sample
    coverage, then trimmed exactly to the video timeline (no -shortest).
    Return verified timeline metadata; replace output only after full decode.
    """
    import json
    import tempfile
    from fractions import Fraction

    def run(args):
        return subprocess.run(args, check=True, capture_output=True).stdout

    def probe(path):
        return json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_frames', '-show_entries',
            'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time',
            '-of', 'json', str(path)]))

    def check_video(path, count=None):
        data = probe(path)
        stream = data['streams'][0]
        if (stream['width'], stream['height']) != (width, height) or Fraction(stream['avg_frame_rate']) != fps:
            raise ValueError(f'input must be {width}x{height} at {fps} fps: {path}')
        stamps = [float(f['best_effort_timestamp_time']) for f in data['frames']]
        if not stamps or any(abs((b-a)-1/fps) > 2e-6 for a,b in zip(stamps,stamps[1:])):
            raise ValueError(f'nonuniform frame timestamps: {path}')
        if count is not None and len(stamps) != count:
            raise RuntimeError(f'expected {count} decoded frames, got {len(stamps)}')
        return len(stamps)

    if not clips or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (fps,width,height)):
        raise ValueError('nonempty clips and positive integer fps/dimensions required')
    total = 0
    args = [ffmpeg, '-y', '-v', 'error']
    filters = []
    for i, (src, start, end) in enumerate(clips):
        count = check_video(src)
        if any(isinstance(x,bool) or not isinstance(x,int) for x in (start,end)) or not 0 <= start < end <= count:
            raise ValueError(f'invalid frame span [{start},{end}) for {count} frames')
        total += end-start
        args += ['-i',str(src)]
        filters.append(f'[{i}:v:0]trim=start_frame={start}:end_frame={end},settb=1/{fps},setpts=N,setsar=1[v{i}]')
    samples = total*48000//fps
    if total*48000 % fps:
        raise ValueError('timeline does not map to integer 48k audio samples')
    if audio_file is not None:
        pcm = run([ffmpeg,'-v','error','-i',str(audio_file),'-map','0:a:0','-vn','-ac','1','-ar','48000','-f','s16le','-'])
        if len(pcm)//2 < samples:
            raise ValueError(f'master audio too short: need {samples} samples, got {len(pcm)//2}')
        args += ['-i',str(audio_file)]
    filters.append(''.join(f'[v{i}]' for i in range(len(clips)))+f'concat=n={len(clips)}:v=1:a=0,settb=1/{fps},setpts=N[v]')
    if audio_file is not None:
        filters.append(f'[{len(clips)}:a:0]aresample=48000,atrim=start_sample=0:end_sample={samples},asetpts=N/SR/TB[a]')
    args += ['-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264',
             '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-r',str(fps),'-fps_mode','cfr',
             '-video_track_timescale',str(fps*1000)]
    if audio_file is not None:
        args += ['-map','[a]','-c:a','aac','-b:a','192k']
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='frame-assembly-',dir=output.parent) as tmp:
        candidate = Path(tmp)/'result.mp4'
        run(args+['-movflags','+faststart',str(candidate)])
        check_video(candidate,total)
        run([ffmpeg,'-v','error','-xerror','-i',str(candidate),'-f','null','-'])
        if audio_file is not None:
            pcm = run([ffmpeg, '-v', 'error', '-xerror', '-i', str(candidate),
                       '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '48000', '-f', 's16le', '-'])
            actual_samples = len(pcm)//2
            if actual_samples != samples:
                raise ValueError(f'decoded final audio has {actual_samples} samples, expected {samples}; '
                                 'use a sample-exact ffmpeg build; destination was not replaced')
        candidate.replace(output)
    return dict(frames=total,fps=fps,width=width,height=height,
                audio_samples=samples if audio_file is not None else None)
