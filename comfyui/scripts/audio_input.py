"""Admit a local soundtrack before any upload or graph submission."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import wave

MAX_AUDIO_BYTES = 32 * 1024 * 1024
SAMPLE_RATE = 48_000


def admit_audio(filename: str, *, frames: int, fps: int = 24) -> tuple[str, bytes]:
    if not isinstance(filename, str) or not filename.strip() or '\x00' in filename:
        raise ValueError('audio must be a nonempty local file path')
    if (type(frames) is not int or type(fps) is not int or fps <= 0 or
            not 0 < frames / fps <= 16):
        raise ValueError('audio duration must be bounded by the video request')
    try:
        with Path(filename).open('rb') as source:
            data = source.read(MAX_AUDIO_BYTES + 1)
    except OSError as exc:
        raise ValueError('audio must be a readable local file') from exc
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise ValueError('audio must be nonempty and at most 32 MiB')

    def run(*arguments):
        try:
            result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError('audio decoding requires working ffmpeg/ffprobe and must finish within 15 seconds') from exc
        if result.returncode:
            raise ValueError('audio could not be decoded safely')
        return result.stdout

    sample_count = round(frames / fps * SAMPLE_RATE)
    restrictions = ('-protocol_whitelist', 'file,pipe', '-format_whitelist', 'wav,mp3,flac,ogg,mov')
    with tempfile.TemporaryDirectory(prefix='h3-source-audio-') as directory:
        source = Path(directory) / 'source'
        source.write_bytes(data)
        probe = run('ffprobe', '-v', 'error', *restrictions, '-select_streams', 'a:0',
                    '-show_entries', 'stream=sample_rate,channels', '-of', 'json', str(source))
        try:
            stream = json.loads(probe)['streams'][0]
            if not 1 <= int(stream['channels']) <= 8 or not 1 <= int(stream['sample_rate']) <= 192_000:
                raise ValueError('unsupported stream')
        except (ValueError, TypeError, KeyError, IndexError):
            raise ValueError('audio must contain a supported audio stream') from None
        pcm = run('ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', *restrictions,
                  '-i', str(source), '-map', '0:a:0', '-vn', '-sn', '-dn',
                  '-t', str(sample_count / SAMPLE_RATE), '-ar', str(SAMPLE_RATE),
                  '-ac', '2', '-f', 's16le', 'pipe:1')
    if len(pcm) != sample_count * 4:
        raise ValueError(f'audio must cover all {frames / fps:.6f} seconds of the rendered video')
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    data = output.getvalue()
    return 'h3_audio_' + hashlib.sha256(data).hexdigest()[:32] + '.wav', data
