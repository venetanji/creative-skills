"""CLI adapter for curated profile graphs; validation precedes all uploads."""
import math
import re
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from creative_comfy_graph.ltx_profiles import build_video, build_flf2v, continuation_timeline

_SUPPORTED = {'t2v', 'i2v', 'ia2v', 'continuation', 'flf2v'}
_ALLOWED = {
    'model_profile', 'prompt', 'seconds', 'frames', 'fps', 'width', 'height',
    'seed', 'prefix', 'negative', 'image', 'audio', 'prev_video', 'prev_frames',
    'previous_retained_end', 'previous_source_start', 'overlap_seconds',
    'overlap_frames', 'overlap_strength', 'audio_start_frame',
    'base_guide_strength', 'refine_guide_strength', 'output_dir', 'timeout',
    'notify_target', 'caption_template', 'timeline_json',
    'first', 'last', 'first_guide_strength', 'last_guide_strength',
}


def _local_file(value):
    """Bare/subfolder names remain Comfy input names; explicit paths are local."""
    path = Path(value).expanduser()
    if path.exists():
        if not path.is_file():
            raise ValueError(f'local input is not a file: {value}')
        return path
    if path.is_absolute() or value.startswith(('./', '../', '~')):
        raise ValueError(f'missing local input: {value}')
    return None


def strict_upload(value):
    """One POST, no path fallback or ambiguous-response retry on profile paths."""
    import core
    path = _local_file(value)
    if path is None:
        return value
    data = path.read_bytes()
    boundary = '----ComfyProfileUploadBoundary'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
    body += data + (f'\r\n--{boundary}\r\nContent-Disposition: form-data; '
                    f'name="type"\r\n\r\ninput\r\n--{boundary}--\r\n').encode()
    request = urllib.request.Request(f'{core.BASE}/upload/image', data=body, method='POST',
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Upload {path}: HTTP {exc.code}: {detail}; no retry/submission') from exc
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f'Ambiguous upload {path}: {exc}; no retry/submission') from exc
    if (not isinstance(result, dict) or not isinstance(result.get('name'), str)
            or not result['name'].strip() or result.get('type', 'input') != 'input'
            or not isinstance(result.get('subfolder', ''), str)):
        raise RuntimeError(f'Invalid/ambiguous upload response for {path}: {result!r}; no submission')
    return '/'.join(v for v in (result.get('subfolder', ''), result['name']) if v)


def _check_audio_coverage(path, start_frame, frames, fps):
    # Probe local files only; dump and server-relative assets never require it.
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=duration:format=duration', '-of', 'json', str(path)],
            capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise ValueError('ffprobe required to preflight local source audio') from None
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'cannot probe local source audio: {exc.stderr}') from exc
    info = json.loads(result.stdout)
    streams = info.get('streams', [])
    duration = streams[0].get('duration') if streams else None
    if duration in (None, 'N/A') and streams:
        duration = info.get('format', {}).get('duration')
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        raise ValueError('cannot determine first audio stream duration') from None
    if not math.isfinite(seconds) or seconds + 1e-6 < (start_frame + frames) / fps:
        raise ValueError('source audio is too short to cover requested source span')


def parse_integer(name, value):
    """CLI integer strings are allowed; JSON floats/bools never coerce."""
    if isinstance(value, str) and re.fullmatch(r'[+-]?[0-9]+', value):
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    return value


def parse_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name} must be a finite number')
    return number


def prepare(cmd, opts, seed, prompt, *, upload, probe, dump=False):
    if cmd not in _SUPPORTED:
        raise ValueError(f'profile path does not support {cmd}')
    unknown = set(opts) - _ALLOWED
    if unknown:
        raise ValueError(f'unverified profile options: {sorted(unknown)}')
    # Presence is not omission: reject invalid source fields before any probe/upload.
    for name in ('image', 'audio', 'prev_video', 'first', 'last'):
        if name in opts:
            if not isinstance(opts[name], str):
                raise ValueError(f'{name} must be a string value')
            if not opts[name].strip():
                raise ValueError(f'{name} must not be empty')
    if cmd in {'i2v', 'ia2v'} and not opts.get('image'):
        raise ValueError(f'{cmd} requires image')
    if cmd == 'ia2v' and not opts.get('audio'):
        raise ValueError('ia2v requires audio')
    if cmd == 'continuation' and not opts.get('prev_video'):
        raise ValueError('continuation requires prev_video')
    if cmd != 'continuation' and opts.get('prev_video'):
        raise ValueError('prev_video is only valid for continuation')
    if cmd in {'t2v', 'continuation'} and opts.get('image'):
        raise ValueError(f'image is not supported by {cmd}')
    if cmd != 'flf2v' and any(k in opts for k in ('first', 'last', 'first_guide_strength', 'last_guide_strength')):
        raise ValueError('first/last options are only valid for flf2v')
    if cmd == 'flf2v':
        conflicts = set(opts) & {'image', 'prev_video', 'prev_frames', 'previous_retained_end',
            'previous_source_start', 'overlap_seconds', 'overlap_frames', 'overlap_strength',
            'audio_start_frame', 'base_guide_strength', 'refine_guide_strength', 'timeline_json'}
        if conflicts:
            raise ValueError(f'flf2v does not accept {sorted(conflicts)}')
        if not opts.get('first') or not opts.get('last'):
            raise ValueError('flf2v requires first and last')
    for name in ('model_profile', 'prompt', 'prefix', 'negative', 'image', 'audio',
                 'prev_video', 'first', 'last', 'output_dir', 'notify_target', 'caption_template', 'timeline_json'):
        if name in opts and not isinstance(opts[name], str):
            raise ValueError(f'{name} must be a string value')
    fps = parse_integer('fps', opts.get('fps', 24))
    if 'seconds' in opts and 'frames' in opts:
        raise ValueError('specify frames or seconds, not both')
    frames = (parse_integer('frames', opts['frames']) if 'frames' in opts else
              int(parse_number('seconds', opts.get('seconds', 2))*fps)//8*8+1)
    seed = parse_integer('seed', seed) if seed is not None else None
    kw = dict(model_profile=opts['model_profile'], workflow=cmd, frames=frames, fps=fps,
              width=parse_integer('width',opts.get('width',512)), height=parse_integer('height',opts.get('height',320)),
              seed=seed, filename_prefix=opts.get('prefix','ltx_profile'), negative=opts.get('negative'))
    for name in ('overlap_frames', 'previous_retained_end', 'previous_source_start', 'audio_start_frame'):
        if name in opts:
            kw[name] = parse_integer(name, opts[name])
    for name in ('overlap_seconds', 'overlap_strength', 'base_guide_strength', 'refine_guide_strength', 'first_guide_strength', 'last_guide_strength'):
        if name in opts:
            kw[name] = parse_number(name, opts[name])
    files = {'image': 'image_filename', 'audio': 'audio_filename', 'prev_video': 'prev_video_filename'}
    for option, argument in files.items():
        if opts.get(option):
            kw[argument] = opts[option]
    if cmd == 'flf2v':
        files.update(first='first_frame_filename', last='last_frame_filename')
        kw.update(first_frame_filename=opts['first'], last_frame_filename=opts['last'])
    if 'prev_frames' in opts:
        kw['previous_frames'] = parse_integer('prev_frames', opts['prev_frames'])
    elif opts.get('prev_video'):
        kw['previous_frames'] = probe(opts['prev_video'])
    builder = build_video
    if cmd == 'flf2v':
        builder = build_flf2v
        kw.pop('workflow')
    graph = builder(prompt, **kw)  # Validate ALL options before upload.
    timeline = None
    if opts.get('prev_video'):
        fields = {k: kw[k] for k in ('previous_frames', 'frames', 'fps', 'previous_retained_end',
                  'previous_source_start', 'overlap_frames', 'overlap_seconds') if k in kw}
        timeline = continuation_timeline(**fields)
    if not dump:
        local = {option: _local_file(opts[option]) for option in files if opts.get(option)}
        if local.get('audio'):
            offset = kw.get('audio_start_frame', timeline['source_span'][0] if timeline else 0)
            _check_audio_coverage(local['audio'], offset, frames, fps)
        for option, argument in files.items():
            if opts.get(option):
                kw[argument] = upload(opts[option])
        graph = builder(prompt, **kw)
    return graph, timeline
