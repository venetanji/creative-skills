"""Narrow, opt-in frame timeline for drama/music profile callers.

No network operations live here. The installed creative-comfy-graph package owns
model/guide rules; the callers own visible durations and retained source spans.
"""
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


def find_script(caller, rel):
    candidates = [Path(caller).resolve().parents[2] / rel,
                  Path('/home/sandbox/.openclaw/skills') / rel,
                  Path.home() / '.openclaw/skills' / rel]
    return next((p for p in candidates if p.exists()), candidates[0])


_COMMON = {'model_profile', 'fps', 'resolution', 'negative', 'tail_buffer_sec',
           'mode', 'seed', 'base_guide_strength', 'refine_guide_strength'}
_SHOT = {'label', 'prompt', 'duration_sec', 'start_sec', 'image', 'continue_from_prev',
         'overlap_frames', 'overlap_seconds', 'overlap_strength'} | (_COMMON - {'model_profile', 'fps', 'resolution', 'tail_buffer_sec'})


def _frames(seconds, fps):
    value = Fraction(str(seconds)) * fps
    if value < 0:
        raise ValueError('time must be nonnegative')
    return math.floor(value + Fraction(1, 2))


def image_selection(spec, key, idx):
    item = spec[key][idx-1]
    mode = item.get('mode', spec['video'].get('mode'))
    if item.get('continue_from_prev') or mode in ('t2v', 'continuation') or item.get('image') == '@none':
        return None
    ref = item.get('image')
    if ref in (None, '@last'):
        return '@last' if idx > 1 else spec.get('anchor_image')
    if ref == '@anchor':
        if not spec.get('anchor_image'):
            raise ValueError('@anchor requires top-level anchor_image')
        return spec['anchor_image']
    if ref.startswith('@'):
        raise ValueError(f'unsupported profile image token: {ref}')
    return ref


def plan(spec, key, cap, project=None, prompts=None):
    from creative_comfy_graph.ltx_profiles import build_video, continuation_timeline
    video = spec.get('video') or {}
    if key == 'shots' and not isinstance((spec.get('audio') or {}).get('file'), str):
        raise ValueError('profile spec.audio.file is required, including with audio.spec generation')
    if key == 'shots' and not spec['audio']['file'].strip():
        raise ValueError('profile spec.audio.file must not be empty')
    unknown = set(video) - _COMMON
    if unknown:
        raise ValueError(f'unsupported profile video options: {sorted(unknown)}')
    for name in ('reference_sheet', 'loras', 'guides', 'transitions', 'anchor_prompt', 'anchor'):
        if spec.get(name):
            raise ValueError(f'unsupported profile option: {name}')
    character = spec.get('character') or {}
    rows, cursor = [], 0
    requested_cursor = Fraction(0)
    fps = video.get('fps', 24)
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError('fps must be a positive integer')
    width, height = video.get('resolution', [1024, 576])
    tail = _frames(video.get('tail_buffer_sec', 0), fps)
    for idx, item in enumerate(spec.get(key) or [], 1):
        unknown = set(item) - _SHOT
        if unknown:
            raise ValueError(f'unsupported profile {key} options: {sorted(unknown)}')
        if character.get('identity_anchor') or character.get('anchor'):
            raise ValueError('unsupported implicit identity_anchor; use explicit image and remove character anchor')
        duration = float(item['duration_sec'])
        if not 0 < duration <= cap:
            raise ValueError(f'duration_sec must be positive and <= {cap}')
        # Round absolute boundaries, not every duration: avoid cumulative drift.
        requested_start = item.get('start_sec', requested_cursor)
        if _frames(requested_start, fps) != cursor:
            raise ValueError('profile source timeline must be contiguous and begin at zero')
        requested_cursor = Fraction(str(requested_start)) + Fraction(str(item['duration_sec']))
        end = _frames(requested_cursor, fps)
        visible = end - cursor
        if visible <= 0:
            raise ValueError('duration must retain at least one frame')
        continuation = bool(item.get('continue_from_prev'))
        mode = item.get('mode', video.get('mode'))
        if mode not in (None, 't2v', 'i2v', 'ia2v', 'continuation'):
            raise ValueError(f'unsupported profile mode: {mode}')
        if mode == 't2v' and item.get('image') not in (None, '@none'):
            raise ValueError('t2v cannot combine image')
        continuation = continuation or mode == 'continuation'
        if continuation and (idx == 1 or item.get('image')):
            raise ValueError('continuation needs a previous clip and cannot combine image')
        if continuation and mode not in (None, 'continuation'):
            raise ValueError('conflicting continuation mode')
        if 'overlap_frames' in item and 'overlap_seconds' in item:
            raise ValueError('conflicting overlap_frames and overlap_seconds')
        selection = image_selection(spec, key, idx)
        image_route = selection is not None
        mode = 'continuation' if continuation else mode or ('ia2v' if image_route else 't2v')
        if mode in ('i2v', 'ia2v') and not image_route:
            raise ValueError(f'{mode} requires an image')
        if project is not None and selection and selection != '@last' and not (project / selection).is_file():
            raise ValueError(f'image missing: {(project / selection).resolve()}')
        if not image_route and any(name in item or name in video for name in ('base_guide_strength', 'refine_guide_strength')):
            raise ValueError('image guide strengths incompatible with non-image workflow')
        overlap = 0
        previous = rows[-1] if continuation else None
        if previous:
            # Ask the package to normalize 1s@24 to 25 BEFORE guide extraction.
            probe = continuation_timeline(previous_frames=previous['render_frames'], frames=1000001,
                fps=fps, previous_retained_end=previous['retain_span'][1],
                previous_source_start=previous['source_span'][0],
                overlap_frames=item.get('overlap_frames'), overlap_seconds=item.get('overlap_seconds', 1))
            overlap = probe['overlap_frames']
        elif any(name in item for name in ('overlap_frames', 'overlap_seconds', 'overlap_strength')):
            raise ValueError('overlap options require continuation')
        frames = max(9, math.ceil((overlap + visible + tail - 1) / 8) * 8 + 1)
        if frames / fps > cap:
            raise ValueError(f'full render including overlap/tail exceeds {cap}s cap')
        timeline = continuation_timeline(previous_frames=previous['render_frames'], frames=frames,
            fps=fps, previous_retained_end=previous['retain_span'][1],
            previous_source_start=previous['source_span'][0], overlap_frames=overlap) if previous else {
                'source_span': [cursor, cursor + frames], 'guide_span': None}
        source_start = timeline['source_span'][0]
        if any(n * 48000 % fps for n in (source_start, source_start + frames, end)):
            raise ValueError('frame boundaries must map to integer 48k audio samples')
        generation = dict(prompt=prompts[idx-1] if prompts else item['prompt'], mode=mode,
            image=selection, seed=item.get('seed', video.get('seed')),
            negative=item.get('negative', video.get('negative')))
        controls = {}
        if image_route:
            controls.update(base_guide_strength=item.get('base_guide_strength', video.get('base_guide_strength', .7)),
                            refine_guide_strength=item.get('refine_guide_strength', video.get('refine_guide_strength', 1)))
        if continuation:
            controls['overlap_strength'] = item.get('overlap_strength', 1)
        generation.update(controls)
        inputs = {}
        if image_route:
            inputs['image_filename'] = 'preflight-image.png'
        if mode != 'i2v':
            inputs.update(audio_filename='preflight-master.wav', audio_start_frame=source_start)
        if continuation:
            inputs.update(prev_video_filename='preflight-previous.mp4', previous_frames=previous['render_frames'],
                previous_retained_end=previous['retain_span'][1], previous_source_start=previous['source_span'][0],
                overlap_frames=overlap)
        # Validate the actual selected graph using inert uploaded-name placeholders.
        build_video(generation['prompt'], workflow=mode, model_profile=video['model_profile'], frames=frames,
            fps=fps, width=width, height=height, seed=generation['seed'], negative=generation['negative'],
            **inputs, **controls)
        defaults = {'base_guide_strength': .7, 'refine_guide_strength': 1., 'overlap_strength': 1.}
        generation.update({name: defaults[name] if value is None else float(value)
                           for name, value in controls.items()})
        rows.append(dict(index=idx, model_profile=video['model_profile'], fps=fps, width=width, height=height,
            render_frames=frames, retain_span=[overlap, overlap + visible],
            source_span=timeline['source_span'], guide_span=timeline['guide_span'],
            visible_source_span=[cursor, end], requested_duration_sec=item['duration_sec'],
            master_audio_start_frame=source_start,
            master_audio_start_sample_48k=source_start * 48000 // fps,
            master_audio_end_sample_48k=(source_start + frames) * 48000 // fps,
            previous_frames=previous['render_frames'] if previous else None,
            previous_retained_end=previous['retain_span'][1] if previous else None,
            previous_source_start=previous['source_span'][0] if previous else None,
            overlap_frames=overlap, generation=generation))
        cursor = end
    if not rows:
        raise ValueError(f'no {key}')
    return rows


def stem(item, idx, key):
    return f"{idx:03d}-{item.get('label', 'shot' if key == 'shots' else 'scene')}"


def master(spec, project, key):
    return (project / ((spec.get('audio') or {}).get('file', '') if key == 'shots' else 'song.mp3')).resolve()


def command(caller, spec, project, idx, rows, key, image):
    item, row = spec[key][idx-1], rows[idx-1]
    mode = row['generation']['mode']
    if (mode in ('i2v', 'ia2v')) != bool(image):
        raise ValueError(f'{mode} has incompatible image selection')
    opts = dict(model_profile=row['model_profile'], frames=row['render_frames'], fps=row['fps'],
                width=row['width'], height=row['height'], prefix=stem(item, idx, key),
                output_dir=str(project / key), timeout=1800, prompt=row['generation']['prompt'])
    if mode != 'i2v':
        opts.update(audio=str(master(spec, project, key)), audio_start_frame=row['master_audio_start_frame'])
    if image:
        opts['image'] = image
    if row['overlap_frames']:
        opts.update(prev_video=str(project / key / (stem(spec[key][idx-2], idx-1, key)+'.mp4')),
                    prev_frames=row['previous_frames'], previous_retained_end=row['previous_retained_end'],
                    previous_source_start=row['previous_source_start'], overlap_frames=row['overlap_frames'],
                    overlap_strength=row['generation']['overlap_strength'])
    for name in ('negative', 'seed', 'base_guide_strength', 'refine_guide_strength'):
        value = row['generation'].get(name)
        if value is not None:
            opts[name] = value
    return [sys.executable, str(find_script(caller, 'comfyui/scripts/comfy_graph.py')), mode] + [
        value for name, val in opts.items() for value in ('--'+name.replace('_', '-'), str(val))]


def audio_samples(path, ffmpeg):
    if not path.is_file():
        raise ValueError(f'master audio missing: {path}')
    result = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:a:0', '-vn', '-ac', '1',
                             '-ar', '48000', '-f', 's16le', '-'], check=True, capture_output=True)
    return len(result.stdout) // 2


def check_audio(spec, project, key, rows, ffmpeg):
    count = audio_samples(master(spec, project, key), ffmpeg)
    needed = max(row['master_audio_end_sample_48k'] for row in rows)
    if count < needed:
        raise ValueError(f'insufficient master audio: need {needed} decoded samples, got {count}; includes render rounding/tail')
    return count


def check_clip(path, row):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames,width,height,avg_frame_rate', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    if (int(stream['nb_read_frames']), stream['width'], stream['height'], Fraction(stream['avg_frame_rate'])) != (
            row['render_frames'], row['width'], row['height'], row['fps']):
        raise ValueError(f'clip does not match persisted full render timeline: {path}')


def metadata_path(project, key, item, idx):
    return project / key / (stem(item, idx, key) + '.timeline.json')


def file_identity(path):
    path = Path(path).resolve()
    digest = None
    if path.is_file():
        with path.open('rb') as source:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return {'path': str(path), 'sha256': digest}


def generation_fingerprint(spec, project, key, idx, rows):
    row = rows[idx-1]
    request = dict(row['generation'], model_profile=row['model_profile'],
                   timeline=row, master=file_identity(master(spec, project, key)))
    selection = row['generation']['image']
    if selection and selection != '@last':
        request['image_identity'] = file_identity(project / selection)
    if row['overlap_frames'] or selection == '@last':
        previous = project / key / (stem(spec[key][idx-2], idx-1, key) + '.mp4')
        request['previous_clip'] = file_identity(previous)
        request['previous_request'] = generation_fingerprint(spec, project, key, idx-1, rows)
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_metadata(spec, project, key, idx, rows):
    path = metadata_path(project, key, spec[key][idx-1], idx)
    saved = json.loads(path.read_text()) if path.exists() else {}
    if (saved.get('timeline') != rows[idx-1] or
            saved.get('generation_fingerprint') != generation_fingerprint(spec, project, key, idx, rows)):
        raise ValueError(f'missing/stale timeline metadata: {path}; rerender this clip and dependent clips '
                         '(remove their MP4/timeline/endpoint files); never infer retained endpoint from clip tail')


def extract_endpoint(path, row, output, ffmpeg):
    subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(path), '-vf',
                    f"select=eq(n\\,{row['retain_span'][1]-1})", '-frames:v', '1', str(output)], check=True)


def render(caller, spec, project, idx, key, rows, run, ffmpeg):
    row, item = rows[idx-1], spec[key][idx-1]
    directory = project / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (stem(item, idx, key) + '.mp4')
    endpoint = directory / (stem(item, idx, key) + '-last.png')
    check_audio(spec, project, key, rows, ffmpeg)
    selection = row['generation']['image']
    if row['overlap_frames'] or selection == '@last':
        check_metadata(spec, project, key, idx-1, rows)
        previous = directory / (stem(spec[key][idx-2], idx-1, key)+'.mp4')
        check_clip(previous, rows[idx-2])
        if selection == '@last':
            image_path = directory / (stem(spec[key][idx-2], idx-1, key)+'-last.png')
            extract_endpoint(previous, rows[idx-2], image_path, ffmpeg)
            image = str(image_path.resolve())
        else:
            image = None
    else:
        image = str((project / selection).resolve()) if selection else None
    if image and not Path(image).is_file():
        raise ValueError(f'image missing: {image}')
    if path.exists():
        check_metadata(spec, project, key, idx, rows)
        check_clip(path, row)
        extract_endpoint(path, row, endpoint, ffmpeg)
        return
    cmd = command(caller, spec, project, idx, rows, key, image)
    # Never mistake an old numbered output for this submission.
    before = set(directory.glob(stem(item, idx, key)+'_*.mp4'))
    if run(cmd) != 0:
        raise RuntimeError('profile generation failed')
    outputs = set(directory.glob(stem(item, idx, key)+'_*.mp4')) - before
    if len(outputs) != 1:
        raise RuntimeError(f'expected one new profile output, got {len(outputs)}')
    produced = outputs.pop()
    check_clip(produced, row)
    produced.rename(path)
    metadata_path(project, key, item, idx).write_text(json.dumps({'timeline': row, 'command': cmd,
        'generation_fingerprint': generation_fingerprint(spec, project, key, idx, rows)}, indent=2)+'\n')
    extract_endpoint(path, row, endpoint, ffmpeg)


def verify_assembled_audio(path, rows, ffmpeg):
    expected = sum(r['retain_span'][1] - r['retain_span'][0] for r in rows) * 48000 // rows[0]['fps']
    actual = audio_samples(path, ffmpeg)
    if actual != expected:
        raise ValueError(f'decoded final audio has {actual} samples, expected {expected}; '
                         'use a sample-exact ffmpeg build (validated system ffmpeg 9), not rounded AAC duration')
    return actual


def invalidate_assembly_metadata(project, key):
    names = ['final']
    if key == 'scenes':
        names += ['final_'+p.stem.removeprefix('song_') for p in project.glob('song_v*.mp3')]
    for name in names:
        (project / (name+'.timeline.json')).unlink(missing_ok=True)


def assemble(caller, spec, project, key, rows, ffmpeg):
    sources = [('final', master(spec, project, key))]
    if key == 'scenes':
        sources += [('final_'+p.stem.removeprefix('song_'), p) for p in sorted(project.glob('song_v*.mp3'))]
    # Invalidate certificates even when clip preflight fails before muxing.
    invalidate_assembly_metadata(project, key)
    clips = []
    for idx, (item, row) in enumerate(zip(spec[key], rows), 1):
        check_metadata(spec, project, key, idx, rows)
        path = project / key / (stem(item, idx, key)+'.mp4')
        check_clip(path, row)
        clips.append((path, *row['retain_span']))
    module_spec = importlib.util.spec_from_file_location('profile_frame_assembly', find_script(caller, 'storyboard/lib/assemble.py'))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    for name, audio in sources:
        result = module.assemble_frame_spans(clips, audio, project / (name+'.mp4'),
            fps=rows[0]['fps'], width=rows[0]['width'], height=rows[0]['height'], ffmpeg=ffmpeg)
        result['decoded_audio_samples'] = verify_assembled_audio(project / (name+'.mp4'), rows, ffmpeg)
        (project / (name+'.timeline.json')).write_text(json.dumps({'assembly': result, 'clips': rows,
            'master_audio': str(audio), 'master_audio_start_sample_48k': 0}, indent=2)+'\n')
