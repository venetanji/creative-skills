"""Offline caller contract tests; no Comfy subprocess may execute here."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(kind):
    path = ROOT / f'{kind}-video/scripts/{kind}_video.py'
    spec = importlib.util.spec_from_file_location(f'{kind}_caller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_spec(kind='drama', tail=0):
    key = 'shots' if kind == 'drama' else 'scenes'
    return {'video': {'model_profile': 'ltx-2.5', 'resolution': [512, 320],
                      'fps': 24, 'tail_buffer_sec': tail, 'negative': 'cuts'},
            'audio': {'file': 'master.wav'}, key: [
                {'label': 'A', 'duration_sec': 49/24, 'prompt': 'ball'},
                {'label': 'B', 'duration_sec': 1, 'continue_from_prev': True, 'prompt': 'roll'},
                {'label': 'C', 'duration_sec': 1, 'continue_from_prev': True, 'prompt': 'roll'}]}


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_profile_plan_retained_endpoint_and_global_offsets(kind):
    m = load(kind)
    rows = m._profile_plan(fixture_spec(kind))
    assert [r['render_frames'] for r in rows] == [49, 49, 49]
    assert [r['retain_span'] for r in rows] == [[0, 49], [25, 49], [25, 49]]
    assert [r['source_span'] for r in rows] == [[0, 49], [24, 73], [48, 97]]
    assert [r['master_audio_start_frame'] for r in rows] == [0, 24, 48]
    assert sum(b-a for a,b in [r['retain_span'] for r in rows[:2]]) == 73
    assert rows[2]['previous_source_start'] == 24
    assert rows[2]['master_audio_start_sample_48k'] == 96000


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_tail_buffer_never_becomes_continuation_endpoint(kind):
    m = load(kind)
    rows = m._profile_plan(fixture_spec(kind, tail=1))
    assert rows[0]['render_frames'] == 73
    assert rows[0]['retain_span'] == [0, 49]
    assert rows[1]['guide_span'] == [24, 49]
    assert rows[1]['previous_retained_end'] == 49
    assert rows[1]['source_span'][0] == 24
    assert rows[2]['source_span'][0] == 48


@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('extra', ['camera_lora', 'fast', 'guides', 'image_refs', 'hdr_lora', 'identity_anchor', 'model', 'surprise_option'])
def test_unsupported_fields_rejected_before_any_render(kind, extra, tmp_path, monkeypatch):
    m = load(kind)
    spec = fixture_spec(kind)
    spec['video'][extra] = True
    monkeypatch.setattr(m, '_run', lambda *a, **k: pytest.fail('must reject before subprocess/upload'))
    with pytest.raises(ValueError, match='unsupported|unverified'):
        if kind == 'drama':
            m._render_shot(spec, tmp_path, 1, spec['shots'][0])
        else:
            m.cmd_scene(spec, tmp_path, 1)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_checkout_helpers_precede_installed_trap(kind, monkeypatch):
    m = load(kind)
    monkeypatch.setattr(Path, 'exists', lambda self: True)
    assert m._find_skill_script('comfyui/scripts/comfy_graph.py') == ROOT / 'comfyui/scripts/comfy_graph.py'
    assert m._find_skill_script('storyboard/lib/assemble.py') == ROOT / 'storyboard/lib/assemble.py'


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_profile_command_uses_master_and_exact_timeline(kind, tmp_path):
    m = load(kind)
    spec = fixture_spec(kind)
    rows = m._profile_plan(spec)
    command = m._profile_command(spec, tmp_path, 3, rows, image=None)
    assert command[:3] == [sys.executable, str(ROOT / 'comfyui/scripts/comfy_graph.py'), 'continuation']
    opts = dict(zip(command[3::2], command[4::2]))
    assert opts['--model-profile'] == 'ltx-2.5'
    assert opts['--frames'] == '49'
    assert opts['--previous-retained-end'] == '49'
    assert opts['--previous-source-start'] == '24'
    assert opts['--overlap-frames'] == '25'
    assert opts['--audio-start-frame'] == '48'
    assert opts['--negative'] == 'cuts'
    assert opts['--audio'] == str(tmp_path / ('master.wav' if kind == 'drama' else 'song.mp3'))
    assert '--seconds' not in command and '--image-refs' not in command


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_selectable_23_and_image_workflows(kind, tmp_path):
    m = load(kind)
    spec = fixture_spec(kind)
    spec['video']['model_profile'] = 'ltx-2.3'
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key] = spec[key][:1]
    spec[key][0].update(image='anchor.png', base_guide_strength=0.65, refine_guide_strength=0.85, seed=7)
    rows = m._profile_plan(spec)
    cmd = m._profile_command(spec, tmp_path, 1, rows, image='anchor.png')
    assert cmd[2] == 'ia2v'
    opts = dict(zip(cmd[3::2], cmd[4::2]))
    assert opts['--model-profile'] == 'ltx-2.3'
    assert opts['--base-guide-strength'] == '0.65'
    assert opts['--refine-guide-strength'] == '0.85'
    assert opts['--seed'] == '7'
    spec[key][0]['mode'] = 'i2v'
    cmd = m._profile_command(spec, tmp_path, 1, m._profile_plan(spec), image='anchor.png')
    assert cmd[2] == 'i2v' and '--audio' not in cmd


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_invalid_overlap_and_noncontiguous_source_fail(kind):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key][1]['overlap_frames'] = 24
    with pytest.raises(ValueError, match='8k'):
        m._profile_plan(spec)
    del spec[key][1]['overlap_frames']
    spec[key][1]['start_sec'] = 9
    with pytest.raises(ValueError, match='contiguous'):
        m._profile_plan(spec)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_actual_render_entrypoint_persists_command_and_retained_end(kind, tmp_path, monkeypatch):
    m = load(kind)
    spec = fixture_spec(kind, tail=1)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key] = spec[key][:1]
    helper = m._profile_helper()
    monkeypatch.setattr(m, '_profile_helper', lambda: helper)
    monkeypatch.setattr(m, '_ffmpeg' if kind == 'drama' else 'FFMPEG', lambda: 'ffmpeg')
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 300000)
    monkeypatch.setattr(helper, 'check_clip', lambda *a: None)
    endpoints, calls = [], []
    monkeypatch.setattr(helper, 'extract_endpoint', lambda path, row, *a: endpoints.append(row['retain_span'][1]))
    monkeypatch.setattr(m, '_slice_audio' if kind == 'drama' else '_slice_song',
                        lambda *a, **k: pytest.fail('profile may not use legacy sliced audio'))
    def run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / key / '001-A_00001_.mp4').touch()
        return 0
    monkeypatch.setattr(m, '_run', run)
    if kind == 'drama':
        m.cmd_shot(spec, tmp_path, 1)
    else:
        m.cmd_scene(spec, tmp_path, 1)
    assert len(calls) == 1 and calls[0][2] == 't2v'
    assert '--audio' in calls[0]
    assert endpoints == [49]  # not full render endpoint 73
    saved = json.loads((tmp_path / key / '001-A.timeline.json').read_text())
    assert saved['command'] == calls[0]
    assert saved['timeline']['render_frames'] == 73
    assert saved['timeline']['retain_span'] == [0,49]


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_short_audio_or_missing_previous_metadata_blocks_submission(kind, tmp_path, monkeypatch):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    helper = m._profile_helper()
    monkeypatch.setattr(m, '_profile_helper', lambda: helper)
    monkeypatch.setattr(m, '_ffmpeg' if kind == 'drama' else 'FFMPEG', lambda: 'ffmpeg')
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 146000)
    monkeypatch.setattr(m, '_run', lambda *a, **k: pytest.fail('no GPU calls'))
    with pytest.raises(ValueError, match='insufficient master audio'):
        m._profile_render(spec, tmp_path, 2)
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 300000)
    with pytest.raises(ValueError, match='missing/stale timeline'):
        m._profile_render(spec, tmp_path, 2)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_explicit_t2v_does_not_implicitly_resolve_previous_image(kind, tmp_path, monkeypatch):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    item = spec[key][1]
    item.pop('continue_from_prev')
    item['mode'] = 't2v'
    spec[key] = spec[key][:2]
    helper = m._profile_helper()
    monkeypatch.setattr(m, '_profile_helper', lambda: helper)
    monkeypatch.setattr(m, '_ffmpeg' if kind == 'drama' else 'FFMPEG', lambda: 'ffmpeg')
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 300000)
    monkeypatch.setattr(m, '_resolve_image', lambda *a: pytest.fail('explicit t2v must not use @last default'))
    class Submitted(Exception): pass
    def run(cmd, **k):
        assert cmd[2] == 't2v' and '--image' not in cmd
        raise Submitted()
    monkeypatch.setattr(m, '_run', run)
    with pytest.raises(Submitted):
        m._profile_render(spec, tmp_path, 2)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_generated_audio_i2v_still_requires_final_master_coverage(kind, tmp_path, monkeypatch):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key] = spec[key][:1]
    spec[key][0].update(mode='i2v', image='anchor.png')
    (tmp_path / 'anchor.png').touch()
    helper = m._profile_helper()
    monkeypatch.setattr(m, '_profile_helper', lambda: helper)
    monkeypatch.setattr(m, '_ffmpeg' if kind == 'drama' else 'FFMPEG', lambda: 'ffmpeg')
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 1)
    with pytest.raises(ValueError, match='insufficient master audio'):
        m._profile_render(spec, tmp_path, 1)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_profile_planning_rounds_absolute_boundaries_without_drift(kind):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key] = [dict(label=str(i), duration_sec=.07, start_sec=str(i*.07), prompt='ball', mode='t2v')
                 for i in range(20)]
    rows = m._profile_plan(spec)
    assert sum(r['retain_span'][1] - r['retain_span'][0] for r in rows) == 34
    for left, right in zip(rows, rows[1:]):
        assert left['visible_source_span'][1] == right['visible_source_span'][0]


def test_implicit_duration_rounding_has_no_cumulative_drift():
    m = load('drama')
    spec = fixture_spec()
    spec['shots'] = [dict(label=str(i), duration_sec=.07, prompt='ball', mode='t2v') for i in range(20)]
    rows = m._profile_plan(spec)
    assert rows[-1]['visible_source_span'][1] == 34


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_incompatible_mode_fields_rejected_during_plan(kind):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key][0].update(mode='t2v', image='anchor.png')
    with pytest.raises(ValueError, match='image'):
        m._profile_plan(spec)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_loaded_cli_without_source_pythonpath(kind, tmp_path):
    import os
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT / f'{kind}-video/scripts/{kind}_video.py'), '--help'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-c', 'import creative_comfy_graph.ltx_profiles'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('field,value', [('base_guide_strength', .3), ('refine_guide_strength', .8), ('overlap_seconds', 2)])
def test_no_accepted_but_ignored_profile_fields(kind, field, value):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key][1][field] = value
    if field == 'overlap_seconds':
        spec[key][1]['overlap_frames'] = 25
    with pytest.raises(ValueError, match='incompatible|conflicting'):
        m._profile_plan(spec)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_requested_duration_and_fourth_continuation_are_not_full_render_length(kind):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    spec[key].append(dict(spec[key][-1], label='D'))
    rows = m._profile_plan(spec)
    assert rows[-1]['source_span'] == [72, 121]
    assert rows[-1]['previous_source_start'] == 48
    assert rows[-1]['visible_source_span'] == [97, 121]
    assert rows[-1]['requested_duration_sec'] == 1
    assert rows[-1]['render_frames'] == 49


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_previous_decoded_count_blocks_gpu_before_guide_extraction(kind, tmp_path, monkeypatch):
    m = load(kind)
    spec = fixture_spec(kind)
    key = 'shots' if kind == 'drama' else 'scenes'
    helper = m._profile_helper()
    monkeypatch.setattr(m, '_profile_helper', lambda: helper)
    monkeypatch.setattr(m, '_ffmpeg' if kind == 'drama' else 'FFMPEG', lambda: 'ffmpeg')
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 300000)
    monkeypatch.setattr(m, '_run', lambda *a, **k: pytest.fail('no GPU'))
    (tmp_path / key).mkdir()
    rows = m._profile_plan(spec)
    helper.metadata_path(tmp_path, key, spec[key][0], 1).write_text(json.dumps({'timeline': rows[0], 'generation_fingerprint':
        helper.generation_fingerprint(spec, tmp_path, key, 1, rows)}))
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=512x320:rate=24',
                    '-frames:v', '9', '-c:v', 'libx264', str(tmp_path / key / '001-A.mp4')], check=True)
    with pytest.raises(ValueError, match='full render timeline'):
        m._profile_render(spec, tmp_path, 2)


def test_profile_audio_validation_rejects_encoder_rounding(tmp_path, monkeypatch):
    m = load('drama')
    helper = m._profile_helper()
    rows = m._profile_plan(dict(fixture_spec(), shots=fixture_spec()['shots'][:2]))
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 145968)
    with pytest.raises(ValueError, match='decoded.*146000|146000.*decoded'):
        helper.verify_assembled_audio(tmp_path / 'final.mp4', rows, 'ffmpeg')


def test_music_all_does_not_enter_unsupported_transition_pass(tmp_path, monkeypatch):
    m = load('music')
    spec = fixture_spec('music')
    (tmp_path / 'song.mp3').touch()
    calls=[]
    for name in ('cmd_song','cmd_anchors','cmd_scene','cmd_assemble'):
        monkeypatch.setattr(m, name, lambda *a, _name=name, **k: calls.append(_name))
    helper = m._profile_helper()
    monkeypatch.setattr(m, '_profile_helper', lambda: helper)
    monkeypatch.setattr(helper, 'audio_samples', lambda *a: 300000)
    monkeypatch.setattr(m, '_probe_duration', lambda *a: 4)
    monkeypatch.setattr(m, 'cmd_transitions', lambda *a: pytest.fail('profile all must skip unrequested transition stage'))
    m.cmd_all(spec, tmp_path, no_gate=True)
    assert calls[-1] == 'cmd_assemble'


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_top_level_anchor_generation_not_silently_ignored(kind):
    m = load(kind)
    spec = fixture_spec(kind)
    spec['anchor_prompt'] = 'generate anchor first'
    with pytest.raises(ValueError, match='unsupported'):
        m._profile_plan(spec)


@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_constructed_command_loads_real_cli_and_installed_profile_package(kind, tmp_path):
    import os
    m = load(kind)
    spec = fixture_spec(kind)
    command = m._profile_command(spec, tmp_path, 3, m._profile_plan(spec), image=None)
    command.insert(2, 'dump')  # Profile dump validates/builds before upload; explicit prev_frames avoids probe.
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(COMFY_URL='http://127.0.0.1:1', COMFY_URL_VIDEO='http://127.0.0.1:1')
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    graph = json.loads(result.stdout)
    trimmed = next(n['inputs'] for n in graph.values() if n['class_type'] == 'TrimAudioDuration')
    assert trimmed['start_index'] == 2
    assert trimmed['duration'] == 49/24
    assert any(n['class_type'] == 'UNETLoader' for n in graph.values())
