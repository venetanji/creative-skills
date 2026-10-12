"""Actual caller CLI assembly, offline synthetic media, normal package environment."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('kind,key', [('drama', 'shots'), ('music', 'scenes')])
def test_actual_cli_assembly_retains_73_frames_and_master(kind, key, tmp_path):
    import yaml
    src = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=128x128:rate=24',
                    '-frames:v', '49', '-c:v', 'libx264', str(src)], check=True)
    audio = tmp_path / ('master.wav' if kind == 'drama' else 'song.mp3')
    # WAV content is intentional: ffmpeg probes content rather than suffix.
    with wave.open(str(audio), 'wb') as w:
        w.setparams((1, 2, 48000, 146000, 'NONE', 'not compressed'))
        w.writeframes(b'\0' * 292000)
    spec = {'video': {'model_profile': 'ltx-2.5', 'resolution': [128, 128]},
            'audio': {'file': str(audio)}, key: [
                {'label': 'A', 'duration_sec': 49/24, 'prompt': 'ball'},
                {'label': 'B', 'duration_sec': 1, 'prompt': 'ball', 'continue_from_prev': True}]}
    caller = ROOT / f'{kind}-video/scripts/{kind}_video.py'
    ms = importlib.util.spec_from_file_location('caller', caller)
    module = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(module)
    rows = module._profile_plan(spec)
    (tmp_path / key).mkdir()
    for idx, label in enumerate(('A', 'B'), 1):
        shutil.copyfile(src, tmp_path / key / f'{idx:03d}-{label}.mp4')
        (tmp_path / key / f'{idx:03d}-{label}.timeline.json').write_text(json.dumps({'timeline': rows[idx-1], 'generation_fingerprint':
            module._profile_helper().generation_fingerprint(spec, tmp_path, key, idx, rows)}))
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(spec))
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, str(caller), 'assemble', str(spec_path)], env=env, cwd=tmp_path, check=True)
    result = json.loads((tmp_path / 'final.timeline.json').read_text())
    assert result['assembly']['frames'] == 73
    assert result['assembly']['audio_samples'] == 146000
    assert result['clips'][1]['retain_span'] == [25, 49]
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(tmp_path / 'final.mp4')],
                            check=True, capture_output=True, text=True)
    assert json.loads(probe.stdout)['streams'][0]['nb_read_frames'] == '73'
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(tmp_path / 'final.mp4'), '-f', 'null', '-'], check=True)
    # Editing visible duration invalidates metadata, rather than replaying overlap.
    spec[key][1]['duration_sec'] = .5
    with pytest.raises(ValueError, match='stale timeline'):
        module.cmd_assemble(spec, tmp_path)
