"""Native IA2V topology and actual CLI admission with guarded backend calls."""
import json
from pathlib import Path
import shutil
import wave

import pytest

from creative_comfy_graph.minimax_h3 import build_ia2v
import test_h3_image_cli as image_cli


def test_native_audio_guide_conditions_sampler_and_preserves_source():
    graph = build_ia2v('singing', image_filename='anchor.png', audio_filename='track.wav', frames=141, seed=0)
    assert graph['6']['inputs']['first_frame'] == ['16', 0]
    assert graph['18']['inputs'] == {'audio': ['17', 0], 'start_index': 0.0, 'duration': 141 / 24}
    assert graph['19']['inputs'] == {'positive': ['6', 0], 'latent': ['6', 1],
                                    'audio_vae': ['5', 0], 'audio': ['18', 0], 'frame_idx': 0}
    assert graph['7']['inputs']['conditioning'] == ['19', 0]
    assert graph['14']['inputs']['audio'] == ['18', 0]
    assert not any(node['class_type'] == 'VAEDecodeAudio' for node in graph.values())
    assert graph['10']['inputs']['steps'] == 8 and graph['8']['inputs']['noise_seed'] == 0


@pytest.mark.parametrize('name', ['', None, '../track.wav', '/track.wav', 'track.wav [output]', 'track.txt'])
def test_unsafe_audio_names_rejected(name):
    with pytest.raises(ValueError):
        build_ia2v('singing', image_filename='anchor.png', audio_filename=name)


@pytest.fixture
def invoke(monkeypatch):
    child = image_cli.IMAGE_CHILD.replace(
        "if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:",
        "if event == 'subprocess.Popen':\n        assert Path(args[0]).name in {'ffmpeg', 'ffprobe'}\n        return\n    if event in {'socket.connect', 'socket.getaddrinfo', 'os.system'}:",
    )
    monkeypatch.setattr(image_cli, 'IMAGE_CHILD', child)
    return image_cli.invoke


def audio_file(tmp_path, seconds=6):
    filename = tmp_path / 'private-track.wav'
    with wave.open(str(filename), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b'\0\0' * round(seconds * 24_000))
    return str(filename)


@pytest.mark.parametrize('dump', [False, True])
def test_ia2v_help_never_uploads(invoke, legacy_dir, tmp_path, dump):
    result, calls = invoke(legacy_dir, tmp_path, [*(['dump'] if dump else []), 'h3-ia2v', '--help'])
    assert result.returncode == 0 and not calls
    assert '--audio' in result.stdout and '--image' in result.stdout


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='audio tools required')
@pytest.mark.parametrize('dump', [False, True])
def test_real_ia2v_cli_admits_audio_before_upload(invoke, legacy_dir, tmp_path, dump):
    args = [*(['dump'] if dump else []), 'h3-ia2v', '--prompt', 'singing', '--seed', '0',
            '--image', image_cli.image(tmp_path), '--audio', audio_file(tmp_path)]
    result, calls = invoke(legacy_dir, tmp_path, args)
    assert result.returncode == 0, result.stderr
    graph = json.loads(result.stdout) if dump else next(call['body']['prompt'] for call in calls if call['url'].endswith('/prompt'))
    assert graph['17']['inputs']['audio'].startswith('h3_audio_')
    assert graph['14']['inputs']['audio'] == ['18', 0]
    if dump:
        assert not calls
    else:
        uploads = [call for call in calls if call['url'].endswith('/upload/image')]
        assert len(uploads) == 2 and uploads[1]['upload_name'].endswith('.wav')
        assert len([call for call in calls if call['url'].endswith('/prompt')]) == 1


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='audio tools required')
@pytest.mark.parametrize('case', ['missing', 'empty', 'corrupt', 'short'])
def test_bad_audio_never_uploads_even_valid_image(invoke, legacy_dir, tmp_path, case):
    filename = tmp_path / 'bad.wav'
    if case == 'empty':
        filename.write_bytes(b'')
    elif case == 'corrupt':
        filename.write_bytes(b'not audio')
    elif case == 'short':
        filename = Path(audio_file(tmp_path, seconds=5))
    result, calls = invoke(legacy_dir, tmp_path, ['h3-ia2v', '--prompt', 'singing',
                          '--image', image_cli.image(tmp_path), '--audio', str(filename)])
    assert result.returncode == 2, result.stderr
    assert not calls


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='audio tools required')
def test_audio_upload_failure_does_not_submit(invoke, legacy_dir, tmp_path):
    result, calls = invoke(legacy_dir, tmp_path, ['h3-ia2v', '--prompt', 'singing',
                          '--image', image_cli.image(tmp_path), '--audio', audio_file(tmp_path)],
                          scenario='second-upload-error')
    assert result.returncode != 0 and 'NO_POST' in result.stderr
    assert not any(call['url'].endswith('/prompt') for call in calls)
