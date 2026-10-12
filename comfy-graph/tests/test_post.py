"""Post graphs preserve server descriptions; clip staging stays in the adapter."""
import builtins
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock, call

import pytest

from test_characterization import capture_module, fixtures


def test_post_package_parity(legacy_dir):
    tool = capture_module()
    expected_api, expected_graphs, expected_errors = fixtures()
    actual_api, actual_graphs, actual_errors = tool.capture(legacy_dir, package_family='post')
    assert actual_api['builders'] == expected_api['builders']
    assert actual_api['exports'] == expected_api['exports']
    assert actual_api['constants'] == expected_api['constants']
    assert json.dumps(actual_graphs) == json.dumps(expected_graphs)
    assert actual_errors == expected_errors
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        package = importlib.import_module('creative_comfy_graph.post')
        assert modules['post'].Path is Path
        assert modules['post'].WorkflowGraph is package.WorkflowGraph
        for name, (family, _) in tool.BUILDERS.items():
            if family == 'post':
                assert getattr(modules['post'], name) is getattr(package, name)
        assert not hasattr(package, 'stage_clips_for_concat')


@pytest.mark.parametrize('prefix', ['/synthetic/nonexistent', 'missing-server-relative'])
def test_post_paths_are_descriptions(legacy_dir, prefix):
    tool = capture_module()
    audio = prefix + '/song.wav'
    video = prefix + '/clip.mp4'
    folder = prefix + '/staged'
    with tool.boundaries(legacy_dir) as attempts:
        post = importlib.import_module('creative_comfy_graph.post')
        stems = post.extract_stems(audio)
        assert stems == {
            '0': {'class_type': 'MelBandRoFormerModelLoader', 'inputs': {'model_name': 'MelBandRoformer_fp16.safetensors'}},
            '1': {'class_type': 'LoadAudio', 'inputs': {'audio': audio, 'audioUI': ''}},
            '2': {'class_type': 'MelBandRoFormerSampler', 'inputs': {'model': ['0', 0], 'audio': ['1', 0]}},
            '3': {'class_type': 'SaveAudio', 'inputs': {'audio': ['2', 0], 'filename_prefix': 'stems_stem0'}},
            '4': {'class_type': 'SaveAudio', 'inputs': {'audio': ['2', 1], 'filename_prefix': 'stems_stem1'}},
        }
        transcript = post.transcribe(audio)
        assert transcript == {
            '0': {'class_type': 'LoadAudio', 'inputs': {'audio': audio, 'audioUI': ''}},
            '1': {'class_type': 'Apply Whisper', 'inputs': {'audio': ['0', 0], 'model': 'large-v3-turbo', 'language': 'auto', 'prompt': ''}},
            '2': {'class_type': 'Save SRT', 'inputs': {'alignment': ['1', 1], 'name': 'transcript_segments'}},
            '3': {'class_type': 'Save SRT', 'inputs': {'alignment': ['1', 2], 'name': 'transcript_words'}},
            '4': {'class_type': 'ShowText|pysssss', 'inputs': {'text': ['1', 0]}},
            '5': {'class_type': 'ShowText|pysssss', 'inputs': {'text': ['2', 0]}},
            '6': {'class_type': 'ShowText|pysssss', 'inputs': {'text': ['3', 0]}},
        }
        assert list(transcript) == ['0', '1', '2', '3', '4', '5', '6']
        assert post.concat_videos_ffmpeg(folder, audio_filename=audio, filename_prefix='unused') == {
            '0': {'class_type': 'MergingVideoByPlenty', 'inputs': {'video_path': '/app/ComfyUI/input/' + folder, 'output_path': '/app/ComfyUI/output'}},
            '1': {'class_type': 'AddAudio', 'inputs': {'video_path': ['0', 0], 'audio_from': 'audio_file', 'file_path': '/app/ComfyUI/input/' + audio, 'delay_play': 0, 'output_path': '/app/ComfyUI/output'}},
            '2': {'class_type': 'ShowText|pysssss', 'inputs': {'text': ['1', 0]}},
        }
        assert post.concat_videos([video], audio_filename=audio, trim_starts=[0.5], trim_durations=[1.25]) == {
            '0': {'class_type': 'LoadVideo', 'inputs': {'file': video}},
            '1': {'class_type': 'GetVideoComponents', 'inputs': {'video': ['0', 0]}},
            '2': {'class_type': 'GetImageRangeFromBatch', 'inputs': {'images': ['1', 0], 'start_index': 12, 'num_frames': 30}},
            '3': {'class_type': 'CreateVideo', 'inputs': {'images': ['2', 0], 'fps': 24.0, 'audio': ['1', 1]}},
            '4': {'class_type': 'SaveVideo', 'inputs': {'video': ['3', 0], 'filename_prefix': 'concat', 'format': 'mp4', 'codec': 'h264'}},
        }
        assert not attempts


def test_stage_adapter_boundary(legacy_dir, monkeypatch):
    tool = capture_module()
    with tool.boundaries(legacy_dir) as attempts, tool.legacy_modules(legacy_dir) as modules:
        package = importlib.import_module('creative_comfy_graph.post')
        assert not hasattr(package, 'stage_clips_for_concat')
        upload = Mock(side_effect=['server-a-renamed.mp4', 'server-b-renamed.mp4'])
        monkeypatch.setattr(modules['core'], 'upload_image', upload)
        result = modules['post'].stage_clips_for_concat(
            [Path('/synthetic/nonexistent/a.mp4'), 'missing-relative/b.mp4'], 'ordered-clips')
        assert result == ['server-a-renamed.mp4', 'server-b-renamed.mp4']
        assert upload.call_args_list == [
            call('/synthetic/nonexistent/a.mp4', subfolder='ordered-clips', upload_as='001_a.mp4'),
            call('missing-relative/b.mp4', subfolder='ordered-clips', upload_as='002_b.mp4'),
        ]
        assert modules['post'].stage_clips_for_concat([], 'empty') == []
        assert upload.call_count == 2
        assert not attempts


def test_post_package_no_transport_import(legacy_dir, monkeypatch):
    tool = capture_module()
    name = 'creative_comfy_graph.post'
    saved = sys.modules.pop(name, None)
    parent = importlib.import_module('creative_comfy_graph')
    previous = vars(parent).pop('post', None)
    original = builtins.__import__
    forbidden = {'core', 'flux2', 'tts', 'post', 'ltx2', 'comfy_graph', 'urllib', 'socket', 'subprocess', 'pathlib'}

    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in forbidden, 'package imported transport/adapter: ' + name
        return original(name, *args, **kwargs)

    try:
        with tool.boundaries(legacy_dir) as attempts, monkeypatch.context() as patcher:
            patcher.setattr(builtins, '__import__', guarded)
            post = importlib.import_module(name)
            assert not hasattr(post, 'stage_clips_for_concat')
            assert post.extract_stems('server.wav')
            assert post.transcribe('server.wav')
            assert post.concat_videos_ffmpeg('staged', audio_filename='server.wav')
            assert post.concat_videos(['server.mp4'])
            assert not attempts
    finally:
        sys.modules.pop(name, None)
        if saved is not None:
            sys.modules[name] = saved
        vars(parent).pop('post', None)
        if previous is not None:
            setattr(parent, 'post', previous)
