"""Execute actual CLI with external HTTP replaced; never contact a backend."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_runtime import clean_environment

# This double sits below real submission/history/download logic. Unexpected
# HTTP routes and socket/media/GPU activity fail instead of reaching the host.
CHILD = r'''
import io, json, runpy, sys, urllib.error, urllib.request
from pathlib import Path
script, scenario, *args = sys.argv[1:]
sys.path[0] = str(Path(script).parent)
sys.argv = [script, *args]
calls = []
def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
        raise AssertionError('forbidden side effect: ' + event)
sys.addaudithook(guard)
def fake_http(request, **kwargs):
    url = request.full_url if hasattr(request, 'full_url') else request
    method = request.get_method() if hasattr(request, 'get_method') else 'GET'
    body = json.loads(request.data) if hasattr(request, 'data') and request.data else None
    calls.append({'url':url, 'method':method, 'body':body})
    if url == 'http://video.invalid:8190/queue' and method == 'GET':
        return io.BytesIO(b'{"queue_running": [], "queue_pending": []}')
    if url == 'http://video.invalid:8190/prompt' and method == 'POST':
        if scenario == 'ambiguous':
            raise urllib.error.URLError('lost POST response')
        return io.BytesIO(json.dumps({'node_errors':{'6':'bad'}} if scenario == 'reject'
                                    else {'prompt_id':'fixture-id','node_errors':{}}).encode())
    if url == 'http://video.invalid:8190/history/fixture-id' and method == 'GET':
        status = {'status_str':'success','completed':True,'messages':[['execution_success',{}]]}
        if scenario == 'error':
            status = {'status_str':'error','completed':True,'messages':[['execution_error',{'exception_message':'offline error'}]]}
        outputs = {} if scenario == 'no-output' else {'15':{'images':[
            {'filename':'fixture_00001.mp4','subfolder':'video','type':'output'}]}}
        return io.BytesIO(json.dumps({'fixture-id':{'status':status,'outputs':outputs}}).encode())
    if url.startswith('http://video.invalid:8190/view?filename=fixture_00001.mp4&subfolder=video&type=output&'):
        if scenario == 'download-error':
            raise urllib.error.URLError('download failed')
        return io.BytesIO(b'offline-download-fixture')
    raise AssertionError('unexpected HTTP: ' + method + ' ' + url)
urllib.request.urlopen = fake_http
try:
    runpy.run_path(script, run_name='__main__')
finally:
    assert not any(n == 'torch' or n.startswith('comfy.') for n in sys.modules)
    print('H3_TEST_RECEIPT=' + json.dumps(calls), file=sys.stderr)
'''


def invoke(legacy_dir, tmp_path, args, scenario='success', *, notify_env=False):
    env = clean_environment(sys.executable)
    env.update(COMFY_URL_VIDEO='http://video.invalid:8190/',
               COMFY_URL='http://default.invalid:8188', COMFY_URL_FLUX='http://flux.invalid:8188',
               OPENCLAW_MEDIA_DIR=str(tmp_path / 'outputs'))
    env.pop('OPENCLAW_NOTIFY_TARGET', None)
    if notify_env:
        env['OPENCLAW_NOTIFY_TARGET'] = 'test-env'
    result = subprocess.run([sys.executable, '-c', CHILD,
        str(legacy_dir / 'comfy_graph.py'), scenario, *args],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    receipts = [line.split('=',1)[1] for line in result.stderr.splitlines() if line.startswith('H3_TEST_RECEIPT=')]
    assert len(receipts) == 1, result.stderr
    return result, json.loads(receipts[0])


def test_h3_actual_dump_and_json_cli_override(legacy_dir, tmp_path):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(dict(prompt='one bell', width=896,height=512,frames=141,seed=42)))
    result, calls = invoke(legacy_dir, tmp_path, ['dump','h3-t2v','--input-json',str(spec),'--seed','0'])
    assert result.returncode == 0, result.stderr
    graph = json.loads(result.stdout)
    assert graph['8']['inputs']['noise_seed'] == 0
    assert graph['6']['inputs']['length'] == 141
    assert graph['6']['inputs']['width'] == 896
    assert graph['6']['inputs']['height'] == 512
    assert calls == []
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('args', [
    ['--audio','local.wav'], ['--image','local.png'], ['--references','ref.png'],
    ['--negative','bad'], ['--cfg','1'], ['--steps','4'], ['--fps','25'],
    ['--unet','/unsafe/model'], ['--model-profile','ltx-2.5'], ['--seconds','5'],
    ['--wat'], ['--wid','864'], ['unexpected'], ['--width'], ['--input-json'],
    ['--width','864.0'], ['--seed','nan'], ['--frames','125'], ['--seed','-1'],
    ['--width','865'], ['--timeout','0'], ['--prefix','../outside'], ['--prompt',''],
])
def test_invalid_flags_fail_locally_without_output(legacy_dir, tmp_path, args):
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v','--prompt','bell',*args])
    assert result.returncode == 2, result.stderr
    assert calls == []
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('spec', [
    {'prompt':'bell','width':True}, {'prompt':'bell','height':480.0},
    {'prompt':'bell','frames':124.0}, {'prompt':'bell','seed':False},
    {'prompt':'bell','seed':float('inf')}, {'prompt':'bell','width':'864'},
    {'prompt':'bell','audio':'local.wav'}, {'prompt':'bell','negative':None},
    {'prompt':'bell','steps':8}, {'prompt':'bell','control_video':'x'},
    {'prompt':'bell','timeout':True}, {'prompt':'bell','output_dir':True},
    {'prompt':'bell','notify_target':[]}, {'prompt':'bell','caption_template':{}},
    {'prompt':'bell','input_json':'other.json'}, [], {'prompt':'bell','seed':None},
    {'prompt':'bell','notify-target':'a','notify_target':'b'},
])
def test_json_invalid_types_and_options_rejected_before_transport(legacy_dir, tmp_path, spec):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v','--input-json',str(path)])
    assert result.returncode == 2, result.stderr
    assert calls == []
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('prefix', [[], ['dump']])
def test_help_exits_successfully_without_submission(legacy_dir, tmp_path, prefix):
    result, calls = invoke(legacy_dir, tmp_path, [*prefix,'h3-t2v','--help'])
    assert result.returncode == 0, result.stderr
    assert 'FL eight-step' in result.stdout
    assert calls == []


def test_missing_prompt_no_submission(legacy_dir, tmp_path):
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v'])
    assert result.returncode == 2
    assert calls == []


def test_actual_dispatch_posts_once_then_downloads_native_output(legacy_dir, tmp_path):
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v','--prompt','one bell',
        '--seed','0','--frames','124','--width','864','--height','480','--prefix','video/offline'])
    assert result.returncode == 0, result.stderr
    assert [c['method'] for c in calls] == ['GET','POST','GET','GET']
    assert calls[0]['url'] == 'http://video.invalid:8190/queue'
    assert calls[1]['url'] == 'http://video.invalid:8190/prompt'
    graph = calls[1]['body']['prompt']
    assert graph['8']['inputs']['noise_seed'] == 0
    assert graph['15']['inputs']['filename_prefix'] == 'video/offline'
    assert graph['10']['inputs']['steps'] == 8
    assert (tmp_path / 'outputs/fixture_00001.mp4').read_bytes() == b'offline-download-fixture'
    assert 'saved:' in result.stdout


@pytest.mark.parametrize('scenario, message, count', [
    ('error','failed',2), ('no-output','without a downloadable',2),
    ('download-error','Failed to download',3), ('reject','Prompt rejected',1),
    ('ambiguous','do not resubmit',1),
])
def test_terminal_transport_failures_are_nonzero_no_resubmission(legacy_dir, tmp_path, scenario, message, count):
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v','--prompt','bell'], scenario)
    assert result.returncode != 0
    assert message in result.stderr
    assert len(calls) == count + 1
    assert calls[0]['url'] == 'http://video.invalid:8190/queue'
    assert sum(c['method'] == 'POST' for c in calls) == 1
    assert not (tmp_path / 'outputs/fixture_00001.mp4').exists()


@pytest.mark.parametrize('template', ['{unknown}', '{'], ids=['unknown', 'incomplete'])
@pytest.mark.parametrize('source', ['cli', 'json'])
def test_caption_red_regression_before_transport(legacy_dir, tmp_path, template, source):
    if source == 'json':
        path = tmp_path / 'spec.json'
        path.write_text(json.dumps({'prompt': 'bell', 'notify-target': 'test',
                                    'caption-template': template}))
        args = ['--input-json', str(path)]
    else:
        args = ['--prompt', 'bell', '--notify-target', 'test', '--caption-template', template]
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v', *args])
    assert result.returncode == 2, (result.stdout, result.stderr, calls)
    assert calls == []
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('template', [
    '{unknown}', '{', '}', '{}', '{0}', '{filename.name}', '{filename[0]}',
    '{prompt:d}', '{filename!x}', '{prompt:{id}}', '{prompt:1000000000}',
    '{prompt:.1000000000s}', '{prompt:1025}', '{prompt:.1025s}',
    '{prompt:+s}', '{prompt:=10}', 'x' * 4097, '{prompt}' * 33,
    '{prompt:1024}' * 17,
])
@pytest.mark.parametrize('source', ['cli', 'json-dash', 'json-underscore'])
@pytest.mark.parametrize('mode', ['notify', 'no-notify', 'env', 'dump'])
def test_caption_invalid_admission_zero_http(legacy_dir, tmp_path, template, source, mode):
    opts = {'prompt': 'bell', 'caption_template': template}
    if mode == 'notify':
        opts['notify_target'] = 'test'
    if source == 'cli':
        args = [part for key, value in opts.items()
                for part in ('--' + key.replace('_', '-'), value)]
    else:
        path = tmp_path / 'spec.json'
        spec = {key.replace('_', '-') if source == 'json-dash' else key: value
                for key, value in opts.items()}
        path.write_text(json.dumps(spec))
        args = ['--input-json', str(path)]
    result, calls = invoke(legacy_dir, tmp_path,
                           [*(['dump'] if mode == 'dump' else []), 'h3-t2v', *args],
                           notify_env=mode == 'env')
    assert result.returncode == 2, result.stderr
    assert 'caption_template' in result.stderr
    assert calls == []
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('template, expected', [
    ('{filename} {prompt}', 'fixture_00001.mp4 bell'),
    ('{filename}|{workflow}|{id}|{ts}|{prompt}', 'fixture_00001.mp4|||'),
    ('{{bell}} {filename!s} {prompt!r} {prompt!a}', "{bell} fixture_00001.mp4 'bell' 'bell'"),
    ('{prompt:*>8s}|{prompt:^8}|{filename:.7s}', '****bell|  bell  |fixture'),
    ('{prompt:05s}|{prompt:>6.3s}', 'bell0|   bel'),
    ('{prompt:1024.1024s}', 'bell'),
    ('{prompt}' * 32, 'bell' * 32),
    ('{prompt:1024}' * 16, 'bell'),
    ('x' * 4096, 'x' * 4096),
    ('{prompt:.0s}{{}}', '{}'),
])
@pytest.mark.parametrize('source', ['cli', 'json-env'])
def test_caption_valid_notification_transport(legacy_dir, tmp_path, template, expected, source):
    if source == 'json-env':
        path = tmp_path / 'spec.json'
        path.write_text(json.dumps({'prompt': 'bell', 'caption-template': template}))
        args = ['--input-json', str(path)]
    else:
        args = ['--prompt', 'bell', '--notify-target', 'test', '--caption-template', template]
    result, calls = invoke(legacy_dir, tmp_path, ['h3-t2v', *args], notify_env=source == 'json-env')
    assert result.returncode == 0, result.stderr
    assert [c['method'] for c in calls] == ['GET', 'POST', 'GET', 'GET']
    assert calls[0]['url'] == 'http://video.invalid:8190/queue'
    assert calls[1]['url'] == 'http://video.invalid:8190/prompt'
    assert (tmp_path / 'outputs/fixture_00001.mp4').read_bytes() == b'offline-download-fixture'
    target = 'test-env' if source == 'json-env' else 'test'
    assert '[NOTIFY_URL 0] target=' + target in result.stdout
    assert 'caption=' + expected in result.stdout
