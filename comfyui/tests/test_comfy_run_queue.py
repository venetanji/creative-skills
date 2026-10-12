"""Actual standalone CLI, guarded from network/GPU/package fallback."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

CHILD = r'''
import io, json, runpy, sys, time, urllib.request
from pathlib import Path
script, scenario, *args = sys.argv[1:]
sys.path[0] = str(Path(script).parent)
sys.argv = [script, *args]
calls, clock = [], [0.0]
def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'}:
        raise AssertionError('forbidden: ' + event)
sys.addaudithook(guard)
class RejectGraphPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'creative_comfy_graph' or fullname.startswith('creative_comfy_graph.'):
            raise AssertionError('standalone must not import graph package')
sys.meta_path.insert(0, RejectGraphPackage())
time.monotonic = lambda: clock[0]
time.sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
def http(request, **kwargs):
    url = request.full_url if hasattr(request, 'full_url') else request
    method = request.get_method() if hasattr(request, 'get_method') else 'GET'
    calls.append([method, url, clock[0]])
    if (method, url) == ('GET', 'http://standalone.invalid/queue'):
        if scenario == 'malformed':
            return io.BytesIO(b'{"queue_running": []}')
        running = [[1, 'other']] if scenario == 'busy-running' else []
        pending = [[1, 'other']] if scenario == 'busy-pending' else []
        return io.BytesIO(json.dumps({'queue_running': running, 'queue_pending': pending}).encode())
    if (method, url) == ('POST', 'http://standalone.invalid/prompt'):
        assert json.loads(request.data) == {'prompt': {'1': {'class_type': 'OfflineFixture', 'inputs': {}}}}
        return io.BytesIO(b'{"prompt_id": "standalone-p"}')
    raise AssertionError('unexpected HTTP: ' + method + ' ' + url)
urllib.request.urlopen = http
try:
    runpy.run_path(script, run_name='__main__')
finally:
    assert not any(n == 'torch' or n.startswith('comfy.') for n in sys.modules)
    assert 'creative_comfy_graph' not in sys.modules and 'core' not in sys.modules
    print('QUEUE_TEST_RECEIPT=' + json.dumps(calls), file=sys.stderr)
'''


@pytest.mark.parametrize('scenario', ['busy-running', 'busy-pending', 'malformed', 'success'])
def test_actual_no_wait_still_requires_admission(tmp_path, scenario):
    workflow = tmp_path / 'workflow.json'
    workflow.write_text(json.dumps({'1': {'class_type': 'OfflineFixture', 'inputs': {}}}))
    script = Path(__file__).parents[1] / 'scripts/comfy_run.py'
    env = dict(os.environ, COMFY_URL='http://standalone.invalid',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', CHILD, str(script), scenario,
                             str(workflow), '--timeout', '5', '--no-wait'],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    receipts = [line.split('=', 1)[1] for line in result.stderr.splitlines()
                if line.startswith('QUEUE_TEST_RECEIPT=')]
    assert len(receipts) == 1, result.stderr
    calls = json.loads(receipts[0])
    if scenario == 'success':
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'standalone-p'
        assert calls == [['GET', 'http://standalone.invalid/queue', 0],
                         ['POST', 'http://standalone.invalid/prompt', 0]]
    else:
        assert result.returncode != 0
        assert 'NO_POST' in result.stderr
        assert all(c[:2] == ['GET', 'http://standalone.invalid/queue'] for c in calls)
        assert len(calls) == (1 if scenario == 'malformed' else 3)


def test_generic_post_helper_does_not_gate(monkeypatch):
    import importlib.util
    import io
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('standalone_upload_double', scripts / 'comfy_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def http(request, **kwargs):
        calls.append((request.get_method(), request.full_url))
        return io.BytesIO(b'{"name": "fixture"}')
    monkeypatch.setattr(module.urllib.request, 'urlopen', http)
    assert module.post_json('/upload/custom', {'fixture': True}) == {'name': 'fixture'}
    assert calls == [('POST', module.BASE + '/upload/custom')]
