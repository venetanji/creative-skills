"""Both real submission routes with only HTTP and the clock replaced."""
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest

EMPTY = {'queue_running': [], 'queue_pending': []}
DONE = {'p': {'status': {'status_str': 'success', 'completed': True}, 'outputs': {}}}


@pytest.fixture(params=['core', 'standalone'])
def route(request, monkeypatch, tmp_path):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    filename = 'core.py' if request.param == 'core' else 'comfy_run.py'
    spec = importlib.util.spec_from_file_location('queue_test_' + request.param, scripts / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [0.0]
    sleeps, calls = [], []
    monkeypatch.setenv('OPENCLAW_MEDIA_DIR', str(tmp_path))
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'time', lambda: clock[0])
    def sleep(seconds):
        assert seconds > 0
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(module.time, 'sleep', sleep)
    def install(responses):
        values = iter(responses)
        def open_(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else req
            method = req.get_method() if hasattr(req, 'get_method') else 'GET'
            assert url.startswith(module.BASE + '/'), url
            calls.append((method, url.removeprefix(module.BASE), clock[0], kwargs['timeout'],
                          req.data if hasattr(req, 'data') else None))
            response = next(values)
            if isinstance(response, tuple):
                response, elapsed = response
                clock[0] += elapsed
            if isinstance(response, Exception):
                raise response
            return io.BytesIO(response if isinstance(response, bytes) else json.dumps(response).encode())
        monkeypatch.setattr(module.urllib.request, 'urlopen', open_)
    def submit(workflow=None, timeout=5):
        graph = {} if workflow is None else workflow
        if request.param == 'core':
            return module._submit_and_wait(graph, tmp_path, timeout=timeout)
        return module.run_workflow(graph, tmp_path, no_wait=True, timeout=timeout)[0]
    return module, request.param, submit, install, clock, sleeps, calls


@pytest.mark.parametrize('field', ['queue_running', 'queue_pending'])
def test_busy_queue_expires_without_post(route, field):
    _, _, submit, install, clock, sleeps, calls = route
    busy = dict(EMPTY, **{field: [[1, 'other-job']]})
    install([busy] * 20)
    with pytest.raises(TimeoutError, match='NO_POST'):
        submit()
    assert calls and all(c[0:2] == ('GET', '/queue') for c in calls)
    assert clock[0] == 5
    assert sleeps == [2, 2, 1]
    assert len(calls) == 3
    assert all(0 < c[3] <= 5 - c[2] for c in calls)


@pytest.mark.parametrize('field', ['queue_running', 'queue_pending'])
def test_eventually_empty_posts_immediately_after_fresh_get(route, field):
    _, _, submit, install, clock, sleeps, calls = route
    busy = dict(EMPTY, **{field: [[1, 'other-job']]})
    install([busy, EMPTY, {'prompt_id': 'p'}, DONE])
    assert submit({'1': {'class_type': 'Fixture', 'inputs': {}}}) == 'p'
    assert [(c[0], c[1]) for c in calls[:3]] == [('GET', '/queue'), ('GET', '/queue'), ('POST', '/prompt')]
    assert calls[1][2] == calls[2][2] == 2
    assert json.loads(calls[2][4]) == {'prompt': {'1': {'class_type': 'Fixture', 'inputs': {}}}}
    assert sum(c[0] == 'POST' for c in calls) == 1
    assert sleeps == [2]


@pytest.mark.parametrize('bad', [
    {}, {'queue_running': []}, {'queue_pending': []}, [], None,
    {'queue_running': None, 'queue_pending': []},
    {'queue_running': [], 'queue_pending': None},
    {'queue_running': {}, 'queue_pending': []},
    {'queue_running': [], 'queue_pending': '[]'},
    {'queue_running': False, 'queue_pending': []},
    b'not json', b'\xff',
])
def test_malformed_queue_fails_closed_without_post(route, bad):
    _, _, submit, install, _, sleeps, calls = route
    install([bad])
    with pytest.raises(RuntimeError, match='NO_POST'):
        submit()
    assert [(c[0], c[1]) for c in calls] == [('GET', '/queue')]
    assert sleeps == []


def http_error(code):
    return urllib.error.HTTPError('http://test/queue', code, 'fixture', {}, io.BytesIO(b'queue diagnostic'))


@pytest.mark.parametrize('failure', [urllib.error.URLError('connect'), TimeoutError('read'), ConnectionError('reset'), http_error(503)])
def test_read_only_transient_queue_retry_then_one_post(route, failure):
    _, _, submit, install, _, sleeps, calls = route
    install([failure, EMPTY, {'prompt_id': 'p'}, DONE])
    assert submit() == 'p'
    assert [(c[0], c[1]) for c in calls[:3]] == [('GET', '/queue'), ('GET', '/queue'), ('POST', '/prompt')]
    assert sleeps == [2]
    assert sum(c[0] == 'POST' for c in calls) == 1


@pytest.mark.parametrize('code', [400, 401, 403, 404, 429])
def test_permanent_queue_http_fails_closed(route, code):
    _, _, submit, install, _, sleeps, calls = route
    install([http_error(code)])
    with pytest.raises(RuntimeError, match='NO_POST') as caught:
        submit()
    assert str(code) in str(caught.value) and 'queue diagnostic' in str(caught.value)
    assert [(c[0], c[1]) for c in calls] == [('GET', '/queue')]
    assert sleeps == []


def test_transient_queue_failures_exhaust_deadline_without_post(route):
    _, _, submit, install, clock, sleeps, calls = route
    install([urllib.error.URLError('down')] * 20)
    with pytest.raises(TimeoutError, match='NO_POST'):
        submit()
    assert clock[0] == 5
    assert sleeps == [2, 3]
    assert len(calls) == 2
    assert all(c[:2] == ('GET', '/queue') for c in calls)


def test_late_empty_response_cannot_admit_after_deadline(route):
    _, _, submit, install, _, _, calls = route
    install([(EMPTY, 5), {'prompt_id': 'p'}, DONE])
    with pytest.raises(TimeoutError, match='NO_POST'):
        submit()
    assert [(c[0], c[1]) for c in calls] == [('GET', '/queue')]


def test_every_invocation_rechecks_queue(route):
    _, _, submit, install, _, _, calls = route
    install([EMPTY, {'prompt_id': 'p'}, DONE] if route[1] == 'core' else [EMPTY, {'prompt_id': 'p'}])
    assert submit() == 'p'
    install([dict(EMPTY, queue_pending=[[1, 'another']])] * 20)
    with pytest.raises(TimeoutError, match='NO_POST'):
        submit()
    assert sum(c[0] == 'POST' for c in calls) == 1
    assert calls[-1][:2] == ('GET', '/queue')


def test_serialization_failure_has_zero_http(route):
    _, _, submit, install, _, _, calls = route
    install([])
    with pytest.raises(TypeError):
        submit({'bad': object()})
    assert calls == []


@pytest.mark.parametrize('failure', [urllib.error.URLError('lost response'), TimeoutError('lost response'), http_error(500), b'not json'])
def test_admitted_post_failure_is_never_retried(route, failure):
    _, _, submit, install, _, sleeps, calls = route
    install([EMPTY, failure])
    with pytest.raises(RuntimeError, match='resubmit|resubmission'):
        submit()
    assert [(c[0], c[1]) for c in calls] == [('GET', '/queue'), ('POST', '/prompt')]
    assert sleeps == []


@pytest.mark.parametrize('route', ['core'], indirect=True)
def test_core_queue_wait_consumes_history_deadline(route):
    _, kind, submit, install, clock, _, calls = route
    busy = dict(EMPTY, queue_running=[[1, 'other']])
    install([busy, EMPTY, {'prompt_id': 'p'}] + [{}] * 10)
    with pytest.raises(TimeoutError, match='prompt_id=p.*do not resubmit'):
        submit(timeout=5)
    assert clock[0] == 5
    assert sum(c[0] == 'POST' for c in calls) == 1
    histories = [c for c in calls if c[1] == '/history/p']
    assert histories and all(c[3] <= 5 - c[2] for c in histories)


@pytest.mark.parametrize('route', ['standalone'], indirect=True)
def test_standalone_history_retains_separate_budget(route):
    module, kind, _, install, clock, _, calls = route
    busy = dict(EMPTY, queue_pending=[[1, 'other']])
    install([busy, EMPTY, {'prompt_id': 'p'}] + [{}] * 10)
    with pytest.raises(TimeoutError, match='prompt_id=p.*do not resubmit'):
        module.run_workflow({}, no_wait=False, timeout=5)
    assert clock[0] == 8  # legacy history polling interval may cross its own deadline
    assert sum(c[0] == 'POST' for c in calls) == 1


def test_expired_admission_budget_has_zero_http(route):
    _, _, submit, install, _, _, calls = route
    install([])
    with pytest.raises(TimeoutError, match='NO_POST'):
        submit(timeout=0)
    assert calls == []


def test_deadline_crossed_between_empty_read_and_post_is_unsubmitted(route, monkeypatch):
    module, _, submit, install, _, _, calls = route
    reads = [0]
    def monotonic():
        reads[0] += 1
        return 0 if reads[0] <= 3 else 5
    monkeypatch.setattr(module.time, 'monotonic', monotonic)
    install([EMPTY, {'prompt_id': 'p'}, DONE])
    with pytest.raises(TimeoutError, match='NO_POST'):
        submit(timeout=5)
    assert [(c[0], c[1]) for c in calls] == [('GET', '/queue')]
