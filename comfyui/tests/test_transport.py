"""HTTP doubles only: submission must never retry an ambiguous POST."""
import importlib.util
import io
import json
from pathlib import Path
import urllib.error
import pytest

spec=importlib.util.spec_from_file_location('transport_core',Path(__file__).parents[1]/'scripts/core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


@pytest.fixture
def http(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    calls=[];clock=[0]
    monkeypatch.setenv('OPENCLAW_MEDIA_DIR',str(tmp_path))
    monkeypatch.setattr(core.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(core.time,'time',lambda:clock[0])
    monkeypatch.setattr(core.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    def install(responses):
        values=iter([{'queue_running': [], 'queue_pending': []}, *responses])
        def open_(request,**kwargs):
            url=request.full_url if hasattr(request,'full_url') else request
            method=request.get_method() if hasattr(request,'get_method') else 'GET'
            calls.append((method,url,kwargs.get('timeout')))
            result=next(values)
            if isinstance(result,Exception):raise result
            return io.BytesIO(json.dumps(result).encode() if isinstance(result,(dict,list)) else result)
        monkeypatch.setattr(core.urllib.request,'urlopen',open_)
        return calls
    return install


@pytest.mark.parametrize('failure', ['400','500','timeout','connection'])
def test_post_not_retried_preserves_diagnostics(http,tmp_path,failure):
    body=b'{"error":"invalid_prompt","node_errors":{"7":{"errors":["width below min"]}}}'
    error=(urllib.error.HTTPError('http://test/prompt',int(failure),'rejected',{},io.BytesIO(body))
           if failure.isdigit() else TimeoutError('lost response') if failure=='timeout' else urllib.error.URLError('lost connection'))
    calls=http([error]*40)
    with pytest.raises(RuntimeError) as caught:
        core._submit_and_wait({},tmp_path,timeout=5)
    assert len(calls)==2
    assert calls[0][1].endswith('/queue')
    if failure.isdigit():assert 'width below min' in str(caught.value) and 'node_errors' in str(caught.value)
    else:assert 'ambiguous' in str(caught.value).lower()


def entry(status='success',completed=True,kind='output'):
    return {'p':{'status':{'status_str':status,'completed':completed,'messages':[] if status in ('success','running') else [['execution_error',{'node_id':'7'}]]},
        'outputs':{'1':{'images':[{'filename':'preview.mp4','type':'input'}]},
                   '2':{'images':[{'filename':'result.mp4','type':kind}]}}}}


def test_waits_for_terminal_success_and_downloads_only_output(http,tmp_path):
    calls=http([{'prompt_id':'p'},entry('running',False),entry(),b'video'])
    assert core._submit_and_wait({},tmp_path,timeout=8)=='p'
    assert sum('/history/' in u for _,u,_ in calls)==2
    assert sum('/view?' in u for _,u,_ in calls)==1
    assert 'filename=result.mp4' in calls[-1][1] and 'type=output' in calls[-1][1]
    assert (tmp_path/'result.mp4').read_bytes()==b'video'
    assert not (tmp_path/'preview.mp4').exists()


@pytest.mark.parametrize('status,completed',[('error',False),('error',True),('interrupted',True)])
def test_terminal_errors_fail_even_with_previews(http,tmp_path,status,completed):
    calls=http([{'prompt_id':'p'},entry(status,completed)])
    with pytest.raises(RuntimeError,match='7'):
        core._submit_and_wait({},tmp_path,timeout=5)
    assert len(calls)==3


def test_selected_route_requires_output_asset(http,tmp_path):
    calls=http([{'prompt_id':'p'},entry(kind='temp')])
    with pytest.raises(RuntimeError,match='output'):
        core._submit_and_wait({},tmp_path,timeout=5,require_output=True)
    assert len(calls)==3


def test_history_get_retry_bounded_by_deadline_no_resubmit(http,tmp_path):
    calls=http([{'prompt_id':'p'}]+[urllib.error.URLError('down')]*40)
    with pytest.raises(TimeoutError,match='p'):
        core._submit_and_wait({},tmp_path,timeout=5)
    assert sum(method=='POST' for method,_,_ in calls)==1
    assert len(calls)<7


def test_running_outputs_timeout_without_download(http,tmp_path):
    calls=http([{'prompt_id':'p'}]+[entry('running',False)]*10)
    with pytest.raises(TimeoutError,match='p'):
        core._submit_and_wait({},tmp_path,timeout=5)
    assert not any('/view?' in url for _,url,_ in calls)


def test_download_failure_is_not_selected_route_success(http,tmp_path):
    http([{'prompt_id':'p'},entry(),urllib.error.URLError('download failed')])
    with pytest.raises(RuntimeError,match='download'):
        core._submit_and_wait({},tmp_path,timeout=5,require_output=True)


@pytest.mark.parametrize('response', [{'node_errors':{'7':'invalid'},'prompt_id':'p'}, {'error':'invalid'}, []])
def test_invalid_submission_response_fails_once(http,tmp_path,response):
    calls=http([response])
    with pytest.raises(RuntimeError,match='response|rejected'):
        core._submit_and_wait({},tmp_path,timeout=5)
    assert len(calls)==2
    assert calls[0][1].endswith('/queue')


def test_malformed_submission_json_is_ambiguous_and_not_retried(http,tmp_path):
    calls=http([b'not json'])
    with pytest.raises(RuntimeError,match='ambiguous'):
        core._submit_and_wait({},tmp_path,timeout=5)
    assert len(calls)==2
    assert calls[0][1].endswith('/queue')


def test_retry_transient_history_then_success(http,tmp_path):
    error=urllib.error.HTTPError('http://test/history/p',503,'down',{},io.BytesIO(b'down'))
    calls=http([{'prompt_id':'p'},error,entry(),b'video'])
    core._submit_and_wait({},tmp_path,timeout=5,require_output=True)
    assert sum(method=='POST' for method,_,_ in calls)==1
    assert (tmp_path/'result.mp4').exists()


def test_profile_main_requires_output_transport(monkeypatch,tmp_path):
    import sys
    scripts=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    s=importlib.util.spec_from_file_location('selected_cli',scripts/'comfy_graph.py')
    cli=importlib.util.module_from_spec(s);s.loader.exec_module(cli)
    calls=[]
    monkeypatch.setattr(cli,'_submit_and_wait',lambda *args,**kwargs:calls.append(kwargs))
    monkeypatch.setattr(sys,'argv',['comfy_graph.py','t2v','--model-profile','ltx-2.5','--seed','0'])
    cli.main()
    assert calls[0]['require_output'] is True
