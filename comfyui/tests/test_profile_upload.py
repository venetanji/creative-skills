"""Selected profile upload is single-attempt/fail-closed, unlike legacy upload."""
import importlib.util
import io
import json
from pathlib import Path
import urllib.error
import pytest


def load_cli(monkeypatch):
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('flf_cli',scripts/'comfy_graph.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    return cli


@pytest.mark.parametrize('slot',['first','last','audio','prev_video'])
@pytest.mark.parametrize('failure',['400','timeout','malformed','missing_name','wrong_type'])
def test_failed_upload_blocks_prompt_once(monkeypatch,tmp_path,slot,failure):
    import sys
    cli=load_cli(monkeypatch)
    file=tmp_path/('a.wav' if slot=='audio' else 'v.mp4' if slot=='prev_video' else 'f.png')
    if slot=='audio':
        import wave
        with wave.open(str(file),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000);f.writeframes(b'\0\0'*98000)
    else:file.write_bytes(b'fixture')
    calls=[]
    def open_(request,**kw):
        calls.append(request.full_url)
        if failure=='400':raise urllib.error.HTTPError(request.full_url,400,'bad',{},io.BytesIO(b'{"error":"original upload diagnostic"}'))
        if failure=='timeout':raise TimeoutError('original lost response')
        return io.BytesIO(b'not json' if failure=='malformed' else json.dumps({} if failure=='missing_name' else {'name':'x','type':'output'}).encode())
    monkeypatch.setattr(cli.urllib.request,'urlopen',open_)
    monkeypatch.setattr(cli,'_submit_and_wait',lambda *a,**k:pytest.fail('prompt submission after failed upload'))
    opts=dict(model_profile='ltx-2.5',seed=0);cmd='continuation' if slot=='prev_video' else 'flf2v'
    if cmd=='continuation':opts.update(prev_video=str(file),prev_frames=49)
    else:opts.update(first='remote-f.png',last='remote-l.png');opts[slot]=str(file)
    spec=tmp_path/'opts.json';spec.write_text(json.dumps(opts))
    monkeypatch.setattr(sys,'argv',['comfy_graph.py',cmd,'--input-json',str(spec)])
    with pytest.raises(RuntimeError) as caught:cli.main()
    assert len(calls)==1 and calls[0].endswith('/upload/image')
    assert 'upload' in str(caught.value).lower()
    if failure in ('400','timeout'):assert 'original' in str(caught.value)


def test_strict_upload_subfolder_return(monkeypatch,tmp_path):
    cli=load_cli(monkeypatch);file=tmp_path/'f.png';file.write_bytes(b'fixture')
    monkeypatch.setattr(cli.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b'{"name":"f.png","subfolder":"guides","type":"input"}'))
    import ltx_profile
    assert ltx_profile.strict_upload(str(file))=='guides/f.png'


def test_legacy_upload_fallback_unchanged(monkeypatch,tmp_path):
    cli=load_cli(monkeypatch);file=tmp_path/'f.png';file.write_bytes(b'fixture')
    def fail(*a,**k):raise TimeoutError('legacy failure')
    monkeypatch.setattr(cli.urllib.request,'urlopen',fail)
    assert cli.upload_if_local(str(file))==str(file)


@pytest.mark.parametrize('extra',[dict(last=''),dict(first=True),dict(first_guide_strength=True),
    dict(last_guide_strength=1.2),dict(frames=97.1),dict(seed=True),dict(image='x'),
    dict(audio_start_frame=0),dict(fast=True),dict(ckpt='x'),dict(seconds=4)])
def test_actual_json_invalid_flf_rejects_before_transport(tmp_path,extra):
    import os,subprocess,sys
    opts=dict(model_profile='ltx-2.5',first='f.png',last='l.png',frames=97,width=768,height=512,seed=0);opts.update(extra)
    path=tmp_path/'input.json';path.write_text(json.dumps(opts))
    env=dict(os.environ,COMFY_URL='http://127.0.0.1:1',COMFY_URL_VIDEO='http://127.0.0.1:1');env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/comfy_graph.py'),
        'flf2v','--input-json',str(path)],capture_output=True,text=True,env=env)
    assert result.returncode!=0 and 'ValueError' in result.stderr
    assert 'Connection refused' not in result.stderr and 'prompt_id:' not in result.stderr


_FILE_CASES = [(cmd, 'audio') for cmd in ('t2v', 'i2v', 'ia2v', 'continuation', 'flf2v')] + [
    ('t2v', 'image'), ('i2v', 'image'), ('ia2v', 'image'),
    ('t2v', 'prev_video'), ('continuation', 'prev_video'),
    ('flf2v', 'first'), ('flf2v', 'last')]


def selected_options(cmd):
    opts = dict(model_profile='ltx-2.5', seed=0, frames=97, width=768, height=512)
    if cmd in ('i2v', 'ia2v'):
        opts['image'] = 'guides/first.png'
    if cmd == 'ia2v':
        opts['audio'] = 'sounds/master.wav'
    if cmd == 'continuation':
        opts.update(prev_video='guides/A.mp4', prev_frames=97)
    if cmd == 'flf2v':
        opts.update(first='guides/first.png', last='guides/last.png')
    return opts


@pytest.mark.parametrize('profile', ['ltx-2.3', 'ltx-2.5'])
@pytest.mark.parametrize('dump', [False, True])
@pytest.mark.parametrize('value', ['', ' \t\n'])
@pytest.mark.parametrize('cmd,field', _FILE_CASES)
def test_actual_main_present_blank_file_rejects_before_transport(monkeypatch, tmp_path, profile, dump, value, cmd, field):
    import sys
    import ltx_profile
    cli = load_cli(monkeypatch)
    opts = selected_options(cmd)
    opts.update(model_profile=profile)
    # A valid local file must not be uploaded before another field rejects.
    local = tmp_path / 'first.png'
    local.write_bytes(b'fixture')
    if cmd == 'flf2v':
        opts['first'] = str(local)
    elif cmd in ('i2v', 'ia2v'):
        opts['image'] = str(local)
    opts[field] = value
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(opts))
    def forbidden(*args, **kwargs):
        pytest.fail('blank filename reached upload/probe/prompt transport')
    monkeypatch.setattr(ltx_profile, 'strict_upload', forbidden)
    monkeypatch.setattr(cli.urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(cli, '_submit_and_wait', forbidden)
    monkeypatch.setattr(cli, '_probe_frame_count', forbidden)
    monkeypatch.setattr(sys, 'argv', ['comfy_graph.py', *(['dump'] if dump else []), cmd, '--input-json', str(path)])
    with pytest.raises(ValueError):
        cli.main()


@pytest.mark.parametrize('cmd', ['t2v', 'i2v', 'continuation', 'flf2v'])
@pytest.mark.parametrize('dump', [False, True])
@pytest.mark.parametrize('supplied', [False, True])
def test_actual_main_optional_audio_omission_and_remote_reference(monkeypatch, tmp_path, capsys, cmd, dump, supplied):
    import sys
    cli = load_cli(monkeypatch)
    opts = selected_options(cmd)
    if supplied:
        opts['audio'] = 'sounds/master.wav'
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(opts))
    submitted = []
    monkeypatch.setattr(cli.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('remote reference uploaded'))
    monkeypatch.setattr(cli, '_submit_and_wait', lambda graph, *a, **k: submitted.append(graph))
    monkeypatch.setattr(sys, 'argv', ['comfy_graph.py', *(['dump'] if dump else []), cmd, '--input-json', str(path)])
    cli.main()
    graph = json.loads(capsys.readouterr().out) if dump else submitted[0]
    assert len(submitted) == (0 if dump else 1)
    audio = [n['inputs']['audio'] for n in graph.values() if n['class_type'] == 'LoadAudio']
    assert audio == (['sounds/master.wav'] if supplied else [])
    assert any(n['class_type'] == 'LTXVEmptyLatentAudio' for n in graph.values()) == (not supplied)
