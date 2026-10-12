"""Real CLI, guarded HTTP doubles; no sockets, GPU or subprocess media work."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from test_h3_cli import CHILD
from test_runtime import clean_environment
PIL=pytest.importorskip('PIL.Image')
IMAGE_CHILD=CHILD.replace("body = json.loads(request.data) if hasattr(request, 'data') and request.data else None", "body = json.loads(request.data) if hasattr(request, 'data') and request.data and not url.endswith('/upload/image') else None").replace("if url == 'http://video.invalid:8190/queue'", """if url == 'http://video.invalid:8190/upload/image' and method == 'POST':
        import re
        name = re.search(b'filename="([^"]+)"', request.data).group(1).decode()
        calls[-1]['upload_name'] = name
        if scenario == 'upload-error' or (scenario == 'second-upload-error' and len(calls) == 2):
            raise urllib.error.URLError('offline upload error')
        if scenario == 'unsafe-upload':
            name = '../escape.png'
        response = {'name':name,'type':'input','subfolder':''}
        if scenario == 'upload-wrong-type': response['type'] = 'output'
        elif scenario == 'upload-missing-type': response.pop('type')
        elif scenario == 'upload-wrong-folder': response['subfolder'] = 'unexpected'
        elif scenario == 'upload-missing-folder': response.pop('subfolder')
        return io.BytesIO(json.dumps(response).encode())
    if url == 'http://video.invalid:8190/queue'""")

def invoke(legacy_dir,tmp_path,args,scenario='success'):
    env=clean_environment(sys.executable)
    env.update(COMFY_URL_VIDEO='http://video.invalid:8190/',COMFY_URL='http://default.invalid',COMFY_URL_FLUX='http://flux.invalid',OPENCLAW_MEDIA_DIR=str(tmp_path/'outputs'))
    env.pop('OPENCLAW_NOTIFY_TARGET',None)
    r=subprocess.run([sys.executable,'-c',IMAGE_CHILD,str(legacy_dir/'comfy_graph.py'),scenario,*args],cwd=tmp_path,env=env,capture_output=True,text=True)
    lines=[l.split('=',1)[1] for l in r.stderr.splitlines() if l.startswith('H3_TEST_RECEIPT=')]
    assert len(lines)==1,r.stderr
    return r,json.loads(lines[0])

def image(tmp_path,name='source.png',fmt='PNG',size=(48,32)):
    p=tmp_path/name;PIL.new('RGB',size,(51,92,103)).save(p,format=fmt);return str(p)

@pytest.mark.parametrize('mode',['h3-i2v','h3-r2v'])
@pytest.mark.parametrize('dump',[False,True])
def test_help_zero_http(legacy_dir,tmp_path,mode,dump):
    r,c=invoke(legacy_dir,tmp_path,[*(['dump'] if dump else []),mode,'--help'])
    assert r.returncode==0,r.stderr
    assert c==[]

@pytest.mark.parametrize('mode',['h3-i2v','h3-r2v'])
def test_offline_full_graph_stable_managed_name(legacy_dir,tmp_path,mode):
    a=image(tmp_path); opts=['--image',a] if mode=='h3-i2v' else ['--references',a,image(tmp_path,'second.jpg','JPEG')]
    args=['dump',mode,'--prompt','Use <Picture 1>. Audio: bell.','--seed','0',*opts]
    r,c=invoke(legacy_dir,tmp_path,args);assert r.returncode==0,r.stderr
    g=json.loads(r.stdout);assert c==[];assert g['8']['inputs']['noise_seed']==0
    loads=[n['inputs']['image'] for n in g.values() if n['class_type']=='LoadImage']
    assert all(n.startswith('h3_') and str(tmp_path) not in n for n in loads)
    again,_=invoke(legacy_dir,tmp_path,args);assert r.stdout==again.stdout
    assert not (tmp_path/'outputs').exists()

@pytest.mark.parametrize('mode',['h3-i2v','h3-r2v'])
@pytest.mark.parametrize('flags', [['--strength','1'],['--audio','x'],['--last','x'],['--negative','x'],['--sampler','res_multistep'],['--steps','20'],['--fps','24'],['--width','864.0'],['--frames','125'],['--seed','true'],['--caption-template','{bad}']])
def test_bad_options_before_upload(legacy_dir,tmp_path,mode,flags):
    a=image(tmp_path);o=['--image',a] if mode=='h3-i2v' else ['--references',a]
    r,c=invoke(legacy_dir,tmp_path,[mode,'--prompt','Use <Picture 1>',*o,*flags]);assert r.returncode==2,r.stderr;assert c==[]

@pytest.mark.parametrize('bad',['missing','corrupt','gif','huge','animated','extension'])
def test_all_references_admit_atomically(legacy_dir,tmp_path,bad):
    a=image(tmp_path);p=tmp_path/'bad.png'
    if bad=='corrupt':p.write_bytes(b'not an image')
    elif bad=='gif':PIL.new('RGB',(32,32)).save(p,format='GIF')
    elif bad=='huge':PIL.new('RGB',(8193,1)).save(p)
    elif bad=='animated':PIL.new('RGB',(32,32)).save(p,save_all=True,append_images=[PIL.new('RGB',(32,32),'red')])
    elif bad=='extension':PIL.new('RGB',(32,32)).save(p,format='JPEG')
    r,c=invoke(legacy_dir,tmp_path,['h3-r2v','--prompt','Use <Picture 1> and <Picture 2>','--references',a,str(p)])
    assert r.returncode==2,r.stderr;assert c==[]

@pytest.mark.parametrize('refs',[[],['one.png']*3,'one.png',[True],[None]])
def test_json_ref_type_count_before_transport(legacy_dir,tmp_path,refs):
    p=tmp_path/'spec.json';p.write_text(json.dumps({'prompt':'Use <Picture 1>','references':refs}))
    r,c=invoke(legacy_dir,tmp_path,['h3-r2v','--input-json',str(p)]);assert r.returncode==2,r.stderr;assert c==[]

@pytest.mark.parametrize('mode',['h3-i2v','h3-r2v'])
def test_uploads_video_endpoint_then_fresh_queue_exact_one_post(legacy_dir,tmp_path,mode):
    a=image(tmp_path);o=['--image',a] if mode=='h3-i2v' else ['--references',a,image(tmp_path,'b.jpg','JPEG')]
    r,c=invoke(legacy_dir,tmp_path,[mode,'--prompt','Use <Picture 1>. Audio: bell','--seed','0',*o]);assert r.returncode==0,r.stderr
    n=1 if mode=='h3-i2v' else 2
    assert [x['method'] for x in c]==['POST']*n+['GET','POST','GET','GET']
    assert all(x['url']=='http://video.invalid:8190/upload/image' for x in c[:n])
    assert c[n]['url']=='http://video.invalid:8190/queue'
    posts=[x for x in c if x['url'].endswith('/prompt')];assert len(posts)==1
    g=posts[0]['body']['prompt'];assert g['8']['inputs']['noise_seed']==0
    assert [x['inputs']['image'] for x in g.values() if x['class_type']=='LoadImage']==[x['upload_name'] for x in c[:n]]

@pytest.mark.parametrize('scenario',['upload-error','second-upload-error','unsafe-upload'])
def test_upload_failure_no_prompt_post(legacy_dir,tmp_path,scenario):
    r,c=invoke(legacy_dir,tmp_path,['h3-r2v','--prompt','Use <Picture 1>','--references',image(tmp_path),image(tmp_path,'b.jpg','JPEG')],scenario)
    assert r.returncode!=0;assert 'NO_POST' in r.stderr;assert not any(x['url'].endswith('/prompt') for x in c)
    assert not (tmp_path/'outputs').exists()


def test_remote_unbounded_collision_suffix_rejected_no_post(legacy_dir,tmp_path):
    global IMAGE_CHILD
    original=IMAGE_CHILD
    IMAGE_CHILD=IMAGE_CHILD.replace("name = '../escape.png'", "name = name.rsplit('.',1)[0] + '_' + '9'*201 + '.png'")
    try:
        r,c=invoke(legacy_dir,tmp_path,['h3-i2v','--prompt','bell','--image',image(tmp_path)],'unsafe-upload')
        assert r.returncode!=0
        assert 'NO_POST' in r.stderr
        assert not any(x['url'].endswith('/prompt') for x in c)
    finally:IMAGE_CHILD=original

@pytest.mark.parametrize('bad',['truncated','bytes','pixels'])
def test_decoder_length_and_pixel_bounds_before_upload(legacy_dir,tmp_path,bad):
    p=Path(image(tmp_path))
    if bad=='truncated':p.write_bytes(p.read_bytes()[:45])
    elif bad=='bytes':
        with p.open('ab') as stream:stream.truncate(32*1024*1024+1)
    else:PIL.new('RGB',(4097,4097)).save(p)
    r,c=invoke(legacy_dir,tmp_path,['h3-i2v','--prompt','bell','--image',str(p)])
    assert r.returncode==2,r.stderr
    assert c==[]


def test_admitted_bytes_cannot_change_between_uploads(legacy_dir,tmp_path):
    global IMAGE_CHILD
    a=image(tmp_path);b=image(tmp_path,'second.jpg','JPEG')
    import hashlib
    expected=hashlib.sha256(Path(b).read_bytes()).hexdigest()
    original=IMAGE_CHILD
    IMAGE_CHILD=IMAGE_CHILD.replace("calls[-1]['upload_name'] = name", """calls[-1]['upload_name'] = name
        import hashlib
        payload = request.data.split(bytes([13,10,13,10]),1)[1].split(bytes([13,10]) + b'--',1)[0]
        calls[-1]['image_sha256'] = hashlib.sha256(payload).hexdigest()
        if len(calls) == 1:
            Path('second.jpg').write_bytes(b'changed after admission')""")
    try:
        r,c=invoke(legacy_dir,tmp_path,['h3-r2v','--prompt','Use <Picture 1>','--references',a,b])
        assert r.returncode==0,r.stderr
        assert c[1]['image_sha256']==expected
        assert expected in c[1]['upload_name']
        assert Path(b).read_bytes()==b'changed after admission'
    finally:IMAGE_CHILD=original


@pytest.mark.parametrize('mode', ['h3-i2v', 'h3-r2v'])
@pytest.mark.parametrize('scenario', ['upload-wrong-type', 'upload-missing-type',
                                     'upload-wrong-folder', 'upload-missing-folder'])
def test_upload_location_metadata_must_match_before_prompt(legacy_dir, tmp_path, mode, scenario):
    path = image(tmp_path)
    flags = ['--image', path] if mode == 'h3-i2v' else ['--references', path]
    result, calls = invoke(legacy_dir, tmp_path, [mode, '--prompt', 'bell', *flags], scenario)
    assert result.returncode != 0
    assert 'NO_POST' in result.stderr
    assert not any(call['url'].endswith('/prompt') for call in calls)
    assert not (tmp_path / 'outputs').exists()
