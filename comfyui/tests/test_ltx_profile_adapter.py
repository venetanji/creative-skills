import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('profile_adapter',Path(__file__).parents[1]/'scripts'/'ltx_profile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def forbidden(*args):
    raise AssertionError('unexpected upload/probe')


@pytest.mark.parametrize('opts', [dict(camera_lora='static'),dict(fast=True),dict(ic_loras='x'),dict(model_profile='bogus'),dict(image='x',prev_video='b',prev_frames=49)])
def test_reject_before_upload(opts):
    o={'model_profile':'ltx-2.5','image':'x.png'};o.update(opts)
    with pytest.raises(ValueError):
        m.prepare('i2v',o,0,'ball',upload=forbidden,probe=forbidden)


def test_dump_no_upload_exact_timeline():
    g,t=m.prepare('continuation',dict(model_profile='ltx-2.5',prev_video='A.mp4',prev_frames=49,
              frames=49,overlap_seconds=1,audio='master.wav'),4202,'ball',upload=forbidden,probe=forbidden,dump=True)
    assert t['overlap_frames']==25
    assert t['source_span']==[24,73]
    assert next(n['inputs'] for n in g.values() if n['class_type']=='TrimAudioDuration')['start_index']==1


def test_unsupported_workflow_before_upload():
    with pytest.raises(ValueError):
        m.prepare('transition',{'model_profile':'ltx-2.5'},1,'ball',upload=forbidden,probe=forbidden)


@pytest.mark.parametrize('cmd,extra', [('t2v',{'image':'x'}),('t2v',{'overlap_frames':25}),('i2v',{'image':'x','prev_video':'y','prev_frames':49})])
def test_reject_conflicting_workflow(cmd,extra):
    with pytest.raises(ValueError):
        m.prepare(cmd,dict(model_profile='ltx-2.5',**extra),1,'ball',upload=forbidden,probe=forbidden)


@pytest.mark.parametrize('key,value', [('width',64),('height',64),('frames',16385),('fps',121),('seed',-1),
    ('frames',49.9),('fps',24.9),('width',512.9),('height',320.9),('seed',1.9),
    ('fps',True),('seed',True),('prev_frames',49.9),('previous_retained_end',49.9),
    ('previous_source_start',24.9),('overlap_frames',25.9),('audio_start_frame',24.9)])
def test_strict_inputs_before_upload(key,value):
    opts=dict(model_profile='ltx-2.5',prev_video='A',prev_frames=49,audio='master.wav')
    opts[key]=value
    with pytest.raises(ValueError):
        m.prepare('continuation',opts,opts.get('seed',0),'ball',upload=forbidden,probe=forbidden)


@pytest.mark.parametrize('key', ['seed','width','height','fps','frames','prev_frames',
    'previous_retained_end','previous_source_start','overlap_frames','audio_start_frame'])
@pytest.mark.parametrize('value', [True,49.9])
def test_actual_input_json_rejects_integer_coercion(tmp_path,key,value):
    import json, os, subprocess, sys
    opts=dict(model_profile='ltx-2.5',prev_video='A',prev_frames=49,audio='master.wav')
    opts[key]=value
    path=tmp_path/'input.json';path.write_text(json.dumps(opts))
    env=dict(os.environ,COMFY_URL='http://127.0.0.1:1',COMFY_URL_VIDEO='http://127.0.0.1:1')
    env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/comfy_graph.py'),
        'dump','continuation','--input-json',str(path)],capture_output=True,text=True,env=env)
    assert result.returncode != 0
    assert 'integer' in result.stderr


def test_actual_json_seed_zero_and_cli_integer_strings(tmp_path):
    import json, os, subprocess, sys
    path=tmp_path/'input.json';path.write_text(json.dumps(dict(model_profile='ltx-2.5',seed=0)))
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/comfy_graph.py'),
        'dump','t2v','--input-json',str(path),'--frames','49','--fps','24'],capture_output=True,text=True,check=True,env=env)
    graph=json.loads(result.stdout)
    assert [n['inputs']['noise_seed'] for n in graph.values() if n['class_type']=='RandomNoise']==[0,1]


@pytest.mark.parametrize('tail', [['--input-json'],['--frames'],['stray'],['--prompt']])
def test_selected_cli_malformed_input_errors(tail):
    import subprocess, sys
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/comfy_graph.py'),
        'dump','t2v','--model-profile','ltx-2.5',*tail],capture_output=True,text=True)
    assert result.returncode != 0


@pytest.mark.parametrize('cmd,extra', [('t2v',{'base_guide_strength':.7}),
    ('continuation',{'prev_video':'A','prev_frames':49,'refine_guide_strength':1}),
    ('i2v',{'image':'x','overlap_strength':1}),('t2v',{'previous_source_start':0}),
    ('t2v',{'overlap_seconds':1}),('t2v',{'prev_frames':49})])
def test_shared_orphan_and_strength_admission(cmd,extra):
    with pytest.raises(ValueError):
        m.prepare(cmd,dict(model_profile='ltx-2.5',**extra),0,'ball',upload=forbidden,probe=forbidden)


def test_flf_dump_strengths_seed_and_placeholders():
    g,t=m.prepare('flf2v',dict(model_profile='ltx-2.5',first='first.png',last='last.png',
          frames=49,first_guide_strength=.2,last_guide_strength=.9,audio='master.wav'),0,
          'ball',upload=forbidden,probe=forbidden,dump=True)
    assert t is None
    assert [n['inputs']['strength'] for n in g.values() if n['class_type']=='LTXVAddGuide']==[.2,.9,.2,.9]
    assert [n['inputs']['image'] for n in g.values() if n['class_type']=='LoadImage']==['first.png','last.png']


@pytest.mark.parametrize('extra',[dict(first=None),dict(last=''),dict(first=[]),dict(last=True),
    dict(image='x'),dict(prev_video='x',prev_frames=49),dict(audio_start_frame=0),
    dict(overlap_frames=25),dict(base_guide_strength=.7),dict(refine_guide_strength=1),
    dict(first_guide_strength=True),dict(last_guide_strength=2),dict(fast=True),dict(ckpt='x')])
def test_flf_adapter_preflight_all_before_upload(extra):
    opts=dict(model_profile='ltx-2.5',first='f.png',last='l.png');opts.update(extra)
    with pytest.raises(ValueError):m.prepare('flf2v',opts,0,'ball',upload=forbidden,probe=forbidden)


@pytest.mark.parametrize('cmd',['t2v','i2v','continuation'])
def test_orphan_endpoint_fields_reject(cmd):
    opts=dict(model_profile='ltx-2.5',first='f',last='l')
    if cmd=='i2v':opts['image']='i'
    if cmd=='continuation':opts.update(prev_video='v',prev_frames=49)
    with pytest.raises(ValueError):m.prepare(cmd,opts,0,'ball',upload=forbidden,probe=forbidden)


def test_actual_flf_input_json_dump_without_network(tmp_path):
    import json,os,subprocess,sys
    path=tmp_path/'input.json';path.write_text(json.dumps(dict(model_profile='ltx-2.5',
        first='f.png',last='l.png',seed=0,frames=97,width=768,height=512,first_guide_strength=.3,last_guide_strength=.8)))
    env=dict(os.environ,COMFY_URL='http://127.0.0.1:1',COMFY_URL_VIDEO='http://127.0.0.1:1');env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'scripts/comfy_graph.py'),
        'dump','flf2v','--input-json',str(path)],capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stderr
    g=json.loads(result.stdout)
    assert [n['inputs']['noise_seed'] for n in g.values() if n['class_type']=='RandomNoise']==[0,1]
    assert [n['inputs']['frame_idx'] for n in g.values() if n['class_type']=='LTXVAddGuide']==[0,96,0,96]
    assert [n['inputs']['strength'] for n in g.values() if n['class_type']=='LTXVAddGuide']==[.3,.8,.3,.8]


def test_known_missing_local_before_any_upload(tmp_path):
    first=tmp_path/'f.png';first.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='local'):
        m.prepare('flf2v',dict(model_profile='ltx-2.5',first=str(first),last=str(tmp_path/'missing.png')),
                  0,'ball',upload=forbidden,probe=forbidden)


def test_short_local_master_before_upload(tmp_path):
    import wave
    audio=tmp_path/'short.wav'
    with wave.open(str(audio),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000);f.writeframes(b'\0\0'*4800)
    with pytest.raises(ValueError,match='short|cover'):
        m.prepare('flf2v',dict(model_profile='ltx-2.5',first='f.png',last='l.png',audio=str(audio)),
                  0,'ball',upload=forbidden,probe=forbidden)
