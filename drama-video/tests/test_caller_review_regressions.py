"""Caller review regressions; no GPU or paid audio service calls."""
import json
from pathlib import Path
import pytest
from test_profile_callers import load, fixture_spec

@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('invalid', [{'mode':'i2v'}, {'mode':'ia2v'}, {'seed':True}, {'seed':1.5}])
def test_invalid_later_request_blocks_paid_audio(kind, invalid, tmp_path, monkeypatch):
    m = load(kind); spec = fixture_spec(kind); key = 'shots' if kind == 'drama' else 'scenes'
    spec[key][1].pop('continue_from_prev'); spec[key][1].update(invalid)
    if 'mode' in invalid: spec[key][1]['image'] = '@none'
    monkeypatch.setattr(m, 'cmd_audio' if kind == 'drama' else 'cmd_song', lambda *a: pytest.fail('paid audio before validation'))
    with pytest.raises(ValueError, match='image|seed'): m.cmd_all(spec, tmp_path, no_gate=True)

@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_true_builder_contract(kind, monkeypatch):
    import creative_comfy_graph.ltx_profiles as p
    original = p.build_video; calls=[]
    def build(prompt, **kwargs):
        calls.append((prompt, kwargs)); return original(prompt, **kwargs)
    monkeypatch.setattr(p, 'build_video', build)
    m=load(kind); spec=fixture_spec(kind); key='shots' if kind=='drama' else 'scenes'
    spec[key][0].update(image='anchor.png', seed=0, mode='ia2v', base_guide_strength=.4)
    m._profile_plan(spec)
    first=calls[0][1]; second=calls[1][1]
    assert first['workflow']=='ia2v' and first['image_filename'] and first['audio_filename']
    assert first['seed']==0 and first['negative']=='cuts'
    assert first['base_guide_strength']==.4 and 'overlap_strength' not in first
    assert second['workflow']=='continuation' and second['prev_video_filename']
    assert second['previous_retained_end']==49 and second['audio_start_frame']==24
    assert 'base_guide_strength' not in second

@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('explicit', [False, True])
def test_direct_last_requires_previous_metadata(kind, explicit, tmp_path, monkeypatch):
    m=load(kind); spec=fixture_spec(kind); key='shots' if kind=='drama' else 'scenes'
    spec[key][1].pop('continue_from_prev'); spec[key]=spec[key][:2]
    if explicit: spec[key][1]['image']='@last'
    h=m._profile_helper(); monkeypatch.setattr(m,'_profile_helper',lambda:h)
    monkeypatch.setattr(h,'audio_samples',lambda *a:300000)
    (tmp_path/key).mkdir(); (tmp_path/key/'001-A-last.png').write_bytes(b'stale')
    monkeypatch.setattr(m,'_run',lambda *a,**k:pytest.fail('stale endpoint submitted'))
    with pytest.raises(ValueError,match='missing/stale timeline'): m._profile_render(spec,tmp_path,2)

@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_anchor_is_used_by_actual_render(kind,tmp_path,monkeypatch):
    m=load(kind); spec=fixture_spec(kind); key='shots' if kind=='drama' else 'scenes'; spec[key]=spec[key][:1]
    spec['anchor_image']='anchor.png'; (tmp_path/'anchor.png').write_bytes(b'image')
    h=m._profile_helper(); monkeypatch.setattr(m,'_profile_helper',lambda:h)
    monkeypatch.setattr(h,'audio_samples',lambda *a:300000)
    class Submitted(Exception): pass
    def run(cmd,**k):
        assert cmd[2]=='ia2v' and cmd[cmd.index('--image')+1]==str(tmp_path/'anchor.png'); raise Submitted
    monkeypatch.setattr(m,'_run',run)
    with pytest.raises(Submitted):m._profile_render(spec,tmp_path,1)

@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('edit', ['prompt','seed','negative','mode','master','upstream','image'])
def test_fingerprint_rejects_request_edits(kind,edit,tmp_path):
    m=load(kind); h=m._profile_helper(); spec=fixture_spec(kind); key='shots' if kind=='drama' else 'scenes'
    (tmp_path/key).mkdir(); master=tmp_path/('master.wav' if kind=='drama' else 'song.mp3');master.write_bytes(b'master')
    (tmp_path/key/'001-A.mp4').write_bytes(b'A');(tmp_path/key/'002-B.mp4').write_bytes(b'B')
    (tmp_path/'anchor.png').write_bytes(b'image')
    spec[key][0]['image']='anchor.png'
    rows=m._profile_plan(spec)
    for idx in (1,2):
        h.metadata_path(tmp_path,key,spec[key][idx-1],idx).write_text(json.dumps({'timeline':rows[idx-1],
            'generation_fingerprint':h.generation_fingerprint(spec,tmp_path,key,idx,rows)}))
    h.check_metadata(spec,tmp_path,key,2,rows)
    if edit=='upstream':(tmp_path/key/'001-A.mp4').write_bytes(b'rerendered A')
    elif edit=='master':master.write_bytes(b'edited master')
    elif edit=='image':(tmp_path/'anchor.png').write_bytes(b'edited image')
    elif edit=='mode':spec[key][0]['mode']='i2v'
    else:spec[key][0][edit]={'prompt':'new prompt','seed':0,'negative':'new negative'}[edit]
    rows=m._profile_plan(spec)
    with pytest.raises(ValueError,match='stale.*rerender'):h.check_metadata(spec,tmp_path,key,2,rows)


def test_music_all_long_master_and_unused_variant_are_not_freeze_tail(tmp_path,monkeypatch):
    m=load('music'); spec=fixture_spec('music'); spec['scenes']=spec['scenes'][:1];spec['scenes'][0]['duration_sec']='2.0416666666666665'
    (tmp_path/'song.mp3').touch();(tmp_path/'song_v1.mp3').touch()
    h=m._profile_helper();monkeypatch.setattr(m,'_profile_helper',lambda:h)
    monkeypatch.setattr(h,'audio_samples',lambda *a:480000)
    calls=[]
    for name in ('cmd_song','cmd_anchors','cmd_scene','cmd_assemble'):monkeypatch.setattr(m,name,lambda *a,_name=name,**k:calls.append(_name))
    monkeypatch.setattr(m,'_probe_duration',lambda *a:pytest.fail('legacy variant/freeze check'))
    m.cmd_all(spec,tmp_path,no_gate=True)
    assert calls==['cmd_song','cmd_anchors','cmd_scene','cmd_assemble']

@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('mode', ['i2v','ia2v'])
def test_first_explicit_image_mode_needs_image(kind,mode):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes'
    spec[key][0]['mode']=mode
    with pytest.raises(ValueError,match='requires an image'):m._profile_plan(spec)

@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_future_last_plan_does_not_require_unrendered_endpoint(kind,tmp_path):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes'
    spec[key]=spec[key][:2]
    spec[key][1].pop('continue_from_prev');spec[key][1]['image']='@last'
    rows=m._profile_plan(spec,tmp_path)
    assert rows[1]['generation']['mode']=='ia2v' and rows[1]['generation']['image']=='@last'

@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_missing_later_literal_image_blocks_audio_and_batch_gpu(kind,tmp_path,monkeypatch):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes'
    spec[key][1].pop('continue_from_prev');spec[key][1]['image']='not-present.png'
    monkeypatch.setattr(m,'cmd_audio' if kind=='drama' else 'cmd_song',lambda *a:pytest.fail('paid audio'))
    monkeypatch.setattr(m,'_run',lambda *a,**k:pytest.fail('GPU'))
    with pytest.raises(ValueError,match='image missing'):m.cmd_all(spec,tmp_path,no_gate=True)
    if kind=='music':(tmp_path/'song.mp3').touch()
    with pytest.raises(ValueError,match='image missing'):
        (m.cmd_shots if kind=='drama' else m.cmd_scenes)(spec,tmp_path)

@pytest.mark.parametrize('kind', ['drama', 'music'])
@pytest.mark.parametrize('independent',[False,True])
def test_last_reextracts_retained_endpoint_but_independent_image_is_exempt(kind,independent,tmp_path,monkeypatch):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes';spec[key]=spec[key][:2]
    spec[key][1].pop('continue_from_prev');spec['video']['tail_buffer_sec']=1
    directory=tmp_path/key;directory.mkdir();previous=directory/'001-A.mp4';previous.write_bytes(b'previous')
    endpoint=directory/'001-A-last.png';endpoint.write_bytes(b'stale full tail')
    if independent:
        spec[key][1]['image']='independent.png';(tmp_path/'independent.png').write_bytes(b'independent')
    h=m._profile_helper();monkeypatch.setattr(m,'_profile_helper',lambda:h);monkeypatch.setattr(h,'audio_samples',lambda *a:500000)
    rows=m._profile_plan(spec)
    h.metadata_path(tmp_path,key,spec[key][0],1).write_text(json.dumps({'timeline':rows[0],
        'generation_fingerprint':h.generation_fingerprint(spec,tmp_path,key,1,rows)}))
    checked=[]
    def check(path,row):checked.append(path)
    def extract(path,row,out,ffmpeg):
        assert path==previous and row['retain_span'][1]==49 and row['render_frames']==73
        out.write_bytes(b'fresh retained endpoint')
    monkeypatch.setattr(h,'check_clip',check);monkeypatch.setattr(h,'extract_endpoint',extract)
    if independent:monkeypatch.setattr(h,'check_metadata',lambda *a:pytest.fail('independent image has no upstream dependency'))
    class Submitted(Exception):pass
    def run(cmd,**k):
        image=Path(cmd[cmd.index('--image')+1]);assert image.read_bytes()==(b'independent' if independent else b'fresh retained endpoint');raise Submitted
    monkeypatch.setattr(m,'_run',run)
    with pytest.raises(Submitted):m._profile_render(spec,tmp_path,2)
    assert checked==([] if independent else [previous])

@pytest.mark.parametrize('kind', ['drama', 'music'])
def test_failed_assembly_invalidates_old_success_metadata(kind,tmp_path,monkeypatch):
    m=load(kind);h=m._profile_helper();spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes'
    final=tmp_path/'final.mp4';final.write_bytes(b'previous valid output')
    cert=tmp_path/'final.timeline.json';cert.write_text('old success')
    def reject(*a):raise ValueError('stale clip')
    monkeypatch.setattr(h,'check_metadata',reject)
    with pytest.raises(ValueError,match='stale clip'):h.assemble(m.__file__,spec,tmp_path,key,m._profile_plan(spec),'ffmpeg')
    assert not cert.exists() and final.read_bytes()==b'previous valid output'


def test_master_preflight_uses_first_audio_stream(tmp_path):
    import subprocess
    m=load('drama');h=m._profile_helper();spec=fixture_spec();spec['shots']=spec['shots'][:1]
    audio=tmp_path/'multitrack.mkv'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=sample_rate=48000:duration=1',
        '-f','lavfi','-i','sine=sample_rate=48000:duration=4','-map','0:a','-map','1:a','-c:a','pcm_s16le',
        '-disposition:a:0','0','-disposition:a:1','default',str(audio)],check=True)
    spec['audio']['file']=str(audio)
    with pytest.raises(ValueError,match='got 48000'):h.check_audio(spec,tmp_path,'shots',m._profile_plan(spec),'ffmpeg')

@pytest.mark.parametrize('kind', ['drama','music'])
def test_command_uses_same_expanded_prompt_as_preflight(kind,tmp_path):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes'
    token='character' if kind=='drama' else 'hero';spec[key][0]['prompt']='{'+token+'} rolls'
    if kind=='drama':spec['character']={'description':'red ball'}
    else:spec['subjects']={'hero':'red ball'}
    rows=m._profile_plan(spec)
    cmd=m._profile_command(spec,tmp_path,1,rows,None)
    assert cmd[cmd.index('--prompt')+1]==rows[0]['generation']['prompt']=='red ball rolls'

@pytest.mark.parametrize('kind', ['drama','music'])
def test_invalid_assembly_plan_removes_previous_certificate(kind,tmp_path):
    m=load(kind);spec=fixture_spec(kind);spec['video']['seed']=True
    cert=tmp_path/'final.timeline.json';cert.write_text('previous success')
    with pytest.raises(ValueError,match='seed'):m.cmd_assemble(spec,tmp_path)
    assert not cert.exists()


def test_missing_drama_master_contract_rejects_before_paid_audio(tmp_path,monkeypatch):
    m=load('drama');spec=fixture_spec();spec['audio']={'spec':'paid-dialogue.yaml'}
    monkeypatch.setattr(m,'cmd_audio',lambda *a:pytest.fail('missing destination accepted'))
    with pytest.raises(ValueError,match='audio.file'):m.cmd_all(spec,tmp_path,no_gate=True)

@pytest.mark.parametrize('kind',['drama','music'])
def test_known_short_master_blocks_paid_audio_stage(kind,tmp_path,monkeypatch):
    m=load(kind);spec=fixture_spec(kind);master=tmp_path/('master.wav' if kind=='drama' else 'song.mp3');master.touch()
    h=m._profile_helper();monkeypatch.setattr(m,'_profile_helper',lambda:h);monkeypatch.setattr(h,'audio_samples',lambda *a:1)
    monkeypatch.setattr(m,'cmd_audio' if kind=='drama' else 'cmd_song',lambda *a:pytest.fail('known invalid master before paid audio'))
    with pytest.raises(ValueError,match='insufficient master audio'):m.cmd_all(spec,tmp_path,no_gate=True)

@pytest.mark.parametrize('kind',['drama','music'])
def test_null_strength_is_default_not_invalid_cli_string(kind,tmp_path):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes'
    spec[key][1]['overlap_strength']=None
    rows=m._profile_plan(spec);cmd=m._profile_command(spec,tmp_path,2,rows,None)
    assert cmd[cmd.index('--overlap-strength')+1]=='1.0'
    assert rows[1]['generation']['overlap_strength']==1.0

@pytest.mark.parametrize('kind',['drama','music'])
def test_equivalent_strength_values_have_same_normalized_fingerprint(kind,tmp_path):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes';h=m._profile_helper()
    spec[key][1]['overlap_strength']=1
    a=h.generation_fingerprint(spec,tmp_path,key,2,m._profile_plan(spec))
    spec[key][1]['overlap_strength']=1.0
    b=h.generation_fingerprint(spec,tmp_path,key,2,m._profile_plan(spec))
    assert a==b

@pytest.mark.parametrize('kind',['drama','music'])
def test_reused_last_clip_still_requires_previous_metadata(kind,tmp_path,monkeypatch):
    m=load(kind);spec=fixture_spec(kind);key='shots' if kind=='drama' else 'scenes';spec[key]=spec[key][:2]
    spec[key][1].pop('continue_from_prev');directory=tmp_path/key;directory.mkdir()
    for name in ('001-A.mp4','002-B.mp4'): (directory/name).write_bytes(b'clip')
    h=m._profile_helper();monkeypatch.setattr(m,'_profile_helper',lambda:h)
    monkeypatch.setattr(h,'audio_samples',lambda *a:300000);monkeypatch.setattr(h,'check_clip',lambda *a:None)
    monkeypatch.setattr(h,'extract_endpoint',lambda *a:None)
    rows=m._profile_plan(spec)
    h.metadata_path(tmp_path,key,spec[key][1],2).write_text(json.dumps({'timeline':rows[1],
        'generation_fingerprint':h.generation_fingerprint(spec,tmp_path,key,2,rows)}))
    with pytest.raises(ValueError,match='001-A.timeline.json'):m._profile_render(spec,tmp_path,2)
