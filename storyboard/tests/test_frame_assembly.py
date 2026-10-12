import importlib.util
from pathlib import Path
import subprocess
import wave
import pytest

spec=importlib.util.spec_from_file_location('assemble',Path(__file__).parents[1]/'lib'/'assemble.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_exact_frames_and_master_audio(tmp_path):
    src=tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=24',
                    '-frames:v','49','-c:v','libx264',str(src)],check=True)
    audio=tmp_path/'master.wav'
    with wave.open(str(audio),'wb') as w:
        w.setparams((1,2,48000,146000,'NONE','not compressed'));w.writeframes(b'\0'*292000)
    out=tmp_path/'out.mp4'
    result=module.assemble_frame_spans([(src,0,49),(src,25,49)],audio,out,fps=24,width=64,height=64)
    assert result['frames']==73
    assert result['audio_samples']==146000
    with pytest.raises(ValueError,match='span'):
        module.assemble_frame_spans([(src,25,50)],None,out,fps=24,width=64,height=64)


def test_reject_short_audio(tmp_path):
    src=tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=64x64:rate=24',
                    '-frames:v','9','-c:v','libx264',str(src)],check=True)
    audio=tmp_path/'short.wav'
    with wave.open(str(audio),'wb') as w:
        w.setparams((1,2,48000,2,'NONE','not compressed'));w.writeframes(b'\0'*4)
    with pytest.raises(ValueError,match='audio'):
        module.assemble_frame_spans([(src,0,9)],audio,tmp_path/'out.mp4',fps=24,width=64,height=64)


@pytest.mark.parametrize('default_second',[False,True])
def test_multitrack_coverage_selects_muxed_first_stream(tmp_path,default_second):
    src=tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=64x64:rate=24',
                    '-frames:v','49','-c:v','libx264',str(src)],check=True)
    audio=tmp_path/'master.mka'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','anullsrc=r=48000:cl=mono:d=0.1',
        '-f','lavfi','-i','anullsrc=r=48000:cl=stereo:d=3','-map','0:a:0','-map','1:a:0',
        '-c:a','pcm_s16le','-disposition:a:0','0','-disposition:a:1','default' if default_second else '0',str(audio)],check=True)
    out=tmp_path/'out.mp4';out.write_bytes(b'previous accepted output')
    with pytest.raises(ValueError,match='master audio too short'):
        module.assemble_frame_spans([(src,0,49)],audio,out,fps=24,width=64,height=64)
    assert out.read_bytes()==b'previous accepted output'


def test_candidate_32_sample_discrepancy_preserves_destination(tmp_path,monkeypatch):
    src=tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=64x64:rate=24',
                    '-frames:v','49','-c:v','libx264',str(src)],check=True)
    audio=tmp_path/'master.wav'
    with wave.open(str(audio),'wb') as w:
        w.setparams((1,2,48000,98000,'NONE','not compressed'));w.writeframes(b'\0'*196000)
    out=tmp_path/'out.mp4';out.write_bytes(b'previous accepted output')
    real_run=subprocess.run
    def run(args,**kw):
        result=real_run(args,**kw)
        if 's16le' in args and 'result.mp4' in str(args):
            result.stdout=result.stdout[:-64]
        return result
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(ValueError,match='decoded.*audio.*97968.*98000'):
        module.assemble_frame_spans([(src,0,49)],audio,out,fps=24,width=64,height=64)
    assert out.read_bytes()==b'previous accepted output'
