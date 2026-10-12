"""Native image/ref recipes and import-free deployed V3 API expansion."""
import importlib.util
import json
from pathlib import Path
import pytest
import creative_comfy_graph.minimax_h3 as h3
F = Path(__file__).parent / 'fixtures'

def test_i2v_exact_fixture():
    p=json.loads((F/'h3_i2v_prototype.json').read_text())
    assert h3.build_i2v(p['6']['inputs']['prompt'], image_filename='h3_fixture_first.png', seed=20261002, filename_prefix='video/minimax_h3_bell_20261002') == p

def test_r2v_exact_fixture():
    p=json.loads((F/'h3_r2v_prototype.json').read_text())
    assert h3.build_r2v(p['6']['inputs']['prompt'], reference_filenames=['h3_fixture_1.png','h3_fixture_2.jpg'], seed=20261002, filename_prefix='video/minimax_h3_bell_20261002') == p

@pytest.mark.parametrize('refs',[[],['a.png']*3,'a.png',[True],[None],['../a.png'],['/a.png'],['a\\b.png'],['a.png [output]'],['a.png',3]])
def test_ref_names_and_count_reject(refs):
    with pytest.raises(ValueError):h3.build_r2v('Use <Picture 1>', reference_filenames=refs)

@pytest.mark.parametrize('image',['',None,True,'../a.png','a.png [input]','/a.png','a.gif','x'*201+'.png'])
def test_image_name_reject(image):
    with pytest.raises(ValueError):h3.build_i2v('bell',image_filename=image)

@pytest.mark.parametrize('mode',['i2v','r2v'])
@pytest.mark.parametrize('kwargs',[{'strength':1},{'last_frame':'x'},{'audio_filename':'x'},{'steps':20},{'sampler':'res_multistep'},{'negative':'x'},{'fps':24},{'seed':True},{'seed':'0'},{'frames':125},{'width':865}])
def test_strict_image_admission(mode,kwargs):
    fn=getattr(h3,'build_'+mode); extra={'image_filename':'a.png'} if mode=='i2v' else {'reference_filenames':['a.png']}
    with pytest.raises((ValueError,TypeError)):fn('Use <Picture 1>',**extra,**kwargs)

@pytest.mark.parametrize('value',[True,1,'other',None])
def test_ref_size_reject(value):
    with pytest.raises(ValueError):h3.build_r2v('Use <Picture 1>',reference_filenames=['a.png'],ref_image_size=value)

@pytest.mark.parametrize('refs',[['a.png'],['a.png','b.jpg']])
@pytest.mark.parametrize('size',['match','max'])
def test_schema_flat_links_and_v3_execution_dictionary(refs,size):
    graph=h3.build_r2v('Use <Picture 1>',reference_filenames=refs,ref_image_size=size,seed=0)
    schema=json.loads((F/'h3_image_native_schema.json').read_text())
    spec=importlib.util.spec_from_file_location('native_dynamic_contract',F/'h3_v3_dynamic_contract.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cond=graph['6']['inputs']; fields,hidden,data=m.get_finalized_class_inputs(schema['MiniMaxH3ReferenceToVideo']['input'],cond)
    assert set(fields['required']) <= cond.keys()
    assert cond.keys() <= (fields['required'].keys() | fields['optional'].keys())
    # Backend receives a ref_images dict after resolving each direct IMAGE link.
    values={k:('resolved-'+k if k.startswith('ref_images.') else v) for k,v in cond.items()}
    nested=m.build_nested_inputs(values,data)
    assert list(nested['ref_images']) == ['ref_image_'+str(i) for i in range(len(refs))]
    assert nested['ref_images']['ref_image_0']=='resolved-ref_images.ref_image_0'
    assert nested['ref_videos']==nested['ref_audios']==nested['ref_video_audios']=={}
    assert graph['8']['inputs']['noise_seed']==0
    assert graph['10']['inputs']==dict(model=['0',0],scheduler='simple',steps=20,denoise=1.0)
    assert not any(n['class_type'] in {'MiniMaxH3SigmaShift','LoraLoaderModelOnly'} for n in graph.values())
    for nid,node in graph.items():
        for name,value in node['inputs'].items():
            if isinstance(value,list):
                kind=(fields['required']|fields['optional'])[name][0] if nid=='6' else (schema[node['class_type']]['input'].get('required',{})|schema[node['class_type']]['input'].get('optional',{}))[name][0]
                assert schema[graph[value[0]]['class_type']]['output'][value[1]]==kind


def test_native_ref_active_switches_not_turbo_branch():
    recipe=json.loads((F/'h3_upstream_ref_recipe.json').read_text())
    nodes={n['id']:n for n in recipe['nodes']};links={l[0]:l for l in recipe['links']}
    assert nodes[146]['widgets_values']==[False]
    for nid in (141,142):
        ins={i['name']:i['link'] for i in nodes[nid]['inputs']}
        assert links[ins['switch']][1]==146
        active=links[ins['on_false']][1]
        assert active==(127 if nid==141 else 143)
    assert nodes[143]['widgets_values'][0]==20
    assert nodes[127]['widgets_values'][0]==h3.REF_UNET
    assert nodes[123]['widgets_values']==['res_multistep']
    assert nodes[124]['widgets_values']==['simple',20,1]
    assert links[{i['name']:i['link'] for i in nodes[124]['inputs']}['model']][1]==127
    assert links[{i['name']:i['link'] for i in nodes[126]['inputs']}['model']][1]==141


def test_native_first_frame_has_no_strength_or_ltx_snap():
    import ast
    a=ast.parse((F/'h3_native_i2v_contract.py').read_text())
    cls=next(n for n in a.body if isinstance(n,ast.ClassDef))
    execute=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    assert [x.arg for x in execute.args.args][-2:]==['first_frame','last_frame']
    assert 'strength' not in ast.unparse(cls)
    resizes=[n for n in ast.walk(execute) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_resize']
    assert [n.args[-1].value for n in resizes]==['disabled','center']
    assert 'resolved_frame_index' in ast.unparse(execute)
    assert 'frame_count - 1' in ast.unparse(execute)
