"""Copied legacy CLI fixtures: deliberate process launches, guarded child I/O."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_characterization import capture_module

MESSAGE = 'creative-comfy-graph is required; install it into the Python environment running this script.'


def clean_environment(python):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.pop('VIRTUAL_ENV', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PATH=str(Path(python).parent) + os.pathsep + env['PATH'])
    return env


def cli_child(python, legacy_dir, cwd, mode):
    # Read only the supplied CLI/test sources before installing the same strict
    # boundary used by characterization; no package source path is inserted.
    code = r'''
import importlib.util, json, sys
from pathlib import Path
legacy, tool_path, mode = sys.argv[1:]
spec = importlib.util.spec_from_file_location('capture', tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
path = Path(legacy) / 'comfy_graph.py'
source = path.read_text()
with tool.boundaries(legacy) as attempts, tool.legacy_modules(legacy) as modules:
    namespace = {'__name__': 'copied_cli', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)
    if mode == 'exports':
        import ast
        tree = ast.parse(source)
        exports = {alias.asname or alias.name: node.module for node in tree.body
                   if isinstance(node, ast.ImportFrom) and node.module in modules
                   for alias in node.names}
        for name, module in exports.items():
            assert namespace[name] is getattr(modules[module], name), name
        for name in ('core', 'flux2', 'tts', 'post', 'ltx2'):
            assert namespace[name] is modules[name], name
        print(json.dumps({'exports': sorted(exports), 'aliases': sorted(modules)}))
    else:
        args = [] if mode == 'usage' else ['dump', 't2i', '--prompt', 'fixture', '--seed', '42']
        sys.argv = [str(path), *args]
        namespace['main']()
    assert not attempts
'''
    tool = Path(__file__).parent / 'tools' / 'capture_baseline.py'
    return subprocess.run([str(python), '-c', code, str(legacy_dir), str(tool), mode],
                          cwd=cwd, env=clean_environment(python), text=True, capture_output=True)



def entrypoint_child(python, legacy_dir, cwd, mode):
    code = r'''
import importlib.util, runpy, sys
from pathlib import Path
legacy, tool_path, mode = sys.argv[1:]
legacy = Path(legacy).resolve()
spec = importlib.util.spec_from_file_location('capture', tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
assert not {'core', 'flux2', 'ltx2', 'tts', 'post'} & sys.modules.keys()
# Match python <copied-script>: only this explicitly supplied sibling directory
# replaces the -c cwd entry. Package resolution still uses the installed venv.
sys.path[0] = str(legacy)
path = legacy / 'comfy_graph.py'
args = [] if mode == 'usage' else ['dump', 't2i', '--prompt', 'fixture', '--seed', '42']
sys.argv = [str(path), *args]
with tool.boundaries(legacy, script_entrypoint=True) as attempts:
    try:
        runpy.run_path(str(path), run_name='__main__')
    finally:
        assert not attempts, attempts
'''
    tool = Path(__file__).parent / 'tools' / 'capture_baseline.py'
    return subprocess.run([str(python), '-c', code, str(legacy_dir), str(tool), mode],
                          cwd=cwd, env=clean_environment(python), text=True, capture_output=True)


@pytest.mark.parametrize('mode', ['usage', 'dump'])
def test_runtime_actual_entrypoint(legacy_dir, tmp_path, mode):
    result = entrypoint_child(sys.executable, legacy_dir, tmp_path, mode)
    assert result.returncode == (1 if mode == 'usage' else 0), result.stderr
    assert result.stderr == ''
    if mode == 'usage':
        assert 'Usage:' in result.stdout
    else:
        tool = capture_module()
        with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
            expected = modules['flux2'].flux2_text_to_image('fixture', seed=42)
        assert json.loads(result.stdout) == expected

def test_runtime_cli_no_arguments(legacy_dir, tmp_path):
    result = cli_child(sys.executable, legacy_dir, tmp_path, 'usage')
    assert result.returncode == 1, result.stderr
    assert 'Usage:' in result.stdout
    assert result.stderr == ''


def test_runtime_cli_guarded_dump(legacy_dir, tmp_path):
    result = cli_child(sys.executable, legacy_dir, tmp_path, 'dump')
    assert result.returncode == 0, result.stderr
    tool = capture_module()
    with tool.boundaries(legacy_dir), tool.legacy_modules(legacy_dir) as modules:
        expected = modules['flux2'].flux2_text_to_image('fixture', seed=42)
    assert json.loads(result.stdout) == expected
    assert result.stderr == ''


def test_runtime_cli_exports(legacy_dir, tmp_path):
    result = cli_child(sys.executable, legacy_dir, tmp_path, 'exports')
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)['exports']) == {
        '_submit_and_wait', 'upload_if_local', 'flux2_text_to_image',
        'flux2_single_image_edit', 'flux2_double_image_edit',
        'flux2_double_image_edit_multiprompt', 'flux2_multiple_angles',
        'flux2_multi_reference_edit', 'flux2_multi_reference_edit_multiprompt',
        'ltx2_text_to_video', 'ltx2_image_to_video', 'ltx2_image_audio_to_video',
        'ltx2_ingredients_to_video', 'ltx2_first_last_frame_to_video',
        'ltx2_multi_guide_to_video', 'ltx2_continuation_to_video',
        'extract_last_frame', 'qwen_tts', 'extract_stems', 'transcribe', 'concat_videos',
    }


def test_plain_python_child_same_environment(tmp_path):
    probe = "import json,sys,os,creative_comfy_graph as p; print(json.dumps({'executable':os.path.realpath(sys.executable),'prefix':sys.prefix,'package':p.__file__}))"
    parent = '''import subprocess, sys
probe = sys.argv[1]
exec(probe)
result = subprocess.run(['python3', '-c', probe], check=True, text=True, capture_output=True)
print(result.stdout, end='')
'''
    result = subprocess.run(['uv', 'run', '--no-project', '--python', sys.executable,
                             'python', '-c', parent, probe], cwd=tmp_path,
                            env=clean_environment(sys.executable), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    parent_info, child_info = map(json.loads, result.stdout.splitlines())
    assert parent_info == child_info
    assert Path(parent_info['prefix']) == Path(sys.prefix)
    print(json.dumps(parent_info, sort_keys=True))


DIRECT_MODULES = ('core', 'flux2', 'tts', 'post', 'ltx2')
IMPORT_FAILURES = ('missing', 'nested', 'transitive', 'export', 'runtime')


def assert_direct_import_failure(python, legacy_dir, cwd, module, failure):
    # Each child starts without preloaded core/families. Synthetic failures model
    # a broken installed package; 'missing' uses real empty-venv resolution.
    code = r'''
import builtins, importlib, importlib.util, sys
from pathlib import Path
legacy, tool_path, module, failure = sys.argv[1:]
legacy = Path(legacy).resolve()
spec = importlib.util.spec_from_file_location('capture', tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
assert not {'core', 'flux2', 'tts', 'post', 'ltx2'} & sys.modules.keys()
sys.path[0] = str(legacy)
errors = {
    'nested': ModuleNotFoundError('broken package submodule', name='creative_comfy_graph.graph'),
    'transitive': ModuleNotFoundError('broken transitive dependency', name='nested_dependency'),
    'export': ImportError('broken export'),
    'runtime': RuntimeError('broken module'),
}
expected = errors.get(failure)
original = builtins.__import__
def broken(name, *args, **kwargs):
    if name == 'creative_comfy_graph' or name.startswith('creative_comfy_graph.'):
        raise expected
    return original(name, *args, **kwargs)
with tool.boundaries(legacy, script_entrypoint=True) as attempts:
    if expected is not None:
        builtins.__import__ = broken
    try:
        importlib.import_module(module)
    except Exception as error:
        if expected is None:
            assert type(error) is ImportError, repr(error)
            assert str(error) == 'creative-comfy-graph is required; install it into the Python environment running this script.'
            assert isinstance(error.__cause__, ModuleNotFoundError)
            assert error.__cause__.name == 'creative_comfy_graph'
        else:
            assert error is expected, repr(error)
    else:
        raise AssertionError('direct import unexpectedly succeeded')
    finally:
        builtins.__import__ = original
        assert not attempts, attempts
print('direct import failure passed:', module, failure)
'''
    tool = Path(__file__).parent / 'tools' / 'capture_baseline.py'
    result = subprocess.run([str(python), '-c', code, str(legacy_dir), str(tool), module, failure],
                            cwd=cwd, env=clean_environment(python), text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ''
    assert result.stdout.strip() == f'direct import failure passed: {module} {failure}'


@pytest.fixture(scope='module')
def empty_python(tmp_path_factory):
    env = tmp_path_factory.mktemp('missing-distribution') / 'empty'
    result = subprocess.run(['uv', 'venv', '--python', sys.executable, str(env)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return env / 'bin/python'


@pytest.mark.artifact_driver
@pytest.mark.parametrize('module', DIRECT_MODULES)
@pytest.mark.parametrize('failure', IMPORT_FAILURES)
def test_runtime_direct_import_failures(empty_python, legacy_dir, tmp_path, module, failure):
    assert_direct_import_failure(empty_python, legacy_dir, tmp_path, module, failure)


def assert_missing_runtime(python, legacy_dir, cwd):
    probe = subprocess.run([str(python), '-c',
                            "import importlib.util; assert importlib.util.find_spec('creative_comfy_graph') is None"],
                           cwd=cwd, env=clean_environment(python), text=True, capture_output=True)
    assert probe.returncode == 0, probe.stderr
    for mode in ('exports', 'usage', 'dump'):
        launch = cli_child if mode == 'exports' else entrypoint_child
        result = launch(python, legacy_dir, cwd, mode)
        assert result.returncode == 1, result.stdout + result.stderr
        assert result.stderr.rstrip().splitlines()[-1] == 'ImportError: ' + MESSAGE
        assert 'BoundaryViolation' not in result.stderr
        assert result.stdout == ''
        if mode != 'exports':
            assert 'comfy_graph.py' in result.stderr, 'failure must occur inside the actual CLI'


@pytest.mark.artifact_driver
def test_runtime_no_distribution_fixture(legacy_dir, tmp_path):
    env = tmp_path / 'empty'
    result = subprocess.run(['uv', 'venv', '--python', sys.executable, str(env)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert_missing_runtime(env / 'bin/python', legacy_dir, tmp_path)


@pytest.mark.parametrize('mode', ['usage', 'dump'])
def test_runtime_entrypoint_detects_broken_startup(legacy_dir, tmp_path, mode):
    import shutil
    scripts = tmp_path / 'copied-scripts'
    scripts.mkdir()
    for name in ('core', 'flux2', 'ltx2', 'tts', 'post', 'comfy_graph'):
        shutil.copy2(legacy_dir / (name + '.py'), scripts / (name + '.py'))
    cli = scripts / 'comfy_graph.py'
    source = cli.read_text()
    entrypoint = 'if __name__ == "__main__":'
    assert source.count(entrypoint) == 1
    cli.write_text(source.replace(entrypoint, entrypoint + '\n    raise RuntimeError("broken-entrypoint-sentinel")'))
    result = entrypoint_child(sys.executable, scripts, tmp_path, mode)
    assert result.returncode == 1
    assert 'RuntimeError: broken-entrypoint-sentinel' in result.stderr
    assert result.stdout == ''


def test_runtime_entrypoint_requires_natural_sibling_imports(legacy_dir, tmp_path):
    import shutil
    scripts = tmp_path / 'copied-scripts'
    scripts.mkdir()
    for name in ('core', 'flux2', 'ltx2', 'post', 'comfy_graph'):
        shutil.copy2(legacy_dir / (name + '.py'), scripts / (name + '.py'))
    # Deliberately omit tts.py: preloaded modules would hide this broken layout.
    result = entrypoint_child(sys.executable, scripts, tmp_path, 'usage')
    assert result.returncode == 1
    assert "ModuleNotFoundError: No module named 'tts'" in result.stderr
    assert 'comfy_graph.py' in result.stderr
    assert result.stdout == ''


@pytest.mark.parametrize(('attack', 'label'), [
    ("import urllib.request; urllib.request.urlopen('http://invalid.test')", 'urlopen'),
    ("import subprocess; subprocess.run(['must-not-launch'])", 'Popen'),
    ("from pathlib import Path; Path('/synthetic/client-media.png').exists()", 'filesystem:/synthetic/client-media.png'),
])
def test_runtime_entrypoint_guards_precede_script_execution(legacy_dir, tmp_path, attack, label):
    scripts = tmp_path / 'copied-scripts'
    scripts.mkdir()
    (scripts / 'comfy_graph.py').write_text(attack + '\n')
    result = entrypoint_child(sys.executable, scripts, tmp_path, 'usage')
    assert result.returncode == 1
    assert 'BoundaryViolation: ' + label in result.stderr
    assert result.stdout == ''
