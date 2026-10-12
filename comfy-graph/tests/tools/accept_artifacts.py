"""Build and accept standalone distributions without source/package path tricks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def run(command, *, cwd, env=None):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    if result.returncode:
        raise RuntimeError(f'command exited {result.returncode}: {command}')
    return result.stdout


def only(directory, pattern):
    paths = list(directory.glob(pattern))
    assert len(paths) == 1, paths
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--legacy-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--expected-families', required=True)
    parser.add_argument('--python', action='append', required=True)
    args = parser.parse_args()
    project, legacy, work = (p.resolve() for p in (args.project, args.legacy_dir, args.work_dir))
    assert not work.is_relative_to(project.parent), 'work directory must be outside checkout'
    families = set(args.expected_families.split(','))
    assert 'primitives' in families and families <= {'primitives', 'flux2', 'tts', 'post', 'ltx2'}
    assert set(args.python) == {'3.11', '3.12'}, 'both actual supported interpreter versions required'
    work.mkdir(parents=True, exist_ok=False)
    cwd = work / 'empty-cwd'
    cwd.mkdir()
    tests, scripts = work / 'tests', work / 'copied-skill/scripts'
    shutil.copytree(project / 'tests', tests, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc'))
    scripts.mkdir(parents=True)
    tracked = run(['git', 'ls-files', '--', '*.py'], cwd=legacy).splitlines()
    assert 'comfy_graph.py' in tracked and 'core.py' in tracked
    # Explicit pending helpers may still be untracked in a review-only worktree.
    for name in ('h3.py', 'queue_admission.py', 'audio_input.py', 'ltx_profile.py'):
        if (legacy / name).is_file() and name not in tracked:
            tracked.append(name)
    for name in tracked:
        if '/' not in name and name != 'test_all_workflows.py':
            shutil.copy2(legacy / name, scripts / name)
    requirements = legacy.parent / 'requirements-h3.txt'
    assert requirements.is_file(), 'image caller dependency declaration is required'
    copied_requirements = scripts.parent / 'requirements-h3.txt'
    shutil.copy2(requirements, copied_requirements)
    # Test helpers are explicitly loaded; production imports remain normal.
    import importlib.util
    spec = importlib.util.spec_from_file_location('artifact_checks', tests / 'test_packaging.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    dist, rebuilt = work / 'dist', work / 'rebuilt'
    run(['uv', 'build', project, '--out-dir', dist], cwd=cwd)
    wheel, sdist = only(dist, '*.whl'), only(dist, '*.tar.gz')
    license_text = checks.inspect_wheel(wheel, families)
    assert checks.inspect_sdist(sdist) == license_text == (project / 'LICENSE').read_bytes()
    extracted = work / 'sdist-source'
    extracted.mkdir()
    with tarfile.open(sdist, 'r:gz') as archive:
        # inspect_sdist rejects absolute/traversal paths, links and special files.
        archive.extractall(extracted, filter='data')
    run(['uv', 'build', only(extracted, '*'), '--wheel', '--out-dir', rebuilt], cwd=cwd)
    rebuilt_wheel = only(rebuilt, '*.whl')
    assert checks.inspect_wheel(rebuilt_wheel, families) == license_text
    evidence = {'families': sorted(families), 'artifacts': {}, 'environments': []}
    for path in (wheel, sdist, rebuilt_wheel):
        evidence['artifacts'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    base_env = dict(os.environ)
    base_env.pop('PYTHONPATH', None)
    base_env.pop('PYTHONHOME', None)
    base_env.pop('VIRTUAL_ENV', None)
    base_env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    for version in args.python:
        for label, artifact in (('wheel', wheel), ('rebuilt', rebuilt_wheel)):
            environment = work / f'py{version}-{label}'
            run(['uv', 'venv', '--python', version, environment], cwd=cwd)
            python = environment / 'bin/python'
            run(['uv', 'pip', 'install', '--python', python, '--link-mode=copy', '--no-deps', artifact], cwd=cwd)
            run(['uv', 'pip', 'install', '--python', python, '--link-mode=copy', '-r', copied_requirements], cwd=cwd)
            run(['uv', 'pip', 'install', '--python', python, '--link-mode=copy', 'pytest'], cwd=cwd)
            env = dict(base_env, PATH=str(environment / 'bin') + os.pathsep + base_env['PATH'],
                       COMFY_GRAPH_TEST_WHEEL=str(artifact), COMFY_GRAPH_TEST_SDIST=str(sdist),
                       COMFY_GRAPH_TEST_PROJECT=str(project))
            probe = '''import json, sys, creative_comfy_graph as p
from pathlib import Path
assert sys.version_info[:2] == tuple(map(int, sys.argv[1].split('.')))
assert Path(p.__file__).resolve().is_relative_to(Path(sys.prefix))
assert 'site-packages' in Path(p.__file__).parts
print(json.dumps({'version':sys.version,'executable':sys.executable,'prefix':sys.prefix,'package':p.__file__}))'''
            info = json.loads(run([python, '-c', probe, version], cwd=cwd, env=env))
            output = run([python, '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider', tests,
                          '--legacy-dir', scripts, '--expected-families', args.expected_families,
                          '-m', 'not artifact_driver', '--strict-markers', '-W', 'error'], cwd=cwd, env=env)
            assert 'skipped' not in output, 'artifact tests must exercise every inner gate'
            evidence['environments'].append(dict(info, artifact=str(artifact), result=output.splitlines()[-1]))
        # A separate empty interpreter proves no source/user-site/installed-skill fallback.
        empty = work / f'py{version}-missing'
        run(['uv', 'venv', '--python', version, empty], cwd=cwd)
        # Use the provisioned wheel's pytest to launch the empty interpreter fixture;
        # the actual CLI/import child has no installed distribution or pytest.
        negative = f'''import importlib.util
from pathlib import Path
path = Path({str(tests / 'test_runtime.py')!r})
spec = importlib.util.spec_from_file_location('runtime_checks', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.assert_missing_runtime(Path({str(empty / 'bin/python')!r}), Path({str(scripts)!r}), Path({str(cwd)!r}))
print('missing-package fixture passed: Python {version}')
'''
        # pytest supplies test-module import paths without changing production resolution.
        negative_test = tests / 'test_artifact_missing.py'
        direct = f'''
import pytest
from test_runtime import DIRECT_MODULES, IMPORT_FAILURES, assert_direct_import_failure
from pathlib import Path

@pytest.mark.parametrize('module', DIRECT_MODULES)
@pytest.mark.parametrize('failure', IMPORT_FAILURES)
def test_direct_import_failures(module, failure):
    assert_direct_import_failure(Path({str(empty / 'bin/python')!r}), Path({str(scripts)!r}), Path({str(cwd)!r}), module, failure)
'''
        negative_test.write_text(direct + '\ndef test_missing_distribution():\n' +
                                 ''.join('    ' + line + '\n' for line in negative.splitlines()))
        run([python, '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider', negative_test,
             '--legacy-dir', scripts, '--expected-families', args.expected_families], cwd=cwd, env=env)
        negative_test.unlink()
    (work / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
