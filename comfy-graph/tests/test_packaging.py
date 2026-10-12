"""Distribution metadata and archive allowlists; no builds in inner tests."""
from email.parser import BytesParser
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile

import pytest


def inspect_wheel(path, families):
    modules = {'__init__.py', 'graph.py', 'ltx_profiles.py', 'minimax_h3.py'} | {family + '.py' for family in families - {'primitives'}}
    prefix = 'creative_comfy_graph-0.1.0.dist-info/'
    allowed = {'creative_comfy_graph/' + name for name in modules} | {
        prefix + name for name in ('METADATA', 'WHEEL', 'RECORD', 'licenses/LICENSE')}
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == allowed
        metadata = BytesParser().parsebytes(archive.read(prefix + 'METADATA'))
        assert metadata['Name'] == 'creative-comfy-graph'
        assert metadata['Version'] == '0.1.0'
        assert metadata['Requires-Python'] == '>=3.11'
        assert not metadata.get_all('Requires-Dist')
        assert metadata['License-Expression'] == 'MIT'
        license_text = archive.read(prefix + 'licenses/LICENSE')
        assert b'Copyright (c) 2026 Venetanji' in license_text
        assert 'Root-Is-Purelib: true' in archive.read(prefix + 'WHEEL').decode()
    return license_text


def inspect_sdist(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        files = set()
        for member in members:
            p = PurePosixPath(member.name)
            assert not p.is_absolute() and '..' not in p.parts
            assert p.parts[0] == 'creative_comfy_graph-0.1.0'
            assert member.isfile(), member.name
            relative = PurePosixPath(*p.parts[1:])
            files.add(str(relative))
            assert (str(relative) in {'README.md', 'LTX-PROFILES.md', 'H3-PROFILE.md', 'RECOVERY.md', 'LICENSE', 'pyproject.toml', 'PKG-INFO'} or
                    (relative.parts[0] == 'src' and relative.suffix == '.py') or
                    (relative.parts[0] == 'tests' and relative.suffix in {'.py', '.json'})), member.name
            assert not any(part.startswith('.') or part == '__pycache__' for part in relative.parts)
        assert {'README.md', 'LTX-PROFILES.md', 'H3-PROFILE.md', 'LICENSE', 'pyproject.toml', 'PKG-INFO',
                'src/creative_comfy_graph/__init__.py', 'src/creative_comfy_graph/graph.py',
                'tests/fixtures/api.json', 'tests/fixtures/graphs.json', 'tests/fixtures/errors.json'} <= files
        return archive.extractfile('creative_comfy_graph-0.1.0/LICENSE').read()


def test_packaging_installed_metadata():
    metadata = importlib.metadata.metadata('creative-comfy-graph')
    assert metadata['Version'] == '0.1.0'
    assert metadata['Requires-Python'] == '>=3.11'
    assert not metadata.get_all('Requires-Dist')


def test_packaging_supplied_artifacts(expected_families):
    if not os.environ.get('COMFY_GRAPH_TEST_WHEEL'):
        pytest.skip('archive assertions run in artifact acceptance')
    license_text = inspect_wheel(os.environ['COMFY_GRAPH_TEST_WHEEL'], expected_families)
    assert inspect_sdist(os.environ['COMFY_GRAPH_TEST_SDIST']) == license_text


def test_packaging_wheel_only_import_location():
    if not os.environ.get('COMFY_GRAPH_TEST_WHEEL'):
        pytest.skip('editable development install; artifact acceptance requires site-packages')
    import creative_comfy_graph as package
    path = Path(package.__file__).resolve()
    assert path.is_relative_to(Path(sys.prefix))
    assert 'site-packages' in path.parts
    assert not path.is_relative_to(Path(os.environ['COMFY_GRAPH_TEST_PROJECT']))


@pytest.mark.artifact_driver
def test_packaging_artifact_driver_available():
    assert (Path(__file__).parent / 'tools' / 'accept_artifacts.py').is_file()


@pytest.mark.parametrize('member', ['.env', '../escape.py', 'skills/core.py',
                                    'models/model.safetensors', 'tests/__pycache__/cache.pyc',
                                    'src/creative_comfy_graph/media.png'])
def test_packaging_rejects_unwanted_sdist_members(tmp_path, member):
    import io
    required = {'README.md', 'LTX-PROFILES.md', 'H3-PROFILE.md', 'LICENSE', 'pyproject.toml', 'PKG-INFO',
                'src/creative_comfy_graph/__init__.py', 'src/creative_comfy_graph/graph.py',
                'tests/fixtures/api.json', 'tests/fixtures/graphs.json', 'tests/fixtures/errors.json'}

    def archive_at(path, names):
        with tarfile.open(path, 'w:gz') as archive:
            for name in sorted(names):
                info = tarfile.TarInfo('creative_comfy_graph-0.1.0/' + name)
                info.size = 1
                archive.addfile(info, io.BytesIO(b'x'))

    valid, invalid = tmp_path / 'valid.tar.gz', tmp_path / 'invalid.tar.gz'
    archive_at(valid, required)
    assert inspect_sdist(valid) == b'x'
    archive_at(invalid, required | {member})
    with pytest.raises(AssertionError):
        inspect_sdist(invalid)


@pytest.mark.artifact_driver
def test_packaging_driver_scopes_copy_mode_to_installs(monkeypatch, tmp_path):
    import importlib.util
    import shutil

    path = Path(__file__).parent / 'tools' / 'accept_artifacts.py'
    spec = importlib.util.spec_from_file_location('accept_artifacts', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    project = tmp_path / 'checkout' / 'project'
    legacy, work = tmp_path / 'legacy', tmp_path / 'accept'
    project.mkdir(parents=True)
    legacy.mkdir()
    (legacy / 'h3.py').write_text('# pending local adapter')
    (legacy / 'queue_admission.py').write_text('# pending shared admission helper')
    (legacy.parent / 'requirements-h3.txt').write_text('Pillow>=10\n')
    (legacy / 'unrelated.py').write_text('# must not copy arbitrary untracked scripts')
    copied = []
    # Only exercise command construction; real archive/runtime checks run in acceptance.
    monkeypatch.setattr(shutil, 'copytree', lambda *a, **kw: None)
    monkeypatch.setattr(shutil, 'copy2', lambda source, dest, **kw: copied.append(Path(source).name))
    import test_packaging as checks
    monkeypatch.setattr(importlib.util, 'spec_from_file_location', lambda *a: spec)
    monkeypatch.setattr(importlib.util, 'module_from_spec', lambda *a: checks)
    monkeypatch.setattr(spec.loader, 'exec_module', lambda *a: None)
    monkeypatch.setattr(checks, 'inspect_wheel', lambda *a: b'license')
    monkeypatch.setattr(checks, 'inspect_sdist', lambda *a: b'license')
    (project / 'LICENSE').write_bytes(b'license')
    commands = []

    class InstallsReached(Exception):
        pass

    def run(command, *, cwd, env=None):
        command = list(map(str, command))
        commands.append(command)
        if command[:2] == ['git', 'ls-files']:
            return 'core.py\ncomfy_graph.py\n'
        if command[:2] == ['uv', 'build']:
            out = Path(command[command.index('--out-dir') + 1])
            out.mkdir()
            (out / 'fixture.whl').write_bytes(b'wheel')
            if '--wheel' not in command:
                import io
                with tarfile.open(out / 'fixture.tar.gz', 'w:gz') as archive:
                    member = tarfile.TarInfo('fixture/pyproject.toml')
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b'x'))
        if command[:3] == ['uv', 'pip', 'install'] and command[-1] == 'pytest':
            raise InstallsReached
        return ''

    monkeypatch.setattr(driver, 'run', run)
    monkeypatch.setattr(sys, 'argv', [str(path), '--project', str(project),
                        '--legacy-dir', str(legacy), '--work-dir', str(work),
                        '--expected-families', 'primitives,flux2,tts,post,ltx2',
                        '--python', '3.11', '--python', '3.12'])
    before = dict(os.environ)
    with pytest.raises(InstallsReached):
        driver.main()
    assert 'h3.py' in copied, 'pending H3 adapter must reach copied-script runtime tests'
    assert 'queue_admission.py' in copied, 'shared admission helper must accompany both submission routes'
    assert 'requirements-h3.txt' in copied, 'image caller dependency declaration must accompany copied scripts'
    assert 'unrelated.py' not in copied
    installs = [command for command in commands if command[:3] == ['uv', 'pip', 'install']]
    python = str(work / 'py3.11-wheel/bin/python')
    assert installs == [
        ['uv', 'pip', 'install', '--python', python, '--link-mode=copy', '--no-deps', str(work / 'dist/fixture.whl')],
        ['uv', 'pip', 'install', '--python', python, '--link-mode=copy', '-r', str(work / 'copied-skill/requirements-h3.txt')],
        ['uv', 'pip', 'install', '--python', python, '--link-mode=copy', 'pytest'],
    ]
    assert all('--link-mode=copy' not in command for command in commands if command not in installs)
    assert dict(os.environ) == before
