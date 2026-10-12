"""Offline golden capture. CLI accepts only the pinned, hashed legacy source.

This is a fail-closed test boundary, not a sandbox for hostile Python. Import
I/O is allowed for Python source and the interpreter; client files, discovery,
transport and process execution are denied before any legacy module executes.
"""
import _io
import argparse
import ast
import builtins
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import textwrap
import sys
import time
import urllib.request
from unittest.mock import patch

TESTS = Path(__file__).absolute().parents[1]
# Loading this test-only matrix is explicit, not a package-source fallback.
_spec = importlib.util.spec_from_file_location('characterization_cases', TESTS / 'cases.py')
_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cases)
BUILDERS, CASES = _cases.BUILDERS, _cases.CASES
PIN = '741bab9bb1da495e96949286695fccfd3ec931ff'
FAMILIES = ('flux2', 'tts', 'post', 'ltx2')
CONSTANTS = {
    'ltx2': ('CKPT', 'TEXT_ENCODER', 'UPSCALER', 'DISTILLED_LORA', 'INGREDIENTS_LORA',
             'INGREDIENTS_NEG', 'NEG_DEFAULT', 'SIGMAS_PASS1', 'SIGMAS_PASS2',
             'SAMPLER_PASS1', 'SAMPLER_PASS2', 'CAMERA_LORAS', 'TRANSITION_LORA', 'TRANSITION_TRIGGER'),
    'post': ('COMFY_INPUT_DIR', 'COMFY_OUTPUT_DIR'),
}
SOURCE_HASHES = {
    'core.py': 'adc60e9e4450118a23e49d21683ef66703a2cc5a3e0b6d63452fa6cb444f8cda',
    'flux2.py': 'bb1067bd55e50c40e6f70309e6d0c8ca74be4f2affcf3c2fd72fcbf78396ed9b',
    'tts.py': '52199be9ad85b49965cb0b28ea1b5de7904fa52f9f9927e1c0f7c272213afdf9',
    'post.py': '98599aec072e6728698426a459bdba66ed36052f65074f36025928ce4cd77f95',
    'ltx2.py': 'ce5ac93c52f0a2af51af5283aff039c122c80d477c3c8955bf52785cea614455',
}


class BoundaryViolation(BaseException):
    """BaseException prevents legacy broad except Exception from swallowing I/O."""


@contextmanager
def boundaries(legacy_dir, write_paths=(), *, script_entrypoint=False):
    legacy_dir = Path(os.path.abspath(legacy_dir))
    allowed_sources = {legacy_dir / (name + '.py') for name in ('core', *FAMILIES)}
    interpreter_roots = tuple(Path(p) for p in (sys.base_prefix, sys.prefix))
    package_root = TESTS.parent / 'src' / 'creative_comfy_graph'
    # Snapshot trusted import directories before installing the guards. Metadata
    # permissions are exact paths, never a blanket interpreter-root allowance.
    # Walk directories only; do not follow symlinks or enter OpenClaw directories.
    import_directories = set()
    for root in (*interpreter_roots, package_root):
        for directory, children, _ in os.walk(root, followlinks=False):
            children[:] = [name for name in children
                           if name not in ('.openclaw', '.openclaw-gateway')]
            import_directories.add(Path(directory))
    if script_entrypoint:
        # Natural copied-script imports need the entrypoint, sibling directory
        # metadata and exact cache read paths (but never bytecode writes).
        allowed_sources.add(legacy_dir / 'comfy_graph.py')
        allowed_sources.update(Path(importlib.util.cache_from_source(str(path)))
                               for path in tuple(allowed_sources))
        import_directories.update((legacy_dir, legacy_dir / '__pycache__'))
    write_paths = {Path(os.path.abspath(p)) for p in write_paths}
    attempts = []
    old_open, old_io_open, old_os_open = builtins.open, io.open, os.open
    old_stat, old_lstat, old_readlink = os.stat, os.lstat, os.readlink
    old_open_code = _io.open_code
    old_listdir, old_scandir = os.listdir, os.scandir

    def deny(label):
        def blocked(*args, **kwargs):
            attempts.append(label)
            raise BoundaryViolation(label)
        return blocked

    def check(path, writing=False, directory=False):
        if isinstance(path, int):
            deny('file-descriptor-access')()
        p = Path(os.path.abspath(os.fsdecode(path)))
        if writing:
            allowed = p in write_paths
        else:
            # Explicit archive files; ordinary imports within Python's runtime
            # and installed packages. OpenClaw discovery never becomes allowed.
            runtime = any(p.is_relative_to(root) for root in interpreter_roots)
            imported_package = p.is_relative_to(package_root) and p.suffix in ('.py', '.pyc')
            allowed = (p in allowed_sources or p == TESTS / 'cases.py' or
                       (runtime and p.suffix in ('.py', '.pyc', '.so', '.pth')) or
                       imported_package or (directory and p in import_directories))
        if '.openclaw' in p.parts or '.openclaw-gateway' in p.parts:
            allowed = False
        if not allowed:
            deny('filesystem:' + str(p))()

    def guarded_open(path, mode='r', *args, **kwargs):
        check(path, any(c in mode for c in 'wax+'))
        return old_open(path, mode, *args, **kwargs)

    def guarded_io_open(path, mode='r', *args, **kwargs):
        check(path, any(c in mode for c in 'wax+'))
        return old_io_open(path, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        check(path, bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        return old_os_open(path, flags, *args, **kwargs)

    def guarded_open_code(path):
        check(path)
        return old_open_code(path)

    def probe(original):
        def guarded(path, *args, **kwargs):
            check(path, directory=True)
            return original(path, *args, **kwargs)
        return guarded

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'dont_write_bytecode', True))
        stack.enter_context(patch.object(time, 'time', return_value=1700000000.125))
        stack.enter_context(patch.dict(os.environ, {'LTX_SAGE_ATTENTION': 'auto'}))
        for owner, name in ((socket, 'create_connection'), (socket, 'getaddrinfo'),
                            (socket, 'gethostbyname'), (socket, 'gethostbyname_ex'),
                            (socket, 'gethostbyaddr'), (socket.socket, 'connect'),
                            (socket.socket, 'connect_ex'), (socket.socket, 'sendto'),
                            (urllib.request, 'urlopen'), (urllib.request, 'urlretrieve'),
                            (urllib.request.OpenerDirector, 'open'), (subprocess, 'Popen'),
                            (os, 'system'), (os, 'popen'), (os, 'fork'), (os, 'posix_spawn'),
                            (os, 'posix_spawnp'), (os, 'execve'), (os, 'execv'),
                            (Path, 'home')):
            if hasattr(owner, name):
                stack.enter_context(patch.object(owner, name, deny(name)))
        for owner, name, replacement in (
            (builtins, 'open', guarded_open), (io, 'open', guarded_io_open),
            (os, 'open', guarded_os_open), (os, 'stat', probe(old_stat)),
            (os, 'lstat', probe(old_lstat)), (os, 'readlink', probe(old_readlink)),
            (_io, 'open_code', guarded_open_code), (io, 'open_code', guarded_open_code),
            (os, 'listdir', probe(old_listdir)),
            (os, 'scandir', probe(old_scandir)),
        ):
            stack.enter_context(patch.object(owner, name, replacement))
        for name in ('mkdir', 'makedirs', 'remove', 'unlink', 'rename', 'replace', 'rmdir'):
            stack.enter_context(patch.object(os, name, deny(name)))
        yield attempts


@contextmanager
def legacy_modules(directory):
    names = ('core', *FAMILIES)
    saved = {name: sys.modules.get(name) for name in names}
    modules = {}
    try:
        for name in names:
            path = Path(directory) / (name + '.py')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # Explicit source read uses the guarded open, unlike loader-internal
            # open_code. No CLI, sibling discovery or sys.path insertion.
            source = path.read_text()
            exec(compile(source, str(path), 'exec'), module.__dict__)
            modules[name] = module
        yield modules
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def api_record(function):
    signature = inspect.signature(function)
    source = function.__init__ if inspect.isclass(function) else function
    node = ast.parse(textwrap.dedent(inspect.getsource(source))).body[0]
    dump_options = {'show_empty': True} if sys.version_info >= (3, 13) else {}
    return {'signature': str(signature), 'arguments_ast': ast.dump(node.args, include_attributes=False, **dump_options),
            'returns_ast': ast.dump(node.returns) if node.returns else None}


def capture(directory, package_family=None):
    directory = Path(os.path.abspath(directory))
    with boundaries(directory) as attempts, legacy_modules(directory) as modules:
        api = {'provenance': {'revision': PIN, 'clock': 1700000000.125,
                              'source_sha256': {name + '.py': hashlib.sha256((directory / (name + '.py')).read_bytes()).hexdigest()
                                                for name in ('core', *FAMILIES)}},
               'builders': {}, 'exports': {}, 'constants': {}}
        api['primitives'] = {}
        for class_name, methods in (('NodeRef', ('__getitem__', 'as_link', '__repr__')),
                                    ('WorkflowGraph', ('node', 'to_dict', 'to_json'))):
            cls = getattr(modules['core'], class_name)
            api['primitives'][class_name] = api_record(cls)
            for method in methods:
                api['primitives'][class_name + '.' + method] = api_record(getattr(cls, method))
        for family in FAMILIES:
            module = modules[family]
            api['exports'][family] = [name for name in vars(module) if not name.startswith('_')]
            api['constants'][family] = {name: getattr(module, name) for name in CONSTANTS.get(family, ())}
        if package_family:
            # Normal installed import only; expected missing modules must fail.
            modules[package_family] = importlib.import_module('creative_comfy_graph.' + package_family)
            api['constants'][package_family] = {name: getattr(modules[package_family], name)
                                                for name in CONSTANTS.get(package_family, ())}
        for name, (family, _) in BUILDERS.items():
            api['builders'][name] = api_record(getattr(modules[family], name))
        graphs, errors = {}, {}
        for case in CASES:
            if case['attention'] is None:
                os.environ.pop('LTX_SAGE_ATTENTION', None)
            else:
                os.environ['LTX_SAGE_ATTENTION'] = case['attention']
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    graph = getattr(modules[case['family']], case['builder'])(**case['kwargs'])
                except Exception as exc:
                    errors[case['id']] = {'type': type(exc).__name__, 'message': str(exc),
                                          'stdout': stdout.getvalue(), 'stderr': stderr.getvalue()}
                else:
                    graphs[case['id']] = {'graph': graph, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue()}
        assert not attempts, f'Forbidden boundary attempts: {attempts}'
    return api, graphs, errors


def uncovered_parameters(directory):
    missing = {}
    with boundaries(directory), legacy_modules(directory) as modules:
        for name, (family, _) in BUILDERS.items():
            params = inspect.signature(getattr(modules[family], name)).parameters
            custom = next(case['kwargs'] for case in CASES if case['id'] == name + '/custom')
            gaps = [key for key, param in params.items()
                    if param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
                    and (key not in custom or (param.default is not param.empty and custom[key] == param.default))]
            if gaps:
                missing[name] = gaps
    return missing


def exercise_boundaries(directory, temp):
    media = Path(temp) / 'client.png'
    evidence = {}

    def socket_probe(method):
        # Close sockets even when the intentional boundary exception interrupts
        # the operation; strict artifact runs treat ResourceWarning as an error.
        with socket.socket() as connection:
            getattr(connection, method)(('127.0.0.1', 9))

    with boundaries(directory), legacy_modules(directory) as modules:
        actions = {
            'socket-connect': lambda: socket_probe('connect'),
            'socket-connect-ex': lambda: socket_probe('connect_ex'),
            'connection': lambda: socket.create_connection(('127.0.0.1', 9)),
            'dns': lambda: socket.getaddrinfo('invalid.test', 80),
            'hostname': lambda: socket.gethostbyname('invalid.test'),
            'urllib': lambda: urllib.request.urlopen('http://invalid.test'),
            'retrieve': lambda: urllib.request.urlretrieve('http://invalid.test', media),
            'opener': lambda: urllib.request.build_opener().open('http://invalid.test'),
            'subprocess': lambda: subprocess.run(['must-not-launch']),
            'system': lambda: os.system('must-not-launch'),
            'popen': lambda: os.popen('must-not-launch'),
            'open': lambda: open(media, 'rb'),
            'write': lambda: open(media, 'wb'),
            'os-open': lambda: os.open(media, os.O_RDONLY),
            'path-read': lambda: media.read_bytes(),
            'path-text': lambda: media.read_text(),
            'path-exists': lambda: media.exists(),
            'path-is-file': lambda: media.is_file(),
            'path-stat': lambda: media.stat(),
            'listdir': lambda: os.listdir(temp),
            'glob': lambda: list(Path(temp).glob('*')),
            'home': lambda: Path.home(),
            'openclaw': lambda: Path('/home/synthetic/.openclaw/config.json').exists(),
            'mkdir': lambda: media.mkdir(),
            'upload': lambda: modules['core'].upload_image(str(media)),
            'upload-if-local': lambda: modules['flux2'].upload_if_local(str(media)),
            'stage-clips': lambda: modules['post'].stage_clips_for_concat([str(media)], 'test'),
            'submit': lambda: modules['core']._submit_and_wait({}, temp),
        }
        for name, action in actions.items():
            try:
                action()
            except BoundaryViolation:
                evidence[name] = 'blocked'
            else:
                evidence[name] = 'NOT BLOCKED'
    return evidence


def serialize(result):
    return tuple(json.dumps(value, indent=2, ensure_ascii=True) + '\n' for value in result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Provenance check before any legacy import. No git/subprocess is needed.
    with boundaries(args.legacy_dir):
        hashes = {name + '.py': hashlib.sha256((args.legacy_dir / (name + '.py')).read_bytes()).hexdigest()
                  for name in ('core', *FAMILIES)}
    if hashes != SOURCE_HASHES:
        parser.error('legacy source does not match immutable revision ' + PIN)
    result = capture(args.legacy_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / (name + '.json') for name in ('api', 'graphs', 'errors')]
    with boundaries(args.legacy_dir, write_paths=paths):
        for path, text in zip(paths, serialize(result)):
            path.write_text(text)
    print(f'Captured {len(result[0]["builders"])} builders: {len(result[1])} graphs, {len(result[2])} errors; zero forbidden calls')


if __name__ == '__main__':
    main()
