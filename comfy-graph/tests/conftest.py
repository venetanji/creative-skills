from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption('--legacy-dir', type=Path, required=True)
    parser.addoption('--expected-families', default='none')


@pytest.fixture(scope='session')
def legacy_dir(request):
    return request.config.getoption('--legacy-dir')


@pytest.fixture(scope='session')
def expected_families(request):
    value = request.config.getoption('--expected-families')
    families = set(value.split(','))
    assert families <= {'none', 'primitives', 'flux2', 'tts', 'post', 'ltx2'}
    assert 'none' not in families or families == {'none'}
    return families - {'none'}


def pytest_configure(config):
    # Copied tests intentionally have no repository pyproject configuration.
    config.addinivalue_line('markers', 'artifact_driver: outer build/venv orchestration, excluded from artifact inner runs')
