#!/usr/bin/env python3
"""Resumable YouTube Data API v3 uploader; credentials come from a private helper."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

API_ROOT = 'https://www.googleapis.com'
VIDEO_ENDPOINT = API_ROOT + '/upload/youtube/v3/videos'
VIDEO_LIST_ENDPOINT = API_ROOT + '/youtube/v3/videos'
CHANNEL_LIST_ENDPOINT = API_ROOT + '/youtube/v3/channels'
TRANSIENT = {429, 500, 502, 503, 504}
PRIVACY = {'private', 'unlisted', 'public'}
CHUNK_ALIGNMENT = 256 * 1024


class UploadError(RuntimeError):
    """Safe-to-display failure; never includes tokens or resumable URLs."""


class TransportError(UploadError):
    pass


class HelperTokenProvider:
    """Run an operator-owned helper; capture its short-lived token privately."""

    def __init__(self, helper: Path, account: str):
        helper = helper.expanduser()
        if not helper.is_absolute() or not helper.is_file() or not os.access(helper, os.X_OK):
            raise UploadError('Token helper must be an absolute executable path')
        if helper.is_symlink():
            raise UploadError('Token helper must not be a symlink')
        self.helper = helper.resolve()
        repo = Path(__file__).resolve().parents[2]
        if _inside(self.helper, repo):
            raise UploadError('Token helper must be private and outside the shared repository')
        self.account = account
        self._token: str | None = None

    def __call__(self, force_refresh: bool = False) -> str:
        if self._token is not None and not force_refresh:
            return self._token
        args = [str(self.helper), '--account', self.account]
        if force_refresh:
            args.append('--force-refresh')
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=90, check=False)
            if result.returncode != 0:
                raise UploadError('Private token helper failed; its output was suppressed')
            token = json.loads(result.stdout).get('access_token')
        except UploadError:
            raise
        except Exception:
            raise UploadError('Private token helper returned invalid output; details suppressed') from None
        if not isinstance(token, str) or not token:
            raise UploadError('Private token helper returned no access token')
        self._token = token
        return token


class UrllibTransport:
    def __call__(self, url: str, method: str, data: bytes | None,
                 headers: dict[str, str], timeout: int):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            # urllib exceptions can contain the resumable URL; do not propagate it.
            raise TransportError('YouTube transport failed; resumable state was retained') from None


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _state_path(path: Path) -> Path:
    if path.is_symlink():
        raise UploadError('State file must not be a symlink')
    path = path.expanduser().absolute()
    repo = Path(__file__).resolve().parents[2]
    resolved_parent = path.parent.resolve()
    resolved = resolved_parent / path.name
    if _inside(resolved, repo):
        raise UploadError('Upload state must be stored outside the repository')
    resolved_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = resolved_parent.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise UploadError('State directory must be owned by this user and private (0700)')
    if resolved.exists():
        info = resolved.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise UploadError('State file must be a private regular file owned by this user')
    return resolved


@contextmanager
def _state_lock(path: Path):
    lock_path = path.with_name(path.name + '.lock')
    if lock_path.is_symlink():
        raise UploadError('State lock must not be a symlink')
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError:
        raise UploadError('Cannot open private state lock') from None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise UploadError('State lock must be a private regular file owned by this user')
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _save_state(path: Path, state: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix='.youtube-upload-', suffix='.tmp', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(state, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _fingerprint(source: Path) -> dict:
    if source.is_symlink() or not source.is_file() or source.suffix.lower() != '.mp4':
        raise UploadError('Source must be a regular, non-symlink .mp4 file')
    source = source.resolve()
    before = source.stat()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = source.stat()
    if after.st_size <= 0:
        raise UploadError('Source MP4 must not be empty')
    attrs = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, name) != getattr(after, name) for name in attrs):
        raise UploadError('Source changed while its identity was being computed')
    return {'path': str(source), 'size': after.st_size, 'sha256': digest.hexdigest(),
            'device': after.st_dev, 'inode': after.st_ino, 'mtime_ns': after.st_mtime_ns,
            'ctime_ns': after.st_ctime_ns}


def _validate_request(account: str, channel: str, title: str, category: str,
                      privacy: str, notify: bool) -> dict:
    if not account or not channel or not title or len(title) > 100:
        raise UploadError('Account, expected channel ID, and a 1..100 character title are required')
    if not re.fullmatch(r'\d{1,3}', str(category)):
        raise UploadError('Category ID must be a numeric YouTube category')
    if privacy not in PRIVACY:
        raise UploadError('Privacy must be private, unlisted, or public')
    return {'account': account, 'expected_channel': channel, 'title': title,
            'category_id': str(category), 'privacy': privacy,
            'notify_subscribers': bool(notify)}


def _valid_session_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        return (parsed.scheme == 'https' and parsed.hostname == 'www.googleapis.com'
                and parsed.path.startswith('/upload/youtube/v3/videos')
                and bool(parsed.query))
    except Exception:
        return False


def _offset(headers: dict[str, str], total: int) -> int:
    value = headers.get('Range') or headers.get('range')
    if value is None:
        return 0
    match = re.fullmatch(r'bytes=0-(\d+)', value.strip())
    if not match:
        raise UploadError('YouTube returned an invalid resumable range')
    result = int(match.group(1)) + 1
    if result > total:
        raise UploadError('YouTube resumable range exceeds the source size')
    return result


class ResumableUploader:
    def __init__(self, token_provider: Callable[[bool], str], transport=None,
                 chunk_size: int = 16 * 1024 * 1024, max_retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep):
        if chunk_size < CHUNK_ALIGNMENT or chunk_size % CHUNK_ALIGNMENT:
            raise ValueError('Chunk size must be a positive multiple of 256 KiB')
        self.tokens = token_provider
        self.transport = transport or UrllibTransport()
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.sleep = sleep

    def _request(self, url: str, method: str, data: bytes | None,
                 headers: dict[str, str] | None = None, retry_401: bool = True):
        token = self.tokens(False)
        request_headers = dict(headers or {})
        request_headers['Authorization'] = 'Bearer ' + token
        try:
            result = self.transport(url, method, data, request_headers, 180)
        except UploadError:
            raise
        except Exception:
            raise TransportError('YouTube transport failed; resumable state was retained') from None
        if result[0] == 401 and retry_401:
            self.tokens(True)
            request_headers['Authorization'] = 'Bearer ' + self.tokens(False)
            try:
                result = self.transport(url, method, data, request_headers, 180)
            except UploadError:
                raise
            except Exception:
                raise TransportError('YouTube transport failed after token refresh; state was retained') from None
        return result

    def _verify_account_channel(self, request: dict) -> None:
        query = urllib.parse.urlencode({'part': 'id', 'mine': 'true'})
        status, _, body = self._request(CHANNEL_LIST_ENDPOINT + '?' + query, 'GET', None)
        if status != 200:
            raise UploadError('Cannot confirm the authorized account channel (HTTP %d); upload stopped' % status)
        try:
            channels = {item['id'] for item in json.loads(body).get('items', [])}
        except Exception:
            raise UploadError('YouTube returned incomplete account-channel data; upload stopped') from None
        if request['expected_channel'] not in channels:
            raise UploadError('Authorized account does not own the expected channel; upload stopped')

    def _verify(self, video_id: str, request: dict) -> dict:
        query = urllib.parse.urlencode({'part': 'snippet,status,processingDetails,contentDetails', 'id': video_id})
        status, _, body = self._request(VIDEO_LIST_ENDPOINT + '?' + query, 'GET', None)
        if status != 200:
            raise UploadError('Upload exists but YouTube read-back verification failed (HTTP %d)' % status)
        try:
            items = json.loads(body).get('items', [])
            video = items[0]
            actual_channel = video['snippet']['channelId']
            actual_privacy = video['status']['privacyStatus']
            actual_title = video['snippet']['title']
        except Exception:
            raise UploadError('Upload exists but YouTube returned incomplete verification data') from None
        if actual_channel != request['expected_channel']:
            raise UploadError('Upload exists on an unexpected channel; do not re-upload')
        if actual_privacy != request['privacy']:
            raise UploadError('Upload exists with different privacy; no automatic privacy change was made')
        if actual_title != request['title']:
            raise UploadError('Upload exists with a different title; do not re-upload')
        processing = video.get('processingDetails', {}).get('processingStatus', 'unknown')
        return {'video_id': video_id, 'url': 'https://www.youtube.com/watch?v=' + video_id,
                'channel_id': actual_channel, 'title': actual_title,
                'privacy': actual_privacy, 'upload_status': video['status'].get('uploadStatus', 'unknown'),
                'processing_status': processing, 'content_details': video.get('contentDetails', {})}

    def _complete(self, state_path: Path, state: dict, video_id: str) -> dict:
        state['status'] = 'uploaded'
        state['video_id'] = video_id
        state.pop('session_url', None)
        _save_state(state_path, state)
        return self._verify(video_id, state['request'])

    def upload(self, source: Path, state_path: Path, account: str,
               expected_channel: str, title: str, category_id: str,
               privacy: str = 'private', notify_subscribers: bool = False) -> dict:
        source = Path(source).expanduser()
        state_path = _state_path(Path(state_path))
        with _state_lock(state_path):
            return self._upload_locked(source, state_path, account, expected_channel,
                                       title, category_id, privacy, notify_subscribers)

    def _upload_locked(self, source: Path, state_path: Path, account: str,
                       expected_channel: str, title: str, category_id: str,
                       privacy: str, notify_subscribers: bool) -> dict:
        source = Path(source).expanduser()
        state_path = _state_path(Path(state_path))
        identity = _fingerprint(source)
        request = _validate_request(account, expected_channel, title, category_id,
                                    privacy, notify_subscribers)
        try:
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        except Exception:
            raise UploadError('Cannot read private upload state; preserve it for inspection') from None
        if state is not None:
            if state.get('source') != identity:
                raise UploadError('Source identity differs from the saved upload state')
            if state.get('request') != request:
                raise UploadError('Upload metadata/privacy differs from saved state; no change or new upload made')
        # Confirm account ownership before resuming, creating a session, or read-back.
        self._verify_account_channel(request)
        if state is not None:
            if state.get('status') == 'initiation_uncertain':
                raise UploadError('Session creation is ambiguous; reconcile the account before any new session')
            if state.get('video_id'):
                return self._verify(state['video_id'], request)
        else:
            state = {'source': identity, 'request': request, 'status': 'ready'}

        if not state.get('session_url'):
            if state.get('status') not in ('ready', 'start_rejected'):
                raise UploadError('No resumable session is recorded; manual reconciliation is required')
            state['status'] = 'initiating'
            _save_state(state_path, state)
            params = urllib.parse.urlencode({'uploadType': 'resumable', 'part': 'snippet,status',
                                             'notifySubscribers': str(request['notify_subscribers']).lower()})
            metadata = {'snippet': {'title': title, 'categoryId': request['category_id']},
                        'status': {'privacyStatus': privacy}}
            headers = {'Content-Type': 'application/json; charset=UTF-8',
                       'X-Upload-Content-Length': str(identity['size']),
                       'X-Upload-Content-Type': 'video/mp4'}
            try:
                status, response_headers, body = self._request(
                    VIDEO_ENDPOINT + '?' + params, 'POST', json.dumps(metadata).encode(), headers)
            except TransportError:
                state['status'] = 'initiation_uncertain'
                _save_state(state_path, state)
                raise UploadError('Session creation outcome is ambiguous; inspect the account before retrying') from None
            except UploadError:
                state['status'] = 'start_rejected'
                _save_state(state_path, state)
                raise
            location = response_headers.get('Location') or response_headers.get('location')
            if status not in (200, 201):
                state['status'] = 'initiation_uncertain' if status in TRANSIENT else 'start_rejected'
                state['last_http_status'] = status
                _save_state(state_path, state)
                if status in TRANSIENT:
                    raise UploadError('Session creation returned a transient server response; reconcile before retrying')
                raise UploadError('YouTube rejected resumable session creation (HTTP %d); no automatic retry' % status)
            if not location or not _valid_session_url(location):
                state['status'] = 'initiation_uncertain'
                _save_state(state_path, state)
                raise UploadError('YouTube session response was incomplete; reconcile before retrying')
            state['status'] = 'uploading'
            state['session_url'] = location
            _save_state(state_path, state)

        session = state['session_url']
        if not _valid_session_url(session):
            raise UploadError('Saved resumable URL is invalid; preserve state for inspection')
        total = identity['size']
        failures = 0
        stuck = 0
        last_server_offset = None
        while True:
            try:
                status, headers, body = self._request(
                    session, 'PUT', b'', {'Content-Length': '0',
                    'Content-Range': 'bytes */%d' % total})
            except TransportError:
                failures += 1
                if failures > self.max_retries:
                    raise UploadError('Could not reconcile resumable position; state retained') from None
                self.sleep(min(2 ** (failures - 1), 8))
                continue
            if status in (200, 201):
                try:
                    video_id = json.loads(body)['id']
                except Exception:
                    raise UploadError('YouTube completed the upload without a readable video ID') from None
                return self._complete(state_path, state, video_id)
            if status == 404 or status == 410:
                raise UploadError('Resumable session expired; do not create a replacement until reconciled')
            if status != 308:
                if status in TRANSIENT and failures < self.max_retries:
                    failures += 1
                    self.sleep(min(2 ** (failures - 1), 8))
                    continue
                raise UploadError('YouTube position query failed (HTTP %d); state retained' % status)
            offset = _offset(headers, total)
            if last_server_offset is not None and offset < last_server_offset:
                raise UploadError('YouTube resumable position moved backwards')
            if last_server_offset is None or offset > last_server_offset:
                failures = 0
            last_server_offset = offset
            if offset == total:
                stuck += 1
                if stuck > 2:
                    raise UploadError('YouTube has all bytes but has not finalized; state retained')
                continue
            if offset > total:
                raise UploadError('YouTube offset exceeds source size')
            current = offset
            try:
                with source.open('rb') as stream:
                    st = os.fstat(stream.fileno())
                    expected = (identity['device'], identity['inode'], identity['size'],
                                identity['mtime_ns'], identity['ctime_ns'])
                    if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != expected:
                        raise UploadError('Source changed after its identity was recorded')
                    stream.seek(current)
                    chunk = stream.read(min(self.chunk_size, total - current))
                    st = os.fstat(stream.fileno())
                    if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != expected:
                        raise UploadError('Source changed while a chunk was read')
                if not chunk:
                    raise UploadError('Unexpected end of source; state retained')
                end = current + len(chunk) - 1
                status, response_headers, body = self._request(
                    session, 'PUT', chunk, {'Content-Type': 'video/mp4',
                    'Content-Length': str(len(chunk)),
                    'Content-Range': 'bytes %d-%d/%d' % (current, end, total)},
                    retry_401=False)
            except UploadError as error:
                if not isinstance(error, TransportError):
                    raise
                failures += 1
                if failures > self.max_retries:
                    raise UploadError('Chunk transfer interrupted; resumable state retained') from None
                self.sleep(min(2 ** (failures - 1), 8))
                continue
            if status == 401:
                self.tokens(True)
                failures += 1
                if failures > self.max_retries:
                    raise UploadError('Authorization failed after refresh; state retained')
                continue  # next operation reconciles server offset before any retry
            if status in (200, 201):
                try:
                    video_id = json.loads(body)['id']
                except Exception:
                    raise UploadError('YouTube completed the upload without a readable video ID') from None
                return self._complete(state_path, state, video_id)
            if status == 308:
                accepted = _offset(response_headers, total)
                if accepted <= current:
                    stuck += 1
                    if stuck > self.max_retries:
                        raise UploadError('YouTube made no upload progress; state retained')
                else:
                    stuck = 0
                    failures = 0
                    last_server_offset = accepted
                continue
            if status in TRANSIENT and failures < self.max_retries:
                failures += 1
                self.sleep(min(2 ** (failures - 1), 8))
                continue
            raise UploadError('YouTube chunk request failed (HTTP %d); state retained' % status)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Upload or resume one MP4 through YouTube Data API v3.')
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path, help='Private JSON state file outside this repository')
    parser.add_argument('--account', required=True, help='Google account name used by the private token helper')
    parser.add_argument('--token-helper', required=True, type=Path,
                        help='Operator-owned executable returning JSON access_token; never stores secrets in this repo')
    parser.add_argument('--expected-channel', required=True, help='Channel ID from read-only YouTube channel listing')
    parser.add_argument('--title', required=True)
    parser.add_argument('--category-id', required=True)
    parser.add_argument('--privacy', choices=sorted(PRIVACY), default='private')
    parser.add_argument('--notify-subscribers', action='store_true', default=False)
    parser.add_argument('--chunk-mib', type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        helper = HelperTokenProvider(args.token_helper, args.account)
        uploader = ResumableUploader(helper, chunk_size=args.chunk_mib * 1024 * 1024)
        result = uploader.upload(args.source, args.state, args.account, args.expected_channel,
                                 args.title, args.category_id, args.privacy,
                                 args.notify_subscribers)
        print(json.dumps(result, sort_keys=True))
        return 0
    except UploadError as error:
        print('YOUTUBE_UPLOAD_STOPPED: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
