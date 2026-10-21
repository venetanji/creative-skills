import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.youtube_upload import ResumableUploader, UploadError, _state_lock


SESSION = 'https://www.googleapis.com/upload/youtube/v3/videos?upload_id=opaque'
VIDEO_ID = 'test-video-123'
CHANNEL = 'test-channel-id'


def video(privacy='private', channel=CHANNEL):
    return json.dumps({'items': [{'id': VIDEO_ID, 'snippet': {
        'channelId': channel, 'title': 'A test upload', 'categoryId': '10'},
        'status': {'privacyStatus': privacy, 'uploadStatus': 'uploaded'},
        'processingDetails': {'processingStatus': 'processing'},
        'contentDetails': {'duration': 'PT1S'}}]}).encode()


class Tokens:
    def __init__(self):
        self.refreshes = 0

    def __call__(self, force_refresh=False):
        if force_refresh:
            self.refreshes += 1
        return 'access-token-never-persist'


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.all_calls = []
        self.channel_response = (200, {}, json.dumps({'items': [{'id': CHANNEL}]}).encode())

    def __call__(self, url, method, data, headers, timeout):
        call = (url, method, data, dict(headers))
        self.all_calls.append(call)
        if '/youtube/v3/channels?' in url:
            return self.channel_response
        self.calls.append(call)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class ResumableUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'x' * (256 * 1024 + 3))
        self.state = self.root / 'private' / 'upload.json'

    def tearDown(self):
        self.temp.cleanup()

    def uploader(self, responses, tokens=None):
        return ResumableUploader(tokens or Tokens(), Transport(responses), chunk_size=256 * 1024, sleep=lambda _: None)

    def upload(self, uploader, privacy='private'):
        return uploader.upload(
            source=self.source, state_path=self.state, account='authorized@example.com',
            expected_channel=CHANNEL, title='A test upload', category_id='10',
            privacy=privacy, notify_subscribers=False)

    def test_same_state_lock_serializes_concurrent_uploader_runs(self):
        self.state.parent.mkdir(mode=0o700)
        first = _state_lock(self.state)
        first.__enter__()
        entered_second = threading.Event()

        def second_run():
            with _state_lock(self.state):
                entered_second.set()

        thread = threading.Thread(target=second_run)
        thread.start()
        time.sleep(0.03)
        self.assertFalse(entered_second.is_set())
        first.__exit__(None, None, None)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(entered_second.is_set())
        self.assertEqual(self.state.with_name(self.state.name + '.lock').stat().st_mode & 0o777, 0o600)

    def test_one_upload_is_private_resumable_and_verified_without_leaking_token(self):
        transport = Transport([
            (200, {'Location': SESSION}, b''),  # create resumable session
            (308, {}, b''),  # query starts at byte zero
            (308, {'Range': 'bytes=0-262143'}, b''),  # acknowledged first chunk
            (308, {'Range': 'bytes=0-262143'}, b''),  # reconcile before final chunk
            (201, {}, json.dumps({'id': VIDEO_ID}).encode()),
            (200, {}, video()),
        ])
        tokens = Tokens()
        result = self.upload(ResumableUploader(tokens, transport, chunk_size=256 * 1024, sleep=lambda _: None))
        self.assertEqual(result['video_id'], VIDEO_ID)
        self.assertEqual(result['processing_status'], 'processing')
        init = transport.calls[0]
        self.assertEqual(init[1], 'POST')
        self.assertIn('notifySubscribers=false', init[0])
        metadata = json.loads(init[2])
        self.assertEqual(metadata['status']['privacyStatus'], 'private')
        self.assertEqual(metadata['snippet']['title'], 'A test upload')
        chunks = [c for c in transport.calls if c[1] == 'PUT' and c[2]]
        self.assertEqual([c[3]['Content-Range'] for c in chunks], [
            'bytes 0-262143/262147', 'bytes 262144-262146/262147'])
        self.assertTrue(transport.calls[1][3]['Content-Range'].startswith('bytes */'))
        saved = json.dumps(json.loads(self.state.read_text()))
        self.assertNotIn('access-token-never-persist', saved)
        self.assertNotIn(SESSION, saved)  # completed state no longer needs bearer-like session URL
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.state.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(tokens.refreshes, 0)

    def test_account_channel_is_verified_before_any_upload_request(self):
        transport = Transport([(200, {'Location': SESSION}, b'')])
        transport.channel_response = (200, {}, json.dumps({'items': [{'id': 'different-channel'}]}).encode())
        with self.assertRaisesRegex(UploadError, 'expected channel'):
            self.upload(ResumableUploader(Tokens(), transport, sleep=lambda _: None))
        self.assertEqual(len(transport.all_calls), 1)
        self.assertEqual(transport.all_calls[0][1], 'GET')
        self.assertIn('/youtube/v3/channels?', transport.all_calls[0][0])

    def test_401_refreshes_once_then_reconciles_before_resending_chunk(self):
        transport = Transport([
            (200, {'Location': SESSION}, b''),
            (308, {}, b''),
            (401, {}, b''),
            (308, {}, b''),  # reconcile after refresh; server accepted no bytes
            (308, {'Range': 'bytes=0-262143'}, b''),
            (308, {'Range': 'bytes=0-262143'}, b''),  # reconcile before final chunk
            (201, {}, json.dumps({'id': VIDEO_ID}).encode()),
            (200, {}, video()),
        ])
        tokens = Tokens()
        result = self.upload(ResumableUploader(tokens, transport, chunk_size=256 * 1024, sleep=lambda _: None))
        self.assertEqual(result['video_id'], VIDEO_ID)
        self.assertEqual(tokens.refreshes, 1)
        puts = [c for c in transport.calls if c[1] == 'PUT']
        self.assertEqual(puts[1][3]['Content-Range'], 'bytes 0-262143/262147')
        self.assertEqual(puts[2][2], b'')

    def test_token_helper_failure_does_not_create_an_ambiguous_upload_state(self):
        def broken_token(_force_refresh=False):
            raise UploadError('private helper unavailable')
        transport = Transport([])
        uploader = ResumableUploader(broken_token, transport, sleep=lambda _: None)
        with self.assertRaisesRegex(UploadError, 'helper unavailable'):
            self.upload(uploader)
        self.assertFalse(self.state.exists())
        self.assertEqual(transport.calls, [])
        self.assertEqual(len(transport.all_calls), 0)

    def test_transient_chunk_errors_are_bounded_and_keep_the_same_session(self):
        transport = Transport([
            (200, {'Location': SESSION}, b''),
            (308, {}, b''),
            (503, {}, b''),
            (308, {}, b''),
            (503, {}, b''),
            (308, {}, b''),
            (503, {}, b''),
        ])
        uploader = ResumableUploader(Tokens(), transport, chunk_size=256 * 1024,
                                     max_retries=2, sleep=lambda _: None)
        with self.assertRaisesRegex(UploadError, 'chunk request failed'):
            self.upload(uploader)
        self.assertEqual(sum(call[1] == 'POST' for call in transport.calls), 1)
        self.assertEqual(sum(call[1] == 'PUT' and bool(call[2]) for call in transport.calls), 3)
        self.assertEqual(json.loads(self.state.read_text())['status'], 'uploading')

    def test_ambiguous_session_creation_is_persisted_and_never_blindly_retried(self):
        transport = Transport([OSError('connection reset')])
        uploader = ResumableUploader(Tokens(), transport, max_retries=0, sleep=lambda _: None)
        with self.assertRaisesRegex(UploadError, 'ambiguous'):
            self.upload(uploader)
        self.assertEqual(json.loads(self.state.read_text())['status'], 'initiation_uncertain')
        with self.assertRaisesRegex(UploadError, 'reconcile'):
            self.upload(ResumableUploader(Tokens(), Transport([]), sleep=lambda _: None))
        self.assertEqual(len(transport.calls), 1)

    def test_resume_reconciles_same_source_and_does_not_create_second_session(self):
        # First attempt leaves an acknowledged first chunk then a transport failure.
        first = Transport([
            (200, {'Location': SESSION}, b''),
            (308, {}, b''),
            (308, {'Range': 'bytes=0-262143'}, b''),
            OSError('timeout'),
        ])
        with self.assertRaises(UploadError):
            self.upload(ResumableUploader(Tokens(), first, chunk_size=256 * 1024, max_retries=0, sleep=lambda _: None))
        # On resume, query first. Google reports the first chunk was accepted.
        resumed = Transport([
            (308, {'Range': 'bytes=0-262143'}, b''),
            (201, {}, json.dumps({'id': VIDEO_ID}).encode()),
            (200, {}, video()),
        ])
        result = self.upload(ResumableUploader(Tokens(), resumed, chunk_size=256 * 1024, sleep=lambda _: None))
        self.assertEqual(result['video_id'], VIDEO_ID)
        self.assertEqual(resumed.calls[0][3]['Content-Range'], 'bytes */262147')
        self.assertFalse(any(c[1] == 'POST' for c in resumed.calls))

    def test_existing_state_rejects_changed_source_or_privacy_before_network(self):
        transport = Transport([
            (200, {'Location': SESSION}, b''),
            (308, {}, b''),
            OSError('stop after initiation'),
        ])
        with self.assertRaises(UploadError):
            self.upload(ResumableUploader(Tokens(), transport, chunk_size=256 * 1024, max_retries=0, sleep=lambda _: None))
        self.source.write_bytes(b'different')
        untouched = Transport([])
        with self.assertRaisesRegex(UploadError, '[Ss]ource'):
            self.upload(ResumableUploader(Tokens(), untouched, sleep=lambda _: None))
        self.assertEqual(untouched.calls, [])
        self.source.write_bytes(b'x' * (256 * 1024 + 3))
        self.state.unlink()
        setup = Transport([
            (200, {'Location': SESSION}, b''),
            (308, {}, b''),
            OSError('stop after creating a known session'),
        ])
        with self.assertRaises(UploadError):
            self.upload(ResumableUploader(Tokens(), setup, chunk_size=256 * 1024,
                                          max_retries=0, sleep=lambda _: None))
        with self.assertRaisesRegex(UploadError, 'privacy'):
            self.upload(ResumableUploader(Tokens(), untouched, sleep=lambda _: None), 'unlisted')
        self.assertEqual(untouched.calls, [])

    def test_completed_video_is_verified_instead_of_uploaded_again(self):
        self.state.parent.mkdir(mode=0o700)
        self.state.write_text(json.dumps({
            'status': 'uploaded', 'video_id': VIDEO_ID,
            'source': {'path': str(self.source.resolve()), 'size': self.source.stat().st_size,
                       'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                       'device': self.source.stat().st_dev, 'inode': self.source.stat().st_ino,
                       'mtime_ns': self.source.stat().st_mtime_ns,
                       'ctime_ns': self.source.stat().st_ctime_ns},
            'request': {'account': 'authorized@example.com', 'expected_channel': CHANNEL,
                        'title': 'A test upload', 'category_id': '10', 'privacy': 'private',
                        'notify_subscribers': False},
        }))
        os.chmod(self.state, 0o600)
        transport = Transport([(200, {}, video())])
        result = self.upload(ResumableUploader(Tokens(), transport, sleep=lambda _: None))
        self.assertEqual(result['video_id'], VIDEO_ID)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][1], 'GET')

    def test_actual_privacy_mismatch_is_reported_without_privacy_update(self):
        transport = Transport([
            (200, {'Location': SESSION}, b''),
            (308, {}, b''),
            (201, {}, json.dumps({'id': VIDEO_ID}).encode()),
            (200, {}, video(privacy='unlisted')),
        ])
        with self.assertRaisesRegex(UploadError, 'different privacy'):
            self.upload(ResumableUploader(Tokens(), transport, sleep=lambda _: None))
        self.assertEqual(transport.calls[-1][1], 'GET')
        self.assertFalse(any(url.startswith('https://www.googleapis.com/youtube/v3/videos?') and method != 'GET'
                             for url, method, _, _ in transport.calls))

    def test_channel_or_actual_privacy_mismatch_is_not_silently_changed(self):
        transport = Transport([
            (200, {'Location': SESSION}, b''), (308, {}, b''),
            (308, {'Range': 'bytes=0-262143'}, b''),
            (201, {}, json.dumps({'id': VIDEO_ID}).encode()),
            (200, {}, video(privacy='unlisted', channel='other-channel')),
        ])
        with self.assertRaisesRegex(UploadError, 'channel'):
            self.upload(ResumableUploader(Tokens(), transport, chunk_size=256 * 1024, sleep=lambda _: None))
        self.assertFalse(any(c[1] == 'PUT' and b'privacyStatus' in (c[2] or b'') for c in transport.calls))


if __name__ == '__main__':
    unittest.main()
